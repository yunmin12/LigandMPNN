"""
Stage 6: Vina Post-processing
Parse PDBQT results, select 10 best poses, create complex PDBs

Selection criteria:
1. Filter by COM distance to target ligand
2. Cluster by RMSD and rank by COM distance
3. Select 10 diverse poses
"""
import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from Bio.PDB import PDBParser, PDBIO, Superimposer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from utils.status_tracker import StatusTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VinaPostProcessor:
    def __init__(self, status_tracker: StatusTracker):
        self.status_tracker = status_tracker
        self.parser = PDBParser(QUIET=True)
        self.io = PDBIO()
    
    def parse_vina_pdbqt(self, pdbqt_file: Path) -> List[dict]:
        """
        Parse Vina output PDBQT file
        Returns: list of dicts with {pose_id, score, pdbqt_lines}
        """
        poses = []
        current_pose = None
        current_lines = []
        
        with open(pdbqt_file, 'r') as f:
            for line in f:
                if line.startswith('MODEL'):
                    if current_pose is not None:
                        current_pose['pdbqt_lines'] = current_lines
                        poses.append(current_pose)
                    
                    pose_id = int(line.split()[1])
                    current_pose = {'pose_id': pose_id, 'score': None}
                    current_lines = []
                
                elif line.startswith('REMARK VINA RESULT:'):
                    # Format: REMARK VINA RESULT:    -7.8      0.000      0.000
                    parts = line.split()
                    if len(parts) >= 4:
                        current_pose['score'] = float(parts[3])
                
                elif line.startswith('ENDMDL'):
                    if current_pose is not None:
                        current_pose['pdbqt_lines'] = current_lines
                        poses.append(current_pose)
                        current_pose = None
                        current_lines = []
                
                elif current_pose is not None:
                    # Skip REMARK lines, keep ATOM/HETATM
                    if line.startswith(('ATOM', 'HETATM')):
                        current_lines.append(line)
        
        return poses
    
    def pdbqt_to_pdb(self, pdbqt_lines: List[str], output_pdb: Path, 
                     het_id: str = 'LIG', chain_id: str = 'Z') -> bool:
        """
        Convert PDBQT lines to PDB format
        """
        try:
            with open(output_pdb, 'w') as f:
                for line in pdbqt_lines:
                    # Convert ATOM/HETATM line
                    if line.startswith(('ATOM', 'HETATM')):
                        # Change resname and chain
                        pdb_line = line[:17] + het_id.ljust(3)[:3] + line[20:21] + chain_id + line[22:]
                        f.write(pdb_line)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert PDBQT to PDB: {e}")
            return False
    
    def calculate_ligand_rmsd(self, pdb1: Path, pdb2: Path) -> Optional[float]:
        """
        Calculate RMSD between two ligand PDB files
        Uses heavy atoms only
        """
        try:
            struct1 = self.parser.get_structure('lig1', pdb1)
            struct2 = self.parser.get_structure('lig2', pdb2)
            
            # Get heavy atoms
            atoms1 = [atom for atom in struct1.get_atoms() if atom.element != 'H']
            atoms2 = [atom for atom in struct2.get_atoms() if atom.element != 'H']
            
            if len(atoms1) != len(atoms2):
                logger.warning(f"Atom count mismatch: {len(atoms1)} vs {len(atoms2)}")
                return None
            
            if len(atoms1) == 0:
                return None
            
            # Calculate RMSD using Superimposer
            super_imposer = Superimposer()
            super_imposer.set_atoms(atoms1, atoms2)
            rmsd = super_imposer.rms
            
            return rmsd
            
        except Exception as e:
            logger.error(f"Failed to calculate RMSD: {e}")
            return None
    
    def calculate_com_distance(self, pose_pdb: Path, target_pdb: Path) -> Optional[float]:
        """
        Calculate center of mass distance between docked pose and target ligand
        Uses heavy atoms only
        """
        try:
            pose_struct = self.parser.get_structure('pose', pose_pdb)
            target_struct = self.parser.get_structure('target', target_pdb)
            
            # Get heavy atom coordinates
            pose_coords = np.array([
                atom.coord for atom in pose_struct.get_atoms() 
                if atom.element != 'H'
            ])
            target_coords = np.array([
                atom.coord for atom in target_struct.get_atoms() 
                if atom.element != 'H'
            ])
            
            if len(pose_coords) == 0 or len(target_coords) == 0:
                return None
            
            # Calculate centroids
            pose_center = pose_coords.mean(axis=0)
            target_center = target_coords.mean(axis=0)
            
            # Euclidean distance
            distance = np.linalg.norm(pose_center - target_center)
            
            return distance
            
        except Exception as e:
            logger.error(f"Failed to calculate COM distance: {e}")
            return None
    
    def extract_target_ligand(self, target_pdb: Path, output_pdb: Path,
                              chain_id: str = 'Z', resid: int = 1) -> bool:
        """
        Extract target ligand from complex PDB
        """
        try:
            structure = self.parser.get_structure('complex', target_pdb)
            
            # Find and extract ligand
            ligand_atoms = []
            for model in structure:
                if chain_id in model:
                    for residue in model[chain_id]:
                        if residue.id[1] == resid:
                            for atom in residue:
                                ligand_atoms.append(atom)
            
            if not ligand_atoms:
                raise ValueError(f"No ligand found at chain {chain_id}, resid {resid}")
            
            # Save ligand
            class LigandSelect:
                def __init__(self, atoms):
                    self.atoms = atoms
                
                def accept_atom(self, atom):
                    return atom in self.atoms
            
            self.io.set_structure(structure)
            self.io.save(str(output_pdb), LigandSelect(ligand_atoms))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract target ligand: {e}")
            return False
    
    def combine_receptor_ligand(self, receptor_pdb: Path, ligand_pdb: Path, 
                                output_pdb: Path) -> bool:
        """
        Combine receptor and ligand into single complex PDB
        """
        try:
            with open(output_pdb, 'w') as out_f:
                # Write receptor
                with open(receptor_pdb, 'r') as rec_f:
                    for line in rec_f:
                        if line.startswith(('ATOM', 'HETATM', 'TER')):
                            out_f.write(line)
                
                # Write ligand
                with open(ligand_pdb, 'r') as lig_f:
                    for line in lig_f:
                        if line.startswith(('ATOM', 'HETATM')):
                            out_f.write(line)
                
                out_f.write('END\n')
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to combine receptor and ligand: {e}")
            return False
    
    def cluster_and_select_poses(self, poses_df: pd.DataFrame, n_select: int = 10,
                                  com_distance_threshold: float = 15.0) -> pd.DataFrame:
        """
        Cluster poses by RMSD and select n_select diverse poses
        
        Selection strategy:
        1. Filter out poses with COM distance > threshold
        2. Cluster by RMSD
        3. Rank by COM distance within each cluster
        4. Select representatives from clusters
        """
        logger.info(f"\nClustering and selecting {n_select} poses...")
        logger.info(f"  Total poses: {len(poses_df)}")
        
        # Step 1: Filter by COM distance
        if 'com_distance' in poses_df.columns:
            filtered_df = poses_df[poses_df['com_distance'] <= com_distance_threshold].copy()
            logger.info(f"  After COM distance filter (<={com_distance_threshold}Å): {len(filtered_df)}")
        else:
            filtered_df = poses_df.copy()
        
        if len(filtered_df) == 0:
            logger.warning("  No poses passed COM distance filter!")
            return pd.DataFrame()
        
        # If we have fewer poses than requested, return all
        if len(filtered_df) <= n_select:
            logger.info(f"  Selecting all {len(filtered_df)} poses")
            return filtered_df
        
        # Step 2: Build RMSD distance matrix
        logger.info("  Building RMSD distance matrix...")
        n_poses = len(filtered_df)
        rmsd_matrix = np.zeros((n_poses, n_poses))
        
        pose_pdbs = filtered_df['ligand_pdb'].tolist()
        
        for i in range(n_poses):
            for j in range(i+1, n_poses):
                rmsd = self.calculate_ligand_rmsd(Path(pose_pdbs[i]), Path(pose_pdbs[j]))
                if rmsd is not None:
                    rmsd_matrix[i, j] = rmsd
                    rmsd_matrix[j, i] = rmsd
                else:
                    rmsd_matrix[i, j] = 999.0
                    rmsd_matrix[j, i] = 999.0
        
        # Step 3: Hierarchical clustering
        logger.info("  Performing hierarchical clustering...")
        
        # Convert to condensed distance matrix
        condensed_dist = squareform(rmsd_matrix)
        
        # Perform clustering (average linkage)
        Z = linkage(condensed_dist, method='average')
        
        # Determine number of clusters (adaptive based on data)
        n_clusters = min(n_select, max(3, n_poses // 3))
        clusters = fcluster(Z, n_clusters, criterion='maxclust')
        
        filtered_df['cluster'] = clusters
        
        logger.info(f"  Created {len(set(clusters))} clusters")
        
        # Step 4: Select representatives from each cluster
        selected_indices = []
        
        for cluster_id in sorted(set(clusters)):
            cluster_df = filtered_df[filtered_df['cluster'] == cluster_id].copy()
            
            # Rank by COM distance (lower is better)
            cluster_df = cluster_df.sort_values('com_distance')
            
            # Select best from this cluster
            n_from_cluster = max(1, int(len(cluster_df) * n_select / n_poses))
            selected_indices.extend(cluster_df.head(n_from_cluster).index.tolist())
        
        # Ensure we have exactly n_select poses
        if len(selected_indices) > n_select:
            # Keep top n_select by COM distance
            temp_df = filtered_df.loc[selected_indices]
            temp_df = temp_df.sort_values('com_distance')
            selected_indices = temp_df.head(n_select).index.tolist()
        elif len(selected_indices) < n_select:
            # Add more poses to reach n_select
            remaining = filtered_df.drop(selected_indices)
            remaining = remaining.sort_values('com_distance')
            additional = remaining.head(n_select - len(selected_indices)).index.tolist()
            selected_indices.extend(additional)
        
        selected_df = filtered_df.loc[selected_indices].copy()
        selected_df = selected_df.sort_values('com_distance')
        selected_df['rank'] = range(1, len(selected_df) + 1)
        
        logger.info(f"  Selected {len(selected_df)} poses")
        
        return selected_df
    
    def process_config(self, row: pd.Series, config_id: str, runs_dir: Path,
                      current_num: int = None, total_num: int = None) -> bool:
        """
        Process one config: parse results, select poses, create complex PDBs
        
        Args:
            row: DataFrame row
            config_id: Config ID
            runs_dir: Runs directory
            current_num: Current config number (for progress tracking)
            total_num: Total configs (for progress tracking)
        """
        # Check if can run this stage
        if not self.status_tracker.can_run_stage(config_id, '6_vina_post'):
            self.status_tracker.mark_status(config_id, '6_vina_post', 'skip',
                                           'Previous stage not completed')
            return False
        
        try:
            run_dir = runs_dir / config_id
            prepared_dir = run_dir / 'prepared'
            docking_dir = run_dir / 'docking'
            final_dir = run_dir / 'final'
            final_dir.mkdir(parents=True, exist_ok=True)
            
            # Format progress indicator
            progress_str = ""
            if current_num is not None and total_num is not None:
                progress_str = f" ({current_num}/{total_num})"
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing {config_id}{progress_str}")
            logger.info(f"{'='*70}")
            
            # Get target ligand PDB for COM distance calculation
            target_blurred_pdb = Path(row['target_blurred_path'])
            target_ligand_pdb = final_dir / 'target_ligand.pdb'
            
            if not self.extract_target_ligand(target_blurred_pdb, target_ligand_pdb):
                raise ValueError("Failed to extract target ligand")
            
            # Get off-target ligand HET ID
            het_id = str(row.get('offtarget_het_id', 'LIG'))[:3]
            
            # Parse all Vina outputs
            all_poses = []
            
            for seed in range(5):  # 5 seeds
                pdbqt_file = docking_dir / f'out_seed{seed}.pdbqt'
                
                if not pdbqt_file.exists():
                    logger.debug(f"  Output file not found: {pdbqt_file}")
                    continue
                
                poses = self.parse_vina_pdbqt(pdbqt_file)
                logger.info(f"  Seed {seed}: {len(poses)} poses")
                
                for pose in poses:
                    # Convert PDBQT to PDB
                    pose_pdb = final_dir / f'pose_seed{seed}_model{pose["pose_id"]}.pdb'
                    
                    if not self.pdbqt_to_pdb(pose['pdbqt_lines'], pose_pdb, het_id=het_id):
                        continue
                    
                    # Calculate COM distance
                    com_dist = self.calculate_com_distance(pose_pdb, target_ligand_pdb)
                    
                    all_poses.append({
                        'seed': seed,
                        'model': pose['pose_id'],
                        'score': pose['score'],
                        'ligand_pdb': str(pose_pdb),
                        'com_distance': com_dist
                    })
            
            if len(all_poses) == 0:
                raise ValueError("No poses were successfully parsed")
            
            logger.info(f"  Total poses parsed: {len(all_poses)}")
            
            # Create DataFrame
            poses_df = pd.DataFrame(all_poses)
            
            # Cluster and select 10 poses
            selected_df = self.cluster_and_select_poses(poses_df, n_select=10)
            
            if len(selected_df) == 0:
                raise ValueError("No poses selected after clustering")
            
            logger.info(f"\nSelected {len(selected_df)} poses:")
            for idx, pose in selected_df.iterrows():
                logger.info(f"  Rank {pose['rank']}: seed{pose['seed']}_model{pose['model']} "
                          f"(score={pose['score']:.2f}, COM={pose['com_distance']:.2f}Å)")
            
            # Create complex PDBs for selected poses
            receptor_pdb = prepared_dir / 'receptor.pdb'
            
            for idx, pose in selected_df.iterrows():
                ligand_pdb = Path(pose['ligand_pdb'])
                complex_pdb = final_dir / f'complex_rank{pose["rank"]}_seed{pose["seed"]}_model{pose["model"]}.pdb'
                
                if not self.combine_receptor_ligand(receptor_pdb, ligand_pdb, complex_pdb):
                    logger.warning(f"  Failed to create complex PDB: {complex_pdb.name}")
            
            # Save selection summary
            summary_csv = final_dir / 'selected_poses.csv'
            selected_df.to_csv(summary_csv, index=False)
            logger.info(f"\n  Saved summary: {summary_csv}")
            
            logger.info(f"✓ {config_id}: Post-processing completed")
            self.status_tracker.mark_status(config_id, '6_vina_post', 'success')
            return True
            
        except Exception as e:
            logger.error(f"✗ {config_id}: Post-processing failed: {e}")
            self.status_tracker.mark_status(config_id, '6_vina_post', 'failed', str(e))
            return False


def main():
    p = argparse.ArgumentParser(description="Stage 6: Vina Post-processing")
    p.add_argument("--csv", required=True, help="Input CSV (output from stage 5)")
    p.add_argument("--runs_dir", required=True, help="Runs directory")
    p.add_argument("--status_dir", required=True, help="Status tracking directory")
    args = p.parse_args()
    
    # Initialize
    status_tracker = StatusTracker(Path(args.status_dir))
    processor = VinaPostProcessor(status_tracker)
    
    # Load CSV
    df = pd.read_csv(args.csv)
    logger.info(f"Processing {len(df)} configs")
    
    # Process each config
    updated_rows = []
    success_count = 0
    failed_count = 0
    total_configs = len(df)
    
    for idx, row in df.iterrows():
        config_id = row.get('config_id', f"config_{idx:06d}")
        current_num = idx + 1
        
        if processor.process_config(row, config_id, Path(args.runs_dir),
                                   current_num=current_num, total_num=total_configs):
            # Only add successful rows to output CSV
            row_dict = row.to_dict()
            updated_rows.append(row_dict)
            success_count += 1
        else:
            failed_count += 1
    
    # Save updated CSV (only successful rows)
    output_csv = Path(args.runs_dir).parent / "stage6_output.csv"
    pd.DataFrame(updated_rows).to_csv(output_csv, index=False)
    logger.info(f"Saved {success_count} successful entries to {output_csv}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"=== STAGE 6 SUMMARY ===")
    print(f"{'='*70}")
    print(f"Total processed:  {len(df)}")
    print(f"Success:          {success_count}")
    print(f"Failed:           {failed_count}")
    print(f"Success rate:     {100*success_count/len(df) if len(df) > 0 else 0:.1f}%")
    print(f"Output CSV:       {output_csv}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
