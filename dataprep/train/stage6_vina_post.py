"""
Stage 6: Vina Post-processing
Parse PDBQT results, select 10 best poses, create complex PDBs

=== SELECTION PARAMETERS ===
- Docking seeds: 3 (seed 0, 1, 2)
- Poses per seed: 5 (model 1-5)
- Total input poses: ~15 per config
- Target output poses: 10

=== FILTERING CRITERIA ===
- Clash threshold: 1.5 Å (heavy atoms only)
- Max clashes allowed: 0 (relaxed to min+2 if all clash)
- COM distance threshold: 15.0 Å (to target ligand centroid)

=== CLUSTERING METHOD ===
- RMSD calculation: heavy atoms only (exclude H)
- Clustering method: hierarchical (average linkage)
- Selection strategy: diverse representatives from clusters

=== OUTPUT FORMAT ===
- Ligand records: HETATM
- Ligand chain: Z
- Element column (77-78): filled
- Bond connectivity: CONECT records included
- Complex PDB: receptor (ATOM) + ligand (HETATM)

Selection criteria:
1. Filter by COM distance to target ligand
2. Cluster by RMSD and rank by COM distance
3. Check for clashes with receptor
4. Select 10 diverse poses

Sanity checks:
- Ligand records are HETATM
- Element column (77-78) is properly filled
- No steric clashes with receptor
"""
import os
import sys
import re
import logging
import argparse
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from Bio.PDB import PDBParser, PDBIO, Superimposer, NeighborSearch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from utils.status_tracker import StatusTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Element symbol mapping for common atom types in PDBQT
PDBQT_ELEMENT_MAP = {
    'C': 'C', 'A': 'C', 'NA': 'N', 'N': 'N', 'OA': 'O', 'O': 'O',
    'SA': 'S', 'S': 'S', 'H': 'H', 'HD': 'H', 'HS': 'H',
    'P': 'P', 'F': 'F', 'Cl': 'Cl', 'CL': 'Cl', 'Br': 'Br', 'BR': 'Br',
    'I': 'I', 'Mg': 'Mg', 'MG': 'Mg', 'Ca': 'Ca', 'CA': 'Ca',
    'Mn': 'Mn', 'MN': 'Mn', 'Fe': 'Fe', 'FE': 'Fe', 'Zn': 'Zn', 'ZN': 'Zn'
}


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
    
    def pdbqt_to_sdf_to_pdb(self, pdbqt_lines: List[str], output_pdb: Path, 
                            het_id: str = 'LIG', chain_id: str = 'Z', resid: int = 1) -> bool:
        """
        Convert PDBQT to PDB via SDF to preserve chemical structure and bond information
        
        Process:
        1. Write PDBQT lines to temporary file
        2. Convert PDBQT -> SDF using Open Babel (preserves bond info)
        3. Convert SDF -> PDB using Open Babel
        4. Post-process PDB to ensure proper format (HETATM, chain, etc.)
        
        Ensures:
        - Chemical structure and bonds preserved
        - Records are HETATM (not ATOM)
        - Residue name is set correctly
        - Chain ID is set correctly
        - Element column (77-78) is properly filled
        """
        try:
            # Create temporary files
            temp_dir = output_pdb.parent / 'temp'
            temp_dir.mkdir(exist_ok=True)
            
            temp_pdbqt = temp_dir / f'{output_pdb.stem}_temp.pdbqt'
            temp_sdf = temp_dir / f'{output_pdb.stem}_temp.sdf'
            temp_pdb = temp_dir / f'{output_pdb.stem}_temp.pdb'
            
            # Step 1: Write PDBQT to temporary file
            with open(temp_pdbqt, 'w') as f:
                for line in pdbqt_lines:
                    f.write(line)
            
            # Step 2: Convert PDBQT to SDF using Open Babel
            # Do NOT use --gen3D as it will generate new coordinates instead of preserving docking results
            cmd_pdbqt_to_sdf = [
                'obabel',
                str(temp_pdbqt),
                '-O', str(temp_sdf)
            ]
            
            result = subprocess.run(
                cmd_pdbqt_to_sdf,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"obabel PDBQT->SDF failed: {result.stderr}")
                return False
            
            if not temp_sdf.exists():
                logger.error(f"SDF file not created: {temp_sdf}")
                return False
            
            # Step 3: Convert SDF to PDB using Open Babel
            # -b: write bond connectivity as CONECT records
            cmd_sdf_to_pdb = [
                'obabel',
                str(temp_sdf),
                '-O', str(temp_pdb),
                '-b'  # Write CONECT records for bond connectivity
            ]
            
            result = subprocess.run(
                cmd_sdf_to_pdb,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"obabel SDF->PDB failed: {result.stderr}")
                return False
            
            if not temp_pdb.exists():
                logger.error(f"PDB file not created: {temp_pdb}")
                return False
            
            # Step 4: Post-process PDB to ensure proper format
            self._postprocess_ligand_pdb(temp_pdb, output_pdb, het_id, chain_id, resid)
            
            # Clean up temporary files
            temp_pdbqt.unlink(missing_ok=True)
            temp_sdf.unlink(missing_ok=True)
            temp_pdb.unlink(missing_ok=True)
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("obabel conversion timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to convert PDBQT->SDF->PDB: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _postprocess_ligand_pdb(self, input_pdb: Path, output_pdb: Path,
                                het_id: str, chain_id: str, resid: int):
        """
        Post-process ligand PDB to ensure proper format
        - Change to HETATM records
        - Set residue name, chain ID, residue ID
        - Ensure element column is filled
        - Preserve CONECT records for bond connectivity
        """
        processed_lines = []
        conect_lines = []
        atom_serial = 1
        old_to_new_serial = {}  # Map old serial numbers to new ones for CONECT
        
        with open(input_pdb, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    # Store old serial number
                    old_serial = int(line[6:11].strip()) if len(line) > 11 else atom_serial
                    
                    # Parse existing line
                    atom_name = line[12:16] if len(line) > 16 else '    '
                    x = line[30:38] if len(line) > 38 else '       '
                    y = line[38:46] if len(line) > 46 else '       '
                    z = line[46:54] if len(line) > 54 else '       '
                    occupancy = line[54:60].strip() if len(line) > 60 else '1.00'
                    bfactor = line[60:66].strip() if len(line) > 66 else '0.00'
                    element = line[76:78].strip() if len(line) > 78 else ''
                    
                    # Infer element if missing
                    if not element:
                        element = self._infer_element(atom_name.strip())
                    
                    # Default values
                    if not occupancy:
                        occupancy = '1.00'
                    if not bfactor:
                        bfactor = '0.00'
                    
                    # Map old serial to new serial
                    old_to_new_serial[old_serial] = atom_serial
                    
                    # Construct proper PDB line with HETATM record
                    pdb_line = (
                        f"HETATM{atom_serial:5d} {atom_name} "
                        f"{het_id:>3s} {chain_id}{resid:4d}    "
                        f"{x}{y}{z}{float(occupancy):>6.2f}{float(bfactor):>6.2f}"
                        f"          {element:>2s}  \n"
                    )
                    
                    processed_lines.append(pdb_line)
                    atom_serial += 1
                
                elif line.startswith('CONECT'):
                    # Store CONECT lines for later processing
                    conect_lines.append(line)
        
        # Process CONECT records with updated serial numbers
        processed_conect = []
        for line in conect_lines:
            parts = line.split()
            if len(parts) < 2:
                continue
            
            try:
                # First number is the atom serial
                old_serial = int(parts[1])
                if old_serial not in old_to_new_serial:
                    continue
                
                new_serial = old_to_new_serial[old_serial]
                
                # Bonded atoms
                bonded = []
                for i in range(2, len(parts)):
                    old_bonded = int(parts[i])
                    if old_bonded in old_to_new_serial:
                        bonded.append(old_to_new_serial[old_bonded])
                
                if bonded:
                    # Format: CONECT serial bond1 bond2 bond3 bond4
                    conect_str = f"CONECT{new_serial:5d}"
                    for b in bonded:
                        conect_str += f"{b:5d}"
                    conect_str += "\n"
                    processed_conect.append(conect_str)
            
            except (ValueError, IndexError):
                continue
        
        # Write processed PDB
        with open(output_pdb, 'w') as f:
            # Write atom records
            for line in processed_lines:
                f.write(line)
            # Write CONECT records
            for line in processed_conect:
                f.write(line)
            f.write('END\n')
    
    def _pdbqt_type_to_element(self, pdbqt_type: str, atom_name: str) -> str:
        """
        Convert PDBQT atom type to element symbol
        
        Args:
            pdbqt_type: PDBQT atom type (e.g., 'C', 'OA', 'HD', 'NA')
            atom_name: PDB atom name for fallback
        
        Returns:
            Element symbol (e.g., 'C', 'O', 'H', 'N')
        """
        # Try direct mapping
        if pdbqt_type in PDBQT_ELEMENT_MAP:
            return PDBQT_ELEMENT_MAP[pdbqt_type]
        
        # Try first character of pdbqt_type
        if pdbqt_type and pdbqt_type[0] in PDBQT_ELEMENT_MAP:
            return PDBQT_ELEMENT_MAP[pdbqt_type[0]]
        
        # Fallback: infer from atom name
        atom_name_stripped = atom_name.strip()
        if atom_name_stripped:
            # First character is usually the element for most atoms
            # But some atoms like 'CA' (alpha carbon) vs 'CA' (calcium) need care
            first_char = atom_name_stripped[0]
            if first_char.isalpha():
                # Check for two-letter elements
                if len(atom_name_stripped) >= 2:
                    two_char = atom_name_stripped[:2].upper()
                    if two_char in ['CL', 'BR', 'FE', 'ZN', 'MG', 'CA', 'MN']:
                        # Check if this looks like a metal (usually no number after)
                        if len(atom_name_stripped) == 2 or not atom_name_stripped[2].isalpha():
                            return two_char.capitalize()
                return first_char.upper()
        
        return 'C'  # Default to carbon
    
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
    
    def check_clash(self, ligand_pdb: Path, receptor_pdb: Path, 
                    clash_threshold: float = 1.5) -> Tuple[bool, int]:
        """
        Check for steric clashes between ligand and receptor
        
        Args:
            ligand_pdb: Path to ligand PDB file
            receptor_pdb: Path to receptor PDB file
            clash_threshold: Distance threshold for clash (Angstroms)
        
        Returns:
            Tuple of (has_clash, n_clashes)
        """
        try:
            ligand_struct = self.parser.get_structure('ligand', ligand_pdb)
            receptor_struct = self.parser.get_structure('receptor', receptor_pdb)
            
            # Get ligand heavy atom coordinates
            ligand_coords = np.array([
                atom.coord for atom in ligand_struct.get_atoms() 
                if atom.element != 'H'
            ])
            
            # Get receptor heavy atom coordinates
            receptor_coords = np.array([
                atom.coord for atom in receptor_struct.get_atoms() 
                if atom.element != 'H'
            ])
            
            if len(ligand_coords) == 0 or len(receptor_coords) == 0:
                return False, 0
            
            # Build KD-tree for receptor atoms for efficient distance queries
            receptor_tree = cKDTree(receptor_coords)
            
            # Check for clashes
            n_clashes = 0
            for lig_coord in ligand_coords:
                # Find atoms within clash threshold
                nearby = receptor_tree.query_ball_point(lig_coord, clash_threshold)
                n_clashes += len(nearby)
            
            has_clash = n_clashes > 0
            
            return has_clash, n_clashes
            
        except Exception as e:
            logger.error(f"Failed to check clash: {e}")
            return False, 0
    
    def sanity_check_pdb(self, pdb_path: Path, fix: bool = True) -> Tuple[bool, List[str]]:
        """
        Perform sanity checks on a PDB file and optionally fix issues
        
        Checks:
        1. Ligand records should be HETATM
        2. Element column (77-78) should not be empty
        3. Basic format validation
        
        Args:
            pdb_path: Path to PDB file
            fix: If True, fix issues and rewrite the file
        
        Returns:
            Tuple of (all_ok, list of issues found)
        """
        issues = []
        fixed_lines = []
        needs_fix = False
        
        try:
            with open(pdb_path, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                if not line.startswith(('ATOM', 'HETATM')):
                    fixed_lines.append(line)
                    continue
                
                # Ensure line is long enough
                if len(line.rstrip()) < 54:
                    issues.append(f"Line {i+1}: Too short ({len(line.rstrip())} chars)")
                    continue
                
                # Pad line to 80 characters
                line = line.rstrip().ljust(80) + '\n'
                
                # Check 1: Ligand records should be HETATM (chain Z)
                chain_id = line[21] if len(line) > 21 else ''
                if chain_id == 'Z' and line.startswith('ATOM  '):
                    issues.append(f"Line {i+1}: Ligand record is ATOM, should be HETATM")
                    line = 'HETATM' + line[6:]
                    needs_fix = True
                
                # Check 2: Element column (77-78) should not be empty
                element = line[76:78].strip() if len(line) > 77 else ''
                if not element:
                    # Try to infer element from atom name
                    atom_name = line[12:16].strip()
                    inferred_element = self._infer_element(atom_name)
                    issues.append(f"Line {i+1}: Empty element column, inferred '{inferred_element}' from '{atom_name}'")
                    # Fix by inserting element
                    line = line[:76] + f"{inferred_element:>2s}" + line[78:]
                    needs_fix = True
                
                fixed_lines.append(line)
            
            # Write fixed file if needed
            if fix and needs_fix:
                with open(pdb_path, 'w') as f:
                    f.writelines(fixed_lines)
                logger.debug(f"Fixed {len(issues)} issues in {pdb_path.name}")
            
            all_ok = len(issues) == 0
            return all_ok, issues
            
        except Exception as e:
            logger.error(f"Failed sanity check for {pdb_path}: {e}")
            return False, [str(e)]
    
    def _infer_element(self, atom_name: str) -> str:
        """
        Infer element symbol from atom name
        """
        atom_name = atom_name.strip()
        if not atom_name:
            return 'C'
        
        # Common patterns
        # First check for two-letter elements
        if len(atom_name) >= 2:
            two_char = atom_name[:2].upper()
            if two_char in ['CL', 'BR', 'FE', 'ZN', 'MG', 'MN', 'CU', 'CO', 'NI']:
                return two_char[0] + two_char[1].lower()
        
        # Single letter element from first character
        first_char = atom_name[0].upper()
        if first_char in 'CNOSHPF':
            return first_char
        
        return 'C'  # Default
    
    def extract_target_ligand(self, target_pdb: Path, output_pdb: Path,
                              chain_id: str = 'Z', resid: int = 1) -> bool:
        """
        Extract target ligand from complex PDB using simple line-based approach
        """
        try:
            ligand_lines = []
            
            with open(target_pdb, 'r') as f:
                for line in f:
                    if line.startswith(('ATOM', 'HETATM')):
                        # Check chain ID (column 22, 0-indexed: 21)
                        if len(line) > 21 and line[21] == chain_id:
                            ligand_lines.append(line)
            
            if not ligand_lines:
                raise ValueError(f"No ligand found at chain {chain_id}")
            
            # Write ligand to output file
            with open(output_pdb, 'w') as f:
                for line in ligand_lines:
                    f.write(line)
                f.write('END\n')
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract target ligand: {e}")
            return False
    
    def combine_receptor_ligand(self, receptor_pdb: Path, ligand_pdb: Path, 
                                output_pdb: Path) -> bool:
        """
        Combine receptor and ligand into single complex PDB
        Preserves CONECT records from ligand
        """
        try:
            with open(output_pdb, 'w') as out_f:
                # Write receptor
                with open(receptor_pdb, 'r') as rec_f:
                    for line in rec_f:
                        if line.startswith(('ATOM', 'HETATM', 'TER')):
                            out_f.write(line)
                
                # Write ligand (including CONECT records)
                with open(ligand_pdb, 'r') as lig_f:
                    for line in lig_f:
                        if line.startswith(('ATOM', 'HETATM', 'CONECT')):
                            out_f.write(line)
                
                out_f.write('END\n')
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to combine receptor and ligand: {e}")
            return False
    
    def cluster_and_select_poses(self, poses_df: pd.DataFrame, receptor_pdb: Path,
                                  n_select: int = 10,
                                  com_distance_threshold: float = 15.0,
                                  max_clashes: int = 0) -> pd.DataFrame:
        """
        Cluster poses by RMSD and select n_select diverse poses
        
        Selection strategy:
        1. Filter out poses with clashes
        2. Filter out poses with COM distance > threshold
        3. Cluster by RMSD
        4. Rank by COM distance within each cluster
        5. Select representatives from clusters
        """
        logger.info(f"\nClustering and selecting {n_select} poses...")
        logger.info(f"  Total poses: {len(poses_df)}")
        
        # Step 0: Filter by clashes
        if 'n_clashes' in poses_df.columns:
            no_clash_df = poses_df[poses_df['n_clashes'] <= max_clashes].copy()
            logger.info(f"  After clash filter (<={max_clashes} clashes): {len(no_clash_df)}")
        else:
            no_clash_df = poses_df.copy()
        
        if len(no_clash_df) == 0:
            logger.warning("  No poses passed clash filter! Trying with relaxed threshold...")
            # Try with slightly relaxed threshold if all poses have clashes
            if 'n_clashes' in poses_df.columns:
                min_clashes = poses_df['n_clashes'].min()
                no_clash_df = poses_df[poses_df['n_clashes'] <= min_clashes + 2].copy()
                logger.info(f"  With relaxed clash filter (<={min_clashes + 2}): {len(no_clash_df)}")
        
        # Step 1: Filter by COM distance
        if 'com_distance' in no_clash_df.columns:
            filtered_df = no_clash_df[no_clash_df['com_distance'] <= com_distance_threshold].copy()
            logger.info(f"  After COM distance filter (<={com_distance_threshold}Å): {len(filtered_df)}")
        else:
            filtered_df = no_clash_df.copy()
        
        if len(filtered_df) == 0:
            logger.warning("  No poses passed COM distance filter!")
            return pd.DataFrame()
        
        # If we have fewer poses than requested, return all
        if len(filtered_df) <= n_select:
            logger.info(f"  Selecting all {len(filtered_df)} poses")
            selected_df = filtered_df.copy()
            selected_df = selected_df.sort_values('com_distance')
            selected_df['rank'] = range(1, len(selected_df) + 1)
            return selected_df
        
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
                      current_num: int = None, total_num: int = None) -> tuple:
        """
        Process one config: parse results, select poses, create complex PDBs
        
        Args:
            row: DataFrame row
            config_id: Config ID
            runs_dir: Runs directory
            current_num: Current config number (for progress tracking)
            total_num: Total configs (for progress tracking)
            
        Returns:
            Tuple of (success: bool, pose_paths: list of str)
        """
        # Check if can run this stage
        if not self.status_tracker.can_run_stage(config_id, '6_vina_post'):
            self.status_tracker.mark_status(config_id, '6_vina_post', 'skip',
                                           'Previous stage not completed')
            return False, []
        
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
            
            # Get receptor for clash detection
            receptor_pdb = prepared_dir / 'receptor.pdb'
            if not receptor_pdb.exists():
                raise ValueError(f"Receptor PDB not found: {receptor_pdb}")
            
            # Parse all Vina outputs (dynamically detect seeds)
            # Only process if docking log files exist (indicates successful docking)
            all_poses = []
            seeds_found = []
            
            # Try to find all available seed files
            for seed in range(10):  # Check up to 10 seeds
                pdbqt_file = docking_dir / f'out_seed{seed}.pdbqt'
                
                # Check if log file exists (either vina_seed*.log or unidock_seed*.log)
                vina_log = docking_dir / f'vina_seed{seed}.log'
                unidock_log = docking_dir / f'unidock_seed{seed}.log'
                
                if not (vina_log.exists() or unidock_log.exists()):
                    # No log file means docking didn't complete successfully
                    logger.debug(f"  Seed {seed}: No log file found (skipping)")
                    continue
                
                if not pdbqt_file.exists():
                    logger.warning(f"  Seed {seed}: Log exists but output PDBQT not found")
                    continue
                
                seeds_found.append(seed)
                poses = self.parse_vina_pdbqt(pdbqt_file)
                logger.info(f"  Seed {seed}: {len(poses)} poses")
                
                for pose in poses:
                    # Convert PDBQT to PDB via SDF (preserves chemical structure)
                    pose_pdb = final_dir / f'pose_seed{seed}_model{pose["pose_id"]}.pdb'
                    
                    if not self.pdbqt_to_sdf_to_pdb(pose['pdbqt_lines'], pose_pdb, het_id=het_id):
                        continue
                    
                    # Calculate COM distance
                    com_dist = self.calculate_com_distance(pose_pdb, target_ligand_pdb)
                    
                    # Check for clashes with receptor
                    has_clash, n_clashes = self.check_clash(pose_pdb, receptor_pdb, clash_threshold=1.5)
                    
                    all_poses.append({
                        'seed': seed,
                        'model': pose['pose_id'],
                        'score': pose['score'],
                        'ligand_pdb': str(pose_pdb),
                        'com_distance': com_dist,
                        'has_clash': has_clash,
                        'n_clashes': n_clashes
                    })
            
            if len(seeds_found) == 0:
                raise ValueError("No docking output files found")
            
            logger.info(f"  Found {len(seeds_found)} seeds: {seeds_found}")
            
            if len(all_poses) == 0:
                raise ValueError("No poses were successfully parsed")
            
            logger.info(f"  Total poses parsed: {len(all_poses)}")
            
            # Create DataFrame
            poses_df = pd.DataFrame(all_poses)
            
            # Log clash statistics
            if 'has_clash' in poses_df.columns:
                clash_stats = poses_df['has_clash'].value_counts()
                logger.info(f"  Poses with clashes: {clash_stats.get(True, 0)}, without: {clash_stats.get(False, 0)}")
            
            # Cluster and select 10 poses (with clash filtering)
            selected_df = self.cluster_and_select_poses(poses_df, receptor_pdb, n_select=10)
            
            if len(selected_df) == 0:
                raise ValueError("No poses selected after clustering")
            
            # Log selected poses with clash info
            logger.info(f"\nSelected {len(selected_df)} poses:")
            for idx, pose in selected_df.iterrows():
                clash_info = f", clashes={pose.get('n_clashes', 'N/A')}" if 'n_clashes' in pose else ""
                logger.info(f"  Rank {pose['rank']}: seed{pose['seed']}_model{pose['model']} "
                          f"(score={pose['score']:.2f}, COM={pose['com_distance']:.2f}Å{clash_info})")
            
            # Create complex PDBs for selected poses
            complex_pdb_paths = []  # Collect absolute paths
            for idx, pose in selected_df.iterrows():
                ligand_pdb = Path(pose['ligand_pdb'])
                complex_pdb = final_dir / f'complex_rank{int(pose["rank"])}_seed{int(pose["seed"])}_model{int(pose["model"])}.pdb'
                
                if not self.combine_receptor_ligand(receptor_pdb, ligand_pdb, complex_pdb):
                    logger.warning(f"  Failed to create complex PDB: {complex_pdb.name}")
                    continue
                
                # Perform sanity check on the complex PDB
                all_ok, issues = self.sanity_check_pdb(complex_pdb, fix=True)
                if not all_ok:
                    logger.debug(f"  Fixed {len(issues)} issues in {complex_pdb.name}")
                
                # Add absolute path to list
                complex_pdb_paths.append(str(complex_pdb.absolute()))
            
            # Save selection summary
            summary_csv = final_dir / 'selected_poses.csv'
            selected_df.to_csv(summary_csv, index=False)
            logger.info(f"\n  Saved summary: {summary_csv}")
            logger.info(f"  Created {len(complex_pdb_paths)} complex PDBs")
            
            logger.info(f"✓ {config_id}: Post-processing completed")
            self.status_tracker.mark_status(config_id, '6_vina_post', 'success')
            return True, complex_pdb_paths
            
        except Exception as e:
            logger.error(f"✗ {config_id}: Post-processing failed: {e}")
            self.status_tracker.mark_status(config_id, '6_vina_post', 'failed', str(e))
            return False, []


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
        
        success, pose_paths = processor.process_config(row, config_id, Path(args.runs_dir),
                                                       current_num=current_num, total_num=total_configs)
        
        if success:
            # Only add successful rows to output CSV
            row_dict = row.to_dict()
            # Add offtarget_pose_path column with semicolon-separated absolute paths
            row_dict['offtarget_pose_path'] = ';'.join(pose_paths) if pose_paths else ''
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
