"""
Vina Post-processing: Parse PDBQT results, filter poses, calculate metrics
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Superimposer
import argparse
import subprocess
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_file):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_vina_pdbqt(pdbqt_file):
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
            
            elif line.startswith('REMARK VINA RESULT'):
                # Parse score from: REMARK VINA RESULT:    -8.5      0.000      0.000
                parts = line.split()
                score = float(parts[3])
                current_pose['score'] = score
            
            elif line.startswith('ENDMDL'):
                current_lines.append(line)
                current_pose['pdbqt_lines'] = current_lines
                poses.append(current_pose)
                current_pose = None
                current_lines = []
            
            else:
                if current_pose is not None:
                    current_lines.append(line)
    
    return poses


def pdbqt_to_pdb(pdbqt_lines, output_pdb):
    """Convert PDBQT lines to PDB file"""
    with open(output_pdb, 'w') as f:
        for line in pdbqt_lines:
            if line.startswith(('ATOM', 'HETATM')):
                # PDBQT has extra columns, need to truncate
                pdb_line = line[:66] + '\n'
                f.write(pdb_line)
            elif line.startswith(('MODEL', 'ENDMDL', 'TER', 'END')):
                f.write(line)


def check_clashes(pose_pdb, receptor_pdb, clash_distance=2.0):
    """
    Check for clashes between ligand and receptor
    Returns: True if severe clashes exist, False otherwise
    """
    parser = PDBParser(QUIET=True)
    
    try:
        ligand_structure = parser.get_structure('ligand', pose_pdb)
        receptor_structure = parser.get_structure('receptor', receptor_pdb)
    except:
        logger.warning(f"Failed to parse structures for clash detection: {pose_pdb}")
        return True  # Assume clash if parsing fails
    
    # Get ligand atoms
    ligand_atoms = []
    for atom in ligand_structure.get_atoms():
        if atom.element != 'H':
            ligand_atoms.append(atom.coord)
    
    if not ligand_atoms:
        return True
    
    ligand_atoms = np.array(ligand_atoms)
    
    # Get receptor atoms
    receptor_atoms = []
    for atom in receptor_structure.get_atoms():
        if atom.element != 'H':
            receptor_atoms.append(atom.coord)
    
    if not receptor_atoms:
        return True
    
    receptor_atoms = np.array(receptor_atoms)
    
    # Calculate minimum distance
    min_distance = float('inf')
    for lig_coord in ligand_atoms:
        distances = np.linalg.norm(receptor_atoms - lig_coord, axis=1)
        min_dist = np.min(distances)
        min_distance = min(min_distance, min_dist)
    
    has_clash = min_distance < clash_distance
    
    return has_clash


def calculate_ligand_rmsd(pose_pdb, target_ligand_pdb):
    """
    Calculate RMSD between docked pose and target ligand
    Uses heavy atoms only
    """
    parser = PDBParser(QUIET=True)
    
    try:
        pose_structure = parser.get_structure('pose', pose_pdb)
        target_structure = parser.get_structure('target', target_ligand_pdb)
    except:
        logger.warning(f"Failed to parse structures for RMSD: {pose_pdb}")
        return None
    
    # Get heavy atom coordinates
    pose_atoms = []
    for atom in pose_structure.get_atoms():
        if atom.element != 'H':
            pose_atoms.append(atom)
    
    target_atoms = []
    for atom in target_structure.get_atoms():
        if atom.element != 'H':
            target_atoms.append(atom)
    
    # Check if same number of atoms
    if len(pose_atoms) != len(target_atoms):
        logger.warning(f"Different number of atoms: pose={len(pose_atoms)}, target={len(target_atoms)}")
        # Try to match by distance if different sizes
        if len(pose_atoms) == 0 or len(target_atoms) == 0:
            return None
        
        # Calculate centroid distance as proxy
        pose_coords = np.array([atom.coord for atom in pose_atoms])
        target_coords = np.array([atom.coord for atom in target_atoms])
        pose_center = np.mean(pose_coords, axis=0)
        target_center = np.mean(target_coords, axis=0)
        rmsd = np.linalg.norm(pose_center - target_center)
        return rmsd
    
    # Calculate RMSD using superimposer
    super_imposer = Superimposer()
    super_imposer.set_atoms(target_atoms, pose_atoms)
    rmsd = super_imposer.rms
    
    return rmsd


def calculate_pocket_overlap(pose_pdb, target_ligand_pdb, receptor_pdb, cutoff=5.0):
    """
    Calculate Jaccard similarity of interacting residues
    Returns: overlap score (0-1)
    """
    parser = PDBParser(QUIET=True)
    
    try:
        pose_structure = parser.get_structure('pose', pose_pdb)
        target_structure = parser.get_structure('target', target_ligand_pdb)
        receptor_structure = parser.get_structure('receptor', receptor_pdb)
    except:
        return 0.0
    
    def get_interacting_residues(ligand_structure, receptor_structure, cutoff):
        """Get set of residue IDs within cutoff of ligand"""
        ligand_coords = np.array([atom.coord for atom in ligand_structure.get_atoms() if atom.element != 'H'])
        
        interacting = set()
        for residue in receptor_structure.get_residues():
            for atom in residue.get_atoms():
                if atom.element == 'H':
                    continue
                distances = np.linalg.norm(ligand_coords - atom.coord, axis=1)
                if np.min(distances) <= cutoff:
                    # Use chain + resid as unique identifier
                    res_id = f"{residue.parent.id}_{residue.id[1]}"
                    interacting.add(res_id)
                    break
        
        return interacting
    
    pose_pocket = get_interacting_residues(pose_structure, receptor_structure, cutoff)
    target_pocket = get_interacting_residues(target_structure, receptor_structure, cutoff)
    
    if not pose_pocket or not target_pocket:
        return 0.0
    
    # Jaccard similarity
    intersection = len(pose_pocket & target_pocket)
    union = len(pose_pocket | target_pocket)
    
    jaccard = intersection / union if union > 0 else 0.0
    
    return jaccard


def process_all_seeds(config, dirs):
    """
    Process all Vina output files from multiple seeds
    Returns: DataFrame with all poses and metrics
    """
    all_poses = []
    
    receptor_pdb = dirs['prepared'] / 'receptor.pdb'
    target_ligand_pdb = dirs['prepared'] / 'target_ligand.pdb'
    
    n_seeds = config['docking']['n_seeds']
    
    for seed in range(n_seeds):
        pdbqt_file = dirs['docking'] / f'vina_out_seed{seed}.pdbqt'
        
        if not pdbqt_file.exists():
            logger.warning(f"Vina output not found: {pdbqt_file}")
            continue
        
        logger.info(f"Processing seed {seed}: {pdbqt_file}")
        
        # Parse PDBQT
        poses = parse_vina_pdbqt(pdbqt_file)
        
        for pose in poses:
            pose_name = f"seed{seed}_pose{pose['pose_id']}"
            pose_pdb = dirs['filtered'] / f"{pose_name}.pdb"
            
            # Convert to PDB
            pdbqt_to_pdb(pose['pdbqt_lines'], pose_pdb)
            
            # Check clashes
            has_clash = check_clashes(
                pose_pdb, 
                receptor_pdb, 
                config['filtering']['clash_distance']
            )
            
            # Calculate RMSD
            rmsd = calculate_ligand_rmsd(pose_pdb, target_ligand_pdb)
            
            # Calculate pocket overlap
            pocket_overlap = calculate_pocket_overlap(
                pose_pdb,
                target_ligand_pdb,
                receptor_pdb
            )
            
            all_poses.append({
                'pose_name': pose_name,
                'seed': seed,
                'pose_id': pose['pose_id'],
                'vina_score': pose['score'],
                'has_clash': has_clash,
                'ligand_rmsd': rmsd,
                'pocket_overlap': pocket_overlap,
                'pdb_file': str(pose_pdb),
                'status': 'raw'
            })
    
    df = pd.DataFrame(all_poses)
    return df


def filter_poses(df, config):
    """
    Filter poses based on criteria
    """
    logger.info(f"\nFiltering poses...")
    logger.info(f"  Total poses: {len(df)}")
    
    # Filter by Vina score
    score_cutoff = config['filtering']['vina_score_cutoff']
    df_filtered = df[df['vina_score'] <= score_cutoff].copy()
    logger.info(f"  After score filter (>{score_cutoff}): {len(df_filtered)}")
    
    # Filter by clashes
    df_filtered = df_filtered[~df_filtered['has_clash']].copy()
    logger.info(f"  After clash filter: {len(df_filtered)}")
    
    # Limit number of poses
    max_poses = config['filtering']['max_poses_to_relax']
    if len(df_filtered) > max_poses:
        df_filtered = df_filtered.nsmallest(max_poses, 'vina_score')
        logger.info(f"  Limited to top {max_poses} poses")
    
    # Flag possible allosteric binding
    df_filtered['possible_allosteric'] = (
        (df_filtered['ligand_rmsd'] > 10.0) | 
        (df_filtered['pocket_overlap'] < 0.3)
    )
    
    n_allosteric = df_filtered['possible_allosteric'].sum()
    if n_allosteric > 0:
        logger.warning(f"  {n_allosteric} poses flagged as possible allosteric binding!")
    
    df_filtered['status'] = 'filtered'
    
    return df_filtered


def main():
    parser = argparse.ArgumentParser(description='Post-process Vina docking results')
    parser.add_argument('config', help='Configuration YAML file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get directories
    output_dir = Path(config['output_dir'])
    dirs = {
        'prepared': output_dir / 'prepared',
        'docking': output_dir / 'docking',
        'filtered': output_dir / 'filtered',
        'relaxed': output_dir / 'relaxed',
        'final': output_dir / 'final',
        'logs': output_dir / 'logs',
    }
    
    logger.info("\n" + "="*60)
    logger.info("VINA POST-PROCESSING")
    logger.info("\n" + "="*60)
    
    # Process all seeds
    logger.info("\n1. Parsing and analyzing all poses...")
    df_all = process_all_seeds(config, dirs)
    
    # Save raw results
    raw_csv = dirs['filtered'] / 'poses_raw.csv'
    df_all.to_csv(raw_csv, index=False)
    logger.info(f"\nSaved raw results: {raw_csv}")
    
    # Filter poses
    logger.info("\n2. Filtering poses...")
    df_filtered = filter_poses(df_all, config)
    
    # Save filtered results
    filtered_csv = dirs['filtered'] / 'poses_filtered.csv'
    df_filtered.to_csv(filtered_csv, index=False)
    logger.info(f"\nSaved filtered results: {filtered_csv}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("POST-PROCESSING COMPLETE")
    logger.info("\n" + "="*60)
    logger.info(f"\nResults:")
    logger.info(f"  Total poses: {len(df_all)}")
    logger.info(f"  Filtered poses: {len(df_filtered)}")
    logger.info(f"  Possible allosteric: {df_filtered['possible_allosteric'].sum()}")
    
    if len(df_filtered) > 0:
        logger.info(f"\nScore statistics:")
        logger.info(f"  Best score: {df_filtered['vina_score'].min():.2f}")
        logger.info(f"  Mean score: {df_filtered['vina_score'].mean():.2f}")
        logger.info(f"  Worst score: {df_filtered['vina_score'].max():.2f}")
        
        logger.info(f"\nRMSD statistics:")
        logger.info(f"  Min RMSD: {df_filtered['ligand_rmsd'].min():.2f} Å")
        logger.info(f"  Mean RMSD: {df_filtered['ligand_rmsd'].mean():.2f} Å")
        logger.info(f"  Max RMSD: {df_filtered['ligand_rmsd'].max():.2f} Å")
        
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Open analysis.ipynb to:")
        logger.info(f"     - Calculate ProLIF interaction fingerprints")
        logger.info(f"     - Visualize poses")
        logger.info(f"     - Select poses for relaxation")
        logger.info(f"  2. Run: python fastrelax_prep.py {args.config}")
    else:
        logger.error("\nNo poses passed filtering!")
        logger.info("  Consider:")
        logger.info("  - Lowering score cutoff")
        logger.info("  - Adjusting clash distance")
        logger.info("  - Increasing box size or running blind docking")


if __name__ == '__main__':
    main()