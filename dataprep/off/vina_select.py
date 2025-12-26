"""
Vina Post-processing: Parse PDBQT results, filter poses, calculate metrics
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Superimposer
import argparse
import subprocess
import logging
from collections import defaultdict

# ProLIF for interaction fingerprints
import prolif as plf
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


### Post-processing functions ###
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
    """Convert PDBQT lines to PDB file (safe for RDKit)"""
    with open(output_pdb, "w") as f:
        for line in pdbqt_lines:
            if line.startswith(("ATOM", "HETATM")):
                # Columns based on PDBQT format
                atom_name = line[12:16]
                res_name  = line[17:20]
                chain_id  = line[21]
                res_seq   = line[22:26]
                x = line[30:38]
                y = line[38:46]
                z = line[46:54]

                # Infer element from atom name (fallback-safe)
                element = atom_name.strip()[0]
                if element.isdigit():
                    element = atom_name.strip()[1]

                pdb_line = (
                    f"{line[:6]}"
                    f"{line[6:11]}"
                    f"{atom_name}"
                    f"{res_name:>4}"
                    f"{chain_id}"
                    f"{res_seq}"
                    f"    {x}{y}{z}"
                    f"  1.00  0.00          {element:>2}\n"
                )
                f.write(pdb_line)

            elif line.startswith(("MODEL", "ENDMDL", "TER", "END")):
                f.write(line)

def make_complex_pdb(receptor_pdb: Path, ligand_pdb: Path, out_pdb: Path) -> None:
    """Write a protein-ligand complex PDB (simple concat)."""
    with open(out_pdb, "w") as w:
        with open(receptor_pdb, "r") as r:
            for line in r:
                if line.startswith("END"):
                    continue
                w.write(line)
        w.write("TER\n")
        with open(ligand_pdb, "r") as l:
            for line in l:
                if line.startswith(("MODEL", "ENDMDL", "END")):
                    continue
                w.write(line)
        w.write("END\n")

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


def calculate_ligand_com(pose_pdb, target_ligand_pdb):
    """
    Calculate COM distance between docked pose and target ligand
    Uses heavy atoms only
    """
    parser = PDBParser(QUIET=True)
    
    try:
        pose_structure = parser.get_structure('pose', pose_pdb)
        target_structure = parser.get_structure('target', target_ligand_pdb)
    except:
        logger.warning(f"Failed to parse structures for COM distance: {pose_pdb}")
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
    
    # Calculate centroid distance
    pose_coords = np.array([atom.coord for atom in pose_atoms])
    target_coords = np.array([atom.coord for atom in target_atoms])
    pose_center = np.mean(pose_coords, axis=0)
    target_center = np.mean(target_coords, axis=0)
    com_dist = np.linalg.norm(pose_center - target_center)
    
    return com_dist


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
    
    # n_seeds = config['docking']['n_seeds']
    n_seeds = 3
    
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
            
            complex_pdb = dirs["filtered"] / f"{pose_name}_complex.pdb"
            try:
                make_complex_pdb(receptor_pdb, pose_pdb, complex_pdb)
            except Exception as e:
                logger.warning(f"Failed to create complex PDB for {pose_name}: {e}")

            # Check clashes
            has_clash = check_clashes(
                pose_pdb, 
                receptor_pdb, 
                config['filtering']['clash_distance']
            )
            
            # Calculate COM distance
            com_dist = calculate_ligand_com(pose_pdb, target_ligand_pdb)
            
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
                'ligand_com_dist': com_dist,
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
        (df_filtered['ligand_com_dist'] > 10.0) | 
        (df_filtered['pocket_overlap'] < 0.3)
    )
    
    n_allosteric = df_filtered['possible_allosteric'].sum()
    if n_allosteric > 0:
        logger.warning(f"  {n_allosteric} poses flagged as possible allosteric binding!")
    
    df_filtered['status'] = 'filtered'
    
    return df_filtered

### Pose selection functions ###
def plot_score_com_dist(df: pd.DataFrame, dirs: dict):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Score distribution
    df['vina_score'] = pd.to_numeric(df['vina_score'], errors='coerce')
    axes[0].hist(df['vina_score'], bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(df['vina_score'].median(), color='red', linestyle='--', label='Median')
    axes[0].set_xlabel('Vina Score (kcal/mol)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Score Distribution')
    axes[0].legend()

    # COM distance distribution
    axes[1].hist(df['ligand_com_dist'], bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(df['ligand_com_dist'].median(), color='red', linestyle='--', label='Median')
    axes[1].set_xlabel('Ligand COM Distance (Å)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('COM Distance Distribution')
    axes[1].legend()

    # Score vs COM distance scatter plot
    scatter = axes[2].scatter(df['ligand_com_dist'], df['vina_score'], 
                            c=df['pocket_overlap'], cmap='viridis', 
                            s=100, alpha=0.6, edgecolor='black')
    axes[2].set_xlabel('Ligand COM Distance (Å)')
    axes[2].set_ylabel('Vina Score (kcal/mol)')
    axes[2].set_title('Score vs COM Distance (colored by pocket overlap)')
    cbar = plt.colorbar(scatter, ax=axes[2])
    cbar.set_label('Pocket Overlap')

    # Flag allosteric
    # if df['possible_allosteric'].sum() > 0:
    #     allosteric = df[df['possible_allosteric']]
    #     axes[2].scatter(allosteric['ligand_com_dist'], allosteric['vina_score'],
    #                    marker='x', s=200, c='red', label='Possible allosteric', linewidths=3)
    #     axes[2].legend()

    plt.tight_layout()
    plt.savefig(dirs['filtered'] / 'score_com_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ Saved: score_com_distribution.png")

def calculate_prolif_fingerprints(pose_pdb: Path, receptor_pdb: Path) -> dict:
    """
    Calculate interaction fingerprints using ProLIF
    Returns: DataFrame with interaction counts
    """
    try:
        # Load structures
        prot_rdkit = Chem.MolFromPDBFile(str(receptor_pdb), removeHs=False)
        lig_rdkit = Chem.MolFromPDBFile(str(pose_pdb), removeHs=False, sanitize=False)
        
        if prot_rdkit is None:
            raise ValueError("RDKit failed to read receptor PDB (prot_rdkit is None)")
        if lig_rdkit is None:
            raise ValueError("RDKit failed to read pose PDB (lig_rdkit is None)")

        try:
            Chem.SanitizeMol(lig_rdkit)
        except Exception as se:
            raise ValueError(f"Ligand sanitize failed: {se}")
        
        prot = plf.Molecule(prot_rdkit)
        lig = plf.Molecule(lig_rdkit)

        # Define interactions to detect
        fp = plf.Fingerprint(
            interactions=[
                'HBDonor', 'HBAcceptor', 'PiStacking', 'PiCation',
                'Hydrophobic', 'MetalAcceptor'
            ]
        )
        
        # Calculate fingerprint
        fp.run_from_iterable([lig], prot)
        
        # Convert to dataframe
        df_fp = fp.to_dataframe()

        # Count interactions
        interaction_counts = {}
        for col in df_fp.columns:
            interaction_type = col[1]  # (residue, interaction_type)
            if interaction_type not in interaction_counts:
                interaction_counts[interaction_type] = 0
            interaction_counts[interaction_type] += df_fp[col].sum()
        
        return interaction_counts, df_fp
    
    except Exception as e:
        print(f"Warning: ProLIF failed for {pose_pdb}: {e}")
        return {}, None

def compute_pose_fingerprints(df: pd.DataFrame, dirs: dict) -> pd.DataFrame:
    """
    Calculate fingerprints for all poses
    """
    receptor_pdb = dirs['prepared'] / 'receptor.pdb'

    print("Calculating ProLIF fingerprints for all poses...")
    print("This may take a few minutes...\n")

    interaction_data = []

    for idx, row in df.iterrows():
        pose_name = row['pose_name']
        pose_pdb = row['pdb_file']
        
        print(f"Processing {pose_name}...", end=' ')
        
        counts, df_fp = calculate_prolif_fingerprints(pose_pdb, receptor_pdb)

        if counts:
            counts['pose_name'] = pose_name
            interaction_data.append(counts)
            print("✓")
        else:
            print("✗")

    # Create interaction dataframe
    df_interactions = pd.DataFrame(interaction_data).fillna(0)
    if not df_interactions.empty:
        df_interactions.set_index('pose_name', inplace=True)
    else:
        df_interactions = pd.DataFrame(index=pd.Index([], name='pose_name'))

    # Merge with main dataframe
    df = df.merge(df_interactions, left_on='pose_name', right_index=True, how='left')

    # Save updated dataframe
    interaction_csv = dirs['filtered'] / 'poses_with_interactions.csv'
    df.to_csv(interaction_csv, index=False)

    print(f"\n✓ Saved interactions: {interaction_csv}")
    print(f"\nInteraction summary:")
    print(df_interactions.describe())

    return df_interactions

def plot_int_heatmap(df_interactions: pd.DataFrame, dirs: dict) -> list:
    # Create interaction heatmap
    interaction_cols = df_interactions.columns.tolist()

    if len(interaction_cols) > 0:
        plt.figure(figsize=(12, 8))
        
        # Sort by total interactions
        df_interactions['total'] = df_interactions.sum(axis=1)
        df_sorted = df_interactions.sort_values('total', ascending=False).drop('total', axis=1)
        
        sns.heatmap(df_sorted, cmap='YlOrRd', annot=True, fmt='.0f', 
                    cbar_kws={'label': 'Interaction Count'})
        plt.xlabel('Interaction Type')
        plt.ylabel('Pose')
        plt.title('Interaction Fingerprint Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(dirs['filtered'] / 'interaction_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Saved: interaction_heatmap.png")
    else:
        print("No interactions detected")

def robust_minmax(s: pd.Series, q_low=0.05, q_high=0.95) -> pd.Series:
    """
    Robust min-max normalization to [0,1] using quantile clipping
    5th and 95th percentiles are used to clip extremes
    Params:
        - s: input Series
        - q_low: lower quantile
        - q_high: upper quantile
    Returns: normalized Series
    """
    lo = s.quantile(q_low)
    hi = s.quantile(q_high)
    s_clip = s.clip(lower=lo, upper=hi)

    den = (hi - lo)
    if den == 0 or pd.isna(den):
        return pd.Series(0.5, index=s.index)
    return (s_clip - lo) / den

def pose_select(df: pd.DataFrame, df_interactions: pd.DataFrame, dirs: dict) -> pd.DataFrame:
    """
    Calculate composite score for ranking
    Lower is better: combine vina score, COM distance, and interaction counts
    Params:
        - df: DataFrame with pose data
        - interaction_cols: list of interaction count columns
    Returns: DataFrame with selected poses
    """
    # Normalize metrics to 0-1 scale
    # df['score_norm'] = (df['vina_score'] - df['vina_score'].min()) / (df['vina_score'].max() - df['vina_score'].min())
    # df['com_dist_norm'] = (df['ligand_com_dist'] - df['ligand_com_dist'].min()) / (df['ligand_com_dist'].max() - df['ligand_com_dist'].min())

    # Robust min-max normalization
    df["score_norm"] = robust_minmax(df["vina_score"], q_low=0.05, q_high=0.95)
    df["com_dist_norm"]  = robust_minmax(df["ligand_com_dist"],   q_low=0.05, q_high=0.95)

    interaction_cols = df_interactions.columns.tolist()

    # Count total interactions (higher is better, so invert)
    if len(interaction_cols) > 0:
        df['total_interactions'] = df[interaction_cols].sum(axis=1)
        df['interactions_norm'] = 1 - (df['total_interactions'] - df['total_interactions'].min()) / (df['total_interactions'].max() - df['total_interactions'].min())
    else:
        df['interactions_norm'] = 0.5

    # Composite score (lower is better)
    df['composite_score'] = (
        0.5 * df['score_norm'] +      # 50% weight on vina score
        0.2 * df['rmsd_norm'] +        # 20% weight on RMSD
        0.3 * df['interactions_norm']  # 30% weight on interactions (inverted)
    )

    # Sort by composite score
    df_sorted = df.sort_values('composite_score')

    print("Top 10 poses by composite score:")
    print(df_sorted[[
        'pose_name', 'vina_score', 'ligand_com_dist', 'pocket_overlap',
        'total_interactions', 'composite_score', 'possible_allosteric'
    ]].head(10))

    # Select top N for relaxation
    # n_to_relax = min(config['filtering']['max_poses_to_relax'], len(df_sorted))
    n_to_relax = 10
    df_selected = df_sorted.head(n_to_relax)

    print(f"\n✓ Selected {len(df_selected)} poses for FastRelax")

    # Save selection
    selected_csv = dirs['filtered'] / 'poses_selected_for_relax.csv'
    df_selected.to_csv(selected_csv, index=False)
    print(f"✓ Saved: {selected_csv}")

    return df_selected

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
    
    # filtered_csv = dirs['filtered'] / 'poses_filtered.csv'
    # if filtered_csv.exists():
    #     logger.info(f"\nFiltered results already exist: {filtered_csv}")
    #     logger.info("Skipping post-processing.")

    #     df_filtered = pd.read_csv(filtered_csv)

    #     print(f"Loaded {len(df_filtered)} filtered poses")
    #     print(f"\nScore range: {df_filtered['vina_score'].min():.2f} to {df_filtered['vina_score'].max():.2f} kcal/mol")
    #     print(f"COM Distance range: {df_filtered['ligand_com_dist'].min():.2f} to {df_filtered['ligand_com_dist'].max():.2f} Å")
    #     print(f"Pocket overlap range: {df_filtered['pocket_overlap'].min():.3f} to {df_filtered['pocket_overlap'].max():.3f}")
    #     print(f"\nPossible allosteric binding: {df_filtered['possible_allosteric'].sum()} poses")

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
        
        logger.info(f"\nCOM Distance statistics:")
        logger.info(f"  Min COM Distance: {df_filtered['ligand_com_dist'].min():.2f} Å")
        logger.info(f"  Mean COM Distance: {df_filtered['ligand_com_dist'].mean():.2f} Å")
        logger.info(f"  Max COM Distance: {df_filtered['ligand_com_dist'].max():.2f} Å")
        
        logger.info(f"\nNext steps:")
        logger.info(f"  Vina pose selection.")
    else:
        logger.error("\nNo poses passed filtering!")
        logger.info("  Consider:")
        logger.info("  - Lowering score cutoff")
        logger.info("  - Adjusting clash distance")
        logger.info("  - Increasing box size or running blind docking")

    # Plot vina score and COM distance distributions
    plot_score_com_dist(df_filtered, dirs)

    # Calculate interaction fingerprints
    df_interactions = compute_pose_fingerprints(df_filtered, dirs)
    plot_int_heatmap(df_interactions, dirs)

    # Pose selection
    df_selected = pose_select(df_filtered, df_interactions, dirs)

    if len(df_selected) > 0:
        logger.info("\nPost-processing and pose selection complete!")
        logger.info(f"\n Selected poses:")
        logger.info(f"{df_selected[['pose_name', 'vina_score', 'ligand_com_dist', 'pocket_overlap', 'composite_score']]}")

        logger.info(f"\nNext steps:")
        logger.info(f"  Review the selected poses and prepare for relaxation")
        logger.info(f"  Run: python fastrelax_prep.py {args.config}")
        logger.info(f"  Input CSV: {dirs['filtered'] / 'poses_selected_for_relax.csv'}")
    else: 
        logger.error("\nNo poses selected for relaxation!")

if __name__ == '__main__':
        main()
 