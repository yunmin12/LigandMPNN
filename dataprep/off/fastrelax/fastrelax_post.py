"""
FastRelax Post-processing: Analyze FastRelax results from multiple complex directories.
Calculates Rosetta scores, RMSD, and IFP metrics.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
import logging

import pyrosetta
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.scoring import CA_rmsd, all_atom_rmsd

import yaml
import warnings
from plip.structure.preparation import PDBComplex
from plip.basic import config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_params_file(relaxed_dir):
    """Find the 3-letter ligand params file in relaxed directory."""
    params_files = glob.glob(os.path.join(relaxed_dir, "*.params"))
    for pf in params_files:
        basename = os.path.basename(pf)
        name_part = basename.replace(".params", "")
        if len(name_part) == 3:
            return pf, name_part
    return None, None


def get_ligand_residue_index(pose):
    """Get the residue index of the ligand (last residue)."""
    return pose.total_residue()


def calculate_rosetta_scores(pose, scorefxn):
    """Calculate Rosetta total score and fa_rep."""
    total_score = scorefxn(pose)
    fa_rep = scorefxn.get_weight(pyrosetta.rosetta.core.scoring.fa_rep) * \
             pose.energies().total_energies()[pyrosetta.rosetta.core.scoring.fa_rep]
    return total_score, fa_rep


def calculate_ligand_rmsd(pose1, pose2):
    """Calculate ligand RMSD between two poses (last residue)."""
    lig_idx = pose1.total_residue()
    
    # Get ligand atoms from both poses
    lig1 = pose1.residue(lig_idx)
    lig2 = pose2.residue(lig_idx)
    
    if lig1.natoms() != lig2.natoms():
        logger.warning(f"Ligand atom count mismatch: {lig1.natoms()} vs {lig2.natoms()}")
        return np.nan
    
    # Calculate RMSD manually for ligand heavy atoms
    coords1 = []
    coords2 = []
    for i in range(1, lig1.natoms() + 1):
        if not lig1.atom_is_hydrogen(i):
            coords1.append([lig1.xyz(i).x, lig1.xyz(i).y, lig1.xyz(i).z])
            coords2.append([lig2.xyz(i).x, lig2.xyz(i).y, lig2.xyz(i).z])
    
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    
    if len(coords1) == 0:
        return np.nan
    
    rmsd = np.sqrt(np.mean(np.sum((coords1 - coords2)**2, axis=1)))
    return rmsd


def calculate_receptor_rmsd(pose1, pose2):
    """Calculate receptor CA-RMSD between two poses (excluding last residue)."""
    # Use CA_rmsd for receptor, excluding ligand
    nres = pose1.total_residue() - 1  # Exclude ligand
    return CA_rmsd(pose1, pose2, 1, nres)


def calculate_ifp_plip(complex_pdb):
    """
    Extract interaction fingerprint by PLIP.
    Returns dict: {interaction_type: set(residue_ids)} or None if failed
    """
    config.NOFIX = True
    config.SILENT = True

    interaction_dict = {
        "hbond": set(),
        "hydrophobic": set(),
        "saltbridge": set(),
        "pistacking": set(),
        "halogen": set(),
        "metal": set()
    }

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="invalid value encountered in arccos"
    )

    try:
        mol = PDBComplex()
        mol.load_pdb(str(complex_pdb))
        mol.analyze()
        
        # Check if any ligands found
        if not mol.ligands:
            logger.warning(f"No ligands found in {complex_pdb}")
            return None
        
        # Get interactions for first ligand
        if not mol.interaction_sets:
            logger.warning(f"No interactions found in {complex_pdb}")
            return interaction_dict
        
        interactions = next(iter(mol.interaction_sets.values()))
        # print([attr for attr in dir(interactions) if not attr.startswith("_")])

        # Hydrogen bonds
        if hasattr(interactions, 'hbonds_ldon') and hasattr(interactions, 'hbonds_pdon'):
            hbonds = interactions.hbonds_ldon + interactions.hbonds_pdon
            for hb in hbonds:
                interaction_dict["hbond"].add(
                    (hb.reschain, hb.restype, hb.resnr)
                )

        # Hydrophobic contacts
        if hasattr(interactions, 'all_hydrophobic_contacts'):
            for hp in interactions.all_hydrophobic_contacts:
                interaction_dict["hydrophobic"].add(
                    (hp.reschain, hp.restype, hp.resnr)
                )

        # Salt bridges
        if hasattr(interactions, 'saltbridge_lneg') and hasattr(interactions, 'saltbridge_pneg'):
            saltbridges = interactions.saltbridge_lneg + interactions.saltbridge_pneg
            for sb in saltbridges:
                interaction_dict["saltbridge"].add(
                    (sb.reschain, sb.restype, sb.resnr)
                )

        # Pi-stacking
        if hasattr(interactions, 'pistacking'):
            for ps in interactions.pistacking:
                interaction_dict["pistacking"].add(
                    (ps.reschain, ps.restype, ps.resnr)
                )

        # Halogen bonds
        if hasattr(interactions, 'halogen_bonds'):
            for hx in interactions.halogen_bonds:
                interaction_dict["halogen"].add(
                    (hx.reschain, hx.restype, hx.resnr)
                )

        # Metal coordination
        if hasattr(interactions, 'metal_complexes'):
            for mt in interactions.metal_complexes:
                interaction_dict["metal"].add(
                    (mt.reschain, mt.restype, mt.resnr)
                )

        return interaction_dict

    except Exception as e:
        logger.warning(f"PLIP failed on {complex_pdb}: {e}")
        return None


def calculate_ifp_similarity(ifp1, ifp2):
    """
    Calculate Tanimoto similarity from two PLIP IFP dictionaries.
    Returns: float (0-1) or None if calculation fails
    """
    try:
        if ifp1 is None or ifp2 is None:
            return None
        
        set1 = set().union(*ifp1.values())
        set2 = set().union(*ifp2.values())

        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = set1 & set2
        union = set1 | set2

        if len(union) == 0:
            return 0.0
        
        ifp_sim = len(intersection) / len(union)
        # print(f"Calculated IFP similarity: {ifp_sim}")
        return ifp_sim
    
    except Exception as e:
        logger.warning(f"IFP similarity calculation failed: {e}")
        return None

def load_config_yaml(config_path):
    """Load configuration YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) 
        return config
    except Exception as e:
        logger.error(f"Failed to load config {config_path}: {e}")
        return None


def process_complex_dir(complex_dir, base_dir):
    """Process a single complex directory and return metrics for all seeds."""
    logger.info(f"Processing {complex_dir}")
    
    results = []
    complex_name = os.path.basename(complex_dir)
    
    # Find paths
    receptor_pdb = os.path.join(complex_dir, "prepared", "receptor.pdb")
    relaxed_dir = os.path.join(complex_dir, "relaxed")
    filtered_dir = os.path.join(complex_dir, "filtered")
    
    if not os.path.exists(filtered_dir):
        logger.warning(f"Filtered directory not found: {filtered_dir}")
        return results
    
    # Find params file
    params_file, het_id = find_params_file(relaxed_dir)
    if not params_file:
        logger.warning(f"No 3-letter params file found in {relaxed_dir}")
        return results
    
    logger.info(f"Found ligand: {het_id}, params: {params_file}")
    
    # Initialize PyRosetta with params
    try:
        pyrosetta.init(f"-beta -extra_res_fa {params_file} -mute all")
        scorefxn = pyrosetta.get_fa_scorefxn()
    except Exception as e:
        logger.error(f"Failed to initialize PyRosetta: {e}")
        return results
    
    # Load target complex
    target_complex_pdb = os.path.join(filtered_dir, "target_complex.pdb")
    if not os.path.exists(target_complex_pdb):
        logger.warning(f"Target complex not found: {target_complex_pdb}")
        target_pose = None
    else:
        try:
            target_pose = pose_from_pdb(target_complex_pdb)
            logger.info(f"Loaded target complex: {target_complex_pdb}")
        except Exception as e:
            logger.error(f"Failed to load target complex: {e}")
            target_pose = None
    
    # Process each seed
    seed_poses = {}
    for seed_idx in range(5):
        seed_name = f"seed{seed_idx}"
        before_pdb = os.path.join(filtered_dir, f"{seed_name}_complex.pdb")
        after_pdb = os.path.join(filtered_dir, f"{seed_name}_complex_betarelax_111.pdb")
        
        if not os.path.exists(before_pdb):
            logger.warning(f"Before relax PDB not found: {before_pdb}")
            continue
        
        if not os.path.exists(after_pdb):
            logger.warning(f"After relax PDB not found: {after_pdb}")
            continue
        
        try:
            # Load poses
            pose_before = pose_from_pdb(before_pdb)
            pose_after = pose_from_pdb(after_pdb)
            seed_poses[seed_idx] = pose_after
            
            # Calculate Rosetta scores
            total_score_before, fa_rep_before = calculate_rosetta_scores(pose_before, scorefxn)
            total_score_after, fa_rep_after = calculate_rosetta_scores(pose_after, scorefxn)
            total_score_ba = total_score_after - total_score_before
            fa_rep_ba = fa_rep_after - fa_rep_before
            
            # Calculate RMSD (before/after)
            lig_rmsd_ba = calculate_ligand_rmsd(pose_before, pose_after)
            rec_rmsd_ba = calculate_receptor_rmsd(pose_before, pose_after)
            
            # Calculate IFP if available
            ifp_before = None
            ifp_after = None
            ifp_sim = None
            if calculate_ifp_plip is not None:
                try:
                    ifp_before = calculate_ifp_plip(before_pdb)
                    ifp_after = calculate_ifp_plip(after_pdb)
                    if calculate_ifp_similarity is not None:
                        ifp_sim = calculate_ifp_similarity(ifp_before, ifp_after)
                except Exception as e:
                    logger.warning(f"Failed to calculate IFP: {e}")
            
            result = {
                'complex': complex_name,
                'seed': seed_name,
                'het_id': het_id,
                'total_score': total_score_after,
                'Δtotal_score': total_score_ba,
                'fa_rep': fa_rep_after,
                'Δfa_rep': fa_rep_ba,
                'lig_rmsd_before_after': lig_rmsd_ba,
                'rec_rmsd_before_after': rec_rmsd_ba,
                'ifp_similarity': ifp_sim,
                'before_pdb': before_pdb,
                'after_pdb': after_pdb
            }
            
            results.append(result)
            logger.info(f"  {seed_name}: score={total_score_after:.2f}, lig_rmsd={lig_rmsd_ba:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to process {seed_name}: {e}")
            continue
    
    # Calculate RMSD among seeds
    if len(seed_poses) > 1:
        seed_indices = sorted(seed_poses.keys())
        for i, result in enumerate(results):
            if result['seed'] not in [f"seed{idx}" for idx in seed_indices]:
                continue
            
            seed_idx = int(result['seed'].replace('seed', ''))
            if seed_idx not in seed_poses:
                continue
            
            pose_i = seed_poses[seed_idx]
            
            lig_rmsds = []
            rec_rmsds = []
            for other_idx in seed_indices:
                if other_idx == seed_idx:
                    continue
                pose_j = seed_poses[other_idx]
                lig_rmsds.append(calculate_ligand_rmsd(pose_i, pose_j))
                rec_rmsds.append(calculate_receptor_rmsd(pose_i, pose_j))
            
            result['lig_rmsd_among_seeds_mean'] = np.nanmean(lig_rmsds) if lig_rmsds else np.nan
            result['lig_rmsd_among_seeds_std'] = np.nanstd(lig_rmsds) if lig_rmsds else np.nan
            result['rec_rmsd_among_seeds_mean'] = np.nanmean(rec_rmsds) if rec_rmsds else np.nan
            result['rec_rmsd_among_seeds_std'] = np.nanstd(rec_rmsds) if rec_rmsds else np.nan
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Post-process FastRelax results')
    parser.add_argument('--base_dir', type=str, help='Base directory containing runs/complex_****_off** directories')
    parser.add_argument('--output', type=str, default='fastrelax_post_summary.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    base_dir = os.path.join(args.base_dir, "runs")
    output_csv = os.path.join(args.base_dir, "eval", args.output)
    
    if not os.path.exists(base_dir):
        logger.error(f"Base directory does not exist: {base_dir}")
        sys.exit(1)
    
    # Find all complex directories
    complex_dirs = sorted(glob.glob(os.path.join(base_dir, "complex_*_off[0-9][0-9]")))
    
    if not complex_dirs:
        logger.error(f"No complex directories found in {base_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(complex_dirs)} complex directories")
    
    # Process all complexes
    all_results = []
    for complex_dir in complex_dirs:
        results = process_complex_dir(complex_dir, base_dir)
        all_results.extend(results)
    
    # Create DataFrame and save
    if not all_results:
        logger.error("No results collected!")
        sys.exit(1)
    
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    
    logger.info(f"Saved {len(df)} records to {output_csv}")
    logger.info(f"Summary statistics:")
    logger.info(f"  Mean total_score: {df['total_score'].mean():.2f}")
    logger.info(f"  Mean fa_rep: {df['fa_rep'].mean():.2f}")
    logger.info(f"  Mean lig_rmsd_before_after: {df['lig_rmsd_before_after'].mean():.3f}")
    logger.info(f"  Mean rec_rmsd_before_after: {df['rec_rmsd_before_after'].mean():.3f}")


if __name__ == '__main__':
    main()