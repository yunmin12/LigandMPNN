"""
Vina Post-processing: Parse PDBQT results, convert to complex PDB, calculate metrics
Processes multiple complexes with multiple seeds
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select, Superimposer
import argparse
import logging
from collections import defaultdict
import glob
import re
import warnings
from plip.structure.preparation import PDBComplex
from plip.basic import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_vina_pdbqt(pdbqt_file):
    """
    Parse Vina output PDBQT file
    Returns: dict with best pose {pose_id, score, pdbqt_lines}
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
    
    # Return best pose (MODEL 1)
    if poses:
        return poses[0]
    return None

def pdbqt_to_pdb_lines(pdbqt_lines):
    """Convert PDBQT lines to PDB format lines"""
    pdb_lines = []
    for line in pdbqt_lines:
        if line.startswith(('ATOM', 'HETATM')):
            # PDBQT has extra columns (charge, type), truncate to standard PDB
            pdb_line = line[:66] + '\n'
            pdb_lines.append(pdb_line)
    return pdb_lines

def combine_receptor_ligand(receptor_pdb, ligand_pdb_lines, output_pdb):
    """
    Combine receptor PDB file and ligand PDB lines into a single complex PDB.
    Assigns the ligand chain 'Z' by default.
    Returns: 
        True if successful, False otherwise
    """
    try:
        with open(receptor_pdb, 'r') as r:
            receptor_lines = [
                line for line in r
                if not line.startswith(('END', 'ENDMDL'))
            ]

        ligand_clean = [
            line for line in ligand_pdb_lines
            if not line.startswith(('END', 'ENDMDL'))
        ]

        # Rewrite ligand chian ID to specified chain (default 'Z')
        ligand_with_chain = []
        for line in ligand_clean:
            if line.startswith(('ATOM', 'HETATM')):
                # Change chain ID (column 22, index 21)
                line = line[:21] + 'Z' + line[22:]
            ligand_with_chain.append(line)

        with open(output_pdb, 'w') as f:
            f.writelines(receptor_lines)
            # ensure separation between receptor and ligand
            f.write('TER\n')
            f.writelines(ligand_with_chain)
            f.write('END\n')

        return True

    except Exception as e:
        logger.error(f"Failed to combine PDB files: {e}")
        return False

def extract_ligand_from_complex(complex_pdb, output_ligand_pdb, ligand_chain=None, ligand_resname=None):
    """
    Extract ligand from original complex PDB
    If ligand_chain is provided, extract by chain ID
    Otherwise, extract by common ligand resnames (not standard amino acids)
    """
    parser = PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure('complex', complex_pdb)
    except Exception as e:
        logger.error(f"Failed to parse complex PDB {complex_pdb}: {e}")
        return False
    
    # Standard amino acids to exclude
    standard_residues = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'HOH', 'WAT'  # Water
    }
    
    ligand_atoms = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check if this is ligand
                is_ligand = False
                
                if ligand_chain and chain.id == ligand_chain:
                    is_ligand = True
                elif ligand_resname and residue.resname == ligand_resname:
                    is_ligand = True
                elif residue.resname not in standard_residues and residue.id[0] == ' ':
                    # HETATM that's not water or standard residue
                    is_ligand = True
                
                if is_ligand:
                    for atom in residue:
                        ligand_atoms.append(atom)
    
    if not ligand_atoms:
        logger.warning(f"No ligand atoms found in {complex_pdb}")
        return False
    
    # Save ligand
    io = PDBIO()
    
    class LigandSelect(Select):
        def __init__(self, atom_list):
            self.atom_list = atom_list
        
        def accept_atom(self, atom):
            return atom in self.atom_list
    
    io.set_structure(structure)
    io.save(str(output_ligand_pdb), LigandSelect(ligand_atoms))
    
    logger.info(f"Extracted {len(ligand_atoms)} ligand atoms from {complex_pdb}")
    return True

def calculate_ligand_rmsd_between_seeds(ligand_pdb_1, ligand_pdb_2):
    """
    Calculate RMSD between two ligand PDB files (different seeds).
    Uses heavy atoms only.
    Returns: float (RMSD in Angstroms) or None if calculation fails
    """
    parser = PDBParser(QUIET=True)
    
    try:
        structure_1 = parser.get_structure('ligand_1', ligand_pdb_1)
        structure_2 = parser.get_structure('ligand_2', ligand_pdb_2)
    except Exception as e:
        logger.warning(f"Failed to parse ligand structures for seed RMSD: {e}")
        return None
    
    # Get heavy atom coordinates
    atoms_1 = []
    for atom in structure_1.get_atoms():
        if atom.element != 'H':
            atoms_1.append(atom)
    
    atoms_2 = []
    for atom in structure_2.get_atoms():
        if atom.element != 'H':
            atoms_2.append(atom)
    
    # Check if same number of atoms
    if len(atoms_1) != len(atoms_2):
        logger.warning(f"Different number of atoms between seeds: seed1={len(atoms_1)}, seed2={len(atoms_2)}")
        return None
    
    if len(atoms_1) == 0:
        logger.warning("No heavy atoms found in ligand structures")
        return None
    
    # Calculate RMSD using superimposer
    super_imposer = Superimposer()
    super_imposer.set_atoms(atoms_1, atoms_2)
    rmsd = super_imposer.rms
    
    return rmsd

def calculate_ligand_com_distance(pose_pdb, target_ligand_pdb):
    """
    Calculate centroid (COM) distance between docked pose and target ligand
    Uses heavy atoms only
    """
    parser = PDBParser(QUIET=True)

    try:
        pose_structure = parser.get_structure('pose', pose_pdb)
        target_structure = parser.get_structure('target', target_ligand_pdb)
    except Exception as e:
        logger.warning(f"Failed to parse structures for COM distance: {e}")
        return None

    # Collect heavy-atom coordinates
    pose_coords = np.array(
        [atom.coord for atom in pose_structure.get_atoms() if atom.element != 'H']
    )
    target_coords = np.array(
        [atom.coord for atom in target_structure.get_atoms() if atom.element != 'H']
    )

    if pose_coords.size == 0 or target_coords.size == 0:
        logger.warning("No heavy atoms found for COM distance calculation")
        return None

    # Centroids
    pose_center = pose_coords.mean(axis=0)
    target_center = target_coords.mean(axis=0)

    # Euclidean distance
    distance = np.linalg.norm(pose_center - target_center)
    # print(f"Calculated COM distance: {distance}")
    return distance

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

def discover_complexes_for_config(base_dir, config_num):
    """
    Discover all complex_ids that have the given config_num.
    Globs for: runs/complex_*_off_{config_num:02d}
    Returns: sorted list of complex_ids
    """
    runs_dir = Path(base_dir) / 'runs'
    
    if not runs_dir.exists():
        logger.error(f"Runs directory not found: {runs_dir}")
        return []
    
    complex_ids = set()
    pattern = f"complex_*_off{config_num:02d}"
    
    for run_dir in runs_dir.glob(pattern):
        match = re.search(r'complex_(\d+)_off(\d+)', run_dir.name)
        if match:
            complex_id = int(match.group(1))
            complex_ids.add(complex_id)
    
    return sorted(complex_ids)

def find_run_dir(base_dir, complex_id, config_num):
    """
    Find the run directory for a given complex_id and config_num.
    Pattern: runs/complex_{complex_id:04d}_off_{config_num:02d}/
    Returns: (run_path, run_id) or (None, None) if not found
    """
    pattern = f"complex_{complex_id:04d}_off{config_num:02d}"
    runs_dir = Path(base_dir) / 'runs'
    
    run_dir = runs_dir / pattern
    
    if not run_dir.exists():
        return None, None
    
    run_id = run_dir.name
    return run_dir, run_id


def process_complex(base_dir, complex_id, config_num):
    """
    Process a single complex with given complex_id and config_num.
    Returns: list of result dicts
    """
    results = []
    
    # Find the run directory for this complex
    run_path, run_id = find_run_dir(base_dir, complex_id, config_num)
    
    if not run_path:
        logger.warning(f"No run directory found for complex_{complex_id:04d}_off{config_num:02d}")
        return results
    
    logger.info(f"\nProcessing complex_{complex_id:04d}_off{config_num:02d}")
    logger.info(f"  Run directory: {run_path.name}")
    logger.info(f"  Run ID: {run_id}")
    
    # Load config for this complex
    config_dir = Path(base_dir) / f'configs_{config_num:02d}'
    
    if not config_dir.exists():
        logger.error(f"Config directory not found: {config_dir}")
        return results
    
    config_file = config_dir / f'complex_{complex_id:04d}_off{config_num:02d}.yaml'
    
    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        return results
    
    config = load_config_yaml(config_file)
    
    if not config:
        return results
    
    # Get target PDB
    target_pdb = config.get('target_pdb', '')
    
    logger.info(f"  Config file: {config_file.name}")
    logger.info(f"  Target PDB: {target_pdb}")
    
    # Extract PDB ID and HET ID from target PDB filename
    pdb_id = None
    het_id = None
    
    if target_pdb:
        target_pdb_basename = os.path.splitext(os.path.basename(target_pdb))[0]
        m = re.match(r'^([A-Za-z0-9]{4})_([A-Za-z0-9]+)_', target_pdb_basename)
        if m:
            pdb_id = m.group(1)
            het_id = m.group(2)
            logger.info(f"  PDB ID: {pdb_id}, HET ID: {het_id}")
        else:
            logger.warning(f"  Could not extract PDB ID and HET ID from {target_pdb_basename}")
            pdb_id = 'UNKNOWN'
            het_id = 'UNKNOWN'
    else:
        logger.warning(f"  No target_pdb found in config")
        pdb_id = 'UNKNOWN'
        het_id = 'UNKNOWN'
    
    # Get paths
    docking_dir = run_path / 'docking'
    receptor_pdb = run_path / 'prepared' / 'receptor.pdb'
    filtered_dir = run_path / 'filtered'
    
    # Create filtered directory
    filtered_dir.mkdir(parents=True, exist_ok=True)
    
    # Check receptor exists
    if not receptor_pdb.exists():
        logger.error(f"  Receptor not found: {receptor_pdb}")
        return results
    
    # Extract target ligand (only if target_pdb exists)
    target_ligand_pdb = None
    if target_pdb and Path(target_pdb).exists():
        target_ligand_pdb = filtered_dir / 'target_ligand.pdb'
        extract_ligand_from_complex(target_pdb, target_ligand_pdb, 'Z', het_id)
    else:
        if target_pdb:
            logger.warning(f"  Target PDB not found: {target_pdb}")
        else:
            logger.warning(f"  No target PDB specified in config")
    
        # Store seed ligand PDBs for inter-seed RMSD calculation
    seed_ligand_pdbs = {}

    # Process each seed (0-4)
    for seed in range(5):
        pdbqt_file = docking_dir / f'vina_out_seed{seed}.pdbqt'
        
        if not pdbqt_file.exists():
            logger.warning(f"  Complex ID {complex_id:04d} Seed {seed}: PDBQT not found")
            results.append({
                'complex_id': complex_id,
                'config_num': config_num,
                'pdb_id': pdb_id,
                'ligand_het_id': het_id,
                'run_id': run_id,
                'seed': seed,
                'vina_score': None,
                'ligand_com_distance': None,
                'ligand_ifp_similarity': None,
                'ligand_rmsd_current_seed': None,
                'ligand_rmsd_avg_all_seeds': None,
                'pose_ifp': None,
                'target_ifp': None, 
                'target_pdb_path': target_pdb,
                'complex_pdb_path': None,
                'complex_pdb_saved': False,
                'num_receptor_atoms': 0,
                'num_ligand_atoms': 0,
                'notes': 'PDBQT file not found'
            })
            continue
        
        # Parse PDBQT - get best pose
        best_pose = parse_vina_pdbqt(pdbqt_file)

        if not best_pose:
            logger.warning(f"  Seed {seed}: Failed to parse PDBQT")
            results.append({
                'complex_id': complex_id,
                'config_num': config_num,
                'pdb_id': pdb_id,
                'ligand_het_id': het_id,
                'run_id': run_id,
                'seed': seed,
                'vina_score': None,
                'ligand_com_distance': None,
                'ligand_ifp_similarity': None,
                'ligand_rmsd_current_seed': None,
                'ligand_rmsd_avg_all_seeds': None,
                'pose_ifp': None,
                'target_ifp': None, 
                'target_pdb_path': target_pdb,
                'complex_pdb_path': None,
                'complex_pdb_saved': False,
                'num_receptor_atoms': 0,
                'num_ligand_atoms': 0,
                'notes': 'Failed to parse PDBQT'
            })
            continue
        
        vina_score = best_pose['score']

        # Convert ligand PDBQT to PDB lines
        ligand_pdb_lines = pdbqt_to_pdb_lines(best_pose['pdbqt_lines'])
        num_ligand_atoms = len(ligand_pdb_lines)
        
        # Save ligand only
        ligand_pdb = filtered_dir / f'seed{seed}_ligand.pdb'
        with open(ligand_pdb, 'w') as f:
            f.writelines(ligand_pdb_lines)
        
        # Store ligand PDB path for inter-seed RMSD calculation
        seed_ligand_pdbs[seed] = ligand_pdb

        # Combine with receptor
        complex_pdb = filtered_dir / f'seed{seed}_complex.pdb'
        save_success = combine_receptor_ligand(receptor_pdb, ligand_pdb_lines, complex_pdb)
        
        # Count receptor atoms
        num_receptor_atoms = 0
        if receptor_pdb.exists():
            with open(receptor_pdb, 'r') as f:
                num_receptor_atoms = sum(1 for line in f if line.startswith(('ATOM', 'HETATM')))
        
        # Calculate COM distance and IFP
        com_dist = None
        pose_ifp = None
        target_ifp = None
        ifp_sim = None
        notes = []
        
        if target_ligand_pdb and target_ligand_pdb.exists() and save_success:
            # Calculate ligand COM distance
            com_dist = calculate_ligand_com_distance(str(ligand_pdb), str(target_ligand_pdb))
            if com_dist is None:
                notes.append('COM distance calculation failed')
            
            # Calculate IFP by PLIP using complex PDB files
            try:
                pose_ifp = calculate_ifp_plip(str(complex_pdb))
                if pose_ifp is None:
                    notes.append('Pose IFP calculation failed')
            except Exception as e:
                logger.warning(f"  Seed {seed} Pose IFP error: {e}")
                pose_ifp = None
                notes.append('Pose IFP calculation error')
            
            try:
                # For target, create temporary complex with target ligand
                target_complex_pdb = filtered_dir / f'target_complex.pdb'
                target_ligand_pdb_for_ifp = filtered_dir / 'target_ligand.pdb'
                
                if target_ligand_pdb_for_ifp.exists():
                    # Read target ligand and receptor
                    with open(receptor_pdb, 'r') as r:
                        receptor_lines = [
                            line for line in r
                            if not line.startswith(('END', 'ENDMDL'))
                        ]
                    
                    with open(target_ligand_pdb_for_ifp, 'r') as l:
                        ligand_lines = [
                            line for line in l
                            if not line.startswith(('END', 'ENDMDL'))
                        ]
                    
                    # Create temporary target complex
                    with open(target_complex_pdb, 'w') as f:
                        f.writelines(receptor_lines)
                        f.write('TER\n')
                        f.writelines(ligand_lines)
                        f.write('END\n')
                    
                    target_ifp = calculate_ifp_plip(str(target_complex_pdb))
                    if target_ifp is None:
                        notes.append('Target IFP calculation failed')
            except Exception as e:
                logger.warning(f"  Seed {seed} Target IFP error: {e}")
                target_ifp = None
                notes.append('Target IFP calculation error')
            
            # Calculate IFP similarity
            if pose_ifp is not None and target_ifp is not None:
                ifp_sim = calculate_ifp_similarity(pose_ifp, target_ifp)
                if ifp_sim is None:
                    notes.append('IFP similarity calculation failed')
        else:
            if not target_ligand_pdb or not target_ligand_pdb.exists():
                notes.append('Target ligand not available')
            if not save_success:
                notes.append('Failed to save complex PDB')
        
        # Calculate ligand RMSD between current seed and other seeds
        ligand_rmsd_current_seed = None
        if len(seed_ligand_pdbs) > 0:
            rmsd_values = []
            for other_seed, other_ligand_pdb in seed_ligand_pdbs.items():
                if other_seed != seed:
                    rmsd = calculate_ligand_rmsd_between_seeds(
                        str(seed_ligand_pdbs[seed]),
                        str(other_ligand_pdb)
                    )
                    if rmsd is not None:
                        rmsd_values.append(rmsd)
            if rmsd_values:
                ligand_rmsd_current_seed = sum(rmsd_values) / len(rmsd_values)
        
        # Placeholder for ligand_rmsd_avg_all_seeds (will be filled later)
        results.append({
            'complex_id': complex_id,
            'config_num': config_num,
            'pdb_id': pdb_id,
            'ligand_het_id': het_id,
            'run_id': run_id,
            'seed': seed,
            'vina_score': vina_score,
            'ligand_com_distance': com_dist,
            'ligand_ifp_similarity': ifp_sim,
            'ligand_rmsd_current_seed': ligand_rmsd_current_seed,
            'ligand_rmsd_avg_all_seeds': None,  # Will be updated below
            'pose_ifp': str(pose_ifp) if pose_ifp else None,
            'target_ifp': str(target_ifp) if target_ifp else None,
            'target_pdb_path': target_pdb,
            'complex_pdb_path': str(complex_pdb) if save_success else None,
            'complex_pdb_saved': save_success,
            'num_receptor_atoms': num_receptor_atoms,
            'num_ligand_atoms': num_ligand_atoms,
            'notes': '; '.join(notes) if notes else 'OK'
        })
        
        com_dist_str = f"{com_dist:.2f} Å" if com_dist is not None else "N/A"
        ifp_sim_str = f"{ifp_sim:.2f}" if ifp_sim is not None else "N/A"
        ligand_rmsd_current_seed_str = f"{ligand_rmsd_current_seed:.2f} Å" if ligand_rmsd_current_seed is not None else "N/A"
        logger.info(f"  Seed {seed}: Vina Score={vina_score:.2f}, COM Distance={com_dist_str}, IFP Similarity={ifp_sim_str}, Ligand RMSD among seeds={ligand_rmsd_current_seed_str}")
    
    # Calculate average ligand RMSD among all seeds and update results
    all_seeds = sorted(seed_ligand_pdbs.keys())
    ligand_rmsd_values = []
    for i in range(len(all_seeds)):
        for j in range(i + 1, len(all_seeds)):
            seed_i = all_seeds[i]
            seed_j = all_seeds[j]
            rmsd = calculate_ligand_rmsd_between_seeds(
                str(seed_ligand_pdbs[seed_i]),
                str(seed_ligand_pdbs[seed_j])
            )
            if rmsd is not None:
                ligand_rmsd_values.append(rmsd)
    
    ligand_rmsd_avg_all_seeds = None
    if ligand_rmsd_values:
        ligand_rmsd_avg_all_seeds = sum(ligand_rmsd_values) / len(ligand_rmsd_values)
        logger.info(f"  Average Ligand RMSD among all seeds: {ligand_rmsd_avg_all_seeds:.2f} Å")
    else:
        logger.info(f"  No Ligand RMSD values calculated among seeds.")
    
    # Update all results with ligand_rmsd_avg_all_seeds
    for result in results:
        result['ligand_rmsd_avg_all_seeds'] = ligand_rmsd_avg_all_seeds
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Post-process Vina docking results for BDB dataset')
    parser.add_argument('--base_dir', required=True,
                       help='Base directory containing runs and configs')
    parser.add_argument('--config_num', type=int, required=True,
                       help='Config number to process (required)')
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    config_num = args.config_num
    
    logger.info("\n" + "="*60)
    logger.info("VINA POST-PROCESSING")
    logger.info("="*60)
    logger.info(f"\nBase directory: {base_dir}")
    logger.info(f"Config number: {config_num}")
    
    # Discover all complexes for this config number
    complex_ids = discover_complexes_for_config(base_dir, config_num)
    
    if not complex_ids:
        logger.error(f"No complexes found for config_num {config_num}")
        return
    
    logger.info(f"\nDiscovered {len(complex_ids)} complex(es) for config_{config_num}:")
    logger.info(f"  Complex IDs: {complex_ids}")
    
    # Process all complexes with this config number
    all_results = []
    
    for complex_id in complex_ids:
        results = process_complex(base_dir, complex_id, config_num)
        all_results.extend(results)
    
    if not all_results:
        logger.error("No results generated!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    eval_dir = base_dir / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_dir / f'vina_post_summary_config{config_num:02d}.csv'
    df.to_csv(output_path, index=False)
    
    logger.info("\n" + "="*60)
    logger.info("POST-PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"\nResults saved to: {output_path}")
    logger.info(f"\nTotal entries: {len(df)}")
    logger.info(f"Successful complex PDB saves: {df['complex_pdb_saved'].sum()}")
    
    if len(df) > 0:
        # Statistics
        valid_scores = df[df['vina_score'].notna()]
        if len(valid_scores) > 0:
            logger.info(f"\nVina Score statistics:")
            logger.info(f"  Best: {valid_scores['vina_score'].min():.2f}")
            logger.info(f"  Mean: {valid_scores['vina_score'].mean():.2f}")
            logger.info(f"  Worst: {valid_scores['vina_score'].max():.2f}")
        
        valid_com_dist = df[df['ligand_com_distance'].notna()]
        if len(valid_com_dist) > 0:
            logger.info(f"\nLigand COM distance statistics:")
            logger.info(f"  Min: {valid_com_dist['ligand_com_distance'].min():.2f} Å")
            logger.info(f"  Mean: {valid_com_dist['ligand_com_distance'].mean():.2f} Å")
            logger.info(f"  Max: {valid_com_dist['ligand_com_distance'].max():.2f} Å")
        
        valid_ifp = df[df['ligand_ifp_similarity'].notna()]
        if len(valid_ifp) > 0:
            logger.info(f"\nIFP Similarity statistics:")
            logger.info(f"  Min: {valid_ifp['ligand_ifp_similarity'].min():.2f}")
            logger.info(f"  Mean: {valid_ifp['ligand_ifp_similarity'].mean():.2f}")
            logger.info(f"  Max: {valid_ifp['ligand_ifp_similarity'].max():.2f}")
        
        # Check for issues
        issues = df[df['notes'] != 'OK']
        if len(issues) > 0:
            logger.warning(f"\n{len(issues)} entries had issues:")
            issue_counts = issues['notes'].value_counts()
            for note, count in issue_counts.items():
                logger.warning(f"  {note}: {count}")

if __name__ == '__main__':
    main()