"""
FastRelax Preparation: Prepare inputs for Rosetta FastRelax
Generates ligand params, complex PDBs, and SLURM scripts
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
import argparse
import subprocess
import logging
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LigandSelect(Select):
    """Select only ligand atoms"""
    def __init__(self, ligand_chain='X', ligand_resid=1):
        self.ligand_chain = ligand_chain
        self.ligand_resid = ligand_resid
        
    def accept_residue(self, residue):
        return True  # Accept all for now


def load_config(config_file):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def combine_receptor_ligand(receptor_pdb, ligand_pdb, output_pdb, ligand_chain='X', ligand_resid=1):
    """
    Combine receptor and ligand into single PDB file
    Assign ligand to chain X with resid 1
    """
    parser = PDBParser(QUIET=True)
    receptor_structure = parser.get_structure('receptor', receptor_pdb)
    ligand_structure = parser.get_structure('ligand', ligand_pdb)
    
    # Get ligand residue
    ligand_residue = None
    for residue in ligand_structure.get_residues():
        ligand_residue = residue
        break
    
    if ligand_residue is None:
        raise ValueError(f"No residue found in ligand PDB: {ligand_pdb}")
    
    # Write combined PDB
    with open(output_pdb, 'w') as out:
        # Write receptor
        with open(receptor_pdb, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM', 'TER')):
                    out.write(line)
        
        # Write ligand with modified chain and resid
        ligand_resname = ligand_residue.resname
        atom_serial = 1
        
        with open(ligand_pdb, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    # Modify chain and resid
                    new_line = (
                        line[:21] +  # Up to chain
                        ligand_chain +  # New chain
                        f"{ligand_resid:>4}" +  # New resid
                        line[26:]  # Rest of line
                    )
                    out.write(new_line)
        
        out.write("END\n")
    
    logger.info(f"Combined receptor and ligand: {output_pdb}")
    return ligand_resname


def generate_ligand_params(ligand_pdb, output_params, ligand_resname='LIG'):
    """
    Generate Rosetta ligand params file using molfile_to_params.py
    """
    # First convert PDB to MOL/SDF if needed
    ligand_path = Path(ligand_pdb)
    mol_file = ligand_path.parent / f"{ligand_path.stem}.mol"
    
    # Convert PDB to MOL using obabel
    cmd = [
        'obabel',
        str(ligand_pdb),
        '-O', str(mol_file),
        '-h'  # Add hydrogens
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Converted ligand to MOL: {mol_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert ligand to MOL: {e}")
        raise
    
    # Generate params using molfile_to_params.py
    # This script is part of Rosetta
    cmd = [
        'python2',  # molfile_to_params often requires Python 2
        '/home/yunmin/src/Rosetta/main/source/scripts/python/public/molfile_to_params.py',
        str(mol_file),
        '-n', ligand_resname,  # Residue name
        '-p', str(output_params.stem),  # Output prefix
        '--clobber'  # Overwrite existing
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=str(output_params.parent)
        )
        logger.info(f"Generated ligand params: {output_params}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate params: {e.stderr}")
        raise
    
    # The script generates <name>.params, find it
    generated_params = output_params.parent / f"{output_params.stem}.params"
    if generated_params.exists() and generated_params != output_params:
        shutil.move(str(generated_params), str(output_params))
    
    if not output_params.exists():
        raise FileNotFoundError(f"Params file not generated: {output_params}")
    
    return output_params


def create_ligand_flags_file(params_file, output_flags):
    """
    Create ligand_flags file for Rosetta
    Format:
    -extra_res_fa <params_file>
    """
    with open(output_flags, 'w') as f:
        f.write(f"-extra_res_fa {params_file}\n")
    
    logger.info(f"Created ligand flags file: {output_flags}")


def identify_pocket_residues(complex_pdb, ligand_chain='X', ligand_resid=1, radius=8.0):
    """
    Identify pocket residues within radius of ligand
    Returns: list of residue indices (Rosetta numbering, 1-indexed)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', complex_pdb)
    
    # Get ligand coordinates
    ligand_coords = []
    for model in structure:
        for chain in model:
            if chain.id == ligand_chain:
                for residue in chain:
                    if residue.id[1] == ligand_resid:
                        for atom in residue.get_atoms():
                            if atom.element != 'H':
                                ligand_coords.append(atom.coord)
    
    if not ligand_coords:
        logger.warning(f"No ligand found at chain {ligand_chain}, resid {ligand_resid}")
        return []
    
    ligand_coords = np.array(ligand_coords)
    
    # Find pocket residues
    pocket_residues = []
    rosetta_index = 1
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip ligand itself
                if chain.id == ligand_chain and residue.id[1] == ligand_resid:
                    continue
                
                # Skip non-protein residues
                if residue.id[0] != ' ':
                    continue
                
                # Check distance to ligand
                for atom in residue.get_atoms():
                    if atom.element == 'H':
                        continue
                    
                    distances = np.linalg.norm(ligand_coords - atom.coord, axis=1)
                    min_dist = np.min(distances)
                    
                    if min_dist <= radius:
                        pocket_residues.append(rosetta_index)
                        break
                
                rosetta_index += 1
    
    logger.info(f"Found {len(pocket_residues)} pocket residues within {radius}Å")
    return pocket_residues


def generate_fastrelax_slurm_script(config, dirs, pose_name, complex_pdb, ligand_flags):
    """
    Generate SLURM script for FastRelax using existing script structure
    """
    script_file = dirs['relaxed'] / f"fastrelax_{pose_name}.sh"
    log_file = dirs['logs'] / f"fastrelax_{pose_name}.log"
    output_prefix = dirs['relaxed'] / pose_name
    
    rosetta_scripts = config['paths']['rosetta_scripts']
    rosetta_db = config['paths']['rosetta_db']
    xml_script = "/home/yunmin/proj/LigandMPNN/fastrelax/relax_sc_simple_hb_offrepack_edit.xml"
    
    with open(script_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name=relax_{pose_name}\n")
        f.write(f"#SBATCH --partition={config['slurm']['partition']}\n")
        f.write(f"#SBATCH --time={config['slurm']['time']}\n")
        f.write(f"#SBATCH --mem={config['slurm']['mem']}\n")
        f.write(f"#SBATCH --cpus-per-task=4\n")
        f.write(f"#SBATCH --output={log_file}\n")
        f.write(f"\n")
        f.write(f"echo \"Starting FastRelax for {pose_name}\"\n")
        f.write(f"echo \"Complex: {complex_pdb}\"\n")
        f.write(f"echo \"Ligand flags: {ligand_flags}\"\n")
        f.write(f"\n")
        
        # Rosetta command similar to run_fastrelax_norepack.sh
        f.write(f"{rosetta_scripts} \\\n")
        f.write(f"  -database {rosetta_db} \\\n")
        f.write(f"  -parser:protocol {xml_script} \\\n")
        f.write(f"  -s {complex_pdb} \\\n")
        f.write(f"  @{ligand_flags} \\\n")
        f.write(f"  -out:prefix {output_prefix}_ \\\n")
        f.write(f"  -out:pdb \\\n")
        f.write(f"  -out:file:scorefile {output_prefix}_score.sc \\\n")
        f.write(f"  -nstruct {config['relaxation']['nstruct']} \\\n")
        f.write(f"  -overwrite \\\n")
        f.write(f"  -jd2:failed_job_exception false\n")
        f.write(f"\n")
        f.write(f"echo \"FastRelax for {pose_name} completed\"\n")
    
    # Make executable
    os.chmod(script_file, 0o755)
    
    logger.info(f"Generated SLURM script: {script_file}")
    return script_file


def prepare_pose_for_relaxation(pose_row, config, dirs):
    """
    Prepare a single pose for FastRelax
    Returns: dict with prepared files
    """
    pose_name = pose_row['pose_name']
    ligand_pdb = pose_row['pdb_file']
    
    logger.info(f"\nPreparing pose: {pose_name}")
    
    # Get receptor PDB
    receptor_pdb = dirs['prepared'] / 'receptor.pdb'
    
    # Combine receptor and ligand
    complex_pdb = dirs['relaxed'] / f"{pose_name}_input.pdb"
    ligand_resname = combine_receptor_ligand(
        receptor_pdb,
        ligand_pdb,
        complex_pdb,
        ligand_chain='X',
        ligand_resid=1
    )
    
    # Generate ligand params
    params_file = dirs['relaxed'] / f"{pose_name}_ligand.params"
    generate_ligand_params(ligand_pdb, params_file, ligand_resname)
    
    # Create ligand flags file
    flags_file = dirs['relaxed'] / f"{pose_name}_ligand_flags"
    create_ligand_flags_file(params_file, flags_file)
    
    # Identify pocket residues (optional, for reference)
    pocket_residues = identify_pocket_residues(
        complex_pdb,
        ligand_chain='X',
        ligand_resid=1,
        radius=config['relaxation']['pocket_radius']
    )
    
    # Generate SLURM script
    slurm_script = generate_fastrelax_slurm_script(
        config,
        dirs,
        pose_name,
        complex_pdb,
        flags_file
    )
    
    return {
        'pose_name': pose_name,
        'complex_pdb': str(complex_pdb),
        'params_file': str(params_file),
        'flags_file': str(flags_file),
        'slurm_script': str(slurm_script),
        'n_pocket_residues': len(pocket_residues),
        'pocket_residues': ','.join(map(str, pocket_residues))
    }


def main():
    parser = argparse.ArgumentParser(description='Prepare FastRelax inputs')
    parser.add_argument('config', help='Configuration YAML file')
    parser.add_argument('--poses', help='Specific poses to prepare (comma-separated)', default=None)
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
    
    logger.info("="*60)
    logger.info("FASTRELAX PREPARATION")
    logger.info("="*60)
    
    # Load filtered poses
    filtered_csv = dirs['filtered'] / 'poses_filtered.csv'
    if not filtered_csv.exists():
        logger.error(f"Filtered poses CSV not found: {filtered_csv}")
        logger.info("Please run vina_postprocess.py first")
        return
    
    df = pd.read_csv(filtered_csv)
    logger.info(f"\nLoaded {len(df)} filtered poses")
    
    # Select specific poses if specified
    if args.poses:
        pose_names = [p.strip() for p in args.poses.split(',')]
        df = df[df['pose_name'].isin(pose_names)]
        logger.info(f"Preparing {len(df)} specified poses: {pose_names}")
    
    if len(df) == 0:
        logger.error("No poses to prepare!")
        return
    
    # Prepare each pose
    logger.info(f"\nPreparing {len(df)} poses for FastRelax...")
    
    prepared_data = []
    slurm_scripts = []
    
    for idx, row in df.iterrows():
        try:
            result = prepare_pose_for_relaxation(row, config, dirs)
            prepared_data.append(result)
            slurm_scripts.append(result['slurm_script'])
        except Exception as e:
            logger.error(f"Failed to prepare {row['pose_name']}: {e}")
            continue
    
    # Save preparation summary
    df_prepared = pd.DataFrame(prepared_data)
    prep_csv = dirs['relaxed'] / 'fastrelax_prepared.csv'
    df_prepared.to_csv(prep_csv, index=False)
    logger.info(f"\nSaved preparation summary: {prep_csv}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PREPARATION COMPLETE")
    logger.info("="*60)
    logger.info(f"\nPrepared {len(prepared_data)} poses for FastRelax:")
    for data in prepared_data:
        logger.info(f"  - {data['pose_name']}: {data['n_pocket_residues']} pocket residues")
    
    logger.info(f"\nGenerated files:")
    logger.info(f"  - {len(prepared_data)} input complexes")
    logger.info(f"  - {len(prepared_data)} params files")
    logger.info(f"  - {len(prepared_data)} flags files")
    logger.info(f"  - {len(prepared_data)} SLURM scripts")
    
    logger.info(f"\nNext steps:")
    logger.info(f"  Submit FastRelax jobs to server:")
    for script in slurm_scripts:
        logger.info(f"    sbatch {script}")
    
    logger.info(f"\n  After relaxation completes, continue with analysis.ipynb")


if __name__ == '__main__':
    main()