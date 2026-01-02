"""
AutoDock Vina Preparation
Ligand & Receptor preparation, search box definition
Generates Vina config files and SLURM scripts
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
import argparse
import subprocess
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LigandSelect(Select):
    """Select only ligand residues"""
    def __init__(self, chain_id, resid):
        self.chain_id = chain_id
        self.resid = resid
        
    def accept_residue(self, residue):
        return (residue.parent.id == self.chain_id and 
                residue.id[1] == int(self.resid))


class ProteinSelect(Select):
    """Select protein residues (exclude ligand and waters)"""
    def __init__(self, exclude_chain=None, exclude_resid=None):
        self.exclude_chain = exclude_chain
        self.exclude_resid = exclude_resid
        
    def accept_residue(self, residue):
        # Exclude waters
        if residue.resname in ['HOH', 'WAT']:
            return False
        # Exclude target ligand
        if residue.id[0] != ' ':
            return False
        return True


def load_config(config_file):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config):
    """Create output directory structure"""
    output_dir = Path(config['output_dir'])
    dirs = {
        'prepared': output_dir / 'prepared',
        'docking': output_dir / 'docking',
        'filtered': output_dir / 'filtered',
        'relaxed': output_dir / 'relaxed',
        'final': output_dir / 'final',
        'logs': output_dir / 'logs',
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directory structure in {output_dir}")
    return dirs


def get_ligand_com_and_bounds(pdb_file, chain_id, resid):
    """
    Get center of mass and bounding box of target ligand
    Returns: com (x,y,z), box_size (x,y,z)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)
    
    coords = []
    for residue in structure[0][chain_id]:
        if residue.id[1] == int(resid):
            for atom in residue.get_atoms():
                if atom.element != 'H':  # Heavy atoms only
                    coords.append(atom.coord)
    
    if not coords:
        raise ValueError(f"No ligand found at chain {chain_id}, resid {resid}")
    
    coords = np.array(coords)
    com = np.mean(coords, axis=0)
    
    # Calculate bounding box
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    box_size = max_coords - min_coords
    
    logger.info(f"Target ligand COM: {com}")
    logger.info(f"Target ligand size: {box_size}")
    
    return com, box_size


def extract_ligand(pdb_file, chain_id, resid, output_pdb):
    """Extract ligand from complex and save as separate PDB"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)
    
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb), LigandSelect(chain_id, resid))
    
    logger.info(f"Extracted target ligand to {output_pdb}")


def clean_pdb(input_pdb, output_pdb):
    """
    Clean PDB file: keep only ATOM records, remove HETATM
    This helps with template matching in mk_prepare_receptor.py
    """
    atom_count = 0
    with open(input_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith('ATOM'):
                f_out.write(line)
                atom_count += 1
    
    logger.info(f"Cleaned PDB: {atom_count} ATOM records written to {output_pdb}")
    return atom_count


def prepare_receptor(pdb_file, chain_id, resid, output_pdb, output_pdbqt):
    """
    Prepare receptor: remove ligand and waters, save as PDB and PDBQT
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)
    
    # Save receptor PDB (no ligand, no waters)
    io = PDBIO()
    io.set_structure(structure)
    temp_pdb = str(output_pdb).replace('.pdb', '_temp.pdb')
    io.save(temp_pdb, ProteinSelect(chain_id, resid))
    
    # Clean PDB file (remove HETATM records that might cause issues)
    atom_count = clean_pdb(temp_pdb, str(output_pdb))
    
    if atom_count == 0:
        logger.error(f"Generated receptor PDB is empty!")
        return None
    
    logger.info(f"Prepared receptor PDB: {output_pdb} ({atom_count} atoms)")
    
    # Convert to PDBQT using mk_prepare_receptor (meeko)
    # Important: mk_prepare_receptor adds .pdbqt extension automatically
    output_base = str(output_pdbqt).replace('.pdbqt', '')
    log_file = str(output_pdb).replace('.pdb', '_prep.log')
    
    # Try method 1: mk_prepare_receptor with default settings
    logger.info("Attempting receptor preparation with mk_prepare_receptor.py...")
    cmd = [
        'mk_prepare_receptor.py',
        '-i', str(output_pdb),
        '-o', output_base,
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Write full output to log file
        with open(log_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("mk_prepare_receptor.py output\n")
            f.write("="*60 + "\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
            f.write("\n\nReturn code: " + str(result.returncode) + "\n")
        
        logger.info(f"mk_prepare_receptor.py log saved to: {log_file}")
        
        # Check if pdbqt file was created
        expected_pdbqt = output_base + '.pdbqt'
        
        if os.path.exists(expected_pdbqt):
            file_size = os.path.getsize(expected_pdbqt)
            logger.info(f"Generated PDBQT file size: {file_size} bytes")
            
            if file_size > 0:
                # Rename to expected filename if different
                if expected_pdbqt != str(output_pdbqt):
                    os.rename(expected_pdbqt, str(output_pdbqt))
                logger.info(f"✓ Prepared receptor PDBQT: {output_pdbqt}")
                
                # Show first few lines to verify content
                with open(output_pdbqt, 'r') as f:
                    first_lines = [f.readline() for _ in range(5)]
                    logger.info(f"First lines of PDBQT:\n{''.join(first_lines)}")
                
                return output_pdbqt
            else:
                logger.error(f"PDBQT file is empty!")
        else:
            logger.error(f"PDBQT file not created at expected location: {expected_pdbqt}")
        
        # If we got here, mk_prepare_receptor failed
        logger.warning("mk_prepare_receptor.py failed or produced empty file")
        logger.warning("Check the log file for details: " + log_file)
        
        # Show stderr to user
        if result.stderr:
            logger.error("Error output from mk_prepare_receptor.py:")
            for line in result.stderr.split('\n')[:20]:  # Show first 20 lines
                if line.strip():
                    logger.error(f"  {line}")
        
    except subprocess.TimeoutExpired:
        logger.error("mk_prepare_receptor.py timed out")
        return None
    except FileNotFoundError:
        logger.error("mk_prepare_receptor.py not found. Is meeko installed?")
        logger.info("Install with: pip install meeko")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None
    
    # Fallback: Try obabel if mk_prepare_receptor failed
    logger.warning("Attempting fallback method with obabel...")
    try:
        cmd = [
            'obabel',
            str(output_pdb),
            '-O', str(output_pdbqt),
            '-xr'  # Rigid receptor
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if os.path.exists(output_pdbqt) and os.path.getsize(output_pdbqt) > 0:
            logger.info(f"✓ Prepared receptor PDBQT with obabel: {output_pdbqt}")
            return output_pdbqt
        else:
            logger.error("obabel also produced empty file")
            return None
            
    except FileNotFoundError:
        logger.error("obabel not found. Please install OpenBabel")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"obabel failed: {e}")
        return None
    
    # return None

def prepare_ligand(ligand_file, output_pdbqt):
    """
    Prepare ligand: convert to PDBQT
    Supports SDF, MOL2, PDB input formats
    """
    ligand_path = Path(ligand_file)
    
    # mk_prepare_ligand adds .pdbqt extension automatically
    output_base = str(output_pdbqt).replace('.pdbqt', '')
    
    cmd = [
        'mk_prepare_ligand.py',
        '-i', str(ligand_path),
        '-o', output_base
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        expected_pdbqt = output_base + '.pdbqt'
        
        if os.path.exists(expected_pdbqt):
            if expected_pdbqt != str(output_pdbqt):
                os.rename(expected_pdbqt, str(output_pdbqt))
            
            if os.path.getsize(output_pdbqt) > 0:
                logger.info(f"Prepared off-target ligand PDBQT: {output_pdbqt}")
                return output_pdbqt
            else:
                logger.error("Generated ligand PDBQT is empty")
        else:
            logger.error(f"Ligand PDBQT not created")
            
        if result.stderr:
            logger.error(f"mk_prepare_ligand.py errors: {result.stderr}")
        
        return None
        
    except subprocess.TimeoutExpired:
        logger.error("mk_prepare_ligand.py timed out")
        return None
    except FileNotFoundError:
        logger.error("mk_prepare_ligand.py not found. Is meeko installed?")
        return None
    except Exception as e:
        logger.error(f"Unexpected error preparing ligand: {e}")
        return None


def generate_vina_config(config, dirs, com, box_size, seed):
    """Generate Vina configuration file for a specific seed"""
    
    # Add padding to box size
    padding = config['docking']['box_padding']
    box_size_padded = box_size + 2 * padding
    
    # Ensure minimum box size
    box_size_final = np.maximum(box_size_padded, [20, 20, 20])
    
    config_file = dirs['prepared'] / f'vina_config_seed{seed}.txt'
    
    with open(config_file, 'w') as f:
        f.write(f"receptor = {dirs['prepared']}/receptor.pdbqt\n")
        f.write(f"ligand = {dirs['prepared']}/offtarget_ligand.pdbqt\n")
        f.write(f"\n")
        f.write(f"center_x = {com[0]:.3f}\n")
        f.write(f"center_y = {com[1]:.3f}\n")
        f.write(f"center_z = {com[2]:.3f}\n")
        f.write(f"\n")
        f.write(f"size_x = {box_size_final[0]:.1f}\n")
        f.write(f"size_y = {box_size_final[1]:.1f}\n")
        f.write(f"size_z = {box_size_final[2]:.1f}\n")
        f.write(f"\n")
        f.write(f"out = {dirs['docking']}/vina_out_seed{seed}.pdbqt\n")
        f.write(f"\n")
        f.write(f"exhaustiveness = {config['docking']['exhaustiveness']}\n")
        f.write(f"num_modes = {config['docking']['n_poses']}\n")
        f.write(f"seed = {seed}\n")
        f.write(f"cpu = {config['docking']['cpu']}\n")
    
    logger.info(f"Generated Vina config for seed {seed}: {config_file}")
    return config_file


def generate_vina_slurm_script(config, dirs):
    """Generate SLURM script for Vina docking"""
    
    script_file = dirs['docking'] / f'run_vina.sh'
    
    vina_path = config['paths']['vina']
    log_file = dirs['logs'] / f'vina.log'
    
    with open(script_file, 'w') as f:
        # f.write(f"#!/bin/bash\n")
        # f.write(f"#SBATCH --job-name=vina_s{seed}\n")
        # f.write(f"#SBATCH --partition={config['slurm']['partition']}\n")
        # f.write(f"#SBATCH --mem={config['slurm']['mem']}\n")
        # f.write(f"#SBATCH --cpus-per-task={config['docking']['cpu']}\n")
        # f.write(f"#SBATCH --output={log_file}\n")
        # f.write(f"\n")
        # f.write("set -euo pipefail\n\n")
        f.write("echo \"🚀 Starting Vina docking jobs\"\n")
        for seed in range(config['docking']['n_seeds']):
            config_file = dirs['prepared'] / f'vina_config_seed{seed}.txt'
            f.write(f"echo \"Starting Vina docking seed {seed}\"\n")
            f.write(f"echo \"Config: {config_file}\"\n")
            f.write(f"\n")
            f.write(f"{vina_path} --config {config_file}\n")
            f.write(f"\n")
            f.write(f"echo \"Vina docking seed {seed} completed\"\n")
            f.write(f"\n")
        f.write("echo \"✅ All Vina docking jobs completed\"\n")
    
    # Make executable
    os.chmod(script_file, 0o755)
    
    logger.info(f"Generated SLURM script: {script_file}")
    return script_file


def main():
    parser = argparse.ArgumentParser(description='Prepare files for Vina docking')
    parser.add_argument('config', help='Configuration YAML file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup directories
    dirs = setup_directories(config)
    
    logger.info("="*60)
    logger.info("VINA PREPARATION")
    logger.info("="*60)
    
    # Extract target ligand COM and bounds
    logger.info("\n1. Analyzing target ligand...")
    com, box_size = get_ligand_com_and_bounds(
        config['target_pdb'],
        config['target_ligand_chain'],
        config['target_ligand_resid']
    )
    
    # Extract target ligand
    target_ligand_pdb = dirs['prepared'] / 'target_ligand.pdb'
    extract_ligand(
        config['target_pdb'],
        config['target_ligand_chain'],
        config['target_ligand_resid'],
        target_ligand_pdb
    )
    
    # Prepare receptor
    logger.info("\n2. Preparing receptor...")
    receptor_pdb = dirs['prepared'] / 'receptor.pdb'
    receptor_pdbqt = dirs['prepared'] / 'receptor.pdbqt'
    
    result = prepare_receptor(
        config['target_pdb'],
        config['target_ligand_chain'],
        config['target_ligand_resid'],
        receptor_pdb,
        receptor_pdbqt
    )
    
    if result is None:
        logger.error("\n❌ Receptor preparation failed!")
        logger.error("Please check:")
        logger.error("  1. Is meeko or openbabel installed?")
        logger.error("  2. Is the PDB file properly formatted?")
        logger.error("  3. Check the log file in the prepared/ directory")
        sys.exit(1)
    
    # Prepare off-target ligand
    logger.info("\n3. Preparing off-target ligand...")
    offtarget_pdbqt = dirs['prepared'] / 'offtarget_ligand.pdbqt'
    result = prepare_ligand(config['offtarget_ligand'], offtarget_pdbqt)
    
    if result is None:
        logger.error("\n❌ Ligand preparation failed!")
        sys.exit(1)
    
    # Generate Vina configs and SLURM scripts
    logger.info(f"\n4. Generating Vina configs and SLURM scripts for {config['docking']['n_seeds']} seeds...")
    for seed in range(config['docking']['n_seeds']):
        generate_vina_config(config, dirs, com, box_size, seed)
    script = generate_vina_slurm_script(config, dirs)
    
    # Summary
    print("\n" + "="*60)
    print("PREPARATION COMPLETE")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  - Receptor PDBQT: {receptor_pdbqt}")
    print(f"  - Off-target ligand PDBQT: {offtarget_pdbqt}")
    print(f"  - Target ligand PDB: {target_ligand_pdb}")
    print(f"  - {config['docking']['n_seeds']} Vina config files")
    print(f"  - SLURM scripts: {script}")
    
    print(f"\nNext steps:")
    print(f"  Submit Vina jobs to server:")
    print(f"    sbatch {script}")
    
    print(f"\n  After docking completes, run:")
    print(f"    python vina_postprocess.py {args.config}")


if __name__ == '__main__':
    main()
