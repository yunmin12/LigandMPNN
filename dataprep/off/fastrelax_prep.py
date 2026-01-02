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
import re

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


def reassign_ligand_chain(complex_pdb, output_pdb, from_chain='Z', to_chain='X', to_resid=1):
    """
    Reassign ligand chain from Z to X with resid 1 for Rosetta
    """
    with open(complex_pdb, 'r') as f_in:
        lines = f_in.readlines()
    
    with open(output_pdb, 'w') as f_out:
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):
                chain = line[21]
                if chain == from_chain:
                    # Reassign chain and resid
                    new_line = (
                        line[:21] +  # Up to chain
                        to_chain +  # New chain
                        f"{to_resid:>4}" +  # New resid
                        line[26:]  # Rest of line
                    )
                    f_out.write(new_line)
                else:
                    f_out.write(line)
            else:
                f_out.write(line)
    
    logger.info(f"Reassigned ligand chain {from_chain} -> {to_chain} with resid {to_resid}")


def generate_ligand_params(ligand_pdb, output_params, ligand_het_id):
    """
    Generate Rosetta ligand params file using mol2genparams.py
    ligand_het_id: 3-letter ligand HET ID
    """
    ligand_path = Path(ligand_pdb)
    
    # Step 1: Convert PDB to MOL2 using obabel
    mol2_file = ligand_path.parent / f"{ligand_path.stem}.mol2"
    
    cmd = [
        'obabel',
        '-ipdb', str(ligand_pdb),
        '-h',
        '-omol2',
        '-O', str(mol2_file)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Converted ligand PDB to MOL2: {mol2_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert ligand to MOL2: {e.stderr}")
        raise
    
    # Step 2: Generate params using mol2genparams.py
    cmd = [
        '/apps/repos/rosetta/source/scripts/python/public/generic_potential/mol2genparams.py',
        '-s', str(mol2_file),
        '--nm', ligand_het_id,
        '--prefix', ligand_het_id
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=str(output_params.parent)
        )
        logger.info(f"Generated ligand params using mol2genparams.py")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate params: {e.stderr}")
        raise
    
    # The script generates <het_id>.params, rename it to the desired output
    generated_params = output_params.parent / f"{ligand_het_id}.params"
    
    if generated_params.exists() and generated_params != output_params:
        shutil.move(str(generated_params), str(output_params))
    
    if not output_params.exists():
        raise FileNotFoundError(f"Params file not generated: {output_params}")
    
    logger.info(f"Params file saved: {output_params}")
    return output_params


def create_ligand_flags_file(params_file, output_flags):
    """
    Create ligand_flags file for Rosetta
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
    Generate SLURM script for FastRelax
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
        f.write(f"#SBATCH --mem={config['slurm']['mem']}\n")
        f.write(f"#SBATCH --cpus-per-task=4\n")
        f.write(f"#SBATCH --output={log_file}\n")
        f.write(f"\n")
        f.write(f"echo \"Starting FastRelax for {pose_name}\"\n")
        f.write(f"echo \"Complex: {complex_pdb}\"\n")
        f.write(f"echo \"Ligand flags: {ligand_flags}\"\n")
        f.write(f"\n")
        
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
    
    os.chmod(script_file, 0o755)
    
    logger.info(f"Generated SLURM script: {script_file}")
    return script_file


def prepare_seed_for_relaxation(seed, complex_id, config_num, config, dirs, ligand_het_id):
    """
    Prepare a single seed for FastRelax from existing filtered/seed{seed}_complex.pdb
    """
    pose_name = f"complex_{complex_id:04d}_config{config_num:02d}_seed{seed}"
    
    logger.info(f"\nPreparing pose: {pose_name}")
    
    # Input files (already exist from vina_post.py)
    seed_complex_pdb = dirs['filtered'] / f'seed{seed}_complex.pdb'
    seed_ligand_pdb = dirs['filtered'] / f'seed{seed}_ligand.pdb'
    
    if not seed_complex_pdb.exists():
        logger.error(f"Complex PDB not found: {seed_complex_pdb}")
        return None
    
    if not seed_ligand_pdb.exists():
        logger.error(f"Ligand PDB not found: {seed_ligand_pdb}")
        return None
    
    # Reassign ligand chain from Z to X with resid 1 for Rosetta
    complex_pdb = dirs['relaxed'] / f"{pose_name}_input.pdb"
    reassign_ligand_chain(seed_complex_pdb, complex_pdb, from_chain='Z', to_chain='X', to_resid=1)
    
    # Generate ligand params using the extracted HET ID
    params_file = dirs['relaxed'] / f"{pose_name}_ligand.params"
    generate_ligand_params(seed_ligand_pdb, params_file, ligand_het_id)
    
    # Create ligand flags file
    flags_file = dirs['relaxed'] / f"{pose_name}_ligand_flags"
    create_ligand_flags_file(params_file, flags_file)
    
    # Identify pocket residues
    pocket_residues = identify_pocket_residues(
        complex_pdb,
        ligand_chain='X',
        ligand_resid=1,
        radius=config['relaxation']['pocket_radius']
    )
    
    # Generate SLURM script
    # slurm_script = generate_fastrelax_slurm_script(
    #     config,
    #     dirs,
    #     pose_name,
    #     complex_pdb,
    #     flags_file
    # )
    
    return {
        'pose_name': pose_name,
        'seed': seed,
        'complex_pdb': str(complex_pdb),
        'params_file': str(params_file),
        'flags_file': str(flags_file),
        # 'slurm_script': str(slurm_script),
        'n_pocket_residues': len(pocket_residues),
        'pocket_residues': ','.join(map(str, pocket_residues))
    }


def main():
    parser = argparse.ArgumentParser(description='Prepare FastRelax inputs from Vina docking results')
    parser.add_argument('config', help='Configuration YAML file (complex_{complex_id:04d}_off{config_num:02d}.yaml)')
    parser.add_argument('--seeds', help='Specific seeds to prepare (comma-separated, e.g., 0,1,2)', default=None)
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract complex_id and config_num from filename
    config_path = Path(args.config)
    match = re.search(r'complex_(\d+)_off(\d+)', config_path.name)
    if not match:
        logger.error(f"Could not extract complex_id and config_num from filename: {config_path.name}")
        logger.error("Expected format: complex_XXXX_offYY.yaml")
        return
    
    complex_id = int(match.group(1))
    config_num = int(match.group(2))
    
    logger.info(f"Complex ID: {complex_id}")
    logger.info(f"Config number: {config_num}")
    
    # Extract ligand HET ID from target_pdb basename
    # Format: {PDB_ID}_{HET_ID}_std_bb_keephet_ala.pdb
    target_pdb = config.get('target_pdb', '')
    if not target_pdb:
        logger.error("target_pdb not found in config!")
        return
    
    target_pdb_basename = os.path.basename(target_pdb)
    match_het = re.match(r'^[A-Za-z0-9]{4}_([A-Za-z0-9]{3})_', target_pdb_basename)
    if match_het:
        ligand_het_id = match_het.group(1)
        logger.info(f"Ligand HET ID: {ligand_het_id}")
    else:
        logger.error(f"Could not extract HET ID from target_pdb: {target_pdb_basename}")
        logger.error("Expected format: PDBID_HET_std_bb_keephet_ala.pdb")
        return
    
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
    
    # Create relaxed directory if not exists
    dirs['relaxed'].mkdir(parents=True, exist_ok=True)
    dirs['logs'].mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("FASTRELAX PREPARATION FROM VINA DOCKING RESULTS")
    logger.info("="*60)
    
    # Determine which seeds to process
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
        logger.info(f"\nProcessing specified seeds: {seeds}")
    else:
        # Auto-detect seeds from filtered directory
        seeds = []
        for seed in range(5):  # Seeds 0-4
            seed_complex = dirs['filtered'] / f'seed{seed}_complex.pdb'
            if seed_complex.exists():
                seeds.append(seed)
        logger.info(f"\nAuto-detected {len(seeds)} seeds: {seeds}")
    
    if not seeds:
        logger.error("No seeds found in filtered directory!")
        logger.error(f"Expected to find seed[0-4]_complex.pdb in: {dirs['filtered']}")
        return
    
    # Prepare each seed
    logger.info(f"\nPreparing {len(seeds)} seed(s) for FastRelax...")
    
    prepared_data = []
    slurm_scripts = []
    input_pdb_paths = []
    
    for seed in seeds:
        try:
            result = prepare_seed_for_relaxation(seed, complex_id, config_num, config, dirs, ligand_het_id)
            if result:
                prepared_data.append(result)
                # slurm_scripts.append(result['slurm_script'])
                # Collect absolute path of input PDB
                input_pdb_paths.append(os.path.abspath(result['complex_pdb']))
        except Exception as e:
            logger.error(f"Failed to prepare seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not prepared_data:
        logger.error("No seeds were successfully prepared!")
        return
    
    # Save preparation summary
    df_prepared = pd.DataFrame(prepared_data)
    prep_csv = dirs['relaxed'] / 'fastrelax_prepared.csv'
    df_prepared.to_csv(prep_csv, index=False)
    logger.info(f"\nSaved preparation summary: {prep_csv}")
    
    # Write pdb_list.txt with absolute paths
    pdb_list_file = dirs['relaxed'] / 'pdb_list.txt'
    with open(pdb_list_file, 'w') as f:
        for pdb_path in input_pdb_paths:
            f.write(f"{pdb_path}\n")
    logger.info(f"Saved PDB list: {pdb_list_file}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PREPARATION COMPLETE")
    logger.info("="*60)
    logger.info(f"\nPrepared {len(prepared_data)} seed(s) for FastRelax:")
    for data in prepared_data:
        logger.info(f"  - {data['pose_name']}: {data['n_pocket_residues']} pocket residues")
    
    logger.info(f"\nGenerated files in: {dirs['relaxed']}")
    logger.info(f"  - {len(prepared_data)} input complexes (*_input.pdb)")
    logger.info(f"  - {len(prepared_data)} params files (*_ligand.params)")
    logger.info(f"  - {len(prepared_data)} flags files (*_ligand_flags)")
    # logger.info(f"  - {len(prepared_data)} SLURM scripts (fastrelax_*.sh)")
    logger.info(f"  - pdb_list.txt with {len(input_pdb_paths)} entries")
    
    logger.info(f"\nNext steps:")
    logger.info(f"  Submit FastRelax jobs")
    # for script in slurm_scripts:
    #     logger.info(f"    sbatch {script}")


if __name__ == '__main__':
    main()