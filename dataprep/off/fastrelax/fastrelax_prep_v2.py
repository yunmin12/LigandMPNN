"""
Prepare ligand params files for FastRelax from Vina docking results.
Converts PDBQT -> MOL2 -> Params and aligns coordinates.
"""

import os
import sys
import argparse
import subprocess
import logging
import csv
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd, cwd=None, description=""):
    """Run shell command and handle errors"""
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        if result.stdout:
            logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to {description}: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False


def convert_pdbqt_to_sdf(pdbqt_file, sdf_file, seed):
    """Convert PDBQT to SDF using obabel (first model only)"""
    cmd = [
        'obabel',
        str(pdbqt_file),
        '-f', '1',  # First model
        '-l', '1',  # Last model (only first)
        '-osdf',
        '-O', str(sdf_file)
    ]
    return run_command(cmd, description=f"convert PDBQT to SDF for seed{seed}")


def convert_sdf_to_mol2(sdf_file, mol2_file, seed):
    """Convert SDF to MOL2 using obabel"""
    cmd = [
        'obabel',
        str(sdf_file),
        '-omol2',
        '-O', str(mol2_file)
    ]

    if mol2_file.exists() and mol2_file.stat().st_size > 0:
        sdf_file.unlink()  # Remove sdf file
    
    return run_command(cmd, description=f"convert SDF to MOL2 for seed{seed}")


def convert_pdbqt_to_pdb(pdbqt_file, pdb_file, seed):
    """Convert PDBQT to PDB using obabel (first model only)"""
    cmd = [
        'obabel',
        str(pdbqt_file),
        '-f', '1',  # First model
        '-l', '1',  # Last model (only first)
        '-opdb',
        '-O', str(pdb_file)
    ]
    return run_command(cmd, description=f"convert PDBQT to PDB for seed{seed}")


def generate_params_from_mol2(mol2_file, het_id, output_dir):
    """Generate Rosetta params file from MOL2 with generic potential options"""
    cmd = [
        '/apps/repos/rosetta/source/scripts/python/public/generic_potential/mol2genparams.py',
        '-s', str(mol2_file),
        '--nm', het_id,
        '--prefix', het_id
    ]
    
    success = run_command(
        cmd,
        cwd=str(output_dir),
        description=f"generate params for {het_id}"
    )
    
    if not success:
        logger.warning(f"Failed to generate params for {het_id}")
        return None
    
    # Check if params file was generated
    params_file = output_dir / f"{het_id}.params"
    pdb_file = output_dir / f"{het_id}_0001.pdb"
    
    if not params_file.exists():
        logger.warning(f"Params file not generated: {params_file}")
        return None
    
    if not pdb_file.exists():
        logger.warning(f"PDB file not generated: {pdb_file}")
        return None
    
    logger.info(f"Generated params: {params_file}")
    logger.info(f"Generated PDB: {pdb_file}")
    
    return params_file, pdb_file


def align_params_to_vina(params_pdb, vina_pdbs, frx_script):
    """Run frx_superimpose_params_pdb.py to align params PDB to Vina PDBs"""
    cmd = [
        'python',
        str(frx_script),
        str(params_pdb)
    ] + [str(pdb) for pdb in vina_pdbs]
    
    return run_command(cmd, description="align params PDB to Vina PDBs")

def _pdb_lines_single_model(p: Path):
    """Read PDB file and return lines for single model (skip MODEL/ENDMDL/END)"""
    out = []
    with p.open("r") as f:
        for line in f:
            rec = line[:6].strip()
            if rec in ("MODEL", "ENDMDL", "END"):
                continue
            out.append(line.rstrip("\n"))
    return out

def concat_receptor_ligand(receptor_pdb: Path, ligand_pdb: Path, output_pdb: Path) -> bool:
    """Concatenate receptor and ligand PDB files into a single PDB file."""
    try:
        r = _pdb_lines_single_model(receptor_pdb)
        l = _pdb_lines_single_model(ligand_pdb)

        if r and r[-1][:3] != "TER":
            r.append("TER")

        with output_pdb.open("w") as w:
            for line in r:
                w.write(line + "\n")
            for line in l:
                w.write(line + "\n")
            w.write("END\n")
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser(
        description='Prepare ligand params files from Vina docking PDBQT files for FastRelax.'
    )
    p.add_argument(
        '--base_dir',
        type=str,
        help='Base directory (runs/complex_*_off*) containing subdirectories'
    )
    p.add_argument(
        '--het_id',
        type=str,
        help='3-letter HET ID for the ligand (e.g., ATP, HEM, LIG)'
    )
    p.add_argument(
        '--seeds',
        type=int,
        default=5,
        help='Number of seeds to process (default: 5)'
    )
    p.add_argument(
        '--csv_path',
        type=str,
        help='Path to summary CSV file (default: fastrelax_prep_summary.csv)',
    )
    
    args = p.parse_args()
    
    # Parse arguments
    base_dir = Path(args.base_dir).resolve()
    het_id = args.het_id.upper()
    seeds = list(range(args.seeds))
    
    # Validate HET ID (should be 3 characters)
    if len(het_id) != 3:
        logger.warning(f"HET ID '{het_id}' is not 3 characters. Rosetta may have issues.")
    
    # Setup directories
    prepared_dir = base_dir / 'prepared'
    docking_dir = base_dir / 'docking'
    filtered_dir = base_dir / 'filtered'
    relaxed_dir = base_dir / 'relaxed'
    
    if not docking_dir.exists():
        logger.error(f"Docking directory not found: {docking_dir}")
        return 1
    
    filtered_dir.mkdir(parents=True, exist_ok=True)
    relaxed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("="*5 + " LIGAND PARAMS PREPARATION " + "="*5)
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"HET ID: {het_id}")
    logger.info(f"Seeds: {seeds}")
    logger.info("")
    
    # Initialize summary tracking
    summary = {
        'complex_id': base_dir.name,
        'het_id': het_id,
        'params_pdb': None,
        'stage_completed': 'NONE',
        'num_pdbs_converted': 0,
        'num_pdbs_aligned': 0,
        'num_complexes_concatenated': 0,
        'status': 'FAILED',
        'timestamp': datetime.now().isoformat(),
        'notes': ''
    }
    
    # Step 1 & 2: Convert PDBQT to MOL2 and generate params for first seed
    logger.info("Step 1-2: Converting PDBQT to MOL2 and generating params...")
    
    seed0_pdbqt = docking_dir / 'vina_out_seed0.pdbqt'
    seed0_sdf = filtered_dir / 'seed0_ligand.sdf'
    seed0_mol2 = filtered_dir / 'seed0_ligand.mol2'
    
    if not seed0_pdbqt.exists():
        logger.error(f"Seed 0 PDBQT not found: {seed0_pdbqt}")
        logger.warning("⚠️ Skipping this complex due to missing seed0 PDBQT")
        summary['notes'] = 'Missing seed0 PDBQT file'
        summary['stage_completed'] = 'FAILED'
        return summary, args.csv_path
    
    # Convert seed0 to MOL2
    if not convert_pdbqt_to_sdf(seed0_pdbqt, seed0_sdf, 0):
        logger.warning("⚠️ Failed to convert seed0 PDBQT to SDF, skipping this complex")
        summary['notes'] = 'Failed to convert PDBQT to SDF'
        summary['stage_completed'] = 'FAILED'
        return summary, args.csv_path
    
    if not convert_sdf_to_mol2(seed0_sdf, seed0_mol2, 0):
        logger.warning("⚠️ Failed to convert seed0 SDF to MOL2, skipping this complex")
        summary['notes'] = 'Failed to convert SDF to MOL2'
        summary['stage_completed'] = 'FAILED'
        seed0_sdf.unlink()  # Remove sdf file
        return summary, args.csv_path
    
    # Generate params from seed0 MOL2
    params_result = generate_params_from_mol2(seed0_mol2, het_id, relaxed_dir)
    if not params_result:
        logger.warning("⚠️ Failed to generate params from MOL2, skipping this complex")
        summary['notes'] = 'Failed to generate params from MOL2'
        summary['stage_completed'] = 'FAILED'
        return summary, args.csv_path
    
    params_file, params_pdb = params_result
    summary['params_pdb'] = str(params_pdb.relative_to(base_dir.parent))
    summary['stage_completed'] = 'PARAMS_GENERATED'
    
    logger.info(f"✓ Generated params: {params_file}")
    logger.info(f"✓ Generated params PDB: {params_pdb}")
    logger.info("")
    
    # Step 3: Convert all PDBQT seeds to PDB
    logger.info("Step 3: Converting PDBQT files to PDB for all seeds...")
    
    vina_pdbs = []
    for seed in seeds:
        pdbqt_file = docking_dir / f'vina_out_seed{seed}.pdbqt'
        pdb_file = filtered_dir / f'seed{seed}_ligand.pdb'
        
        if not pdbqt_file.exists():
            logger.warning(f"⚠️ PDBQT not found for seed {seed}, skipping this seed")
            continue
        
        if convert_pdbqt_to_pdb(pdbqt_file, pdb_file, seed):
            vina_pdbs.append(pdb_file)
            summary['num_pdbs_converted'] += 1
            logger.info(f"✓ Converted seed{seed}: {pdb_file}")
        else:
            logger.warning(f"⚠️ Failed to convert seed {seed}, skipping this seed")
    
    if not vina_pdbs:
        logger.warning("⚠️ No Vina PDB files were successfully converted, skipping this complex")
        summary['notes'] = 'No Vina PDB files converted'
        return summary, args.csv_path
    
    logger.info(f"\n✓ Successfully converted {len(vina_pdbs)} Vina PDB files")
    logger.info("")
    
    # Step 4: Align params PDB to Vina PDBs
    logger.info("Step 4: Aligning params PDB coordinates to Vina PDBs...")
    
    frx_script = Path('/home/yunmin/proj/LigandMPNN/dataprep/off/frx_superimpose_params_pdb.py')
    if not frx_script.exists():
        logger.error(f"Superimpose script not found: {frx_script}")
        logger.warning("⚠️ Superimpose script not found, skipping alignment")
        summary['notes'] = 'Superimpose script not found'
    else:
        if align_params_to_vina(params_pdb, vina_pdbs, frx_script):
            logger.info("✓ Successfully aligned params PDB to all Vina PDBs")
            summary['stage_completed'] = 'PARAMS_ALIGNED'
            # Count aligned PDBs
            for seed in seeds:
                if (filtered_dir / f'seed{seed}_ligand_aligned.pdb').exists():
                    summary['num_pdbs_aligned'] += 1
        else:
            logger.warning("⚠️ Failed to align params PDB to Vina PDBs, skipping alignment step")
            summary['notes'] = 'Failed to align params PDB'
    
    logger.info("")
    
    # Step 5: Concatenate receptor and aligned ligand PDBs
    logger.info("Step 5: Concatenating receptor and aligned ligand PDBs...")
    receptor_pdb = prepared_dir / 'receptor.pdb'

    pdb_list_txt = relaxed_dir / 'complex_pdb_list.txt'
    complex_paths = []
    
    if not receptor_pdb.exists():
        logger.warning(f"⚠️ Receptor PDB not found: {receptor_pdb}, skipping concatenation step")
        summary['notes'] = 'Receptor PDB not found'
    else:
        concatenated_count = 0
        for seed in seeds:
            aligned_ligand_pdb = filtered_dir / f'seed{seed}_ligand_aligned.pdb'
            output_pdb = filtered_dir / f'seed{seed}_complex.pdb'
            
            if not aligned_ligand_pdb.exists():
                logger.warning(f"⚠️ Aligned ligand PDB not found for seed {seed}, skipping this seed")
                continue
            
            if concat_receptor_ligand(receptor_pdb, aligned_ligand_pdb, output_pdb):
                logger.info(f"✓ Created concatenated PDB for seed{seed}: {output_pdb}")
                concatenated_count += 1
                complex_paths.append(str(output_pdb.resolve())) # append absolute path
            else:
                logger.warning(f"⚠️ Failed to concatenate for seed {seed}, skipping this seed")
        
        summary['num_complexes_concatenated'] = concatenated_count
        if concatenated_count > 0:
            logger.info(f"✓ Successfully concatenated {concatenated_count} complexes")
            logger.info(f"✓ Wrote pdb list to {pdb_list_txt}")
            summary['stage_completed'] = 'COMPLETE'
            # Write complex paths to pdb_list.txt
            with open(pdb_list_txt, 'w') as f:
                for path in complex_paths:
                    f.write(f"{path}\n")
    
    logger.info("")

    # Determine final status
    if summary['stage_completed'] == 'COMPLETE':
        summary['status'] = 'SUCCESS'
    elif summary['stage_completed'] == 'PARAMS_ALIGNED':
        summary['status'] = 'PARTIAL_SUCCESS'
    elif summary['stage_completed'] == 'PARAMS_GENERATED':
        summary['status'] = 'PARTIAL_SUCCESS'
    elif summary['stage_completed'] == 'FAILED':
        summary['status'] = 'FAILED'

    # Summary
    logger.info("\n" + "="*60)
    logger.info("PREPARATION COMPLETE")
    logger.info("="*60)
    logger.info(f"\nGenerated files in {filtered_dir}:")
    logger.info(f"  - {params_file.name}")
    logger.info(f"  - {params_pdb.name}")
    
    generated_count = 0
    for seed in seeds:
        if (filtered_dir / f'seed{seed}_ligand.pdb').exists():
            logger.info(f"  - seed{seed}_ligand.pdb")
            generated_count += 1
            if (filtered_dir / f'seed{seed}_ligand_aligned.pdb').exists():
                logger.info(f"  - seed{seed}_ligand_aligned.pdb")
    
    if generated_count > 0:
        logger.info("\nNext step:")
        logger.info(f"  Use {params_file} and seed*_complex.pdb for FastRelax")
    
    logger.info(f"\nStatus: {summary['status']}")
    logger.info(f"Stage Completed: {summary['stage_completed']}")
    logger.info("="*60)
    
    return summary, args.csv_path

def write_summary_csv(summary, csv_file):
    """Write summary information to CSV file"""
    # Check if file exists to determine if we need to write header
    file_exists = csv_file.exists()
    
    fieldnames = [
        'complex_id',
        'het_id',
        'params_pdb',
        'stage_completed',
        'num_pdbs_converted',
        'num_pdbs_aligned',
        'num_complexes_concatenated',
        'status',
        'timestamp',
        'notes'
    ]
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary)


if __name__ == '__main__':
    summary, csv_path = main()

    # Write summary to CSV
    csv_path = Path(csv_path) if csv_path else None
    
    if isinstance(summary, dict) and csv_path is not None:
        write_summary_csv(summary, csv_path)
        logger.info(f"Summary csv is saved at {csv_path}")
 
    sys.exit(0)