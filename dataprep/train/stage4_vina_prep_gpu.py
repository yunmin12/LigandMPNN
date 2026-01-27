"""
Stage 4: Vina Preparation
Prepare receptor.pdbqt and off-target ligand.pdbqt for docking
"""
import os
import sys
import logging
import argparse
import yaml
import subprocess
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from utils.status_tracker import StatusTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProteinSelect(Select):
    """Select protein residues only (exclude ligand and waters)"""
    def accept_residue(self, residue):
        # Exclude waters
        if residue.resname in ['HOH', 'WAT', 'H2O', 'DOD']:
            return False
        # Exclude HETATM (ligands)
        if residue.id[0] != ' ':
            return False
        return True


class VinaPreparator:
    def __init__(self, status_tracker: StatusTracker):
        self.status_tracker = status_tracker
        self.parser = PDBParser(QUIET=True)
        self.io = PDBIO()
        
        # Check dependencies at init
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required tools are available"""
        try:
            import gemmi
            logger.info(f"✓ gemmi {gemmi.__version__} found")
        except ImportError:
            logger.error("✗ gemmi not found")
            logger.error("  Install: pip install gemmi")
            raise RuntimeError("gemmi required by meeko")
        
        try:
            result = subprocess.run(['mk_prepare_receptor.py', '--help'],
                                   capture_output=True, timeout=5)
            logger.info("✓ mk_prepare_receptor.py found")
        except Exception as e:
            logger.error(f"✗ mk_prepare_receptor.py not found: {e}")
            raise
        
        try:
            result = subprocess.run(['mk_prepare_ligand.py', '--help'],
                                   capture_output=True, timeout=5)
            logger.info("✓ mk_prepare_ligand.py found")
        except Exception as e:
            logger.error(f"✗ mk_prepare_ligand.py not found: {e}")
            raise
    
    def get_ligand_com_and_bounds(self, pdb_file: Path, chain_id: str = 'Z', resid: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get center of mass and bounding box of target ligand
        Returns: com (x,y,z), box_size (x,y,z)
        """
        structure = self.parser.get_structure('complex', pdb_file)
        
        coords = []
        for residue in structure[0][chain_id]:
            if residue.id[1] == resid:
                for atom in residue:
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
    
    def extract_receptor(self, pdb_file: Path, output_pdb: Path) -> bool:
        """
        Extract receptor from complex (remove ligand and waters)
        """
        try:
            structure = self.parser.get_structure('complex', pdb_file)
            
            # Save receptor (protein only)
            self.io.set_structure(structure)
            temp_pdb = str(output_pdb).replace('.pdb', '_temp.pdb')
            self.io.save(temp_pdb, ProteinSelect())
            
            # Clean PDB file (keep only ATOM records)
            atom_count = 0
            with open(temp_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
                for line in f_in:
                    if line.startswith('ATOM'):
                        f_out.write(line)
                        atom_count += 1
            
            # Remove temp file
            Path(temp_pdb).unlink()
            
            if atom_count == 0:
                raise ValueError("Generated receptor PDB is empty")
            
            logger.info(f"Extracted receptor: {atom_count} atoms")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract receptor: {e}")
            return False
    
    def prepare_receptor_pdbqt(self, receptor_pdb: Path, output_pdbqt: Path) -> bool:
        """
        Convert receptor PDB to PDBQT using mk_prepare_receptor.py (meeko)
        """
        try:
            output_base = str(output_pdbqt).replace('.pdbqt', '')
            log_file = str(receptor_pdb).replace('.pdb', '_prep.log')
            
            cmd = [
                'mk_prepare_receptor.py',
                '-i', str(receptor_pdb),
                '-o', output_base,
                '--allow_bad_res',
                '--default_altloc', 'A',
                '-p', '-j'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            # Write log
            with open(log_file, 'w') as f:
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
            
            # Check if PDBQT was created
            expected_pdbqt = output_base + '.pdbqt'
            if not Path(expected_pdbqt).exists():
                raise ValueError("PDBQT file not created")
            
            # Check file size
            if Path(expected_pdbqt).stat().st_size == 0:
                raise ValueError("PDBQT file is empty")
            
            # Validate PDBQT (check for ROOT/BRANCH - should NOT exist in receptor)
            with open(expected_pdbqt) as f:
                content = f.read()
                if any(tag in content for tag in ['ROOT', 'BRANCH', 'TORSDOF']):
                    raise ValueError("Receptor PDBQT contains ligand tags (ROOT/BRANCH)")
            
            logger.info(f"Created receptor PDBQT: {expected_pdbqt}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("mk_prepare_receptor.py timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to prepare receptor PDBQT: {e}")
            return False
    
    def prepare_ligand_pdbqt(self, ligand_file: Path, output_pdbqt: Path) -> bool:
        """
        Convert ligand to PDBQT using mk_prepare_ligand.py (meeko)
        Supports SDF, MOL2, PDB input formats
        
        If the input file is already in PDBQT format, just copy it.
        For 2D SDF files, generate 3D coordinates first using Open Babel.
        """
        try:
            import shutil
            
            # Check if the input file is already PDBQT format
            # (either by extension or by content)
            is_pdbqt = False
            is_2d_sdf = False
            
            if ligand_file.suffix.lower() == '.pdbqt':
                is_pdbqt = True
            else:
                # Check content for PDBQT markers (ROOT, BRANCH, TORSDOF)
                with open(ligand_file, 'r') as f:
                    content = f.read()
                    if any(marker in content for marker in ['ROOT', 'BRANCH', 'TORSDOF']):
                        is_pdbqt = True
                        logger.info(f"  Detected PDBQT format in {ligand_file.name}")
                    # Check if it's a 2D SDF (look for V2000/V3000 and check z-coords)
                    elif 'V2000' in content or 'V3000' in content:
                        # Check if Z coordinates are all 0 (2D structure)
                        lines = content.split('\n')
                        atom_section = False
                        all_z_zero = True
                        for line in lines:
                            if 'V2000' in line or 'V3000' in line:
                                atom_section = True
                                continue
                            if atom_section and len(line) > 30:
                                parts = line.split()
                                if len(parts) >= 4:
                                    try:
                                        z = float(parts[2])
                                        if abs(z) > 0.001:
                                            all_z_zero = False
                                            break
                                    except (ValueError, IndexError):
                                        pass
                            if 'M  END' in line:
                                break
                        if all_z_zero:
                            is_2d_sdf = True
                            logger.info(f"  Detected 2D SDF: {ligand_file.name}, will generate 3D coordinates")
            
            if is_pdbqt:
                # Just copy the file
                shutil.copy2(ligand_file, output_pdbqt)
                logger.info(f"Copied existing PDBQT: {output_pdbqt}")
                return True
            
            # For 2D SDF, generate 3D coordinates first using RDKit, then use mk_prepare_ligand.py
            if is_2d_sdf:
                try:
                    from rdkit import Chem
                    from rdkit.Chem import AllChem
                    
                    temp_3d_sdf = str(ligand_file).replace('.sdf', '_3d.sdf')
                    
                    # Read molecule (sanitize=False to handle problematic molecules)
                    suppl = Chem.SDMolSupplier(str(ligand_file), removeHs=False, sanitize=False)
                    mol = None
                    for m in suppl:
                        if m is not None:
                            mol = m
                            break
                    
                    if mol is None:
                        raise ValueError("Could not read molecule from SDF")
                    
                    # Sanitize
                    try:
                        Chem.SanitizeMol(mol)
                    except:
                        pass  # Continue even if sanitization fails
                    
                    # Add hydrogens
                    mol = Chem.AddHs(mol, addCoords=True)
                    
                    # Generate 3D coordinates with multiple attempts
                    embed_result = -1
                    for attempt in range(3):
                        try:
                            embed_result = AllChem.EmbedMolecule(mol, randomSeed=42+attempt, 
                                                                maxAttempts=10, useRandomCoords=True)
                            if embed_result == 0:
                                break
                        except:
                            continue
                    
                    if embed_result != 0:
                        raise ValueError(f"Could not generate 3D coordinates after 3 attempts")
                    
                    # Optimize geometry
                    try:
                        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                    except:
                        pass  # Continue even if optimization fails
                    
                    # Write 3D SDF
                    writer = Chem.SDWriter(temp_3d_sdf)
                    writer.write(mol)
                    writer.close()
                    
                    if not Path(temp_3d_sdf).exists() or Path(temp_3d_sdf).stat().st_size == 0:
                        raise ValueError("3D SDF file not created")
                    
                    logger.info(f"  Generated 3D SDF using RDKit: {Path(temp_3d_sdf).name}")
                    
                except Exception as e:
                    logger.error(f"RDKit 3D generation failed: {e}")
                    # Cleanup
                    try:
                        if Path(temp_3d_sdf).exists():
                            Path(temp_3d_sdf).unlink()
                    except:
                        pass
                    raise ValueError(f"Failed to generate 3D coordinates: {e}")
                
                # Step 2: Convert 3D SDF to PDBQT using mk_prepare_ligand.py
                output_base = str(output_pdbqt).replace('.pdbqt', '')
                
                cmd_pdbqt = [
                    'mk_prepare_ligand.py',
                    '-i', temp_3d_sdf,
                    '-o', output_base
                ]
                
                result = subprocess.run(cmd_pdbqt, capture_output=True, text=True, timeout=120)
                
                # mk_prepare_ligand.py writes to output_base (no extension)
                # Rename to add .pdbqt extension if needed
                output_no_ext = Path(output_base)
                if output_no_ext.exists() and output_no_ext.stat().st_size > 0:
                    if not output_pdbqt.exists():
                        output_no_ext.rename(output_pdbqt)
                
                # Clean up temp file
                try:
                    Path(temp_3d_sdf).unlink()
                except:
                    pass
                
                if output_pdbqt.exists() and output_pdbqt.stat().st_size > 0:
                    logger.info(f"Created ligand PDBQT (2D→3D→PDBQT): {output_pdbqt}")
                    return True
                else:
                    logger.error(f"mk_prepare_ligand.py failed: {result.stderr}")
                    raise ValueError("PDBQT file not created from 3D SDF")
            
            # For PDB files, use obabel directly to PDBQT
            is_pdb_format = ligand_file.suffix.lower() == '.pdb'
            
            if is_pdb_format:
                cmd = [
                    'obabel', str(ligand_file),
                    '-O', str(output_pdbqt),
                    '-h',       # Add hydrogens
                    '-p', '7.4'  # Protonate at physiological pH
                ]
                
                logger.info(f"  Using obabel for PDB format: {ligand_file.name}")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if output_pdbqt.exists() and output_pdbqt.stat().st_size > 0:
                    logger.info(f"Created ligand PDBQT (via obabel): {output_pdbqt}")
                    return True
                else:
                    logger.warning(f"obabel conversion failed: {result.stderr}")
            
            # Convert using mk_prepare_ligand.py (supports SDF/MOL2/MOL only)
            if ligand_file.suffix.lower() in ['.sdf', '.mol2', '.mol']:
                output_base = str(output_pdbqt).replace('.pdbqt', '')
                
                cmd = [
                    'mk_prepare_ligand.py',
                    '-i', str(ligand_file),
                    '-o', output_base
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                # Check if PDBQT was created
                expected_pdbqt = output_pdbqt
                if Path(expected_pdbqt).exists() and Path(expected_pdbqt).stat().st_size > 0:
                    logger.info(f"Created ligand PDBQT: {expected_pdbqt}")
                    return True
                else:
                    logger.error(f"mk_prepare_ligand.py failed: {result.stderr}")
                    raise ValueError("PDBQT file not created")
            
            # If we got here, no conversion method worked
            raise ValueError(f"Unsupported ligand format: {ligand_file.suffix}")
            
        except subprocess.TimeoutExpired:
            logger.error("Ligand preparation timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to prepare ligand PDBQT: {e}")
            return False
    
    def generate_vina_configs(self, com: np.ndarray, box_size: np.ndarray, 
                             prepared_dir: Path, n_seeds: int = 3, 
                             box_padding: float = 5.0) -> bool:
        """
        Generate Vina configuration files for multiple seeds
        """
        try:
            # Add padding to box size
            box_size_padded = box_size + 2 * box_padding
            
            # Ensure minimum box size
            box_size_final = np.maximum(box_size_padded, [20, 20, 20])
            
            for seed in range(n_seeds):
                config_file = prepared_dir / f'vina_config_seed{seed}.txt'
                
                with open(config_file, 'w') as f:
                    f.write(f"receptor = {prepared_dir / 'receptor.pdbqt'}\n")
                    f.write(f"ligand = {prepared_dir / 'off_ligand.pdbqt'}\n")
                    f.write(f"\n")
                    f.write(f"center_x = {com[0]:.3f}\n")
                    f.write(f"center_y = {com[1]:.3f}\n")
                    f.write(f"center_z = {com[2]:.3f}\n")
                    f.write(f"\n")
                    f.write(f"size_x = {box_size_final[0]:.1f}\n")
                    f.write(f"size_y = {box_size_final[1]:.1f}\n")
                    f.write(f"size_z = {box_size_final[2]:.1f}\n")
                    f.write(f"\n")
                    f.write(f"exhaustiveness = 32\n")
                    f.write(f"num_modes = 5\n")
                    f.write(f"seed = {seed}\n")
                    f.write(f"\n")
                    f.write(f"out = {prepared_dir.parent / 'docking' / f'out_seed{seed}.pdbqt'}\n")
                
                logger.info(f"Generated Vina config: seed{seed}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate Vina configs: {e}")
            return False
    
    def process_config(self, row: pd.Series, config_id: str, runs_dir: Path, 
                      current_num: int = None, total_num: int = None) -> bool:
        """
        Process one config: prepare receptor and ligand PDBQT files
        
        Args:
            row: DataFrame row
            config_id: Config ID
            runs_dir: Runs directory
            current_num: Current config number (for progress tracking)
            total_num: Total configs (for progress tracking)
        """
        # Check if can run this stage
        if not self.status_tracker.can_run_stage(config_id, '4_vina_prep'):
            self.status_tracker.mark_status(config_id, '4_vina_prep', 'skip',
                                           'Previous stage not completed')
            return False
        
        try:
            # Get paths from CSV
            target_blurred_pdb = Path(row['target_blurred_path'])
            offtarget_ligand = Path(row['offtarget_ligand_path'])
            
            if not target_blurred_pdb.exists():
                raise ValueError(f"Target blurred PDB not found: {target_blurred_pdb}")
            if not offtarget_ligand.exists():
                raise ValueError(f"Off-target ligand not found: {offtarget_ligand}")
            
            # Create run directory
            run_dir = runs_dir / config_id
            prepared_dir = run_dir / 'prepared'
            docking_dir = run_dir / 'docking'
            prepared_dir.mkdir(parents=True, exist_ok=True)
            docking_dir.mkdir(parents=True, exist_ok=True)
            
            # Format progress indicator
            progress_str = ""
            if current_num is not None and total_num is not None:
                progress_str = f" ({current_num}/{total_num})"
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing {config_id}{progress_str}")
            logger.info(f"{'='*70}")
            
            # Step 1: Get target ligand COM and bounds
            logger.info("1. Analyzing target ligand...")
            com, box_size = self.get_ligand_com_and_bounds(target_blurred_pdb)
            
            # Step 2: Extract receptor
            logger.info("2. Extracting receptor...")
            receptor_pdb = prepared_dir / 'receptor.pdb'
            if not self.extract_receptor(target_blurred_pdb, receptor_pdb):
                raise ValueError("Failed to extract receptor")
            
            # Step 3: Prepare receptor PDBQT
            logger.info("3. Preparing receptor PDBQT...")
            receptor_pdbqt = prepared_dir / 'receptor.pdbqt'
            if not self.prepare_receptor_pdbqt(receptor_pdb, receptor_pdbqt):
                raise ValueError("Failed to prepare receptor PDBQT")
            
            # Step 4: Prepare ligand PDBQT
            logger.info("4. Preparing off-target ligand PDBQT...")
            ligand_pdbqt = prepared_dir / 'off_ligand.pdbqt'
            if not self.prepare_ligand_pdbqt(offtarget_ligand, ligand_pdbqt):
                raise ValueError("Failed to prepare ligand PDBQT")
            
            # Step 5: Generate Vina configs
            logger.info("5. Generating Vina configs...")
            if not self.generate_vina_configs(com, box_size, prepared_dir):
                raise ValueError("Failed to generate Vina configs")
            
            logger.info(f"✓ {config_id}: Vina preparation completed")
            self.status_tracker.mark_status(config_id, '4_vina_prep', 'success')
            return True
            
        except Exception as e:
            logger.error(f"✗ {config_id}: Vina preparation failed: {e}")
            self.status_tracker.mark_status(config_id, '4_vina_prep', 'failed', str(e))
            return False


def main():    
    p = argparse.ArgumentParser(description="Stage 4: Vina Preparation")
    p.add_argument("--csv", required=True, help="Input CSV (output from stage 3)")
    p.add_argument("--runs_dir", required=True, help="Runs directory")
    p.add_argument("--status_dir", required=True, help="Status tracking directory")
    p.add_argument("--use_gpu", action='store_true', help="Use GPU acceleration (ignored, for compatibility)")
    p.add_argument("--exhaustiveness", type=int, default=32, help="Exhaustiveness (default: 32)")
    p.add_argument("--num_modes", type=int, default=5, help="Number of modes per seed (default: 5)")
    args = p.parse_args()
    
    # Initialize
    status_tracker = StatusTracker(Path(args.status_dir))
    preparator = VinaPreparator(status_tracker)
    
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
        
        if preparator.process_config(row, config_id, Path(args.runs_dir), 
                                     current_num=current_num, total_num=total_configs):
            # Only add successful rows to output CSV
            row_dict = row.to_dict()
            updated_rows.append(row_dict)
            success_count += 1
        else:
            failed_count += 1
    
    # Save updated CSV (only successful rows)
    output_csv = Path(args.runs_dir).parent / "stage4_output.csv"
    pd.DataFrame(updated_rows).to_csv(output_csv, index=False)
    logger.info(f"Saved {success_count} successful entries to {output_csv}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"=== STAGE 4 SUMMARY ===")
    print(f"{'='*70}")
    print(f"Total processed:  {len(df)}")
    print(f"Success:          {success_count}")
    print(f"Failed:           {failed_count}")
    print(f"Success rate:     {100*success_count/len(df) if len(df) > 0 else 0:.1f}%")
    print(f"Output CSV:       {output_csv}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()