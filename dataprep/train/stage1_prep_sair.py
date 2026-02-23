"""
Stage 1 (SAIR variant): Prepare target structures from SAIR dataset.

SAIR-specific processing:
1. Copy CIF files from pre-computed AlphaFold structures
2. Parse CIF and convert to PDB format
3. Apply standardization and blurring

Directory: /proj/home/ibs/aipd_lab/yunmin_kim/data/sair_structures/structures/
File naming: {uniprot_id}.cif

Output structure:
{base_dir}/
└── {uniprot_id}/
    └── target/
        ├── {target_complex_id}_std.pdb       # Standardized PDB
        └── {target_complex_id}_blurred.pdb   # Blurred backbone
"""

import argparse
import sys
import logging
from pathlib import Path
import pandas as pd
import shutil
from typing import Dict, Tuple, Optional

from Bio.PDB import PDBIO, MMCIFParser, Select, PDBParser, Structure, Model, Chain, Atom
from Bio.PDB.Polypeptide import is_aa
import numpy as np
from utils.status_tracker import StatusTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Stage name for status tracking
STAGE_NAME = '1_prep_sair'

# Constants
WATER_RESNAMES = {'HOH', 'WAT', 'H2O', 'DOD', 'TIP', 'SOL'}
METAL_ION_RESNAMES = {
    'NA','K','CL','CA','MG','ZN','FE','CU','MN','CO','NI',
    'CD','HG','PB','SR','CS','BA','TI','V','CR','MO','SE','W','RB','YB','AL'
}
BACKBONE_ATOMS = {"N", "CA", "C", "O"}

# SAIR structures directory
SAIR_STRUCTURES_DIR = Path("/proj/home/ibs/aipd_lab/yunmin_kim/data/sair_structures/structures")


class BackboneOnlySelect(Select):
    """Select backbone atoms + CB, keep HETATM (ligands)"""
    def __init__(self):
        self.n_protein_atoms_kept = 0
        self.n_protein_atoms_removed = 0
        self.n_hetero_atoms_kept = 0
        
    def accept_atom(self, atom):
        parent_res = atom.get_parent()
        hetflag = parent_res.id[0]
        
        # Protein residues - keep backbone + CB
        if hetflag == " ":
            if is_aa(parent_res, standard=True):
                # Keep backbone + CB
                if atom.name in BACKBONE_ATOMS or atom.name == "CB":
                    self.n_protein_atoms_kept += 1
                    return True
                else:
                    self.n_protein_atoms_removed += 1
                    return False
        
        # HETATM - keep all (ligand)
        self.n_hetero_atoms_kept += 1
        return True


class SAIRTargetPreparator:
    def __init__(self, base_dir: str, status_tracker: StatusTracker):
        self.base_dir = Path(base_dir)
        self.status_tracker = status_tracker
        self.cif_parser = MMCIFParser(QUIET=True)
        self.pdb_parser = PDBParser(QUIET=True)
        self.pdb_io = PDBIO()
        
        # Statistics
        self.n_skipped_existing = 0
        self.auto_selected_ligands = []
        
        if not SAIR_STRUCTURES_DIR.exists():
            logger.error(f"SAIR structures directory not found: {SAIR_STRUCTURES_DIR}")
            raise FileNotFoundError(f"SAIR directory missing: {SAIR_STRUCTURES_DIR}")
        
        logger.info(f"SAIR structures directory: {SAIR_STRUCTURES_DIR}")
    
    def copy_cif_from_sair(self, valid_pdb_id: str, out_dir: Path) -> Optional[Path]:
        """
        Copy CIF file from SAIR structures directory.
        
        Args:
            valid_pdb_id: PDB ID (SAIR uses this for CIF naming)
            out_dir: Output directory for temporary CIF
            
        Returns:
            Path to copied CIF file, or None if not found
        """
        cif_filename = f"{valid_pdb_id}.cif"
        source_cif = SAIR_STRUCTURES_DIR / cif_filename
        
        if not source_cif.exists():
            logger.error(f"SAIR CIF not found: {source_cif}")
            return None
        
        out_dir.mkdir(parents=True, exist_ok=True)
        dest_cif = out_dir / cif_filename
        
        try:
            shutil.copy2(source_cif, dest_cif)
            logger.debug(f"Copied CIF: {source_cif} -> {dest_cif}")
            return dest_cif
        except Exception as e:
            logger.error(f"Failed to copy CIF {source_cif}: {e}")
            return None
    
    def cif_to_pdb(self, cif_path: Path, pdb_path: Path) -> bool:
        """
        Convert CIF to PDB format (keeps all atoms including ligands).
        
        Args:
            cif_path: Input CIF file
            pdb_path: Output PDB file
            
        Returns:
            True if successful
        """
        try:
            structure = self.cif_parser.get_structure('structure', str(cif_path))
            self.pdb_io.set_structure(structure)
            
            pdb_path.parent.mkdir(parents=True, exist_ok=True)
            self.pdb_io.save(str(pdb_path))
            
            return True
        except Exception as e:
            logger.error(f"CIF to PDB conversion failed for {cif_path}: {e}")
            return False
    
    def find_ligand_in_pdb(self, pdb_file: Path) -> tuple:
        """Find ligand in SAIR PDB file (auto-select largest organic ligand)."""
        try:
            structure = self.pdb_parser.get_structure('complex', pdb_file)
            
            ligand_candidates = []  # (chain_id, resid, resname, n_atoms)
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        hetflag = residue.id[0]
                        if hetflag.startswith('H'):  # HETATM
                            resname = residue.resname
                            # Skip water and metal ions
                            if resname in WATER_RESNAMES or resname in METAL_ION_RESNAMES:
                                continue
                            
                            n_atoms = len(list(residue.get_atoms()))
                            ligand_candidates.append((chain.id, residue.id[1], resname, n_atoms))
            
            if ligand_candidates:
                # Select largest ligand
                ligand_candidates.sort(key=lambda x: x[3], reverse=True)
                best = ligand_candidates[0]
                logger.info(f"Auto-selected ligand: {best[2]} (chain {best[0]}, {best[3]} atoms)")
                return best[0], best[1], best[2], ligand_candidates
            
            logger.warning("No suitable ligand found in PDB")
            return None, None, None, None
            
        except Exception as e:
            logger.error(f"Error finding ligand: {e}")
            return None, None, None, None
    
    def extract_and_standardize(self, pdb_file: Path, chain_id: str, 
                                resid: int, resname: str, output_file: Path) -> bool:
        """
        Extract and standardize ligand (same as stage1_prep_target):
        1. Find target ligand in pdb
        2. Renumber ligand to Chain Z, resnum 1
        3. Remove other HETATM/water
        4. Save standardized pdb
        """
        try:
            structure = self.pdb_parser.get_structure('complex', pdb_file)
            
            ligand_found = False
            for model in structure:
                target_chain = None
                target_res = None

                # 1) Find target ligand residue
                for chain in model:
                    if chain.id != chain_id:
                        continue
                    for residue in chain:
                        if (residue.id[1] == resid and 
                            residue.resname == resname and
                            residue.id[0].startswith('H')): # HETATM
                            target_chain = chain
                            target_res = residue
                            ligand_found = True
                            break
                    if target_res is not None:
                        break
                if target_res is None:
                    continue

                # 2) Detach ligand from original chain
                orig_id = target_res.id
                het_field = orig_id[0]
                target_chain.detach_child(orig_id)

                # 3) Create chain Z
                if model.has_id('Z'):
                    chain_z = model['Z']
                else:
                    chain_z = Chain.Chain('Z')
                    model.add(chain_z)

                new_id = (het_field, 1, ' ')  # Standardize to resid 1
                if chain_z.has_id(new_id):
                    chain_z.detach_child(new_id)
                
                # 4) Add ligand to chain Z with standardized residue ID
                new_res = target_res.copy()
                new_res.id = new_id
                chain_z.add(new_res)

                # 5) Remove all other HETATM/water residues
                for chain in list(model):
                    for residue in list(chain):
                        resname_check = residue.resname
                        hetero_flag = residue.id[0]

                        # retain target ligand only (now in Chain Z, ResID 1)
                        if chain.id == 'Z' and residue.id[1] == 1:
                            continue
                        
                        # remove water molecules
                        if resname_check in WATER_RESNAMES:
                            chain.detach_child(residue.id)
                            continue
                        
                        # remove metal ions (by residue name)
                        if resname_check in METAL_ION_RESNAMES:
                            chain.detach_child(residue.id)
                            continue
                        
                        # remove all other HETATM records
                        if hetero_flag.strip():
                            chain.detach_child(residue.id)
                            continue
                
                ligand_found = True
                break

            if not ligand_found:
                logger.warning(f"Ligand {resname} not found at chain {chain_id}, resid {resid}")
                return False
            
            # Save standardized complex pdb
            self.pdb_io.set_structure(structure)
            self.pdb_io.save(str(output_file))
            
            logger.debug(f"Standardized ligand {resname} → Chain Z, ResID 1")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract/standardize ligand: {e}")
            return False
    
    def apply_sequence_blurring(self, input_pdb: Path, output_pdb: Path) -> bool:
        """
        Apply sequence blurring:
        - Non-GLY residues → ALA (keep backbone + CB)
        - GLY → GLY (keep backbone only, no virtual CB to avoid steric clash)
        """
        try:
            structure = self.pdb_parser.get_structure('input', input_pdb)
            
            # First pass: Change protein residues to ALA (except GLY)
            for model in structure:
                for chain in model:
                    for residue in list(chain):
                        hetflag = residue.id[0]
                        # Only modify standard protein residues
                        if hetflag == " " and is_aa(residue, standard=True):
                            # Keep GLY as GLY (no virtual CB to avoid steric clash)
                            if residue.resname != "GLY":
                                residue.resname = "ALA"
            
            # Second pass: Save with selector (keeps backbone + CB, removes side-chains)
            selector = BackboneOnlySelect()
            self.pdb_io.set_structure(structure)
            self.pdb_io.save(str(output_pdb), selector)
            
            logger.info(f"Sequence blurring completed: {output_pdb}")
            logger.info(f"  Protein atoms kept: {selector.n_protein_atoms_kept}")
            logger.info(f"  Protein atoms removed: {selector.n_protein_atoms_removed}")
            logger.info(f"  HETATM atoms kept: {selector.n_hetero_atoms_kept}")
            
            return True
            
        except Exception as e:
            logger.error(f"Sequence blurring failed: {e}")
            return False
    
    def validate_ligand_present(self, pdb_file: Path) -> bool:
        """Check if the PDB file contains HETATM (ligand) records."""
        try:
            with open(pdb_file) as f:
                for line in f:
                    if line.startswith('HETATM'):
                        return True
            return False
        except Exception as e:
            logger.error(f"Failed to validate ligand in {pdb_file}: {e}")
            return False
    
    def process_row(self, row: pd.Series, target_complex_id: str, uniprot_id: str) -> Tuple[bool, Dict]:
        """
        Process a single SAIR target row.
        
        Args:
            row: Input CSV row
            target_complex_id: Pre-assigned complex ID (e.g., "P12345_tar_000001")
            uniprot_id: UniProt ID
            
        Returns:
            (success, updated_data_dict)
        """
        # Check if already processed
        if self.status_tracker.check_status(target_complex_id, STAGE_NAME) == 'success':
            logger.info(f"[SKIP] {target_complex_id} - already processed")
            self.n_skipped_existing += 1
            
            # Return existing paths from row if available
            existing_std = row.get('target_std_path', '')
            existing_blurred = row.get('target_blurred_path', '')
            
            if existing_std and existing_blurred:
                return True, {
                    'target_std_path': existing_std,
                    'target_blurred_path': existing_blurred
                }
        
        # Define output paths
        protein_dir = self.base_dir / uniprot_id / "target"
        protein_dir.mkdir(parents=True, exist_ok=True)
        
        target_std_path = protein_dir / f"{target_complex_id}_std.pdb"
        target_blurred_path = protein_dir / f"{target_complex_id}_blurred.pdb"
        
        # Temporary directory for CIF conversion
        temp_dir = protein_dir / "_temp"
        temp_dir.mkdir(exist_ok=True)
        temp_pdb = None
        
        try:
            # Get valid_pdb_id from row
            valid_pdb_id = row.get('valid_pdb_id', '')
            if not valid_pdb_id or pd.isna(valid_pdb_id) or str(valid_pdb_id).strip().upper() == 'NAN':
                logger.error(f"[SAIR] Missing valid_pdb_id for {target_complex_id}")
                self.status_tracker.mark_status(target_complex_id, STAGE_NAME, 'failed', 'missing_pdb_id')
                return False, {}
            
            valid_pdb_id = str(valid_pdb_id).strip()
            
            # Step 1: Copy CIF from SAIR
            logger.info(f"[SAIR] Copying CIF for {target_complex_id} (PDB: {valid_pdb_id})")
            cif_path = self.copy_cif_from_sair(valid_pdb_id, temp_dir)
            if not cif_path:
                self.status_tracker.mark_status(target_complex_id, STAGE_NAME, 'failed', 'cif_not_found')
                return False, {}
            
            # Step 2: Convert CIF to PDB (temp)
            temp_pdb = temp_dir / f"{valid_pdb_id}_temp.pdb"
            logger.info(f"[SAIR] Converting CIF to PDB: {target_complex_id}")
            if not self.cif_to_pdb(cif_path, temp_pdb):
                self.status_tracker.mark_status(target_complex_id, STAGE_NAME, 'failed', 'cif_conversion_failed')
                return False, {}
            
            # Step 3: Find ligand in converted PDB
            logger.info(f"[SAIR] Finding ligand in PDB: {target_complex_id}")
            chain_id, resid, resname, candidates = self.find_ligand_in_pdb(temp_pdb)
            if not chain_id:
                logger.error(f"[SAIR] No ligand found in PDB: {target_complex_id}")
                self.status_tracker.mark_status(target_complex_id, STAGE_NAME, 'failed', 'no_ligand_found')
                return False, {}
            
            # Step 4: Extract and standardize (ligand → Chain Z, ResID 1)
            logger.info(f"[SAIR] Standardizing ligand {resname}: {target_complex_id}")
            if not self.extract_and_standardize(temp_pdb, chain_id, resid, resname, target_std_path):
                self.status_tracker.mark_status(target_complex_id, STAGE_NAME, 'failed', 'standardization_failed')
                return False, {}
            
            # Step 5: Apply sequence blurring
            logger.info(f"[SAIR] Applying sequence blurring: {target_complex_id}")
            if not self.apply_sequence_blurring(target_std_path, target_blurred_path):
                self.status_tracker.mark_status(target_complex_id, STAGE_NAME, 'failed', 'blurring_failed')
                return False, {}
            
            # Step 6: Validate ligand is present in output PDB
            if not self.validate_ligand_present(target_std_path):
                logger.error(f"[SAIR] No ligand in standardized PDB: {target_complex_id}")
                self.status_tracker.mark_status(target_complex_id, STAGE_NAME, 'failed', 'no_ligand_std')
                target_std_path.unlink(missing_ok=True)
                target_blurred_path.unlink(missing_ok=True)
                return False, {}
            
            # Cleanup temp
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Mark success
            self.status_tracker.mark_status(target_complex_id, STAGE_NAME, 'success', 'sair_processed')
            
            logger.info(f"[SUCCESS] {target_complex_id}")
            return True, {
                'target_std_path': str(target_std_path),
                'target_blurred_path': str(target_blurred_path)
            }
            
        except Exception as e:
            logger.error(f"[FAILED] {target_complex_id}: {e}")
            self.status_tracker.mark_status(target_complex_id, STAGE_NAME, 'failed', str(e)[:200])
            return False, {}
        finally:
            # Clean up temp files
            if temp_pdb and temp_pdb.exists():
                temp_pdb.unlink(missing_ok=True)
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def extract_ligand_only(self, pdb_file: Path, output_pdb: Path) -> bool:
        """
        Extract only the largest organic ligand from PDB file (no protein).
        """
        try:
            chain_id, resid, resname, candidates = self.find_ligand_in_pdb(pdb_file)
            if not chain_id:
                logger.warning(f"No ligand found for ligand-only extraction: {pdb_file}")
                return False

            structure = self.pdb_parser.get_structure('complex', str(pdb_file))

            # Selector that keeps only the target ligand residue
            _chain_id, _resid, _resname = chain_id, resid, resname

            class SingleLigandSelect(Select):
                def accept_residue(self, residue):
                    return (
                        residue.parent.id == _chain_id
                        and residue.id[1] == _resid
                        and residue.resname == _resname
                    )

            output_pdb.parent.mkdir(parents=True, exist_ok=True)
            self.pdb_io.set_structure(structure)
            self.pdb_io.save(str(output_pdb), SingleLigandSelect())

            if not output_pdb.exists() or output_pdb.stat().st_size == 0:
                logger.warning(f"Ligand-only PDB is empty: {output_pdb}")
                return False

            logger.debug(f"Extracted ligand-only PDB: {output_pdb}")
            return True
        except Exception as e:
            logger.error(f"Ligand-only extraction failed for {pdb_file}: {e}")
            return False

    def process_offtarget_row(self, row: pd.Series, offtarget_complex_id: str, uniprot_id: str) -> Tuple[bool, Dict]:
        """
        Process a single SAIR off-target row: extract the ligand-only PDB.

        Args:
            row: Input CSV row
            offtarget_complex_id: Pre-assigned complex ID (e.g., "P12345_off_000001")
            uniprot_id: UniProt ID

        Returns:
            (success, updated_data_dict) — updated_data_dict contains 'ligand_path'
        """
        OFFTARGET_STAGE = '1_prep_sair_offtarget'

        # Skip if already processed and file still exists
        if self.status_tracker.check_status(offtarget_complex_id, OFFTARGET_STAGE) == 'success':
            existing_path = row.get('ligand_path', '')
            if existing_path and Path(str(existing_path)).exists():
                logger.info(f"[SKIP offtarget] {offtarget_complex_id} - already processed")
                return True, {'ligand_path': existing_path}

        # Define output path
        ligand_dir = self.base_dir / uniprot_id / "offtargets" / "from_sair"
        ligand_dir.mkdir(parents=True, exist_ok=True)
        ligand_pdb = ligand_dir / f"{offtarget_complex_id}.pdb"

        temp_dir = ligand_dir / "_temp"
        temp_dir.mkdir(exist_ok=True)
        temp_pdb = None

        try:
            valid_pdb_id = row.get('valid_pdb_id', '')
            if not valid_pdb_id or pd.isna(valid_pdb_id) or str(valid_pdb_id).strip().upper() == 'NAN':
                logger.error(f"[SAIR offtarget] Missing valid_pdb_id for {offtarget_complex_id}")
                self.status_tracker.mark_status(offtarget_complex_id, OFFTARGET_STAGE, 'failed', 'missing_pdb_id')
                return False, {}

            valid_pdb_id = str(valid_pdb_id).strip()

            # Step 1: Copy CIF from SAIR
            cif_path = self.copy_cif_from_sair(valid_pdb_id, temp_dir)
            if not cif_path:
                self.status_tracker.mark_status(offtarget_complex_id, OFFTARGET_STAGE, 'failed', 'cif_not_found')
                return False, {}

            # Step 2: Convert CIF to PDB (temp)
            temp_pdb = temp_dir / f"{valid_pdb_id}_temp.pdb"
            if not self.cif_to_pdb(cif_path, temp_pdb):
                self.status_tracker.mark_status(offtarget_complex_id, OFFTARGET_STAGE, 'failed', 'cif_conversion_failed')
                return False, {}

            # Step 3: Extract ligand-only PDB
            if not self.extract_ligand_only(temp_pdb, ligand_pdb):
                self.status_tracker.mark_status(offtarget_complex_id, OFFTARGET_STAGE, 'failed', 'ligand_extraction_failed')
                return False, {}

            # Cleanup temp
            shutil.rmtree(temp_dir, ignore_errors=True)

            self.status_tracker.mark_status(offtarget_complex_id, OFFTARGET_STAGE, 'success', 'offtarget_ligand_extracted')
            logger.info(f"[SUCCESS offtarget] {offtarget_complex_id} -> {ligand_pdb}")
            return True, {'ligand_path': str(ligand_pdb)}

        except Exception as e:
            logger.error(f"[FAILED offtarget] {offtarget_complex_id}: {e}")
            self.status_tracker.mark_status(offtarget_complex_id, OFFTARGET_STAGE, 'failed', str(e)[:200])
            return False, {}
        finally:
            if temp_pdb and temp_pdb.exists():
                temp_pdb.unlink(missing_ok=True)
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description="Stage 1 (SAIR): Prepare SAIR target structures")
    p.add_argument("--csv", required=True, help="Input indexed CSV (SAIR rows only)")
    p.add_argument("--base_dir", required=True, help="Base output directory")
    p.add_argument("--status_dir", required=True, help="Status tracking directory")
    args = p.parse_args()
    
    # Initialize
    status_tracker = StatusTracker(Path(args.status_dir))
    preparator = SAIRTargetPreparator(args.base_dir, status_tracker)
    
    # Load CSV
    df = pd.read_csv(args.csv)
    logger.info(f"Loaded {len(df)} total entries")
    
    # Validate required columns
    required_cols = ['uniprot_id', 'target_type', 'target_complex_id', 'offtarget_complex_id', 'data_source']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Required columns missing: {missing_cols}")
        sys.exit(1)
    
    # Filter SAIR rows only
    df_sair = df[df['data_source'] == 'SAIR'].copy()
    logger.info(f"Filtered {len(df_sair)} SAIR entries")
    
    if len(df_sair) == 0:
        logger.warning("No SAIR entries found in input CSV")
        # Save empty output
        output_csv = Path(args.base_dir) / "stage1_sair_output.csv"
        df_sair.to_csv(output_csv, index=False)
        logger.info(f"Empty output saved to: {output_csv}")
        return
    
    # Separate target and off-target
    target_df = df_sair[df_sair['target_type'] == 'target'].copy()
    off_target_df = df_sair[df_sair['target_type'] != 'target'].copy()
    
    n_proteins = df_sair['uniprot_id'].nunique()
    
    logger.info(f"Found {n_proteins} unique SAIR proteins")
    logger.info(f"Processing {len(target_df)} SAIR target entries")
    logger.info(f"Passing through {len(off_target_df)} SAIR off-target entries")
    
    if len(target_df) == 0:
        logger.error("No SAIR target entries found!")
        sys.exit(1)
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"=== STAGE 1 (SAIR) INPUT ANALYSIS ===")
    print(f"{'='*70}")
    print(f"Unique SAIR proteins:    {n_proteins}")
    print(f"SAIR target entries:     {len(target_df)}")
    print(f"SAIR off-target entries: {len(off_target_df)}")
    print(f"{'='*70}\n")
    
    # Process target rows
    updated_target_rows = []
    success_count = 0
    failed_count = 0
    
    for idx, row in target_df.iterrows():
        target_complex_id = row['target_complex_id']
        uniprot_id = str(row['uniprot_id']).strip()
        if uniprot_id.upper() == 'NAN' or not uniprot_id:
            logger.warning(f"Missing uniprot_id for row {idx}, skipping")
            failed_count += 1
            continue
        
        success, updated_data = preparator.process_row(row, target_complex_id, uniprot_id)
        
        if success:
            row_dict = row.to_dict()
            row_dict.update(updated_data)
            updated_target_rows.append(row_dict)
            success_count += 1
        else:
            failed_count += 1
    
    # Process off-target rows: extract ligand-only PDB and record ligand_path
    off_target_rows = []
    off_success_count = 0
    off_failed_count = 0

    for idx, row in off_target_df.iterrows():
        offtarget_complex_id = row['offtarget_complex_id']
        uniprot_id_off = str(row['uniprot_id']).strip()
        if uniprot_id_off.upper() == 'NAN' or not uniprot_id_off:
            logger.warning(f"Missing uniprot_id for off-target row {idx}, skipping")
            off_failed_count += 1
            off_target_rows.append(row.to_dict())
            continue

        success_off, updated_off = preparator.process_offtarget_row(row, offtarget_complex_id, uniprot_id_off)

        row_dict = row.to_dict()
        if success_off:
            row_dict.update(updated_off)  # adds 'ligand_path'
            off_success_count += 1
        else:
            off_failed_count += 1
            logger.warning(f"[SAIR offtarget] Could not extract ligand for {offtarget_complex_id}; row included without ligand_path")

        off_target_rows.append(row_dict)

    # Combine and save
    all_rows = updated_target_rows + off_target_rows
    output_csv = Path(args.base_dir) / "stage1_sair_output.csv"

    if all_rows:
        pd.DataFrame(all_rows).to_csv(output_csv, index=False)
        logger.info(f"Saved {len(updated_target_rows)} processed SAIR target entries")
        logger.info(f"Saved {len(off_target_rows)} SAIR off-target entries (with ligand_path)")
    else:
        logger.error("No entries to save!")
        pd.DataFrame(columns=['target_complex_id', 'target_std_path', 'target_blurred_path', 'ligand_path']).to_csv(output_csv, index=False)

    # Summary
    print(f"{'='*70}")
    print(f"=== STAGE 1 (SAIR) SUMMARY ===")
    print(f"{'='*70}")
    print(f"Unique proteins:       {n_proteins}")
    print(f"SAIR target entries:   {len(target_df)}")
    print(f"  Skipped (exists):    {preparator.n_skipped_existing}")
    print(f"  Success (new):       {success_count - preparator.n_skipped_existing}")
    print(f"  Total success:       {success_count}")
    print(f"  Failed:              {failed_count}")
    print(f"  Success rate:        {100*success_count/len(target_df) if len(target_df) > 0 else 0:.1f}%")
    print(f"SAIR off-target entries: {len(off_target_df)}")
    print(f"  Ligand extracted:    {off_success_count}")
    print(f"  Failed:              {off_failed_count}")
    print(f"  Success rate:        {100*off_success_count/len(off_target_df) if len(off_target_df) > 0 else 0:.1f}%")
    print(f"Total output rows:     {len(all_rows)}")
    print(f"Output CSV:            {output_csv}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
