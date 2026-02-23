"""
Stage 2: Save target and off-target ligand files
Based on save_ligand.py

Note: Run after Stage 1 (PDB preparation)
Skips already saved ligands based on actual file existence
"""
import os
import sys
import logging
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
from rdkit import Chem
from rdkit.Chem import AllChem
import requests
import pandas as pd
import argparse
from typing import Optional, Tuple, Tuple as TupleTy, Dict
from utils.status_tracker import StatusTracker
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LigandExtractor(Select):
    """BioPython Select class to extract specific ligand from PDB"""
    def __init__(self, chain_id: str, res_id: int, res_name: str):
        self.chain_id = chain_id
        self.res_id = res_id
        self.res_name = res_name
        
    def accept_residue(self, residue):
        return (residue.parent.id == self.chain_id and 
                residue.id[1] == self.res_id and
                residue.resname == self.res_name)


class LigandSaver:
    def __init__(self, base_dir: str, status_tracker: StatusTracker, max_heavy_atoms: Optional[int] = None):
        self.base_dir = Path(base_dir)
        self.status_tracker = status_tracker
        self.max_heavy_atoms = max_heavy_atoms
        
        # Error statistics tracking
        self.smiles_error_stats = defaultdict(int)
        self.smiles_error_examples = defaultdict(list)
        
        # Track ligands rejected by the heavy-atom size filter
        self.too_large_ligands: list = []
        
        # Pre-existing ligand directories to check first
        self.pre_existing_dirs = [
            Path("/scratch/yunmin/data/graph/lmpnn/bdb_pdb/off_inputs/ligand_sdf")
        ]
        
        # Per-protein directories (set dynamically in process_row)
        self.from_pdb_dir = None
        self.from_pubchem_dir = None
        self.from_smiles_dir = None
            
        self.parser = PDBParser(QUIET=True)
        self.io = PDBIO()
        
        # Track skipped entries
        self.n_skipped_existing = 0
    
    def check_pre_existing(self, het_id: str) -> Tuple[Optional[str], str]:
        """
        Check if ligand file already exists in pre-existing directories
        Returns: (file_path, source) or (None, 'not_found')
        """
        if pd.isna(het_id) or het_id is None:
            return None, 'missing_het_id'
        
        het_id = str(het_id).strip()
        
        if not het_id or het_id.lower() == 'nan':
            return None, 'missing_het_id'
        
        for pre_dir in self.pre_existing_dirs:
            if not pre_dir.exists():
                continue
            
            # Look for files matching the HET ID
            for pattern in [f"{het_id}*.sdf", f"{het_id}*.pdb", f"{het_id}*.mol2"]:
                files = list(pre_dir.glob(pattern))
                if files:
                    logger.info(f"Found pre-existing ligand {het_id} at {files[0]}")
                    return str(files[0]), 'pre_existing'
        
        return None, 'not_found'
    
    def download_pdb(self, pdb_id: str) -> Optional[Path]:
        """Download PDB file from RCSB"""
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        temp_file = self.from_pdb_dir / f"{pdb_id}_full.pdb"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(temp_file, 'w') as f:
                f.write(response.text)
            
            logger.info(f"Downloaded PDB {pdb_id}")
            return temp_file
        except Exception as e:
            logger.error(f"Failed to download PDB {pdb_id}: {e}")
            return None
    
    def extract_ligand_from_pdb(self, pdb_file: Path, het_id: str, pdb_id: str) -> Optional[Path]:
        """Extract ligand from PDB file"""
        try:
            structure = self.parser.get_structure('complex', pdb_file)
            
            # Find ligand
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.resname == het_id and residue.id[0].startswith('H'):
                            chain_id = chain.id
                            res_id = residue.id[1]
                            
                            output_file = self.from_pdb_dir / f"{pdb_id}_{het_id}.pdb"
                            
                            self.io.set_structure(structure)
                            self.io.save(str(output_file), LigandExtractor(chain_id, res_id, het_id))
                            
                            logger.info(f"Extracted {het_id} from {pdb_id}")
                            return output_file
            
            logger.warning(f"Ligand {het_id} not found in {pdb_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract ligand from {pdb_id}: {e}")
            return None
        
    def download_from_pubchem(self, het_id: str) -> Optional[Path]:
        """Download ligand from PubChem by name/HET ID"""
        # Validate het_id before making request
        if not het_id or het_id.lower() == 'nan':
            logger.warning(f"Skipping PubChem download for invalid HET ID: {het_id}")
            return None
        
        output_file = self.from_pubchem_dir / f"{het_id}_pubchem.sdf"
        
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{het_id}/SDF"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_file, 'w') as f:
                f.write(response.text)
            
            # Validate SDF
            mol = Chem.MolFromMolFile(str(output_file))
            if mol is None:
                output_file.unlink()
                logger.warning(f"Invalid SDF downloaded for {het_id}")
                return None
            
            logger.info(f"Downloaded {het_id} from PubChem")
            return output_file
            
        except Exception as e:
            logger.warning(f"Failed to download from PubChem ({het_id}): {e}")
            if output_file.exists():
                output_file.unlink()
            return None
    
    def generate_from_smiles(self, smiles: str, identifier: str, complex_id: str = None) -> Tuple[Optional[Path], str]:
        """Generate 3D structure from SMILES with robust methods and detailed error tracking
        
        Args:
            smiles: SMILES string
            identifier: Unique identifier (e.g., 'PDB_HET' or 'complex_id')
            
        Returns:
            (output_file_path, error_type) where error_type is one of:
            - 'success': Successfully generated
            - 'invalid_smiles': SMILES parsing failed
            - 'sanitization_failed': Molecule sanitization failed
            - 'molecule_too_large': Heavy atom count exceeds threshold (skipped)
            - 'embed_failed': All embedding methods failed
            - 'optimization_failed': Force field optimization failed
            - 'write_failed': File writing failed
            - 'unknown_error': Unexpected error
        """
        output_file = self.from_smiles_dir / f"{identifier}_generated.sdf"
        error_details = []
        
        try:
            # Step 1: Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                error_msg = f"Invalid SMILES syntax: {smiles[:50]}..."
                logger.error(error_msg)
                error_details.append(error_msg)
                return None, 'invalid_smiles'
            
            # Step 2: Sanitize molecule
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                error_msg = f"Sanitization failed: {str(e)}"
                logger.error(error_msg)
                error_details.append(error_msg)
                return None, 'sanitization_failed'
            
            # Step 2.5: Reject oversized molecules (not drug-like; embed would take hours)
            num_heavy_pre = mol.GetNumHeavyAtoms()
            if self.max_heavy_atoms is not None and num_heavy_pre > self.max_heavy_atoms:
                error_msg = (
                    f"Molecule too large ({num_heavy_pre} heavy atoms > {self.max_heavy_atoms}); "
                    f"skipping SMILES-based 3D generation"
                )
                logger.warning(error_msg)
                error_details.append(error_msg)
                self.too_large_ligands.append({
                    'complex_id': complex_id or identifier,
                    'identifier': identifier,
                    'smiles': smiles[:120],
                    'heavy_atoms': num_heavy_pre,
                })
                self.smiles_error_stats['molecule_too_large'] += 1
                if len(self.smiles_error_examples['molecule_too_large']) < 10:
                    self.smiles_error_examples['molecule_too_large'].append({
                        'complex_id': complex_id or identifier,
                        'smiles': smiles[:100],
                        'heavy_atoms': num_heavy_pre,
                        'het_id': None,
                        'pdb_id': None,
                    })
                return None, 'molecule_too_large'

            # Step 3: Add hydrogens
            mol = Chem.AddHs(mol)
            num_atoms = mol.GetNumAtoms()
            num_heavy = mol.GetNumHeavyAtoms()
            logger.info(f"Molecule: {num_heavy} heavy atoms, {num_atoms} total atoms")
            
            # Step 4: Generate 3D coordinates with multiple methods
            embed_success = False
            
            # Method 1: ETKDG (most robust, recommended method)
            try:
                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                params.useRandomCoords = False  # Use distance geometry
                params.numThreads = 0  # Use all available threads
                
                result = AllChem.EmbedMolecule(mol, params)
                if result == 0:
                    logger.info(f"Successfully embedded with ETKDGv3")
                    embed_success = True
                else:
                    error_details.append(f"ETKDGv3 failed with code {result}")
            except Exception as e:
                error_details.append(f"ETKDGv3 exception: {str(e)}")
            
            # Method 2: Standard ETKDG (fallback)
            if not embed_success:
                try:
                    params = AllChem.ETKDG()
                    params.randomSeed = 42
                    result = AllChem.EmbedMolecule(mol, params)
                    if result == 0:
                        logger.info(f"Successfully embedded with ETKDG (fallback 1)")
                        embed_success = True
                    else:
                        error_details.append(f"ETKDG failed with code {result}")
                except Exception as e:
                    error_details.append(f"ETKDG exception: {str(e)}")
            
            # Method 3: Basic embedding with random coords (last resort)
            if not embed_success:
                try:
                    result = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
                    if result == 0:
                        logger.info(f"Successfully embedded with random coords (fallback 2)")
                        embed_success = True
                    else:
                        error_details.append(f"Random coords failed with code {result}")
                except Exception as e:
                    error_details.append(f"Random coords exception: {str(e)}")
            
            if not embed_success:
                error_msg = f"All embedding methods failed. Errors: {'; '.join(error_details)}"
                logger.error(error_msg)
                return None, 'embed_failed'
            
            # Step 5: Optimize geometry
            try:
                # Try MMFF first (more accurate)
                props = AllChem.MMFFGetMoleculeProperties(mol)
                if props is not None:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props)
                    if ff is not None:
                        ff.Minimize()
                        logger.info(f"Optimized with MMFF")
                    else:
                        # Fallback to UFF
                        ff = AllChem.UFFGetMoleculeForceField(mol)
                        ff.Minimize()
                        logger.info(f"Optimized with UFF (MMFF unavailable)")
                else:
                    # Fallback to UFF
                    ff = AllChem.UFFGetMoleculeForceField(mol)
                    ff.Minimize()
                    logger.info(f"Optimized with UFF (MMFF properties unavailable)")
            except Exception as e:
                # Don't fail if optimization fails, geometry might still be usable
                error_msg = f"Optimization failed but continuing: {str(e)}"
                logger.warning(error_msg)
                error_details.append(error_msg)
            
            # Step 6: Write to file
            try:
                writer = Chem.SDWriter(str(output_file))
                writer.write(mol)
                writer.close()
                
                # Validate the written file
                test_mol = Chem.MolFromMolFile(str(output_file))
                if test_mol is None:
                    error_msg = f"Written SDF file is invalid"
                    logger.error(error_msg)
                    if output_file.exists():
                        output_file.unlink()
                    return None, 'write_failed'
                
                logger.info(f"Successfully generated 3D structure from SMILES")
                return output_file, 'success'
                
            except Exception as e:
                error_msg = f"Failed to write SDF file: {str(e)}"
                logger.error(error_msg)
                if output_file.exists():
                    output_file.unlink()
                return None, 'write_failed'
            
        except Exception as e:
            error_msg = f"Unexpected error in SMILES generation: {str(e)}"
            logger.error(error_msg)
            error_details.append(error_msg)
            return None, 'unknown_error'
    
    def get_ligand(self, het_id: Optional[str] = None, pdb_id: Optional[str] = None, 
                   smiles: Optional[str] = None, complex_id: str = "unknown") -> Tuple[Optional[str], str, list]:
        """
        Get ligand file path with fallback priority
        Returns: (file_path, source, attempt_log)
        
        Priority order:
        1. Check pre-existing directories (if het_id available)
        2. Extract from PDB (if pdb_id available)
        3. Download from PubChem (if het_id available)
        4. Generate from SMILES (if smiles available)
        
        Args:
            het_id: HET ID of the ligand
            pdb_id: PDB ID to extract from
            smiles: SMILES string
            complex_id: Unique identifier for this complex (for file naming)
        """
        attempt_log = []
        
        # Validate and normalize inputs
        het_id = self._validate_input(het_id)
        pdb_id = self._validate_input(pdb_id)
        smiles = self._validate_input(smiles)
        
        # Priority 1: Check pre-existing directories
        if het_id:
            attempt_log.append(f"1. Checking pre-existing directories for {het_id}...")
            pre_existing_path, source = self.check_pre_existing(het_id)
            if pre_existing_path:
                attempt_log.append(f"   ✓ SUCCESS: Found pre-existing ligand at {pre_existing_path}")
                return pre_existing_path, source, attempt_log
            attempt_log.append(f"   ✗ Not found in pre-existing directories")
        else:
            attempt_log.append(f"1. Skipping pre-existing check (no het_id)")
        
        # Priority 2: Extract from PDB
        if pdb_id:
            attempt_log.append(f"2. Attempting to extract from PDB {pdb_id}...")
            temp_pdb = self.download_pdb(pdb_id)
            if temp_pdb:
                ligand_file = self.extract_ligand_from_pdb(temp_pdb, het_id, pdb_id)
                if temp_pdb.exists():
                    temp_pdb.unlink()
                if ligand_file:
                    attempt_log.append(f"   ✓ SUCCESS: Extracted ligand from PDB")
                    return str(ligand_file), 'pdb', attempt_log
                else:
                    attempt_log.append(f"   ✗ Failed to extract ligand {het_id} from PDB")
            else:
                attempt_log.append(f"   ✗ Failed to download PDB")
        else:
            attempt_log.append(f"2. Skipping PDB extraction (no pdb_id)")
        
        # Priority 3: Download from PubChem
        if het_id:
            attempt_log.append(f"3. Attempting to download from PubChem {het_id}...")
            ligand_file = self.download_from_pubchem(het_id)
            if ligand_file:
                attempt_log.append(f"   ✓ SUCCESS: Downloaded from PubChem")
                return str(ligand_file), 'pubchem', attempt_log
            attempt_log.append(f"   ✗ Failed to download from PubChem")
        else:
            attempt_log.append(f"3. Skipping PubChem download (no het_id)")
        
        # Priority 4: Generate from SMILES
        if smiles:
            attempt_log.append(f"4. Attempting to generate from SMILES...")
            attempt_log.append(f"   SMILES: {smiles[:80]}..." if len(smiles) > 80 else f"   SMILES: {smiles}")
            
            # Create unique identifier for filename
            if pdb_id and het_id:
                identifier = f"{pdb_id}_{het_id}"
            elif pdb_id:
                identifier = f"{pdb_id}_LIG"
            elif het_id:
                identifier = het_id
            else:
                identifier = complex_id
            
            ligand_file, error_type = self.generate_from_smiles(smiles, identifier, complex_id=complex_id)
            
            # Track error statistics
            # Note: 'molecule_too_large' is already tracked inside generate_from_smiles
            if error_type != 'success' and error_type != 'molecule_too_large':
                self.smiles_error_stats[error_type] += 1
                if len(self.smiles_error_examples[error_type]) < 10:  # Keep first 10 examples
                    self.smiles_error_examples[error_type].append({
                        'complex_id': complex_id,
                        'smiles': smiles[:100],
                        'het_id': het_id,
                        'pdb_id': pdb_id
                    })
            
            if ligand_file:
                attempt_log.append(f"   ✓ SUCCESS: Generated from SMILES as {identifier}_generated.sdf")
                return str(ligand_file), 'smiles', attempt_log
            else:
                attempt_log.append(f"   ✗ Failed to generate from SMILES: {error_type}")
        else:
            attempt_log.append(f"4. Skipping SMILES generation (no smiles)")
        
        attempt_log.append(f"\n✗ ALL METHODS FAILED: Could not obtain ligand")
        return None, 'failed', attempt_log
    
    def _validate_input(self, value: Optional[str]) -> Optional[str]:
        """Normalize and validate string input"""
        if pd.isna(value) or value is None:
            return None
        
        value = str(value).strip()
        
        if not value or value.lower() == 'nan':
            return None
        
        return value
    
    def check_existing_ligand(self, complex_id: str) -> Tuple[Optional[str], str]:
        """
        Check if ligand file already exists in output directories
        Returns: (file_path, source) or (None, 'not_found')
        
        Checks for valid ligand files in:
        - from_pdb/
        - from_pubchem/
        - from_smiles/
        - pre_existing directories
        """
        # Check all output directories
        search_dirs = [
            (self.from_pdb_dir, 'pdb'),
            (self.from_pubchem_dir, 'pubchem'),
            (self.from_smiles_dir, 'smiles'),
        ]
        
        # Also check pre-existing directories
        for pre_dir in self.pre_existing_dirs:
            if pre_dir.exists():
                search_dirs.append((pre_dir, 'pre_existing'))
        
        # Look for files matching this complex
        for search_dir, source in search_dirs:
            if not search_dir.exists():
                continue
            
            # Look for common ligand file formats
            for pattern in [f"*{complex_id}*", f"{complex_id}_*"]:
                for ext in ['.sdf', '.pdb', '.mol2', '.mol']:
                    files = list(search_dir.glob(f"{pattern}{ext}"))
                    for f in files:
                        # Validate the file is not empty and is a valid structure
                        if f.stat().st_size > 100:  # Minimum reasonable size
                            try:
                                # Quick validation for SDF/PDB files
                                if ext in ['.sdf', '.mol']:
                                    mol = Chem.MolFromMolFile(str(f))
                                    if mol is not None:
                                        logger.info(f"Found existing valid ligand for {complex_id}: {f}")
                                        return str(f), source
                                elif ext == '.pdb':
                                    # Quick check: file should have ATOM or HETATM lines
                                    with open(f, 'r') as fh:
                                        content = fh.read(1000)
                                        if 'ATOM' in content or 'HETATM' in content:
                                            logger.info(f"Found existing valid ligand for {complex_id}: {f}")
                                            return str(f), source
                            except Exception as e:
                                logger.warning(f"Failed to validate {f}: {e}")
                                continue
        
        return None, 'not_found'
    
    def process_row(self, row: pd.Series, offtarget_complex_id: str, uniprot_id: str) -> Tuple[bool, dict, list]:
        """
        Process one off-target row: save off-target ligand file
        
        CSV columns expected:
        - ligand_het_id: HET ID for the ligand (optional)
        - valid_pdb_id: PDB ID to extract from (optional)
        - ligand_smiles: SMILES string (optional)
        
        Returns:
            (success, updated_data_dict, attempt_log)
        """
        attempt_log = []
        
        try:
            # Set per-protein output directories
            protein_offtarget_dir = self.base_dir / uniprot_id / "offtargets"
            self.from_pdb_dir = protein_offtarget_dir / "from_pdb"
            self.from_pubchem_dir = protein_offtarget_dir / "from_pubchem"
            self.from_smiles_dir = protein_offtarget_dir / "from_smiles"
            
            for dir_path in [self.from_pdb_dir, self.from_pubchem_dir, self.from_smiles_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Check if ligand already exists (skip if found)
            existing_path, existing_source = self.check_existing_ligand(offtarget_complex_id)
            if existing_path:
                attempt_log.append(f"\n{'='*70}")
                attempt_log.append(f"✓ {offtarget_complex_id}: SKIPPED - Ligand already exists")
                attempt_log.append(f"  Source: {existing_source}")
                attempt_log.append(f"  Path: {existing_path}")
                attempt_log.append(f"{'='*70}\n")
                
                self.n_skipped_existing += 1
                self.status_tracker.mark_status(offtarget_complex_id, '2_save_ligands', 'success')
                
                return True, {
                    'ligand_path': existing_path,
                    'ligand_source': existing_source,
                    'offtarget_complex_id': offtarget_complex_id,
                    'skipped': True
                }, attempt_log
            
            # Extract ligand info from CSV columns
            het_id = row.get('ligand_het_id')
            pdb_id = row.get('valid_pdb_id')
            smiles = row.get('ligand_smiles')
            
            # Check that at least one identifier is available
            het_id_valid = self._validate_input(het_id)
            pdb_id_valid = self._validate_input(pdb_id)
            smiles_valid = self._validate_input(smiles)
            
            if not any([het_id_valid, pdb_id_valid, smiles_valid]):
                raise ValueError(
                    "No ligand identifier found. Need at least one of: "
                    "ligand_het_id, valid_pdb_id, or ligand_smiles"
                )
            
            attempt_log.append(f"\n{'='*70}")
            attempt_log.append(f"Processing {offtarget_complex_id} ({uniprot_id})")
            attempt_log.append(f"  HET ID: {het_id if het_id_valid else '(none)'}")
            attempt_log.append(f"  PDB ID: {pdb_id if pdb_id_valid else '(none)'}")
            attempt_log.append(f"  SMILES: {smiles if smiles_valid else '(none)'}")
            attempt_log.append(f"{'='*70}")
            
            # Get ligand with priority order
            ligand_path, ligand_source, method_log = self.get_ligand(
                het_id_valid, pdb_id_valid, smiles_valid, offtarget_complex_id
            )
            attempt_log.extend(method_log)
            
            if not ligand_path:
                raise ValueError(f"Failed to obtain ligand from any source")
            
            attempt_log.append(f"\n✓ {offtarget_complex_id}: Successfully saved ligand from {ligand_source}")
            self.status_tracker.mark_status(offtarget_complex_id, '2_save_ligands', 'success')
            
            return True, {
                'ligand_path': ligand_path,
                'ligand_source': ligand_source,
                'offtarget_complex_id': offtarget_complex_id
            }, attempt_log
            
        except Exception as e:
            attempt_log.append(f"\n✗ {offtarget_complex_id}: FAILED - {str(e)}")
            logger.error(f"{offtarget_complex_id}: {e}")
            self.status_tracker.mark_status(offtarget_complex_id, '2_save_ligands', 'failed', str(e))
            return False, {'offtarget_complex_id': offtarget_complex_id}, attempt_log


def main():
    p = argparse.ArgumentParser(description="Stage 2: Save off-target ligand files (run after Stage 1)")
    p.add_argument("--csv", required=True, help="Input CSV (stage1_output.csv)")
    p.add_argument("--base_dir", required=True, help="Base output directory")
    p.add_argument("--status_dir", required=True, help="Status tracking directory")
    p.add_argument("--verbose", action='store_true', help="Print detailed attempt log")
    p.add_argument("--max_heavy_atoms", type=int, default=None,
                   help="Reject SMILES-based ligands with more than this many heavy atoms. "
                        "Default: no limit.")
    args = p.parse_args()
    
    # Initialize
    status_tracker = StatusTracker(Path(args.status_dir))
    ligand_saver = LigandSaver(args.base_dir, status_tracker, max_heavy_atoms=args.max_heavy_atoms)
    if args.max_heavy_atoms is not None:
        logger.info(f"Heavy-atom size filter: skipping SMILES generation for molecules > {args.max_heavy_atoms} heavy atoms")
    else:
        logger.info("Heavy-atom size filter: disabled (no limit)")
    
    # Load CSV
    df = pd.read_csv(args.csv)
    logger.info(f"Loaded {len(df)} total entries")
    
    # Filter off-target rows only (stage2 saves off-target ligands)
    off_target_df = df[df['target_type'] != 'target'].copy()
    target_df = df[df['target_type'] == 'target'].copy()
    
    logger.info(f"Processing {len(off_target_df)} off-target entries")
    logger.info(f"Passing through {len(target_df)} target entries unchanged")
    
    # Process off-target rows
    updated_off_rows = []
    success_count = 0
    failed_count = 0
    all_logs = []
    
    for idx, row in off_target_df.iterrows():
        offtarget_complex_id = row['offtarget_complex_id']
        uniprot_id = str(row['uniprot_id']).strip()
        if not uniprot_id or uniprot_id.upper() == 'NAN':
            uniprot_id = 'UNKNOWN'
        
        success, updated_data, attempt_log = ligand_saver.process_row(row, offtarget_complex_id, uniprot_id)
        
        # Print attempt log
        for log_line in attempt_log:
            if args.verbose or ('SUCCESS' in log_line or 'FAILED' in log_line or '✗' in log_line):
                logger.info(log_line)
        all_logs.extend(attempt_log)
        
        if success:
            success_count += 1
            row_dict = row.to_dict()
            row_dict.update(updated_data)
            row_dict['ligand_source'] = updated_data['ligand_source']
            updated_off_rows.append(row_dict)
        else:
            failed_count += 1
    
    # Combine: target rows (unchanged) + processed off-target rows
    target_rows = [row.to_dict() for _, row in target_df.iterrows()]
    all_rows = target_rows + updated_off_rows
    
    # Save updated CSV
    output_csv = Path(args.base_dir) / "stage2_output.csv"
    pd.DataFrame(all_rows).to_csv(output_csv, index=False)
    logger.info(f"Saved {len(target_rows)} target entries (pass-through)")
    logger.info(f"Saved {len(updated_off_rows)} off-target entries (processed)")
    
    # Save detailed log
    log_file = Path(args.base_dir) / "stage2_detailed.log"
    with open(log_file, 'w') as f:
        for log_line in all_logs:
            f.write(log_line + '\n')
    
    # Save SMILES error statistics
    stats_file = Path(args.base_dir) / "stage2_smiles_errors.log"
    with open(stats_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SMILES GENERATION ERROR STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        
        if ligand_saver.smiles_error_stats:
            f.write("Error Type Summary:\n")
            f.write("-" * 70 + "\n")
            total_smiles_errors = sum(ligand_saver.smiles_error_stats.values())
            for error_type, count in sorted(ligand_saver.smiles_error_stats.items(), 
                                           key=lambda x: x[1], reverse=True):
                percentage = 100 * count / total_smiles_errors
                f.write(f"{error_type:30s}: {count:5d} ({percentage:5.1f}%)\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("Example Cases for Each Error Type (max 10 per type):\n")
            f.write("=" * 70 + "\n\n")
            
            for error_type, examples in ligand_saver.smiles_error_examples.items():
                f.write(f"\n{error_type.upper()}:\n")
                f.write("-" * 70 + "\n")
                for i, ex in enumerate(examples, 1):
                    f.write(f"  {i}. Complex: {ex['complex_id']}\n")
                    f.write(f"     PDB: {ex['pdb_id']}, HET: {ex['het_id']}\n")
                    f.write(f"     SMILES: {ex['smiles']}\n\n")
        else:
            f.write("No SMILES errors encountered.\n")
    
    # Save too-large ligands to a dedicated CSV for inspection
    too_large_csv = Path(args.base_dir) / "stage2_too_large.csv"
    if ligand_saver.too_large_ligands:
        pd.DataFrame(ligand_saver.too_large_ligands).to_csv(too_large_csv, index=False)
        logger.info(f"Saved {len(ligand_saver.too_large_ligands)} size-rejected ligands to {too_large_csv}")
    else:
        logger.info("No ligands rejected by size filter.")

    # Print summary
    print(f"\n{'='*70}")
    print(f"=== STAGE 2 SUMMARY ===")
    print(f"{'='*70}")
    print(f"Off-target entries:  {len(off_target_df)}")
    print(f"  Skipped (exists):  {ligand_saver.n_skipped_existing}")
    print(f"  Success (new):     {success_count - ligand_saver.n_skipped_existing}")
    print(f"  Total success:     {success_count}")
    print(f"  Rejected (size):   {len(ligand_saver.too_large_ligands)}")
    print(f"  Failed:            {failed_count}")
    print(f"  Success rate:      {100*success_count/len(off_target_df) if len(off_target_df) > 0 else 0:.1f}%")
    print(f"Target entries:      {len(target_df)} (passed through)")
    print(f"Total output rows:   {len(all_rows)}")
    
    if ligand_saver.smiles_error_stats:
        print(f"\nSMILES Generation Errors:")
        total_smiles_errors = sum(ligand_saver.smiles_error_stats.values())
        for error_type, count in sorted(ligand_saver.smiles_error_stats.items(), 
                                       key=lambda x: x[1], reverse=True):
            percentage = 100 * count / total_smiles_errors
            print(f"  {error_type:25s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"{'='*70}")
    print(f"Output CSV:          {output_csv}")
    print(f"Detailed log:        {log_file}")
    print(f"SMILES errors:       {stats_file}")
    if ligand_saver.too_large_ligands:
        print(f"Too-large ligands:   {too_large_csv}  ({len(ligand_saver.too_large_ligands)} entries)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()