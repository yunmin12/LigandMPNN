"""
Stage 1: Save target and off-target ligand files
Based on save_ligand.py
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
from typing import Optional, Tuple, Tuple as TupleTy
from utils.status_tracker import StatusTracker

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
    def __init__(self, output_dir: str, status_tracker: StatusTracker):
        self.output_dir = Path(output_dir)
        self.status_tracker = status_tracker
        
        # Pre-existing ligand directories to check first
        self.pre_existing_dirs = [
            Path("/scratch/yunmin/data/graph/lmpnn/bdb_pdb/off_inputs/ligand_sdf")
        ]
        
        # Create subdirectories
        self.from_pdb_dir = self.output_dir / "from_pdb"
        self.from_pubchem_dir = self.output_dir / "from_pubchem"
        self.from_smiles_dir = self.output_dir / "from_smiles"
        
        for dir_path in [self.from_pdb_dir, self.from_pubchem_dir, self.from_smiles_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.parser = PDBParser(QUIET=True)
        self.io = PDBIO()
    
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
    
    def generate_from_smiles(self, smiles: str, het_id: str) -> Optional[Path]:
        """Generate 3D structure from SMILES"""
        output_file = self.from_smiles_dir / f"{het_id}_generated.sdf"
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Invalid SMILES for {het_id}")
                return None
            
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
                logger.error(f"Failed to embed molecule for {het_id}")
                return None
            
            AllChem.MMFFOptimizeMolecule(mol)
            
            writer = Chem.SDWriter(str(output_file))
            writer.write(mol)
            writer.close()
            
            logger.info(f"Generated 3D structure for {het_id} from SMILES")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to generate from SMILES: {e}")
            return None
    
    def get_ligand(self, het_id: Optional[str] = None, pdb_id: Optional[str] = None, 
                   smiles: Optional[str] = None) -> Tuple[Optional[str], str, list]:
        """
        Get ligand file path with fallback priority
        Returns: (file_path, source, attempt_log)
        
        Priority order:
        1. Check pre-existing directories (if het_id available)
        2. Extract from PDB (if pdb_id available)
        3. Download from PubChem (if het_id available)
        4. Generate from SMILES (if smiles available)
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
            ligand_file = self.generate_from_smiles(smiles, het_id or "generated")
            if ligand_file:
                attempt_log.append(f"   ✓ SUCCESS: Generated from SMILES")
                return str(ligand_file), 'smiles', attempt_log
            attempt_log.append(f"   ✗ Failed to generate from SMILES")
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
    
    def process_row(self, row: pd.Series, complex_id: str) -> Tuple[bool, dict, list]:
        """
        Process one row: save target and off-target ligands
        
        CSV columns expected:
        - ligand_het_id: HET ID for the ligand (optional)
        - valid_pdb_id: PDB ID to extract from (optional)
        - ligand_smiles: SMILES string (optional)
        
        Returns:
            (success, updated_data_dict, attempt_log)
        """
        attempt_log = []
        
        try:
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
            attempt_log.append(f"Processing {complex_id}")
            attempt_log.append(f"  HET ID: {het_id if het_id_valid else '(none)'}")
            attempt_log.append(f"  PDB ID: {pdb_id if pdb_id_valid else '(none)'}")
            attempt_log.append(f"  SMILES: {smiles if smiles_valid else '(none)'}")
            attempt_log.append(f"{'='*70}")
            
            # Get ligand with priority order
            ligand_path, ligand_source, method_log = self.get_ligand(
                het_id_valid, pdb_id_valid, smiles_valid
            )
            attempt_log.extend(method_log)
            
            if not ligand_path:
                raise ValueError(f"Failed to obtain ligand from any source")
            
            attempt_log.append(f"\n✓ {complex_id}: Successfully saved ligand from {ligand_source}")
            self.status_tracker.mark_status(complex_id, '1_save_ligands', 'success')
            
            return True, {
                'ligand_path': ligand_path,
                'ligand_source': ligand_source,
                'complex_id': complex_id
            }, attempt_log
            
        except Exception as e:
            attempt_log.append(f"\n✗ {complex_id}: FAILED - {str(e)}")
            logger.error(f"{complex_id}: {e}")
            self.status_tracker.mark_status(complex_id, '1_save_ligands', 'failed', str(e))
            return False, {'complex_id': complex_id}, attempt_log


def main():
    p = argparse.ArgumentParser(description="Stage 1: Save ligand files")
    p.add_argument("--csv", required=True, help="Input CSV with ligand information")
    p.add_argument("--output_dir", required=True, help="Output directory for ligands")
    p.add_argument("--status_dir", required=True, help="Status tracking directory")
    p.add_argument("--verbose", action='store_true', help="Print detailed attempt log")
    args = p.parse_args()
    
    # Initialize
    status_tracker = StatusTracker(Path(args.status_dir))
    ligand_saver = LigandSaver(
        args.output_dir, 
        status_tracker
    )
    
    # Load CSV
    df = pd.read_csv(args.csv)
    logger.info(f"Processing {len(df)} entries")
    
    # Process each row
    updated_rows = []
    success_count = 0
    failed_count = 0
    all_logs = []
    
    for idx, row in df.iterrows():
        complex_id = f"complex_{idx:06d}"
        
        success, updated_data, attempt_log = ligand_saver.process_row(row, complex_id)
        
        # Print attempt log
        for log_line in attempt_log:
            if args.verbose or ('SUCCESS' in log_line or 'FAILED' in log_line or '✗' in log_line):
                logger.info(log_line)
            all_logs.extend(attempt_log)
        
        if success:
            success_count += 1
            # Update row with new data
            row_dict = row.to_dict()
            row_dict.update(updated_data)
            row_dict['ligand_source'] = updated_data['ligand_source']
            updated_rows.append(row_dict)
        else:
            failed_count += 1
    
    # Save updated CSV (only successful rows)
    output_csv = Path(args.output_dir).parent / "stage1_output.csv"
    pd.DataFrame(updated_rows).to_csv(output_csv, index=False)
    logger.info(f"Saved {success_count} successful entries to {output_csv}")
    
    # Save detailed log
    log_file = Path(args.output_dir).parent / "stage1_detailed.log"
    with open(log_file, 'w') as f:
        for log_line in all_logs:
            f.write(log_line + '\n')
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"=== STAGE 1 SUMMARY ===")
    print(f"{'='*70}")
    print(f"Total processed:  {len(df)}")
    print(f"Success:          {success_count}")
    print(f"Failed:           {failed_count}")
    print(f"Success rate:     {100*success_count/len(df):.1f}%")
    print(f"{'='*70}")
    print(f"Output CSV:       {output_csv}")
    print(f"Detailed log:     {log_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()