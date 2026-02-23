"""
Stage 2 (SAIR variant): Extract off-target ligands from SAIR CIF structures.

SAIR-specific processing:
1. Read CIF files from SAIR structures directory
2. Extract ligand (Chain B, LIG) directly
3. Save ligand as PDB and convert to SDF

Note: Off-target protein structures are not used (only ligands).
No need for standardization - just extract ligand from Chain B.

Output structure:
{base_dir}/
└── {uniprot_id}/
    └── offtargets/
        └── from_sair/
            ├── {offtarget_complex_id}.pdb
            └── {offtarget_complex_id}.sdf
"""

import argparse
import sys
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, Optional

from Bio.PDB import MMCIFParser, PDBIO, Select
from rdkit import Chem
from rdkit.Chem import AllChem
from utils.status_tracker import StatusTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Stage name for status tracking
STAGE_NAME = '2_save_sair'

# SAIR structures directory
SAIR_STRUCTURES_DIR = Path("/proj/home/ibs/aipd_lab/yunmin_kim/data/sair_structures/structures")


class LigandSelect(Select):
    """Select only ligand residues (HETATM)"""
    def __init__(self, ligand_chain='B', ligand_resname='LIG'):
        self.ligand_chain = ligand_chain
        self.ligand_resname = ligand_resname
    
    def accept_chain(self, chain):
        return chain.get_id() == self.ligand_chain
    
    def accept_residue(self, residue):
        # Accept HETATM with matching resname
        if residue.get_id()[0].startswith('H_'):  # HETATM
            resname = residue.get_resname()
            if resname == self.ligand_resname:
                return True
        return False


class SAIRLigandExtractor:
    def __init__(self, base_dir: str, status_tracker: StatusTracker):
        self.base_dir = Path(base_dir)
        self.status_tracker = status_tracker
        self.cif_parser = MMCIFParser(QUIET=True)
        self.pdb_io = PDBIO()
        
        # Statistics
        self.n_skipped_existing = 0
        self.extraction_log = []
        
        if not SAIR_STRUCTURES_DIR.exists():
            logger.error(f"SAIR structures directory not found: {SAIR_STRUCTURES_DIR}")
            raise FileNotFoundError(f"SAIR directory missing: {SAIR_STRUCTURES_DIR}")
    
    def extract_ligand_from_cif(self, cif_path: Path, output_pdb: Path, 
                                ligand_chain: str = 'B', ligand_resname: str = 'LIG') -> bool:
        """
        Extract ligand from SAIR CIF file and save as PDB.
        
        Args:
            cif_path: Input CIF file
            output_pdb: Output ligand PDB file
            ligand_chain: Chain ID containing ligand (default: 'B')
            ligand_resname: Residue name of ligand (default: 'LIG')
            
        Returns:
            True if successful
        """
        try:
            structure = self.cif_parser.get_structure('structure', str(cif_path))
            
            # Extract ligand
            self.pdb_io.set_structure(structure)
            output_pdb.parent.mkdir(parents=True, exist_ok=True)
            
            selector = LigandSelect(ligand_chain, ligand_resname)
            self.pdb_io.save(str(output_pdb), selector)
            
            # Verify ligand was extracted
            if output_pdb.stat().st_size == 0:
                logger.warning(f"No ligand found in {cif_path} (chain={ligand_chain}, resname={ligand_resname})")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Ligand extraction failed for {cif_path}: {e}")
            return False
    
    def pdb_to_sdf(self, pdb_path: Path, sdf_path: Path) -> bool:
        """
        Convert ligand PDB to SDF format using RDKit.
        
        Args:
            pdb_path: Input ligand PDB file
            sdf_path: Output SDF file
            
        Returns:
            True if successful
        """
        try:
            # Read PDB
            mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
            if mol is None:
                logger.error(f"Failed to parse PDB: {pdb_path}")
                return False
            
            # Assign bond orders and add hydrogens if needed
            mol = Chem.AddHs(mol, addCoords=True)
            
            # Write SDF
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(mol)
            writer.close()
            
            return True
        except Exception as e:
            logger.error(f"PDB to SDF conversion failed for {pdb_path}: {e}")
            return False
    
    def process_row(self, row: pd.Series, offtarget_complex_id: str, uniprot_id: str) -> Tuple[bool, Dict]:
        """
        Process a single SAIR off-target row.
        
        Args:
            row: Input CSV row
            offtarget_complex_id: Pre-assigned complex ID (e.g., "P12345_off_000001")
            uniprot_id: UniProt ID
            
        Returns:
            (success, updated_data_dict)
        """
        # Check if already processed
        if self.status_tracker.check_status(offtarget_complex_id, STAGE_NAME) == 'success':
            logger.info(f"[SKIP] {offtarget_complex_id} - already processed")
            self.n_skipped_existing += 1

            # Return existing paths
            existing_pdb = row.get('ligand_path', '')
            existing_sdf = row.get('ligand_path_sdf', '')
            existing_source = row.get('ligand_source', 'sair')

            if existing_pdb and existing_sdf:
                return True, {
                    'ligand_path': existing_pdb,
                    'ligand_path_sdf': existing_sdf,
                    'ligand_source': existing_source
                }

        # Define output paths
        ligand_dir = self.base_dir / uniprot_id / "offtargets" / "from_sair"
        ligand_dir.mkdir(parents=True, exist_ok=True)

        ligand_pdb = ligand_dir / f"{offtarget_complex_id}.pdb"
        ligand_sdf = ligand_dir / f"{offtarget_complex_id}.sdf"

        try:
            # Use pre-computed ligand PDB from stage1 if available
            cif_path = None
            precomputed_pdb = row.get('ligand_path', '')
            if precomputed_pdb and Path(str(precomputed_pdb)).exists():
                logger.info(f"[SAIR] Using pre-computed ligand from stage1: {offtarget_complex_id}")
                ligand_pdb = Path(str(precomputed_pdb))
            else:
                # Fall back to extracting from CIF
                valid_pdb_id = row.get('valid_pdb_id', '')
                if not valid_pdb_id or pd.isna(valid_pdb_id) or str(valid_pdb_id).strip().upper() == 'NAN':
                    logger.error(f"[SAIR] Missing valid_pdb_id for {offtarget_complex_id}")
                    self.status_tracker.mark_status(offtarget_complex_id, STAGE_NAME, 'failed', 'missing_pdb_id')
                    return False, {}

                valid_pdb_id = str(valid_pdb_id).strip()

                # Find SAIR CIF file
                cif_filename = f"{valid_pdb_id}.cif"
                cif_path = SAIR_STRUCTURES_DIR / cif_filename

                if not cif_path.exists():
                    logger.error(f"SAIR CIF not found: {cif_path}")
                    self.status_tracker.mark_status(offtarget_complex_id, STAGE_NAME, 'failed', 'cif_not_found')
                    return False, {}

                # Extract ligand from CIF (Chain B, LIG)
                logger.info(f"[SAIR] Extracting ligand from CIF: {offtarget_complex_id}")
                if not self.extract_ligand_from_cif(cif_path, ligand_pdb, ligand_chain='B', ligand_resname='LIG'):
                    self.status_tracker.mark_status(offtarget_complex_id, STAGE_NAME, 'failed', 'ligand_extraction_failed')
                    return False, {}

            # Convert to SDF
            logger.info(f"[SAIR] Converting to SDF: {offtarget_complex_id}")
            if not self.pdb_to_sdf(ligand_pdb, ligand_sdf):
                logger.warning(f"SDF conversion failed for {offtarget_complex_id}, but PDB exists")
                # Don't fail completely - PDB is still usable
            
            # Log extraction
            self.extraction_log.append({
                'offtarget_complex_id': offtarget_complex_id,
                'uniprot_id': uniprot_id,
                'cif_path': str(cif_path) if cif_path else 'precomputed',
                'ligand_chain': 'B',
                'ligand_resname': 'LIG',
                'output_pdb': str(ligand_pdb),
                'output_sdf': str(ligand_sdf)
            })
            
            # Mark success
            self.status_tracker.mark_status(offtarget_complex_id, STAGE_NAME, 'success', 'sair_ligand_extracted')
            
            logger.info(f"[SUCCESS] {offtarget_complex_id}")
            return True, {
                'ligand_path': str(ligand_pdb),
                'ligand_path_sdf': str(ligand_sdf),
                'ligand_source': 'sair'
            }
            
        except Exception as e:
            logger.error(f"[FAILED] {offtarget_complex_id}: {e}")
            self.status_tracker.mark_status(offtarget_complex_id, STAGE_NAME, 'failed', str(e)[:200])
            return False, {}


def main():
    p = argparse.ArgumentParser(description="Stage 2 (SAIR): Extract ligands from SAIR structures")
    p.add_argument("--csv", required=True, help="Input CSV from stage1_sair")
    p.add_argument("--base_dir", required=True, help="Base output directory")
    p.add_argument("--status_dir", required=True, help="Status tracking directory")
    args = p.parse_args()
    
    # Initialize
    status_tracker = StatusTracker(Path(args.status_dir))
    extractor = SAIRLigandExtractor(args.base_dir, status_tracker)
    
    # Load CSV
    df = pd.read_csv(args.csv)
    logger.info(f"Loaded {len(df)} total entries")
    
    # Validate required columns
    required_cols = ['uniprot_id', 'target_type', 'offtarget_complex_id', 'data_source']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Required columns missing: {missing_cols}")
        sys.exit(1)
    
    # Filter SAIR off-target rows only
    # Note: target_type uses underscore ('off_target'), not hyphen
    df_sair_offtarget = df[(df['data_source'] == 'SAIR') & (df['target_type'] == 'off_target')].copy()
    df_passthrough = df[(df['data_source'] == 'SAIR') & (df['target_type'] == 'target')].copy()
    
    logger.info(f"Found {len(df_sair_offtarget)} SAIR off-target entries to process")
    logger.info(f"Found {len(df_passthrough)} SAIR target entries to pass through")
    
    if len(df_sair_offtarget) == 0:
        logger.warning("No SAIR off-target entries found")
        # Pass through all rows
        output_csv = Path(args.base_dir) / "stage2_sair_output.csv"
        df.to_csv(output_csv, index=False)
        logger.info(f"Output saved to: {output_csv}")
        return
    
    # Print statistics
    n_proteins = df_sair_offtarget['uniprot_id'].nunique()
    print(f"\n{'='*70}")
    print(f"=== STAGE 2 (SAIR) INPUT ANALYSIS ===")
    print(f"{'='*70}")
    print(f"Unique SAIR proteins:      {n_proteins}")
    print(f"SAIR off-target entries:   {len(df_sair_offtarget)}")
    print(f"SAIR target entries:       {len(df_passthrough)} (pass-through)")
    print(f"{'='*70}\n")
    
    # Process off-target rows
    updated_offtarget_rows = []
    success_count = 0
    failed_count = 0
    
    for idx, row in df_sair_offtarget.iterrows():
        offtarget_complex_id = row['offtarget_complex_id']
        uniprot_id = str(row['uniprot_id']).strip()
        if uniprot_id.upper() == 'NAN' or not uniprot_id:
            logger.warning(f"Missing uniprot_id for row {idx}, skipping")
            failed_count += 1
            continue
        
        success, updated_data = extractor.process_row(row, offtarget_complex_id, uniprot_id)
        
        if success:
            row_dict = row.to_dict()
            row_dict.update(updated_data)
            updated_offtarget_rows.append(row_dict)
            success_count += 1
        else:
            failed_count += 1
    
    # Pass through target rows
    target_rows = [row.to_dict() for _, row in df_passthrough.iterrows()]
    
    # Combine and save
    all_rows = updated_offtarget_rows + target_rows
    output_csv = Path(args.base_dir) / "stage2_sair_output.csv"
    
    if all_rows:
        pd.DataFrame(all_rows).to_csv(output_csv, index=False)
        logger.info(f"Saved {len(updated_offtarget_rows)} processed SAIR off-target entries")
        logger.info(f"Saved {len(target_rows)} pass-through SAIR target entries")
    else:
        logger.error("No entries to save!")
        pd.DataFrame(columns=['offtarget_complex_id', 'ligand_path', 'ligand_path_sdf', 'ligand_source']).to_csv(output_csv, index=False)
    
    # Save extraction log
    if extractor.extraction_log:
        log_csv = Path(args.base_dir) / "stage2_sair_extraction_log.csv"
        log_df = pd.DataFrame(extractor.extraction_log)
        log_df.to_csv(log_csv, index=False)
        logger.info(f"Extraction log saved: {log_csv}")
    
    # Summary
    print(f"{'='*70}")
    print(f"=== STAGE 2 (SAIR) SUMMARY ===")
    print(f"{'='*70}")
    print(f"Unique proteins:         {n_proteins}")
    print(f"SAIR off-target entries: {len(df_sair_offtarget)}")
    print(f"  Skipped (exists):      {extractor.n_skipped_existing}")
    print(f"  Success (new):         {success_count - extractor.n_skipped_existing}")
    print(f"  Total success:         {success_count}")
    print(f"  Failed:                {failed_count}")
    print(f"  Success rate:          {100*success_count/len(df_sair_offtarget) if len(df_sair_offtarget) > 0 else 0:.1f}%")
    print(f"Target entries:          {len(target_rows)} (passed through)")
    print(f"Total output rows:       {len(all_rows)}")
    print(f"Output CSV:              {output_csv}")
    if extractor.extraction_log:
        log_csv = Path(args.base_dir) / "stage2_sair_extraction_log.csv"
        print(f"Extraction log:          {log_csv}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
