#!/usr/bin/env python3
"""
Extract target ligands from PDB files and standardize to Chain Z, ResID 1
"""
import pandas as pd
import requests
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select, Structure, Model, Chain
import logging
from typing import Optional
import time
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

WATER_RESNAMES = {'HOH', 'WAT', 'H2O', 'DOD'}
METAL_ION_RESNAMES = {
    'NA','K','CL','CA','MG','ZN','FE','CU','MN','CO','NI',
    'CD','HG','PB','SR','CS','BA','TI','V','CR','MO','SE','W','RB','YB','AL'
}

class LigandExtractor(Select):
    """Extract specific ligand and renumber to chain Z, resid 1"""
    def __init__(self, target_chain: str, target_resid: int, target_resname: str):
        self.target_chain = target_chain
        self.target_resid = target_resid
        self.target_resname = target_resname
        
    def accept_residue(self, residue):
        return (residue.parent.id == self.target_chain and 
                residue.id[1] == self.target_resid and
                residue.resname == self.target_resname)

class TargetLigandExtractor:
    def __init__(self, output_dir: str, csv_path: str):
        self.output_dir = Path(output_dir)
        self.csv_path = csv_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.parser = PDBParser(QUIET=True)
        self.io = PDBIO()
        
        # Cache for already processed PDBs
        self.processed_pdbs = {}
        
        # Track failure reasons
        self.failure_reasons = Counter()
        
    def download_pdb(self, pdb_id: str) -> Optional[Path]:
        """Download PDB file from RCSB"""
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        temp_file = self.output_dir / f"{pdb_id}_temp.pdb"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(temp_file, 'w') as f:
                f.write(response.text)
            
            logger.debug(f"Downloaded PDB {pdb_id}")
            return temp_file
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.failure_reasons['pdb_not_found'] += 1
            else:
                self.failure_reasons['download_http_error'] += 1
            logger.warning(f"Failed to download PDB {pdb_id}: HTTP {e.response.status_code}")
            return None
        except Exception as e:
            self.failure_reasons['download_other_error'] += 1
            logger.warning(f"Failed to download PDB {pdb_id}: {e}")
            return None
    
    def find_ligand_in_pdb(self, pdb_file: Path, het_id: str):
        """Find ligand location in PDB"""
        try:
            structure = self.parser.get_structure('complex', pdb_file)
            
            # Collect all HET residues for debugging
            all_het_residues = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0].startswith('H'):  # HETATM
                            all_het_residues.append(residue.resname)
                            if residue.resname == het_id:
                                return chain.id, residue.id[1], residue.resname
            
            # Log what was found vs what was expected
            unique_hets = set(all_het_residues)
            logger.debug(f"Expected {het_id}, found HET residues: {unique_hets}")
            self.failure_reasons[f'ligand_not_found_in_pdb'] += 1
            
            return None, None, None
        except Exception as e:
            self.failure_reasons['pdb_parsing_error'] += 1
            logger.error(f"Error parsing PDB: {e}")
            return None, None, None
    
    def is_metal_ion(self, residue: Chain) -> bool:
        atoms = list(residue.get_atoms())
        if len(atoms) == 1:
            return False
        atom = atoms[0]
        return atom.element.strip().upper() in METAL_ION_RESNAMES

    def extract_and_standardize(self, pdb_file: Path, chain_id: str, 
                                resid: int, resname: str, pdb_id: str) -> Optional[Path]:
        """ 
        1. Find target ligand in pdb
        2. Renumber ligand to Chain Z, resnum 1
        3. Remove other HETATM/water
        4. Save standardized pdb
        """
        try:
            structure = self.parser.get_structure('complex', pdb_file)
            
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
                        resname = residue.resname
                        hetero_flag = residue.id[0]

                        # retain target ligand only (now in Chain Z, ResID 1)
                        if chain.id == 'Z' and residue.id[1] == 1:
                            continue
                        
                        # remove water molecules
                        if resname in WATER_RESNAMES:
                            chain.detach_child(residue.id)
                            continue
                        
                        # remove metal ions (by residue name)
                        if resname in METAL_ION_RESNAMES:
                            chain.detach_child(residue.id)
                            continue
                        
                        # remove all other HETATM records (hetero_flag is not blank)
                        if hetero_flag.strip():
                            chain.detach_child(residue.id)
                            continue
                
                ligand_found = True
                break

            if not ligand_found:
                self.failure_reasons['extraction_failed'] += 1
                logger.warning(f"Ligand {resname} not found at chain {chain_id}, resid {resid}")
                return None
            
            # Save standardized complex pdb
            output_file = self.output_dir / f"{pdb_id}_{resname}_std.pdb"
            self.io.set_structure(structure)
            self.io.save(str(output_file))
            
            logger.debug(f"Standardized ligand {resname} in {pdb_id} → Chain Z, ResID 1")
            return output_file
            
        except Exception as e:
            self.failure_reasons['extraction_exception'] += 1
            logger.error(f"Failed to extract/standardize ligand: {e}")
            return None
    
    def process_pdb(self, pdb_id: str, het_id: str) -> Optional[Path]:
        """Download PDB, find ligand, extract and standardize"""
        # Check cache
        cache_key = f"{pdb_id}_{het_id}"
        if cache_key in self.processed_pdbs:
            return self.processed_pdbs[cache_key]
        
        # Check if already exists
        output_file = self.output_dir / f"{pdb_id}_{het_id}_target.pdb"
        if output_file.exists():
            logger.debug(f"Using cached target ligand: {output_file}")
            self.processed_pdbs[cache_key] = output_file
            return output_file
        
        # Download PDB
        temp_pdb = self.download_pdb(pdb_id)
        if not temp_pdb:
            self.processed_pdbs[cache_key] = None
            return None
        
        # Find ligand in PDB
        chain_id, resid, resname = self.find_ligand_in_pdb(temp_pdb, het_id)
        if not chain_id:
            logger.debug(f"Could not find ligand {het_id} in PDB {pdb_id}")
            temp_pdb.unlink()
            self.processed_pdbs[cache_key] = None
            return None
        
        # Extract and standardize
        standardized_file = self.extract_and_standardize(
            temp_pdb, chain_id, resid, resname, pdb_id
        )
        
        # Clean up temp file
        if temp_pdb.exists():
            temp_pdb.unlink()
        
        # Cache result
        self.processed_pdbs[cache_key] = standardized_file
        return standardized_file
    
    def process_csv(self):
        """Process CSV and add target ligand paths"""
        logger.info(f"Reading CSV from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        
        # Check how many have valid_pdb_id
        valid_pdb_count = df[df['target_type'] == 'target']['valid_pdb_id'].notna().sum()
        logger.info(f"Total entries: {len(df)}")
        logger.info(f"Entries with valid_pdb_id: {valid_pdb_count} ({valid_pdb_count/len(df)*100:.1f}%)")
        
        target_ligand_paths = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Progress: {idx}/{len(df)}")
            
            # Only extract for target_type == "target"
            if row.get('target_type') != 'target':
                target_ligand_paths.append(None)
                continue

            pdb_id = row.get('valid_pdb_id')
            het_id = str(row['ligand_het_id']).strip()
            
            if pd.isna(pdb_id):
                self.failure_reasons['no_valid_pdb_id'] += 1
                target_ligand_paths.append(None)
                continue
            
            # Process PDB and extract target ligand
            target_file = self.process_pdb(pdb_id, het_id)
            target_ligand_paths.append(str(target_file.resolve()) if target_file else None)
            
            # Rate limiting
            time.sleep(0.3)
        
        # Add column to dataframe
        df['target_ligand_path'] = target_ligand_paths
        
        # Save updated CSV
        output_csv = Path(self.csv_path).parent / "bindingdb_with_all_ligand_paths.csv"
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved updated CSV to {output_csv}")
        
        # Statistics
        success_count = df['target_ligand_path'].notna().sum()
        logger.info(f"\n=== Target Ligand Extraction Statistics ===")
        logger.info(f"Successfully extracted: {success_count}/{len(df)} ({success_count/len(df)*100:.1f}%)")
        logger.info(f"Failed: {len(df) - success_count}")
        
        # Detailed failure breakdown
        logger.info(f"\n=== Failure Reasons ===")
        for reason, count in self.failure_reasons.most_common():
            percentage = count / len(df) * 100
            logger.info(f"{reason}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    # CSV_PATH = "/scratch/yunmin/data/graph/lmpnn/bdb_pdb/Kd/off_inputs/ligand_sdf/bindingdb_with_ligand_paths.csv"
    CSV_PATH = "/scratch/yunmin/data/graph/lmpnn/bdb_pdb/off_inputs/counts/bindingdb_count_set3.csv"
    OUTPUT_DIR = "/scratch/yunmin/data/graph/lmpnn/bdb_pdb/off_inputs/target_std"
    
    extractor = TargetLigandExtractor(OUTPUT_DIR, CSV_PATH)
    extractor.process_csv()