import pandas as pd
import requests
import os
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
from typing import Optional, Tuple
import time
import shutil
from glob import glob


# Setup logging
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

class OffTargetLigandDownloader:
    def __init__(self, output_dir: str, csv_path: str):
        self.output_dir = Path(output_dir)
        self.csv_path = csv_path
        
        # Create subdirectories
        self.from_pdb_dir = self.output_dir / "from_pdb"
        self.from_pubchem_dir = self.output_dir / "from_pubchem"
        self.from_smiles_dir = self.output_dir / "from_smiles"
        
        for dir_path in [self.from_pdb_dir, self.from_pubchem_dir, self.from_smiles_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.parser = PDBParser(QUIET=True)
        self.io = PDBIO()
        
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
            logger.warning(f"Failed to download PDB {pdb_id}: {e}")
            return None
    
    def extract_ligand_from_pdb(self, pdb_file: Path, het_id: str, pdb_id: str) -> Optional[Path]:
        """Extract ligand from PDB file"""
        try:
            structure = self.parser.get_structure('complex', pdb_file)
            
            # Find ligand residue
            ligand_found = False
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.resname == het_id and residue.id[0].startswith('H'):  # HETATM
                            ligand_chain = chain.id
                            ligand_resid = residue.id[1]
                            ligand_found = True
                            break
                    if ligand_found:
                        break
                if ligand_found:
                    break
            
            if not ligand_found:
                logger.warning(f"Ligand {het_id} not found in PDB {pdb_id}")
                return None
            
            # Extract ligand
            output_file = self.from_pdb_dir / f"{pdb_id}_{het_id}.pdb"
            self.io.set_structure(structure)
            self.io.save(str(output_file), LigandExtractor(ligand_chain, ligand_resid, het_id))
            
            logger.info(f"Extracted ligand {het_id} from PDB {pdb_id}")
            return output_file
            
        except Exception as e:
            logger.warning(f"Failed to extract ligand from {pdb_id}: {e}")
            return None
        finally:
            # Clean up temporary file
            if pdb_file.exists():
                pdb_file.unlink()
    
    def download_from_pubchem(self, het_id: str) -> Optional[Path]:
        """Download ligand from PubChem by name/HET ID"""
        output_file = self.from_pubchem_dir / f"{het_id}_pubchem.sdf"
        
        # Try direct download first
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{het_id}/SDF"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            # Validate SDF
            mol = Chem.SDMolSupplier(str(output_file), removeHs=False)[0]
            if mol is None:
                logger.warning(f"Invalid SDF from PubChem for {het_id}")
                output_file.unlink()
                return None
                
            logger.info(f"Downloaded {het_id} from PubChem")
            return output_file
            
        except Exception as e:
            logger.warning(f"Failed to download from PubChem for {het_id}: {e}")
            if output_file.exists():
                output_file.unlink()
            return None
    
    def generate_from_smiles(self, smiles: str, het_id: str) -> Optional[Path]:
        """Generate 3D structure from SMILES"""
        output_file = self.from_smiles_dir / f"{het_id}_generated.sdf"
        
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES for {het_id}: {smiles}")
                return None
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result != 0:
                logger.warning(f"Failed to embed molecule for {het_id}")
                return None
            
            # Optimize geometry with UFF
            AllChem.UFFOptimizeMolecule(mol)
            
            # Write SDF
            writer = Chem.SDWriter(str(output_file))
            writer.write(mol)
            writer.close()
            
            logger.info(f"Generated 3D structure from SMILES for {het_id}")
            return output_file
            
        except Exception as e:
            logger.warning(f"Failed to generate from SMILES for {het_id}: {e}")
            if output_file.exists():
                output_file.unlink()
            return None
    
    def get_ligand(self, row: pd.Series) -> Tuple[Optional[str], str]:
        """
        Get ligand file path with fallback priority
        Returns: (file_path, source)
        """
        het_id = str(row['ligand_het_id']).strip()
        pdb_id = row.get('valid_pdb_id')
        smiles = row.get('ligand_smiles')

        # Priority 0: Check pre-existing directory
        pre_existing_dirs = [
            Path("/scratch/yunmin/data/graph/lmpnn/bdb_pdb/Kd/off_inputs/ligand_sdf"),
            Path("/scratch/yunmin/data/graph/lmpnn/bdb_pdb/IC50/off_inputs/ligand_sdf"),
        ]
        for pre_dir in pre_existing_dirs:
            for ext in [".sdf", ".pdb", ".mol"]:
                pattern = str(pre_dir / "*" / f"{het_id}{ext}")
                for candidate_path in glob(pattern):
                    candidate = Path(candidate_path)
                    if candidate.exists():
                        dest = self.output_dir / candidate.name
                        shutil.copy(candidate, dest)
                        logger.info(f"Found pre-existing ligand for {het_id} at {candidate}, copied to {dest}")
                        return str(dest), candidate.parent.name
            
        # Priority 1: Extract from PDB
        if pd.notna(pdb_id):
            pdb_file = self.download_pdb(pdb_id)
            if pdb_file:
                ligand_file = self.extract_ligand_from_pdb(pdb_file, het_id, pdb_id)
                if ligand_file:
                    return str(ligand_file), "from_pdb"
        
        # Priority 2: Download from PubChem
        ligand_file = self.download_from_pubchem(het_id)
        if ligand_file:
            return str(ligand_file), "from_pubchem"
        
        # Priority 3: Generate from SMILES
        if pd.notna(smiles):
            ligand_file = self.generate_from_smiles(smiles, het_id)
            if ligand_file:
                return str(ligand_file), "from_smiles"
        
        return None, "failed"
    
    def process_csv(self):
        """Process entire CSV and add ligand paths"""
        logger.info(f"Reading CSV from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        
        logger.info(f"Processing {len(df)} entries")
        
        results = []
        sources = []
        
        for idx, row in df.iterrows():
            if idx % 10 == 0:
                logger.info(f"Progress: {idx}/{len(df)}")
            
            # Only process off-target rows
            if row.get('target_type') != 'off_target':
                results.append(None)
                sources.append("not_off_target")
                continue

            ligand_path, source = self.get_ligand(row)
            results.append(ligand_path)
            sources.append(source)
            
            # Rate limiting for API calls
            time.sleep(0.5)
        
        # Add columns to dataframe
        df['offtarget_ligand_path'] = results
        df['offtarget_ligand_source'] = sources
        
        # Save updated CSV
        orig_name = Path(self.csv_path).stem
        output_csv = self.output_dir / f"{orig_name}_with_ligand_paths.csv"
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved updated CSV to {output_csv}")
        
        # Print statistics
        source_counts = df['offtarget_ligand_source'].value_counts()
        logger.info("\n=== Download Statistics ===")
        for source, count in source_counts.items():
            logger.info(f"{source}: {count} ({count/len(df)*100:.1f}%)")
        
        success_rate = (len(df[df['offtarget_ligand_source'] != 'failed']) / len(df[df['target_type'] == 'off_target'])) * 100
        logger.info(f"Overall success rate: {success_rate:.1f}%")

if __name__ == "__main__":
    # CSV_PATH = f"/scratch/yunmin/data/graph/lmpnn/bdb_pdb/off_inputs/counts/bindingdb_count_set3.csv"
    OUTPUT_DIR = f"/scratch/yunmin/data/graph/lmpnn/bdb_pdb/off_inputs/ligand_sdf"
    CSV_PATH = "/scratch/yunmin/data/graph/train/identity50/csvs/06_sample_example/train_set_sampled_20targets_100offtargets.csv"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    downloader = OffTargetLigandDownloader(OUTPUT_DIR, CSV_PATH)
    downloader.process_csv()