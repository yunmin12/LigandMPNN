"""
Validate PDB IDs from BindingDB filtered data and select the best structure per protein.

Steps:
1. Protein validation (UniProt ID match via SIFTS)
2. Ligand validation (HET ID match)
3. Mutation check (wild-type)
4. Best resolution
"""

import pandas as pd
import requests
import json
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import gzip
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️  Warning: RDKit not available. SMILES comparison will be limited.")

# Solvent/common molecules to exclude
SOLVENT_IDS = {
    "HOH","WAT","DOD",  # water
    "CL","NA","K","CA","MG","ZN","MN","CO","CU","NI","IOD",  # common ions
    "SO4","PO4",  # anions
    "GOL","EDO","PEG","PG4","MPD","TRS","MES","ACE","IPA","BME","FMT",  # common buffers/additives
    "SEP", "TPO"    # modified residues
}


class TeeLogger:
    """Logger that writes to both stdout and file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8', buffering=1)  # Line buffering
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


class PDBCache:
    """Cache system for PDB metadata."""
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir) / "metadata"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get(self, pdb_id):
        cache_file = self.cache_dir / f"{pdb_id.upper()}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        return None
    
    def save(self, pdb_id, data):
        cache_file = self.cache_dir / f"{pdb_id.upper()}.json"
        cache_file.write_text(json.dumps(data, indent=2))


class PDBValidator:
    """Validate and select best PDB structures."""
    def __init__(self, cache_dir, assay_type, verbose=True):
        self.cache = PDBCache(cache_dir)
        self.cif_dir = Path(cache_dir) / "cif"
        self.pdb_dir = Path(cache_dir) / f"{assay_type}_val_pdb"
        self.sifts_dir = Path(cache_dir) / "sifts"  # SIFTS cache
        self.cif_dir.mkdir(parents=True, exist_ok=True)
        self.pdb_dir.mkdir(parents=True, exist_ok=True)
        self.sifts_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Chem comp cache for InChI key lookups
        self.chem_comp_cache = {}
    
    def _smiles_to_inchikey(self, smiles):
        """Convert SMILES to InChIKey using RDKit."""
        if not RDKIT_AVAILABLE or not smiles:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                inchikey = Chem.inchi.MolToInchiKey(mol)
                return inchikey
        except Exception as e:
            if self.verbose:
                print(f"    Warning: Failed to convert SMILES to InChIKey: {e}")
        return None
    
    def _compare_smiles(self, smiles1, smiles2):
        """Compare two SMILES strings for chemical equivalence."""
        if not RDKIT_AVAILABLE or not smiles1 or not smiles2:
            return False
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            if mol1 and mol2:
                # Compare InChIKeys (most reliable)
                inchikey1 = Chem.inchi.MolToInchiKey(mol1)
                inchikey2 = Chem.inchi.MolToInchiKey(mol2)
                return inchikey1 == inchikey2
        except Exception:
            pass
        return False
    
    def _fetch_sifts_uniprot_mapping(self, pdb_id):
        """
        Fetch SIFTS UniProt mapping using PDBe REST API.
        Returns list of UniProt accessions for this PDB entry.
        """
        pdb_id_lower = pdb_id.lower()
        cache_file = self.sifts_dir / f"{pdb_id_lower}_sifts.json"
        
        # Check cache
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        
        try:
            url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id_lower}"
            response = requests.get(url, timeout=30)
            
            if response.status_code != 200:
                if self.verbose:
                    print(f"    SIFTS: No mapping available (HTTP {response.status_code})")
                cache_file.write_text(json.dumps([]))
                return []
            
            data = response.json()
            
            # Extract all UniProt accessions from the mapping
            uniprot_ids = set()
            
            for pdb_entry_data in data.get(pdb_id_lower, {}).values():
                # Each entry can have multiple UniProt mappings
                for uniprot_data in pdb_entry_data.get('UniProt', {}).values():
                    uniprot_acc = uniprot_data.get('identifier')
                    if uniprot_acc:
                        uniprot_ids.add(uniprot_acc)
            
            result = sorted(list(uniprot_ids))
            
            # Cache the result
            cache_file.write_text(json.dumps(result, indent=2))
            
            if self.verbose and result:
                print(f"    SIFTS: Found {len(result)} UniProt ID(s): {', '.join(result)}")
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"    SIFTS: Error fetching mapping: {e}")
            cache_file.write_text(json.dumps([]))
            return []
    
    def fetch_metadata(self, pdb_id):
        """Fetch and cache PDB metadata."""
        pdb_id = pdb_id.upper().strip()
        
        # Check cache first
        cached = self.cache.get(pdb_id)
        if cached:
            self.cache_hits += 1
            return cached
        
        self.cache_misses += 1
        if self.verbose:
            print(f"  Fetching metadata for {pdb_id}...")
        
        try:
            # Download and parse mmCIF
            cif_path = self._download_cif(pdb_id)
            metadata = self._parse_cif(cif_path, pdb_id)
            
            # Use SIFTS for UniProt mapping (primary method)
            sifts_uniprots = self._fetch_sifts_uniprot_mapping(pdb_id)
            
            if sifts_uniprots:
                # SIFTS is more reliable, use it as primary source
                metadata['uniprot_ids'] = sifts_uniprots
                metadata['uniprot_source'] = 'SIFTS'
            else:
                # Fallback to mmCIF if SIFTS unavailable
                metadata['uniprot_source'] = 'mmCIF'
                if not metadata['uniprot_ids']:
                    if self.verbose:
                        print(f"    ⚠️  No UniProt IDs found from SIFTS or mmCIF")
            
            # Fetch ligands using multiple methods
            metadata['ligands'] = self._fetch_ligands_comprehensive(pdb_id)
            
            # Cache the result
            self.cache.save(pdb_id, metadata)
            
            # Rate limiting (be polite to APIs)
            time.sleep(0.1)
            
            return metadata
            
        except Exception as e:
            print(f"  ⚠️  Error fetching {pdb_id}: {e}")
            return None
    
    def _download_cif(self, pdb_id):
        """Download mmCIF file from RCSB."""
        cif_path = self.cif_dir / f"{pdb_id}.cif"
        
        if cif_path.exists():
            return cif_path
        
        # Try gzipped first (smaller download)
        url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            # Decompress and save
            cif_path.write_bytes(gzip.decompress(response.content))
            return cif_path
        
        # Fallback to uncompressed
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        cif_path.write_bytes(response.content)
        return cif_path
    
    def _parse_cif(self, cif_path, pdb_id):
        """Extract metadata from mmCIF file."""
        try:
            mmcif_dict = MMCIF2Dict(str(cif_path))
        except Exception as e:
            raise ValueError(f"Failed to parse mmCIF for {pdb_id}: {e}")
        
        metadata = {
            "pdb_id": pdb_id,
            "uniprot_ids": [],
            "mutations": [],
            "resolution": None,
            "release_date": None
        }
        
        # Extract UniProt IDs from mmCIF (fallback method)
        if "_struct_ref.db_name" in mmcif_dict:
            for i, db_name in enumerate(mmcif_dict["_struct_ref.db_name"]):
                if db_name == "UNP":
                    uniprot_id = mmcif_dict["_struct_ref.pdbx_db_accession"][i]
                    metadata["uniprot_ids"].append(uniprot_id)
        
        # Extract mutations
        if "_pdbx_struct_mod_residue.label_comp_id" in mmcif_dict:
            mutations = []
            comp_ids = mmcif_dict["_pdbx_struct_mod_residue.label_comp_id"]
            seq_ids = mmcif_dict["_pdbx_struct_mod_residue.label_seq_id"]
            details = mmcif_dict.get("_pdbx_struct_mod_residue.details", [""] * len(comp_ids))
            
            for comp_id, seq_id, detail in zip(comp_ids, seq_ids, details):
                if "MUTATION" in detail.upper() or "ENGINEERED" in detail.upper():
                    mutations.append(f"{comp_id}{seq_id}")
            
            metadata["mutations"] = mutations
        
        # Extract resolution
        if "_refine.ls_d_res_high" in mmcif_dict:
            try:
                metadata["resolution"] = float(mmcif_dict["_refine.ls_d_res_high"][0])
            except (ValueError, IndexError):
                pass
        elif "_em_3d_reconstruction.resolution" in mmcif_dict:
            # For cryo-EM structures
            try:
                metadata["resolution"] = float(mmcif_dict["_em_3d_reconstruction.resolution"][0])
            except (ValueError, IndexError):
                pass
        
        # Extract release date
        if "_pdbx_database_status.recvd_initial_deposition_date" in mmcif_dict:
            try:
                metadata["release_date"] = mmcif_dict["_pdbx_database_status.recvd_initial_deposition_date"][0]
            except IndexError:
                pass
        
        return metadata
    
    def _rcsb_entry_summary(self, pdb_id, timeout=15):
        """Fetch RCSB entry summary JSON."""
        try:
            url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None
    
    def _rcsb_polymer_entity_summary(self, pdb_id, timeout=15):
        """Fetch RCSB polymer entity summary JSON."""
        try:
            url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None
    
    def _rcsb_chem_comp(self, chem_id, timeout=15):
        """Fetch RCSB chem_comp; cached."""
        if chem_id in self.chem_comp_cache:
            return self.chem_comp_cache[chem_id]
        
        try:
            url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{chem_id}"
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                self.chem_comp_cache[chem_id] = None
                return None
            data = r.json()
            self.chem_comp_cache[chem_id] = data
            return data
        except Exception:
            self.chem_comp_cache[chem_id] = None
            return None
    
    def _fetch_ligands_comprehensive(self, pdb_id):
        """
        Fetch ligands using multiple methods (similar to bindingdb_v2_2.py).
        Tries multiple RCSB API endpoints to find ligand HET IDs.
        """
        entry_summary = self._rcsb_entry_summary(pdb_id)
        polymer_entity_summary = self._rcsb_polymer_entity_summary(pdb_id)
        
        if not entry_summary and not polymer_entity_summary:
            return []
        
        comp_ids = set()
        
        # Method 1: rcsb_binding_affinity
        if entry_summary and entry_summary.get('rcsb_binding_affinity') is not None:
            comp_ids.update([c['comp_id'] for c in entry_summary.get('rcsb_binding_affinity', [])])
        
        # Method 2: nonpolymer_bound_components
        if entry_summary and entry_summary.get("nonpolymer_bound_components") is not None:
            comp_ids.update(entry_summary.get("nonpolymer_bound_components", []))
        
        # Method 3: entity_poly.rcsb_non_std_monomers
        if polymer_entity_summary and polymer_entity_summary.get("entity_poly") is not None:
            comp_ids.update(polymer_entity_summary.get("entity_poly", {}).get("rcsb_non_std_monomers", []))
        
        # Method 4: rcsb_polymer_entity_container_identifiers.chem_comp_nstd_monomers
        if polymer_entity_summary and polymer_entity_summary.get("rcsb_polymer_entity_container_identifiers") is not None:
            comp_ids.update(
                polymer_entity_summary.get("rcsb_polymer_entity_container_identifiers", {})
                .get("chem_comp_nstd_monomers", [])
            )
        
        # Filter out solvents and get InChI keys
        ligands = []
        for cid in comp_ids:
            if not cid or cid in SOLVENT_IDS or cid == "UNK":
                continue
            
            # Fetch chem_comp details
            cc = self._rcsb_chem_comp(cid)
            inchikey = None
            name = None
            
            if isinstance(cc, dict):
                # Try rcsb_chem_comp_descriptor
                inchikey = cc.get("rcsb_chem_comp_descriptor", {}).get("InChIKey")
                name = cc.get("chem_comp", {}).get("name")
                
                # Fallback: pdbx_chem_comp_descriptor
                if not inchikey:
                    for d in (cc.get("pdbx_chem_comp_descriptor") or []):
                        if (d.get("type") or "").lower() == "inchikey" and d.get("descriptor"):
                            inchikey = d["descriptor"]
                            break
            
            # Get SMILES too
            smiles = None
            if isinstance(cc, dict):
                smiles = cc.get("rcsb_chem_comp_descriptor", {}).get("SMILES")
                if not smiles:
                    for d in (cc.get("pdbx_chem_comp_descriptor") or []):
                        if (d.get("type") or "").lower() == "smiles" and d.get("descriptor"):
                            smiles = d["descriptor"]
                            break
            
            ligands.append({
                "het_id": cid,
                "name": name or "N/A",
                "inchikey": inchikey or "N/A",
                "smiles": smiles or "N/A"
            })
        
        return ligands
    
    def download_pdb_structure(self, pdb_id):
        """Download final PDB structure file."""
        pdb_id = pdb_id.upper().strip()
        pdb_path = self.pdb_dir / f"{pdb_id}.pdb"
        
        if pdb_path.exists():
            return pdb_path
        
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            pdb_path.write_bytes(response.content)
            return pdb_path
        except Exception as e:
            print(f"  ⚠️  Failed to download PDB {pdb_id}: {e}")
            return None
    
    def validate_protein(self, row, idx, total):
        """Validate PDB IDs for a single protein row."""
        protein_key = row['protein_key']
        pdb_str = str(row.get('complex_pdb_id', '')).strip()
        
        if not pdb_str or pdb_str == 'nan':
            return None, "No PDB IDs"
        
        # Parse PDB IDs (split by comma)
        pdb_ids = [p.strip().upper() for p in pdb_str.split(';') if p.strip()]
        
        if not pdb_ids:
            return None, "No valid PDB IDs"
        
        print(f"\n{'='*80}")
        print(f"Protein: {protein_key} ({idx}/{total})")
        print(f"Original PDBs: {','.join(pdb_ids)}")
        
        # STEP 1: Protein UniProt validation (using SIFTS)
        print(f"\n[STEP 1: UniProt Validation via SIFTS]")
        expected_uniprot = str(row.get('uniprot_id', '')).strip()
        print(f"  Expected UniProt: {expected_uniprot}")
        candidates_step1 = []
        
        for pdb_id in pdb_ids:
            metadata = self.fetch_metadata(pdb_id)
            if not metadata:
                print(f"  ✗ {pdb_id} - Failed to fetch metadata → DROPPED")
                continue
            
            uniprot_ids = metadata.get('uniprot_ids', [])
            source = metadata.get('uniprot_source', 'unknown')
            
            if expected_uniprot in uniprot_ids:
                candidates_step1.append(pdb_id)
                print(f"  ✓ {pdb_id} - UniProt match: {expected_uniprot} (source: {source})")
            else:
                found = ', '.join(uniprot_ids) if uniprot_ids else 'None'
                print(f"  ✗ {pdb_id} - UniProt mismatch: found {found}, expected {expected_uniprot} (source: {source}) → DROPPED")
        
        print(f"  Remaining: {','.join(candidates_step1) if candidates_step1 else 'None'}")
        
        if not candidates_step1:
            return None, "No UniProt match"
        
        # STEP 2: Ligand InChIKey/SMILES validation
        print(f"\n[STEP 2: Ligand Validation (InChIKey/SMILES)]")
        expected_inchikey = str(row.get('ligand_inchikey', '')).strip()
        expected_smiles = str(row.get('canonical_smiles', '')).strip()
        print(f"  Expected InChIKey: {expected_inchikey}")
        print(f"  Expected SMILES: {expected_smiles[:60]}..." if len(expected_smiles) > 60 else f"  Expected SMILES: {expected_smiles}")
        
        # Convert expected SMILES to InChIKey if RDKit available
        expected_inchikey_from_smiles = None
        if RDKIT_AVAILABLE and expected_smiles:
            expected_inchikey_from_smiles = self._smiles_to_inchikey(expected_smiles)
            if expected_inchikey_from_smiles:
                print(f"  InChIKey from SMILES: {expected_inchikey_from_smiles}")
        
        candidates_step2 = []
        
        for pdb_id in candidates_step1:
            metadata = self.fetch_metadata(pdb_id)
            ligands = metadata.get('ligands', [])
            
            # Show all ligands found in this PDB
            print(f"\n  {pdb_id} - Found {len(ligands)} ligand(s):")
            if ligands:
                for lig in ligands:
                    het_id = lig.get('het_id', 'N/A')
                    name = lig.get('name', 'N/A')
                    inchikey = lig.get('inchikey', 'N/A')
                    smiles = lig.get('smiles', 'N/A')
                    print(f"    • {het_id}: {name}")
                    if inchikey != 'N/A':
                        print(f"      InChIKey: {inchikey}")
                    if smiles != 'N/A' and len(smiles) < 100:
                        print(f"      SMILES: {smiles}")
            else:
                print(f"    (No ligands found)")
            
            # Check for ligand match using InChIKey or SMILES
            match_found = False
            match_method = None
            
            for lig in ligands:
                lig_inchikey = lig.get('inchikey', '')
                lig_smiles = lig.get('smiles', '')
                
                # Method 1: Direct InChIKey comparison
                if expected_inchikey and lig_inchikey and lig_inchikey != 'N/A':
                    if expected_inchikey == lig_inchikey:
                        match_found = True
                        match_method = f"InChIKey match: {lig['het_id']}"
                        break
                
                # Method 2: InChIKey from SMILES comparison
                if expected_inchikey_from_smiles and lig_inchikey and lig_inchikey != 'N/A':
                    if expected_inchikey_from_smiles == lig_inchikey:
                        match_found = True
                        match_method = f"InChIKey (from SMILES) match: {lig['het_id']}"
                        break
                
                # Method 3: SMILES comparison using RDKit
                if RDKIT_AVAILABLE and expected_smiles and lig_smiles and lig_smiles != 'N/A':
                    if self._compare_smiles(expected_smiles, lig_smiles):
                        match_found = True
                        match_method = f"SMILES match: {lig['het_id']}"
                        break
            
            if match_found:
                candidates_step2.append(pdb_id)
                print(f"  ✓ {pdb_id} - Ligand MATCH: {match_method}")
            else:
                print(f"  ✗ {pdb_id} - No ligand match found → DROPPED")
        
        print(f"\n  Remaining: {','.join(candidates_step2) if candidates_step2 else 'None'}")
        
        if not candidates_step2:
            return None, "No ligand match"
        
        # STEP 3: Mutation check
        print(f"\n[STEP 3: Mutation Check]")
        candidates_step3 = []
        
        for pdb_id in candidates_step2:
            metadata = self.fetch_metadata(pdb_id)
            mutations = metadata.get('mutations', [])
            
            if not mutations:
                candidates_step3.append(pdb_id)
                print(f"  ✓ {pdb_id} - Wild-type (no mutations)")
            else:
                print(f"  ✗ {pdb_id} - Has mutations: {', '.join(mutations[:5])} → DROPPED")
        
        print(f"  Remaining: {','.join(candidates_step3) if candidates_step3 else 'None'}")
        
        if not candidates_step3:
            return None, "No wild-type structures"
        
        # STEP 4: Best resolution
        print(f"\n[STEP 4: Resolution Ranking]")
        
        if len(candidates_step3) == 1:
            best_pdb = candidates_step3[0]
            print(f"  Only 1 candidate, no ranking needed")
        else:
            # Sort by resolution (lower is better)
            ranked = []
            for pdb_id in candidates_step3:
                metadata = self.fetch_metadata(pdb_id)
                resolution = metadata.get('resolution', 999.9)
                ranked.append((pdb_id, resolution))
                res_str = f"{resolution:.2f}Å" if resolution else "N/A"
                print(f"  {pdb_id}: resolution={res_str}")
            
            # Sort: resolution ASC
            ranked.sort(key=lambda x: (x[1] if x[1] else 999.9))
            best_pdb = ranked[0][0]
            print(f"  → Selected {best_pdb} (best resolution)")
        
        print(f"\n[FINAL SELECTED: {best_pdb}]")
        
        # Download PDB structure
        pdb_path = self.download_pdb_structure(best_pdb)
        if pdb_path:
            print(f"  ✓ Downloaded: {pdb_path}")
        
        return best_pdb, "Valid"


def main():
    parser = argparse.ArgumentParser(description="Validate PDB IDs from ChEMBL")
    parser.add_argument("--assay_type", choices=["Kd", "IC50"], default="Kd",
                       help="Assay type (default: Kd)")
    parser.add_argument("--out_dir", default=".", help="Output directory")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "pdb_cache"
    
    # Input file
    # input_csv = out_dir / "filtered" / f"bindingdb_{args.assay_type}_with_pdb_val.csv"
    input_csv = out_dir / "filtered" / f"chembl_{args.assay_type}_with_pdb_val.csv"
    
    if not input_csv.exists():
        print(f"❌ Input file not found: {input_csv}")
        return
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = out_dir / "structures" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pdb_validation_{args.assay_type}_{timestamp}.log"
    logger = TeeLogger(log_file)
    sys.stdout = logger
    
    try:
        print(f"{'='*80}")
        print(f"PDB VALIDATION STARTED")
        print(f"{'='*80}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Assay type: {args.assay_type}")
        print(f"Log file: {log_file}")
        print(f"{'='*80}\n")
        
        print(f"Loading {input_csv}...")
        df = pd.read_csv(input_csv)
        total_rows = len(df)
        print(f"Loaded {total_rows} rows\n")
        df = df.rename(columns={"pdb_ids": "complex_pdb_id", "accession": "uniprot_id"})
        
        # Verify required columns
        required_cols = ['protein_key', 'complex_pdb_id', 'uniprot_id', 'ligand_inchikey', 'canonical_smiles']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        print(f"✓ Required columns present: {required_cols}\n")
        
        # Initialize validator
        validator = PDBValidator(cache_dir, assay_type=args.assay_type, verbose=True)
        
        # Validate each row
        results = []
        for idx, row in df.iterrows():
            try:
                valid_pdb, reason = validator.validate_protein(row, idx + 1, total_rows)
                results.append({
                    'protein_key': row['protein_key'],
                    'original_pdbs': row.get('complex_pdb_id', ''),
                    'valid_pdb_id': valid_pdb if valid_pdb else '',
                    'validation_status': reason
                })
                
                # Save checkpoint every 50 proteins
                if (idx + 1) % 50 == 0:
                    checkpoint_df = pd.DataFrame(results)
                    checkpoint_csv = out_dir / f"pdb_validation_checkpoint_{args.assay_type}_{timestamp}.csv"
                    checkpoint_df.to_csv(checkpoint_csv, index=False)
                    print(f"\n  💾 Checkpoint saved: {idx + 1}/{total_rows} proteins validated\n")
                    
            except Exception as e:
                print(f"\n❌ ERROR processing protein {idx + 1}: {e}\n")
                results.append({
                    'protein_key': row.get('protein_key', 'UNKNOWN'),
                    'original_pdbs': row.get('complex_pdb_id', ''),
                    'valid_pdb_id': '',
                    'validation_status': f'Error: {str(e)}'
                })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results)
        
        # Merge results back to original DataFrame
        df['valid_pdb_id'] = summary_df['valid_pdb_id']
        df['validation_status'] = summary_df['validation_status']
        output_csv = out_dir / "filtered" / f"chembl_{args.assay_type}_with_pdb_val_saved_pdb.csv"
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Saved full results: {output_csv}")

        df_valid = df[df['valid_pdb_id'] != ''].copy()
        output_valid_csv = out_dir / "filtered" / f"chembl_{args.assay_type}_with_pdb_val_valid_pdb.csv"
        df_valid.to_csv(output_valid_csv, index=False)
        print(f"✅ Saved valid entries: {output_valid_csv}")

        # Save summary
        summary_dir = out_dir / "structures" / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = summary_dir / f"pdb_validation_summary_{args.assay_type}_{timestamp}.csv"
        summary_df.to_csv(summary_csv, index=False)
        
        print(f"\n{'='*80}")
        print(f"VALIDATION COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Saved summary: {summary_csv}")
        print(f"✅ Saved log: {log_file}")
        print(f"\nCache statistics:")
        print(f"  Cache hits: {validator.cache_hits}")
        print(f"  Cache misses: {validator.cache_misses}")
        print(f"\nValidation summary:")
        print(summary_df['validation_status'].value_counts())
        
        # Final statistics
        valid_count = (summary_df['valid_pdb_id'] != '').sum()
        print(f"\n✅ Successfully validated: {valid_count}/{total_rows} entries ({100*valid_count/total_rows:.1f}%)")
        print(f"   Downloaded PDB structures in: {cache_dir / f'{args.assay_type}_val_pdb'}")
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    finally:
        # Always close logger
        logger.close()
        sys.stdout = logger.terminal


if __name__ == "__main__":
    main()