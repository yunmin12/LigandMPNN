"""
Validate PDB IDs from BindingDB filtered data and select the best structure per protein.

Steps:
1. Protein validation (exact UniProt ID match)
2. Ligand validation (HET ID match)
3. Mutation check (wild-type)
4. Best resolution
"""

import pandas as pd
import requests
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import gzip


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
    def __init__(self, cache_dir, verbose=True):
        self.cache = PDBCache(cache_dir)
        self.cif_dir = Path(cache_dir) / "cif"
        self.pdb_dir = Path(cache_dir) / "pdb"
        self.cif_dir.mkdir(parents=True, exist_ok=True)
        self.pdb_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
    
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
            
            # Enrich with ligand data from PDBe
            metadata['ligands'] = self._fetch_ligands_pdbe(pdb_id)
            
            # Cache the result
            self.cache.save(pdb_id, metadata)
            
            # Rate limiting (be polite to RCSB)
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
        
        # Extract UniProt IDs
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
    
    def _fetch_ligands_pdbe(self, pdb_id):
        """Fetch ligand information from PDBe API."""
        url = f"https://www.ebi.ac.uk/pdbe/api/pdb/compound/in_pdb/{pdb_id}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return []
            
            data = response.json()
            ligands = []
            
            for pdb_key, compounds in data.items():
                for compound in compounds:
                    ligands.append({
                        "het_id": compound.get("chem_comp_id", ""),
                        "name": compound.get("chem_comp_name", ""),
                        "inchikey": compound.get("inchikey", ""),
                        "smiles": compound.get("smiles", "")
                    })
            
            return ligands
            
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️  PDBe API error for {pdb_id}: {e}")
            return []
    
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
    
    def validate_protein(self, row):
        """Validate PDB IDs for a single protein row."""
        protein_key = row['protein_key']
        pdb_str = str(row.get('complex_pdb_id', '')).strip()
        
        if not pdb_str or pdb_str == 'nan':
            return None, "No PDB IDs"
        
        # Parse PDB IDs (split by comma)
        pdb_ids = [p.strip().upper() for p in pdb_str.split(',') if p.strip()]
        
        if not pdb_ids:
            return None, "No valid PDB IDs"
        
        print(f"\n{'='*80}")
        print(f"Protein: {protein_key}")
        print(f"Original PDBs: {','.join(pdb_ids)}")
        
        # STEP 1: Protein UniProt validation
        print(f"\n[STEP 1: UniProt Validation]")
        expected_uniprot = str(row.get('uniprot_id', '')).strip()
        candidates_step1 = []
        
        for pdb_id in pdb_ids:
            metadata = self.fetch_metadata(pdb_id)
            if not metadata:
                print(f"  ✗ {pdb_id} - Failed to fetch metadata → DROPPED")
                continue
            
            if expected_uniprot in metadata['uniprot_ids']:
                candidates_step1.append(pdb_id)
                print(f"  ✓ {pdb_id} - UniProt match: {expected_uniprot}")
            else:
                found = ', '.join(metadata['uniprot_ids']) if metadata['uniprot_ids'] else 'None'
                print(f"  ✗ {pdb_id} - UniProt mismatch: found {found}, expected {expected_uniprot} → DROPPED")
        
        print(f"  Remaining: {','.join(candidates_step1) if candidates_step1 else 'None'}")
        
        if not candidates_step1:
            return None, "No UniProt match"
        
        # STEP 2: Ligand HET ID validation
        print(f"\n[STEP 2: Ligand HET ID Validation]")
        expected_het_id = str(row.get('ligand_het_id', '')).strip().upper()
        print(f"  Expected HET ID: {expected_het_id}")
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
                    print(f"    • {het_id}: {name}")
                    if inchikey != 'N/A':
                        print(f"      InChI: {inchikey[:27]}...")
            else:
                print(f"    (No ligands found)")
            
            # Check for HET ID match
            ligand_het_ids = [lig['het_id'].upper() for lig in ligands if lig['het_id']]
            
            if expected_het_id in ligand_het_ids:
                candidates_step2.append(pdb_id)
                print(f"  ✓ {pdb_id} - Ligand HET ID MATCH: {expected_het_id}")
            else:
                found_hets = ', '.join(ligand_het_ids[:5]) if ligand_het_ids else 'None'
                print(f"  ✗ {pdb_id} - HET ID mismatch: found {found_hets}, expected {expected_het_id} → DROPPED")
        
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
        
        # STEP 4: Best resolution + recent date
        print(f"\n[STEP 4: Resolution & Date Ranking]")
        
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
    parser = argparse.ArgumentParser(description="Validate PDB IDs from BindingDB")
    parser.add_argument("--assay_type", choices=["Kd", "IC50"], default="Kd",
                       help="Assay type (default: Kd)")
    parser.add_argument("--out_dir", default=".", help="Output directory")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "pdb_cache"
    
    # Input file
    input_csv = out_dir / f"bindingdb_{args.assay_type}_best_per_protein_w_1210.csv"
    
    if not input_csv.exists():
        print(f"❌ Input file not found: {input_csv}")
        return
    
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")
    
    # Initialize validator
    validator = PDBValidator(cache_dir, verbose=True)
    
    # Validate each row
    results = []
    for idx, row in df.iterrows():
        valid_pdb, reason = validator.validate_protein(row)
        results.append({
            'protein_key': row['protein_key'],
            'original_pdbs': row.get('complex_pdb_id', ''),
            'valid_pdb_id': valid_pdb if valid_pdb else '',
            'validation_status': reason
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # Add valid_pdb_id to original DataFrame
    df['valid_pdb_id'] = summary_df['valid_pdb_id']
    
    # Save outputs
    output_csv = out_dir / f"bindingdb_{args.assay_type}_valid.csv"
    summary_csv = out_dir / f"pdb_validation_summary.csv"
    
    df.to_csv(output_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\n{'='*80}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Saved validated data: {output_csv}")
    print(f"✅ Saved summary: {summary_csv}")
    print(f"\nCache statistics:")
    print(f"  Cache hits: {validator.cache_hits}")
    print(f"  Cache misses: {validator.cache_misses}")
    print(f"\nValidation summary:")
    print(summary_df['validation_status'].value_counts())
    
    # Final statistics
    valid_count = (summary_df['valid_pdb_id'] != '').sum()
    print(f"\n✅ Successfully validated: {valid_count}/{len(df)} entries")
    print(f"   Downloaded PDB structures in: {cache_dir / 'pdb'}")


if __name__ == "__main__":
    main()