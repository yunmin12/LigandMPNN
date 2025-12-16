"""
Find proteins that appear in both BindingDB and SAIR datasets.
Compares uniprot_id from BindingDB with protein column from SAIR.
"""

import pandas as pd
from pathlib import Path

# Paths
# BINDINGDB_CSV = "/scratch/yunmin/data/graph/bdb/clustered/bindingdb_IC50_clustered_stage1.csv"
BINDINGDB_CSV = "/scratch/yunmin/data/graph/bdb/filtered/bindingdb_IC50_with_pdb_val_valid_pdb.csv"
SAIR_CSV = "/scratch/yunmin/data/graph/sair/filtered/sair_pIC50_all_val.csv"
OUT_DIR = Path("/scratch/yunmin/data/graph/sair/common_proteins")

def main():
    print("="*70)
    print("FINDING COMMON PROTEINS: BindingDB vs SAIR")
    print("="*70)
    
    # Load datasets
    print(f"\nLoading BindingDB: {BINDINGDB_CSV}")
    bdb_df = pd.read_csv(BINDINGDB_CSV)
    print(f"  Loaded {len(bdb_df):,} entries")
    print(f"  Unique proteins (uniprot_id): {bdb_df['uniprot_id'].nunique()}")
    
    print(f"\nLoading SAIR: {SAIR_CSV}")
    sair_df = pd.read_csv(SAIR_CSV)
    print(f"  Loaded {len(sair_df):,} entries")
    print(f"  Unique proteins: {sair_df['protein'].nunique()}")
    
    # Find common proteins
    bdb_proteins = set(bdb_df['uniprot_id'].dropna().unique())
    sair_proteins = set(sair_df['protein'].dropna().unique())
    
    common_proteins = bdb_proteins & sair_proteins
    bdb_only = bdb_proteins - sair_proteins
    sair_only = sair_proteins - bdb_proteins
    
    print(f"\n{'='*70}")
    print("OVERLAP ANALYSIS")
    print(f"{'='*70}")
    print(f"BindingDB proteins: {len(bdb_proteins)}")
    print(f"SAIR proteins: {len(sair_proteins)}")
    print(f"Common proteins: {len(common_proteins)}")
    print(f"  - BindingDB only: {len(bdb_only)}")
    print(f"  - SAIR only: {len(sair_only)}")
    
    if len(common_proteins) == 0:
        print("\n❌ No common proteins found!")
        return
    
    # Filter datasets for common proteins
    bdb_common = bdb_df[bdb_df['uniprot_id'].isin(common_proteins)].copy()
    sair_common = sair_df[sair_df['protein'].isin(common_proteins)].copy()
    
    print(f"\nFiltered entries:")
    print(f"  - BindingDB: {len(bdb_common):,} entries")
    print(f"  - SAIR: {len(sair_common):,} entries")
    
    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    bdb_out = OUT_DIR / "bindingdb_common_proteins_all.csv"
    sair_out = OUT_DIR / "sair_common_proteins_all.csv"
    proteins_out = OUT_DIR / "common_protein_list_all.txt"
    
    bdb_common.to_csv(bdb_out, index=False)
    sair_common.to_csv(sair_out, index=False)
    
    with open(proteins_out, 'w') as f:
        for protein in sorted(common_proteins):
            f.write(f"{protein}\n")
    
    print(f"\n{'='*70}")
    print("SAVED OUTPUTS")
    print(f"{'='*70}")
    print(f"✅ {bdb_out}")
    print(f"✅ {sair_out}")
    print(f"✅ {proteins_out}")
    
    # Per-protein summary
    summary = []
    for protein in sorted(common_proteins):
        bdb_count = (bdb_common['uniprot_id'] == protein).sum()
        sair_count = (sair_common['protein'] == protein).sum()
        
        summary.append({
            'protein': protein,
            'bindingdb_entries': bdb_count,
            'sair_entries': sair_count,
            'total_entries': bdb_count + sair_count
        })
    
    summary_df = pd.DataFrame(summary)
    summary_out = OUT_DIR / "common_proteins_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    
    print(f"✅ {summary_out}")
    
    print(f"\n{'='*70}")
    print("TOP 10 PROTEINS BY TOTAL ENTRIES")
    print(f"{'='*70}")
    print(summary_df.nlargest(10, 'total_entries').to_string(index=False))
    
    # Find common protein-ligand pairs
    print(f"\n{'='*70}")
    print("FINDING COMMON PROTEIN-LIGAND PAIRS")
    print(f"{'='*70}")
    
    common_ligand_pairs = []
    
    for protein in sorted(common_proteins):
        bdb_protein = bdb_common[bdb_common['uniprot_id'] == protein]
        sair_protein = sair_common[sair_common['protein'] == protein]
        
        # Get unique SMILES for this protein in both datasets
        bdb_smiles = set(bdb_protein['ligand_smiles'].dropna().unique())
        sair_smiles = set(sair_protein['ligand_smiles'].dropna().unique())
        
        # Find common SMILES
        common_smiles = bdb_smiles & sair_smiles
        
        if len(common_smiles) > 0:
            for smiles in common_smiles:
                # Get all BindingDB entries with this protein-ligand pair
                bdb_matches = bdb_protein[bdb_protein['ligand_smiles'] == smiles]
                # Get all SAIR entries with this protein-ligand pair
                sair_matches = sair_protein[sair_protein['ligand_smiles'] == smiles]
                
                for _, bdb_row in bdb_matches.iterrows():
                    for _, sair_row in sair_matches.iterrows():
                        common_ligand_pairs.append({
                            'protein': protein,
                            'ligand_smiles': smiles,
                            'bindingdb_ligand_het_id': bdb_row['ligand_het_id'],
                            'bindingdb_target_type': bdb_row.get('target_type', 'N/A'),
                            'sair_entry_id': sair_row['entry_id'],
                            'sair_index': sair_row['index'],
                            'sair_target_type': sair_row.get('target_type', 'N/A')
                        })
    
    if len(common_ligand_pairs) > 0:
        common_ligands_df = pd.DataFrame(common_ligand_pairs)
        ligands_out = OUT_DIR / "common_protein_ligand_pairs.csv"
        common_ligands_df.to_csv(ligands_out, index=False)
        
        print(f"Found {len(common_ligand_pairs)} matching protein-ligand pairs!")
        print(f"  - Unique proteins with matching ligands: {common_ligands_df['protein'].nunique()}")
        print(f"  - Unique SMILES matched: {common_ligands_df['ligand_smiles'].nunique()}")
        print(f"\n✅ {ligands_out}")
        
        print(f"\nTop 10 proteins by matching ligands:")
        ligand_counts = common_ligands_df.groupby('protein').size().sort_values(ascending=False)
        print(ligand_counts.head(10).to_string())
        
        # First, identify which pairs have PDB structures in BindingDB
        print(f"\n{'='*70}")
        print("IDENTIFYING PAIRS WITH PDB STRUCTURES")
        print(f"{'='*70}")
        
        pdb_pairs = set()
        for _, row in bdb_df.iterrows():
            protein = row['uniprot_id']
            smiles = row['ligand_smiles']
            pdb_id = row.get('valid_pdb_id', None)
            if pd.notna(pdb_id) and str(pdb_id).strip():
                pdb_pairs.add((protein, smiles))
        
        print(f"Found {len(pdb_pairs)} unique protein-ligand pairs with PDB structures")
        
        # Check for existing CIF structure files (only for pairs with PDB)
        print(f"\n{'='*70}")
        print("CHECKING FOR EXISTING STRUCTURE FILES (with PDB validation)")
        print(f"{'='*70}")
        
        structures_dir = Path("/scratch/yunmin/data/graph/sair/structures")
        existing_cifs = []
        
        for _, row in common_ligands_df.iterrows():
            protein = row['protein']
            smiles = row['ligand_smiles']
            cif_filename = f"sample_{row['sair_entry_id']}_model_{row['sair_index']}.cif"
            cif_path = structures_dir / cif_filename
            
            # Only include if it has both PDB and CIF
            has_pdb = (protein, smiles) in pdb_pairs
            if has_pdb and cif_path.exists():
                existing_cifs.append(str(cif_path))
        
        print(f"CIF files with PDB validation: {len(existing_cifs)}")
        
        # Save list of existing files (only those with PDB)
        if existing_cifs:
            existing_out = OUT_DIR / "existing_cif_files_with_pdb.txt"
            with open(existing_out, 'w') as f:
                for cif_path in existing_cifs:
                    f.write(f"{cif_path}\n")
            print(f"\n✅ {existing_out}")
        
        # Add PDB and CIF existence flags to dataframe
        common_ligands_df['has_pdb'] = common_ligands_df.apply(
            lambda row: (row['protein'], row['ligand_smiles']) in pdb_pairs,
            axis=1
        )
        common_ligands_df['cif_exists'] = common_ligands_df.apply(
            lambda row: (structures_dir / f"sample_{row['sair_entry_id']}_model_{row['sair_index']}.cif").exists(),
            axis=1
        )
        common_ligands_df['cif_path'] = common_ligands_df.apply(
            lambda row: str(structures_dir / f"sample_{row['sair_entry_id']}_model_{row['sair_index']}.cif") if (structures_dir / f"sample_{row['sair_entry_id']}_model_{row['sair_index']}.cif").exists() else None,
            axis=1
        )
        
        # Extract PDB information from BindingDB dataset
        print(f"\n{'='*70}")
        print("EXTRACTING PDB STRUCTURE INFORMATION")
        print(f"{'='*70}")
        
        pdb_matches = []
        
        for _, row in common_ligands_df.iterrows():
            protein = row['protein']
            smiles = row['ligand_smiles']
            
            # Find matching entries in BindingDB (which now only has PDB entries)
            matches = bdb_df[
                (bdb_df['uniprot_id'] == protein) & 
                (bdb_df['ligand_smiles'] == smiles)
            ]
            
            if len(matches) > 0:
                for _, pdb_row in matches.iterrows():
                    pdb_id = pdb_row.get('valid_pdb_id', None)
                    if pd.notna(pdb_id) and str(pdb_id).strip():
                        pdb_matches.append({
                            'protein': protein,
                            'ligand_smiles': smiles,
                            'valid_pdb_id': pdb_id,
                            'ligand_het_id': pdb_row.get('ligand_het_id', None)
                        })
        
        if pdb_matches:
            pdb_matches_df = pd.DataFrame(pdb_matches)
            pdb_out = OUT_DIR / "common_pairs_with_pdb.csv"
            pdb_matches_df.to_csv(pdb_out, index=False)
            
            print(f"Found {len(pdb_matches)} protein-ligand pairs with PDB structures!")
            print(f"  - Unique proteins with PDB: {pdb_matches_df['protein'].nunique()}")
            print(f"  - Unique PDB IDs: {pdb_matches_df['valid_pdb_id'].nunique()}")
            print(f"\n✅ {pdb_out}")
            
            # Save list of PDB IDs
            pdb_list_out = OUT_DIR / "valid_pdb_ids.txt"
            with open(pdb_list_out, 'w') as f:
                for pdb_id in sorted(pdb_matches_df['valid_pdb_id'].unique()):
                    f.write(f"{pdb_id}\n")
            print(f"✅ {pdb_list_out}")
            
            # Merge PDB info into main dataframe
            pdb_info = pdb_matches_df.groupby(['protein', 'ligand_smiles'])['valid_pdb_id'].apply(lambda x: ';'.join(sorted(set(x)))).reset_index()
            pdb_info.columns = ['protein', 'ligand_smiles', 'valid_pdb_ids']
            
            common_ligands_df = common_ligands_df.merge(pdb_info, on=['protein', 'ligand_smiles'], how='left')
        else:
            print("No PDB structures found for common protein-ligand pairs")
            common_ligands_df['valid_pdb_ids'] = None
        
        # Save updated dataframe with CIF and PDB status
        ligands_with_cif_out = OUT_DIR / "common_protein_ligand_pairs_with_cif.csv"
        common_ligands_df.to_csv(ligands_with_cif_out, index=False)
        print(f"✅ {ligands_with_cif_out}")
        
    else:
        print("No common protein-ligand pairs found (same protein + same SMILES)")
    
    print(f"\n{'='*70}")
    print(f"✅ Complete! Found {len(common_proteins)} common proteins")
    if len(common_ligand_pairs) > 0:
        print(f"✅ Found {len(common_ligand_pairs)} matching protein-ligand pairs")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
