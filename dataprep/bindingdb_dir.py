"""
Organize PDB files into structured directories based on bindingdb clustered data.
"""

import argparse
import os
import shutil
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Organize PDB files by protein and ligand targets'
    )
    parser.add_argument(
        '--assay',
        choices=['Kd', 'IC50'],
        help='Type of data to process (Kd or IC50)'
    )
    args = parser.parse_args()
    
    assay_type = args.assay
    
    # Define paths
    csv_path = f'/scratch/yunmin/data/graph/bdb/clustered/bindingdb_{assay_type}_clustered_stage1.csv'
    pdb_source_dir = f'/scratch/yunmin/data/graph/bdb/pdb_cache/{assay_type}_val_pdb'
    output_base_dir = f'/scratch/yunmin/data/graph/lmpnn/bdb_pdb/{assay_type}/lmpnn_in'
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    # Read CSV
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Track statistics
    total_rows = len(df)
    processed = 0
    skipped = 0
    copied = 0
    
    # Process each row
    for idx, row in df.iterrows():
        try:
            uniprot_id = str(row['uniprot_id'])
            target_type = str(row['target_type'])
            ligand_het_id = str(row['ligand_het_id'])
            valid_pdb_id = str(row['valid_pdb_id'])
            
            # Skip rows with missing data
            if pd.isna(row['uniprot_id']) or pd.isna(row['target_type']) or \
               pd.isna(row['ligand_het_id']) or pd.isna(row['valid_pdb_id']):
                print(f"Warning: Missing data at row {idx}, skipping")
                skipped += 1
                continue
            
            # Determine target directory name
            if target_type == 'target':
                target_dir = 'tar'
            elif target_type == 'off_target':
                target_dir = 'off'
            else:
                print(f"Warning: Unknown target_type '{target_type}' at row {idx}, skipping")
                skipped += 1
                continue
            
            # Create directory structure
            dest_dir = Path(output_base_dir) / uniprot_id / target_dir / ligand_het_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Source and destination PDB paths
            source_pdb = Path(pdb_source_dir) / f"{valid_pdb_id}.pdb"
            dest_pdb = dest_dir / f"{valid_pdb_id}.pdb"
            
            # Copy PDB file if it exists
            if source_pdb.exists():
                shutil.copy2(source_pdb, dest_pdb)
                copied += 1
            else:
                print(f"Warning: Source PDB not found: {source_pdb}")
                skipped += 1
            
            processed += 1
            
            # Progress update
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{total_rows} rows...")
                
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            skipped += 1
            continue
    
    # Print summary
    print("\n" + "="*50)
    print(f"Processing complete!")
    print(f"Total rows: {total_rows}")
    print(f"Processed: {processed}")
    print(f"Files copied: {copied}")
    print(f"Skipped: {skipped}")
    print("="*50)


if __name__ == '__main__':
    main()