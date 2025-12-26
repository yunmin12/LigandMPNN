"""
Count off-targets that share the same protein as PDB-validated targets.
Input CSV: 
    off_inputs/ligand_sdf/bindingdb_with_ligand_paths.csv
Output CSV: 
    off_inputs/tar_pdb_off_count.csv
"""

import os
import pandas as pd

# Configurations
LIG_THRESHOLD = 3  # Minimum number of ligands (both targets and off-targets) per protein
INPUT_CSV = "/scratch/yunmin/data/graph/bdb/filtered/bindingdb_Kd_with_pdb_val_saved_pdb.csv"

OUT_CSV="/scratch/yunmin/data/graph/lmpnn/bdb_pdb/Kd/off_inputs/tar_pdb_off_count.csv"
INPUT_FILTERED_CSV = f"/scratch/yunmin/data/graph/lmpnn/bdb_pdb/Kd/off_inputs/bindingdb_Kd_set{LIG_THRESHOLD}.csv"
OUT_FILTERED_CSV = f"/scratch/yunmin/data/graph/lmpnn/bdb_pdb/Kd/off_inputs/tar_pdb_off_count_set{LIG_THRESHOLD}.csv"

# Load dataframe
df = pd.read_csv(INPUT_CSV)
df['target_type'] = df['target_type'].astype(str).str.strip().str.lower()

PROT = 'protein_key'
LIG  = 'ligand_het_id'
PDB  = 'valid_pdb_id'

# Targets with valid_pdb_id (valid_target)
targets = (
    df[(df['target_type'] == 'target') & (df[PDB].notna())]
    [[PROT, 'uniprot_id', LIG, PDB]]
    .dropna(subset=[PROT, LIG, PDB])
    .drop_duplicates()
    .rename(columns={
        LIG: 'target_id',
        PDB: 'target_valid_pdb_id'
    })
    .sort_values([PROT, 'target_id'])
)

# Off-targets bind to the same protein
off_summary = (
    df[df['target_type'] == 'off_target']
    [[PROT, LIG]]
    .dropna(subset=[PROT, LIG])
    .drop_duplicates()
    .groupby(PROT)[LIG]
    .agg(
        off_target_ids=lambda s: ';'.join(sorted(set(s))),
        n_off_targets=lambda s: s.nunique()
    )
    .reset_index()
)

# Attach off-targets to the same (protein, valid_target) set if exists
result = (
    targets.merge(off_summary, on=PROT, how='inner')
    .sort_values([PROT, 'target_id'])
    .reset_index(drop=True)
)

# n_targets per protein in this df
result['n_targets'] = result.groupby('uniprot_id')['target_id'].transform('nunique')
result = result[
  ['uniprot_id',
   'n_targets',
   'target_id',
   'target_valid_pdb_id',
   'off_target_ids',
   'n_off_targets',
  ]
]

print(result)

result.to_csv(OUT_CSV, index=False)
print(f"\nCSV saved in {OUT_CSV}")

print("\n===== Proteins with PDB-validated targets and corresponding off-targets =====")
print("Threshold: n_targets & n_off_targets >= k")
print(f"current threshold = {LIG_THRESHOLD}")
print("threshold >= 1:", result.loc[(result['n_targets'] >= 1) & (result['n_off_targets'] >= 1), 'uniprot_id'].nunique())
print("threshold >= 3:", result.loc[(result['n_targets'] >= 3) & (result['n_off_targets'] >= 3), 'uniprot_id'].nunique())
print("threshold >= 5:", result.loc[(result['n_targets'] >= 5) & (result['n_off_targets'] >= 5), 'uniprot_id'].nunique())
print("threshold >= 7:", result.loc[(result['n_targets'] >= 7) & (result['n_off_targets'] >= 7), 'uniprot_id'].nunique())
print("threshold >= 10:", result.loc[(result['n_targets'] >= 10) & (result['n_off_targets'] >= 10), 'uniprot_id'].nunique())

filtered = result.loc[(result["n_targets"] >= LIG_THRESHOLD) & (result["n_off_targets"] >= LIG_THRESHOLD)].reset_index(drop=True)
filtered.to_csv(OUT_FILTERED_CSV, index=False)

filtered_uniprots = filtered['uniprot_id'].unique().tolist()
if len(filtered_uniprots) == 0:
    print("No proteins passed the filter. No filtered input CSV will be written.")
else:
    # attach filtered columns to the filtered original input CSV
    join_cols = ['uniprot_id', 'off_target_ids', 'n_targets', 'n_off_targets']
    join_df = filtered[join_cols].drop_duplicates(subset=['uniprot_id'])
    input_filtered_df = df[df['uniprot_id'].isin(filtered_uniprots)].reset_index(drop=True)
    input_filtered_df = input_filtered_df.merge(join_df, on='uniprot_id', how='left')
    input_filtered_df.to_csv(INPUT_FILTERED_CSV, index=False)
    print(f"Filtered input CSV saved in {INPUT_FILTERED_CSV}")
