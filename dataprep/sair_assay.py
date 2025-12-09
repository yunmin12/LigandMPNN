"""
SAIR dataset curation for {protein, target, off-target} dataset based on pIC50 values.
1. Classify target/off-target for each (entry_id, index, pIC50) combination
    Target: pIC50 >= 7.5 (high affinity, <= 100 nM)
    Off-target: pIC50 <= 5.5 (low affinity, >= 1 uM)
2. Select best model among classified entries
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# ---------- Configuration ----------
INPUT_PARQUET = "/scratch/yunmin/data/db/graph/sair/sair.parquet"

def parse_args():
    p = argparse.ArgumentParser(description="SAIR pIC50-based curation")
    p.add_argument("--out_dir", dest="out_dir", required=False, default=".",
                   help="Output directory for CSV files")
    p.add_argument("--min_confidence", type=float, default=0.80,
                   help="Minimum confidence_score (default: 0.80)")
    p.add_argument("--min_plddt", type=float, default=0.80,
                   help="Minimum complex_plddt (default: 0.80)")
    p.add_argument("--min_iplddt", type=float, default=0.70,
                   help="Minimum complex_iplddt (default: 0.70)")
    p.add_argument("--min_iptm", type=float, default=0.75,
                   help="Minimum iptm (default: 0.75)")
    return p.parse_args()

# ---------- Utility Functions ----------
def first_nonnull(x):
    """Return first non-null value from series."""
    for v in x:
        if pd.notna(v) and str(v).strip() != "":
            return v
    return np.nan

def select_best_model(models_df):
    """
    Multi-stage filtering for models of same entry_id with same pIC50.
    Returns: best model (single row)
    """
    # Stage 1: Validation (must pass all three)
    valid = models_df[
        (models_df['all_passed'] == True) &
        (models_df['passes_valence_checks'] == True) &
        (models_df['passes_kekulization'] == True)
    ]
    
    if len(valid) == 0:
        if models_df['confidence_score'].notna().any():
            return models_df.loc[models_df['confidence_score'].idxmax()]
        else:
            return models_df.iloc[0]
    
    # Stage 2: Zero clashes preferred
    zero_clash = valid[valid['number_clashes'] == 0]
    candidates = zero_clash if len(zero_clash) > 0 else valid
    
    # Stage 3: Highest confidence_score
    candidates_with_conf = candidates[candidates['confidence_score'].notna()]

    if len(candidates_with_conf) == 0:
        if candidates['vina_score_min'].notna().any():
            return candidates.loc[candidates['vina_score_min'].idxmin()]
        else:
            return candidates.iloc[0]
    
    best_conf_idx = candidates['confidence_score'].idxmax()
    best_conf_val = candidates.loc[best_conf_idx, 'confidence_score']
    
    # Stage 4: Among models within confidence threshold, pick lowest vina_score_min
    confidence_threshold = 0.02
    high_conf = candidates_with_conf[
        candidates_with_conf['confidence_score'] >= (best_conf_val - confidence_threshold)
    ]
    
    if len(high_conf) > 1 and high_conf['vina_score_min'].notna().any():
        return high_conf.loc[high_conf['vina_score_min'].idxmin()]
    else:
        return high_conf.loc[high_conf['confidence_score'].idxmax()]

# ---------- Main Processing ----------
def main():
    args = parse_args()
    
    print("Loading SAIR data...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} rows")
    
    # ---------- Step 1: Validate Entry/Sequence/IC50 Integrity ----------
    print("\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)
    
    # Check 1: Each entry_id should have exactly 5 models (index 0-4)
    entry_counts = df.groupby('entry_id')['index'].nunique()
    bad_counts = entry_counts[entry_counts != 5]
    if len(bad_counts) > 0:
        print(f"  ⚠️  WARNING: {len(bad_counts)} entries don't have 5 models")
        print(f"  Dropping these entries...")
        df = df[~df['entry_id'].isin(bad_counts.index)].copy()
    else:
        print(f"  ✅ All {len(entry_counts)} entries have 5 models")
    
    # Check 2: Each entry_id should have unique protein-SMILES pairs
    entry_proteins = df.groupby('entry_id')['protein'].nunique()
    entry_smiles = df.groupby('entry_id')['SMILES'].nunique()
    bad_proteins = entry_proteins[entry_proteins != 1]
    bad_smiles = entry_smiles[entry_smiles != 1]
    
    entries_to_drop = set()
    if len(bad_proteins) > 0:
        print(f"  ⚠️  WARNING: {len(bad_proteins)} entries have multiple proteins")
        entries_to_drop.update(bad_proteins.index)
    if len(bad_smiles) > 0:
        print(f"  ⚠️  WARNING: {len(bad_smiles)} entries have multiple SMILES")
        entries_to_drop.update(bad_smiles.index)
    
    if entries_to_drop:
        print(f"  Dropping {len(entries_to_drop)} entries with inconsistent protein/SMILES...")
        df = df[~df['entry_id'].isin(entries_to_drop)].copy()
    else:
        print(f" ✅ All entries have unique protein-SMILES pairs")
    
    # Check 3: Each entry_id should have the same sequence across all models
    entry_sequences = df.groupby('entry_id')['sequence'].nunique()
    bad_sequences = entry_sequences[entry_sequences != 1]
    
    if len(bad_sequences) > 0:
        print(f" ⚠️ WARNING: {len(bad_sequences)} entries have inconsistent sequences")
        print(f"  Dropping these entries...")
        df = df[~df['entry_id'].isin(bad_sequences.index)].copy()
    else:
        print(f" ✅ All entries have consistent sequences")
    
    print(f"\nAfter validation: {len(df)} rows, {df['entry_id'].nunique()} unique entries")

    # Check 4: Filter for valid sequences and pIC50
    df["sequence"] = df["sequence"].fillna("").str.strip().str.upper()
    df = df[df["sequence"].str.len() > 0].copy()
    
    df["pIC50"] = pd.to_numeric(df["pIC50"], errors="coerce")
    df = df[df["pIC50"].notna()].copy()
    print(f"After filters: {len(df)} rows")
    
    # Create ligand identifier
    df["ligand_smiles"] = df["SMILES"]
    df = df[df["ligand_smiles"].notna()].copy()
    print(f" ✅ All sequences and pIC50 values are valid")
    
    # ---------- Step 2: Classify Each Row (entry_id, index, pIC50) ----------
    print("\n" + "="*60)
    print("CLASSIFICATION")
    print("="*60)
    
    def classify_by_pIC50(pic50_value):
        """Classify based on pIC50 threshold."""
        if pd.isna(pic50_value):
            return None
        if pic50_value >= 7.5:
            return "target"
        elif pic50_value <= 5.5:
            return "off_target"
        else:
            return None
    
    df["target_type"] = df["pIC50"].apply(classify_by_pIC50)
    
    # Remove intermediate affinity
    df_classified = df[df["target_type"].notna()].copy()
    print(f"Classified rows: {len(df_classified)}")
    print(f"  - Targets (pIC50 ≥ 7.5): {(df_classified['target_type'] == 'target').sum()}")
    print(f"  - Off-targets (pIC50 ≤ 6): {(df_classified['target_type'] == 'off_target').sum()}")
    
    # Filter for proteins with both types
    protein_types = df_classified.groupby('protein')['target_type'].apply(lambda x: set(x))
    proteins_with_both = protein_types[protein_types.map(len) > 1].index
    df_classified = df_classified[df_classified['protein'].isin(proteins_with_both)].copy()
    print(f"After filtering proteins with both types: {len(df_classified)} rows")
    print(f"  - Proteins with both: {len(proteins_with_both)}")
    
    # ---------- Step 3: Apply Confidence Filters ----------
    print("\n" + "="*60)
    print("CONFIDENCE SCORE FILTERING")
    print("="*60)
    
    print(f"Applying confidence filters:")
    print(f"  - confidence_score ≥ {args.min_confidence}")
    print(f"  - complex_plddt ≥ {args.min_plddt}")
    print(f"  - complex_iplddt ≥ {args.min_iplddt}")
    print(f"  - iptm ≥ {args.min_iptm}")
    
    before_count = len(df_classified)
    
    # Apply filters (handle NaN values)
    df_classified = df_classified[
        (df_classified['confidence_score'].fillna(0) >= args.min_confidence) &
        (df_classified['complex_plddt'].fillna(0) >= args.min_plddt) &
        (df_classified['complex_iplddt'].fillna(0) >= args.min_iplddt) &
        (df_classified['iptm'].fillna(0) >= args.min_iptm)
    ].copy()
    
    after_count = len(df_classified)
    filtered_out = before_count - after_count
    
    print(f"\nFiltering results:")
    print(f"  - Before: {before_count} models")
    print(f"  - After: {after_count} models")
    print(f"  - Filtered out: {filtered_out} ({100*filtered_out/before_count:.1f}%)")
    
    # Check if we still have proteins with both types
    protein_types_after = df_classified.groupby('protein')['target_type'].apply(lambda x: set(x))
    proteins_with_both_after = protein_types_after[protein_types_after.map(len) > 1].index
    df_classified = df_classified[df_classified['protein'].isin(proteins_with_both_after)].copy()
    
    print(f"  - Proteins with both types after filtering: {len(proteins_with_both_after)}")
    
    if len(df_classified) == 0:
        print("  ⚠️  WARNING: No data left after confidence filtering!")
        print("  Consider relaxing the thresholds.")
        return
    
    # ---------- Step 4: Select Best Model ----------
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)

    best_models = []
    for (entry_id, pic50), group in df_classified.groupby(['entry_id', 'pIC50']):
        # if there are multiple models with same pIC50
        if len(group) > 1:
            # then use ranking to select best
            best = select_best_model(group)
        # Only one model with this pIC50
        else:
            # then just take it
            best = group.iloc[0]
        
        best_models.append(best)
    
    best_models_df = pd.DataFrame(best_models)
    print(f"Selected {len(best_models_df)} best models")
    
    # For each entry_id, if there are multiple pIC50 values, take the best per target_type
    final_best = []
    
    for entry_id, group in best_models_df.groupby('entry_id'):
        for target_type in group['target_type'].unique():
            type_group = group[group['target_type'] == target_type]
            
            if target_type == 'target':
                # Take highest pIC50 for targets
                best = type_group.loc[type_group['pIC50'].idxmax()]
            else:  # off_target
                # Take lowest pIC50 for off-targets
                best = type_group.loc[type_group['pIC50'].idxmin()]
            
            final_best.append(best)
    
    final = pd.DataFrame(final_best)
    final['sequence_length'] = final['sequence'].str.len()
    
    print(f"Final selection: {len(final)} models")
    print(f"  - Unique entry_ids: {final['entry_id'].nunique()}")
    print(f"  - Targets: {(final['target_type'] == 'target').sum()}")
    print(f"  - Off-targets: {(final['target_type'] == 'off_target').sum()}")
    
    # ---------- Step 5: Plot Histogram ----------
    print("\nGenerating histogram...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    targets = final[final["target_type"] == "target"]["pIC50"]
    off_targets = final[final["target_type"] == "off_target"]["pIC50"]
    
    min_val = min(off_targets.min(), targets.min())
    max_val = max(off_targets.max(), targets.max())
    
    bins_before_5_5 = np.linspace(min_val, 5.5, 15)
    bins_between = np.linspace(5.5, 7.5, 10)
    bins_after_7_5 = np.linspace(7.5, max_val, 15)
    
    # Combine and remove duplicates
    bins = np.unique(np.concatenate([bins_before_5_5, bins_between, bins_after_7_5]))
    
    ax.hist(targets, bins=bins, alpha=0.6, label='Target (pIC50 ≥ 7.5)', color='blue', edgecolor='black')
    ax.hist(off_targets, bins=bins, alpha=0.6, label='Off-target (pIC50 ≤ 5.5)', color='red', edgecolor='black')
    
    ax.set_xlabel('pIC50', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of pIC50 in SAIR', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Vertical lines for cutoffs
    ax.axvline(7.5, color='green', linestyle='--', linewidth=2, label='Target cutoff (pIC50 = 7.5)')
    ax.axvline(5.5, color='orange', linestyle='--', linewidth=2, label='Off-target cutoff (pIC50 = 5.5)')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    hist_path = Path(args.out_dir) / "sair_pIC50_best_histogram.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved histogram: {hist_path}")
    
    # ---------- Step 6: Save Outputs ----------
    output_cols = [
        'entry_id', 'index', 'protein', 'sequence', 'sequence_length',
        'ligand_smiles', 'pIC50', 'target_type',
        'source', 'assay', 'family', 'description',
        'vina_score_min', 'number_clashes', 'internal_energy',
        'confidence_score', 'complex_plddt', 'complex_iplddt', 'ptm', 'iptm'
    ]
    
    final_out = final[output_cols].sort_values(['protein', 'target_type', 'pIC50'], ascending=[True, True, False])
    
    out_all_path = Path(args.out_dir) / "sair_pIC50_best_models_filtered.csv"
    final_out.to_csv(out_all_path, index=False)
    print(f"✅ Saved best models: {out_all_path} ({len(final_out)} rows)")
    
    # Best per protein
    best_target_idx = final_out[final_out["target_type"] == "target"].groupby("protein")["pIC50"].idxmax()
    best_off_idx = final_out[final_out["target_type"] == "off_target"].groupby("protein")["pIC50"].idxmin()
    
    best_per_protein = pd.concat([
        final_out.loc[best_target_idx],
        final_out.loc[best_off_idx]
    ]).sort_values(["protein", "target_type"])
    
    out_best_path = Path(args.out_dir) / "sair_pIC50_best_per_protein_filtered.csv"
    best_per_protein.to_csv(out_best_path, index=False)
    print(f"✅ Saved best per protein: {out_best_path} ({len(best_per_protein)} rows)")
    
    # ---------- Summary Statistics ----------
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total unique proteins: {final_out['protein'].nunique()}")
    print(f"Total unique entry_ids: {final_out['entry_id'].nunique()}")
    print(f"Total models selected: {len(final_out)}")
    print(f"\nTarget distribution:")
    print(final_out['target_type'].value_counts())
    print(f"\npIC50 statistics:")
    print(f"  - Target (mean ± std): {targets.mean():.2f} ± {targets.std():.2f}")
    print(f"  - Off-target (mean ± std): {off_targets.mean():.2f} ± {off_targets.std():.2f}")

if __name__ == "__main__":
    main()