"""
SAIR dataset curation for {protein, target, off-target} dataset based on pIC50 values.
Pipeline:
1. Validate pIC50 values per entry_id (std, range, median checks)
2. Classify target/off-target per entry_id using representative pIC50
3. Select best model among 5 models per entry_id
4. Apply confidence score filters
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

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

def validate_pIC50_values(pic50_values):
    """
    Validate pIC50 values and return representative value.
    
    Criteria:
    - std < 0.5 → Use MEDIAN
    - std ≥ 0.5 → DISCARD
    - Range crosses categories (min ≤ 5.5 AND max ≥ 7.5) → DISCARD
    - Median in gray zone (5.5 < median < 7.5) → DISCARD
    
    Returns:
        (is_valid, representative_pIC50, reason)
    """
    pic50_arr = np.array(pic50_values)
    
    # If only one unique value, use it directly
    if len(np.unique(pic50_arr)) == 1:
        val = pic50_arr[0]
        # Check if in gray zone
        if 5.5 < val < 7.5:
            return False, None, "median_in_gray_zone"
        return True, val, "single_value"
    
    # Calculate statistics
    std = np.std(pic50_arr, ddof=1)
    median = np.median(pic50_arr)
    min_val = np.min(pic50_arr)
    max_val = np.max(pic50_arr)
    
    # Check std threshold
    if std >= 0.5:
        return False, None, f"high_std_{std:.3f}"
    
    # Check if range crosses both category boundaries
    if min_val <= 5.5 and max_val >= 7.5:
        return False, None, f"crosses_categories_{min_val:.2f}-{max_val:.2f}"
    
    # Check if median in gray zone
    if 5.5 < median < 7.5:
        return False, None, f"median_in_gray_zone_{median:.2f}"
    
    return True, median, f"valid_std_{std:.3f}"

def select_best_model(models_df):
    """
    Multi-stage filtering for models of same entry_id.
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
    
    best_conf_idx = candidates_with_conf['confidence_score'].idxmax()
    best_conf_val = candidates_with_conf.loc[best_conf_idx, 'confidence_score']
    
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
    start_time = datetime.now()
    
    # Initialize metadata
    metadata = {
        "pipeline": "SAIR pIC50-based curation",
        "version": "2.0",
        "timestamp": start_time.isoformat(),
        "input": {
            "source": INPUT_PARQUET
        },
        "filtering_criteria": {
            "pIC50_validation": {
                "max_std": 0.5,
                "discard_gray_zone": "5.5 < median < 7.5",
                "discard_category_crossing": "min ≤ 5.5 AND max ≥ 7.5"
            },
            "target_cutoff_pIC50": 7.5,
            "off_target_cutoff_pIC50": 5.5,
            "confidence_filters": {
                "min_confidence_score": args.min_confidence,
                "min_complex_plddt": args.min_plddt,
                "min_complex_iplddt": args.min_iplddt,
                "min_iptm": args.min_iptm
            }
        },
        "validation_steps": [],
        "statistics": {}
    }
    
    print("Loading SAIR data...")
    df = pd.read_parquet(INPUT_PARQUET)
    initial_rows = len(df)
    print(f"Loaded {initial_rows} rows")
    
    metadata["statistics"]["initial_rows"] = initial_rows
    
    # ---------- Step 1: Basic Validation ----------
    print("\n" + "="*60)
    print("BASIC VALIDATION CHECKS")
    print("="*60)
    
    # Check 1: Each entry_id should have exactly 5 models (index 0-4)
    entry_counts = df.groupby('entry_id')['index'].nunique()
    bad_counts = entry_counts[entry_counts != 5]
    if len(bad_counts) > 0:
        print(f"  ⚠️  WARNING: {len(bad_counts)} entries don't have 5 models")
        print(f"  Dropping these entries...")
        df = df[~df['entry_id'].isin(bad_counts.index)].copy()
        metadata["validation_steps"].append({
            "step": "Check 1: 5 models per entry",
            "dropped_entries": len(bad_counts),
            "remaining_rows": len(df)
        })
    else:
        print(f"  ✅ All {len(entry_counts)} entries have 5 models")
        metadata["validation_steps"].append({
            "step": "Check 1: 5 models per entry",
            "dropped_entries": 0,
            "remaining_rows": len(df)
        })
    
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
        metadata["validation_steps"].append({
            "step": "Check 2: Unique protein-SMILES per entry",
            "dropped_entries": len(entries_to_drop),
            "remaining_rows": len(df)
        })
    else:
        print(f"  ✅ All entries have unique protein-SMILES pairs")
        metadata["validation_steps"].append({
            "step": "Check 2: Unique protein-SMILES per entry",
            "dropped_entries": 0,
            "remaining_rows": len(df)
        })
    
    # Check 3: Each entry_id should have the same sequence across all models
    entry_sequences = df.groupby('entry_id')['sequence'].nunique()
    bad_sequences = entry_sequences[entry_sequences != 1]
    
    if len(bad_sequences) > 0:
        print(f"  ⚠️  WARNING: {len(bad_sequences)} entries have inconsistent sequences")
        print(f"  Dropping these entries...")
        df = df[~df['entry_id'].isin(bad_sequences.index)].copy()
        metadata["validation_steps"].append({
            "step": "Check 3: Consistent sequences per entry",
            "dropped_entries": len(bad_sequences),
            "remaining_rows": len(df)
        })
    else:
        print(f"  ✅ All entries have consistent sequences")
        metadata["validation_steps"].append({
            "step": "Check 3: Consistent sequences per entry",
            "dropped_entries": 0,
            "remaining_rows": len(df)
        })
    
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
    print(f"  ✅ All sequences and pIC50 values are valid")
    
    metadata["validation_steps"].append({
        "step": "Check 4: Valid sequences and pIC50",
        "remaining_rows": len(df)
    })
    
    # ---------- PHASE 1: pIC50 Validation ----------
    print("\n" + "="*60)
    print("PHASE 1: pIC50 VALIDATION (per entry_id)")
    print("="*60)
    
    # Group by entry_id and validate pIC50 values
    pic50_validation_results = []
    
    for entry_id, group in df.groupby('entry_id'):
        pic50_values = group['pIC50'].values
        is_valid, representative_pic50, reason = validate_pIC50_values(pic50_values)
        
        pic50_validation_results.append({
            'entry_id': entry_id,
            'is_valid': is_valid,
            'representative_pIC50': representative_pic50,
            'validation_reason': reason,
            'original_std': np.std(pic50_values, ddof=1) if len(pic50_values) > 1 else 0.0,
            'original_median': np.median(pic50_values),
            'original_min': np.min(pic50_values),
            'original_max': np.max(pic50_values)
        })
    
    validation_df = pd.DataFrame(pic50_validation_results)
    
    # Statistics
    total_entries = len(validation_df)
    valid_entries = validation_df['is_valid'].sum()
    invalid_entries = total_entries - valid_entries
    
    print(f"Total entries: {total_entries}")
    print(f"  ✅ Valid: {valid_entries} ({100*valid_entries/total_entries:.1f}%)")
    print(f"  ❌ Invalid: {invalid_entries} ({100*invalid_entries/total_entries:.1f}%)")
    
    # Breakdown of invalid reasons
    if invalid_entries > 0:
        print("\nInvalid entry breakdown:")
        invalid_reasons = validation_df[~validation_df['is_valid']]['validation_reason'].value_counts()
        for reason, count in invalid_reasons.items():
            print(f"  - {reason}: {count}")
    else:
        invalid_reasons = pd.Series(dtype=int)
    
    # Merge back with original data
    df = df.merge(validation_df[['entry_id', 'is_valid', 'representative_pIC50']], on='entry_id')
    
    # Filter only valid entries
    df_valid = df[df['is_valid']].copy()
    print(f"\nAfter pIC50 validation: {len(df_valid)} rows ({df_valid['entry_id'].nunique()} entries)")
    
    metadata["statistics"]["pic50_validation"] = {
        "total_entries": int(total_entries),
        "valid_entries": int(valid_entries),
        "invalid_entries": int(invalid_entries),
        "invalid_breakdown": {k: int(v) for k, v in invalid_reasons.items()} if invalid_entries > 0 else {}
    }
    
    # ---------- PHASE 2: Classification (per entry_id) ----------
    print("\n" + "="*60)
    print("PHASE 2: CLASSIFICATION (per entry_id using representative pIC50)")
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
    
    # Classify per entry_id (not per row)
    entry_classifications = df_valid.groupby('entry_id')['representative_pIC50'].first().apply(classify_by_pIC50)
    entry_classifications_df = entry_classifications.to_frame('target_type').reset_index()
    
    # Merge back to all rows
    df_valid = df_valid.drop(columns=['target_type'], errors='ignore')
    df_valid = df_valid.merge(entry_classifications_df, on='entry_id')
    
    # All should be classified (no gray zone after validation)
    print(f"Classified entries: {df_valid['entry_id'].nunique()}")
    print(f"  - Targets (pIC50 ≥ 7.5): {(entry_classifications == 'target').sum()}")
    print(f"  - Off-targets (pIC50 ≤ 5.5): {(entry_classifications == 'off_target').sum()}")
    
    # Filter for proteins with both types
    entry_protein_map = df_valid.groupby('entry_id')['protein'].first()
    entry_type_map = df_valid.groupby('entry_id')['target_type'].first()
    
    protein_type_check = pd.DataFrame({
        'protein': entry_protein_map,
        'target_type': entry_type_map
    })
    
    protein_types = protein_type_check.groupby('protein')['target_type'].apply(lambda x: set(x))
    proteins_with_both = protein_types[protein_types.map(len) > 1].index
    
    entries_to_keep = protein_type_check[protein_type_check['protein'].isin(proteins_with_both)].index
    df_classified = df_valid[df_valid['entry_id'].isin(entries_to_keep)].copy()
    
    print(f"\nAfter filtering proteins with both types:")
    print(f"  - Proteins: {len(proteins_with_both)}")
    print(f"  - Entries: {df_classified['entry_id'].nunique()}")
    print(f"  - Rows (5 models per entry): {len(df_classified)}")
    
    metadata["statistics"]["after_classification"] = {
        "proteins_with_both_types": int(len(proteins_with_both)),
        "entries": int(df_classified['entry_id'].nunique()),
        "total_rows": int(len(df_classified))
    }
    
    # ---------- PHASE 3: Select Best Model (per entry_id) ----------
    print("\n" + "="*60)
    print("PHASE 3: BEST MODEL SELECTION (per entry_id)")
    print("="*60)

    best_models = []
    for entry_id, group in df_classified.groupby('entry_id'):
        best = select_best_model(group)
        best_models.append(best)
    
    best_models_df = pd.DataFrame(best_models)
    print(f"Selected {len(best_models_df)} best models (1 per entry_id)")
    print(f"  - Targets: {(best_models_df['target_type'] == 'target').sum()}")
    print(f"  - Off-targets: {(best_models_df['target_type'] == 'off_target').sum()}")
    
    metadata["statistics"]["after_model_selection"] = {
        "total_models": int(len(best_models_df)),
        "targets": int((best_models_df['target_type'] == 'target').sum()),
        "off_targets": int((best_models_df['target_type'] == 'off_target').sum())
    }
    
    # ---------- PHASE 4: Apply Confidence Filters ----------
    print("\n" + "="*60)
    print("PHASE 4: CONFIDENCE FILTERING")
    print("="*60)
    
    print(f"Applying confidence filters:")
    print(f"  - confidence_score ≥ {args.min_confidence}")
    print(f"  - complex_plddt ≥ {args.min_plddt}")
    print(f"  - complex_iplddt ≥ {args.min_iplddt}")
    print(f"  - iptm ≥ {args.min_iptm}")
    
    before_count = len(best_models_df)
    
    # Apply filters (handle NaN values)
    df_filtered = best_models_df[
        (best_models_df['confidence_score'].fillna(0) >= args.min_confidence) &
        (best_models_df['complex_plddt'].fillna(0) >= args.min_plddt) &
        (best_models_df['complex_iplddt'].fillna(0) >= args.min_iplddt) &
        (best_models_df['iptm'].fillna(0) >= args.min_iptm)
    ].copy()
    
    after_count = len(df_filtered)
    filtered_out = before_count - after_count
    
    print(f"\nFiltering results:")
    print(f"  - Before: {before_count} models")
    print(f"  - After: {after_count} models")
    print(f"  - Filtered out: {filtered_out} ({100*filtered_out/before_count:.1f}%)")
    
    metadata["statistics"]["confidence_filtering"] = {
        "before": int(before_count),
        "after": int(after_count),
        "filtered_out": int(filtered_out),
        "percentage_filtered": round(100 * filtered_out / before_count, 2) if before_count > 0 else 0
    }
    
    # Check if we still have proteins with both types
    entry_protein_map = df_filtered.groupby('entry_id')['protein'].first()
    entry_type_map = df_filtered.groupby('entry_id')['target_type'].first()
    
    protein_type_check = pd.DataFrame({
        'protein': entry_protein_map,
        'target_type': entry_type_map
    })
    
    protein_types_after = protein_type_check.groupby('protein')['target_type'].apply(lambda x: set(x))
    proteins_with_both_after = protein_types_after[protein_types_after.map(len) > 1].index
    
    entries_to_keep_final = protein_type_check[protein_type_check['protein'].isin(proteins_with_both_after)].index
    final = df_filtered[df_filtered['entry_id'].isin(entries_to_keep_final)].copy()
    
    print(f"  - Proteins with both types after filtering: {len(proteins_with_both_after)}")
    print(f"  - Final entries: {len(final)}")
    
    if len(final) == 0:
        print("  ⚠️  WARNING: No data left after confidence filtering!")
        print("  Consider relaxing the thresholds.")
        metadata["statistics"]["final_status"] = "FAILED: No data after confidence filtering"
        # Save metadata even on failure
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = out_dir / "sair_curation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        return
    
    final['sequence_length'] = final['sequence'].str.len()
    
    # Use representative_pIC50 for final output
    final['pIC50'] = final['representative_pIC50']
    
    print(f"\nFinal selection: {len(final)} models")
    print(f"  - Unique proteins: {final['protein'].nunique()}")
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
    
    bins = np.unique(np.concatenate([bins_before_5_5, bins_between, bins_after_7_5]))
    
    ax.hist(targets, bins=bins, alpha=0.6, label='Target (pIC50 ≥ 7.5)', color='blue', edgecolor='black')
    ax.hist(off_targets, bins=bins, alpha=0.6, label='Off-target (pIC50 ≤ 5.5)', color='red', edgecolor='black')
    
    ax.set_xlabel('pIC50', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Representative pIC50 in SAIR', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax.axvline(7.5, color='green', linestyle='--', linewidth=2, label='Target cutoff (pIC50 = 7.5)')
    ax.axvline(5.5, color='orange', linestyle='--', linewidth=2, label='Off-target cutoff (pIC50 = 5.5)')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = out_dir / "sair_pIC50_best_histogram_val.png"
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
    
    out_all_path = out_dir / "sair_pIC50_all_val.csv"
    final_out.to_csv(out_all_path, index=False)
    print(f"✅ Saved all models: {out_all_path} ({len(final_out)} rows)")
    
    # Best per protein
    best_target_idx = final_out[final_out["target_type"] == "target"].groupby("protein")["pIC50"].idxmax()
    best_off_idx = final_out[final_out["target_type"] == "off_target"].groupby("protein")["pIC50"].idxmin()
    
    best_per_protein = pd.concat([
        final_out.loc[best_target_idx],
        final_out.loc[best_off_idx]
    ]).sort_values(["protein", "target_type"])
    
    out_best_path = out_dir / "sair_pIC50_best_per_protein_val.csv"
    best_per_protein.to_csv(out_best_path, index=False)
    print(f"✅ Saved best per protein: {out_best_path} ({len(best_per_protein)} rows)")
    
    # ---------- Final Statistics ----------
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
    
    # Update metadata with final statistics
    end_time = datetime.now()
    metadata["statistics"]["final"] = {
        "unique_proteins": int(final_out['protein'].nunique()),
        "unique_entry_ids": int(final_out['entry_id'].nunique()),
        "total_models": int(len(final_out)),
        "targets": int((final_out['target_type'] == 'target').sum()),
        "off_targets": int((final_out['target_type'] == 'off_target').sum()),
        "pIC50_stats": {
            "target": {
                "mean": float(targets.mean()),
                "std": float(targets.std()),
                "min": float(targets.min()),
                "max": float(targets.max())
            },
            "off_target": {
                "mean": float(off_targets.mean()),
                "std": float(off_targets.std()),
                "min": float(off_targets.min()),
                "max": float(off_targets.max())
            }
        }
    }
    
    metadata["output_files"] = {
        "all_models": str(out_all_path),
        "best_per_protein": str(out_best_path),
        "histogram": str(hist_path)
    }
    
    metadata["execution_time"] = {
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds()
    }
    
    # Save metadata
    metadata_path = out_dir / "sair_curation_metadata_val.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✅ Saved metadata: {metadata_path}")

if __name__ == "__main__":
    main()