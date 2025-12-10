"""
BindingDB curation for {protein, target, off-target} dataset based on Kd/IC50 values (no mutations).
Target: Kd/IC50 <= 31.6 nM (high affinity)
Off-target: Kd/IC50 >= 3160 nM (low affinity)
"""

import pandas as pd
import numpy as np
import argparse
import json
import csv
import re
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# ---------- Configuration ----------
INPUT_TSV = "/scratch/yunmin/data/db/BindingDB_All.tsv"

def parse_args():
    p = argparse.ArgumentParser(description="BindingDB Kd/IC50-based curation")
    p.add_argument("--out_dir", dest="out_dir", required=False, default=".",
                   help="Output directory for CSV files")
    p.add_argument("--assay_type", dest="assay_type", required=False, default="Kd",
                   choices=["Kd", "IC50"], help="Assay type to use for curation (default: Kd)")
    return p.parse_args()

# ---------- Utility Functions ----------
def median_ignore_nan(values):
    """Calculate median ignoring NaN values."""
    vals = [v for v in values if pd.notna(v)]
    return np.nan if not vals else float(np.median(vals))

def first_nonnull(x):
    """Return first non-null value from series."""
    for v in x:
        if pd.notna(v) and str(v).strip() != "":
            return v
    return np.nan

# ---------- Main Processing ----------
def main():
    args = parse_args()
    start_time = datetime.now()
    
    # Initialize metadata
    metadata = {
        "pipeline": f"BindingDB {args.assay_type}-based curation",
        "version": "1.0",
        "timestamp": start_time.isoformat(),
        "input": {
            "source": INPUT_TSV
        },
        "filtering_criteria": {
            "assay_type": args.assay_type,
            "target_cutoff_nM": 31.6,
            "off_target_cutoff_nM": 3160,
            "intermediate_range_nM": [31.6, 3160],
            "multichain_filter": "single chain only",
            "sequence_filter": "non-empty sequences only"
        },
        "validation_steps": [],
        "statistics": {}
    }

    print("Loading BindingDB data...")
    # Load TSV with proper handling
    df = pd.read_csv(
        INPUT_TSV,
        sep="\t",
        engine="python",
        dtype=str,
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
        on_bad_lines="warn",
        thousands=None
    )
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)
    initial_rows = len(df)
    print(f"Loaded {initial_rows} rows")
    
    metadata["statistics"]["initial_rows"] = initial_rows

    # ---------- Step 1: Filter multichain complexes ----------
    print("Filtering single-chain complexes...")
    chain_col = "Number of Protein Chains in Target (>1 implies a multichain complex)"
    if chain_col in df.columns:
        chains = pd.to_numeric(
            df[chain_col].str.extract(r"(\d+)", expand=False), 
            errors="coerce"
        )
        before_multichain = len(df)
        df = df[chains.fillna(1) <= 1].copy()
        after_multichain = len(df)
        
        metadata["validation_steps"].append({
            "step": "Multichain filter",
            "before": before_multichain,
            "after": after_multichain,
            "dropped": before_multichain - after_multichain
        })
    
    print(f"After multichain filter: {len(df)} rows")
    
    # ---------- Step 2: Filter for valid sequences ----------
    print("Filtering for valid protein sequences...")
    seq_col = "BindingDB Target Chain Sequence 1"
    if seq_col not in df.columns:
        raise ValueError(f"Column '{seq_col}' not found")
    
    before_seq = len(df)
    df[seq_col] = df[seq_col].fillna("").str.strip().str.upper()
    df = df[df[seq_col].str.len() > 0].copy()
    after_seq = len(df)
    
    print(f"After sequence filter: {after_seq} rows")
    
    metadata["validation_steps"].append({
        "step": "Sequence filter",
        "before": before_seq,
        "after": after_seq,
        "dropped": before_seq - after_seq
    })
    
    # ---------- Step 3: Filter for assay measurements only ----------
    print(f"Filtering for {args.assay_type} measurements...")
    assay_col = f"{args.assay_type} (nM)"
    if assay_col not in df.columns:
        raise ValueError(f"Column '{assay_col}' not found in input TSV")
    
    # Keep original assay values before conversion
    df[f"original_{args.assay_type}_nM"] = df[assay_col].copy()
    
    def parse_multi_nM(cell):
        """Parse multi-value or inequality assay values."""
        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            return []
        vals = []
        for tok in re.split(r"[;,\s]+", str(cell).strip()):
            if not tok:
                continue
            s = tok.replace(",", "").strip()
            # Handle > and < symbols
            if s.startswith((">", "<")):
                s = s[1:].strip()
            try:
                v = float(s)
                if v > 0:
                    vals.append(v)
            except:
                pass
        return vals

    def to_single_nM(cell):
        """Convert cell to single numeric assay value (median of multiple values)."""
        xs = parse_multi_nM(cell)
        return float(np.median(xs)) if xs else np.nan

    # Convert assay to numeric
    before_assay = len(df)
    df[assay_col] = df[assay_col].apply(to_single_nM)
    df = df[df[assay_col].notna()].copy()
    after_assay = len(df)
    
    print(f"After {args.assay_type} filter: {after_assay} rows")
    
    metadata["validation_steps"].append({
        "step": f"{args.assay_type} filter",
        "before": before_assay,
        "after": after_assay,
        "dropped": before_assay - after_assay
    })
    
    # ---------- Step 4: Create protein_key (UniProt + Organism) ----------
    print("Creating protein identifiers...")
    uniprot_col = "UniProt (SwissProt) Primary ID of Target Chain 1"
    organism_col = "Target Source Organism According to Curator or DataSource"
    
    df["protein_key"] = (
        df[uniprot_col].fillna("UNK") + "_" + 
        df[organism_col].fillna("UNK")
    )

    # ---------- Step 5: Aggregate by protein + sequence + ligand ----------
    print("Aggregating data...")
    group_cols = ["protein_key", seq_col, "Ligand InChI Key"]
    
    before_agg = len(df)
    
    agg = df.groupby(group_cols).agg(
        uniprot_id=(uniprot_col, first_nonnull),
        organism=(organism_col, first_nonnull),
        target_name=("Target Name", first_nonnull),
        ligand_name=("BindingDB Ligand Name", first_nonnull),
        ligand_smiles=("Ligand SMILES", first_nonnull),
        ligand_het_id=("Ligand HET ID in PDB", lambda x: ";".join(sorted(set(str(v) for v in x.dropna() if str(v).strip())))),
        original_assay_values=(f"original_{args.assay_type}_nM", lambda x: ";".join(sorted(set(str(v) for v in x.dropna() if str(v).strip())))),
        median_assay_nM=(assay_col, lambda x: median_ignore_nan(pd.to_numeric(x, errors='coerce'))),
        complex_pdb_id=("PDB ID(s) for Ligand-Target Complex", 
                       lambda x: ";".join(sorted(set(str(v) for v in x.dropna() if str(v).strip())))),
        doi=("Article DOI", first_nonnull),
        curation_source=("Curation/DataSource", first_nonnull)
    ).reset_index()
    
    after_agg = len(agg)
    print(f"After aggregation: {after_agg} unique protein-ligand pairs")
    
    metadata["validation_steps"].append({
        "step": "Aggregation by protein-ligand pairs",
        "before": before_agg,
        "after": after_agg,
        "aggregated": before_agg - after_agg
    })

    # ---------- Step 6: Classify target vs off-target ----------
    def classify_by_assay(assay_value):
        """Classify ligand as target or off-target based on assay threshold."""
        if pd.isna(assay_value):
            return np.nan
        if assay_value <= 31.6:
            return "target"
        elif assay_value >= 3160:
            return "off_target"
        else:
            return None
    
    print("Classifying target/off-target...")
    agg["target_type"] = agg["median_assay_nM"].apply(classify_by_assay)
    
    # Remove intermediate affinity entries
    before_classification = len(agg)
    agg = agg[agg["target_type"].isin(["target", "off_target"])].copy()
    after_classification = len(agg)
    
    n_targets = (agg['target_type'] == 'target').sum()
    n_off_targets = (agg['target_type'] == 'off_target').sum()
    
    print(f"After classification: {after_classification} rows (targets + off-targets)")
    print(f"  - Targets ({args.assay_type} ≤ 31.6 nM): {n_targets}")
    print(f"  - Off-targets ({args.assay_type} ≥ 3160 nM): {n_off_targets}")
    
    metadata["statistics"]["classification"] = {
        "before": before_classification,
        "after": after_classification,
        "intermediate_removed": before_classification - after_classification,
        "targets": int(n_targets),
        "off_targets": int(n_off_targets)
    }

    # ---------- Step 7: Prepare output columns ----------
    print("Preparing output...")
    output_cols = [
        "protein_key",
        "uniprot_id", 
        "target_name",
        "organism",
        seq_col,
        "Ligand InChI Key",
        "ligand_smiles",
        "ligand_name",
        "ligand_het_id",
        "original_assay_values",
        "median_assay_nM",
        "target_type",
        "curation_source",
        "doi",
        "complex_pdb_id"
    ]
    
    final = agg[output_cols].rename(columns={
        seq_col: "sequence",
        "Ligand InChI Key": "ligand_inchikey",
        "organism": "target_source"
    })
    final.insert(final.columns.get_loc("sequence") + 1, "sequence_length", final["sequence"].str.len())
    
    # Filter proteins with both targets and off-targets
    print("Filtering proteins with both target and off-target ligands...")
    protein_types = final.groupby('protein_key')['target_type'].apply(lambda x: set(x))
    proteins_with_both = protein_types[protein_types.map(len) > 1].index

    before_both = len(final)
    final = final[final['protein_key'].isin(proteins_with_both)].copy()
    after_both = len(final)
    
    print(f"After filtering for proteins with both types: {after_both} rows")
    print(f"  - Proteins: {len(proteins_with_both)}")
    
    metadata["statistics"]["both_types_filter"] = {
        "before": before_both,
        "after": after_both,
        "proteins_with_both": int(len(proteins_with_both))
    }

    # Sort by protein and assay
    final = final.sort_values(["protein_key", "median_assay_nM"]).reset_index(drop=True)

    # ---------- Step 8: Plot histograms ----------
    print("Generating histograms...")
    
    # Histogram 1: Raw assay values (before classification)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_raw_values = df[assay_col].dropna()
    
    min_val_raw = all_raw_values.min()
    max_val_raw = all_raw_values.max()
    
    bins_before_100_raw = np.logspace(np.log10(max(min_val_raw, 0.01)), np.log10(100), 20)
    bins_between_raw = np.logspace(np.log10(100), np.log10(3160), 20)
    bins_after_3160_raw = np.logspace(np.log10(3160), np.log10(max_val_raw), 20)
    bins_raw = np.unique(np.concatenate([bins_before_100_raw, bins_between_raw, bins_after_3160_raw]))
    
    ax.hist(all_raw_values, bins=bins_raw, alpha=0.7, color='gray', edgecolor='black')
    
    ax.set_xscale('log')
    ax.set_xlabel(f'{args.assay_type} (nM, log scale)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of {args.assay_type} in BindingDB (all)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax.axvline(100, color='green', linestyle='--', linewidth=2, label='Target cutoff (100 nM)')
    ax.axvline(3160, color='orange', linestyle='--', linewidth=2, label='Off-target cutoff (3160 nM)')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_raw_path = out_dir / f"bindingdb_{args.assay_type}_raw_histogram.png"
    plt.savefig(hist_raw_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved raw histogram: {hist_raw_path}")
    
    # Histogram 2: Classified median values (targets vs off-targets)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    targets = final[final["target_type"] == "target"]["median_assay_nM"]
    off_targets = final[final["target_type"] == "off_target"]["median_assay_nM"]
    
    min_val = min(targets.min(), off_targets.min())
    max_val = max(targets.max(), off_targets.max())
    
    bins_before_100 = np.logspace(np.log10(min_val), np.log10(100), 20)
    bins_between = np.logspace(np.log10(100), np.log10(3160), 20)
    bins_after_3160 = np.logspace(np.log10(3160), np.log10(max_val), 20)
    bins = np.unique(np.concatenate([bins_before_100, bins_between, bins_after_3160]))

    ax.hist(targets, bins=bins, alpha=0.6, label=f'Target (≤100 nM)', color='blue', edgecolor='black')
    ax.hist(off_targets, bins=bins, alpha=0.6, label=f'Off-target (≥3160 nM)', color='red', edgecolor='black')
    
    ax.set_xscale('log')
    ax.set_xlabel(f'{args.assay_type} (nM, log scale)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of {args.assay_type} in BindingDB', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax.axvline(100, color='green', linestyle='--', linewidth=2, label='Target cutoff (100 nM)')
    ax.axvline(3160, color='orange', linestyle='--', linewidth=2, label='Off-target cutoff (3160 nM)')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    hist_classified_path = out_dir / f"bindingdb_{args.assay_type}_classified_histogram.png"
    plt.savefig(hist_classified_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved classified histogram: {hist_classified_path}")

    # ---------- Step 9: Save outputs ----------
    out_all_path = out_dir / f"bindingdb_{args.assay_type}_all_1210.csv"
    out_pdb_path = out_dir / f"bindingdb_{args.assay_type}_with_pdb_1210.csv"
    out_best_wo_path = out_dir / f"bindingdb_{args.assay_type}_best_per_protein_wo_1210.csv"
    out_best_w_path = out_dir / f"bindingdb_{args.assay_type}_best_per_protein_w_1210.csv"

    # Output 1: All entries
    final.to_csv(out_all_path, index=False)
    print(f"✅ Saved all entries: {out_all_path} ({len(final)} rows)")
    
    # Output 2: Entries with PDB structures
    targets_with_pdb = final[
        (final["target_type"] == "target") &
        final["complex_pdb_id"].notna() & 
        (final["complex_pdb_id"].str.strip() != "") &
        (final["complex_pdb_id"].str.strip() != "nan")
    ]
    proteins_with_target_pdb = set(targets_with_pdb['protein_key'].unique())
    final_with_pdb = final[final['protein_key'].isin(proteins_with_target_pdb)].copy()
    
    final_with_pdb.to_csv(out_pdb_path, index=False)
    print(f"✅ Saved PDB entries: {out_pdb_path} ({len(final_with_pdb)} rows)")
    print(f"  - Proteins with at least one target PDB: {len(proteins_with_target_pdb)}")

    # Output 3: Best ligand per protein (without PDB requirement)
    best_target_idx = final[final["target_type"] == "target"].groupby("protein_key")["median_assay_nM"].idxmin()
    best_off_idx = final[final["target_type"] == "off_target"].groupby("protein_key")["median_assay_nM"].idxmax()

    best_per_protein_wo = pd.concat([
        final.loc[best_target_idx],
        final.loc[best_off_idx]
    ], axis=0).sort_values(["protein_key", "target_type"]).reset_index(drop=True)
    
    best_per_protein_wo.to_csv(out_best_wo_path, index=False)
    print(f"✅ Saved best ligands per protein: {out_best_wo_path} ({len(best_per_protein_wo)} rows)")
    print(f"  - Proteins: {best_per_protein_wo['protein_key'].nunique()}")

    # Output 4: Best ligand per protein (with PDB requirement for target)
    final_with_target_pdb = final[final['protein_key'].isin(proteins_with_target_pdb)].copy()
    
    targets_pdb_only = final_with_target_pdb[
        (final_with_target_pdb["target_type"] == "target") &
        final_with_target_pdb["complex_pdb_id"].notna() & 
        (final_with_target_pdb["complex_pdb_id"].str.strip() != "") &
        (final_with_target_pdb["complex_pdb_id"].str.strip() != "nan")
    ]
    best_target_idx = targets_pdb_only.groupby("protein_key")["median_assay_nM"].idxmin()
    
    off_targets_all = final_with_target_pdb[final_with_target_pdb["target_type"] == "off_target"]
    best_off_idx = off_targets_all.groupby("protein_key")["median_assay_nM"].idxmax()

    best_per_protein_w = pd.concat([
        targets_pdb_only.loc[best_target_idx],
        off_targets_all.loc[best_off_idx]
    ], axis=0).sort_values(["protein_key", "target_type"]).reset_index(drop=True)
    
    best_per_protein_w.to_csv(out_best_w_path, index=False)
    print(f"✅ Saved best ligands per protein (target with PDB, top off-target): {out_best_w_path} ({len(best_per_protein_w)} rows)")
    print(f"  - Proteins: {best_per_protein_w['protein_key'].nunique()}")
    print(f"  - Targets (all with PDB): {(best_per_protein_w['target_type'] == 'target').sum()}")
    print(f"  - Off-targets (no PDB requirement): {(best_per_protein_w['target_type'] == 'off_target').sum()}")
    
    # Verify all proteins have both types
    check_types = best_per_protein_w.groupby('protein_key')['target_type'].apply(lambda x: set(x))
    proteins_missing_type = check_types[check_types.map(len) < 2]
    if len(proteins_missing_type) > 0:
        print(f"  ⚠️  WARNING: {len(proteins_missing_type)} proteins missing one type")
    else:
        print(f"  ✅ All proteins have both target and off-target")
    
    # ---------- Final Statistics ----------
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total unique proteins: {final['protein_key'].nunique()}")
    print(f"Total unique ligands: {final['ligand_inchikey'].nunique()}")
    print(f"Total protein-ligand pairs: {len(final)}")
    print(f"  - With PDB structures: {len(final_with_pdb)}")
    print(f"  - Without PDB structures: {len(final) - len(final_with_pdb)}")
    print(f"\nTarget distribution:")
    print(final['target_type'].value_counts())
    
    # Count proteins with both types
    protein_types_final = final.groupby('protein_key')['target_type'].apply(lambda x: set(x))
    both_count = sum(1 for s in protein_types_final if len(s) > 1)
    print(f"\nProteins with both targets and off-targets: {both_count}")
    
    print(f"\n{args.assay_type} statistics:")
    print(f"  - Target (mean ± std): {targets.mean():.2f} ± {targets.std():.2f} nM")
    print(f"  - Off-target (mean ± std): {off_targets.mean():.2f} ± {off_targets.std():.2f} nM")
    
    # Update metadata with final statistics
    end_time = datetime.now()
    
    metadata["statistics"]["final"] = {
        "unique_proteins": int(final['protein_key'].nunique()),
        "unique_ligands": int(final['ligand_inchikey'].nunique()),
        "total_pairs": len(final),
        "with_pdb": len(final_with_pdb),
        "without_pdb": len(final) - len(final_with_pdb),
        "targets": int((final['target_type'] == 'target').sum()),
        "off_targets": int((final['target_type'] == 'off_target').sum()),
        "proteins_with_both": int(both_count),
        "assay_stats": {
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
        "all": str(out_all_path),
        "with_pdb": str(out_pdb_path),
        "best_per_protein_wo_pdb": str(out_best_wo_path),
        "best_per_protein_w_pdb": str(out_best_w_path),
        "histogram_raw": str(hist_raw_path),
        "histogram_classified": str(hist_classified_path)
    }
    
    metadata["execution_time"] = {
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds()
    }
    
    # Save metadata
    metadata_path = out_dir / f"bindingdb_{args.assay_type}_curation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✅ Saved metadata: {metadata_path}")

if __name__ == "__main__":
    main()
