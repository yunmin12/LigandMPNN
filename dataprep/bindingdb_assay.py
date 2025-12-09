"""
BindingDB curation for {protein, target, off-target} dataset based on Kd/IC50 values (no mutations).
Target: Kd/IC50 <= 31.6 nM (high affinity)
Off-target: Kd/IC50 >= 3160 nM (low affinity)
"""

import pandas as pd
import numpy as np
import argparse
import csv
import re
from pathlib import Path
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
    print(f"Loaded {len(df)} rows")

    # ---------- Step 1: Filter multichain complexes ----------
    print("Filtering single-chain complexes...")
    chain_col = "Number of Protein Chains in Target (>1 implies a multichain complex)"
    if chain_col in df.columns:
        chains = pd.to_numeric(
            df[chain_col].str.extract(r"(\d+)", expand=False), 
            errors="coerce"
        )
        df = df[chains.fillna(1) <= 1].copy()
    print(f"After multichain filter: {len(df)} rows")
    
    # ---------- Step 2: Filter for valid sequences ----------
    print("Filtering for valid protein sequences...")
    seq_col = "BindingDB Target Chain Sequence 1"
    if seq_col not in df.columns:
        raise ValueError(f"Column '{seq_col}' not found")
    
    df[seq_col] = df[seq_col].fillna("").str.strip().str.upper()
    df = df[df[seq_col].str.len() > 0].copy()
    print(f"After sequence filter: {len(df)} rows")
    
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
    df[assay_col] = df[assay_col].apply(to_single_nM)
    df = df[df[assay_col].notna()].copy()
    print(f"After {args.assay_type} filter: {len(df)} rows")
    
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
    
    print(f"After aggregation: {len(agg)} unique protein-ligand pairs")

    # ---------- Step 6: Classify target vs off-target ----------
    def classify_by_assay(assay_value):
        """Classify ligand as target or off-target based on assay threshold.
        By Cheng-Prusoff approximation, Kd ~ IC50 for competitive inhibitors."""
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
    agg = agg[agg["target_type"].isin(["target", "off_target"])].copy()
    print(f"After classification: {len(agg)} rows (targets + off-targets)")
    print(f"  - Targets ({args.assay_type} ≤ 31.6 nM): {(agg['target_type'] == 'target').sum()}")
    print(f"  - Off-targets ({args.assay_type} ≥ 3160 nM): {(agg['target_type'] == 'off_target').sum()}")

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

    final = final[final['protein_key'].isin(proteins_with_both)].copy()
    print(f"After filtering for proteins with both types: {len(final)} rows")

    # Sort by protein and assay
    final = final.sort_values(["protein_key", "median_assay_nM"]).reset_index(drop=True)

    # ---------- Step 8: Plot histograms ----------
    print("Generating histograms...")
    
    # Histogram 1: Raw assay values (before classification)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get all raw values after validation (before classification)
    all_raw_values = df[assay_col].dropna()
    
    min_val_raw = all_raw_values.min()
    max_val_raw = all_raw_values.max()
    
    # Create log-spaced bins for raw data
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
    
    # Vertical lines for cutoffs
    ax.axvline(100, color='green', linestyle='--', linewidth=2, label='Target cutoff (100 nM)')
    ax.axvline(3160, color='orange', linestyle='--', linewidth=2, label='Off-target cutoff (3160 nM)')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    hist_raw_path = Path(args.out_dir) / f"bindingdb_{args.assay_type}_raw_histogram.png"
    plt.savefig(hist_raw_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved raw histogram: {hist_raw_path}")
    
    # Histogram 2: Classified median values (targets vs off-targets)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    targets = final[final["target_type"] == "target"]["median_assay_nM"]
    off_targets = final[final["target_type"] == "off_target"]["median_assay_nM"]
    
    min_val = min(targets.min(), off_targets.min())
    max_val = max(targets.max(), off_targets.max())
    
    # Create log-spaced bins, ensuring 100 and 3160 are bin edges
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
    
    # Vertical lines for cutoffs
    ax.axvline(100, color='green', linestyle='--', linewidth=2, label='Target cutoff (100 nM)')
    ax.axvline(3160, color='orange', linestyle='--', linewidth=2, label='Off-target cutoff (3160 nM)')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    hist_classified_path = Path(args.out_dir) / f"bindingdb_{args.assay_type}_classified_histogram.png"
    plt.savefig(hist_classified_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved classified histogram: {hist_classified_path}")

    # ---------- Step 9: Save outputs ----------
    out_all_path = Path(args.out_dir) / f"bindingdb_{args.assay_type}_all_1209.csv"
    out_pdb_path = Path(args.out_dir) / f"bindingdb_{args.assay_type}_with_pdb_1209.csv"
    out_best_path = Path(args.out_dir) / f"bindingdb_{args.assay_type}_best_per_protein_1209.csv"

    # Output 1: All entries
    final.to_csv(out_all_path, index=False)
    print(f"✅ Saved all entries: {out_all_path} ({len(final)} rows)")
    
    # Output 2: Entries with PDB structures
    final_with_pdb = final[
        final["complex_pdb_id"].notna() & 
        (final["complex_pdb_id"].str.strip() != "") &
        (final["complex_pdb_id"].str.strip() != "nan")
    ].copy()
    
    final_with_pdb.to_csv(out_pdb_path, index=False)
    print(f"✅ Saved PDB entries: {out_pdb_path} ({len(final_with_pdb)} rows)")

    # Output 3: Best ligand per protein
    best_target_idx = final_with_pdb[final_with_pdb["target_type"] == "target"].groupby("protein_key")["median_assay_nM"].idxmin()
    best_target = final_with_pdb.loc[best_target_idx]
    best_off_idx = final_with_pdb[final_with_pdb["target_type"] == "off_target"].groupby("protein_key")["median_assay_nM"].idxmax()
    best_off = final_with_pdb.loc[best_off_idx]

    best_per_protein = pd.concat([best_target, best_off], axis=0).sort_values(["protein_key", "target_type"]).reset_index(drop=True)
    best_per_protein.to_csv(out_best_path, index=False)
    print(f"✅ Saved best ligands per protein (top target & top off-target): {out_best_path} ({len(best_per_protein)} rows)")
    
    # ---------- Summary Statistics ----------
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
    print(f"\nProteins with both targets and off-targets: ", end="")
    
    # Count proteins with both types
    protein_types = final.groupby('protein_key')['target_type'].apply(
        lambda x: set(x)
    )
    both = sum(1 for s in protein_types if len(s) > 1)
    print(f"{both}")

if __name__ == "__main__":
    main()
