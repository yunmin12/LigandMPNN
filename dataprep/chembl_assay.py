"""
ChEMBL curation for {protein, target, off-target} dataset based on Kd/IC50 values.
Target: Kd/IC50 <= 31.6 nM (high affinity)
Off-target: Kd/IC50 >= 3160 nM (low affinity)

Pipeline (following BindingDB structure):
1. Parse and aggregate assay measurements per protein-ligand pair
2. Validate assay values (convert to pIC50, check std < 0.5, convert back to nM)
3. Classify target/off-target using representative assay value
4. Filter for proteins with both types
"""

import pandas as pd
import numpy as np
import argparse
import json
import sqlite3
import requests
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# ---------- Configuration ----------
INPUT_DB = "/scratch/yunmin/data/graph/chembl/chembl_36/chembl_36_sqlite/chembl_36.db"

def parse_args():
    p = argparse.ArgumentParser(description="ChEMBL Kd/IC50-based curation")
    p.add_argument("--out_dir", dest="out_dir", required=False, default=".",
                   help="Output directory for CSV files")
    p.add_argument("--assay_type", dest="assay_type", required=False, default="Kd",
                   choices=["Kd", "IC50"], help="Assay type to use for curation (default: Kd)")
    p.add_argument("--confidence_cutoff", dest="confidence_cutoff", required=False, 
                   default=8, type=int,
                   help="Minimum confidence score for assays (0-9, default: 8)")
    return p.parse_args()

# ---------- Utility Functions ----------
def first_nonnull(x):
    """Return first non-null value from series."""
    for v in x:
        if pd.notna(v) and str(v).strip() != "":
            return v
    return np.nan

def nM_to_pIC50(nM_value):
    """Convert nM to pIC50: pIC50 = -log10(nM * 1e-9)"""
    if nM_value <= 0:
        return np.nan
    return -np.log10(nM_value * 1e-9)

def pIC50_to_nM(pIC50_value):
    """Convert pIC50 to nM: nM = 10^(-pIC50) / 1e-9"""
    return 10**(-pIC50_value) / 1e-9

def validate_assay_values(assay_values_nM):
    """
    Validate Kd/IC50 values and return representative value.
    
    Strategy:
    1. Convert nM values to pIC50 scale
    2. Apply same validation as SAIR (std < 0.5)
    3. Return representative value in nM
    
    Criteria (in pIC50 space):
    - std < 0.5 → Use MEDIAN (convert back to nM)
    - std ≥ 0.5 → DISCARD
    - Range crosses categories (min ≤ 5.5 AND max ≥ 7.5 in pIC50) → DISCARD
      (equivalent to: max ≥ 3160 nM AND min ≤ 31.6 nM)
    - Median in gray zone (5.5 < median < 7.5 in pIC50) → DISCARD
      (equivalent to: 31.6 < median < 3160 nM)
    
    Returns:
        (is_valid, representative_nM, reason)
    """
    # Filter out invalid values
    valid_nM = [v for v in assay_values_nM if v > 0]
    if not valid_nM:
        return False, None, "no_valid_values"
    
    assay_arr = np.array(valid_nM)
    
    # If only one unique value, use it directly
    if len(np.unique(assay_arr)) == 1:
        val_nM = assay_arr[0]
        # Check if in gray zone (31.6 < val < 3160 nM)
        if 31.6 < val_nM < 3160:
            return False, None, "median_in_gray_zone"
        return True, val_nM, "single_value"
    
    # Convert to pIC50 for validation
    pIC50_values = np.array([nM_to_pIC50(v) for v in assay_arr])
    
    # Calculate statistics in pIC50 space
    std_pIC50 = np.std(pIC50_values, ddof=1)
    median_pIC50 = np.median(pIC50_values)
    min_pIC50 = np.min(pIC50_values)
    max_pIC50 = np.max(pIC50_values)
    
    # Check std threshold (same as SAIR)
    if std_pIC50 >= 0.5:
        return False, None, f"high_std_pIC50_{std_pIC50:.3f}"
    
    # Check if range crosses both category boundaries
    # In pIC50: target ≥ 7.5, off-target ≤ 5.5
    # In nM: target ≤ 31.6, off-target ≥ 3160
    # Crosses if: min_pIC50 ≤ 5.5 AND max_pIC50 ≥ 7.5
    if min_pIC50 <= 5.5 and max_pIC50 >= 7.5:
        min_nM = pIC50_to_nM(max_pIC50)  # Note: pIC50 is inverted
        max_nM = pIC50_to_nM(min_pIC50)
        return False, None, f"crosses_categories_{min_nM:.2f}-{max_nM:.2f}nM"
    
    # Check if median in gray zone (5.5 < median_pIC50 < 7.5)
    # Equivalent to: 31.6 < median_nM < 3160
    if 5.5 < median_pIC50 < 7.5:
        median_nM = pIC50_to_nM(median_pIC50)
        return False, None, f"median_in_gray_zone_{median_nM:.2f}nM"
    
    # Valid: convert median back to nM
    representative_nM = pIC50_to_nM(median_pIC50)
    return True, representative_nM, f"valid_std_pIC50_{std_pIC50:.3f}"

# ---------- Database Query Functions ----------
def fetch_pdb_from_uniprot(uniprot_accession):
    """Fetch PDB IDs from UniProt REST API."""
    if not uniprot_accession or pd.isna(uniprot_accession):
        return None
    
    try:
        url = f"https://www.uniprot.org/uniprot/{uniprot_accession}.txt"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            pdb_ids = []
            for line in response.text.split('\n'):
                if line.startswith('DR   PDB;'):
                    # Format: DR   PDB; 1ABC, A/B/C=1-100, etc.
                    parts = line.split(';')
                    if len(parts) >= 2:
                        pdb_id = parts[1].strip().split(',')[0].strip()
                        pdb_ids.append(pdb_id)
            
            time.sleep(0.1)  # Rate limiting
            return ';'.join(pdb_ids) if pdb_ids else None
    except Exception as e:
        print(f"Warning: Could not fetch PDB for {uniprot_accession}: {e}")
        return None

def query_chembl_activities(conn, assay_type, confidence_cutoff):
    """Query ChEMBL database for activities with specified assay type."""
    
    query = f"""
    SELECT DISTINCT
        -- Target information
        td.chembl_id as target_chembl_id,
        cs.component_id,
        cs.accession,
        cs.sequence,
        cs.description as target_name,
        cs.organism as target_organism,
        cs.tax_id as target_tax_id,
        
        -- Ligand information
        md.chembl_id as ligand_chembl_id,
        md.molregno,
        cst.standard_inchi_key,
        cst.canonical_smiles,
        md.pref_name as compound_name,
        
        -- Activity data
        act.standard_value,
        act.standard_relation,
        act.standard_units,
        act.pchembl_value,
        act.standard_type,
        act.activity_id,
        
        -- Assay information
        ass.chembl_id as assay_chembl_id,
        ass.assay_type,
        ass.confidence_score,
        
        -- Document information
        doc.chembl_id as doc_chembl_id,
        doc.doi,
        doc.pubmed_id,
        doc.doc_id
        
    FROM activities act
    
    INNER JOIN assays ass ON act.assay_id = ass.assay_id
    INNER JOIN target_dictionary td ON ass.tid = td.tid
    INNER JOIN target_components tc ON td.tid = tc.tid
    INNER JOIN component_sequences cs ON tc.component_id = cs.component_id
    INNER JOIN molecule_dictionary md ON act.molregno = md.molregno
    INNER JOIN compound_structures cst ON md.molregno = cst.molregno
    LEFT JOIN docs doc ON act.doc_id = doc.doc_id
    
    WHERE
        act.standard_type = '{assay_type}'
        AND act.standard_value IS NOT NULL
        AND act.standard_units = 'nM'
        AND td.target_type = 'SINGLE PROTEIN'
        AND ass.confidence_score >= {confidence_cutoff}
        AND cs.sequence IS NOT NULL
        AND LENGTH(cs.sequence) > 0
        AND cs.component_type = 'PROTEIN'
        AND tc.homologue = 0
        AND cst.standard_inchi_key IS NOT NULL
    
    ORDER BY td.chembl_id, md.chembl_id
    """
    
    print(f"\nQuerying ChEMBL database for {assay_type} activities...")
    print(f"Confidence cutoff: >={confidence_cutoff}")
    
    df = pd.read_sql_query(query, conn)
    
    print(f"Retrieved {len(df)} activity records")
    return df

# ---------- Main Processing ----------
def main():
    args = parse_args()
    start_time = datetime.now()
    
    # Initialize metadata
    metadata = {
        "pipeline": f"ChEMBL {args.assay_type}-based curation (v2.0 - pIC50 validation)",
        "version": "2.0",
        "timestamp": start_time.isoformat(),
        "input": {
            "source": INPUT_DB,
            "database": "ChEMBL 36"
        },
        "filtering_criteria": {
            "assay_type": args.assay_type,
            "confidence_score_cutoff": args.confidence_cutoff,
            "assay_validation": {
                "method": "pIC50 conversion",
                "max_std_pIC50": 0.5,
                "discard_gray_zone_pIC50": "5.5 < median < 7.5 (31.6-3160 nM)",
                "discard_category_crossing_pIC50": "min ≤ 5.5 AND max ≥ 7.5"
            },
            "target_cutoff_nM": 31.6,
            "off_target_cutoff_nM": 3160,
            "intermediate_range_nM": [31.6, 3160],
            "target_type_filter": "SINGLE PROTEIN",
            "sequence_filter": "non-empty sequences only",
            "component_type_filter": "PROTEIN only",
            "homologue_filter": "exclude homologues"
        },
        "validation_steps": [],
        "statistics": {}
    }

    # Connect to ChEMBL database
    print("Connecting to ChEMBL database...")
    conn = sqlite3.connect(INPUT_DB)
    
    try:
        # ---------- Step 1: Query database ----------
        print("\n" + "="*60)
        print("STEP 1: DATABASE QUERY")
        print("="*60)
        
        df = query_chembl_activities(conn, args.assay_type, args.confidence_cutoff)
        initial_rows = len(df)
        
        metadata["statistics"]["initial_rows"] = initial_rows
        
        if initial_rows == 0:
            print("ERROR: No data retrieved from database!")
            return
        
        # ---------- Step 2: Parse assay values ----------
        print("\n" + "="*60)
        print("STEP 2: PARSE ASSAY VALUES")
        print("="*60)
        
        print("Processing standard_value and standard_relation...")
        
        # Handle relation symbols (>, <, =)
        def parse_standard_value(row):
            """Parse standard value considering relation symbols."""
            value = row['standard_value']
            relation = row['standard_relation']
            
            if pd.isna(value):
                return None
            
            # Convert to float
            try:
                val = float(value)
                if val > 0:
                    return val
            except:
                pass
            
            return None
        
        df['parsed_value_nM'] = df.apply(parse_standard_value, axis=1)
        
        # Filter rows with valid parsed values
        before_parse = len(df)
        df = df[df['parsed_value_nM'].notna()].copy()
        after_parse = len(df)
        
        print(f"  After parsing: {after_parse} rows")
        print(f"  Dropped: {before_parse - after_parse} rows")
        
        metadata["validation_steps"].append({
            "step": f"{args.assay_type} parsing",
            "before": before_parse,
            "after": after_parse,
            "dropped": before_parse - after_parse
        })
        
        # ---------- Step 3: Create protein_key ----------
        print("\n" + "="*60)
        print("STEP 3: CREATE PROTEIN IDENTIFIERS")
        print("="*60)
        
        # Use component_id + organism as protein key
        df["protein_key"] = (
            "CHEMBL_" + df["target_chembl_id"].astype(str) + "_" + 
            "COMP_" + df["component_id"].astype(str) + "_" +
            df["target_organism"].fillna("UNK").astype(str).str.replace(" ", "_")
        )
        
        print(f"Created protein keys for {df['protein_key'].nunique()} unique proteins")
        
        # ---------- Step 4: Aggregate by protein + ligand ----------
        print("\n" + "="*60)
        print("STEP 4: AGGREGATION")
        print("="*60)
        
        group_cols = ["protein_key", "sequence", "standard_inchi_key"]
        
        before_agg = len(df)
        
        # Aggregate all assay values into lists
        agg = df.groupby(group_cols).agg(
            target_chembl_id=("target_chembl_id", first_nonnull),
            component_id=("component_id", first_nonnull),
            accession=("accession", first_nonnull),
            target_name=("target_name", first_nonnull),
            target_organism=("target_organism", first_nonnull),
            target_tax_id=("target_tax_id", first_nonnull),
            ligand_chembl_id=("ligand_chembl_id", first_nonnull),
            molregno=("molregno", first_nonnull),
            canonical_smiles=("canonical_smiles", first_nonnull),
            compound_name=("compound_name", first_nonnull),
            assay_chembl_ids=("assay_chembl_id", lambda x: ";".join(sorted(set(str(v) for v in x.dropna() if str(v).strip())))),
            doc_chembl_ids=("doc_chembl_id", lambda x: ";".join(sorted(set(str(v) for v in x.dropna() if str(v).strip())))),
            doi=("doi", lambda x: ";".join(sorted(set(str(v) for v in x.dropna() if str(v).strip())))),
            pubmed_id=("pubmed_id", lambda x: ";".join(sorted(set(str(v) for v in x.dropna() if str(v).strip())))),
            # pdb_ids=("pdb_ids", lambda x: ";".join(sorted(set(str(v) for v in x.dropna() if str(v).strip() and str(v).strip() != "None")))),
            assay_values_nM=("parsed_value_nM", lambda x: list(x.dropna())),
            original_values_string=("standard_value", lambda x: ";".join(sorted(set(str(v) for v in x.dropna() if str(v).strip())))),
            n_measurements=("activity_id", "count")
        ).reset_index()
        
        after_agg = len(agg)
        print(f"Aggregated to {after_agg} unique protein-ligand pairs")
        
        metadata["validation_steps"].append({
            "step": "Aggregation by protein-ligand pairs",
            "before": before_agg,
            "after": after_agg,
            "aggregated": before_agg - after_agg
        })
        
        # ---------- Step 5: Validate assay values ----------
        print("\n" + "="*60)
        print("STEP 5: ASSAY VALUE VALIDATION (pIC50 method)")
        print("="*60)
        
        validation_results = []
        
        for idx, row in agg.iterrows():
            assay_vals = row["assay_values_nM"]
            is_valid, representative_nM, reason = validate_assay_values(assay_vals)
            
            # Calculate stats for tracking
            if len(assay_vals) > 0:
                pIC50_vals = [nM_to_pIC50(v) for v in assay_vals]
                std_pIC50 = np.std(pIC50_vals, ddof=1) if len(pIC50_vals) > 1 else 0.0
                median_pIC50 = np.median(pIC50_vals)
                min_nM = min(assay_vals)
                max_nM = max(assay_vals)
            else:
                std_pIC50 = np.nan
                median_pIC50 = np.nan
                min_nM = np.nan
                max_nM = np.nan
            
            validation_results.append({
                'is_valid': is_valid,
                'representative_assay_nM': representative_nM,
                'validation_reason': reason,
                'std_pIC50': std_pIC50,
                'original_min_nM': min_nM,
                'original_max_nM': max_nM
            })
        
        validation_df = pd.DataFrame(validation_results)
        agg = pd.concat([agg, validation_df], axis=1)
        
        # Statistics
        total_pairs = len(agg)
        valid_pairs = agg['is_valid'].sum()
        invalid_pairs = total_pairs - valid_pairs
        
        print(f"Total protein-ligand pairs: {total_pairs}")
        print(f"  ✅ Valid: {valid_pairs} ({100*valid_pairs/total_pairs:.1f}%)")
        print(f"  ❌ Invalid: {invalid_pairs} ({100*invalid_pairs/total_pairs:.1f}%)")
        
        # Breakdown of invalid reasons
        if invalid_pairs > 0:
            print("\nInvalid pair breakdown:")
            invalid_reasons = agg[~agg['is_valid']]['validation_reason'].value_counts()
            for reason, count in invalid_reasons.items():
                print(f"  - {reason}: {count}")
        else:
            invalid_reasons = pd.Series(dtype=int)
        
        # Filter only valid pairs
        agg_valid = agg[agg['is_valid']].copy()
        
        print(f"\nAfter validation: {len(agg_valid)} pairs")
        
        metadata["statistics"]["assay_validation"] = {
            "total_pairs": int(total_pairs),
            "valid_pairs": int(valid_pairs),
            "invalid_pairs": int(invalid_pairs),
            "invalid_breakdown": {k: int(v) for k, v in invalid_reasons.items()} if invalid_pairs > 0 else {}
        }
        
        # ---------- Step 6: Classify target vs off-target ----------
        print("\n" + "="*60)
        print("STEP 6: CLASSIFICATION")
        print("="*60)
        
        def classify_by_assay(assay_value):
            """Classify ligand as target or off-target based on assay threshold."""
            if pd.isna(assay_value):
                return None
            if assay_value <= 31.6:
                return "target"
            elif assay_value >= 3160:
                return "off_target"
            else:
                return None
        
        agg_valid["target_type"] = agg_valid["representative_assay_nM"].apply(classify_by_assay)
        
        # Remove entries that didn't pass classification
        before_classification = len(agg_valid)
        agg_classified = agg_valid[agg_valid["target_type"].isin(["target", "off_target"])].copy()
        after_classification = len(agg_classified)
        
        n_targets = (agg_classified['target_type'] == 'target').sum()
        n_off_targets = (agg_classified['target_type'] == 'off_target').sum()
        
        print(f"After classification: {after_classification} pairs (targets + off-targets)")
        print(f"  - Targets ({args.assay_type} ≤ 31.6 nM): {n_targets}")
        print(f"  - Off-targets ({args.assay_type} ≥ 3160 nM): {n_off_targets}")
        print(f"  - Removed (gray zone): {before_classification - after_classification}")
        
        metadata["statistics"]["classification"] = {
            "before": before_classification,
            "after": after_classification,
            "intermediate_removed": before_classification - after_classification,
            "targets": int(n_targets),
            "off_targets": int(n_off_targets)
        }
        
        # ---------- Step 7: Remove conflicting ligands ----------
        print("\n" + "="*60)
        print("STEP 7: REMOVE CONFLICTING LIGANDS")
        print("="*60)
        
        before_conflict = len(agg_classified)
        
        # Find ligands that appear as both target and off-target for the same protein
        ligand_types = agg_classified.groupby(['protein_key', 'standard_inchi_key'])['target_type'].apply(lambda x: set(x))
        conflicting_pairs = ligand_types[ligand_types.map(len) > 1].reset_index()
        
        if len(conflicting_pairs) > 0:
            print(f"Found {len(conflicting_pairs)} ligands appearing as both target and off-target")
            print("Removing all entries for these protein-ligand pairs...")
            
            # Create set of (protein_key, standard_inchi_key) to remove
            conflicts_set = set(zip(conflicting_pairs['protein_key'], conflicting_pairs['standard_inchi_key']))
            
            # Filter out conflicting pairs
            agg_classified = agg_classified[
                ~agg_classified.apply(lambda row: (row['protein_key'], row['standard_inchi_key']) in conflicts_set, axis=1)
            ].copy()
        else:
            print("No conflicting ligands found")
        
        after_conflict = len(agg_classified)
        removed_conflicts = before_conflict - after_conflict
        
        print(f"After removing conflicts: {after_conflict} pairs")
        print(f"  - Removed: {removed_conflicts} entries")
        
        metadata["statistics"]["conflict_removal"] = {
            "before": before_conflict,
            "after": after_conflict,
            "conflicting_ligands": int(len(conflicting_pairs)),
            "entries_removed": removed_conflicts
        }
        
        # ---------- Step 8: Filter proteins with both types ----------
        print("\n" + "="*60)
        print("STEP 8: FILTER PROTEINS WITH BOTH TYPES")
        print("="*60)
        
        protein_types = agg_classified.groupby('protein_key')['target_type'].apply(lambda x: set(x))
        proteins_with_both = protein_types[protein_types.map(len) > 1].index
        
        before_both = len(agg_classified)
        final = agg_classified[agg_classified['protein_key'].isin(proteins_with_both)].copy()
        after_both = len(final)
        
        print(f"Proteins with both types: {len(proteins_with_both)}")
        print(f"Protein-ligand pairs: {after_both}")
        
        metadata["statistics"]["both_types_filter"] = {
            "before": before_both,
            "after": after_both,
            "proteins_with_both": int(len(proteins_with_both))
        }

        # ---------- Step 9: Fetch PDB IDs (final dataset only) ----------
        print("\n" + "="*60)
        print("STEP 9: FETCH PDB IDs FROM UniProt")
        print("="*60)

        print(f"Fetching PDB IDs for {len(final)} unique proteins...")
        print("(This may take a few minutes due to UniProt API rate limiting)")

        # Fetch PDB only for unique proteins in final dataset
        unique_proteins = final[['protein_key', 'accession']].drop_duplicates()
        pdb_cache = {}

        for idx, row in unique_proteins.iterrows():
            accession = row['accession']
            if accession and pd.notna(accession):
                pdb_ids = fetch_pdb_from_uniprot(accession)
                pdb_cache[accession] = pdb_ids
            
            if (idx + 1) % 100 == 0:
                print(f"  Progress: {idx + 1}/{len(unique_proteins)}")

        # Map PDB IDs back to main dataframe
        final['pdb_ids'] = final['accession'].map(pdb_cache)

        n_with_pdb = final['pdb_ids'].notna().sum()
        print(f"✅ Fetched PDB information for {n_with_pdb}/{len(unique_proteins)} unique proteins")

        # ---------- Step 10: Prepare output columns ----------
        print("\n" + "="*60)
        print("PREPARING OUTPUT")
        print("="*60)
        
        output_cols = [
            "protein_key",
            "target_chembl_id",
            "component_id",
            "accession",
            "target_name",
            "target_organism",
            "target_tax_id",
            "sequence",
            "ligand_chembl_id",
            "molregno",
            "standard_inchi_key",
            "canonical_smiles",
            "compound_name",
            "original_values_string",
            "representative_assay_nM",
            "target_type",
            "n_measurements",
            "assay_chembl_ids",
            "doc_chembl_ids",
            "doi",
            "pubmed_id",
            "pdb_ids"
        ]
        
        final = final[output_cols].rename(columns={
            "standard_inchi_key": "ligand_inchikey",
            "target_organism": "target_source",
            f"original_values_string": f"original_{args.assay_type}_nM"
        })
        
        final.insert(final.columns.get_loc("sequence") + 1, "sequence_length", final["sequence"].str.len())
        
        # Sort by protein and assay
        final = final.sort_values(["protein_key", "representative_assay_nM"]).reset_index(drop=True)
        
        # ---------- Step 11: Plot histograms ----------
        print("Generating histograms...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        targets = final[final["target_type"] == "target"]["representative_assay_nM"]
        off_targets = final[final["target_type"] == "off_target"]["representative_assay_nM"]
        
        min_val = min(targets.min(), off_targets.min())
        max_val = max(targets.max(), off_targets.max())
        
        bins_before_31_6 = np.logspace(np.log10(max(min_val, 0.01)), np.log10(31.6), 20)
        bins_between = np.logspace(np.log10(31.6), np.log10(3160), 20)
        bins_after_3160 = np.logspace(np.log10(3160), np.log10(max_val), 20)
        bins = np.unique(np.concatenate([bins_before_31_6, bins_between, bins_after_3160]))
        
        ax.hist(targets, bins=bins, alpha=0.6, label=f'Target (≤31.6 nM)', color='blue', edgecolor='black')
        ax.hist(off_targets, bins=bins, alpha=0.6, label=f'Off-target (≥3160 nM)', color='red', edgecolor='black')
        
        ax.set_xscale('log')
        ax.set_xlabel(f'{args.assay_type} (nM, log scale)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Distribution of Representative {args.assay_type} in ChEMBL', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        ax.axvline(31.6, color='green', linestyle='--', linewidth=2, label='Target cutoff (31.6 nM)')
        ax.axvline(3160, color='orange', linestyle='--', linewidth=2, label='Off-target cutoff (3160 nM)')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        hist_path = out_dir / f"chembl_{args.assay_type}_histogram.png"
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved histogram: {hist_path}")
        
        # ---------- Step 12: Save outputs ----------
        out_all_path = out_dir / f"chembl_{args.assay_type}_all_val.csv"
        out_pdb_path = out_dir / f"chembl_{args.assay_type}_with_pdb_val.csv"
        out_best_wo_path = out_dir / f"chembl_{args.assay_type}_best_per_protein_wo_val.csv"
        out_best_w_path = out_dir / f"chembl_{args.assay_type}_best_per_protein_w_val.csv"
        
        # Output 1: All entries
        final.to_csv(out_all_path, index=False)
        print(f"✅ Saved all entries: {out_all_path} ({len(final)} rows)")
        
        # Output 2: Entries with PDB structures
        targets_with_pdb = final[
            (final["target_type"] == "target") &
            final["pdb_ids"].notna() & 
            (final["pdb_ids"].str.strip() != "") &
            (final["pdb_ids"].str.strip() != "nan")
        ]
        proteins_with_target_pdb = set(targets_with_pdb['protein_key'].unique())
        final_with_pdb = final[final['protein_key'].isin(proteins_with_target_pdb)].copy()
        
        final_with_pdb.to_csv(out_pdb_path, index=False)
        print(f"✅ Saved PDB entries: {out_pdb_path} ({len(final_with_pdb)} rows)")
        print(f"  - Proteins with at least one target PDB: {len(proteins_with_target_pdb)}")
        
        # Output 3: Best ligand per protein (without PDB requirement)
        best_target_idx = final[final["target_type"] == "target"].groupby("protein_key")["representative_assay_nM"].idxmin()
        best_off_idx = final[final["target_type"] == "off_target"].groupby("protein_key")["representative_assay_nM"].idxmax()
        
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
            final_with_target_pdb["pdb_ids"].notna() & 
            (final_with_target_pdb["pdb_ids"].str.strip() != "") &
            (final_with_target_pdb["pdb_ids"].str.strip() != "nan")
        ]
        best_target_idx = targets_pdb_only.groupby("protein_key")["representative_assay_nM"].idxmin()
        
        off_targets_all = final_with_target_pdb[final_with_target_pdb["target_type"] == "off_target"]
        best_off_idx = off_targets_all.groupby("protein_key")["representative_assay_nM"].idxmax()
        
        best_per_protein_w = pd.concat([
            targets_pdb_only.loc[best_target_idx],
            off_targets_all.loc[best_off_idx]
        ], axis=0).sort_values(["protein_key", "target_type"]).reset_index(drop=True)
        
        best_per_protein_w.to_csv(out_best_w_path, index=False)
        print(f"✅ Saved best ligands per protein (target with PDB): {out_best_w_path} ({len(best_per_protein_w)} rows)")
        print(f"  - Proteins: {best_per_protein_w['protein_key'].nunique()}")
        
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
        print(f"\nTarget distribution:")
        print(final['target_type'].value_counts())
        
        print(f"\n{args.assay_type} statistics (representative values):")
        print(f"  - Target (mean ± std): {targets.mean():.2f} ± {targets.std():.2f} nM")
        print(f"  - Off-target (mean ± std): {off_targets.mean():.2f} ± {off_targets.std():.2f} nM")
        
        # Update metadata with final statistics
        end_time = datetime.now()
        
        metadata["statistics"]["final"] = {
            "unique_proteins": int(final['protein_key'].nunique()),
            "unique_ligands": int(final['ligand_inchikey'].nunique()),
            "total_pairs": len(final),
            "with_pdb": len(final_with_pdb),
            "targets": int((final['target_type'] == 'target').sum()),
            "off_targets": int((final['target_type'] == 'off_target').sum()),
            "proteins_with_both": int(len(proteins_with_both)),
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
            "histogram": str(hist_path)
        }
        
        metadata["execution_time"] = {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds()
        }
        
        # Save metadata
        metadata_path = out_dir / f"chembl_{args.assay_type}_curation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\n✅ Saved metadata: {metadata_path}")
        
    finally:
        conn.close()
        print("\nDatabase connection closed.")

if __name__ == "__main__":
    main()