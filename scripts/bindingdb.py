# %%
# Description:
#  - Load BindingDB TSV and sort by Target Name
#  - Detect different sequences within the same Target Name
#  - Summarize and save results to CSV
# Outputs:
#  - 01 bindingdb_sorted.csv
#  - 02 protein_keys_with_variants.csv
#  - 03 sequence_variant_details.csv
#  - 04 substitutions_only.csv
#  - 05 substitutions_only_available.csv
#  - 06 substitutions_only_available_group.csv
#  - 07 reference_sequences.fasta
#  - 08 substitutions_only_final.csv
#  - 09 stats.csv
#  - 10 find_mutation.csv
# %% Import packages
import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from typing import List, Optional
from difflib import SequenceMatcher
import argparse
# %% Path setting
DATA_DIR = Path("/Users/yunmin/Desktop/aipdl/data/db")
TSV_PATH = Path(DATA_DIR/"BindingDB_BindingDB_Articles.tsv")

df = pd.read_csv(TSV_PATH, sep="\t", engine="python", on_bad_lines="skip")
# %% Functions
def clean_seq(seq: str) -> str:
    if pd.isna(seq):
        return ""
    return "".join(str(seq).split()).upper()

def md5_of(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def norm_text(x: str) -> str:
    if pd.isna(x) or x is None:
        return ""
    return " ".join(str(x).strip().split())

def build_protein_key(row: pd.Series) -> str:
    swiss_primary = norm_text(row.get("UniProt (SwissProt) Primary ID of Target Chain", ""))
    swiss_entry   = norm_text(row.get("UniProt (SwissProt) Entry Name of Target Chain", ""))
    trembl_primary= norm_text(row.get("UniProt (TrEMBL) Primary ID of Target Chain", ""))
    swiss_recname = norm_text(row.get("UniProt (SwissProt) Recommended Name of Target Chain", ""))
    target_name   = norm_text(row.get("Target Name", ""))
    organism      = norm_text(row.get("Target Source Organism According to Curator or DataSource", ""))

    ident = swiss_primary or swiss_entry or trembl_primary or swiss_recname or target_name or "UNSPECIFIED_TARGET"
    org   = organism or "UNSPECIFIED_ORG"
    return f"{ident} | {org}"

def align_diff(reference: str, variant: str):
    """Return (diffs_string, n_subs, n_ins, n_del)."""
    if reference == variant:
        return "", 0, 0, 0

    diffs = []
    subs = ins = dels = 0

    sm = SequenceMatcher(a=reference, b=variant)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "replace":
            ref_block = reference[i1:i2]
            var_block = variant[j1:j2]
            m = min(len(ref_block), len(var_block))
            # substitutions
            for k in range(m):
                r = ref_block[k]
                v = var_block[k]
                if r != v:
                    pos = i1 + k + 1
                    diffs.append(f"{r}{pos}{v}")
                    subs += 1
            # deletions
            if len(ref_block) > m:
                start = i1 + m + 1
                end = i2
                if end >= start:
                    if end > start:
                        diffs.append(f"del{start}-{end}")
                    else:
                        diffs.append(f"del{start}")
                    dels += (end - start + 1)
            # insertions
            if len(var_block) > m:
                ins_after = i1 + m
                ins_seq = var_block[m:]
                diffs.append(f"ins{ins_after}:{ins_seq}")
                ins += len(ins_seq)
        elif tag == "delete":
            start = i1 + 1
            end = i2
            if end >= start:
                if end > start:
                    diffs.append(f"del{start}-{end}")
                else:
                    diffs.append(f"del{start}")
                dels += (end - start + 1)
        elif tag == "insert":
            ins_seq = variant[j1:j2]
            diffs.append(f"ins{i1}:{ins_seq}")
            ins += len(ins_seq)
    return "; ".join(diffs), subs, ins, dels

def to_item_set(s):
    if not isinstance(s, str):
        return set()
    out = []
    for part in s.split(";"):
        out.extend(part.split(","))
    return set(x.strip() for x in out if x and isinstance(x, str) and x.strip())

def ligand_same_as_ref_for_nonref(row):
    if bool(row.get("is_reference", False)):
        return np.nan
    var_set = to_item_set(row.get("ligands", np.nan))
    ref_set = to_item_set(row.get("ref_ligands", np.nan))
    if len(ref_set) == 0:
        return np.nan
    return var_set == ref_set

def uniq_join(series, max_items=50):
    vals = [str(v).strip() for v in series.dropna().tolist() if str(v).strip()]
    if not vals:
        return np.nan
    uniq = pd.unique(vals)[:max_items]
    return "; ".join(uniq)
# %% Dataframe processing
rename_candidates = {
    "Target Name": ["Target Name", "Target (Pref Name)", "TargetName"],
    "BindingDB Target Chain Sequence": ["BindingDB Target Chain Sequence", "Target Chain Sequence", "Target Sequence"],
    "Target Source Organism According to Curator or DataSource": [
        "Target Source Organism According to Curator or DataSource", "Target Source Organism"
    ],
    "UniProt (SwissProt) Primary ID of Target Chain": ["UniProt (SwissProt) Primary ID of Target Chain"],
    "UniProt (SwissProt) Entry Name of Target Chain": ["UniProt (SwissProt) Entry Name of Target Chain"],
    "UniProt (SwissProt) Recommended Name of Target Chain": ["UniProt (SwissProt) Recommended Name of Target Chain"],
    "UniProt (TrEMBL) Primary ID of Target Chain": ["UniProt (TrEMBL) Primary ID of Target Chain"],
    "PDB ID(s) of Target Chain": ["PDB ID(s) of Target Chain", "PDB ID(s)"],
    "Binding DB Ligand Name": ["Binding DB Ligand Name", "Ligand Name", "BindingDB Ligand Name"],
    "Ligand SMILES": ["Ligand SMILES", "SMILES", "Canonical SMILES"],
    "Article DOI": ["Article DOI", "DOI", "Document DOI"],
}

cols_lower = {c.lower(): c for c in df.columns}
rename_map = {}
for std, cands in rename_candidates.items():
    for cand in cands:
        if cand.lower() in cols_lower:
            rename_map[cols_lower[cand.lower()]] = std
            break

df = df.rename(columns=rename_map)

for col in rename_candidates.keys():
    if col not in df.columns:
        df[col] = np.nan

df["Target Name (norm)"] = df["Target Name"].map(norm_text)
df["Sequence (clean)"] = df["BindingDB Target Chain Sequence"].map(clean_seq)
df["Sequence MD5"] = df["Sequence (clean)"].map(md5_of)
df["Sequence Length"] = df["Sequence (clean)"].map(len)
df["Organism (norm)"] = df["Target Source Organism According to Curator or DataSource"].map(norm_text)
df["Protein Key"] = df.apply(build_protein_key, axis=1)

df_sorted = df.sort_values(["Protein Key", "Binding DB Ligand Name", "Article DOI"], na_position="last")
out_sorted = DATA_DIR / "01_bindingdb_sorted.csv"
df_sorted.to_csv(out_sorted, index=False)
print(f"Sorted by Protein Key: {out_sorted}")
# %% Protein keys with variants
seq_counts = (
    df_sorted.groupby(["Protein Key", "Sequence MD5"], dropna=False)
    .agg(
        n_records=("Sequence MD5", "size"),
        seq_length=("Sequence Length", "max"),
        example_seq=("Sequence (clean)", "first"),
        target_name=("Target Name (norm)", "first"),
        organism=("Organism (norm)", "first"),
    )
    .reset_index()
)

seq_counts["rank_within_key"] = (
    seq_counts
    .sort_values(["Protein Key", "n_records"], ascending=[True, False])
    .groupby("Protein Key")["n_records"]
    .rank(method="first", ascending=False)
)
seq_counts["is_reference"] = seq_counts["rank_within_key"].eq(1)

key_summary = (
    seq_counts.groupby("Protein Key", dropna=False)
    .agg(
        n_total_records=("n_records", "sum"),
        n_unique_sequences=("Sequence MD5", "nunique"),
        reference_seq_md5=("Sequence MD5", lambda s: s.loc[seq_counts.loc[s.index, "is_reference"]].values[0] if (seq_counts.loc[s.index, "is_reference"]).any() else np.nan),
        organism=("organism", "first"),
    )
    .reset_index()
)
out_key_variants = DATA_DIR / "02_protein_keys_with_variants.csv"
key_summary.to_csv(out_key_variants, index=False)
print(f"Protein keys with variants: {out_key_variants}")
# %% Sequence variant details
variant_details = (
    df_sorted.groupby(["Protein Key", "Sequence MD5"], dropna=False)
    .agg(
        n_records=("Sequence MD5", "size"),
        sequence_length=("Sequence Length", "max"),
        ligands=("Binding DB Ligand Name", lambda s: "; ".join(pd.unique([v for v in s.dropna().astype(str) if v.strip()])[:15])),
        smiles=("Ligand SMILES", lambda s: "; ".join(pd.unique([v for v in s.dropna().astype(str) if v.strip()])[:5])),
        pdb_ids=("PDB ID(s) of Target Chain", lambda s: "; ".join(pd.unique("; ".join([p.strip() for v in s.dropna().astype(str) for p in v.split(",")]).split("; "))[:10])),
        dois=("Article DOI", lambda s: "; ".join(pd.unique([v for v in s.dropna().astype(str) if v.strip()])[:10])),
        example_seq=("Sequence (clean)", "first"),
        target_name=("Target Name (norm)", "first"),
        organism=("Organism (norm)", "first"),
    )
    .reset_index()
    .merge(seq_counts[["Protein Key", "Sequence MD5", "is_reference"]], on=["Protein Key", "Sequence MD5"], how="left")
)

ref_by_key = (
    seq_counts.loc[seq_counts["is_reference"], ["Protein Key", "example_seq"]]
    .set_index("Protein Key")["example_seq"]
    .to_dict()
)

diffs_list = []
nsubs_list = []
nins_list = []
ndel_list = []

for _, row in variant_details.iterrows():
    key = row["Protein Key"]
    seq = row["example_seq"] or ""
    ref = ref_by_key.get(key, "")
    if not ref or not seq:
        diffs, ns, ni, nd = "", 0, 0, 0
    else:
        diffs, ns, ni, nd = align_diff(ref, seq)
    diffs_list.append(diffs)
    nsubs_list.append(ns)
    nins_list.append(ni)
    ndel_list.append(nd)

variant_details["key_mutations"] = diffs_list
variant_details["n_subs"] = nsubs_list
variant_details["n_ins"] = nins_list
variant_details["n_del"] = ndel_list

out_variant_diffs = DATA_DIR / "03_sequence_variant_details.csv"
variant_details_sorted = variant_details.sort_values(
    ["Protein Key", "is_reference", "n_records"],
    ascending=[True, False, False]
)
variant_details_sorted.to_csv(out_variant_diffs, index=False)
print(f"Variant details: {out_variant_diffs}")
# %% Substitutions only
subs_only = variant_details_sorted[
    (variant_details_sorted["n_ins"] == 0) &
    (variant_details_sorted["n_del"] == 0) &
    (variant_details_sorted["n_subs"] > 0)
].copy()
subs_ref = variant_details_sorted[
    (variant_details_sorted["Protein Key"].isin(subs_only["Protein Key"])) &
    (variant_details_sorted["is_reference"]) &
    (variant_details_sorted["target_name"])
].copy()
subs_only_ref = pd.concat([subs_only, subs_ref], ignore_index=True)

subs_info = variant_details_sorted[
    variant_details_sorted["is_reference"]][
        ["Protein Key", "ligands", "pdb_ids", "smiles"]
    ].rename(columns={"ligands": "ref_ligands", "pdb_ids": "ref_pdb_ids", "smiles": "ref_smiles"})
subs_only_ref = subs_only_ref.merge(subs_info, on="Protein Key", how="left")

subs_only_ref["ligand_same_as_ref_for_nonref"] = subs_only_ref.apply(ligand_same_as_ref_for_nonref, axis=1)

subs_only_ref["has_pdb_ids"] = subs_only_ref["pdb_ids"].apply(
    lambda s: len(to_item_set(s)) > 0
)

aff_cols = ["Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)"]
aff_agg = (
    df_sorted
    .groupby(["Protein Key", "Sequence MD5"], dropna=False)[aff_cols]
    .agg(uniq_join)
    .reset_index()
)

subs_only_ref = subs_only_ref.merge(aff_agg, on=["Protein Key", "Sequence MD5"], how="left")

keep_cols = [
    "Protein Key", "target_name", "organism",
    "Sequence MD5", "example_seq", "is_reference",
    "key_mutations", "n_subs", "n_ins", "n_del",
    "ligands", "ref_ligands",
    "smiles", "ref_smiles",
    "pdb_ids", "ref_pdb_ids", "has_pdb_ids",
    "ligand_same_as_ref_for_nonref",
] + aff_cols

out_subs_only = DATA_DIR / "04_substitutions_only.csv"
subs_only_ref.loc[:, keep_cols].sort_values(
    ["Protein Key", "is_reference"],
    ascending=[True, False]
).to_csv(out_subs_only, index=False)
print(f"Substitutions only: {out_subs_only}")
# %% Available substitutions only
def has_any_affinity(row):
    for c in aff_cols:
        v = row.get(c)
        if isinstance(v, str) and v.strip():
            return True
        if pd.notna(v):
            try:
                if any(tok.strip() for tok in str(v).split(";")):
                    return True
            except Exception:
                return True
    return False

nonref = subs_only_ref[subs_only_ref["is_reference"] == False].copy()

cand_nonref = nonref[
    (nonref["ligand_same_as_ref_for_nonref"] == False) &
    (nonref["has_pdb_ids"] == True)
].copy()

cand_nonref = cand_nonref[cand_nonref.apply(has_any_affinity, axis=1)].copy()

cand_nonref = cand_nonref[
    cand_nonref["ref_ligands"].notna() &
    (cand_nonref["ref_ligands"].astype(str).str.strip() != "")
].copy()

ref_rows = subs_only_ref[
    (subs_only_ref["Protein Key"].isin(cand_nonref["Protein Key"])) &
    (subs_only_ref["is_reference"] == True)
].copy()

pairs = pd.concat([cand_nonref, ref_rows], ignore_index=True)

keep_cols = [
    "Protein Key", "target_name", "organism", 
    "Sequence MD5", "example_seq", "is_reference",
    "key_mutations", "n_subs", "n_ins", "n_del",
    "ligands", "ref_ligands", 
    "smiles", "ref_smiles",
    "pdb_ids", "ref_pdb_ids", "has_pdb_ids",
] + aff_cols

pairs = pairs.loc[:, [c for c in keep_cols if c in pairs.columns]].sort_values(
    ["Protein Key", "is_reference"], ascending=[True, False]
)

out_subs_only_available = DATA_DIR / "05_substitutions_only_available.csv"
pairs.to_csv(out_subs_only_available, index=False)
print(f"Available Substitutions only: {out_subs_only_available}")
# %% Substitutions with available (protein, sequence, ligand) group
from collections import Counter
def most_frequent(series):
    vals = [str(v).strip() for v in series.dropna().tolist() if str(v).strip()]
    if not vals:
        return np.nan
    c = Counter(vals)
    return c.most_common(1)[0][0]

def pick_ligand_key_rowwise(df):
    lk = df.get("Ligand InChI Key")
    if lk is None:
        lk = pd.Series([np.nan]*len(df))
    li = df.get("Ligand InChI")
    ls = df.get("Ligand SMILES")
    ln = df.get("Binding DB Ligand Name")
    ligand_key = []
    for i in range(len(df)):
        key = None
        for cand in (lk.iloc[i] if lk is not None else None,
                     li.iloc[i] if li is not None else None,
                     ls.iloc[i] if ls is not None else None,
                     ln.iloc[i] if ln is not None else None):
            if isinstance(cand, str) and cand.strip():
                key = cand.strip()
                break
        ligand_key.append(key if key is not None else "")
    return pd.Series(ligand_key, index=df.index, name="Ligand Key")

need_reconstruct = False
if 'df_sorted' not in globals() or 'subs_plus_refs' not in globals():
    need_reconstruct = True

if need_reconstruct:
    tsvs = sorted(DATA_DIR.glob("*.tsv"))
    csvs = sorted(DATA_DIR.glob("*.csv"))
    src = None
    if tsvs:
        src = max(tsvs, key=lambda p: p.stat().st_size)
        df_raw = pd.read_csv(src, sep="\t", engine="python", on_bad_lines="skip")
    elif csvs:
        src = max(csvs, key=lambda p: p.stat().st_size)
        df_raw = pd.read_csv(src, engine="python")
    else:
        raise FileNotFoundError("Could not find any TSV/CSV in /mnt/data to reconstruct inputs.")

    rename_map = {}
    def map_first(df, targets):
        cols_lower = {c.lower(): c for c in df.columns}
        for t in targets:
            if t.lower() in cols_lower:
                return cols_lower[t.lower()]
        return None

    want = {
        "Target Name": ["Target Name", "Target (Pref Name)"],
        "BindingDB Target Chain Sequence": ["BindingDB Target Chain Sequence", "Target Chain Sequence", "Target Sequence"],
        "Target Source Organism According to Curator or DataSource": ["Target Source Organism According to Curator or DataSource", "Target Source Organism"],
        "Ligand SMILES": ["Ligand SMILES", "SMILES", "Canonical SMILES"],
        "Binding DB Ligand Name": ["Binding DB Ligand Name", "Ligand Name", "BindingDB Ligand Name"],
        "Ligand InChI": ["Ligand InChI"],
        "Ligand InChI Key": ["Ligand InChI Key"],
        "Ki (nM)": ["Ki (nM)"],
        "IC50 (nM)": ["IC50 (nM)"],
        "Kd (nM)": ["Kd (nM)"],
        "EC50 (nM)": ["EC50 (nM)"],
        "Target Name": ["Target Name", "Target (Pref Name)", "TargetName"],
        "BindingDB Target Chain Sequence": ["BindingDB Target Chain Sequence", "Target Chain Sequence", "Target Sequence"],
        "Target Source Organism According to Curator or DataSource": [
            "Target Source Organism According to Curator or DataSource", "Target Source Organism"
        ],
        "UniProt (SwissProt) Primary ID of Target Chain": ["UniProt (SwissProt) Primary ID of Target Chain"],
        "UniProt (SwissProt) Entry Name of Target Chain": ["UniProt (SwissProt) Entry Name of Target Chain"],
        "UniProt (SwissProt) Recommended Name of Target Chain": ["UniProt (SwissProt) Recommended Name of Target Chain"],
        "UniProt (TrEMBL) Primary ID of Target Chain": ["UniProt (TrEMBL) Primary ID of Target Chain"],
        "PDB ID(s) of Target Chain": ["PDB ID(s) of Target Chain", "PDB ID(s)"],
        "Article DOI": ["Article DOI", "DOI", "Document DOI"],
    }

    for std, cands in want.items():
        got = map_first(df_raw, cands)
        if got and got != std:
            rename_map[got] = std
    df_sorted = df_raw.rename(columns=rename_map).copy()
    df_sorted["Target Name"] = df_sorted.get("Target Name", pd.Series(dtype=str)).astype(str)
    df_sorted["Organism (norm)"] = df_sorted.get("Target Source Organism According to Curator or DataSource", pd.Series(dtype=str)).astype(str)
    df_sorted["Sequence (clean)"] = df_sorted.get("BindingDB Target Chain Sequence", pd.Series(dtype=str)).astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    df_sorted["Sequence MD5"] = df_sorted["Sequence (clean)"].map(lambda s: pd.util.hash_pandas_object(pd.Series([s])).astype(str).iloc[0])
    df_sorted["Protein Key"] = df_sorted.apply(build_protein_key, axis=1)
    subs_plus_refs = None

df_sorted["Ligand Key"] = pick_ligand_key_rowwise(df_sorted)

if subs_plus_refs is not None and "Protein Key" in subs_plus_refs.columns and "Sequence MD5" in subs_plus_refs.columns:
    allowed = subs_plus_refs[["Protein Key", "Sequence MD5"]].drop_duplicates()
    df_use = df_sorted.merge(allowed, on=["Protein Key", "Sequence MD5"], how="inner")
else:
    df_use = df_sorted.copy()

aff_cols = ["Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)"]
present_aff = [c for c in aff_cols if c in df_use.columns]

group_cols = ["Protein Key", "Sequence MD5", "Ligand Key"]
agg_spec = {
    "Target Name": most_frequent,
    "Organism (norm)": most_frequent,
    "Binding DB Ligand Name": most_frequent,
    "Ligand SMILES": most_frequent,
    "Ligand InChI": most_frequent,
    "Ligand InChI Key": most_frequent,
    "BindingDB Target Chain Sequence": most_frequent,
}
for c in present_aff:
    agg_spec[c] = uniq_join

df_grouped = (
    df_use
    .groupby(group_cols, dropna=False)
    .agg(agg_spec)
    .reset_index()
)

subs_only_group = DATA_DIR / "06_substitutions_only_available_group.csv"
sort_cols = ["Protein Key", "Sequence MD5", "Ligand Key"]
df_grouped = df_grouped.sort_values(sort_cols).reset_index(drop=True)
df_grouped.to_csv(subs_only_group, index=False)
print(f"Grouped available substitutions only: {subs_only_group}")
# %% Ref sequences FASTA file
fasta_path = DATA_DIR / "07_reference_sequences.fasta"
with open(fasta_path, "w") as fh:
    for key, seq in ref_by_key.items():
        if seq:
            header = key.replace(" ", "_")[:100]
            fh.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i+80] + "\n")
print(f"Reference FASTA: {fasta_path}")
# %% Grouped and substitution only csv with affinities
rename_candidates = {
    "Target Name": ["Target Name", "Target (Pref Name)", "TargetName"],
    "BindingDB Target Chain Sequence": ["BindingDB Target Chain Sequence", "Target Chain Sequence", "Target Sequence"],
    "Target Source Organism According to Curator or DataSource": [
        "Target Source Organism According to Curator or DataSource", "Target Source Organism"
    ],
    "UniProt (SwissProt) Primary ID of Target Chain": ["UniProt (SwissProt) Primary ID of Target Chain"],
    "UniProt (SwissProt) Entry Name of Target Chain": ["UniProt (SwissProt) Entry Name of Target Chain"],
    "UniProt (SwissProt) Recommended Name of Target Chain": ["UniProt (SwissProt) Recommended Name of Target Chain"],
    "UniProt (TrEMBL) Primary ID of Target Chain": ["UniProt (TrEMBL) Primary ID of Target Chain"],
    "PDB ID(s) of Target Chain": ["PDB ID(s) of Target Chain", "PDB ID(s)"],
    "Binding DB Ligand Name": ["Binding DB Ligand Name", "Ligand Name", "BindingDB Ligand Name"],
    "Ligand SMILES": ["Ligand SMILES", "SMILES", "Canonical SMILES"],
    "Ligand InChI": ["Ligand InChI"],
    "Ligand InChI Key": ["Ligand InChI Key"],
    "Article DOI": ["Article DOI", "DOI", "Document DOI"],
    "Ki (nM)": ["Ki (nM)"],
    "IC50 (nM)": ["IC50 (nM)"],
    "Kd (nM)": ["Kd (nM)"],
    "EC50 (nM)": ["EC50 (nM)"],
}
cols_lower = {c.lower(): c for c in df.columns}
rename_map = {}
for std, cands in rename_candidates.items():
    for cand in cands:
        if cand.lower() in cols_lower:
            rename_map[cols_lower[cand.lower()]] = std
            break
df = df.rename(columns=rename_map)
for col in rename_candidates.keys():
    if col not in df.columns:
        df[col] = np.nan

df["Target Name (norm)"] = df["Target Name"].map(norm_text)
df["Organism (norm)"] = df["Target Source Organism According to Curator or DataSource"].map(norm_text)
df["Sequence (clean)"] = df["BindingDB Target Chain Sequence"].map(clean_seq)
df["Sequence MD5"] = df["Sequence (clean)"].map(md5_of)
df["Protein Key"] = df.apply(build_protein_key, axis=1)
df["Ligand Key"] = pick_ligand_key_rowwise(df)

seq_counts = (
    df.groupby(["Protein Key","Sequence MD5"], dropna=False)
      .size().reset_index(name="n_records")
)
ref_idx = seq_counts.sort_values(["Protein Key","n_records"], ascending=[True,False]) \
                    .groupby("Protein Key").head(1)[["Protein Key","Sequence MD5"]]
ref_map = df.merge(ref_idx, on=["Protein Key","Sequence MD5"], how="inner") \
            .drop_duplicates(subset=["Protein Key","Sequence MD5"]) \
            .set_index("Protein Key")["Sequence (clean)"].to_dict()

uniq_seq = df.drop_duplicates(subset=["Protein Key","Sequence MD5"])[
    ["Protein Key","Sequence MD5","Sequence (clean)"]
].copy()
diffs, nsubs, nins, ndel = [], [], [], []
for _, r in uniq_seq.iterrows():
    ref = ref_map.get(r["Protein Key"], "")
    variant = r["Sequence (clean)"] or ""
    d, ns, ni, nd = align_diff(ref, variant) if ref else ("",0,0,0)
    diffs.append(d); nsubs.append(ns); nins.append(ni); ndel.append(nd)
uniq_seq["key_mutations"] = diffs
uniq_seq["n_subs"] = nsubs
uniq_seq["n_ins"] = nins
uniq_seq["n_del"] = ndel
uniq_seq["is_reference"] = False
uniq_seq.loc[
    uniq_seq.set_index(["Protein Key","Sequence MD5"]).index.isin(
        ref_idx.set_index(["Protein Key","Sequence MD5"]).index
    ),
    "is_reference"
] = True
uniq_seq["example_seq"] = uniq_seq["Sequence (clean)"]

df_ann = df.merge(
    uniq_seq[["Protein Key","Sequence MD5","is_reference","example_seq","key_mutations","n_subs","n_ins","n_del"]],
    on=["Protein Key","Sequence MD5"], how="left"
)

aff_cols = [c for c in ["Ki (nM)","IC50 (nM)","Kd (nM)","EC50 (nM)"] if c in df_ann.columns]
group_cols = ["Protein Key","Sequence MD5","Ligand Key"]

agg_dict = {
    "target_name": ("Target Name (norm)", "first"),
    "organism": ("Organism (norm)", "first"),
    "example_seq": ("example_seq","first"),
    "ligand": ("Binding DB Ligand Name","first"),
    "ligand_smiles": ("Ligand SMILES","first"),
    "is_reference": ("is_reference","first"),
    "key_mutations": ("key_mutations","first"),
    "n_subs": ("n_subs","first"),
    "n_ins": ("n_ins","first"),
    "n_del": ("n_del","first"),
    "pdb_ids": ("PDB ID(s) of Target Chain", uniq_join),
}
for c in aff_cols:
    agg_dict[c] = (c, uniq_join)

grouped = (
    df_ann
    .groupby(group_cols, dropna=False)
    .agg(**{new: pd.NamedAgg(column=col, aggfunc=fn) for new,(col,fn) in agg_dict.items()})
    .reset_index()
)

grouped = grouped.sort_values(
    ["Protein Key","Sequence MD5","Ligand Key","is_reference"],
    ascending=[True,True,True,False]
)
selected_grouped = grouped.drop(columns=['n_ins', 'n_del'])
selected_grouped = selected_grouped.drop(columns=group_cols[1:-1])
selected_grouped = selected_grouped[
    selected_grouped['pdb_ids'].notna() &
    selected_grouped['pdb_ids'].astype(str).str.strip().ne("")
    ]
substitution_final_path = DATA_DIR / "08_substitution_final.csv"
selected_grouped.to_csv(substitution_final_path, index=False)
print(f"Grouped available substitutions only (final): {substitution_final_path}")
# %% Per-protein stats
g = grouped.copy()
g = g.drop(columns=['n_ins', 'n_del'])

required = ["Protein Key", "Sequence MD5", "Ligand Key", "is_reference", "n_subs"]
missing = [c for c in required if c not in g.columns]
if missing:
    raise RuntimeError(f"`grouped` is missing required columns: {missing}")

meta = (
    g.sort_values(["Protein Key"])
     .groupby("Protein Key", dropna=False)
     .agg(
         target_name=("target_name", uniq_join) if "target_name" in g.columns else ("Protein Key", "first"),
         organism=("organism", uniq_join) if "organism" in g.columns else ("Protein Key", "first")
     )
     .reset_index()
)

seq_total = g.groupby("Protein Key")["Sequence MD5"].nunique().rename("n_sequences_total")
seq_mut = (
    g[(g["is_reference"] == False) & (g["n_subs"].fillna(0) > 0)]
    .groupby("Protein Key")["Sequence MD5"].nunique()
    .rename("n_sequences_mut")
)
lig_ref = (
    g[g["is_reference"] == True]
    .groupby("Protein Key")["Ligand Key"].nunique()
    .rename("n_ligands_ref")
)
lig_mut = (
    g[(g["is_reference"] == False) & (g["n_subs"].fillna(0) > 0)]
    .groupby("Protein Key")["Ligand Key"].nunique()
    .rename("n_ligands_mut")
)
lig_total = g.groupby("Protein Key")["Ligand Key"].nunique().rename("n_ligands_total")

stats = meta.set_index("Protein Key") \
            .join([seq_total, seq_mut, lig_ref, lig_mut, lig_total]) \
            .fillna(0)

for col in ["n_sequences_total","n_sequences_mut","n_ligands_ref","n_ligands_mut","n_ligands_total"]:
    stats[col] = stats[col].astype(int)
stats = stats.reset_index()

out_stats = DATA_DIR / "09_stats.csv"
stats.to_csv(out_stats, index=False)
print(f"Stats: {out_stats}")
# %% Find mutation info
g = grouped.copy()

def pick_col(df, cands, default=None):
    for c in cands:
        if c in df.columns:
            return c
    return default

g["has_mut"] = g["n_subs"].fillna(0) > 0
g = g[g["n_subs"].fillna(0) < 30]
g = g[g["n_ins"].fillna(0) == 0]
g = g[g["n_del"].fillna(0) == 0]

grp = (
    g.groupby(["Protein Key","Ligand Key"], dropna=False)
     .agg(
         n_sequences=("Sequence MD5","nunique"),
         any_mut=("has_mut","any")
     )
     .reset_index()
)
keep_keys = grp[(grp["n_sequences"] > 1) & (grp["any_mut"])][["Protein Key","Ligand Key"]]
out = g.merge(keep_keys, on=["Protein Key","Ligand Key"], how="inner")
out = out.sort_values(
    ["Protein Key","Ligand Key","is_reference","Sequence MD5"],
    ascending=[True, True, False, True]
)

cols = [
    "Protein Key",
    pick_col(out, ["target_name","Target Name","Target Name (norm)"]),
    # "Ligand Key",
    pick_col(out, ["ligand", "ligand_name","Binding DB Ligand Name"]),
    pick_col(out, ["ligand_smiles", "smiles","Ligand SMILES"]),
    # "Sequence MD5",
    pick_col(out, ["example_seq","Sequence (clean)"]),
    "is_reference",
    pick_col(out, ["key_mutations","diffs_vs_reference"]),
    "n_subs",
    # pick_col(out, ["n_ins"], default=None),
    # pick_col(out, ["n_del"], default=None),
    pick_col(out, ["Ki (nM)"], default=None),
    pick_col(out, ["IC50 (nM)"], default=None),
    pick_col(out, ["Kd (nM)"], default=None),
    pick_col(out, ["EC50 (nM)"], default=None),
    pick_col(out, ["pdb_ids","PDB ID(s) of Target Chain"]),
]
cols = [c for c in cols if c is not None and c in out.columns]

find_mutation_path = DATA_DIR / "10_find_mutation.csv"
out.loc[:, cols].to_csv(find_mutation_path, index=False)
print(f"Sequence mutation data: {find_mutation_path}")
