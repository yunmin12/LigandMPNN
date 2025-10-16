# %%
import pandas as pd
import numpy as np
import requests, re
import argparse
import hashlib, ast
import csv, json
from collections import Counter
from pathlib import Path

# ---------- Config ----------
p = argparse.ArgumentParser(description="BindingDB curation set for ligand specificity evaluation.")
p.add_argument("--out", dest="output_path", required=True, help="Output CSV path")
p.add_argument("--ensure_pdb", dest="ensure_pdb", required=False, default=False, help="Ensure all the ligands to have complex PDB ids. If not, leave ligands with at least one complex PDB ids in the same protein. Default: False.")
p.add_argument("--same_mutation", dest="same_mutation", required=False, default=False, help="Remains off-targets that only has the same key_mutation with the target. Default: False.")
args = p.parse_args()

INPUT_TSV = "/home/yunmin/proj/data/db/BindingDB_BindingDB_Articles.tsv"
OUTPUT_CSV = args.output_path
# OUTPUT_CSV = "/home/yunmin/proj/data/db/15_eval_set_1015.csv"
SOLVENT_IDS = {
    "HOH","WAT","DOD",  # water
    "CL","NA","K","CA","MG","ZN","MN","CO","CU","NI","IOD",  # common ions
    "SO4","PO4",  # anions
    "GOL","EDO","PEG","PG4","MPD","TRS","MES","ACE","IPA","BME","FMT"  # common buffers/additives
}
# ---------- Utility ----------
def md5_seq(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()

def clean_nM(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace(",", "")
    if s.startswith((">", "<")):
        s = s[1:].strip()
    try:
        v = float(s)
        return np.nan if v <= 0 else v
    except:
        return np.nan

def to_pX(nM: float) -> float:
    if pd.isna(nM): return np.nan
    try:
        val = float(nM)
        if val <= 0: return np.nan
        return -np.log10(val * 1e-9)  # nM -> M
    except:
        return np.nan

def median_ignore_nan(values: list) -> float:
    vals = [v for v in values if pd.notna(v)]
    return np.nan if not vals else float(np.median(vals))

def first_nonnull(x: str) -> str:
    for v in x:
        if pd.notna(v) and str(v).strip() != "":
            return v
    return np.nan

# ---------- Load & Clean ----------
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
df = df.rename(columns=lambda x: x.strip())
df.columns = df.columns.str.strip()

# Select useful columns
cols = ["BindingDB Reactant_set_id","Ligand SMILES","Ligand InChI Key",
        "BindingDB Ligand Name","Target Name",
        "Target Source Organism According to Curator or DataSource",
        "Ki (nM)","Kd (nM)","IC50 (nM)","EC50 (nM)",
        "PDB ID(s) for Ligand-Target Complex",
        "BindingDB Target Chain Sequence",
        "UniProt (SwissProt) Primary ID of Target Chain"]
df = df[[c for c in cols if c in df.columns]].copy()

df["BindingDB Target Chain Sequence"] = df["BindingDB Target Chain Sequence"].fillna("").str.strip()
df = df[df["BindingDB Target Chain Sequence"].str.len() > 0]
# for col in ["Ki (nM)","Kd (nM)","IC50 (nM)","EC50 (nM)"]:
#     if col in df.columns:
#         df[col] = df[col].apply(clean_nM)
# %%
# ---------- Compute pX values ----------
for col in ["Ki (nM)","Kd (nM)","IC50 (nM)","EC50 (nM)"]:
    if col in df.columns:
        df[col+"_pX"] = df[col].apply(to_pX)

# Melt into tidy format
melted = []
for atype in ["Ki","Kd","IC50","EC50"]:
    nmcol = f"{atype} (nM)"
    pxcol = f"{atype} (nM)_pX"
    if nmcol not in df: continue
    sub = df[["BindingDB Reactant_set_id","Ligand SMILES","Ligand InChI Key",
              "BindingDB Ligand Name","Target Name",
              "Target Source Organism According to Curator or DataSource",
              nmcol, pxcol, "BindingDB Target Chain Sequence",
              "UniProt (SwissProt) Primary ID of Target Chain",
              "PDB ID(s) for Ligand-Target Complex"]].copy()
    sub["assay_type"] = atype
    sub = sub.rename(columns={nmcol:"affinity_nM", pxcol:"pX"})
    melted.append(sub)
df = pd.concat(melted, ignore_index=True)
# df = df[pd.to_numeric(df["affinity_nM"], errors="coerce").notna()].copy()
# df["pX"] = df["pX"].astype(float)
# df = df[df["pX"].notna()].copy()

# ---------- Median pX per unique condition ----------
group_cols = ["UniProt (SwissProt) Primary ID of Target Chain",
              "Target Source Organism According to Curator or DataSource",
              "Ligand InChI Key","BindingDB Target Chain Sequence","assay_type"]
agg = df.groupby(group_cols).agg(
    target_name=("Target Name", first_nonnull),
    ligand_name=("BindingDB Ligand Name", first_nonnull),
    ligand_smiles=("Ligand SMILES", first_nonnull),
    ligand_inchikey=("Ligand InChI Key", first_nonnull),
    median_pX=("pX", median_ignore_nan),
    median_aff_nM=("affinity_nM", lambda x: median_ignore_nan(pd.to_numeric(x, errors='coerce'))),
    complex_pdb_id=("PDB ID(s) for Ligand-Target Complex", lambda x: ";".join(sorted(set(x.dropna().astype(str)))))
).reset_index()
# %%
# ---------- Determine WT per (protein, organism) ----------
agg["BindingDB Target Chain Sequence"] = agg["BindingDB Target Chain Sequence"].str.upper()
agg["seq_md5"] = agg["BindingDB Target Chain Sequence"].apply(md5_seq)
wt_map = {}
for (prot, org), g in agg.groupby(["UniProt (SwissProt) Primary ID of Target Chain",
                                   "Target Source Organism According to Curator or DataSource"]):
    seq_counts = Counter(g["BindingDB Target Chain Sequence"])
    wt_seq = seq_counts.most_common(1)[0][0]
    wt_map[(prot, org)] = wt_seq

# ---------- Pair WT vs Mutant ----------
records = []
for (prot, org), g in agg.groupby(["UniProt (SwissProt) Primary ID of Target Chain",
                                   "Target Source Organism According to Curator or DataSource"]):
    wt_seq = wt_map[(prot, org)]
    g = g.reset_index(drop=True)
    for i, row in g.iterrows():
        seq = row["BindingDB Target Chain Sequence"]
        if seq == wt_seq:
            continue
        if len(seq) != len(wt_seq): 
            continue
        diffs = [(a,b,idx+1) for idx,(a,b) in enumerate(zip(wt_seq,seq)) if a!=b]
        if not diffs or len(diffs)>100: 
            continue
        key_mut = [f"{a}{pos}{b}" for a,b,pos in diffs]
        rec = row.to_dict()
        rec["wildtype_seq"] = wt_seq
        rec["key_mutation"] = json.dumps(key_mut)
        rec["mutation_count"] = len(key_mut)
        # find matching WT row same ligand & assay_type
        match = g[(g["BindingDB Target Chain Sequence"]==wt_seq) &
                  (g["Ligand InChI Key"]==row["Ligand InChI Key"]) &
                  (g["assay_type"]==row["assay_type"])]
        if len(match)==0: 
            continue
        wt_row = match.iloc[0]
        rec["pX_wt"] = wt_row["median_pX"]
        rec["affinity_wt_nM"] = wt_row["median_aff_nM"]
        rec["delta_pX"] = row["median_pX"] - wt_row["median_pX"]
        dpX = rec["delta_pX"]
        if pd.isna(dpX): 
            rec["target_type"]=np.nan
            rec["target_type_vb"]=np.nan
        elif dpX >= 1.0:  # 10x
            rec["target_type"]="target"
            rec["target_type_vb"]="target"
        elif abs(dpX) < 0.3:  # ±2x
            rec["target_type"]="off-target"
            rec["target_type_vb"]="nochange"
        elif dpX <= -1.0: # -10x
            rec["target_type"]="off-target"
            rec["target_type_vb"]="negative"
        else: 
            rec["target_type"]="off-target"
            rec["target_type_vb"]="below_par"
        records.append(rec)

out = pd.DataFrame(records)
out["protein_key"] = out["UniProt (SwissProt) Primary ID of Target Chain"] + "_" + \
                     out["Target Source Organism According to Curator or DataSource"]
# %%
# ---------- PDB/assay type filtering ----------
# 1-1) For wildtype targets, complex PDB ids are required 
# out = out[
#       ~((out["target_type"] == "target") & 
#       (out["complex_pdb_id"].isna() | (out["complex_pdb_id"].str.strip()==""))
#     ].copy()
# 1-2) Too strong regulation, so mitigated as the requirement of complex PDB ids for both target and off-target
if args.ensure_pdb:
    out = out[
        ~(out["complex_pdb_id"].isna() | (out["complex_pdb_id"].str.strip()==""))
        ].copy()
# 1-3) Allow when there exist at least one complex PDB ids in a protein
else:
    has_pdb_by_protein = out.groupby("protein_key")["complex_pdb_id"].apply(lambda s: (s.fillna("").str.strip() != "").any())
    out = out[out["protein_key"].isin(has_pdb_by_protein[has_pdb_by_protein].index)].copy()

# 2) Drop ligands that have no affinity values at all
affinities = ["affinity_wt_nM","affinity_mut_nM","pX_wt","median_pX"]
out = out.dropna(subset=[c for c in affinities if c in out.columns]).copy()
# 3) Drop proteins that have no 'target' at all (only off-targets)
has_target = out.groupby("protein_key")["target_type"].apply(lambda s: (s == "target").any())
out = out[out["protein_key"].isin(has_target[has_target].index)].copy()
# 4) Drop off-targets that has different key_mutation to target
if args.same_mutation:
    out["key_mut_norm"] = out["key_mutation"].fillna("").apply(lambda s: ";".join(sorted(json.loads(s))) if s.strip().startswith("[") else ";".join(sorted([t.strip() for t in s.split(",") if t.strip()])))
    keep = (out[out["target_type"]=="target"][["protein_key","key_mut_norm"]].dropna().drop_duplicates().assign(_k=1))
    out = out.merge(keep, on=["protein_key","key_mut_norm"], how="inner").drop(columns=["_k"])

# %%
# ---------- Fetch ligand IDs from RCSB ----------
from typing import Optional, List

# First, fill with Ligand HET ID in PDB(s) in BindingDB
def normalize_hetid(het: str) -> str:
    if not isinstance(het, str): 
        return None
    het = het.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{1,3}", het):
        return None
    if het.isdecimal():
        return None
    return het

def use_bindingdb_hetid(pdb_ids: str, hetid: str) -> str:
    het = normalize_hetid(hetid)
    if not het:
        return np.nan
    ids = []
    if isinstance(pdb_ids, str):
        for pid in pdb_ids:
            ids.append(f"{pid}:{het}")
    return ";".join(sorted(set(ids))) if ids else np.nan

out["pdb_ligand_id"] = out.apply(
    lambda r: use_bindingdb_hetid(r.get("complex_pdb_id", np.nan), r.get("bindingdb_pdb_hetid", np.nan)),
    axis=1
)

# Then, fetch ligand ids from rcsb PDB
__CC_CACHE = {}

def rcsb_entry_summary(pdb_id: str, timeout: int = 15) -> Optional[dict]:
    """Fetch RCSB summary JSON; None on failure."""
    try:
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def rcsb_polymer_entity_summary(pdb_id: str, timeout: int = 15) -> Optional[dict]:
    """Fetch RCSB summary JSON; None on failure."""
    try:
        url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def rcsb_chem_comp(chem_id: str, timeout: int = 15) -> Optional[dict]:
    """Fetch RCSB chem_comp; None on failure."""
    try:
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{chem_id}"
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def pdb_ligand_inchikeys(pdb_id: str) -> List[str]:
    """Collect InChIKeys of ligands via chemcomp records."""
    entry_summary = rcsb_entry_summary(pdb_id)
    polymer_entity_summary = rcsb_polymer_entity_summary(pdb_id)
    if not entry_summary and not polymer_entity_summary:
        return [], []

    comp_ids = []
    if entry_summary.get('rcsb_binding_affinity') is not None:
        comp_ids = set([c['comp_id'] for c in entry_summary.get('rcsb_binding_affinity')])
    elif entry_summary.get("nonpolymer_bound_components") is not None:
        comp_ids = (entry_summary.get("nonpolymer_bound_components", []))
    elif polymer_entity_summary.get("entity_poly") is not None:
        comp_ids = (polymer_entity_summary.get("entity_poly", {}).get("rcsb_non_std_monomers", []))
    elif polymer_entity_summary.get("rcsb_polymer_entity_container_identifiers") is not None:
        comp_ids = (polymer_entity_summary.get("rcsb_polymer_entity_container_identifiers", {}).get("chem_comp_nstd_monomers", []))

    keys = []
    for cid in comp_ids:
        if not cid:
            return None
        if cid in __CC_CACHE:
            key = __CC_CACHE[cid]
        else: 
            cc = rcsb_chem_comp(cid)
            key = None
            if isinstance(cc, dict):
                key = cc.get("rcsb_chem_comp_descriptor", {}).get("in_ch_ikey")
                if not key:
                    for d in (cc.get("pdbx_chem_comp_descriptor", {}) or []):
                        if (d.get("type") or "").lower() == "InChIKey" and d.get("descriptor"):
                            key = d["descriptor"]
                            break
            __CC_CACHE[cid] = key
        if key:
            keys.append(key)
    print("Finding...", pdb_id, list(comp_ids))
    return comp_ids

pdb_lig_map = {}
for pdb in sorted(set(",".join(out["complex_pdb_id"].dropna()).split(","))):
    print(pdb)
    if pdb.strip()=="" or len(pdb.strip()) != 4: continue
    pdb_lig_map[pdb] = pdb_ligand_inchikeys(pdb)
print("pdb_lig_map: ", pdb_lig_map)
def match_lig(pdb_ids: str) -> str:
    if not isinstance(pdb_ids, str): return np.nan
    ids = [x.strip().upper() for x in pdb_ids.split(",") if len(x.strip())==4]
    hits = []
    for pid in ids:
        print("pdb id: ", pid)
        for lig in pdb_lig_map.get(pid,[]):
            print("lig: ", lig)
            if lig not in SOLVENT_IDS:
                hits.append(f"{pid}:{lig}")
    return ",".join(sorted(set(hits))) if hits else np.nan

out["pdb_ligand_id"] = out["complex_pdb_id"].apply(match_lig)
# %%
# ---------- Save ----------
# if "ligand_inchikey" in out.columns and "Ligand InChI Key" not in out.columns:
#     out["Ligand InChI Key"] = out["ligand_inchikey"]
sel_cols = ["protein_key","target_name","ligand_name","ligand_smiles",
            "BindingDB Target Chain Sequence","wildtype_seq","key_mutation",
            "target_type","target_type_vb",
            "assay_type","affinity_wt_nM","median_aff_nM","pX_wt","median_pX","delta_pX",
            "complex_pdb_id","pdb_ligand_id"]
out = out[sel_cols]
out = out.rename(columns={
    "BindingDB Target Chain Sequence":"sequence",
    "affinity_wt_nM":"wt_affinity_mid",
    "median_aff_nM":"mut_affinity_mid",
    "pX_wt":"wt_p_aff",
    "median_pX":"mut_p_aff",
    "delta_pX":"delta_p_aff"  # delta_pX = mut_p_aff - wt_p_aff
})
out.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved: {OUTPUT_CSV}, total {len(out)} entries")

# %%
