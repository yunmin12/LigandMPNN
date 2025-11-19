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
# p = argparse.ArgumentParser(description="BindingDB curation set for ligand specificity evaluation.")
# p.add_argument("--out", dest="output_path", required=True, help="Output CSV path")
# p.add_argument("--ensure_pdb", dest="ensure_pdb", required=False, default=False, help="Ensure all the ligands to have complex PDB ids. If not, leave ligands with at least one complex PDB ids in the same protein. Default: False.")
# p.add_argument("--same_mutation", dest="same_mutation", required=False, default=False, help="Remains off-targets that only has the same key_mutation with the target. Default: False.")
# args = p.parse_args()

INPUT_TSV = "/scratch/yunmin/data/db/BindingDB_All.tsv"
# OUTPUT_CSV = args.output_path
OUTPUT_CSV = "/scratch/yunmin/data/db/15_eval_set_all_mitigated.csv"
SOLVENT_IDS = {
    "HOH","WAT","DOD",  # water
    "CL","NA","K","CA","MG","ZN","MN","CO","CU","NI","IOD",  # common ions
    "SO4","PO4",  # anions
    "GOL","EDO","PEG","PG4","MPD","TRS","MES","ACE","IPA","BME","FMT",  # common buffers/additives
    "SEP", "TPO"    # modified residues
}
# ---------- Utility ----------
def md5_seq(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()

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
df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)
def pick(*cands):
    for c in cands:
        if c in df.columns: 
            return c
    return None

COL_CHAINS = pick(
"Number of Protein Chains in Target (>1 implies a multichain complex)"
)
COL_PDB_COMPLEX = pick("PDB ID(s) for Ligand-Target Complex")
COL_HET = pick("Ligand HET ID in PDB")
COL_LIG_SMILES = pick("Ligand SMILES")
COL_LIG_INCHIKEY = pick("Ligand InChI Key","Ligand InChIKey","Ligand InChi Key")
COL_LIG_NAME = pick("BindingDB Ligand Name")
COL_TARGET_NAME = pick("Target Name")
COL_ORG = pick("Target Source Organism According to Curator or DataSource")
COL_SEQ1 = pick("BindingDB Target Chain Sequence 1","BindingDB Target Chain Sequence")
COL_UP1  = pick("UniProt (SwissProt) Primary ID of Target Chain 1",
                "UniProt (SwissProt) Primary ID of Target Chain")

ASSAY_KI   = pick("Ki (nM)")
ASSAY_KD   = pick("Kd (nM)")
ASSAY_IC50 = pick("IC50 (nM)")
ASSAY_EC50 = pick("EC50 (nM)")
assay_cols = [c for c in [ASSAY_KI,ASSAY_KD,ASSAY_IC50,ASSAY_EC50] if c]

missing = [name for name, col in {
    "CHAINS":COL_CHAINS, "PDB_COMPLEX":COL_PDB_COMPLEX, "HET":COL_HET,
    "SMILES":COL_LIG_SMILES, "InChIKey":COL_LIG_INCHIKEY, "LigName":COL_LIG_NAME,
    "Target":COL_TARGET_NAME, "Organism":COL_ORG, "Seq1":COL_SEQ1, "UniProt1":COL_UP1
}.items() if col is None]
if missing:
    print("WARN missing columns:", missing)

if COL_CHAINS:
    chains = pd.to_numeric(df[COL_CHAINS].str.extract(r"(\d+)", expand=False), errors="coerce")
    df = df[chains.fillna(1) <= 1].copy()

rename_map = {}
if COL_PDB_COMPLEX:  rename_map[COL_PDB_COMPLEX] = "PDB ID(s) for Ligand-Target Complex"
if COL_HET:          rename_map[COL_HET] = "Ligand HET ID in PDB"
if COL_LIG_SMILES:   rename_map[COL_LIG_SMILES] = "Ligand SMILES"
if COL_LIG_INCHIKEY: rename_map[COL_LIG_INCHIKEY] = "Ligand InChI Key"
if COL_LIG_NAME:     rename_map[COL_LIG_NAME] = "BindingDB Ligand Name"
if COL_TARGET_NAME:  rename_map[COL_TARGET_NAME] = "Target Name"
if COL_ORG:          rename_map[COL_ORG] = "Target Source Organism According to Curator or DataSource"
if COL_SEQ1:         rename_map[COL_SEQ1] = "BindingDB Target Chain Sequence"
if COL_UP1:          rename_map[COL_UP1]  = "UniProt (SwissProt) Primary ID of Target Chain"
df = df.rename(columns=rename_map)

chain_col="Number of Protein Chains in Target (>1 implies a multichain complex)"
chains = pd.to_numeric(df[chain_col].str.extract(r"(\d+)", expand=False), errors="coerce")
df = df[chains.fillna(1) <= 1].copy()

rename = {}
def pick(old1, old0, new):
    if old1 in df.columns: rename[old1] = new
    elif old0 in df.columns: rename[old0] = new

pick("BindingDB Target Chain Sequence 1", "BindingDB Target Chain Sequence",
    "BindingDB Target Chain Sequence")
pick("UniProt (SwissProt) Primary ID of Target Chain 1",
    "UniProt (SwissProt) Primary ID of Target Chain",
    "UniProt (SwissProt) Primary ID of Target Chain")
pick("PDB ID(s) of Target Chain 1", "PDB ID(s) of Target Chain",
    "PDB ID(s) of Target Chain")

df = df.rename(columns=rename)

# Select useful columns
cols = ["BindingDB Reactant_set_id","Ligand SMILES","Ligand InChI Key",
        "BindingDB Ligand Name","Target Name",
        "Target Source Organism According to Curator or DataSource",
        "Ki (nM)","Kd (nM)","IC50 (nM)","EC50 (nM)",
        "PDB ID(s) for Ligand-Target Complex",
        "BindingDB Target Chain Sequence",
        "UniProt (SwissProt) Primary ID of Target Chain",
        "Ligand HET ID in PDB"]
df = df[[c for c in cols if c in df.columns]].copy()

df["BindingDB Target Chain Sequence"] = df["BindingDB Target Chain Sequence"].fillna("").str.strip()
df = df[df["BindingDB Target Chain Sequence"].str.len() > 0]
# %%
# ---------- Compute pX values ----------
# process multi assay and exception cases (>, <, ;)
def parse_multi_nM(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)): 
        return []
    vals = []
    for tok in re.split(r"[;,\s]+", str(cell).strip()):
        if not tok: 
            continue
        s = tok.replace(",", "").strip()
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
    xs = parse_multi_nM(cell)
    return float(np.median(xs)) if xs else np.nan

for col in ["Ki (nM)", "Kd (nM)", "IC50 (nM)", "EC50 (nM)"]:
    if col in df.columns:
        df[col] = df[col].apply(to_single_nM)

for col in ["Ki (nM)","Kd (nM)","IC50 (nM)","EC50 (nM)"]:
    if col in df.columns:
        df[col+"_pX"] = 9 - np.log10(pd.to_numeric(df[col], errors="coerce"))

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
              "PDB ID(s) for Ligand-Target Complex",
              "Ligand HET ID in PDB"
              ]].copy()
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
    median_aff_nM=("affinity_nM", lambda x: median_ignore_nan(pd.to_numeric(x, errors='coerce'))),
    complex_pdb_id=("PDB ID(s) for Ligand-Target Complex", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
    bindingdb_pdb_hetid=("Ligand HET ID in PDB", first_nonnull), 
).reset_index()

agg["median_pX"] = 9 - np.log10(agg["median_aff_nM"])
# ---------- Determine WT per (protein, organism) ----------
agg["BindingDB Target Chain Sequence"] = agg["BindingDB Target Chain Sequence"].str.upper()
agg["seq_md5"] = agg["BindingDB Target Chain Sequence"].apply(md5_seq)
wt_map = {}
for (prot, org), g in agg.groupby(["UniProt (SwissProt) Primary ID of Target Chain",
                                   "Target Source Organism According to Curator or DataSource"]):
    seq_counts = Counter(g["BindingDB Target Chain Sequence"])
    wt_seq = seq_counts.most_common(1)[0][0]
    wt_map[(prot, org)] = wt_seq
# %%
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
        rec["delta_pX"] = np.log10(float(rec["affinity_wt_nM"]) / float(row["median_aff_nM"]))
        dpX = rec["delta_pX"]
        # ver.2 mitigated
        if pd.isna(dpX): 
            rec["target_type"]=np.nan
            rec["target_type_vb"]=np.nan
        elif dpX >= 0.5:  # 2x
            rec["target_type"]="target"
            rec["target_type_vb"]="target"
        elif dpX <= -0.5: # -2x
            rec["target_type"]="off_target"
            rec["target_type_vb"]="negative"
        else: 
            rec["target_type"]="off_target"
            rec["target_type_vb"]="nochange"
        records.append(rec)

out = pd.DataFrame(records)
out["protein_key"] = out["UniProt (SwissProt) Primary ID of Target Chain"] + "_" + \
                     out["Target Source Organism According to Curator or DataSource"]
# %%
# ---------- PDB/assay type filtering ----------
# 1-2) Ensure all the rows to have complex pdb
out = out[
    ~(out["complex_pdb_id"].isna() | (out["complex_pdb_id"].str.strip()==""))
    ].copy()
# %%
# 1-3) Allow when there exist at least one complex PDB ids in a protein
has_pdb_by_protein = out.groupby("protein_key")["complex_pdb_id"].apply(lambda s: (s.fillna("").str.strip() != "").any())
out = out[out["protein_key"].isin(has_pdb_by_protein[has_pdb_by_protein].index)].copy()
# %%
# 2) Drop ligands that have no affinity values at all
affinities = ["affinity_wt_nM","affinity_mut_nM","pX_wt","median_pX"]
out = out.dropna(subset=[c for c in affinities if c in out.columns]).copy()
# %%
# 3) Drop proteins that have no 'target' at all (only off-targets)
has_target = out.groupby("protein_key")["target_type"].apply(lambda s: (s == "target").any())
out = out[out["protein_key"].isin(has_target[has_target].index)].copy()
# %%
# 4) Drop off-targets that has different key_mutation to target
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
        pid_list = [x.strip().upper() for x in pdb_ids.split(",") if len(x.strip())==4]
        for pid in pid_list:
            ids.append(f"{pid}:{het}")
    return ";".join(sorted(set(ids))) if ids else np.nan

het = out["bindingdb_pdb_hetid"].astype("string").str.upper().str.strip()
het = het.where(het.str.fullmatch(r"[A-Z0-9]{1,3}"), None) \
         .where(~het.str.fullmatch(r"\d+"), None)
out["bindingdb_pdb_hetid"] = het

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
    if pdb.strip()=="" or len(pdb.strip()) != 4: continue
    pdb_lig_map[pdb] = pdb_ligand_inchikeys(pdb)

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

def union_row_ligands(pdb_ids, hetid):
    pairs = {tok for tok in str(use_bindingdb_hetid(pdb_ids, hetid)).split(",") if ":" in tok}
    pid_list = [t[:4].upper() for t in re.split(r"[,\s;]+", str(pdb_ids)) if len(t)>=4]
    for pid in pid_list:
        for lig in pdb_lig_map.get(pid, set()):
            if lig not in SOLVENT_IDS and lig != "UNK":
                pairs.add(f"{pid}:{lig}")
    return ",".join(sorted(set(pairs))) if pairs else np.nan

out["pdb_ligand_id"] = out.apply(
    lambda r: union_row_ligands(r.get("complex_pdb_id", np.nan), r.get("bindingdb_pdb_hetid", np.nan)),
    axis=1
)

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
    "affinity_wt_nM":"wt_affinity_nM",
    "median_aff_nM":"mut_affinity_nM",
    "pX_wt":"wt_pX",
    "median_pX":"mut_pX",
    "delta_pX":"delta_pX"  # delta_pX = mut_p_aff - wt_p_aff
})
out.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved: {OUTPUT_CSV}, total {len(out)} entries")

# %%
