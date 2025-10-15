
import argparse
import requests
import hashlib
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Config
COL_TARGET_NAME = "Target Name"
COL_ORGANISM = "Target Source Organism According to Curator or DataSource"
COL_UNIPROT = "UniProt (SwissProt) Primary ID of Target Chain"
COL_SEQ = "BindingDB Target Chain Sequence"

COL_LIG_INCHIKEY = "Ligand InChI Key"
COL_LIG_SMILES = "Ligand SMILES"
COL_LIG_NAME = "BindingDB Ligand Name"

COL_PDB_IDS = "PDB ID(s) of Target Chain"
COL_PDB_COMPLEX = "PDB ID(s) for Ligand-Target Complex"
COL_DOI = "Article DOI"

COL_KI = "Ki (nM)"
COL_KD = "Kd (nM)"
COL_IC50 = "IC50 (nM)"
COL_EC50 = "EC50 (nM)"

# Helper functions
def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def safe_str(x) -> str:
    from pandas.api.types import is_scalar

    if is_scalar(x):
        return "" if (x is None or (isinstance(x, float) and pd.isna(x))) else str(x).strip()

    if isinstance(x, (pd.Series, pd.Index)):
        if x.size == 0:
            return ""
        non_na = x[x.notna()]
        return "" if non_na.size == 0 else str(non_na.iloc[0]).strip()

    try:
        for e in x:
            if not pd.isna(e):
                return str(e).strip()
        return ""
    except Exception:
        return str(x).strip()

def _agg_join(series: pd.Series) -> str:
    vals = [str(v).strip() for v in series.dropna().astype(str) if str(v).strip() != ""]
    if not vals:
        return ""
    return ";".join(sorted(set(vals), key=lambda x: (x.replace("~","").replace(">","").replace("<",""), x)))

def prune_columns(df: pd.DataFrame) -> pd.DataFrame:
    core_keep = {
        COL_TARGET_NAME, COL_ORGANISM, COL_UNIPROT, COL_SEQ,
        COL_LIG_INCHIKEY, COL_LIG_SMILES, COL_LIG_NAME,
        COL_KI, COL_KD, COL_IC50, COL_EC50,
        COL_PDB_IDS, COL_PDB_COMPLEX, COL_DOI,
    }
    present = [c for c in core_keep if c in df.columns]
    pruned = df[present].copy()
    return pruned

def split_pdb_ids(x: str) -> List[str]:
    if not isinstance(x, str):
        return []
    toks = re.split(r"[;,\s]+", x.strip())
    return [t.upper() for t in toks if t]

def parse_affinity_cell(x) -> Optional[float]:
    if pd.isna(x): return None
    s = str(x).strip()
    m = re.search(r"([0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", s)
    return float(m.group(1)) if m else None

def choose_affinity(row: pd.Series) -> Tuple[Optional[float], Optional[str]]:
    for col in (COL_KI, COL_KD, COL_IC50, COL_EC50):
        if col in row:
            v = parse_affinity_cell(row[col])
            if v is not None and math.isfinite(v):
                return v, col
    return None, None

def coerce_affinities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for src, dst in [
        (COL_KI, "Ki_nM"), (COL_KD, "Kd_nM"),
        (COL_IC50, "IC50_nM"), (COL_EC50, "EC50_nM")
    ]:
        df[dst] = df[src].map(parse_affinity_cell)
    def pick(row):
        for col, tag in [("Ki_nM","Ki"),("Kd_nM","Kd"),("IC50_nM","IC50"),("EC50_nM","EC50")]:
            v = row[col]
            if v is not None and pd.notna(v) and math.isfinite(v):
                return v, tag
        return None, None
    vals = df.apply(pick, axis=1).tolist()
    aff_vals, aff_types = zip(*vals) if vals else ([],[])
    df["affinity"] = list(aff_vals)
    df["affinity_type"] = list(aff_types)
    return df

def extract_mutations_from_target_name(name: str) -> List[str]:
    """Parse mutations from Target Name; returns normalized."""
    s = safe_str(name)
    inside = re.findall(r"\[([^\]]+)\]", s)
    muts = []
    for block in inside:
        for tok in re.split(r"[,\s]+", block.strip()):
            tok = tok.strip()
            if re.match(r"^[A-Z][0-9]+[A-Z]$", tok):
                muts.append(tok)
    return muts

def seq_mutations(ref: str, alt: str) -> Tuple[List[str], bool]:
    """Return list of substitution mutationsand a flag if indel detected."""
    if ref is None or alt is None:
        return [], True
    if len(ref) != len(alt):
        return [], True
    muts = []
    for i, (a, b) in enumerate(zip(ref, alt), start=1):
        if a == b:
            continue
        if a == "-" or b == "-":
            return [], True
        if not (a.isalpha() and b.isalpha()):
            return [], True
        muts.append(f"{a.upper()}{i}{b.upper()}")
    return muts, False

# UniProt, RCSB PDB fetchers
def fetch_uniprot_sequence(uniprot: str, timeout: int = 15) -> Optional[str]:
    """Fetch canonical FASTA from UniProt REST. Returns sequence or None on failure."""
    if not uniprot:
        return None
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot}.fasta"
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        fa = r.text
        seq = "".join(line.strip() for line in fa.splitlines() if not line.startswith(">"))
        return seq if seq else None
    except Exception:
        return None

def rcsb_entry_summary(pdb_id: str, timeout: int = 15) -> Optional[dict]:
    """Fetch RCSB summary JSON; None on failure."""
    try:
        import requests
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
        import requests
        url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def pdb_target_matches_uniprot(pdb_id: str, uniprot: str) -> Optional[bool]:
    """Check if any polymer entity cross-refs include given UniProt ID."""
    summary = rcsb_entry_summary(pdb_id)
    if not summary or not uniprot:
        return None
    try:
        assemblies = summary.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
        for ent_id in assemblies:
            j = requests.get(f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{ent_id}", timeout=15)
            if j.status_code != 200:
                continue
            ent = j.json()
            xrefs = ent.get("rcsb_polymer_entity_container_identifiers", {}).get("reference_sequence_identifiers", [])
            print(xrefs)
            for xf in xrefs:
                if xf.get("database_name") in ("UniProt", "UniProtKB") and xf.get("database_accession") == uniprot:
                    return True
    except Exception:
        return None
    return False

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

__CC_CACHE = {}

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
    return keys, comp_ids

def pdb_has_mutation_flag(pdb_id: str) -> Optional[bool]:
    """True if entry title or entity features indicate mutation."""
    summary = rcsb_entry_summary(pdb_id)
    if not summary:
        return None
    title = safe_str(summary.get("struct", {}).get("title", "")).lower()
    if any(w in title for w in ["mutant", "mutation", "variant"]):
        return True
    return False

# Core pipeline
@dataclass
class Options:
    input_path: str
    output_path: str

def build_protein_key(uniprot: str, organism: str) -> Optional[str]:
    u = safe_str(uniprot)
    o = safe_str(organism)
    if not u or not o:
        return None
    return f"{u}|{o}"

def normalize_bindingdb(df: pd.DataFrame) -> pd.DataFrame:
    cols_needed = [
        COL_TARGET_NAME, COL_ORGANISM, COL_UNIPROT, COL_SEQ,
        COL_LIG_INCHIKEY, COL_LIG_SMILES, COL_LIG_NAME,
        COL_PDB_IDS, COL_PDB_COMPLEX, COL_KI, COL_KD, COL_IC50, COL_EC50
    ]
    for c in cols_needed:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def compute_reference_sequences(df: pd.DataFrame) -> Dict[str, str]:
    """Per Protein Key choose reference: UniProt fetch; fallback to mode-by-length among provided sequences."""
    ref: Dict[str, str] = {}
    for key, sub in df.groupby("Protein Key"):
        uids = {safe_str(u) for u in sub[COL_UNIPROT].dropna().unique()}
        ref_seq = None
        for uid in uids:
            ref_seq = fetch_uniprot_sequence(uid)
            if ref_seq:
                break
        if not ref_seq:
            # fallback: choose most frequent length, then most frequent sequence
            lengths = sub[COL_SEQ].dropna().astype(str).map(len)
            if lengths.empty:
                continue
            target_len = lengths.mode().iloc[0]
            cand = (
                sub[COL_SEQ].dropna().astype(str)
                .loc[lengths == target_len]
                .mode()
            )
            ref_seq = cand.iloc[0] if len(cand) else None
        if ref_seq:
            ref[key] = ref_seq
    return ref

def annotate_variants(df: pd.DataFrame, ref_map: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    seqs = df[COL_SEQ].astype(str).to_list()
    keys = df["Protein Key"].to_list()
    tn_muts = df[COL_TARGET_NAME].astype(str).map(extract_mutations_from_target_name).to_list()

    seq_md5 = [md5(s) for s in seqs]
    is_ref, key_mut, n_subs, indel_flag = [], [], [], []

    for s, k, nmuts in zip(seqs, keys, tn_muts):
        ref = ref_map.get(k)
        if not ref:
            is_ref.append(False); key_mut.append(""); n_subs.append(pd.NA); indel_flag.append(True); continue
        if s == ref:
            is_ref.append(True); key_mut.append(""); n_subs.append(0); indel_flag.append(False); continue
        if len(s) != len(ref):
            is_ref.append(False); key_mut.append(""); n_subs.append(pd.NA); indel_flag.append(True); continue
        muts = []
        has_indel = False
        for i, (a,b) in enumerate(zip(ref, s), start=1):
            if a == b: 
                continue
            if a == "-" or b == "-":
                has_indel = True; break
            muts.append(f"{a}{i}{b}")
        if has_indel:
            is_ref.append(False); key_mut.append(""); n_subs.append(pd.NA); indel_flag.append(True)
        else:
            merged = sorted(set(muts).union(nmuts))
            is_ref.append(False); key_mut.append(";".join(merged)); n_subs.append(len(merged)); indel_flag.append(False)

    df["Sequence"] = seqs
    df["Sequence md5"] = seq_md5
    df["is_reference"] = is_ref
    df["key_mutation"] = key_mut
    df["n_subs"] = n_subs
    df["has_indel"] = indel_flag
    df = df[df["has_indel"] == False].drop(columns=["has_indel"])
    return df

def aggregate_affinity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not set([COL_KI, COL_KD, COL_IC50, COL_EC50]).issubset(df.columns):
        for c in [COL_KI, COL_KD, COL_IC50, COL_EC50]:
            if c not in df.columns:
                df[c] = pd.NA

    all_nan = df[[COL_KI, COL_KD, COL_IC50, COL_EC50]].isna().all(axis=1)
    empty_str = (df[[COL_KI, COL_KD, COL_IC50, COL_EC50]]
                 .fillna("")
                 .applymap(lambda x: str(x).strip() == "")
                 .all(axis=1))
    df = df[~(all_nan | empty_str)].copy()

    grp_cols = ["Protein Key", "Sequence md5", "Ligand InChI Key"]

    agg = df.groupby(grp_cols).agg({
        COL_TARGET_NAME: "first",
        COL_LIG_NAME: "first",
        COL_LIG_SMILES: "first",
        "Sequence": "first",
        "key_mutation": "first",
        "n_subs": "first",
        "is_reference": "first",
        COL_KI: _agg_join,
        COL_KD: _agg_join,
        COL_IC50: _agg_join,
        COL_EC50: _agg_join,
        COL_PDB_IDS: _agg_join,
        COL_PDB_COMPLEX: _agg_join
    }).reset_index()

    return agg

def filter_by_ligand_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Keep proteins having ≥2 ligands measured across ref+mut
       Drop ligands measured in only one of ref/mut."""
    # ligands present in both ref and non-ref for each protein
    out_rows = []
    for prot, sub in df.groupby("Protein Key"):
        lig_ref = set(sub.loc[sub["is_reference"] == True, "Ligand InChI Key"].unique())
        lig_mut = set(sub.loc[sub["is_reference"] == False, "Ligand InChI Key"].unique())
        overlap = lig_ref & lig_mut
        if len(overlap) < 2:
            continue
        keep = sub[sub["Ligand InChI Key"].isin(overlap)].copy()
        out_rows.append(keep)
    if not out_rows:
        return df.iloc[0:0]
    return pd.concat(out_rows, ignore_index=True)

def attach_pdb_lists(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pdb_ids"] = df[COL_PDB_IDS].map(split_pdb_ids)
    df["complex_pdb_ids"] = df[COL_PDB_COMPLEX].map(split_pdb_ids)
    # Filter out proteins with no PDB info at all
    filt = df["pdb_ids"].map(len) + df["complex_pdb_ids"].map(len)
    df = df[filt > 0]
    return df

def validate_pdbs(df: pd.DataFrame, pdb_type: str) -> pd.DataFrame:
    """Add booleans and pdb id if True:
       pdb_target_matches, pdb_has_ligand, pdb_ligand_matches, pdb_is_mutation."""
    df = df.copy()
    # boolean flags
    df[f"{pdb_type}_has_ligand"] = False
    df[f"{pdb_type}_ligand_matches"] = False
    df[f"{pdb_type}_is_mutation"] = False
    # pdb id lists (semicolon-joined)
    df[f"{pdb_type}_has_ligand_ids"] = ""
    df[f"{pdb_type}_ligand_match_ids"] = ""
    df[f"{pdb_type}_is_mutation_ids"] = ""
    df[f"{pdb_type}_ligand_ids"] = ""

    def get_uniprot_from_key(k: str) -> str:
        return k.split("|")[0] if isinstance(k, str) and "|" in k else ""

    for i, row in df.iterrows():
        uniprot = get_uniprot_from_key(row["Protein Key"])
        lig_key = safe_str(row["Ligand InChI Key"])
        pdb_ids = row.get(f"{pdb_type}_ids", []) or []
        if not pdb_ids:
            continue

        ids_has_ligand = []
        ids_ligand_match = []
        ids_mutation = []
        ligand_ids = []

        for pid in pdb_ids:
            lig_keys, lig_ids = pdb_ligand_inchikeys(pid)
            if lig_keys:
                ids_has_ligand.append(pid)
                ligand_ids.append(list(lig_ids))
            if lig_key and lig_keys and lig_key in lig_keys:
                ids_ligand_match.append(pid)

            pm = pdb_has_mutation_flag(pid)
            if pm:
                ids_mutation.append(pid)

        df.at[i, f"{pdb_type}_has_ligand"] = bool(ids_has_ligand)
        df.at[i, f"{pdb_type}_ligand_matches"] = bool(ids_ligand_match)
        df.at[i, f"{pdb_type}_is_mutation"] = bool(ids_mutation)

        df.at[i, f"{pdb_type}_has_ligand_ids"] = ";".join(sorted(set(ids_has_ligand)))
        df.at[i, f"{pdb_type}_ligand_match_ids"] = ";".join(sorted(set(ids_ligand_match)))
        df.at[i, f"{pdb_type}_is_mutation_ids"] = ";".join(sorted(set(ids_mutation)))
        df.at[i, f"{pdb_type}_ligand_ids"] = ";".join("(" + ",".join(l) + ")" for l in ligand_ids)
    return df

def ligand_id_conversion(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    __LIG_ID_CACHE: Dict[str, List[str]] = {}

    def inchi_keys_to_lig_ids(val) -> str:
        if pd.isna(val):
            return ""
        inchikeys = [k for k in re.split(r"[;,\s]+", str(val).strip()) if k]
        ids: list[str] = []
        for key in inchikeys:
            if key in __LIG_ID_CACHE:
                ids.extend(__LIG_ID_CACHE[key])
            else:
                try:
                    lig_id = ligand_ids_from_inchikey(key) or []
                except Exception:
                    lig_id = []
                __LIG_ID_CACHE[key] = lig_id
                ids.extend(lig_id)
        return ";".join(sorted(set(ids)))
    
    df["Ligand ID"] = df["Ligand InChI Key"].map(inchi_keys_to_lig_ids)
    return df

def main(opts: Options):
    # Load
    ext = opts.input_path.lower()
    raw = pd.read_csv(opts.input_path, sep="\t", engine="python", on_bad_lines="skip")
    raw = normalize_bindingdb(raw)
    raw = prune_columns(raw)
    print(f"BindingDB TSV loaded from {opts.input_path}.")

    # Step 1: keys & basic fields
    raw["Protein Key"] = [
        build_protein_key(safe_str(u), safe_str(o))
        for u, o in zip(raw.get(COL_UNIPROT, pd.Series(dtype=str)),
                        raw.get(COL_ORGANISM, pd.Series(dtype=str)))
    ]
    raw = raw[~raw["Protein Key"].isna()]
    raw = raw[raw[COL_SEQ].notna() & raw[COL_LIG_INCHIKEY].notna()]
    raw["Sequence"] = raw[COL_SEQ].astype(str)
    raw["Sequence md5"] = raw["Sequence"].map(md5)
    keys_with_variation = (
        raw.groupby("Protein Key")["Sequence md5"].nunique()
        .loc[lambda s: s >= 2]
        .index
    )
    raw_var = raw[raw["Protein Key"].isin(keys_with_variation)].copy()
    print("Step 1 done!")

    # Step 2: reference mapping & mutation annotation
    ref_map = compute_reference_sequences(raw_var)
    anno = annotate_variants(raw_var, ref_map)
    print("Step 2 done!")

    # Step 3: affinity aggregation and ligand-overlap filtering
    aff = aggregate_affinity(anno)
    aff = filter_by_ligand_overlap(aff)
    print("Step 3 done!")

    # Step 4: PDB columns & validation
    aff = attach_pdb_lists(aff)
    # Drop proteins with absolutely no PDB info
    if aff.empty:
        print("No rows after PDB info attachment. Exiting with empty output.", file=sys.stderr)
        aff.to_csv(opts.output_path, index=False)
        return

    aff = validate_pdbs(aff, 'pdb')
    aff = validate_pdbs(aff, 'complex_pdb')
    print("Step 4 done!")

    # Step 5: final columns & save
    final_cols = [
        "Protein Key",
        COL_TARGET_NAME,
        COL_LIG_NAME,
        COL_LIG_SMILES,
        "Sequence",
        "key_mutation",
        "n_subs",
        "is_reference",
        COL_KI, COL_KD, COL_IC50, COL_EC50,
        "pdb_ids",
        "complex_pdb_ids",
        "pdb_has_ligand",
        "pdb_ligand_matches",
        "pdb_is_mutation",
        "pdb_has_ligand_ids",
        "pdb_ligand_match_ids",
        "pdb_is_mutation_ids",
        "pdb_ligand_ids",
        "complex_pdb_has_ligand",
        "complex_pdb_ligand_matches",
        "complex_pdb_is_mutation",
        "complex_pdb_has_ligand_ids",
        "complex_pdb_ligand_match_ids",
        "complex_pdb_is_mutation_ids",
        "complex_pdb_ligand_ids",
        COL_DOI,
        "Ligand InChI Key",
        "Sequence md5",
    ]

    # Some rows might lack SMILES/Name; ensure columns exist
    for c in final_cols:
        if c not in aff.columns:
            aff[c] = pd.NA

    # One row per (Protein Key, Sequence variant, Ligand InChI Key)
    out = aff[final_cols].copy()
    out.to_csv(opts.output_path, index=False)
    print(f"Saved: {opts.output_path}  (rows={len(out)})")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build BindingDB mutation set for LigandMPNN specificity benchmarking.")
    p.add_argument("--in", dest="input_path", required=True, help="BindingDB TSV/CSV")
    p.add_argument("--out", dest="output_path", required=True, help="Output CSV path")
    args = p.parse_args()
    main(Options(input_path=args.input_path, output_path=args.output_path))
