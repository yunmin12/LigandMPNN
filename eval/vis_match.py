import os, re, json, gzip, glob, argparse
import csv, math, itertools
import warnings, datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================ Amino acid utilities ================
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "SEC":"U","PYL":"O"
}
AA1_SET = set(AA_ORDER)

def aa1(x: str) -> Optional[str]:
    if not x: return None
    x = x.strip().upper()
    if len(x) == 1 and x in AA1_SET: return x
    if len(x) == 3: return AA3_TO_1.get(x.upper(), None)
    return None

# ================ File discovery helpers ================
def discover_files(entry: Dict) -> Dict[str, str]:
    if ("prefix" in entry) == ("dir" in entry):
        raise ValueError("Each version must have exactly one of 'prefix' or 'dir'.")
    if "prefix" in entry:
        pref = entry["prefix"]
        files = {
            "logits": f"{pref}_logits.csv.gz",
            "log_probs": f"{pref}_log_probs.csv.gz",
            "meta": f"{pref}_meta.json",
            "per_res": f"{pref}_per_residue_scores.csv",
            "pt": f"{pref}.pt",
            "topk": f"{pref}_topk.json",
        }
        # comp = glob.glob(f"{pref}*{CONFIG['component_logits_hint']}*.csv*")
        # if comp: files["component"] = comp[0]
        return files
    d = entry["dir"].rstrip("/")
    want_suffixes = {
        "logits": "_logits.csv",
        "logits_gz": "_logits.csv.gz",
        "log_probs": "_log_probs.csv",
        "log_probs_gz": "_log_probs.csv.gz",
        "meta": "_meta.json",
        "per_res": "_per_residue_scores.csv",
        "pt": ".pt",
        "topk": "_topk.json",
    }
    choose = {}
    for k, suf in want_suffixes.items():
        hits = glob.glob(os.path.join(d, f"*{suf}"))
        if hits: choose[k] = hits[0]
    files = {}
    files["logits"] = choose.get("logits") or choose.get("logits_gz")
    files["log_probs"] = choose.get("log_probs") or choose.get("log_probs_gz")
    files["meta"] = choose.get("meta")
    files["per_res"] = choose.get("per_res")
    files["pt"] = choose.get("pt")
    files["topk"] = choose.get("topk")
    # comp = glob.glob(os.path.join(d, f"*{CONFIG['component_logits_hint']}*.csv*"))
    # if comp: files["component"] = comp[0]
    return files

def read_table_maybe_gz(path: str) -> pd.DataFrame:
    if path.endswith(".gz"):
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)

def read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

# ============= Mapping PDB index -> internal idx =============
def parse_mutation_token(tok: str) -> Dict:
    s = tok.strip().upper().replace(" ", "")
    m = re.match(r"^([A-Z])(\d+)([A-Z])?$", s)
    if m:
        nat, resid, mut = m.group(1), int(m.group(2)), m.group(3)
        return {"chain": None, "resid": resid, "natural": nat, "mutant": mut}
    m = re.match(r"^([A-Z]):?(\d+)([A-Z])?$", s)
    if m:
        chain, resid, mut = m.group(1), int(m.group(2)), m.group(3)
        return {"chain": chain, "resid": resid, "natural": None, "mutant": mut}
    m = re.match(r"^([A-Z]):(\d+)([A-Z])?>?([A-Z])?$", s)
    if m:
        chain, resid, nat, mut = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        return {"chain": chain, "resid": resid, "natural": nat, "mutant": mut}
    raise ValueError(f"Cannot parse mutation token: {tok}")

def map_with_per_res(df: pd.DataFrame, chain: Optional[str], resid: int, natural: Optional[str]) -> Optional[int]:
    cols = {c.lower(): c for c in df.columns}
    chain_col = cols.get("chain") or cols.get("chain_id")
    resid_candidates = [cols.get(x) for x in ["pdb_resi","resi","residue","pdb_residue","resid","res_index"] if cols.get(x)]
    aa_col = cols.get("aa") or cols.get("resname") or cols.get("aatype") or cols.get("pdb_aa")
    idx_col = cols.get("index") or cols.get("i") or cols.get("pos") or cols.get("internal_idx") or cols.get("model_idx")
    if idx_col is None:
        df = df.copy().reset_index()
        idx_col = "index"
    dfx = df
    if chain and chain_col and chain_col in df.columns:
        dfx = dfx[dfx[chain_col].astype(str).str.upper() == chain.upper()]
    for rc in resid_candidates:
        if rc is None: 
            continue
        hits = dfx[dfx[rc].astype(str) == str(resid)]
        if len(hits) == 1:
            row = hits.iloc[0]
            if natural and aa_col in dfx.columns:
                aa_here = str(row[aa_col]).strip().upper()
                aa1 = aa_here if len(aa_here)==1 else AA3_TO_1.get(aa_here, None)
                if aa1 and aa1 != natural:
                    warnings.warn(f"Natural AA mismatch at resid {resid}: found {aa1}, expected {natural}. Using index anyway.")
            return int(row[idx_col])
        elif len(hits) > 1 and natural and aa_col in dfx.columns:
            hits2 = hits[hits[aa_col].astype(str).str.upper().str.startswith(natural.upper())]
            if len(hits2) == 1:
                row = hits2.iloc[0]
                return int(row[idx_col])
    return None

def map_with_meta(meta: dict, chain: Optional[str], resid: int, natural: Optional[str]) -> Optional[int]:
    keys = ["pdb_idx_map","pdb_index_map","residue_map","pdb_to_internal"]
    for k in keys:
        if k in meta and isinstance(meta[k], list):
            lst = meta[k]
            cands = [r for r in lst if int(r.get("resi", -1)) == resid and (chain is None or str(r.get("chain","")).upper()==chain.upper())]
            if len(cands)==1:
                r = cands[0]
                if natural and r.get("aa"):
                    aa_here = str(r["aa"]).strip().upper()
                    if len(aa_here)==1 and aa_here != natural:
                        warnings.warn(f"Natural AA mismatch at resid {resid} in meta. Using index anyway.")
                return int(cands[0].get("idx", cands[0].get("internal_idx", cands[0].get("i", -1))))
            elif len(cands)>1 and natural:
                cands2 = [r for r in cands if str(r.get("aa","")).upper().startswith(natural.upper())]
                if len(cands2)==1:
                    r = cands2[0]
                    return int(r.get("idx", r.get("internal_idx", r.get("i", -1))))
    return None

def parse_pdb_atoms(pdb_path: str) -> List[Tuple[str,int,str]]:
    seen = set(); seq = []
    if not os.path.isfile(pdb_path): 
        return seq
    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            chain = line[21].strip() or ""
            resi = line[22:26].strip()
            try: resi = int(resi)
            except: continue
            res3 = line[17:20].strip().upper()
            a1 = AA3_TO_1.get(res3, None)
            key = (chain, resi)
            if key not in seen and a1:
                seen.add(key); seq.append((chain, resi, a1))
    return seq

def map_with_pdb(pdb_path: str, logits_len: int, chain: Optional[str], resid: int, natural: Optional[str]) -> Optional[int]:
    seq = parse_pdb_atoms(pdb_path)
    if not seq:
        return None
    for i,(c,r,a) in enumerate(seq):
        if r==resid and (chain is None or c.upper()==chain.upper()):
            if natural and a != natural:
                warnings.warn(f"Natural AA mismatch at resid {resid} in PDB. Using index anyway.")
            return i if i < logits_len else None
    return None

def map_pdb_to_internal(version_name: str, files: Dict[str,str], pdb_hint: Optional[str], mut_token: str) -> Dict:
    info = parse_mutation_token(mut_token)
    chain = info.get("chain"); resid = info["resid"]
    natural = info.get("natural"); mutant = info.get("mutant")
    idx = None
    if files.get("per_res") and os.path.isfile(files["per_res"]):
        try:
            dfr = pd.read_csv(files["per_res"])
            idx = map_with_per_res(dfr, chain, resid, natural)
        except Exception as e:
            warnings.warn(f"[{version_name}] per_residue_scores read/mapping failed: {e}")
    if idx is None and files.get("meta") and os.path.isfile(files["meta"]):
        try:
            meta = json.load(open(files["meta"], "r"))
            idx = map_with_meta(meta, chain, resid, natural)
        except Exception as e:
            warnings.warn(f"[{version_name}] meta.json mapping failed: {e}")
    if idx is None and pdb_hint and os.path.isfile(pdb_hint):
        try:
            L = None
            if files.get("logits"):
                dfL = read_table_maybe_gz(files["logits"])
                L = len(dfL)
            idx = map_with_pdb(pdb_hint, logits_len=L or 10**9, chain=chain, resid=resid, natural=natural)
        except Exception as e:
            warnings.warn(f"[{version_name}] PDB mapping failed: {e}")
    if idx is None:
        raise RuntimeError(f"[{version_name}] Could not map mutation {mut_token} (chain={chain}, resid={resid}) to internal index.")
    return {"internal_idx": int(idx), "chain":chain, "resid":resid, "natural":natural, "mutant":mutant}

# ============= Logits & predictions =============
def load_logits_table(path: str) -> pd.DataFrame:
    df = read_table_maybe_gz(path)
    cols = [c for c in df.columns if len(str(c))==1 and str(c).upper() in AA_ORDER]
    if len(cols) == 20:
        df = df[sorted(cols, key=lambda x: AA_ORDER.index(x.upper()))]
        df.columns = AA_ORDER
        return df
    if "aa" in df.columns and "logit" in df.columns and "pos" in df.columns:
        pivot = df.pivot_table(index="pos", columns="aa", values="logit")
        pivot = pivot.reindex(columns=AA_ORDER)
        return pivot.reset_index(drop=True)
    raise ValueError(f"Unexpected logits table format: {path}")

def load_component_logits_if_any(files: Dict[str,str]) -> Optional[pd.DataFrame]:
    p = files.get("component")
    if not p or not os.path.isfile(p):
        return None
    df = read_table_maybe_gz(p)
    need = {"pos","aa","target_logit","off_logit"}
    if need.issubset({c.lower() for c in df.columns}):
        aa_col = [c for c in df.columns if c.lower()=="aa"][0]
        df[aa_col] = df[aa_col].map(lambda x: x if isinstance(x,str) and len(x)==1 else None)
        return df
    return None

def load_topk_predictions(path: str) -> Optional[List[List[int]]]:
    if not path or not os.path.isfile(path):
        return None
    data = json.load(open(path,"r"))
    if isinstance(data, dict):
        for k in ["top1_indices","argmax_indices","pred_indices"]:
            if k in data and isinstance(data[k], list):
                return data[k]
        for k in ["sequences","seqs"]:
            if k in data and isinstance(data[k], list):
                seqs = data[k]; out = []
                for s in seqs:
                    if isinstance(s,str):
                        out.append([AA_ORDER.index(ch) if ch in AA_ORDER else -1 for ch in s])
                return out
    elif isinstance(data, list) and data and isinstance(data[0], list):
        return data
    return None

# ============= Plotting =============
def plot_grouped_bars_key_only(mutation_label: str, pos_label: str, logits_by_version: Dict[str, np.ndarray], natural: Optional[str], mutant: Optional[str], out_path: str):
    versions = [k for k in ["v0","v1","v2"] if k in logits_by_version]
    if not versions:
        versions = list(logits_by_version.keys())
    n = len(AA_ORDER)
    x = np.arange(n)
    width = 0.8 / max(1,len(versions))
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, v in enumerate(versions):
        bars = ax.bar(x + i*width, logits_by_version[v], width=width, label=v)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.1,
                f"{height:.2f}",
                ha='center', va='bottom', fontsize=8, rotation=90
            )    
    ax.set_xticks(x + (len(versions)-1)*width/2); ax.set_xticklabels(AA_ORDER)
    ax.set_ylabel("Logit"); ax.set_title(f"{mutation_label} @ {pos_label}")
    ax.legend()
    if natural and natural in AA_ORDER:
        idx = AA_ORDER.index(natural)
        ax.axvline(idx + (len(versions)-1)*width/2, color="green", linestyle="--", linewidth=2)
    if mutant and mutant in AA_ORDER:
        idx = AA_ORDER.index(mutant)
        ax.axvline(idx + (len(versions)-1)*width/2, color="red", linestyle="-.", linewidth=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_window_heatmap(version_name: str, mutation_label: str, chain: Optional[str], resid: int, positions: List[int], df_logits: pd.DataFrame, out_path: str):
    # Build matrix: rows=AA (20), cols=positions
    mat = np.zeros((20, len(positions)), dtype=float)
    for j, pos in enumerate(positions):
        if 0 <= pos < len(df_logits):
            mat[:, j] = df_logits.iloc[pos][AA_ORDER].to_numpy(dtype=float)
        else:
            mat[:, j] = np.nan
    fig, ax = plt.subplots(figsize=(1.2*len(positions)+4, 7))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(20)); ax.set_yticklabels(AA_ORDER)
    col_labels = [str(p) for p in positions]
    ax.set_xticks(np.arange(len(positions))); ax.set_xticklabels(col_labels, rotation=0)
    ax.set_xlabel("Internal position"); ax.set_ylabel("AA")
    ax.set_title(f"{mutation_label} | {version_name} | Pocket window (±)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)

# ============= Main analysis =============
def ensure_dirs(base):
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base,"plots"), exist_ok=True)
    os.makedirs(os.path.join(base,"csv"), exist_ok=True)
    os.makedirs(os.path.join(base,"logs"), exist_ok=True)

def main():
    p = argparse.ArgumentParser()
    # p.add_argument('--mode', choices=['single_ref','multi_pdb'], required=True)
    p.add_argument('--orig_pt')
    p.add_argument('--v1_pt')
    p.add_argument('--v2_pt')
    p.add_argument('--key_mutations', nargs="+", required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--pdb_path', help="Original PDBs for mapping (optional)")
    args = p.parse_args()

    versions = {
        "v0": {"dir": args.orig_pt},
        "v1": {"dir": args.v1_pt},
        "v2": {"dir": args.v2_pt},
    }
    window = 5
    ensure_dirs(args.outdir)
    log_lines = []
    stamp = datetime.datetime.now().isoformat(timespec="seconds")

    # Discover files per version (dirs are supported)
    version_files = {}
    for v, entry in versions.items():
        files = discover_files(entry)
        version_files[v] = files
        log_lines.append(f"[discover] {v}: {files}")

    comp_logits = {v: load_component_logits_if_any(files) for v,files in version_files.items()}
    logits_tables = {}
    for v, files in version_files.items():
        if not files.get("logits") or not os.path.isfile(files["logits"]):
            raise FileNotFoundError(f"{v}: logits file not found")
        logits_tables[v] = load_logits_table(files["logits"])
    topk_by_version = {v: load_topk_predictions(files.get("topk")) for v,files in version_files.items()}

    per_batch_rows = []  # mutation, version, chain, resid, internal_idx, batch, pred_aa, pred_logit
    summary_rows = []    # mutation, version, chain, resid, internal_idx, majority_pred_aa, majority_count, mean_pred_logit
    agreement_rows = []  # mutation, vA, vB, n_batches_compared, agreement_rate

    for mut in args.key_mutations:
        maps = {}
        for v, files in version_files.items():
            pdb_hint = args.pdb_path
            maps[v] = map_pdb_to_internal(v, files, pdb_hint, mut)

        idxs = [m["internal_idx"] for m in maps.values()]
        if len(set(idxs)) > 1:
            warnings.warn(f"[{mut}] Internal indices differ across versions: {maps}")
        ref = list(maps.values())[0]
        center_idx = ref["internal_idx"]
        natural = ref["natural"]; mutant = ref["mutant"]; chain = ref["chain"]; resid = ref["resid"]
        win = window
        positions = list(range(max(0, center_idx - win), center_idx + win + 1))

        # --- Key-index only grouped bar (per request) ---
        logits_by_version = {}
        for v, df in logits_tables.items():
            if 0 <= center_idx < len(df):
                logits_by_version[v] = df.iloc[center_idx].to_numpy(dtype=float)
        pos_label = f"{chain or ''}{resid} (internal {center_idx})"
        plot_path = os.path.join(args.outdir, "plots", f"{mut}_keyidx_grouped_logits.png")
        plot_grouped_bars_key_only(mut, pos_label, logits_by_version, natural, mutant, plot_path)

        # --- Pocket heatmaps for each version ---
        for v, df in logits_tables.items():
            heat_out = os.path.join(args.outdir, "plots", f"{mut}_{v}_pocket_heatmap.png")
            plot_window_heatmap(v, mut, chain, resid, positions, df, heat_out)

        # --- Per-batch predictions @ key site ---
        for v, files in version_files.items():
            df = logits_tables[v]; key_pos = maps[v]["internal_idx"]
            if not (0 <= key_pos < len(df)): continue
            preds = topk_by_version.get(v)
            if preds is not None:
                for b, seq in enumerate(preds):
                    if key_pos < len(seq) and 0 <= seq[key_pos] < 20:
                        aa_idx = int(seq[key_pos])
                        aa_char = AA_ORDER[aa_idx]
                        logit_val = float(df.iloc[key_pos, aa_idx])
                        per_batch_rows.append([mut, v, chain or "", resid, key_pos, b, aa_char, logit_val])
            else:
                row = df.iloc[key_pos].to_numpy(dtype=float)
                aa_idx = int(np.argmax(row)); aa_char = AA_ORDER[aa_idx]; logit_val = float(row[aa_idx])
                per_batch_rows.append([mut, v, chain or "", resid, key_pos, 0, aa_char, logit_val])

        # --- Summary per version ---
        dfpb = pd.DataFrame(per_batch_rows, columns=["mutation","version","chain","resid","internal_idx","batch","pred_aa","pred_logit"])
        sub = dfpb[dfpb["mutation"]==mut]
        for v in sub["version"].unique():
            sv = sub[sub["version"]==v]
            if sv.empty: continue
            mean_logit = float(sv["pred_logit"].mean())
            maj = sv["pred_aa"].value_counts().idxmax()
            maj_count = int(sv["pred_aa"].value_counts().max())
            summary_rows.append([mut, v, chain or "", resid, maps[v]["internal_idx"], maj, maj_count, mean_logit])

        # --- Pairwise agreement across versions (key site, batch-wise) ---
        # Compare by batch index intersection. If batch counts differ, use min count.
        versions = list(version_files.keys())
        for i in range(len(versions)):
            for j in range(i+1, len(versions)):
                va, vb = versions[i], versions[j]
                sva = sub[sub["version"]==va].sort_values("batch")
                svb = sub[sub["version"]==vb].sort_values("batch")
                n = min(len(sva), len(svb))
                if n == 0: 
                    continue
                aa_a = sva.iloc[:n]["pred_aa"].to_numpy()
                aa_b = svb.iloc[:n]["pred_aa"].to_numpy()
                agree = float((aa_a == aa_b).mean())
                agreement_rows.append([mut, va, vb, int(n), agree])

    # Write outputs
    pd.DataFrame(per_batch_rows, columns=["mutation","version","chain","resid","internal_idx","batch","pred_aa","pred_logit"]).to_csv(os.path.join(args.outdir, "csv", "keysite_per_batch_predictions.csv"), index=False)
    pd.DataFrame(summary_rows, columns=["mutation","version","chain","resid","internal_idx","majority_pred_aa","majority_count","mean_pred_logit"]).to_csv(os.path.join(args.outdir, "csv", "keysite_summary.csv"), index=False)
    if agreement_rows:
        pd.DataFrame(agreement_rows, columns=["mutation","version_A","version_B","n_batches_compared","agreement_rate"]).to_csv(os.path.join(args.outdir, "csv", "keysite_pairwise_agreement.csv"), index=False)

    # Log
    with open(os.path.join(args.outdir,"logs","run.log"), "w") as f:
        f.write("\n".join([f"[discover] {v}: {version_files[v]}" for v in version_files] + [f"[done] {datetime.datetime.now().isoformat(timespec='seconds')}"]))

if __name__ == "__main__":
    main()