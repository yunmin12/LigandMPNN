import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, NamedTuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET: Set[str] = set(AA_LIST)
AA_INDEX = {a: i for i, a in enumerate(AA_LIST)}


def softmax_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    s = e.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return e / s


def read_gz_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="infer")


def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def discover_pdb_ids(results_dir: Path) -> List[str]:
    ids = set()
    for p in results_dir.glob("trimmed_*_packed_*_1_*"):
        m = re.match(r"trimmed_([A-Za-z0-9]+)_packed_", p.name)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def discover_replicates(results_dir: Path, pdb_id: str) -> Dict[str, Dict[str, Path]]:
    patt = f"trimmed_{pdb_id}_packed_"
    files = list(results_dir.glob(patt + "*_1_*"))
    out: Dict[str, Dict[str, Path]] = {}
    for p in files:
        name = p.name
        m = re.search(r"packed_(\d+)_1_([A-Za-z0-9_.]+)$", name)
        if not m:
            continue
        rep = m.group(1)
        suffix = m.group(2)
        d = out.setdefault(rep, {})
        if suffix.endswith(".csv.gz") and suffix.startswith("logits"):
            d["logits"] = p
        elif suffix.endswith(".csv.gz") and suffix.startswith("log_probs"):
            d["log_probs"] = p
        elif suffix == "per_residue_scores.csv":
            d["per_res"] = p
        elif suffix == "meta.json":
            d["meta"] = p
        elif suffix == "topk.json":
            d["topk"] = p
        elif suffix.endswith(".pt"):
            d["pt"] = p
    return out


def pivot_any(df: pd.DataFrame, value_col: str):
    # Returns (arr[L,20], aa_cols)
    aa_cols = [c for c in df.columns if str(c).upper() in AA_SET]
    if aa_cols:
        aa_cols = [c.upper() for c in aa_cols]
        return df[aa_cols].to_numpy(copy=True), aa_cols
    cols = {c.lower(): c for c in df.columns}
    idx_key = None
    for k in ["idx", "pos", "position"]:
        if k in cols:
            idx_key = cols[k]
            break
    aa_key = cols.get("aa")
    val_key = cols.get(value_col) or cols.get(value_col.lower())
    if idx_key and aa_key and val_key:
        wide = df.pivot(index=idx_key, columns=aa_key, values=val_key)
        order = [a for a in AA_LIST if a in wide.columns]
        wide = wide[order]
        return wide.to_numpy(copy=True), order
    raise ValueError(f"Cannot interpret dataframe for value {value_col}")


def load_logits_probs(paths: Dict[str, Path]):
    logits = None
    probs = None
    aa_cols = AA_LIST
    if "log_probs" in paths:
        dfp = read_gz_csv(paths["log_probs"])
        arr, aa_cols = pivot_any(dfp, "log_prob")
        probs = np.exp(arr)
        probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    if "logits" in paths:
        dfl = read_gz_csv(paths["logits"])
        arr, aa_cols_l = pivot_any(dfl, "logit")
        logits = arr
        if probs is None:
            probs = softmax_rows(logits)
        aa_cols = aa_cols_l
    if (logits is None or probs is None) and "per_res" in paths:
        dfr = pd.read_csv(paths["per_res"])  # long: idx,residue,aa,logit,log_prob
        try:
            arr, aa_cols = pivot_any(dfr, "log_prob")
            probs = np.exp(arr)
            probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        except Exception:
            pass
        try:
            logits, _ = pivot_any(dfr, "logit")
        except Exception:
            pass
    if logits is None and probs is None:
        raise FileNotFoundError("No usable logits/log_probs/per_residue_scores")
    return logits, probs, aa_cols


def meta_idx_to_pdb_labels(meta_path: Optional[Path], L: int):
    labels = [f"X{i+1}" for i in range(L)]
    if meta_path and meta_path.exists():
        try:
            md = read_json(meta_path)
            rn = md.get("residue_names")
            if isinstance(rn, dict):
                tmp = {}
                for k, v in rn.items():
                    try:
                        i = int(k)
                    except Exception:
                        continue
                    if 0 <= i < L:
                        tmp[i] = str(v)
                if tmp:
                    labels = [tmp.get(i, labels[i]) for i in range(L)]
        except Exception:
            pass
    inv = {lab: i for i, lab in enumerate(labels)}
    return labels, inv

class Mut(NamedTuple):
    chain: str
    pos: int
    wt: Optional[str]
    mt: str

_mut_pat = re.compile(r"^(?P<chain>[A-Za-z0-9]):(?P<pos>\d+):(?:(?P<wt>[A-Za-z])>)?(?P<mt>[A-Za-z])$")

def parse_mut(s: str) -> Mut:
    m = _mut_pat.match(s.strip())
    if not m:
        raise ValueError(f"Bad --mutation format: {s}. Use A:199:E>V or A:199:V")
    chain = m.group("chain")
    pos = int(m.group("pos"))
    wt = m.group("wt")
    mt = m.group("mt").upper()
    if wt:
        wt = wt.upper()
        if wt not in AA_SET:
            raise ValueError(f"Invalid WT AA: {wt}")
    if mt not in AA_SET:
        raise ValueError(f"Invalid MT AA: {mt}")
    return Mut(chain, pos, wt, mt)


def analyze_pdb(results_dir: Path, pdb_id: str, mut: Mut, out_dir: Path):
    reps = discover_replicates(results_dir, pdb_id)
    if not reps:
        print(f"[WARN] No replicates for {pdb_id}")
        return None

    first = next(iter(reps.values()))
    logits0, probs0, _ = load_logits_probs(first)
    L = probs0.shape[0]
    labels, inv_map = meta_idx_to_pdb_labels(first.get("meta"), L)

    key_label = f"{mut.chain}{mut.pos}"
    key_ix = inv_map.get(key_label) or inv_map.get(str(mut.pos))
    if key_ix is None:
        print(f"[WARN] {pdb_id}: cannot map {key_label} to idx; skipping")
        return None

    per_pos_records = []
    key_counts = {aa: 0 for aa in AA_LIST}
    total_reps = 0
    
    key_logit_sums = {aa: 0.0 for aa in AA_LIST}
    key_logit_ns = {aa: 0 for aa in AA_LIST}
    
    key_logprob_sums = {aa: 0.0 for aa in AA_LIST}
    key_logprob_ns   = {aa: 0   for aa in AA_LIST}

    for rep_id, paths in sorted(reps.items(), key=lambda kv: int(kv[0])):
        try:
            logits, probs, aa_cols = load_logits_probs(paths)
        except Exception as e:
            print(f"[WARN] skip {pdb_id} rep {rep_id}: {e}")
            continue
        total_reps += 1
        argmax_idx = np.argmax(probs, axis=1)
        chosen_aa = [aa_cols[i] for i in argmax_idx]
        chosen_prob = probs[np.arange(L), argmax_idx]
        chosen_logit = logits[np.arange(L), argmax_idx] if logits is not None else np.full(L, np.nan)
        row_logits = logits[key_ix]
        for j, aa in enumerate(aa_cols):
            a = aa.upper()
            if a in AA_SET:
                key_logit_sums[a] += float(row_logits[j])
                key_logit_ns[a] += 1
        
        row = probs[key_ix]
        for j, aa in enumerate(aa_cols):
            a = aa.upper()
            if a in AA_SET:
                p = float(row[j])
                lp = np.log(max(p, 1e-12))
                key_logprob_sums[a] += lp
                key_logprob_ns[a] += 1
        
        for i in range(L):
            per_pos_records.append({
                "pdb_id": pdb_id,
                "replicate": int(rep_id),
                "idx": i,
                "pdb_label": labels[i],
                "chosen_aa": chosen_aa[i],
                "chosen_prob": float(chosen_prob[i]),
                "chosen_logit": float(chosen_logit[i])
            })

        aa_k = chosen_aa[key_ix]
        if aa_k in key_counts:
            key_counts[aa_k] += 1

    if total_reps == 0:
        print(f"[WARN] {pdb_id}: no usable replicates")
        return None

    mean_logit = {aa: (key_logit_sums[aa] / key_logit_ns[aa] if key_logit_ns[aa] else np.nan)
                  for aa in AA_LIST}
    mean_lp = {aa: (key_logprob_sums[aa] / key_logprob_ns[aa] 
                    if key_logprob_ns[aa] else np.nan)
                    for aa in AA_LIST}

    per_pos_df = pd.DataFrame(per_pos_records)

    # 1) Per-residue mean chosen prob (x = PDB labels)
    m = per_pos_df.groupby(["idx", "pdb_label"])['chosen_prob'].mean().reset_index()
    per_pos_df = per_pos_df["idx"]
    fig1 = plt.figure(figsize=(14, 4))
    ax1 = fig1.gca()
    ax1.plot(m['idx'], m['chosen_prob'])
    ax1.set_xlabel("Residue (PDB numbering)")
    ax1.set_ylabel("Mean chosen prob across reps")
    ax1.set_title(f"{pdb_id} {mut.chain}: per-residue confidence (mean)")
    ax1.axvline(key_ix, linestyle='--')
    tick_idx = np.linspace(0, L-1, num=min(15, max(5, L//5)), dtype=int)
    ax1.set_xticks(tick_idx)
    ax1.set_xticklabels([m.loc[m['idx']==i, 'pdb_label'].iloc[0] if (m['idx']==i).any() else labels[i] for i in tick_idx], rotation=45, ha='right')
    fig1.tight_layout()
    fig1.savefig(out_dir / f"{pdb_id}_{mut.chain}_per_residue_confidence.png", dpi=180)
    plt.close(fig1)

    half_w = 20
    lo = max(0, key_ix - half_w)
    hi = min(L - 1, key_ix + half_w)

    m_crop = m[(m['idx'] >= lo) & (m['idx'] <= hi)].copy()
    if m_crop.empty:
        pass
    else: 
        fig2_2 = plt.figure(figsize=(10, 3.2))
        axc = fig2_2.gca()
        axc.plot(m_crop['idx'], m_crop['chosen_prob'])
        axc.set_xlabel(f"Residue (PDB numbering)  |  window: ±{half_w}")
        axc.set_ylabel("Mean chosen prob across reps")
        axc.set_title(f"{pdb_id} {mut.chain}: per-residue confidence (crop ±{half_w})")
        axc.axvline(key_ix, linestyle='--')

        tick_idx = np.linspace(lo, hi, num=min(15, max(5, (hi - lo + 1)//2)), dtype=int)
        axc.set_xticks(tick_idx)
        axc.set_xticklabels(
            [m.loc[m['idx']==i, 'pdb_label'].iloc[0] if (m['idx']==i).any() else labels[i]
                for i in tick_idx],
            rotation=45, ha='right'
        )
        fig2_2.tight_layout()
        fig2_2.savefig(out_dir / f"{pdb_id}_{mut.chain}_per_residue_confidence_crop{half_w}.png", dpi=180)
        plt.close(fig2_2)

    # 2) Key-site AA count bar (counts/total) per PDBs
    fracs = {aa: key_counts[aa] / float(total_reps) for aa in AA_LIST}

    mean_lp = {
        aa: (key_logprob_sums[aa] / key_logprob_ns[aa]) if key_logprob_ns[aa] > 0 else np.nan
        for aa in AA_LIST
    }

    xs = np.arange(len(AA_LIST))
    vals = np.array([fracs[a] for a in AA_LIST])
    fig2 = plt.figure(figsize=(10, 4))
    ax2 = fig2.gca()
    bars = ax2.bar(xs, vals)

    if mut.mt in AA_INDEX:
        bars[AA_INDEX[mut.mt]].set_hatch('///')
        bars[AA_INDEX[mut.mt]].set_linewidth(1.5)
    
    ax2.set_xticks(xs)
    ax2.set_xticklabels(AA_LIST)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("# of AA / total")
    ax2.set_title(f"{pdb_id}: key-site AA choices @ {mut.chain}{mut.pos} (N={total_reps})")
    
    labels = []
    for idx, rect in enumerate(bars):
        h = rect.get_height()
        aa = AA_LIST[idx]
        if h > 0:
            lp = mean_lp.get(aa, float("nan"))
            label = f"{aa}\n{lp:.2f}" if np.isfinite(lp) else f"{aa}\nNA"
        else:
            label = ""
        labels.append(label)
    ax2.bar_label(bars, labels=labels, label_type='edge', padding=2, fontsize=8)

    fig2.tight_layout()
    mut_id = f"{mut.chain}{mut.pos}_" + (f"{mut.wt}_{mut.mt}" if mut.wt else f"{mut.mt}")
    fig2.savefig(out_dir / f"{pdb_id}_{mut_id}_keysite_aa_counts.png", dpi=180)
    plt.close(fig2)

    key_rows = [{
        "pdb_id": pdb_id,
        "chain": mut.chain,
        "pos": mut.pos,
        "mut_aa": mut.mt,
        "wt_aa": mut.wt if mut.wt else "",
        "aa": aa,
        "count": key_counts[aa],
        "fraction": fracs[aa],
        "total_reps": total_reps,
        "mean_log_prob": float(mean_lp[aa]) if np.isfinite(mean_lp[aa]) else np.nan,
        "mean_logit": float(mean_logit[aa]) if np.isfinite(mean_logit[aa]) else np.nan  # ← NEW
    } for aa in AA_LIST]

    return {
        "per_pos_df": per_pos_df,
        "key_count_rows": key_rows
    }


def plot_key_aa_ratio(key_df: pd.DataFrame, out_dir: str):
    sites = key_df[['chain', 'pos']].drop_duplicates().to_records(index=False)
    multi_site = len(sites) > 1

    for chain, pos in sites:
        df_site = key_df[(key_df['chain'] == chain) & (key_df['pos'] == pos)].copy()

        if 'mean_log_prob' not in df_site.columns:
            df_site['mean_log_prob'] = np.nan

        agg = (df_site
               .groupby(['pdb_id', 'aa'], as_index=False)
               .agg(fraction=('fraction', 'max'),
                    mean_log_prob=('mean_log_prob', 'mean')))

        aas = sorted(agg['aa'].unique())
        pdbs = sorted(agg['pdb_id'].unique())

        plot_df = (agg.pivot_table(index='pdb_id', columns='aa', values='fraction',
                                   fill_value=0.0, aggfunc='max')
                        .reindex(index=pdbs, columns=aas))

        lp_wide = (agg.pivot_table(index='pdb_id', columns='aa', values='mean_log_prob',
                                   fill_value=np.nan, aggfunc='mean')
                        .reindex(index=plot_df.index, columns=plot_df.columns))

        site_rows = key_df[(key_df['chain'] == chain) & (key_df['pos'] == pos)]
        wt_candidates = site_rows['wt_aa'].dropna().replace('', pd.NA).dropna().unique()
        mt_candidates = site_rows['mut_aa'].dropna().replace('', pd.NA).dropna().unique()
        wt = str(wt_candidates[0]) if len(wt_candidates) else ""
        mt = str(mt_candidates[0]) if len(mt_candidates) else ""
        mut_label = f"{chain}:{pos}:{wt}>{mt}" if wt else f"{chain}:{pos}:{mt}"

        ax = plot_df.plot(kind='bar', figsize=(12, 5))
        ax.set_xlabel("PDB ID")
        ax.set_ylabel("AA Ratio at key site (count / total files)")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Key-site AA ratios per PDB @ {chain}{pos}  |  {mut_label}")
        ax.legend(title="amino acid", bbox_to_anchor=(1.02, 1), loc="upper left")

        for aa, container in zip(plot_df.columns, ax.containers):
            lp_series = lp_wide[aa]
            labels = []
            for i, v in enumerate(container.datavalues):
                lp = lp_series.iloc[i]
                if v > 0:
                    if pd.notna(lp):
                        p = float(np.exp(lp))
                        labels.append(f"{aa}\n{v:.2f}\n{p:.2f}")
                    else:
                        labels.append(f"{aa}\n{v:.2f}\nNA")
                else:
                        labels.append("")
            ax.bar_label(container, labels=labels, label_type='center', padding=2, fontsize=8)

        ax.text(
            0.01, 0.98, mut_label,
            transform=ax.transAxes, ha='left', va='top', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round')
        )

        fig = ax.get_figure()
        fig.tight_layout()
        out_name = f"key_aa_ratio_{chain}{pos}.png" if multi_site else "key_aa_ratio.png"
        fig.savefig(out_dir / out_name, dpi=180)
        plt.close(fig)

def plot_key_aa_logits(key_df: pd.DataFrame, out_dir: str):
    sites = key_df[['chain','pos']].drop_duplicates().to_records(index=False)
    multi_site = len(sites) > 1

    for chain, pos in sites:
        df_site = key_df[(key_df['chain']==chain)&(key_df['pos']==pos)].copy()
        if 'mean_logit' not in df_site.columns:
            print(f"[WARN] mean_logit not found for site {chain}{pos}; skip")
            continue
        
        agg = (df_site.groupby(['aa', 'pdb_id'], as_index=False)
                      .agg(mean_logit=('mean_logit','mean')))

        aas = sorted(agg['aa'].unique())
        pdbs = sorted(agg['pdb_id'].unique())

        logit_wide = (agg.pivot_table(index='aa', columns='pdb_id', values='mean_logit',
                                      fill_value=np.nan, aggfunc='mean')
                         .reindex(index=aas, columns=pdbs))

        wt = df_site['wt_aa'].dropna().replace('', pd.NA).dropna().unique()
        mt = df_site['mut_aa'].dropna().replace('', pd.NA).dropna().unique()
        wt = str(wt[0]) if len(wt)>0 else None
        mt = str(mt[0]) if len(mt)>0 else None
        if wt or mt:
            order = []
            for x in [wt, mt]:
                if x and x in logit_wide.index and x not in order:
                    order.append(x)
            order += [a for a in logit_wide.index if a not in order]
            logit_wide = logit_wide.reindex(index=order)
        
        ax = logit_wide.plot(kind='bar', figsize=(12,5))
        ax.set_xlabel("Amino acid at key mutation site")
        ax.set_ylabel("Mean logit at key site")
        ax.set_title(f"Key-site AA logits per PDB @ {chain}{pos}")
        ax.legend(title="PDB", bbox_to_anchor=(1.02,1), loc="upper left")
        ax.tick_params(axis='x', rotation=0)

        for pdb, container in zip(logit_wide.columns, ax.containers):
            labels = [f"{v:.2f}" if not np.isnan(v) else "" for v in container.datavalues]
            ax.bar_label(container, labels=labels, label_type='edge', padding=2, fontsize=8)
        mut_label = f"{chain}:{pos}:{(wt or '')}>{(mt or '')}" if wt else f"{chain}:{pos}:{(mt or '')}"
        ax.text(0.01, 0.98, mut_label, transform=ax.transAxes, ha='left', va='top',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round'))
        
        fig = ax.get_figure()
        fig.tight_layout()
        out_name = f"key_aa_logits_{chain}{pos}.png" if multi_site else "key_aa_logits.png"
        fig.savefig(out_dir/out_name, dpi=180)
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--mutation', action='append', required=True, help='Repeatable. (e.g. A:199:E>V or A:199:V, if WT is unknown)')
    ap.add_argument('--pdb_ids', default=None, help='Comma-separated PDB IDs. Else, auto-discover')
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    muts = [parse_mut(s) for s in args.mutation]
    pdb_ids = [x.strip() for x in args.pdb_ids.split(',')] if args.pdb_ids else discover_pdb_ids(results_dir)
    if not pdb_ids:
        raise SystemExit("No PDB IDs found; provide --pdb_ids or ensure results_dir has trimmed_{PDB}_packed_* files.")

    all_per_pos = []
    all_key_rows = []

    for mut in muts:
        for pdb_id in pdb_ids:
            res = analyze_pdb(results_dir, pdb_id, mut, out_dir)
            if not res:
                continue
            all_per_pos.append(res["per_pos_df"])  # for per_residue_all_reps.csv
            all_key_rows.extend(res["key_count_rows"])  # for keysite_aa_counts.csv
    
    
    
    if all_key_rows:
        key_df = pd.DataFrame(all_key_rows)
        
        sites = key_df[['chain', 'pos']].drop_duplicates().to_records(index=False)
        multi_site = len(sites) > 1
        
        key_aa_counts_out = f"keysite_aa_counts_{mut.chain}{mut.pos}.csv" if multi_site else "keysite_aa_counts.csv"
        key_df.to_csv(out_dir / key_aa_counts_out, index=False)
        
        # 3) Key-site AA count bar (all PDBs)
        plot_key_aa_ratio(key_df, out_dir)
        plot_key_aa_logits(key_df, out_dir)

    if all_per_pos:
        per_pos_cat = pd.concat(all_per_pos, ignore_index=True)
        per_pos_out = f"per_residue_all_reps_{mut.chain}{mut.pos}.csv" if multi_site else "per_residue_all_reps.csv"
        per_pos_cat.to_csv(out_dir / per_pos_out, index=False)

    print("Done. Outputs in:", out_dir)


if __name__ == '__main__':
    main()
