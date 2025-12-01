"""
Merge and rank FastRelax CSV outputs across seeds. (betarelax, off repack ver.)

Expected columns in input CSVs:
  - totalscore (energy)
  - ddg
  - fa_rep (repulsive term; used for light outlier filtering)
  - tag (identifier)
"""

import argparse
import os
import re
import glob
import shutil
import pandas as pd
import numpy as np


def load_csvs(paths):
    dfs = []
    for p in paths:
        if not os.path.isfile(p):
            print(f"[WARN] CSV not found, skipping: {p}")
            continue
        df = pd.read_csv(p)
        base = os.path.basename(p)
        df["source_csv"] = base
        m = re.search(r"_([0-9]+)\.csv$", base)
        seed = m.group(1) if m else None
        if seed is not None:
            df["seed"] = seed
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No valid CSV files loaded.")
    return pd.concat(dfs, ignore_index=True)


def filter_rank(df, top_n):
    needed = [c for c in ("totalscore",) if c in df.columns]
    df = df.dropna(subset=needed).copy()

    if "output_pdb" not in df.columns:
        df["output_pdb"] = None
    def infer_pdb(row):
        if isinstance(row.get("output_pdb"), str) and row["output_pdb"].strip():
            return row["output_pdb"]
        tag = row.get("tag")
        seed = row.get("seed")
        if not isinstance(tag, str) or not tag.strip():
            return None
        if seed and isinstance(seed, str):
            cand = f"{tag}_betarelax_{seed}.pdb"
        else:
            cand = f"{tag}_betarelax.pdb"
        hits = glob.glob(f"**/{cand}", recursive=True)
        if hits:
            return os.path.abspath(hits[0])
        return cand
    df["output_pdb"] = df.apply(infer_pdb, axis=1)

    if "output_pdb" in df.columns:
        df = df.drop_duplicates(subset=["output_pdb"])
    elif "seed" in df.columns:
        df = df.drop_duplicates(subset=["tag", "seed"])
    else:
        df = df.drop_duplicates(subset=["tag"])

    # total score filter
    q1 = df["totalscore"].quantile(0.25)
    q3 = df["totalscore"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    df = df[df["totalscore"] <= upper].copy()

    # fa_rep filter
    if "fa_rep" in df.columns:
        q3_rep = df["fa_rep"].quantile(0.75)
        iqr_rep = df["fa_rep"].quantile(0.75) - df["fa_rep"].quantile(0.25)
        rep_upper = q3_rep + 1.5 * iqr_rep
        df = df[df["fa_rep"] <= rep_upper].copy()

    # ranking: total score, (if exist) ddg
    sort_cols = ["totalscore"]
    if "ddg" in df.columns:
        sort_cols.append("ddg")
    df = df.sort_values(sort_cols, ascending=True)

    return df.head(top_n).copy()


def copy_pdbs(df, out_dir):
    if out_dir is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    if "output_pdb" not in df.columns:
        print("[WARN] output_pdb column not found; skipping PDB copy.")
        return
    seen = set()
    for _, row in df.iterrows():
        src = row.get("output_pdb")
        if not isinstance(src, str) or not os.path.isfile(src):
            print(f"[WARN] output_pdb missing or not found: {src}")
            continue
        base = os.path.basename(src)
        if base in seen:
            continue
        seen.add(base)
        dst = os.path.join(out_dir, base)
        if os.path.exists(dst):
            stem, ext = os.path.splitext(base)
            i = 1
            while True:
                cand = os.path.join(out_dir, f"{stem}_dup{i}{ext}")
                if not os.path.exists(cand):
                    dst = cand
                    break
                i += 1
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(
        description="Merge and rank betarelax CSV outputs (totalscore/ddg-based)."
    )
    ap.add_argument("--summary_csvs", nargs="+", required=True, help="Input CSVs to merge.")
    ap.add_argument("--out_csv", required=True, help="Path to write merged+filtered CSV.")
    ap.add_argument("--out_dir", default=None, help="Optional dir to copy selected PDBs (needs output_pdb column).")
    ap.add_argument("--top_n", type=int, default=50, help="Number of rows to keep.")
    args = ap.parse_args()

    df_all = load_csvs(args.summary_csvs)
    df_top = filter_rank(df_all, args.top_n)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df_top.to_csv(args.out_csv, index=False)
    copy_pdbs(df_top, args.out_dir)

    print(f"[INFO] Loaded {len(df_all)} rows from {len(args.summary_csvs)} CSVs")
    print(f"[INFO] Selected top {len(df_top)} by totalscore (ddg secondary if present)")
    print(f"[INFO] Wrote: {args.out_csv}")
    if args.out_dir:
        print(f"[INFO] Copied PDBs to: {args.out_dir}")


if __name__ == "__main__":
    main()
