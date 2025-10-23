import numpy as np
import pandas as pd
import os, glob, sys
import argparse

def _choose_plddt(df: pd.DataFrame):
    if "plddt" in df.columns and df["plddt"].notna().any():
        return df["plddt"].to_numpy()
    return df["plddt_pde"].to_numpy()

def filter_self(df: pd.DataFrame, topk: int = 15) -> pd.DataFrame:
    rmsd   = df["rmsd"].to_numpy()
    prmsd  = df["prmsd"].to_numpy()
    kabsch = df["kabsch"].to_numpy() if "kabsch" in df.columns else None
    plddt  = _choose_plddt(df)

    # strict
    # mask = (rmsd <= 2.0) & (plddt >= 0.75) & (prmsd < 3.0)
    # if kabsch is not None:
    #     mask &= (kabsch <= 0.8)
    # loose
    mask = (rmsd <= 2.0)

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return df.iloc[[]]

    # alignment key: plddt(desc) → prmsd(asc) → rmsd(asc)
    key1 = -plddt[idx]
    key2 =  prmsd[idx]
    key3 =  rmsd[idx]
    ord_idx = np.lexsort((key3, key2, key1))
    pick = idx[ord_idx[:min(topk, idx.size)]]
    return df.iloc[pick]

def filter_cross(df: pd.DataFrame, topk: int = 30):
    rmsd   = df["rmsd"].to_numpy()
    prmsd  = df["prmsd"].to_numpy()
    kabsch = df["kabsch"].to_numpy() if "kabsch" in df.columns else None
    plddt  = _choose_plddt(df)

    # strict
    # mask = (rmsd <= 6.0) & (plddt >= 0.70) & (prmsd < 4.0)
    # if kabsch is not None:
    #     mask &= (kabsch <= 1.0)
    # loose
    mask = (rmsd <= 8.0) & (plddt >= 0.65)


    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return df.iloc[[]]

    # alignment key: plddt(desc) → prmsd(asc)
    key1 = -plddt[idx]
    key2 =  prmsd[idx]
    ord_idx = np.lexsort((key2, key1))
    pick = idx[ord_idx[:min(topk, idx.size)]]
    return df.iloc[pick]

def rank_topk_self(df: pd.DataFrame, topk=15) -> pd.DataFrame:
    plddt = _choose_plddt(df)
    prmsd = df["prmsd"].to_numpy()
    rmsd  = df["rmsd"].to_numpy()
    ok = np.isfinite(plddt) & np.isfinite(prmsd) & np.isfinite(rmsd)
    idx = np.flatnonzero(ok)
    if idx.size == 0: return df.iloc[[]]
    key1, key2, key3 = -plddt[idx], prmsd[idx], rmsd[idx]
    ord_idx = np.lexsort((key3, key2, key1))
    return df.iloc[idx[ord_idx[:min(topk, idx.size)]]]

def rank_topk_cross(df: pd.DataFrame, topk=30) -> pd.DataFrame:
    plddt = _choose_plddt(df)
    prmsd = df["prmsd"].to_numpy()
    ok = np.isfinite(plddt) & np.isfinite(prmsd)
    idx = np.flatnonzero(ok)
    if idx.size == 0: return df.iloc[[]]
    key1, key2 = -plddt[idx], prmsd[idx]
    ord_idx = np.lexsort((key2, key1))
    return df.iloc[idx[ord_idx[:min(topk, idx.size)]]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)
    p.add_argument("--type", choices=["tar", "off"], required=True)
    p.add_argument("--mode", choices=["self", "cross"], required=True)
    p.add_argument("--method", choices=["filter", "rank"], default="rank")
    p.add_argument("--topk", type=int, default=None)
    args = p.parse_args()

    # Display the filtering criteria
    lines = [
        ["Mode", "rmsd", "plddt", "prmsd", "kabsch", "Top-k"],
        ["self",  "<=2.0 Å", ">=0.75", "<3.0", "<=0.8", "15"],
        ["cross", "<=6.0 Å", ">=0.70", "<4.0", "<=1.0", "30"]
    ]
    widths = [max(len(x) for x in col) for col in zip(*lines)]

    print("\n   === PLACER Ensemble Filtering Criteria ===")
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for i, row in enumerate(lines):
        print(" | ".join(x.ljust(w) for x, w in zip(row, widths)))
        if i == 0:
            print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))

    summary_rows = []
    USECOLS = ["rmsd","prmsd","plddt","plddt_pde","kabsch"]
    is_self = (args.mode == 'self')
    is_rank = (args.method == 'rank')
    topk = args.topk if args.topk is not None else (15 if is_self else 30)

    subdirs = sorted([d for d in glob.glob(os.path.join(args.base_dir, "*"))])
    for sub in subdirs:
        placer_dir = os.path.join(sub, f"placer_{args.type}")
        filter_dir = os.path.join(sub, f"filter_{args.type}")

        if not (os.path.isdir(placer_dir)):
            continue
        os.makedirs(filter_dir, exist_ok=True)
        print(f"📁 Subdir: {os.path.basename(sub)}")

        files = sorted(glob.glob(os.path.join(placer_dir, "**", "*.csv"), recursive=True))
        if not files:
            print(f"no files: {placer_dir}/*.csv", file=sys.stderr)
            continue

        for file in files:
            header_cols = pd.read_csv(file, nrows=0).columns
            usecols = [c for c in USECOLS if c in header_cols]
            if not set(["rmsd","prmsd"]).issubset(usecols):
                print(f"required columns missing in {file}", file=sys.stderr)
                continue

            df = pd.read_csv(file, usecols=usecols, dtype={c:"float32" for c in usecols})
            filtered_df = filter_self(df, topk=topk) if is_self else filter_cross(df, topk=topk)
            
            # Save filtered csv
            stem = os.path.splitext(os.path.basename(file))[0]
            filter_path = os.path.join(filter_dir, stem + "_filtered.csv")
            filtered_df.to_csv(filter_path, index=False)
            print(f"[{args.type}][{args.mode}] {os.path.basename(file)} ({len(filtered_df)}/{len(df)})")

            if is_rank:
                rank_df = rank_topk_self(df, topk=topk) if is_self else rank_topk_cross(df, topk=topk)
                ranked_indices = rank_df.index.to_list()
            rank_path = os.path.join(filter_dir, stem + "_ranked.csv")
            rank_df.to_csv(rank_path, index=False, float_format="%.4f")

            # Summary csv
            label = os.path.splitext(os.path.basename(file))[0]
            filtered_idx_list = [i+1 for i in filtered_df.index.to_list()] # PDB MODEL is 1-based
            filtered_indices_str = ",".join(map(str, filtered_idx_list))
            ranked_idx_list = [i+1 for i in rank_df.index.to_list()] # PDB MODEL is 1-based
            ranked_indices_str = ",".join(map(str, ranked_idx_list))

            summary_rows.append({
                "label": label,
                "type": args.type,
                "mode": args.mode,
                "filtered_count": len(filtered_df),
                "filtered_model_indices": filtered_indices_str,
                "ranked_count": topk,
                "ranked_model_indices": ranked_indices_str,
            })
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows, columns=[
            "label", "type", "mode", 
            "filtered_count", "filtered_model_indices",
            "ranked_count", "ranked_model_indices",
            ])
        summary_path = os.path.join(args.base_dir, f"ensemble_{args.type}_summary.csv")
        summary_df.to_csv((summary_path), index=False)
        print(f"Summary csv is saved as {summary_path}")
    else:
        print("no rows collected")


if __name__ == '__main__':
    main()
