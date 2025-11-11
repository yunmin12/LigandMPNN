import numpy as np
import pandas as pd
import os, glob, sys
import argparse
from typing import Optional, Dict, Tuple


def _choose_plddt(df: pd.DataFrame):
    if "plddt" in df.columns and df["plddt"].notna().any():
        return df["plddt"].to_numpy()
    if "plddt_pde" in df.columns:
        return df["plddt_pde"].to_numpy()
    return np.full(len(df), np.nan, dtype=np.float32)


def filter_by_thresholds(
    df: pd.DataFrame,
    rmsd_cutoff: float,
    plddt_cutoff: Optional[float],
) -> pd.DataFrame:
    rmsd = df["rmsd"].to_numpy()
    mask = np.isfinite(rmsd) & (rmsd <= rmsd_cutoff)

    if plddt_cutoff is not None:
        plddt = _choose_plddt(df)
        mask &= np.isfinite(plddt) & (plddt >= plddt_cutoff)

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return df.iloc[[]]
    return df.iloc[idx]


def rank_by_rmsd(df: pd.DataFrame, topk: Optional[int]) -> pd.DataFrame:
    if df.empty:
        return df
    ordered = df.sort_values(by="rmsd", ascending=True)
    if topk is None:
        return ordered
    return ordered.iloc[: min(topk, len(ordered))]


def _load_threshold_overrides(csv_path: str) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("threshold CSV must contain a 'label' column")
    has_rmsd = "rmsd_threshold" in df.columns
    has_plddt = "plddt_threshold" in df.columns
    overrides: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for _, row in df.iterrows():
        label = str(row["label"])
        rmsd_val = float(row["rmsd_threshold"]) if has_rmsd and not pd.isna(row["rmsd_threshold"]) else None
        plddt_val = float(row["plddt_threshold"]) if has_plddt and not pd.isna(row["plddt_threshold"]) else None
        overrides[label] = (rmsd_val, plddt_val)
    return overrides


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)
    p.add_argument("--type", choices=["tar", "off"], required=True)
    p.add_argument("--rmsd", type=float, default=None,
                   help="Default maximum RMSD allowed during filtering (Å).")
    p.add_argument("--plddt", type=float, default=None,
                   help="Default minimum pLDDT allowed during filtering.")
    p.add_argument("--topk", type=int, default=50,
                   help="Maximum number of ranked models to keep.")
    p.add_argument("--threshold_csv", type=str, default=None,
                   help="Optional CSV with per-file thresholds (columns: label,rmsd_threshold,plddt_threshold).")
    args = p.parse_args()

    DEFAULT_RMSD = 10.0
    DEFAULT_PLDDT = 0.5

    rmsd_threshold = args.rmsd if args.rmsd is not None else DEFAULT_RMSD
    plddt_threshold = args.plddt if args.plddt is not None else DEFAULT_PLDDT
    topk = args.topk
    threshold_overrides = _load_threshold_overrides(args.threshold_csv) if args.threshold_csv else {}
    
    print("\n   === PLACER Ensemble Filtering Criteria ===")
    plddt_msg = f"{plddt_threshold:.2f}" if plddt_threshold is not None else "N/A"
    print("Default thresholds:")
    print(f"  - RMSD ≤ {rmsd_threshold:.2f} Å")
    print(f"  - pLDDT ≥ {plddt_msg}")
    print(f"  - Top-k: {topk}")
    if threshold_overrides:
        print(f"Per-file overrides loaded: {len(threshold_overrides)}")

    summary_rows = []
    USECOLS = ["rmsd", "plddt", "plddt_pde"]

    subdirs = sorted([d for d in glob.glob(os.path.join(args.base_dir, "*"))])
    for sub in subdirs:
        placer_dir = os.path.join(sub, f"placer_{args.type}_large")
        filter_dir = os.path.join(sub, f"filter_{args.type}_large")

        if not os.path.isdir(placer_dir):
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
            if "rmsd" not in usecols:
                print(f"'rmsd' column missing in {file}", file=sys.stderr)
                continue

            label = os.path.splitext(os.path.basename(file))[0]
            override = threshold_overrides.get(label, (None, None))
            file_rmsd = override[0] if override[0] is not None else rmsd_threshold
            file_plddt = override[1] if override[1] is not None else plddt_threshold
            if file_rmsd is None:
                print(f"rmsd threshold missing for {label}, skipping", file=sys.stderr)
                continue

            df = pd.read_csv(file, usecols=usecols, dtype={c: "float32" for c in usecols})
            filtered_df = filter_by_thresholds(df, file_rmsd, file_plddt)

            # Save filtered csv
            stem = os.path.splitext(os.path.basename(file))[0]
            filter_path = os.path.join(filter_dir, stem + "_filtered_large.csv")
            filtered_df.to_csv(filter_path, index=False)
            log_msg = f"[{args.type}] {os.path.basename(file)} ({len(filtered_df)}/{len(df)})"
            log_msg += f" rmsd≤{file_rmsd:.2f}"
            if file_plddt is not None:
                log_msg += f" plddt≥{file_plddt:.2f}"
            print(log_msg)

            rank_df = rank_by_rmsd(filtered_df, topk=topk)
            rank_path = os.path.join(filter_dir, stem + "_ranked_large.csv")
            rank_df.to_csv(rank_path, index=False, float_format="%.4f")

            # Summary csv
            filtered_idx_list = filtered_df.index.to_list()  # PDB MODEL is 0-based
            filtered_indices_str = ",".join(map(str, filtered_idx_list))
            ranked_idx_list = rank_df.index.to_list()
            ranked_indices_str = ",".join(map(str, ranked_idx_list))

            summary_rows.append({
                "label": label,
                "type": args.type,
                "filtered_count": len(filtered_df),
                "filtered_model_indices": filtered_indices_str,
                "ranked_count": len(ranked_idx_list),
                "ranked_model_indices": ranked_indices_str,
                "rmsd_threshold": file_rmsd,
                "plddt_threshold": file_plddt if file_plddt is not None else "",
            })
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows, columns=[
            "label", "type",
            "filtered_count", "filtered_model_indices",
            "ranked_count", "ranked_model_indices",
            "rmsd_threshold", "plddt_threshold",
        ])
        summary_path = os.path.join(args.base_dir, f"ensemble_{args.type}_summary_large.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary csv is saved as {summary_path}")
    else:
        print("no rows collected")


if __name__ == '__main__':
    main()
