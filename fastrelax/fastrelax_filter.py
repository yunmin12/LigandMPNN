import argparse
import os
import shutil

import numpy as np
import pandas as pd


def filter_and_select(
    df,
    top_n=50,
    ca_min=0.3,
    ca_max=2.0,
    ha_max=3.5,
    use_interface=True,
    w_int=0.1,
):
    base_cols = ["E_after", "CA_RMSD_to_input", "all_atom_RMSD_to_input"]
    df = df.dropna(subset=[c for c in base_cols if c in df.columns]).copy()

    # ---------- 1) Energy-based outlier filter (IQR) ----------
    q1 = df["E_after"].quantile(0.25)
    q3 = df["E_after"].quantile(0.75)
    iqr = q3 - q1
    upper_e = q3 + 1.5 * iqr
    mask_energy = df["E_after"] <= upper_e

    # ---------- 2) RMSD-based filter ----------
    mask_ca = (df["CA_RMSD_to_input"] >= ca_min) & (df["CA_RMSD_to_input"] <= ca_max)
    mask_ha = df["all_atom_RMSD_to_input"] <= ha_max

    mask = mask_energy & mask_ca & mask_ha

    # ---------- 3) Interface deltaG ----------
    mask_int = pd.Series(True, index=df.index)
    if use_interface and ("deltaG_interface" in df.columns):
        q1_int = df["deltaG_interface"].quantile(0.25)
        q3_int = df["deltaG_interface"].quantile(0.75)
        iqr_int = q3_int - q1_int
        upper_int = q3_int + 2.0 * iqr_int
        mask_int = df["deltaG_interface"] <= upper_int

    mask = mask & mask_int

    df_filt = df[mask].copy()

    # ---------- Condition Mitigation (if n < top k) ----------
    # 1. energy + interface (RMSD loosen)
    if len(df_filt) < top_n:
        mask_relaxed = mask_energy & mask_int
        df_filt = df[mask_relaxed].copy()

    # 2. keep only energy
    if len(df_filt) < top_n:
        df_filt = df[mask_energy].copy()

    # 3. whole
    if len(df_filt) < top_n:
        df_filt = df.copy()

    # ---------- 4) Ranking: E_after + interface penalty ----------
    if use_interface and ("deltaG_interface" in df_filt.columns):
        med_int = df_filt["deltaG_interface"].median()
        # penalty on interfaces worse than the median
        int_penalty = np.clip(df_filt["deltaG_interface"] - med_int, 0, None)
        df_filt["score_eff"] = df_filt["E_after"] + w_int * int_penalty
    else:
        df_filt["score_eff"] = df_filt["E_after"]

    df_filt = df_filt.sort_values("score_eff", ascending=True)

    df_top = df_filt.head(top_n).copy()
    return df_top


def main():
    parser = argparse.ArgumentParser(
        description="Merge & filter Rosetta FastRelax decoys from multiple seeds and select top N."
    )
    parser.add_argument(
        "--summary_csvs",
        type=str,
        nargs="+",
        required=True,
        help="List of CSV files (e.g., frx_summary_0.csv frx_summary_1.csv ...). Shell glob is allowed.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Path to save filtered summary CSV (merged, top N).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to copy selected output_pdb files.",
    )
    parser.add_argument("--top_n", type=int, default=50, help="Number of decoys to select.")
    parser.add_argument("--ca_min", type=float, default=0.3, help="Minimum CA RMSD to input.")
    parser.add_argument("--ca_max", type=float, default=2.0, help="Maximum CA RMSD to input.")
    parser.add_argument("--ha_max", type=float, default=3.5, help="Maximum all-atom RMSD to input.")
    parser.add_argument(
        "--no_interface",
        action="store_true",
        help="Do not use deltaG_interface in filtering and ranking.",
    )
    parser.add_argument(
        "--w_int",
        type=float,
        default=0.1,
        help="Weight for interface penalty in effective score (small value recommended).",
    )
    args = parser.parse_args()

    # ---------- Read and merge CSVs ----------
    dfs = []
    for path in args.summary_csvs:
        if not os.path.isfile(path):
            print(f"[WARN] summary_csv not found, skipping: {path}")
            continue
        df_i = pd.read_csv(path)
        df_i["source_csv"] = os.path.basename(path)
        dfs.append(df_i)

    if not dfs:
        raise RuntimeError("No valid summary_csvs were loaded.")

    df_all = pd.concat(dfs, ignore_index=True)

    if "E_before" in df_all.columns and "E_after" in df_all.columns:
        df_all["deltaE"] = df_all["E_after"] - df_all["E_before"]

    use_interface = not args.no_interface

    df_top = filter_and_select(
        df_all,
        top_n=args.top_n,
        ca_min=args.ca_min,
        ca_max=args.ca_max,
        ha_max=args.ha_max,
        use_interface=use_interface,
        w_int=args.w_int,
    )

    # ---------- Save the summary ----------
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_top.to_csv(args.out_csv, index=False)

    # ---------- Copy selected PDBs ----------
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        for _, row in df_top.iterrows():
            src = row.get("output_pdb", None)
            if not isinstance(src, str) or not os.path.isfile(src):
                print(f"[WARN] output_pdb not found, skipping: {src}")
                continue

            base = os.path.basename(src)
            dst = os.path.join(args.out_dir, base)

            if os.path.exists(dst):
                stem, ext = os.path.splitext(base)
                i = 1
                while True:
                    new_name = f"{stem}_dup{i}{ext}"
                    dst_candidate = os.path.join(args.out_dir, new_name)
                    if not os.path.exists(dst_candidate):
                        dst = dst_candidate
                        break
                    i += 1

            shutil.copy2(src, dst)

    print(f"[INFO] Selected {len(df_top)} decoys across {len(dfs)} CSVs.")
    print(f"[INFO] Filtered summary saved to: {args.out_csv}")
    if args.out_dir is not None:
        print(f"[INFO] PDBs copied to: {args.out_dir}")


if __name__ == "__main__":
    main()
