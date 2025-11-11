#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import os.path as osp
import numpy as np

def split_prefix_ligand(path_no_ext: str):
    if "_" not in path_no_ext:
        return path_no_ext, ""
    return path_no_ext.rsplit("_", 1)[0], path_no_ext.rsplit("_", 1)[1]

def box_plot_csv(csv_files, column, outpath):
    vals_list, labels, mins, maxs, meds = [], [], [], [], []

    bases = [osp.splitext(osp.basename(f))[0] for f in csv_files]
    pairs = [split_prefix_ligand(b) for b in bases]
    prefixes = [p for p, _ in pairs]
    ligands = [l for _, l in pairs]

    for f, ligand in zip(csv_files, ligands):
        df = pd.read_csv(f)
        if column not in df.columns:
            print(f"[WARN] '{column}' not found in {f}; skip.")
            continue

        vals = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(vals) == 0:
            print(f"[WARN] No valid data in '{column}' for {f}; skip.")
            continue

        vals_list.append(vals.values)
        labels.append(ligand if ligand != "" else osp.basename(f))
        mins.append(float(np.min(vals)))
        maxs.append(float(np.max(vals)))
        meds.append(float(np.median(vals)))

    if len(vals_list) == 0:
        print("No valid data found. Exit.")
        return

    # prepare DataFrame for seaborn
    merged = pd.DataFrame({
        "value": np.concatenate(vals_list),
        "ligand": np.repeat(labels, [len(v) for v in vals_list])
    })

    common_prefix = prefixes[0] if prefixes else ""

    plt.figure(figsize=(max(6, len(labels) * 1.2), 6))
    sns.boxplot(data=merged, x="ligand", y="value")

    # Manually overlay min/max points & median labels
    ax = plt.gca()

    for i, (mn, mx, md) in enumerate(zip(mins, maxs, meds)):
        xpos = i
        ax.plot(xpos, mn, marker='o', color='blue')
        ax.plot(xpos, mx, marker='o', color='red')

        # median text
        ax.text(
            xpos,
            md,
            f"{md:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black"
        )

    ax.set_ylabel(column)
    ax.set_title(f"Boxplot for {column}\n({common_prefix})")

    plt.xticks()
    plt.tight_layout()
    outname = os.path.join(outpath, f"boxplot_{common_prefix}_{column}.png")
    plt.savefig(outname, dpi=300)
    print(f"[saved] {outname}")


def scatter_rmsd_plddt(csv_files, outpath, x_col="rmsd", y_col="plddt"):
    """Plot per-CSV scatter plots with shared axes and unique colors."""
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette(None, n_colors=len(csv_files))

    plotted = False
    bases = [osp.splitext(osp.basename(f))[0] for f in csv_files]
    prefixes = [split_prefix_ligand(b)[0] for b in bases]
    pairs = [split_prefix_ligand(b) for b in bases]
    ligands = [l for _, l in pairs]

    for idx, (csv_file, ligand) in enumerate(zip(csv_files, ligands)):
        df = pd.read_csv(csv_file)
        if x_col not in df.columns or y_col not in df.columns:
            print(f"[WARN] Missing '{x_col}' or '{y_col}' in {csv_file}; skip.")
            continue

        x_vals = pd.to_numeric(df[x_col], errors="coerce")
        y_vals = pd.to_numeric(df[y_col], errors="coerce")
        mask = x_vals.notna() & y_vals.notna()
        if not mask.any():
            print(f"[WARN] No valid ({x_col}, {y_col}) pairs in {csv_file}; skip.")
            continue

        plt.scatter(
            x_vals[mask],
            y_vals[mask],
            color=palette[idx],
            alpha=0.6,
            label=ligand
        )
        plotted = True

    if not plotted:
        print("No data available for scatter plot; exiting.")
        return

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    common_prefix = prefixes[0] if prefixes else ""
    plt.title(f"{y_col} vs {x_col}\n({common_prefix})")
    plt.legend(title="Ligand", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    outname = os.path.join(outpath, f"scatter_{common_prefix}_{x_col}_vs_{y_col}.png")
    plt.savefig(outname, dpi=300)
    print(f"[saved] {outname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", help="CSV files")
    parser.add_argument("--column", default="rmsd",
                        help="Column to plot (default: rmsd)")
    parser.add_argument("--out_dir", default=".",
                        help="Output directory (default: current dir)")
    args = parser.parse_args()

    box_plot_csv(args.csv, args.column, args.out_dir)
    scatter_rmsd_plddt(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
