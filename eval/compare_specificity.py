import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ensure the parent directory is in the path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_utils import alphabet, featurize, parse_PDB, restype_int_to_str

AA_LABELS = alphabet[:20]
MUTATION_PATTERN = re.compile(
    r"^(?:(?P<chain>[A-Za-z]+):)?(?P<orig>[A-Z])?(?P<resnum>\d+)(?P<mut>[A-Z])?$"
)


def load_score_directory(directory: Path):
    """
    Load all .pt files from a directory and group by stem.
    Returns a dict: stem -> list of loaded tensors/dicts.
    """
    records = {}
    for path in directory.rglob("*.pt"):
        try:
            payload = torch.load(path, map_location="cpu")
            records.setdefault(path.stem, []).append(payload)
        except Exception:
            continue
    return records


def to_list_from_mapping(mapping):
    """
    Convert a mapping (dict or list) to an ordered list.
    If dict, sort by integer keys.
    """
    if isinstance(mapping, dict):
        items = sorted(((int(k), v) for k, v in mapping.items()), key=lambda x: x[0])
        return [v for _, v in items]
    return list(mapping) if isinstance(mapping, (list, tuple)) else []


def average_score_entries(entries):
    """
    Average multiple score entries (from different batches/runs).
    Returns a single dict with averaged logits, native_sequence, and residue_names.
    """
    if not entries:
        return None

    # Average logits
    logits_stack = []
    for entry in entries:
        logits = torch.as_tensor(entry["logits"], dtype=torch.float32)
        if logits.dim() == 3:
            # Shape: [B, L, 21] → average over B
            logits_stack.append(logits.mean(dim=0))
        else:
            raise ValueError("Logits tensor must have shape [B, L, 21]")

    logits_mean = torch.stack(logits_stack).mean(dim=0)  # [L, 21]

    # Get other data from first entry
    first_entry = entries[0]
    native_sequence = first_entry.get("native_sequence") or first_entry.get("S")
    native_sequence = torch.as_tensor(native_sequence, dtype=torch.int64).tolist()

    return {
        "logits": logits_mean,
        "native_sequence": native_sequence,
        "residue_names": to_list_from_mapping(first_entry["residue_names"]),
    }


def average_all_entries(records):
    """
    Average all entries across all keys in records.
    """
    all_entries = []
    for entries in records.values():
        all_entries.extend(entries)
    return average_score_entries(all_entries) if all_entries else None


def parse_mutation_label(label: str, residue_names):
    """
    Parse mutation label (e.g., '123Y', 'A:123Y', 'A:123') to find residue index.
    """
    match = MUTATION_PATTERN.match(label)
    if not match:
        raise ValueError(f"Unable to parse mutation label '{label}'")

    chain_hint = match.group("chain")
    resnum = match.group("resnum")

    candidates = []
    for idx, name in enumerate(residue_names):
        if name is None:
            continue

        # Parse residue name (e.g., 'A123' or 'A123A')
        chain, digits, icode = "", "", ""
        for ch in name:
            if ch.isalpha() and not digits:
                chain += ch
            elif ch.isdigit():
                digits += ch
            else:
                icode += ch

        if digits == resnum and (chain_hint is None or chain_hint == chain):
            candidates.append(idx)

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) == 0:
        raise ValueError(f"Residue {label} not found")
    else:
        raise ValueError(f"Residue {label} is ambiguous; specify chain")


def extract_target_aa(label: str):
    """
    Extract target amino acid from mutation label.
    """
    match = MUTATION_PATTERN.match(label)
    if match and match.group("mut"):
        return match.group("mut").upper()
    return None


def compute_mutation_success(entries, site_idx: int, target_aa: str):
    """
    Compute mutation success rates.
    """
    aa_to_idx = {aa: i for i, aa in enumerate(AA_LABELS)}
    target_idx = aa_to_idx.get(target_aa.upper())
    if target_idx is None:
        raise ValueError(f"Unknown amino acid '{target_aa}'")

    top1_match = top3_match = total = 0
    top1_counts = Counter()

    for entry in entries:
        logits = torch.as_tensor(entry["logits"], dtype=torch.float32)
        if logits.dim() != 3 or site_idx >= logits.shape[1]:
            continue

        slice_logits = logits[:, site_idx, : len(AA_LABELS)]
        for row in slice_logits:
            total += 1
            top1_idx = int(torch.argmax(row).item())
            top1_counts[AA_LABELS[top1_idx]] += 1

            if top1_idx == target_idx:
                top1_match += 1

            topk = torch.topk(row, k=min(3, row.numel()), dim=-1).indices.tolist()
            if target_idx in topk:
                top3_match += 1

    return {
        "top1_match": top1_match,
        "top3_match": top3_match,
        "total": total,
        "top1_counter": top1_counts,
    }


def compute_pocket_indices(pdb_path: Path, cutoff: float):
    """
    Compute pocket residue indices from PDB.
    """
    protein_dict, _, _, icodes, _ = parse_PDB(
        str(pdb_path),
        device="cpu",
        chains=[],
        parse_all_atoms=True,
        parse_atoms_with_zero_occupancy=False,
    )

    # Set chain mask if missing
    if "chain_mask" not in protein_dict:
        length = len(protein_dict.get("R_idx", []))
        protein_dict["chain_mask"] = torch.ones(length, dtype=torch.float32)

    feature_dict = featurize(
        protein_dict,
        cutoff_for_score=cutoff,
        use_atom_context=1,
        number_of_ligand_atoms=16,
        model_type="ligand_mpnn",
    )

    mask_xy = feature_dict.get("mask_XY")
    if mask_xy is None:
        raise RuntimeError(
            "Ligand context not found; cannot determine pocket residues"
        )

    mask_xy = mask_xy[0].cpu().numpy().astype(bool)

    # Build residue names
    R_idx_list = list(protein_dict["R_idx"].cpu().numpy())
    chain_letters = list(protein_dict["chain_letters"])
    residue_names = [
        f"{chain_letters[i]}{R_idx_list[i]}{icodes[i]}"
        for i in range(len(R_idx_list))
    ]

    return np.where(mask_xy)[0], residue_names


def plot_mutation_bars(sample: str, mutation: str, datasets: dict, output_dir: Path):
    """
    Plot bar chart for mutation analysis.
    """
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True
    )
    if len(datasets) == 1:
        axes = [axes]

    highlight_aa = extract_target_aa(mutation)

    for ax, (label, logits) in zip(axes, datasets.items()):
        logits_array = np.asarray(logits, dtype=np.float32)
        aa_labels = AA_LABELS[: len(logits_array)]

        # Color scheme: blue for target, red for max, skyblue for others
        colors = ["skyblue"] * len(logits_array)

        if highlight_aa and highlight_aa in AA_LABELS:
            highlight_idx = AA_LABELS.index(highlight_aa)
            if highlight_idx < len(colors):
                colors[highlight_idx] = "blue"

        if len(logits_array) > 0:
            max_idx = int(np.argmax(logits_array))
            colors[max_idx] = "red"

        ax.bar(
            range(len(logits_array)),
            logits_array,
            color=colors,
            edgecolor="black",
        )
        ax.set_title(f"{label} - {mutation}")
        ax.set_xticks(range(len(logits_array)))
        ax.set_xticklabels(aa_labels, rotation=90)
        ax.set_ylabel("Logit")

        # Add value labels
        for i, value in enumerate(logits_array):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / f"{sample}_{mutation}_logits_bar.png", dpi=300)
    plt.close(fig)


def plot_heatmap(data, residues: list, title: str, output_path: Path):
    """
    Plot heatmap of logit differences.
    """
    fig, ax = plt.subplots(figsize=(max(6, len(residues) * 0.6), 6))

    im = ax.imshow(data, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(AA_LABELS)))
    ax.set_xticklabels(AA_LABELS, rotation=90)
    ax.set_yticks(range(len(residues)))
    ax.set_yticklabels(residues)
    ax.set_title(title)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def create_pocket_summary(
    sample: str,
    pocket_pairs: list,
    native_seq: list,
    target_logits,
    off_logits,
    output_dir: Path,
):
    """
    Create pocket analysis summary table.
    """
    rows = []
    for name, idx in pocket_pairs:
        native_idx = native_seq[idx]
        aa_native = restype_int_to_str.get(native_idx, "?")
        tgt = float(target_logits[idx, native_idx])
        off = float(off_logits[idx, native_idx])

        rows.append(
            {
                "residue": name,
                "native_aa": aa_native,
                "target_logit": tgt,
                "off_target_logit": off,
                "delta": tgt - off,
            }
        )

    df = pd.DataFrame(rows)
    df.sort_values(by="delta", ascending=False, inplace=True)
    df.to_csv(output_dir / f"{sample}_pocket_summary.csv", index=False)
    return df


def process_sample(
    sample: str,
    orig_entry: dict,
    mod_entry: dict,
    off_entry: dict,
    pocket_indices: np.ndarray,
    key_mutations: list,
    output_dir: Path,
    mod_raw_entries: list,
):
    """
    Process a single sample for analysis.
    """
    sample_dir = output_dir / sample
    sample_dir.mkdir(parents=True, exist_ok=True)

    residue_names = mod_entry["residue_names"]
    pocket_pairs = [
        (residue_names[idx], idx)
        for idx in pocket_indices
        if idx < len(residue_names)
    ]

    if not pocket_pairs:
        print(f"[Warning] No pocket residues for sample {sample}")
        return []

    success_rows = []

    # Process key mutations
    for mutation in key_mutations:
        try:
            idx = parse_mutation_label(mutation, residue_names)

            datasets = {
                "Original target": orig_entry["logits"][idx, : len(AA_LABELS)]
                .cpu()
                .numpy(),
                "Modified target": mod_entry["logits"][idx, : len(AA_LABELS)]
                .cpu()
                .numpy(),
                "Original off-target": off_entry["logits"][idx, : len(AA_LABELS)]
                .cpu()
                .numpy(),
            }
            plot_mutation_bars(sample, mutation, datasets, sample_dir)

            # Compute success rates
            target_aa = extract_target_aa(mutation)
            if target_aa and mod_raw_entries:
                success = compute_mutation_success(mod_raw_entries, idx, target_aa)
                success_rows.append(
                    {
                        "sample": sample,
                        "mutation": mutation,
                        "target_aa": target_aa,
                        "top1_match": success["top1_match"],
                        "top3_match": success["top3_match"],
                        "total_ensembles": success["total"],
                        "row_type": "sample",
                    }
                )

        except ValueError as e:
            print(f"[Warning] {e}; skipping {mutation} for {sample}")

    # Create summary and heatmaps
    create_pocket_summary(
        sample,
        pocket_pairs,
        mod_entry["native_sequence"],
        mod_entry["logits"],
        off_entry["logits"],
        sample_dir,
    )

    # Generate heatmaps
    indices_order = [idx for _, idx in pocket_pairs]
    residues_order = [name for name, _ in pocket_pairs]

    # Target vs off-target difference
    delta_matrix = (
        mod_entry["logits"][indices_order, : len(AA_LABELS)]
        - off_entry["logits"][indices_order, : len(AA_LABELS)]
    ).cpu().numpy()
    plot_heatmap(
        delta_matrix,
        residues_order,
        "Modified target - off-target logits (pocket)",
        sample_dir / f"{sample}_delta_heatmap.png",
    )

    # Modified vs original target
    diff_matrix = (
        mod_entry["logits"][indices_order, : len(AA_LABELS)]
        - orig_entry["logits"][indices_order, : len(AA_LABELS)]
    ).cpu().numpy()
    plot_heatmap(
        diff_matrix,
        residues_order,
        "Modified target - original target logits (pocket)",
        sample_dir / f"{sample}_target_comparison_heatmap.png",
    )

    return success_rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare LigandMPNN specificity results"
    )
    parser.add_argument("--orig_tar_scores", type=Path, required=True)
    parser.add_argument("--mod_tar_scores", type=Path, required=True)
    parser.add_argument("--orig_off_scores", type=Path, required=True)
    parser.add_argument("--target_pdb", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--key_mutations", nargs="*", default=[])
    parser.add_argument("--pocket_cutoff", type=float, default=5.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load and process all score data
    orig_tar_raw = load_score_directory(args.orig_tar_scores)
    mod_tar_raw = load_score_directory(args.mod_tar_scores)
    orig_off_raw = load_score_directory(args.orig_off_scores)

    # Average entries
    orig_averaged = {k: average_score_entries(v) for k, v in orig_tar_raw.items()}
    mod_averaged = {k: average_score_entries(v) for k, v in mod_tar_raw.items()}
    orig_off_averaged = {k: average_score_entries(v) for k, v in orig_off_raw.items()}

    # Get default off-target entry
    default_off_entry = average_all_entries(orig_off_raw)

    # Find common samples
    common_samples = sorted(set(orig_averaged) & set(mod_averaged))
    if not common_samples:
        raise RuntimeError("No common samples found")

    # Compute pocket residues
    pocket_indices, _ = compute_pocket_indices(args.target_pdb, args.pocket_cutoff)
    print(f"Found {len(pocket_indices)} pocket residues")

    # Process each sample
    all_success_rows = []
    success_totals = defaultdict(
        lambda: {"top1": 0, "top3": 0, "total": 0, "counter": Counter(), "target": None}
    )

    for sample in common_samples:
        off_entry = orig_off_averaged.get(sample, default_off_entry)
        if off_entry is None:
            print(f"[Warning] No off-target data for {sample}")
            continue

        success_rows = process_sample(
            sample,
            orig_averaged[sample],
            mod_averaged[sample],
            off_entry,
            pocket_indices,
            args.key_mutations,
            args.output_dir,
            mod_tar_raw.get(sample, []),
        )

        all_success_rows.extend(success_rows)

        # Accumulate totals
        for row in success_rows:
            mut = row["mutation"]
            acc = success_totals[mut]
            acc["top1"] += row["top1_match"]
            acc["top3"] += row["top3_match"]
            acc["total"] += row["total_ensembles"]
            if acc["target"] is None:
                acc["target"] = row["target_aa"]

    # Save success summary
    if all_success_rows:
        summary_rows = [
            {
                "sample": "ALL",
                "mutation": mut,
                "target_aa": acc["target"] or "",
                "top1_match": acc["top1"],
                "top3_match": acc["top3"],
                "total_ensembles": acc["total"],
                "row_type": "total",
            }
            for mut, acc in sorted(success_totals.items())
        ]

        success_df = pd.DataFrame(summary_rows + all_success_rows)
        success_df.to_csv(args.output_dir / "mutation_success.csv", index=False)

    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
