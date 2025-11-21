import argparse
import copy
import json
import os
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path

import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_utils import (
    alphabet,
    featurize,
    parse_PDB,
    restype_int_to_str,
)


AA_LABELS = alphabet[:20]
MUTATION_PATTERN = re.compile(
    r"^(?:(?P<chain>[A-Za-z]+):)?(?P<orig>[A-Z])?(?P<resnum>\d+)(?P<mut>[A-Z])?$"
)

def load_score_directory(directory: Path):
    records = {}
    for path in directory.rglob("*.pt"):
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            continue
        key = path.stem
        records.setdefault(key, []).append(payload)
    return records


def average_all_entries(records):
    combined = []
    for entries in records.values():
        combined.extend(entries)
    if not combined:
        return None
    return average_entries(combined)


def to_list_from_mapping(mapping):
    if isinstance(mapping, dict):
        items = sorted(((int(k), v) for k, v in mapping.items()), key=lambda x: x[0])
        return [v for _, v in items]
    if isinstance(mapping, (list, tuple)):
        return list(mapping)
    raise ValueError("Unsupported residue_names structure")


def average_entries(entries):
    logits_stack = []
    for entry in entries:
        logits = torch.as_tensor(entry["logits"], dtype=torch.float32)
        if logits.dim() == 3:
            logits_stack.append(logits.mean(dim=0))
        else:
            raise ValueError("Logits tensor must have shape [B, L, 21]")
    logits_mean = torch.stack(logits_stack).mean(dim=0)
    log_probs = torch.as_tensor(entries[0]["log_probs"], dtype=torch.float32)
    if log_probs.dim() == 3:
        log_probs_mean = log_probs.mean(dim=0)
    else:
        log_probs_mean = log_probs
    native_sequence = entries[0].get("native_sequence")
    if native_sequence is None:
        native_sequence = entries[0].get("S")
    native_sequence = torch.as_tensor(native_sequence, dtype=torch.int64).tolist()
    residue_names = to_list_from_mapping(entries[0]["residue_names"])
    return {
        "logits": logits_mean,
        "log_probs": log_probs_mean,
        "native_sequence": native_sequence,
        "residue_names": residue_names,
    }


def parse_mutation_label(label: str, residue_names):
    match = MUTATION_PATTERN.match(label)
    if not match:
        raise ValueError(f"Unable to parse mutation label '{label}'")
    chain_hint = match.group("chain")
    resnum = match.group("resnum")
    candidates = []
    for idx, name in enumerate(residue_names):
        if name is None:
            continue
        chain = ""
        digits = ""
        icode = ""
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
    if len(candidates) == 0:
        raise ValueError(f"Residue {label} not found in residue list")
    raise ValueError(f"Residue {label} is ambiguous; specify chain explicitly")


def extract_mutation_target_aa(label: str):
    match = MUTATION_PATTERN.match(label)
    if not match:
        raise ValueError(f"Unable to parse mutation label '{label}'")
    mut = match.group("mut")
    if mut:
        return mut.upper()
    return None

def compute_pocket_indices(pdb_path: Path, cutoff: float):
    protein_dict, _, _, icodes, _ = parse_PDB(
        str(pdb_path),
        device="cpu",
        chains=[],
        parse_all_atoms=True,
        parse_atoms_with_zero_occupancy=False,
    )
    if "chain_mask" not in protein_dict:
        mask = protein_dict.get("mask")
        if mask is not None:
            protein_dict["chain_mask"] = mask.clone()
        else:
            length = len(protein_dict["R_idx"]) if "R_idx" in protein_dict else 0
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
        raise RuntimeError("Ligand context not found in PDB; cannot determine pocket residues")
    mask_xy = mask_xy[0].cpu().numpy().astype(bool)
    residue_names = []
    R_idx_list = list(protein_dict["R_idx"].cpu().numpy())
    chain_letters = list(protein_dict["chain_letters"])
    for idx, residue_idx in enumerate(R_idx_list):
        residue_names.append(f"{chain_letters[idx]}{residue_idx}{icodes[idx]}")
    return np.where(mask_xy)[0], residue_names


def ensure_output_dir(base: Path, sample: str):
    out_dir = base / sample
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_key_site_bars(sample, mutation, datasets, output_dir):
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    try:
        highlight_aa = extract_mutation_target_aa(mutation)
    except ValueError:
        highlight_aa = None
    for ax, (label, logits) in zip(axes, datasets.items()):
        logits_array = np.asarray(logits, dtype=np.float32)
        aa_labels = AA_LABELS[: len(logits_array)]
        color_list = ["skyblue"] * len(logits_array)
        highlight_idx = None
        if highlight_aa:
            try:
                highlight_idx = AA_LABELS.index(highlight_aa.upper())
            except ValueError:
                highlight_idx = None
        if highlight_idx is not None and highlight_idx < len(color_list):
            color_list[highlight_idx] = "blue"
        if len(logits_array) > 0:
            max_idx = int(np.argmax(logits_array))
            color_list[max_idx] = "red"
        x_pos = range(len(logits_array))
        ax.set_title(f"{label} - {mutation}")
        bars = ax.bar(x_pos, logits_array, color=color_list, edgecolor="black")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(aa_labels, rotation=90)
        ax.set_ylabel("Logit")
        for bar, value in zip(bars, logits_array):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(output_dir / f"{sample}_{mutation}_logits_bar.png", dpi=300)
    plt.close(fig)


def plot_heatmap(data, residues, title, output_path, cmap="coolwarm", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(max(6, len(residues) * 0.6), 6))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(AA_LABELS)))
    ax.set_xticklabels(AA_LABELS, rotation=90)
    ax.set_yticks(range(len(residues)))
    ax.set_yticklabels(residues)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def summarize_pocket(sample, residues, native_seq, target_logits, off_logits, output_dir):
    rows = []
    for name, idx in residues:
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


def compute_mutation_success(entries, site_idx, target_aa):
    aa_to_idx = {aa: i for i, aa in enumerate(AA_LABELS)}
    target_idx = aa_to_idx.get(target_aa.upper())
    if target_idx is None:
        raise ValueError(f"Unknown amino acid '{target_aa}'")

    top1_match = 0
    top3_match = 0
    total = 0
    top1_counts = Counter()

    for entry in entries:
        logits = torch.as_tensor(entry["logits"], dtype=torch.float32)
        if logits.dim() != 3:
            raise ValueError("Logits tensor must have shape [B, L, 21]")
        if site_idx >= logits.shape[1]:
            raise ValueError(f"Site index {site_idx} out of bounds for logits with length {logits.shape[1]}")
        slice_logits = logits[:, site_idx, : len(AA_LABELS)]
        for row in slice_logits:
            total += 1
            top1_idx = int(torch.argmax(row).item())
            top1_aa = AA_LABELS[top1_idx]
            top1_counts[top1_aa] += 1
            if top1_idx == target_idx:
                top1_match += 1
            topk = torch.topk(row, k=min(3, row.numel()), dim=-1).indices.tolist()
            if target_idx in topk:
                top3_match += 1

    top1_all_str = "; ".join([f"{aa}:{top1_counts.get(aa, 0)}" for aa in AA_LABELS])
    return {
        "top1_match": top1_match,
        "top3_match": top3_match,
        "total": total,
        "top1_all_str": top1_all_str,
        "top1_counter": top1_counts,
    }


def _pick_reference_entry(*entries):
    for entry in entries:
        if entry is None:
            continue
        residue_names = entry.get("residue_names")
        logits = entry.get("logits")
        if residue_names is None or logits is None:
            continue
        if len(residue_names) != logits.shape[0]:
            continue
        return entry
    return None


def plot_total_aggregate(label, entries, pocket_indices, key_mutations, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    mod_entry = entries.get("mod_target")
    orig_entry = entries.get("orig_target")
    off_entry = entries.get("orig_off")

    reference_entry = _pick_reference_entry(mod_entry, orig_entry, off_entry)
    if reference_entry is None:
        print("[Warning] Unable to determine residue names for aggregated plots; skipping total plots")
        return

    residue_names = reference_entry["residue_names"]
    pocket_pairs = [
        (residue_names[idx], idx)
        for idx in pocket_indices
        if idx < len(residue_names)
    ]
    if not pocket_pairs:
        print("[Warning] No pocket residues matched for aggregated plots; skipping total plots")
        return
    indices_order = [idx for _, idx in pocket_pairs]
    residues_order = [name for name, _ in pocket_pairs]

    for mutation in key_mutations:
        try:
            idx = parse_mutation_label(mutation, residue_names)
        except ValueError as exc:
            print(f"[Warning] {exc}; skipping mutation {mutation} for aggregated plot")
            continue
        datasets = {}
        if orig_entry is not None:
            datasets["Original target"] = orig_entry["logits"][idx, : len(AA_LABELS)].cpu().numpy()
        if mod_entry is not None:
            datasets["Modified target"] = mod_entry["logits"][idx, : len(AA_LABELS)].cpu().numpy()
        if off_entry is not None:
            datasets["Original off-target"] = off_entry["logits"][idx, : len(AA_LABELS)].cpu().numpy()
        if not datasets:
            continue
        plot_key_site_bars(label, mutation, datasets, output_dir)

    if mod_entry is not None and off_entry is not None:
        delta_matrix = (
            mod_entry["logits"][indices_order, : len(AA_LABELS)]
            - off_entry["logits"][indices_order, : len(AA_LABELS)]
        ).cpu().numpy()
        plot_heatmap(
            delta_matrix,
            residues_order,
            "Modified target - off-target logits (aggregated pocket)",
            output_dir / f"{label}_delta_heatmap.png",
        )

    if mod_entry is not None and orig_entry is not None:
        diff_mod_vs_orig = (
            mod_entry["logits"][indices_order, : len(AA_LABELS)]
            - orig_entry["logits"][indices_order, : len(AA_LABELS)]
        ).cpu().numpy()
        plot_heatmap(
            diff_mod_vs_orig,
            residues_order,
            "Modified target - original target logits (aggregated pocket)",
            output_dir / f"{label}_target_comparison_heatmap.png",
        )


def main():
    parser = argparse.ArgumentParser(description="Compare LigandMPNN original and modified results")
    parser.add_argument("--orig_tar_scores", type=Path, required=True, help="Path to original scores directory")
    parser.add_argument("--mod_tar_scores", type=Path, required=True, help="Path to modified target scores directory")
    parser.add_argument("--orig_off_scores", type=Path, required=True, help="Path to original off-target scores directory")
    parser.add_argument("--target_pdb", type=Path, required=True, help="Target PDB (with ligand) for pocket detection")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to store analysis outputs")
    parser.add_argument("--key_mutations", nargs="*", default=[], help="Key mutation labels (e.g., T315I, A:T315I)")
    parser.add_argument("--pocket_cutoff", type=float, default=5.0, help="Pocket cutoff distance in Angstroms")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    orig_tar_raw = load_score_directory(args.orig_tar_scores)
    mod_tar_row = load_score_directory(args.mod_tar_scores)
    orig_off_raw = load_score_directory(args.orig_off_scores)

    orig = {k: average_entries(v) for k, v in orig_tar_raw.items()}
    mod_target = {k: average_entries(v) for k, v in mod_tar_row.items()}
    orig_off = {k: average_entries(v) for k, v in orig_off_raw.items()}
    orig_total_entry = average_all_entries(orig_tar_raw)
    mod_total_entry = average_all_entries(mod_tar_row)
    orig_off_default = average_all_entries(orig_off_raw)

    common_samples = sorted(set(orig) & set(mod_target))
    if not common_samples:
        raise RuntimeError("No common samples found between original and modified target score directories")

    pocket_indices, pocket_names_from_pdb = compute_pocket_indices(args.target_pdb, args.pocket_cutoff)
    if any(entry is not None for entry in (orig_total_entry, mod_total_entry, orig_off_default)):
        aggregate_dir = ensure_output_dir(args.output_dir, "total")
        total_entries = {
            "orig_target": orig_total_entry,
            "mod_target": mod_total_entry,
            "orig_off": orig_off_default,
        }
        plot_total_aggregate("total", total_entries, pocket_indices, args.key_mutations, aggregate_dir)

    success_rows = []
    success_totals = defaultdict(
        lambda: {"top1": 0, "top3": 0, "total": 0, "counter": Counter(), "target": None}
    )

    for sample in common_samples:
        out_dir = ensure_output_dir(args.output_dir, sample)
        orig_entry = orig[sample]
        mod_tar_entry = mod_target[sample]
        orig_off_entry = orig_off.get(sample)
        if orig_off_entry is None:
            if orig_off_default is None:
                print(f"[Warning] No off-target scores available for sample {sample}; skipping")
                continue
            print(f"[Info] Using aggregated off-target logits for sample {sample}")
            orig_off_entry = orig_off_default

        residue_names = mod_tar_entry["residue_names"]
        if len(residue_names) != mod_tar_entry["logits"].shape[0]:
            raise RuntimeError("Residue name count mismatch with logits length")

        pocket_pairs = [(residue_names[idx], idx) for idx in pocket_indices]
        pocket_pairs = [pair for pair in pocket_pairs if pair[1] < len(residue_names)]
        if not pocket_pairs:
            print(f"[Warning] No pocket residues matched for sample {sample}; skipping pocket analysis")
            continue

        # Key mutation bar plots
        for mutation in args.key_mutations:
            try:
                idx = parse_mutation_label(mutation, residue_names)
            except ValueError as exc:
                print(f"[Warning] {exc}; skipping mutation {mutation} for sample {sample}")
                continue
            datasets = {
                "Original target": orig_entry["logits"][idx, : len(AA_LABELS)].cpu().numpy(),
                "Modified target": mod_tar_entry["logits"][idx, : len(AA_LABELS)].cpu().numpy(),
                "Original off-target": orig_off_entry["logits"][idx, : len(AA_LABELS)].cpu().numpy(),
            }
            plot_key_site_bars(sample, mutation, datasets, out_dir)

            try:
                target_aa = extract_mutation_target_aa(mutation)
                if target_aa is not None:
                    raw_entries = mod_tar_row.get(sample, [])
                    success = compute_mutation_success(raw_entries, idx, target_aa)
                    acc = success_totals[mutation]
                    acc["top1"] += success["top1_match"]
                    acc["top3"] += success["top3_match"]
                    acc["total"] += success["total"]
                    acc["counter"].update(success["top1_counter"])
                    if acc["target"] is None:
                        acc["target"] = target_aa
                    success_rows.append(
                        {
                            "sample": sample,
                            "mutation": mutation,
                            "target_aa": target_aa,
                            "top1_match": success["top1_match"],
                            "top3_match": success["top3_match"],
                            "total_ensembles": success["total"],
                            "top1_counts": success["top1_all_str"],
                            "row_type": "sample",
                        }
                    )
            except Exception as exc:
                print(f"[Warning] Failed to compute success for {sample} {mutation}: {exc}")

        # Pocket summary table
        summary_df = summarize_pocket(
            sample,
            pocket_pairs,
            mod_tar_entry["native_sequence"],
            mod_tar_entry["logits"],
            orig_off_entry["logits"],
            out_dir,
        )

        residues_order = [name for name, _ in pocket_pairs]
        indices_order = [idx for _, idx in pocket_pairs]

        # Heatmap: target vs off-target logits difference (native AA)
        delta_matrix = (
            mod_tar_entry["logits"][indices_order, : len(AA_LABELS)]
            - orig_off_entry["logits"][indices_order, : len(AA_LABELS)]
        ).cpu().numpy()
        plot_heatmap(
            delta_matrix,
            residues_order,
            "Modified target - off-target logits (pocket)",
            out_dir / f"{sample}_delta_heatmap.png",
        )

        # Heatmap: original vs modified target logits
        diff_mod_vs_orig = (
            mod_tar_entry["logits"][indices_order, : len(AA_LABELS)]
            - orig_entry["logits"][indices_order, : len(AA_LABELS)]
        ).cpu().numpy()
        plot_heatmap(
            diff_mod_vs_orig,
            residues_order,
            "Modified target - original target logits (pocket)",
            out_dir / f"{sample}_target_comparison_heatmap.png",
        )

        # Save summary as JSON for convenience
        summary_path = out_dir / f"{sample}_pocket_summary.json"
        summary_path.write_text(summary_df.to_json(orient="records", indent=2))

    if success_rows:
        summary_rows = []
        for mut, acc in sorted(success_totals.items()):
            top_counts_str = "; ".join([f"{aa}:{acc['counter'].get(aa, 0)}" for aa in AA_LABELS])
            summary_rows.append(
                {
                    "sample": "ALL",
                    "mutation": mut,
                    "target_aa": acc["target"] or "",
                    "top1_match": acc["top1"],
                    "top3_match": acc["top3"],
                    "total_ensembles": acc["total"],
                    "top1_counts": top_counts_str,
                    "row_type": "total",
                }
            )
        success_df = pd.DataFrame(summary_rows + success_rows)
        success_df.to_csv(args.output_dir / "mutation_success.csv", index=False)


if __name__ == "__main__":
    main()
