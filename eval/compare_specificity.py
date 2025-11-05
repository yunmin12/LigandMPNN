import argparse
import copy
import json
import os
import re
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
    pattern = re.compile(
        r"^(?:(?P<chain>[A-Za-z]+):)?(?P<orig>[A-Z])?(?P<resnum>\d+)(?P<mut>[A-Z])?$"
    )
    match = pattern.match(label)
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
    for ax, (label, logits) in zip(axes, datasets.items()):
        ax.set_title(f"{label} - {mutation}")
        bars = ax.bar(range(len(AA_LABELS)), logits, color="skyblue", edgecolor="black")
        ax.set_xticks(range(len(AA_LABELS)))
        ax.set_xticklabels(AA_LABELS, rotation=90)
        ax.set_ylabel("Logit")
        for bar, value in zip(bars, logits):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
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


def main():
    parser = argparse.ArgumentParser(description="Compare LigandMPNN original and modified results")
    parser.add_argument("--orig_scores", type=Path, required=True, help="Path to original scores directory")
    parser.add_argument("--mod_target_scores", type=Path, required=True, help="Path to modified target scores directory")
    parser.add_argument("--mod_off_scores", type=Path, required=True, help="Path to modified off-target scores directory")
    parser.add_argument("--target_pdb", type=Path, required=True, help="Target PDB (with ligand) for pocket detection")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to store analysis outputs")
    parser.add_argument("--key_mutations", nargs="*", default=[], help="Key mutation labels (e.g., T315I, A:T315I)")
    parser.add_argument("--pocket_cutoff", type=float, default=5.0, help="Pocket cutoff distance in Angstroms")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    orig_raw = load_score_directory(args.orig_scores)
    mod_target_raw = load_score_directory(args.mod_target_scores)
    mod_off_raw = load_score_directory(args.mod_off_scores)

    orig = {k: average_entries(v) for k, v in orig_raw.items()}
    mod_target = {k: average_entries(v) for k, v in mod_target_raw.items()}
    mod_off = {k: average_entries(v) for k, v in mod_off_raw.items()}
    mod_off_default = average_all_entries(mod_off_raw)

    common_samples = sorted(set(orig) & set(mod_target))
    if not common_samples:
        raise RuntimeError("No common samples found between original and modified target score directories")

    pocket_indices, pocket_names_from_pdb = compute_pocket_indices(args.target_pdb, args.pocket_cutoff)

    for sample in common_samples:
        out_dir = ensure_output_dir(args.output_dir, sample)
        orig_entry = orig[sample]
        mod_target_entry = mod_target[sample]
        mod_off_entry = mod_off.get(sample)
        if mod_off_entry is None:
            if mod_off_default is None:
                print(f"[Warning] No off-target scores available for sample {sample}; skipping")
                continue
            print(f"[Info] Using aggregated off-target logits for sample {sample}")
            mod_off_entry = mod_off_default

        residue_names = mod_target_entry["residue_names"]
        if len(residue_names) != mod_target_entry["logits"].shape[0]:
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
                "Modified target": mod_target_entry["logits"][idx, : len(AA_LABELS)].cpu().numpy(),
                "Modified off-target": mod_off_entry["logits"][idx, : len(AA_LABELS)].cpu().numpy(),
            }
            plot_key_site_bars(sample, mutation, datasets, out_dir)

        # Pocket summary table
        summary_df = summarize_pocket(
            sample,
            pocket_pairs,
            mod_target_entry["native_sequence"],
            mod_target_entry["logits"],
            mod_off_entry["logits"],
            out_dir,
        )

        residues_order = [name for name, _ in pocket_pairs]
        indices_order = [idx for _, idx in pocket_pairs]

        # Heatmap: target vs off-target logits difference (native AA)
        delta_matrix = (
            mod_target_entry["logits"][indices_order, : len(AA_LABELS)]
            - mod_off_entry["logits"][indices_order, : len(AA_LABELS)]
        ).cpu().numpy()
        plot_heatmap(
            delta_matrix,
            residues_order,
            "Modified target - off-target logits (pocket)",
            out_dir / f"{sample}_delta_heatmap.png",
        )

        # Heatmap: original vs modified target logits
        diff_mod_vs_orig = (
            mod_target_entry["logits"][indices_order, : len(AA_LABELS)]
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


if __name__ == "__main__":
    main()
