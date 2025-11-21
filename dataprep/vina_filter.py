#!/usr/bin/env python3
import math
import argparse
from typing import List, Dict, Tuple, Optional

Atom = Dict[str, float]  # x, y, z, plus some metadata
Pose = Dict[str, object] # score, atoms, lines, index


def parse_pdb_like_atoms(lines: List[str]) -> List[Atom]:
    """
    Parse ATOM/HETATM lines from PDB/PDBQT.
    Returns list of atoms with x, y, z, and is_hydrogen.
    """
    atoms = []
    for line in lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue

        # Try to infer element from atom name (columns 12-16)
        name = line[12:16].strip()
        element = name[0]  # crude but usually okay

        is_h = element.upper() == "H"

        atoms.append({
            "x": x,
            "y": y,
            "z": z,
            "is_h": is_h,
        })
    return atoms


def load_reference_ligand(path: str) -> List[Atom]:
    with open(path) as f:
        lines = f.readlines()
    return parse_pdb_like_atoms(lines)


def load_protein_atoms(path: str) -> List[Atom]:
    with open(path) as f:
        lines = f.readlines()
    atoms = parse_pdb_like_atoms(lines)
    # keep only heavy atoms
    return [a for a in atoms if not a["is_h"]]


def load_vina_poses(pdbqt_path: str) -> List[Pose]:
    """
    Parse Vina PDBQT output with multiple MODELs.
    Extract Vina score from REMARK VINA RESULT line.
    """
    poses: List[Pose] = []
    with open(pdbqt_path) as f:
        lines = f.readlines()

    current_lines: List[str] = []
    current_score: Optional[float] = None
    model_index = 0
    in_model = False

    for line in lines:
        if line.startswith("MODEL"):
            # start new model
            in_model = True
            current_lines = [line]
            current_score = None
            continue

        if in_model:
            current_lines.append(line)

            if line.startswith("REMARK VINA RESULT"):
                # e.g. "REMARK VINA RESULT:   -8.7  0.0  0.0"
                parts = line.split()
                # score is typically the 4th token
                for p in parts:
                    try:
                        current_score = float(p)
                        break
                    except ValueError:
                        continue

            if line.startswith("ENDMDL"):
                # close model
                atoms = parse_pdb_like_atoms(current_lines)
                poses.append({
                    "index": model_index,
                    "score": current_score,
                    "atoms": atoms,
                    "lines": current_lines[:],
                })
                model_index += 1
                in_model = False
                current_lines = []

    # Some older vina outputs might not use MODEL/ENDMDL.
    # As a fallback, treat whole file as one pose.
    if not poses:
        atoms = parse_pdb_like_atoms(lines)
        poses.append({
            "index": 0,
            "score": None,
            "atoms": atoms,
            "lines": lines,
        })

    return poses


def centroid(atoms: List[Atom], include_h: bool = False) -> Tuple[float, float, float]:
    xs, ys, zs = [], [], []
    for a in atoms:
        if not include_h and a["is_h"]:
            continue
        xs.append(a["x"])
        ys.append(a["y"])
        zs.append(a["z"])
    if not xs:
        raise ValueError("No (heavy) atoms found to compute centroid")
    n = float(len(xs))
    return sum(xs)/n, sum(ys)/n, sum(zs)/n


def distance(a: Tuple[float, float, float],
             b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0])**2 +
                     (a[1] - b[1])**2 +
                     (a[2] - b[2])**2)


def min_protein_ligand_distance(protein: List[Atom],
                                ligand: List[Atom],
                                heavy_only: bool = True) -> float:
    min_d2 = float("inf")
    for la in ligand:
        if heavy_only and la["is_h"]:
            continue
        for pa in protein:
            # protein already heavy-only if we filtered earlier
            dx = la["x"] - pa["x"]
            dy = la["y"] - pa["y"]
            dz = la["z"] - pa["z"]
            d2 = dx*dx + dy*dy + dz*dz
            if d2 < min_d2:
                min_d2 = d2
    return math.sqrt(min_d2) if min_d2 < float("inf") else float("inf")


def main():
    parser = argparse.ArgumentParser(
        description="Filter Vina poses using COM distance, energy window, and steric clash."
    )
    parser.add_argument("--ref_ligand", required=True,
                        help="Reference ligand (target ligand) PDB/PDBQT path.")
    parser.add_argument("--protein", required=True,
                        help="Protein (receptor) PDB/PDBQT path.")
    parser.add_argument("--vina_out", required=True,
                        help="Docked ligand Vina PDBQT output path.")
    parser.add_argument("--out_prefix", default="filtered_pose",
                        help="Prefix for output PDBQT files.")
    parser.add_argument("--max_com_dist", type=float, default=4.0,
                        help="Max allowed COM distance (Å) between ref and docked pose.")
    parser.add_argument("--energy_window", type=float, default=3.0,
                        help="Energy window (kcal/mol) above best score.")
    parser.add_argument("--clash_cutoff", type=float, default=2.0,
                        help="Min protein-ligand heavy atom distance to consider a clash.")
    parser.add_argument("--max_poses", type=int, default=5,
                        help="Maximum number of poses to keep after filtering.")
    args = parser.parse_args()

    # 1) Load reference ligand & compute COM
    ref_atoms = load_reference_ligand(args.ref_ligand)
    ref_com = centroid(ref_atoms, include_h=False)

    # 2) Load protein heavy atoms
    protein_atoms = load_protein_atoms(args.protein)

    # 3) Load vina poses
    poses = load_vina_poses(args.vina_out)
    poses_with_scores = [p for p in poses if p["score"] is not None]
    if not poses_with_scores:
        print("Warning: No Vina scores found. Energy filtering will be skipped.")
        best_score = None
    else:
        best_score = min(p["score"] for p in poses_with_scores)

    print(f"Loaded {len(poses)} poses from Vina output.")
    if best_score is not None:
        print(f"Best score: {best_score:.3f} kcal/mol")

    # 4) Apply filters
    filtered: List[Tuple[Pose, float, float]] = []  # pose, com_dist, min_pl_dist

    for p in poses:
        atoms = p["atoms"]
        if not atoms:
            continue

        pose_com = centroid(atoms, include_h=False)
        com_dist = distance(ref_com, pose_com)

        if com_dist > args.max_com_dist:
            # different pocket → reject
            continue

        score = p["score"]
        if best_score is not None and score is not None:
            if score > best_score + args.energy_window:
                # too high energy
                continue

        # steric clash check
        min_pl_dist = min_protein_ligand_distance(protein_atoms, atoms, heavy_only=True)
        if min_pl_dist < args.clash_cutoff:
            # severe clash
            continue

        filtered.append((p, com_dist, min_pl_dist))

    if not filtered:
        print("No poses passed filtering criteria.")
        return

    # 5) Sort by score (ascending) primarily, then by COM distance
    def sort_key(item):
        pose, com_dist, min_pl = item
        s = pose["score"]
        if s is None:
            s = 999.0
        return (s, com_dist)

    filtered.sort(key=sort_key)

    # 6) Save top N poses
    kept = filtered[:args.max_poses]

    for rank, (pose, com_dist, min_pl_dist) in enumerate(kept, start=1):
        score = pose["score"]
        score_str = f"{score:.3f}" if score is not None else "None"
        out_path = f"{args.out_prefix}_rank{rank}.pdbqt"
        with open(out_path, "w") as f:
            f.writelines(pose["lines"])
        print(
            f"Saved {out_path}: "
            f"score={score_str if score is not None else None}, "
            f"COM_dist={com_dist:.2f} Å, "
            f"min_PL_dist={min_pl_dist:.2f} Å"
        )


if __name__ == "__main__":
    main()
