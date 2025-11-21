#!/usr/bin/env python3
import argparse, glob, os, time
from pathlib import Path

import pandas as pd
import pyrosetta
from pyrosetta import pose_from_file, create_score_function
from pyrosetta.rosetta.core.scoring import CA_rmsd, all_atom_rmsd, atom_pair_constraint
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import (
    InitializeFromCommandline,
    RestrictToRepacking,
)
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover, MinMover
from pyrosetta.rosetta.protocols.rigid import RigidBodyPerturbMover
from pyrosetta.rosetta.protocols.simple_moves import SmallMover, ShearMover
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.core.scoring.constraints import ConstraintSet, AtomPairConstraint
from pyrosetta.rosetta.core.scoring.func import HarmonicFunc


def find_ligand_residues(pose, lig_chain=None, lig_resname=None):
    """Find ligand residues by chain or resname."""
    pdb_info = pose.pdb_info()
    lig_idxs = []
    for i in range(1, pose.size() + 1):
        res = pose.residue(i)
        if not res.is_ligand():
            continue
        ch = pdb_info.chain(i)
        name3 = res.name3().strip()
        if lig_chain is not None and ch == lig_chain:
            lig_idxs.append(i)
        elif lig_resname is not None and name3 == lig_resname:
            lig_idxs.append(i)
    return lig_idxs


def find_shell_residues(pose, lig_idxs, cutoff=6.0):
    """Find protein residues within cutoff Å of any ligand residue."""
    shell = set()
    for lig_i in lig_idxs:
        lig_res = pose.residue(lig_i)
        for i in range(1, pose.size() + 1):
            res = pose.residue(i)
            if not res.is_protein():
                continue
            min_dist2 = 1e9
            for a in range(1, lig_res.natoms() + 1):
                if lig_res.atom_type(a).is_hydrogen():
                    continue
                xyz_l = lig_res.xyz(a)
                for b in range(1, res.natoms() + 1):
                    if res.atom_type(b).is_hydrogen():
                        continue
                    xyz_r = res.xyz(b)
                    d2 = xyz_l.distance_squared(xyz_r)
                    if d2 < min_dist2:
                        min_dist2 = d2
            if min_dist2 <= cutoff * cutoff:
                shell.add(i)
    return sorted(shell)


def find_ligand_jump(pose, lig_idxs):
    """
    Heuristic: find a jump that connects protein ↔ ligand.
    We look for any jump whose downstream or upstream residue is in lig_idxs.
    """
    ft = pose.fold_tree()
    lig_set = set(lig_idxs)
    for j in range(1, ft.num_jump() + 1):
        down = ft.downstream_jump_residue(j)
        up = ft.upstream_jump_residue(j)
        if down in lig_set or up in lig_set:
            return j
    raise RuntimeError("Could not find ligand jump – check fold tree / ligand chain.")


def add_ligand_rigid_constraints(pose, lig_idxs, stddev=0.02):
    """
    Add pairwise harmonic constraints between ligand heavy atoms to keep the ligand rigid.
    Translation/rotation is still allowed because only internal distances are penalized.
    """
    heavy_atoms = []
    for idx in lig_idxs:
        res = pose.residue(idx)
        for atom_idx in range(1, res.natoms() + 1):
            if res.atom_type(atom_idx).is_hydrogen():
                continue
            atom_id = AtomID(atom_idx, idx)
            heavy_atoms.append((atom_id, res.xyz(atom_idx)))
    n = len(heavy_atoms)
    if n < 2:
        return 0
    cset = pose.constraint_set().clone() if pose.constraint_set() else ConstraintSet()
    for i in range(n):
        atom_i, xyz_i = heavy_atoms[i]
        for j in range(i + 1, n):
            atom_j, xyz_j = heavy_atoms[j]
            dist = xyz_i.distance(xyz_j)
            func = HarmonicFunc(dist, stddev)
            cst = AtomPairConstraint(atom_i, atom_j, func)
            cset.add_constraint(cst)
    pose.constraint_set(cset)
    return n * (n - 1) // 2


def main():
    ap = argparse.ArgumentParser(
        description="Repack + minimize shell residues with rigid-body ligand motion."
    )
    ap.add_argument("--input_dir", required=True,
                    help="Directory with *_recovered_*.pdb")
    ap.add_argument("--pattern", default="*_recovered_*.pdb",
                    help="Glob pattern inside input_dir (default: *_recovered_*.pdb)")
    ap.add_argument("--params", nargs="+", required=True,
                    help="Ligand .params file(s)")
    ap.add_argument("--outdir", required=True,
                    help="Output directory for ensemble PDBs")
    # ligand identification
    ap.add_argument("--ligand_chain", default=None,
                    help="Ligand chain ID, e.g. B")
    ap.add_argument("--ligand_resname", default=None,
                    help="Ligand 3-letter name, e.g. LIG")
    ap.add_argument("--cutoff", type=float, default=6.0,
                    help="Distance cutoff for shell (Å)")
    ap.add_argument("--weights", default="ref2015",
                    help="ScoreFunction weights (default: ref2015)")
    ap.add_argument("--nstruct", type=int, default=1,
                    help="Number of ensemble members per PDB")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--ligand_constraint_sd", type=float, default=0.02,
                    help="Sigma (Å) for harmonic constraints enforcing ligand rigidity")
    ap.add_argument("--ligand_constraint_weight", type=float, default=10.0,
                    help="Score weight for ligand rigid constraints (atom_pair_constraint)")
    args = ap.parse_args()

    if args.ligand_chain is None and args.ligand_resname is None:
        raise SystemExit("Specify at least one of --ligand_chain or --ligand_resname")

    os.makedirs(args.outdir, exist_ok=True)

    pattern = os.path.join(args.input_dir, args.pattern)
    pdbs = sorted(glob.glob(pattern))
    if not pdbs:
        raise SystemExit(f"No PDB matched pattern: {pattern}")

    # ---- PyRosetta init (torsion-space / no cartesian) ----
    flags = [
        f"-score:weights {args.weights}",
        "-ex1 -ex2aro -use_input_sc",
        "-ignore_zero_occupancy false",
        "-detect_disulf false",
        "-mute all",
        f"-constant_seed -jran {args.seed}",
    ]
    flags += ["-in:file:extra_res_fa"] + args.params

    print("[INFO] PyRosetta init...")
    pyrosetta.init(" ".join(flags))
    sf = create_score_function(args.weights)
    sf.set_weight(atom_pair_constraint, args.ligand_constraint_weight)

    rows = []

    for pdb in pdbs:
        tag = Path(pdb).stem
        print(f"\n📁 [INFO] Processing {pdb} (tag={tag})")
        pose0 = pose_from_file(pdb)
        pdb_info = pose0.pdb_info()

        # protein residue count
        n_aa = sum(1 for i in range(1, pose0.size() + 1)
                if pose0.residue(i).is_protein())
        print(f"[INFO] #protein residues = {n_aa}")

        # ligand residues
        lig_idxs = find_ligand_residues(
            pose0,
            lig_chain=args.ligand_chain,
            lig_resname=args.ligand_resname,
        )
        if not lig_idxs:
            print("[WARN] No ligand residues found; skip this PDB")
            continue
        lig_pdb_tags = [(pdb_info.chain(i), pdb_info.number(i)) for i in lig_idxs]
        print(f"[INFO] ligand residues (chain, num) = {lig_pdb_tags}")
        n_lig_csts = add_ligand_rigid_constraints(pose0, lig_idxs,
                                                  stddev=args.ligand_constraint_sd)
        print(f"[INFO] added {n_lig_csts} ligand rigid-body constraints")

        # shell residues
        shell = find_shell_residues(pose0, lig_idxs, cutoff=args.cutoff)
        shell_size = len(shell)
        print(f"[INFO] shell residues (pose index) within {args.cutoff:.1f} Å: {shell}")
        print(f"[INFO] shell size = {shell_size}")
        if shell_size == 0:
            print("[WARN] Empty shell; skip this PDB")
            continue

        # ligand jump (rigid-body DOF)
        try:
            lig_jump = find_ligand_jump(pose0, lig_idxs)
        except RuntimeError as e:
            print(f"[WARN] {e}; skip this PDB")
            continue
        print(f"[INFO] ligand jump index = {lig_jump}")

        # TaskFactory for repacking
        tf = TaskFactory()
        tf.push_back(InitializeFromCommandline())
        tf.push_back(RestrictToRepacking())
        base_task = tf.create_task_and_apply_taskoperations(pose0)

        shell_set = set(shell)

        # base MoveMap: shell bb/chi + ligand jump only
        base_mm = MoveMap()
        base_mm.set_bb(False)
        base_mm.set_chi(False)
        base_mm.set_jump(False)
        # protein shell residues: bb, chi free
        for i in shell:
            base_mm.set_bb(i, True)
            base_mm.set_chi(i, True)
        # ligand internal DOFs remain False by default
        # rigid-body ligand/protein jump:
        base_mm.set_jump(lig_jump, True)

        # rigid-body perturb mover for randomness
        trans_mag = 0.2  # translation pertubation
        rot_mag = 2.0 # rotation perturbation
        rb_pert = RigidBodyPerturbMover(lig_jump, trans_mag, rot_mag)

        # local backbone randomization movers
        kT = 0.5
        n_moves = 5

        small_mover = SmallMover(base_mm, kT, n_moves)
        shear_mover = ShearMover(base_mm, kT, n_moves)

        small_mover.angle_max( 'H', 2.0 )   # helix
        small_mover.angle_max( 'E', 2.0 )   # sheet
        small_mover.angle_max( 'L', 5.0 )   # loop
        shear_mover.angle_max( 'H', 2.0 )
        shear_mover.angle_max( 'E', 2.0 )
        shear_mover.angle_max( 'L', 5.0 )

        # Pre-score original
        E0 = sf(pose0)

        for k in range(1, args.nstruct + 1):
            print(f"[DBG] {tag}: struct {k}/{args.nstruct}")

            pose = pose0.clone()

            rb_pert.apply(pose)
            n_cycles = 3
            for _ in range(n_cycles):
                small_mover.apply(pose)
                shear_mover.apply(pose)

            # repack + minimize
            # task & packer for this pose
            task = base_task.clone()
            for i in range(1, pose.size() + 1):
                res = pose.residue(i)
                if res.is_ligand():
                    task.nonconst_residue_task(i).prevent_repacking()
                elif res.is_protein() and i not in shell_set:
                    task.nonconst_residue_task(i).prevent_repacking()

            packer = PackRotamersMover(sf, task)
            packer.apply(pose)

            # minimization (torsion-space + rigid-body jump)
            mm = base_mm.clone()
            min_mover = MinMover()
            min_mover.score_function(sf)
            min_mover.movemap(mm)
            # torsion-space minimization
            min_mover.min_type("dfpmin_armijo_nonmonotone")
            min_mover.tolerance(0.001)

            t0 = time.time()
            min_mover.apply(pose)
            dt = time.time() - t0
            print(f"[INFO] {tag}: struct {k} minimization took {dt:.1f}s")

            if args.nstruct > 1:
                out_name = f"{tag}_frx_{args.seed}_{k:02d}.pdb"
            else:
                out_name = f"{tag}_frx_{args.seed}.pdb"
            out_path = os.path.join(args.outdir, out_name)
            pose.dump_pdb(out_path)
            E = sf(pose)
            ca = CA_rmsd(pose0, pose)
            ha = all_atom_rmsd(pose0, pose)

            rows.append({
                "input_pdb": pdb,
                "output_pdb": out_path,
                "tag": tag,
                "struct_idx": k,
                "seed": args.seed,
                "n_protein_res": n_aa,
                "shell_size": shell_size,
                "shell": shell, 
                "lig_jump": lig_jump,
                "E_before": float(E0),
                "E_after": float(E),
                "CA_RMSD_to_input": float(ca),
                "all_atom_RMSD_to_input": float(ha),
                "min_time_s": round(dt, 2),
            })
            print(f"✅ [DONE] {out_name}  E_after={E:.2f}  CA_RMSD={ca:.2f} Å")

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(args.outdir, f"frx_summary_{args.seed}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[WROTE] {csv_path}")
    else:
        print("\n[INFO] No structures were processed.")


if __name__ == "__main__":
    main()
