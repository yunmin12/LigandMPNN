import os
import sys
import numpy as np
import math
from collections import defaultdict

import pyrosetta

# example: 'AtomPair NE1 11 O 7 HARMONIC 3.0 0.5'
CST_STDERR = 0.5


def _pose_index_from_pdb_info(pdb_info, chain, resno):
    """Map a PDB chain/resno to the corresponding pose index."""
    for i in range(1, pdb_info.nres() + 1):
        if pdb_info.chain(i) == chain and pdb_info.number(i) == resno:
            return i
    raise ValueError(f"Could not find pose index for {chain}{resno}")


def extract_dist_cst_from_pdb_use_allatm(
    pdb_in, ptm_resname, bsite_res="", prot_chains=("A", "B"), lig_chain=None
):
    """
    Build AtomPair constraints using pose indices (not raw PDB numbering).

    - Uses all CA atoms from protein chains in `prot_chains` (skip ligand chain).
    - Identifies the first residue with name `ptm_resname` as the ligand
      (optionally restricted to `lig_chain`).
    - For each ligand atom, finds the closest protein CA and writes an AtomPair
      line with pose indices.
    """
    pose = pyrosetta.pose_from_pdb(pdb_in)
    pdb_info = pose.pdb_info()

    # Identify ligand residue (pose index) and collect its atom names/coords.
    lig_pose_idx = None
    lig_atom_names = []
    lig_atom_coords = {}
    for i in range(1, pose.size() + 1):
        if pose.residue(i).name3().strip() != ptm_resname:
            continue
        if lig_chain and pdb_info.chain(i) != lig_chain:
            continue
        lig_pose_idx = i
        res = pose.residue(i)
        for atm_i in range(1, res.natoms() + 1):
            atm_name = res.atom_name(atm_i).strip()
            lig_atom_names.append(atm_name)
            lig_atom_coords[atm_name] = res.xyz(atm_i)
        break
    if lig_pose_idx is None:
        raise ValueError(f"Ligand {ptm_resname} not found in {pdb_in}")

    lig_chain = pdb_info.chain(lig_pose_idx)

    # Collect protein CA coords (pose indices) for the requested chains.
    bsite_pose_indices = []
    if bsite_res:
        # If provided, interpret as comma-separated PDB residue numbers on prot_chains[0].
        resnos = [int(x) for x in bsite_res.split(",") if x]
        for r in resnos:
            try:
                idx = _pose_index_from_pdb_info(pdb_info, prot_chains[0], r)
                bsite_pose_indices.append(idx)
            except ValueError:
                print(f"[WARN] Skipping missing residue {prot_chains[0]}{r} in {pdb_in}")
    else:
        for i in range(1, pose.size() + 1):
            if i == lig_pose_idx:
                continue
            ch = pdb_info.chain(i)
            if ch == lig_chain:
                continue
            if ch not in prot_chains:
                continue
            res = pose.residue(i)
            if res.has("CA"):
                bsite_pose_indices.append(i)

    if not bsite_pose_indices:
        raise ValueError(f"No binding-site CA residues found in chains {prot_chains} for {pdb_in}")

    # Build constraints: for each ligand atom, find nearest protein CA.
    cst_s = []
    for lig_atm in lig_atom_names:
        lig_coord = lig_atom_coords[lig_atm]
        het_CA_d = {}
        for idx in bsite_pose_indices:
            res = pose.residue(idx)
            if not res.has("CA"):
                continue
            ca_coord = res.xyz("CA")
            d = (lig_coord - ca_coord).norm()
            het_CA_d[idx] = d
        if not het_CA_d:
            continue
        closest_idx, closest_d = sorted(het_CA_d.items(), key=lambda item: item[1])[0]
        cst_line = f"AtomPair {lig_atm} {lig_pose_idx} CA {closest_idx} HARMONIC {closest_d} {CST_STDERR}"
        cst_s.append(cst_line)

    return cst_s
    
