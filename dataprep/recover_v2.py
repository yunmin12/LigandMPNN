import argparse, os, glob, sys, copy
from typing import Tuple, Dict, List
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Residue import Residue

def residue_key(res):
    het, resseq, icode = res.id
    chain_id = res.get_parent().id
    return (chain_id, het, int(resseq), (icode or "").strip())

def is_het(res):
    return res.id[0].strip() != ""

def build_index(structure):
    idx: Dict[Tuple[str,str,int,str], Residue] = {}
    model = next(iter(structure))
    for chain in model:
        for res in chain:
            idx[residue_key(res)] = res
    return idx, model

def clone_residue(res):
    new_res = Residue(res.id, res.get_resname(), res.segid)
    for atom in res.get_atoms():
        new_res.add(copy.deepcopy(atom))
    return new_res

def residue_atom_count(res):
    return sum(1 for _ in res.get_atoms())

def select_ligand_targets(idx, chain_filter, resname_set):
    targets: List[Tuple[Tuple[str,str,int,str], Residue]] = []
    for key, res in idx.items():
        if not is_het(res):
            continue
        chain_id = key[0]
        resname = res.get_resname().strip().upper()
        if chain_filter != "*" and chain_id != chain_filter:
            continue
        if resname_set is not None and resname not in resname_set:
            continue
        targets.append((key, res))
    return targets

def replace_ligand_residue(c_idx, key, cleaned_res, frag_res):
    parent_chain = cleaned_res.get_parent()
    removed = residue_atom_count(cleaned_res)
    parent_chain.detach_child(cleaned_res.id)
    new_res = clone_residue(frag_res)
    parent_chain.add(new_res)
    c_idx[key] = new_res
    added = residue_atom_count(new_res)
    return added, removed

def replace_residue_atoms(dst_res, src_res):
    """
    Replace all atoms in dst_res with deep copies from src_res so coordinates,
    occupancies and B-factors mirror the PLACER fragment.
    """
    removed = 0
    for atom in list(dst_res.get_atoms()):
        dst_res.detach_child(atom.id)
        removed += 1

    added = 0
    for atom in src_res.get_atoms():
        dst_res.add(copy.deepcopy(atom))
        added += 1

    dst_res.resname = src_res.get_resname()
    return added, removed

class KeepAll(Select):
    def accept_atom(self, atom): return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)
    p.add_argument("--mode", choices=["tar","off"], required=True)
    p.add_argument("--ligand-chain", "--target-chain", dest="ligand_chain", default="*",
                   help="Chain ID for ligand residues to replace; '*' matches all chains")
    p.add_argument("--ligand-resname", default="*",
                   help="Comma-separated ligand residue names to replace (e.g., 'LIG,MOL'); '*' matches all HETATM residues")
    p.add_argument("--skip-ligand", action="store_true",
                   help="Disable ligand replacement if you want to keep the original cleaned ligand")
    args = p.parse_args()

    if args.ligand_resname == "*" or not args.ligand_resname:
        ligand_resnames = None
    else:
        ligand_resnames = {name.strip().upper() for name in args.ligand_resname.split(",") if name.strip()}

    candidate_cleaned = os.path.join(args.base_dir, f"cleaned_{args.mode}")
    if os.path.isdir(candidate_cleaned):
        subdirs = [args.base_dir]
    else:
        subdirs = sorted([d for d in glob.glob(os.path.join(args.base_dir, "*")) if os.path.isdir(d)])
    for sub in subdirs:
        cleaned_dir = os.path.join(sub, f"cleaned_{args.mode}")
        split_dir   = os.path.join(sub, f"split_{args.mode}")
        recover_dir = os.path.join(sub, f"recover_{args.mode}")
        if not (os.path.isdir(cleaned_dir) and os.path.isdir(split_dir)):
            continue
        os.makedirs(recover_dir, exist_ok=True)
        print(f"📁 Subdir: {os.path.basename(sub)}")

        parser = PDBParser(QUIET=True)
        io = PDBIO()

        files = sorted(glob.glob(os.path.join(split_dir, "*_model_*.pdb")))
        if not files:
            print(f"[recover] no files: {split_dir}/*_model_*.pdb", file=sys.stderr)
            sys.exit(1)

        for frag_path in files:
            stem = os.path.basename(frag_path)[:-4]
            if "_model_" not in stem: 
                print(f"[warn] skip (no _model_): {frag_path}"); 
                continue
            basename, idx = stem.split("_model_")
            cleaned_path = os.path.join(cleaned_dir, f"{basename}.pdb")
            out_path     = os.path.join(recover_dir, f"{basename}_recovered_{idx}.pdb")

            if not os.path.isfile(cleaned_path):
                print(f"[skip] cleaned not found: {cleaned_path}")
                continue

            cleaned = parser.get_structure("cleaned", cleaned_path)
            frag    = parser.get_structure("frag", frag_path)

            c_idx, c_model = build_index(cleaned)
            f_idx, f_model = build_index(frag)

            updated_res, removed_atoms, added_atoms = 0, 0, 0

            for f_chain in f_model:
                for f_res in f_chain:
                    if is_het(f_res):
                        continue
                    key = residue_key(f_res)
                    c_res = c_idx.get(key)
                    if c_res is None:
                        continue
                    added, removed = replace_residue_atoms(c_res, f_res)
                    added_atoms   += added
                    removed_atoms += removed
                    updated_res   += 1

            if not args.skip_ligand:
                targets = select_ligand_targets(c_idx, args.ligand_chain, ligand_resnames)
                if not targets:
                    print(f"[warn] no ligand residues matched the filter (chain={args.ligand_chain}, resname={args.ligand_resname}) in {cleaned_path}")
                for key, c_res in targets:
                    f_res = f_idx.get(key)
                    if f_res is None:
                        print(f"[warn] ligand {key} missing in fragment {frag_path}")
                        continue
                    added, removed = replace_ligand_residue(c_idx, key, c_res, f_res)
                    added_atoms   += added
                    removed_atoms += removed
                    updated_res   += 1

            io.set_structure(cleaned)
            io.save(out_path, select=KeepAll())
            print(f"[recover] {frag_path} -> {out_path} | residues={updated_res}, atoms_added={added_atoms}, atoms_removed={removed_atoms}")

        print("[recover] done.")

if __name__ == "__main__":
    main()
