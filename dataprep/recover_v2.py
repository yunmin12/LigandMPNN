import argparse, os, glob, sys, copy
from typing import Tuple, Dict
from Bio.PDB import PDBParser, PDBIO, Select

def residue_key(res):
    het, resseq, icode = res.id
    chain_id = res.get_parent().id
    return (chain_id, het, int(resseq), (icode or "").strip())

def is_het(res):
    return res.id[0].strip() != ""

def build_index(structure):
    idx: Dict[Tuple[str,str,int,str], object] = {}
    model = next(iter(structure))
    for chain in model:
        for res in chain:
            idx[residue_key(res)] = res
    return idx

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
    p.add_argument("--target-chain", default="Z")  # ligand chain if you also want to treat HETATM
    args = p.parse_args()

    subdirs = sorted([d for d in glob.glob(os.path.join(args.base_dir, "*"))])
    for sub in subdirs:
        cleaned_dir = os.path.join(sub, f"cleaned_{args.mode}")
        split_dir   = os.path.join(sub, f"split_{args.mode}_large")
        recover_dir = os.path.join(sub, f"recover_{args.mode}_large")
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

            c_idx = build_index(cleaned)
            f_model = next(iter(frag))

            updated_res, removed_atoms, added_atoms = 0, 0, 0

            for f_chain in f_model:
                for f_res in f_chain:
                    key = residue_key(f_res)
                    c_res = c_idx.get(key)
                    if c_res is None:
                        continue
                    if not is_het(f_res):
                        added, removed = replace_residue_atoms(c_res, f_res)
                        added_atoms   += added
                        removed_atoms += removed
                        updated_res   += 1
                    else:
                        if f_chain.id != args.target_chain:
                            continue
                        added, removed = replace_residue_atoms(c_res, f_res)
                        added_atoms   += added
                        removed_atoms += removed
                        updated_res   += 1

            io.set_structure(cleaned)
            io.save(out_path, select=KeepAll())
            print(f"[recover] {frag_path} -> {out_path} | residues={updated_res}, atoms_added={added_atoms}, atoms_removed={removed_atoms}")

        print("[recover] done.")

if __name__ == "__main__":
    main()
