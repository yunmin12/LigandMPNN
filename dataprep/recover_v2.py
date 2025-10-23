import argparse, os, glob, sys
from typing import Tuple, Dict
from Bio.PDB import PDBParser, PDBIO, Select

BACKBONE = {"N","CA","C","O","OXT"}

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

def copy_xyz(dst_atom, src_atom):
    dst_atom.set_coord(src_atom.get_coord())
    dst_atom.set_bfactor(src_atom.get_bfactor())
    dst_atom.set_occupancy(src_atom.get_occupancy())

def prune_to_backbone_and_update(dst_res, src_res):
    # 1) drop all sidechain atoms (keep only backbone set + CB if present in src)
    keep = set(BACKBONE)
    src_names = {a.get_name().strip() for a in src_res.get_atoms()}
    if "CB" in src_names:
        keep.add("CB")
    # collect atoms to delete
    to_del = []
    for a in dst_res.get_atoms():
        nm = a.get_name().strip()
        if nm not in keep:
            to_del.append(a)
    for a in to_del:
        parent = a.get_parent()
        parent.detach_child(a.id)

    # 2) update kept atoms from src when available; if kept atom not in src, leave as-is
    src_map = {a.get_name().strip(): a for a in src_res.get_atoms()}
    updated = 0
    for a in dst_res.get_atoms():
        nm = a.get_name().strip()
        if nm in src_map:
            copy_xyz(a, src_map[nm]); updated += 1
    return updated, len(to_del)

class KeepAll(Select):
    def accept_atom(self, atom): return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)
    p.add_argument("--mode", choices=["tar","off"], required=True)
    p.add_argument("--target-chain", default="Z")  # ligand chain if you also want to treat HETATM
    p.add_argument("--update-ligand", type=int, default=0)  # 1 to also update ligand atoms (chain Z)
    args = p.parse_args()

    subdirs = sorted([d for d in glob.glob(os.path.join(args.base_dir, "*"))])
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

            c_idx = build_index(cleaned)
            f_model = next(iter(frag))

            updated_res, deleted_side, updated_atoms = 0, 0, 0

            for f_chain in f_model:
                for f_res in f_chain:
                    key = residue_key(f_res)
                    c_res = c_idx.get(key)
                    if c_res is None:
                        continue
                    if not is_het(f_res):
                        ua, ds = prune_to_backbone_and_update(c_res, f_res)
                        updated_atoms += ua
                        deleted_side  += ds
                        updated_res   += 1
                    else:
                        if not args.update_ligand:
                            continue
                        if f_chain.id != args.target_chain:
                            continue
                        # For ligand: replace coords for common atom names only, do not change residue name
                        src_map = {a.get_name().strip(): a for a in f_res.get_atoms()}
                        for a in c_res.get_atoms():
                            nm = a.get_name().strip()
                            if nm in src_map:
                                copy_xyz(a, src_map[nm]); updated_atoms += 1
                        updated_res += 1

            io.set_structure(cleaned)
            io.save(out_path, select=KeepAll())
            print(f"[recover] {frag_path} -> {out_path} | residues={updated_res}, atoms_updated={updated_atoms}, sidechain_deleted={deleted_side}")

        print("[recover] done.")

if __name__ == "__main__":
    main()
