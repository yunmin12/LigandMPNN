import gemmi
import copy, glob
import argparse, os

BACKBONE = {"N", "CA", "C", "O"}
AA3 = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "MSE","SEC","PYL"
}
WATER = {"HOH","WAT"}
METALS = {"ZN","MG","MN","CA","NA","K","FE","CO","NI","CU","CD"}

def is_polymer_res(res):
    return res.name in AA3

def is_water_res(res):
    return res.name in WATER

def is_nonpoly_res(res):
    return (not is_polymer_res(res)) and (not is_water_res(res))

def upsert_atom(dst_res, src_atom):
    for a in dst_res:
        if a.name == src_atom.name:
            a.pos, a.occ, a.b_iso = src_atom.pos, src_atom.occ, src_atom.b_iso
            return
    dst_res.add_atom(copy.deepcopy(src_atom))

def stitch(full_path: str, crop_path: str, out_path: str):
    full = gemmi.read_structure(full_path)
    crop = gemmi.read_structure(crop_path)

    crop_idx = {}
    for ch in crop[0]:
        for r in ch:
            crop_idx[(ch.name, r.seqid.num, r.seqid.icode)] = r

    for ch in full[0]:
        for r in ch:
            key = (ch.name, r.seqid.num, r.seqid.icode)
            if key in crop_idx and is_polymer_res(r):
                sc_src = crop_idx[key]
                for a in sc_src:
                    if a.name not in BACKBONE:
                        upsert_atom(r, a)

    for ch in crop[0]:
        for r in ch:
            if is_nonpoly_res(r):
                dst_ch = full[0].find_chain(ch.name)
                if dst_ch is None:
                    dst_ch = full[0].add_chain(gemmi.Chain(ch.name))

                dst_r = None
                for rr in dst_ch:
                    if (rr.seqid.num, rr.seqid.icode) == (r.seqid.num, r.seqid.icode) and is_nonpoly_res(rr):
                        dst_r = rr
                        break

                if dst_r is None:
                    dst_ch.add_residue_copy(r)
                else:
                    for i in range(len(dst_r) - 1, -1, -1):
                        del dst_r[i]
                    dst_r.name = r.name
                    for a in r:
                        dst_r.add_atom(copy.deepcopy(a))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    full.write_minimal_pdb(out_path)

def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("--full_pdb", help="Full backbone pdb (RFDAA output).", type=str, required=True)
    arg.add_argument("--split_dir", help="Directory with splited multistate pdbs (PLACER output).", type=str, required=True)
    arg.add_argument("--out_dir", help="Directory to save the outputs.", type=str, required=True)
    args=arg.parse_args()

    full_pdb = args.full_pdb
    split_dir = args.split_dir
    out_dir = args.out_dir
    crops = sorted(glob.glob(os.path.join(split_dir, "*.pdb")))
    for crop in crops:
        base = os.path.splitext(os.path.basename(crop))[0]
        out_path = os.path.join(out_dir, f"recovered_{base}.pdb")
        stitch(full_pdb, crop, out_path)
    print(f"Recovered backbone for {len(crops)} structures... Saved at {out_dir}")

if __name__ == "__main__":
    main()
