"""
Create backbone-only PDB files by removing sidechain atoms.
Use before off-target replacement (e.g. by AutoDock Vina, PLACER, FastRelax)
to reduce bias from sidechain conformations. 
Input:
    - input_pdb_path: path to input PDB file
    - output_pdb_path: path to save backbone-only PDB file
    - keep_cb: whether to keep CB atoms in addition to backbone (N, CA, C, O)
    - keep_hetero: whether to keep heteroatoms (ligands, ions, water) intact
Output:
    - backbone-only PDB file (saved at output_pdb_path)
Usage example:
    - protein backbone only, remove all heteroatoms
        make_backbone_only_pdb("input.pdb", "input_bb_only.pdb", keep_cb=False, keep_hetero=False)
    - protein backbone + ligands ✔︎
        make_backbone_only_pdb("input.pdb", "input_bb_keephet.pdb", keep_cb=False, keep_hetero=True)
    - protein backbone + CB + ligands
        make_backbone_only_pdb("input.pdb", "input_bb_cb.pdb", keep_cb=True, keep_hetero=False)
"""
import os, sys
import argparse
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa

BACKBONE_ATOMS = {"N", "CA", "C", "O"}
BACKBONE_PLUS_CB = {"N", "CA", "C", "O", "CB"}

class BackboneSelect(Select):
    def __init__(self, keep_cb: bool = False, keep_hetero: bool = True, unify_to_ala: bool = True):
        self.allowed = BACKBONE_PLUS_CB if keep_cb else BACKBONE_ATOMS
        self.keep_hetero = keep_hetero
        self.unify_to_ala = unify_to_ala

        self.n_protein_atoms_kept = 0
        self.n_protein_atoms_removed = 0
        self.n_hetero_atoms_kept = 0
        self.n_hetero_atoms_removed = 0

    def accept_atom(self, atom):
        # for protein residues, keep only backbone atoms
        parent_res = atom.get_parent()
        hetflag = parent_res.id[0]  # " " (protein), "W" (water), "H_xxx" (ligand/ion)
        
        # discard alternative locations except blank or 'A'
        if is_aa(parent_res, standard=True): # only when protein residue
            if atom.get_altloc() not in (" ", "A"):
                return 0
        
        # protein residues
        if hetflag == " ":
            if atom.get_name().strip() in self.allowed:
                self.n_protein_atoms_kept += 1
                return 1 
            else:
                self.n_protein_atoms_removed += 1
                return 0

        # hetero residues (ligands/ions/water)
        if self.keep_hetero:
            self.n_hetero_atoms_kept += 1
            return 1
        else:
            self.n_hetero_atoms_removed += 1
            return 0


def make_backbone_only_pdb(
    input_pdb_path: str,
    output_pdb_path: str,
    keep_cb: bool = False,
    keep_hetero: bool = True, 
    unify_to_ala: bool = True,
):
    """
    Create backbone-only version of a PDB:
      - Protein residues: keep N, CA, C, O (and optionally CB)
      - HETATM residues: keep all atoms if keep_hetero=True (ligand, ions, etc.)
    """
    print(f"[INFO] Backbone-only PDB generation started")
    print(f"[INFO] Input : {input_pdb_path}")
    print(f"[INFO] Output: {output_pdb_path}")
    print(f"[INFO] Options: keep_cb={keep_cb}, keep_hetero={keep_hetero}, unify_to_ala={unify_to_ala}")

    if not os.path.exists(input_pdb_path):
        print(f"[ERROR] Input PDB does not exist: {input_pdb_path}", file=sys.stderr)
        return False

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("structure", input_pdb_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse PDB: {e}", file=sys.stderr)
        return False
    
    selector = BackboneSelect(keep_cb=keep_cb, keep_hetero=keep_hetero, unify_to_ala=unify_to_ala)

    try:
        io = PDBIO()
        io.set_structure(structure)
        tmp_path = output_pdb_path + ".tmp"
        io.save(tmp_path, select=selector)

        # If unify_to_ala is True, replace all residue names with ALA
        with open(tmp_path) as fin, open(output_pdb_path, "w") as fout:
            for line in fin:
                if unify_to_ala and line.startswith("ATOM"):
                    line = line[:17] + "ALA" + line[20:]
                fout.write(line)
        os.remove(tmp_path)
    except Exception as e:
        print(f"[ERROR] Failed to write output PDB: {e}", file=sys.stderr)
        return False

    print(f"[SUCCESS] Backbone-only PDB generation completed successfully")
    print(
        f"[SUMMARY] protein atoms kept: {selector.n_protein_atoms_kept}, "
        f"removed: {selector.n_protein_atoms_removed}"
    )
    print(
        f"[SUMMARY] hetero atoms kept: {selector.n_hetero_atoms_kept}, "
        f"removed: {selector.n_hetero_atoms_removed}"
    )

    return True

def main():
    p = argparse.ArgumentParser(description="Create backbone-only PDB file by removing sidechain atoms.")
    p.add_argument("--input_pdb", type=str, required=True, help="Path to input PDB file.")
    p.add_argument("--out_dir", type=str, help="Directory to save backbone-only PDB file.")
    p.add_argument("--output_pdb", type=str, default=None, help="Path to save backbone-only PDB file.")
    # action="store_true" means default is False, set to True if flag is provided
    p.add_argument("--keep_cb", action="store_true", help="Whether to keep CB atoms.") # default: False
    p.add_argument("--keep_hetero", action="store_true", help="Whether to keep heteroatoms (ligands, ions, water).") # default: True
    p.add_argument("--unify_to_ala", action="store_true", help="Whether to unify all protein residues to ALA.") # default: True
    args = p.parse_args()

    input_basename = os.path.splitext(os.path.basename(args.input_pdb))[0]
    os.makedirs(args.out_dir, exist_ok=True)

    if args.output_pdb is None:
        if args.keep_cb==False and args.keep_hetero==False:
            output_pdb_path = os.path.join(args.out_dir, f"{input_basename}_bb_only.pdb")
            if args.unify_to_ala:
                output_pdb_path = os.path.join(args.out_dir, f"{input_basename}_bb_only_ala.pdb")
        elif args.keep_cb==False and args.keep_hetero==True:
            output_pdb_path = os.path.join(args.out_dir, f"{input_basename}_bb_keephet.pdb")
            if args.unify_to_ala:
                output_pdb_path = os.path.join(args.out_dir, f"{input_basename}_bb_keephet_ala.pdb")
        elif args.keep_cb==True and args.keep_hetero==False:
            output_pdb_path = os.path.join(args.out_dir, f"{input_basename}_bb_cb.pdb")
            if args.unify_to_ala:
                output_pdb_path = os.path.join(args.out_dir, f"{input_basename}_bb_cb_ala.pdb")
    else:
        output_pdb_path = args.output_pdb
    
    success = make_backbone_only_pdb(
        args.input_pdb, 
        output_pdb_path, 
        keep_cb=args.keep_cb, 
        keep_hetero=args.keep_hetero, 
        unify_to_ala=args.unify_to_ala
        )
    
    if not success:
        print("[ERROR] Backbone-only PDB generation failed.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()