import os
import requests
import argparse
from typing import Iterable, Set, Optional, Tuple, List
from Bio import PDB
from Bio.PDB.NeighborSearch import NeighborSearch

POLYMER_HETS = {
    "MSE", "SEC", "PYL", "HYP", "HIC",
    "SEP", "TPO", "PTR",
    "CSO", "CME", "MLY", "M3L"
}
SOLVENTS = {"HOH", "WAT", "H2O", "SOL"}
CONTACT_CUTOFF = 5.0  # Å


def fetch_pdb(pdb_input: str, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    if os.path.isfile(pdb_input):
        return pdb_input
    else:
        pdb_id = pdb_input.lower()
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        out_path = os.path.join(save_dir, f"{pdb_id}.pdb")
        r = requests.get(url)
        if r.status_code != 200:
            raise ValueError(f"Cannot fetch {pdb_id} from RCSB")
        with open(out_path, "w") as f:
            f.write(r.text)
        return out_path


def get_chain_ids(structure) -> list[str]:
    return [chain.id for chain in structure[0]]


def _normalize_targets(target_ligands: Optional[Iterable[str]]) -> Set[str]:
    if not target_ligands:
        return set()
    if isinstance(target_ligands, str):
        target_ligands = [target_ligands]
    return {t.strip().upper() for t in target_ligands if t and str(t).strip()}


# --- contact helpers ---
def _is_protein_like(res) -> bool:
    het = (res.id[0] or "").strip()
    rn = res.get_resname().strip().upper()
    return het == "" or rn in POLYMER_HETS

def _collect_ligands(model, targets: Set[str]) -> List[Tuple[PDB.Residue.Residue, str]]:
    ligs = []
    for ch in model:
        for res in ch:
            het = (res.id[0] or "").strip()
            rn = res.get_resname().strip().upper()
            if het == "":
                continue
            if rn in SOLVENTS:
                continue
            if targets and rn not in targets:
                continue
            ligs.append((res, ch.id))
    # if no targets specified & none found, keep all non-solvent heteros
    if not targets and not ligs:
        for ch in model:
            for res in ch:
                het = (res.id[0] or "").strip()
                rn = res.get_resname().strip().upper()
                if het != "" and rn not in SOLVENTS:
                    ligs.append((res, ch.id))
    return ligs

def _chain_contact_score(chain: PDB.Chain.Chain, lig_atoms: List[PDB.Atom.Atom], cutoff: float) -> int:
    prot_atoms = [a for r in chain for a in r if _is_protein_like(r)]
    if not prot_atoms or not lig_atoms:
        return 0
    ns = NeighborSearch(prot_atoms)
    touched = set()
    for la in lig_atoms:
        for pa in ns.search(la.get_coord(), cutoff, "A"):
            touched.add(pa.get_serial_number())
    return len(touched)

def _pick_single_chain(model, ligs, cutoff: float) -> str:
    lig_atoms = []
    for res, _ in ligs:
        lig_atoms.extend(list(res.get_atoms()))
    scores = []
    for ch in model:
        if not any(_is_protein_like(r) for r in ch):
            continue
        score = _chain_contact_score(ch, lig_atoms, cutoff)
        scores.append((score, ch.id))
    if not scores:
        # fallback to first protein-like chain
        for ch in model:
            if any(_is_protein_like(r) for r in ch):
                return ch.id
        return next(iter(model)).id
    scores.sort(key=lambda x: (-x[0], str(x[1])))
    return scores[0][1]

def _ligands_contacting_chain(chain: PDB.Chain.Chain, ligs, cutoff: float) -> Set[Tuple[str, Tuple]]:
    # return set of (lig_chain_id, lig_residue.id) that contact the chain
    keep = set()
    ns = NeighborSearch([a for r in chain for a in r])
    for res, ch_id in ligs:
        contacted = False
        for a in res.get_atoms():
            if ns.search(a.get_coord(), cutoff, "A"):
                contacted = True
                break
        if contacted:
            keep.add((ch_id, res.id))
    return keep


class SelectChainsAndLigands(PDB.Select):
    def __init__(self, targets: Set[str], solvent_names: Set[str]):
        super().__init__()
        self.targets = targets
        self.solvent_names = solvent_names

    def accept_chain(self, chain):
        return True
    
    def accept_residue(self, residue):
        hetflag = residue.id[0].strip()
        resname = residue.get_resname().strip().upper()
        if hetflag != "":
            if resname in self.solvent_names:
                return False
            if resname in POLYMER_HETS:
                return True
            if self.targets and resname not in self.targets:
                return False
        return True


class SelectTargetChains(PDB.Select):
    def __init__(self,
                 keep_chains: Set[str],
                 targets: Set[str],
                 solvent_names: Set[str],
                 keep_ligand_keys: Optional[Set[Tuple[str, Tuple]]] = None):
        super().__init__()
        self.keep_chains = keep_chains
        self.targets = targets
        self.solvent_names = solvent_names
        self.keep_ligand_keys = keep_ligand_keys or set()

    def accept_chain(self, chain):
        # keep only the chosen protein chain and, if present, ligand chains that host kept ligands
        if chain.id in self.keep_chains:
            return True
        # if ligand residues to keep exist on this chain, keep this chain as a ligand carrier
        if any(k[0] == chain.id for k in self.keep_ligand_keys):
            return True
        return False

    def accept_residue(self, residue):
        hetflag = residue.id[0]
        resname = residue.get_resname().strip().upper()
        if hetflag.strip() != "":
            # ligand: keep only selected contacting ligands
            key = (residue.get_parent().id, residue.id)
            if key in self.keep_ligand_keys:
                return True
            # otherwise drop non-selected ligands (and solvents)
            return False
        # protein-like residue
        if resname in self.solvent_names:
            return False
        if resname in POLYMER_HETS:
            return True
        return True


def clean_structure(pdb_path: str, save_path: str, target_ligands: Optional[Iterable[str]] = None):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("input", pdb_path)

    targets = _normalize_targets(target_ligands)
    solvent_names = SOLVENTS

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(save_path, select=SelectChainsAndLigands(targets, solvent_names))


def extract_chain_with_ligand(pdb_path: str, save_path: str, target_ligands: Optional[Iterable[str]]):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("input", pdb_path)
    model = structure[0]

    targets = _normalize_targets(target_ligands)
    solvent_names = SOLVENTS

    # 1) find candidate ligands (by name filter if targets given; else all non-solvent HETs)
    ligs = _collect_ligands(model, targets)

    # 2) pick ONE protein chain with max contacts to ligands (distance-based)
    keep_chain_id = _pick_single_chain(model, ligs, CONTACT_CUTOFF)
    keep_chains = {keep_chain_id}

    # 3) among ligands, keep only those contacting the chosen chain
    kept_lig_keys = _ligands_contacting_chain(
        next(ch for ch in model if ch.id == keep_chain_id),
        ligs,
        CONTACT_CUTOFF
    )

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(save_path, select=SelectTargetChains(keep_chains, targets, solvent_names, kept_lig_keys))


def process_pdb(
    pdb_input: str,
    out_dir: str,
    target_ligands: Optional[Iterable[str]] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    pdb_path = pdb_input if pdb_input.endswith(".pdb") else fetch_pdb(pdb_input, out_dir)

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("input", pdb_path)
    chains = get_chain_ids(structure)

    # robust out name
    base = os.path.splitext(os.path.basename(pdb_path))[0].upper()
    out_path = os.path.join(out_dir, f"trimmed_{base}.pdb")

    if len(chains) == 1:
        clean_structure(pdb_path, out_path, target_ligands)
    else:
        extract_chain_with_ligand(pdb_path, out_path, target_ligands)
    print(f"Saved cleaned PDB: {out_path}")
    return out_path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Trim PDB as a single chain for LigandMPNN.")
    p.add_argument("--pdb", dest="input_pdb", required=True, help="Input PDB path or PDB ID.")
    p.add_argument("--out", dest="out_dir", required=True, help="Output directory.")
    p.add_argument("--target", dest="target_ligands", nargs="*", default=None,
                   help="Target ligand names. If omitted, all non-solvent HETs considered.")
    args = p.parse_args()
    output = process_pdb(args.input_pdb, args.out_dir, args.target_ligands)
