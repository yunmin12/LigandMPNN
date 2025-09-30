import os
import requests
import argparse
from typing import Iterable, Set, Optional
from Bio import PDB

POLYMER_HETS = {
    "MSE", "SEC", "PYL", "HYP", "HIC",
    "SEP", "TPO", "PTR",
    "CSO", "CME", "MLY", "M3L"
}


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
            # Remove water and solvents
            if resname in self.solvent_names:
                return False
            # Conserve polymer HETATM
            if resname in POLYMER_HETS:
                return True
            # Conserve HETATM if only designated as a target
            if self.targets and resname not in self.targets:
                return False
        return True


class SelectTargetChains(PDB.Select):
    def __init__(self, keep_chains: Set[str], targets: Set[str], solvent_names: Set[str]):
        super().__init__()
        self.keep_chains = keep_chains
        self.targets = targets
        self.solvent_names = solvent_names

    def accept_chain(self, chain):
        return chain.id in self.keep_chains

    def accept_residue(self, residue):
        hetflag = residue.id[0]
        resname = residue.get_resname().strip().upper()
        if hetflag.strip() != "":
            if resname in self.solvent_names:
                return False
            if resname in POLYMER_HETS:
                return True
            if self.targets and resname not in self.targets:
                return False
        return True


def clean_structure(pdb_path: str, save_path: str, target_ligands: list[str] = None):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("input", pdb_path)

    if target_ligands is None:
        targets = []
    else: 
        targets = _normalize_targets(target_ligands)
    solvent_names = {"HOH", "WAT", "H2O", "SOL"}

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(save_path, select=SelectChainsAndLigands(targets, solvent_names))


def extract_chain_with_ligand(pdb_path: str, save_path: str, target_ligands: list[str]):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("input", pdb_path)
    model = structure[0]

    targets = _normalize_targets(target_ligands)
    solvent_names = {"HOH", "WAT", "H2O", "SOL"}

    keep_chains = set()
    if targets:
        for chain in model:
            for residue in chain:
                hetflag = residue.id[0]
                if hetflag.strip() != "":
                    resname = residue.get_resname().strip().upper()
                    if resname in targets:
                        keep_chains.add(chain.id)
                        break

    if not keep_chains:
        keep_chains = {chain.id for chain in model}
    print(keep_chains, targets)
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(save_path, select=SelectTargetChains(keep_chains, targets, solvent_names))


def process_pdb(
    pdb_input: str,
    out_dir: str,
    target_ligands: list[str] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    if ".pdb" in pdb_input:
        pdb_path = pdb_input
    else: 
        pdb_path = fetch_pdb(pdb_input, out_dir)
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("input", pdb_path)
    chains = get_chain_ids(structure)

    out_path = os.path.join(out_dir, f"trimmed_{pdb_input.upper()}.pdb")
    if len(chains) == 1:
        clean_structure(pdb_path, out_path, target_ligands)
    else:
        extract_chain_with_ligand(pdb_path, out_path, target_ligands)
    print(f"Saved cleaned PDB: {out_path}")
    return out_path

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Trim PDB as a single chain for LigandMPNN.")
    p.add_argument("--pdb", dest="input_pdb", required=True, help="Input PDB to be trimmed.")
    p.add_argument("--out", dest="out_dir", required=True, help="Output PDB path.")
    p.add_argument("--target", dest="target_ligands", required=False, default=None, help="Designate a chain to keep by bound ligand.")
    args = p.parse_args()
    output = process_pdb(args.input_pdb, args.out_dir, args.target_ligands)
