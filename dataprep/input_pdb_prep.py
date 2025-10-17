import os, re, sys, traceback, requests
import argparse
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdFMCS, rdchem


# ---------- utils ----------
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

def mkdir(p): os.makedirs(p, exist_ok=True)

def download_pdb(pdb_id: str, out_path: str, timeout=30) -> bool:
    url = RCSB_PDB_URL.format(pdb_id=pdb_id.upper())
    r = requests.get(url, timeout=timeout)
    if r.status_code == 200 and ("ATOM" in r.text or "HETATM" in r.text):
        with open(out_path, "w") as f: f.write(r.text)
        return True
    return False

def parse_resname_map(raw: str) -> Dict[str, str]:
    m = {}
    if not isinstance(raw, str) or not raw.strip():
        return m
    s = raw.strip()
    s = s.strip().strip("'").strip('"')
    s = s.replace("：", ":").replace("；", ";").replace("，", ",")
    for tok in re.split(r"[;,\s]+", s):
        if not tok or ":" not in tok:
            continue
        pid, rn = tok.split(":", 1)
        pid = pid.strip().upper()
        rn  = rn.strip().upper()[:3]
        if pid and rn:
            m[pid] = rn
    return m


def smiles_to_3d_pdb_block(smiles: str, resname: str="UNK") -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3(); params.randomSeed = 0xF00D; params.useSmallRingTorsions = True
    ok = AllChem.EmbedMolecule(mol, params)
    if ok != 0:
        confs = AllChem.EmbedMultipleConfs(mol, numConfs=20, params=params)
        if not confs: return None
        mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
        if mp is None: return None
        best_e, best_id = 1e30, confs[0]
        for cid in confs:
            try:
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
                ff.Minimize()
                e = ff.CalcEnergy()
                if e < best_e: best_e, best_id = e, cid
            except Exception: pass
        for cid in list(range(mol.GetNumConformers())):
            if cid != best_id: mol.RemoveConformer(cid)
    else:
        mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
        if mp is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp); ff.Minimize()

    for a in mol.GetAtoms():
        info = rdchem.AtomPDBResidueInfo()
        info.SetResidueName((resname or "UNK")[:3]); info.SetResidueNumber(1)
        info.SetChainId('Z'); info.SetIsHeteroAtom(True)
        a.SetMonomerInfo(info)
    return Chem.MolToPDBBlock(mol)

def read_text(p): 
    with open(p, "r") as f: return f.read()

def write_text(p, s): 
    with open(p, "w") as f: f.write(s)

def remove_resname_from_pdb(pdb_text: str, resname: str) -> str:
    out = []
    for line in pdb_text.splitlines():
        if line.startswith("HETATM") and line[17:20].strip().upper() == (resname or "").upper():
            continue
        if line.startswith("CONECT"):  # drop all
            continue
        out.append(line)
    return "\n".join(out) + "\n"

def combine_protein_and_ligand(protein_pdb_text: str, ligand_pdb_block: str, drop_resname: Optional[str]=None) -> str:
    prot = remove_resname_from_pdb(protein_pdb_text, drop_resname) if drop_resname else protein_pdb_text
    prot_lines = [l for l in prot.splitlines() if not l.startswith("END")]
    lig_lines  = [l for l in ligand_pdb_block.splitlines() if l.startswith(("HETATM","CONECT"))]
    return "\n".join(prot_lines + lig_lines + ["END"]) + "\n"

def pdb_block_to_mol(pdb_block: str) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=True, removeHs=False)
        if mol is None: mol = Chem.MolFromPDBBlock(pdb_block, sanitize=True, removeHs=True)
        return mol
    except Exception:
        return None

def extract_ligand_block(pdb_text: str, resname: str) -> Optional[str]:
    lines = []
    for line in pdb_text.splitlines():
        if line.startswith("HETATM") and line[17:20].strip().upper() == resname.upper():
            lines.append(line)
    if not lines: return None
    return "\n".join(lines) + "\n"

def align_to_pose(mov: Chem.Mol, ref: Chem.Mol) -> Chem.Mol:
    res = rdFMCS.FindMCS([ref, mov], timeout=10, completeRingsOnly=True, ringMatchesRingOnly=True, matchChiralTag=False)
    if res and res.smartsString:
        core = Chem.MolFromSmarts(res.smartsString)
        ref_match = ref.GetSubstructMatch(core)
        mov_match = mov.GetSubstructMatch(core)
        if ref_match and mov_match and len(ref_match)==len(mov_match):
            amap = list(zip(mov_match, ref_match))
            rdMolAlign.AlignMol(mov, ref, atomMap=amap)
            return mov
    # fallback: global
    rdMolAlign.AlignMol(mov, ref)
    return mov

def set_resname_all_atoms(m: Chem.Mol, resname="UNK", chain="Z", resnum=1):
    for a in m.GetAtoms():
        info = rdchem.AtomPDBResidueInfo()
        info.SetResidueName((resname or "UNK")[:3])
        info.SetResidueNumber(resnum)
        info.SetChainId(chain)
        info.SetIsHeteroAtom(True)
        a.SetMonomerInfo(info)

# ---------- core ----------
def process_group(gdf: pd.DataFrame, protein_key: str, out_dir: str, summary_rows: List[dict]):
    safe_key = re.sub(r"[^\w\-\.]", "_", str(protein_key))
    pdir = os.path.join(out_dir, safe_key)
    mkdir(pdir)

    # filter rows with non-empty complex_pdb_id
    rows = []
    for _, r in gdf.iterrows():
        cids = str(r.get("complex_pdb_id","") or "").strip()
        if not cids or cids.lower() in ["nan","none",""]:
            # pass row entirely
            continue
        rows.append(r)
    if not rows: return

    # download PDBs and collect maps
    target_pdb_ids: Set[str] = set()
    off_pdb_ids: Set[str]    = set()
    group_resname_map: Dict[str,str] = {}  # PDBID -> resname (first wins)

    # merged resname map over rows
    for r in rows:
        m = parse_resname_map(str(r.get("pdb_ligand_id","") or ""))
        for k,v in m.items():
            if k not in group_resname_map: group_resname_map[k]=v

    for r in rows:
        ttype = str(r.get("target_type","") or "").strip().lower()
        cids = [p.strip().upper() for p in re.split(r"[,\s]+", str(r["complex_pdb_id"]).strip()) if p.strip()]
        for pid in cids:
            path = os.path.join(pdir, f"{pid}.pdb")
            ok = download_pdb(pid, path)
            if ok:
                if ttype == "off-target": off_pdb_ids.add(pid)
                else: target_pdb_ids.add(pid)

    # build ligand for each row
    row_lig_blocks: Dict[int, str] = {}
    row_lig_resname: Dict[int, str] = {}
    for idx, r in enumerate(rows):
        smiles = str(r.get("ligand_smiles","") or "").strip()
        if not smiles: continue
        cids = [p.strip().upper() for p in re.split(r"[,\s]+", str(r["complex_pdb_id"]).strip()) if p.strip()]
        row_map = parse_resname_map(str(r.get("pdb_ligand_id","") or ""))
        resname = None
        for cid in cids:
            if cid in row_map:
                resname = row_map[cid]
                break
        if resname is None and len(set(group_resname_map.values())) == 1 and len(group_resname_map) > 0:
            resname = list(group_resname_map.values())[0]
        if resname is None:
            resname = "UNK"
        block = smiles_to_3d_pdb_block(smiles, resname=resname)
        if block:
            safe_ttype = re.sub(r"[^\w\-.]", "_", str(r.get("target_type","") or "").strip().lower())
            lig_path = os.path.join(pdir, f"{protein_key}_{safe_ttype}_{idx}_{resname}.pdb")
            write_text(lig_path, block)
            row_lig_blocks[idx]  = block
            row_lig_resname[idx] = resname

    # summary helper
    def emit_summary(saved_pid: str, ttype: str, extra: dict, created: bool, basis: str, saved_t_ids: List[str], saved_o_ids: List[str], saved_t_lids: List[str], saved_o_lids: List[str]):
        row = {
            "protein_key": protein_key,
            "target_type": ttype,
            "protein": extra.get("protein",""),
            "ligand":  extra.get("ligand",""),
            "sequence": extra.get("sequence",""),
            "key_mutation": extra.get("key_mutation",""),
            "saved_target_pdb_id": ";".join(sorted(set(saved_t_ids))) if saved_t_ids else "",
            "saved_off_target_pdb_id": ";".join(sorted(set(saved_o_ids))) if saved_o_ids else "",
            "saved_target_ligand_id": ";".join(sorted(set(saved_t_lids))) if saved_t_lids else "",
            "saved_off_target_ligand_id": ";".join(sorted(set(saved_o_lids))) if saved_o_lids else "",
            "off_target_pdb_created": "yes" if created else "no",
            "placement_basis": basis,
            "saved_pdb_id": saved_pid or "",
        }
        summary_rows.append(row)

    # collect ligand-id summaries
    all_target_rows = [r for r in rows if str(r.get("target_type","")).strip().lower()=="target"]
    all_off_rows    = [r for r in rows if str(r.get("target_type","")).strip().lower()=="off-target"]
    saved_t_lids = []
    saved_o_lids = []
    for r in all_target_rows:
        m = parse_resname_map(str(r.get("pdb_ligand_id","") or "")); saved_t_lids += list(m.values())
    for r in all_off_rows:
        m = parse_resname_map(str(r.get("pdb_ligand_id","") or "")); saved_o_lids += list(m.values())

    # emit download summaries per saved PDB
    for pid in sorted(target_pdb_ids):
        extra = {}
        if "protein" in gdf.columns: extra["protein"]=rows[0].get("protein","")
        if "ligand"  in gdf.columns: extra["ligand"]=rows[0].get("ligand","")
        if "sequence" in gdf.columns: extra["sequence"]=rows[0].get("sequence","")
        if "key_mutation" in gdf.columns: extra["key_mutation"]=rows[0].get("key_mutation","")
        emit_summary(pid, "target", extra, created=False, basis="none",
                     saved_t_ids=list(target_pdb_ids), saved_o_ids=list(off_pdb_ids),
                     saved_t_lids=saved_t_lids, saved_o_lids=saved_o_lids)
    for pid in sorted(off_pdb_ids):
        extra = {}
        if "protein" in gdf.columns: extra["protein"]=rows[0].get("protein","")
        if "ligand"  in gdf.columns: extra["ligand"]=rows[0].get("ligand","")
        if "sequence" in gdf.columns: extra["sequence"]=rows[0].get("sequence","")
        if "key_mutation" in gdf.columns: extra["key_mutation"]=rows[0].get("key_mutation","")
        emit_summary(pid, "off-target", extra, created=False, basis="none",
                     saved_t_ids=list(target_pdb_ids), saved_o_ids=list(off_pdb_ids),
                     saved_t_lids=saved_t_lids, saved_o_lids=saved_o_lids)

    # ---------- placement ----------
    created_any = False

    # Case A: target PDB exists → place each off-target ligand into each target basis
    if target_pdb_ids:
        for o_idx, o_row in enumerate(all_off_rows):
            if o_idx not in row_lig_blocks: continue
            off_block = row_lig_blocks[o_idx]
            off_mol = pdb_block_to_mol(off_block)
            if off_mol is None: 
                raise RuntimeError(f"[{protein_key}] Off-target ligand parse failed (row {o_idx}).")
            off_mol = Chem.AddHs(off_mol, addCoords=True)

            for basis_id in sorted(target_pdb_ids):
                basis_path = os.path.join(pdir, f"{basis_id}.pdb")
                basis_txt  = read_text(basis_path)
                basis_res  = group_resname_map.get(basis_id, None)
                if not basis_res:
                    raise RuntimeError(f"[{protein_key}] Missing resname for basis {basis_id} to replace.")
                tgt_block = extract_ligand_block(basis_txt, basis_res)
                if not tgt_block:
                    raise RuntimeError(f"[{protein_key}] Cannot find ligand {basis_res} in basis {basis_id}.")
                tgt_mol = pdb_block_to_mol(tgt_block)
                if tgt_mol is None: 
                    raise RuntimeError(f"[{protein_key}] Basis ligand parse failed for {basis_id}.")
                tgt_mol = Chem.AddHs(tgt_mol, addCoords=True)

                aligned = align_to_pose(off_mol, tgt_mol)
                set_resname_all_atoms(aligned, resname=basis_res, chain="Z", resnum=1)
                aligned_block = Chem.MolToPDBBlock(aligned)

                combined = combine_protein_and_ligand(basis_txt, aligned_block, drop_resname=basis_res)
                outp = os.path.join(pdir, f"{protein_key}__placed_offtarget__basis-{basis_id}.pdb")
                write_text(outp, combined)
                created_any = True

                extra = {}
                if "protein" in gdf.columns: extra["protein"]=rows[0].get("protein","")
                if "ligand"  in gdf.columns: extra["ligand"]=rows[0].get("ligand","")
                if "sequence" in gdf.columns: extra["sequence"]=rows[0].get("sequence","")
                if "key_mutation" in gdf.columns: extra["key_mutation"]=rows[0].get("key_mutation","")
                emit_summary(basis_id, "off-target", extra, created=True, basis="target",
                             saved_t_ids=list(target_pdb_ids), saved_o_ids=list(off_pdb_ids),
                             saved_t_lids=saved_t_lids, saved_o_lids=saved_o_lids)

    # Case B: no target PDB, but off-target PDB exists and target ligand exists → place target ligand into off-target basis
    elif off_pdb_ids and any(idx for idx,_ in enumerate(all_target_rows) if idx in row_lig_blocks):
        for t_idx, t_row in enumerate(all_target_rows):
            if t_idx not in row_lig_blocks: continue
            tgtlig_block = row_lig_blocks[t_idx]
            tgtlig_mol = pdb_block_to_mol(tgtlig_block)
            if tgtlig_mol is None:
                raise RuntimeError(f"[{protein_key}] Target ligand parse failed (row {t_idx}).")
            tgtlig_mol = Chem.AddHs(tgtlig_mol, addCoords=True)

            for basis_id in sorted(off_pdb_ids):
                basis_path = os.path.join(pdir, f"{basis_id}.pdb")
                basis_txt  = read_text(basis_path)
                basis_res  = group_resname_map.get(basis_id, None)
                if not basis_res:
                    raise RuntimeError(f"[{protein_key}] Missing resname for basis {basis_id} to replace.")
                basis_lig_block = extract_ligand_block(basis_txt, basis_res)
                if not basis_lig_block:
                    raise RuntimeError(f"[{protein_key}] Cannot find ligand {basis_res} in basis {basis_id}.")
                basis_lig = pdb_block_to_mol(basis_lig_block)
                if basis_lig is None:
                    raise RuntimeError(f"[{protein_key}] Basis ligand parse failed for {basis_id}.")
                basis_lig = Chem.AddHs(basis_lig, addCoords=True)

                aligned = align_to_pose(tgtlig_mol, basis_lig)
                set_resname_all_atoms(aligned, resname=basis_res, chain="Z", resnum=1)
                aligned_block = Chem.MolToPDBBlock(aligned)

                combined = combine_protein_and_ligand(basis_txt, aligned_block, drop_resname=basis_res)
                outp = os.path.join(pdir, f"{protein_key}__placed_targetlig__basis-{basis_id}.pdb")
                write_text(outp, combined)
                created_any = True

                extra = {}
                if "protein" in gdf.columns: extra["protein"]=rows[0].get("protein","")
                if "ligand"  in gdf.columns: extra["ligand"]=rows[0].get("ligand","")
                if "sequence" in gdf.columns: extra["sequence"]=rows[0].get("sequence","")
                if "key_mutation" in gdf.columns: extra["key_mutation"]=rows[0].get("key_mutation","")
                emit_summary(basis_id, "target", extra, created=True, basis="off_target",
                             saved_t_ids=list(target_pdb_ids), saved_o_ids=list(off_pdb_ids),
                             saved_t_lids=saved_t_lids, saved_o_lids=saved_o_lids)

    # If neither A nor B created anything, mark failed placement rows (download summaries already emitted)
    if not created_any:
        # nothing extra to emit; downloads already recorded with created=no
        pass

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    mkdir(args.out_dir)
    df = pd.read_csv(args.input_csv)

    # required columns
    req = ["protein_key","target_type","complex_pdb_id","ligand_smiles","pdb_ligand_id"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    summary_rows: List[dict] = []

    for protein_key, gdf in df.groupby("protein_key", sort=False):
        try:
            process_group(gdf, protein_key, args.out_dir, summary_rows)
        except Exception as e:
            # emit a failed summary line per group (no saved_pdb_id)
            extra = {}
            if "protein" in gdf.columns: extra["protein"]=gdf.iloc[0].get("protein","")
            if "ligand"  in gdf.columns: extra["ligand"]=gdf.iloc[0].get("ligand","")
            if "sequence" in gdf.columns: extra["sequence"]=gdf.iloc[0].get("sequence","")
            if "key_mutation" in gdf.columns: extra["key_mutation"]=gdf.iloc[0].get("key_mutation","")
            summary_rows.append({
                "protein_key": protein_key,
                "target_type": "mixed",
                "protein": extra.get("protein",""),
                "ligand": extra.get("ligand",""),
                "sequence": extra.get("sequence",""),
                "key_mutation": extra.get("key_mutation",""),
                "saved_target_pdb_id": "",
                "saved_off_target_pdb_id": "",
                "saved_target_ligand_id": "",
                "saved_off_target_ligand_id": "",
                "off_target_pdb_created": "no",
                "placement_basis": "failed",
                "saved_pdb_id": "",
            })
            print(f"[ERROR] {protein_key}: {e}", file=sys.stderr)
            traceback.print_exc()

    out_summary = os.path.join(args.out_dir, "summary.csv")
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    print(f"✅ Summary saved: {out_summary}")

if __name__ == "__main__":
    main()
