import os, re, sys, traceback, requests
import argparse
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdFMCS, rdchem

# ---------- utils ----------
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

def mkdir(p): os.makedirs(p, exist_ok=True)

def safe_name(s: str) -> str:
    return re.sub(r"[^\w\-.]", "_", str(s))

def norm_ttype(s: str) -> str:
    return (str(s) or "").strip().lower().replace("-", "_")

def download_pdb(pdb_id: str, out_path: str, timeout=30) -> bool:
    url = RCSB_PDB_URL.format(pdb_id=pdb_id.upper())
    r = requests.get(url, timeout=timeout)
    if r.status_code == 200 and ("ATOM" in r.text or "HETATM" in r.text):
        with open(out_path, "w") as f: f.write(r.text)
        return True
    return False

def parse_resname_map(raw: str) -> Dict[str, str]:
    """'6TA3:MZK;6TIW:MZK' -> {'6TA3':'MZK','6TIW':'MZK'} (따옴표/전각/쉼표 정규화)"""
    m = {}
    if not isinstance(raw, str) or not raw.strip():
        return m
    s = raw.strip().strip("'").strip('"')
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

def smiles_to_3d_pdb_block(smiles: str, resname: str="LIG") -> Optional[str]:
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
        info.SetResidueName((resname or "LIG")[:3]); info.SetResidueNumber(1)
        info.SetChainId('Z'); info.SetIsHeteroAtom(True)
        a.SetMonomerInfo(info)
    return Chem.MolToPDBBlock(mol)

def read_text(p):
    with open(p, "r") as f: return f.read()

def write_text(p, s):
    with open(p, "w") as f: f.write(s)

def extract_ligand_block_by_resname(pdb_text: str, resname: str) -> Optional[str]:
    lines = []
    for line in pdb_text.splitlines():
        if line.startswith("HETATM") and line[17:20].strip().upper() == (resname or "").upper():
            lines.append(line)
    if not lines: return None
    return "\n".join(lines) + "\n"

def remove_resname_from_pdb(pdb_text: str, resname: str) -> str:
    out = []
    for line in pdb_text.splitlines():
        if line.startswith("HETATM") and line[17:20].strip().upper() == (resname or "").upper():
            continue
        if line.startswith("CONECT"):  # 간단화: CONECT는 제거
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
    rdMolAlign.AlignMol(mov, ref)
    return mov

def set_resname_all_atoms(m: Chem.Mol, resname="LIG", chain="Z", resnum=1):
    for a in m.GetAtoms():
        info = rdchem.AtomPDBResidueInfo()
        info.SetResidueName((resname or "LIG")[:3])
        info.SetResidueNumber(resnum)
        info.SetChainId(chain)
        info.SetIsHeteroAtom(True)
        a.SetMonomerInfo(info)

# ---------- core ----------
def process_group(gdf: pd.DataFrame, protein_key: str, out_dir: str, summary_rows: List[dict]):
    # Directory: {uniprot_key}_{key_mutation}
    base_key = re.split(r'[_\s]+', str(protein_key).strip())[0]
    if "key_mutation" in gdf.columns and len(gdf["key_mutation"].dropna())>0:
        raw_mut = str(gdf["key_mutation"].dropna().astype(str).iloc[0])
    else:
        raw_mut = "WT"
    mut_list = re.findall(r"[A-Z]\d+[A-Z]", raw_mut)
    mut_part = "_".join(mut_list) if mut_list else "WT"

    safe_root = f"{safe_name(base_key)}_{safe_name(mut_part)}"
    pdir = os.path.join(out_dir, safe_root); mkdir(pdir)
    complex_dir  = os.path.join(pdir, "complex");  mkdir(complex_dir)
    standard_dir = os.path.join(pdir, "standard"); mkdir(standard_dir)
    ligands_dir  = os.path.join(pdir, "ligands");  mkdir(ligands_dir)
    lig_target   = os.path.join(ligands_dir, "target"); mkdir(lig_target)
    lig_off      = os.path.join(ligands_dir, "off");    mkdir(lig_off)
    replaced_dir = os.path.join(pdir, "replaced"); mkdir(replaced_dir)

    def warn(basis_id: str, off_lig_id: str, msg: str):
        summary_rows.append({
            "protein_key": base_key,
            "key_mutation": mut_part,
            "basis_pdb_id": basis_id or "",
            "off_target_ligand_id": off_lig_id or "",
            "status": "warn",
            "message": msg,
            "output_file": ""
        })
        print(f"[WARN] {base_key}_{mut_part} / {basis_id}: {msg}", file=sys.stderr)

    # pass a row with empty complex_pdb_id
    rows: List[pd.Series] = []
    for _, r in gdf.iterrows():
        cids = str(r.get("complex_pdb_id","") or "").strip()
        if not cids or cids.lower() in ["nan","none",""]:
            continue
        rows.append(r)
    if not rows: 
        return

    # PDBID->resname mapping (first appearance is taken)
    group_resname_map: Dict[str,str] = {}
    for r in rows:
        m = parse_resname_map(str(r.get("pdb_ligand_id","") or ""))
        for k,v in m.items():
            if k not in group_resname_map: group_resname_map[k] = v

    # Download complex PDBs
    target_pdb_ids: Set[str] = set()
    off_pdb_ids: Set[str]    = set()
    for r in rows:
        ttype = norm_ttype(r.get("target_type",""))
        cids = [p.strip().upper() for p in re.split(r"[,\s]+", str(r["complex_pdb_id"]).strip()) if p.strip()]
        for pid in cids:
            path = os.path.join(complex_dir, f"{pid}.pdb")
            if os.path.exists(path):
                if ttype == "off_target": off_pdb_ids.add(pid)
                else: target_pdb_ids.add(pid)
                continue
            ok = download_pdb(pid, path)
            if ok:
                if ttype == "off_target": off_pdb_ids.add(pid)
                else: target_pdb_ids.add(pid)

    # Generate and save the ligand PDBs (ligands/target or off/{ligand_id}.pdb)
    row_lig_blocks: Dict[int, str] = {}
    row_lig_resname: Dict[int, str] = {}
    for idx, r in enumerate(rows):
        smiles = str(r.get("ligand_smiles","") or "").strip()
        if not smiles: 
            continue
        cids = [p.strip().upper() for p in re.split(r"[,\s]+", str(r["complex_pdb_id"]).strip()) if p.strip()]
        row_map = parse_resname_map(str(r.get("pdb_ligand_id","") or ""))
        resname = None
        for cid in cids:
            if cid in row_map:
                resname = row_map[cid]; break
        if resname is None and len(set(group_resname_map.values())) == 1 and len(group_resname_map) > 0:
            resname = list(set(group_resname_map.values()))[0]
        if resname is None:
            resname = "LIG"

        block = smiles_to_3d_pdb_block(smiles, resname=resname)
        if block:
            ttype = norm_ttype(r.get("target_type",""))
            lig_dir = lig_target if ttype == "target" else lig_off
            lig_path = os.path.join(lig_dir, f"{resname}.pdb")
            if not os.path.exists(lig_path):
                write_text(lig_path, block)
            row_lig_blocks[idx]  = block
            row_lig_resname[idx] = resname

    def emit_download_summary(saved_pid: str, ttype: str):
        row = {
            "protein_key": base_key,
            "key_mutation": mut_part,
            "target_type": ttype,
            "saved_pdb_id": saved_pid or "",
            "saved_target_pdb_id": ";".join(sorted(target_pdb_ids)) if target_pdb_ids else "",
            "saved_off_target_pdb_id": ";".join(sorted(off_pdb_ids)) if off_pdb_ids else "",
            "saved_target_ligand_id": ";".join(sorted({v for k,v in group_resname_map.items() if k in target_pdb_ids})) if target_pdb_ids else "",
            "saved_off_target_ligand_id": ";".join(sorted({v for k,v in group_resname_map.items() if k in off_pdb_ids})) if off_pdb_ids else "",
            "off_target_pdb_created": "no",
            "placement_basis": "none",
            "status": "info",
            "message": "",
            "output_file": ""
        }
        summary_rows.append(row)

    for pid in sorted(target_pdb_ids):
        emit_download_summary(pid, "target")
    for pid in sorted(off_pdb_ids):
        emit_download_summary(pid, "off-target")

    # ---------- Only when target complex PDB exists ----------
    for basis_id in sorted(target_pdb_ids):
        basis_path = os.path.join(complex_dir, f"{basis_id}.pdb")
        if not os.path.exists(basis_path):
            warn(basis_id, "", "Basis PDB not found on disk.")
            continue
        basis_txt = read_text(basis_path)

        basis_res = group_resname_map.get(basis_id, None)
        if not basis_res:
            warn(basis_id, "", f"Missing mapped ligand residue for basis {basis_id}.")
            continue

        tgt_block = extract_ligand_block_by_resname(basis_txt, basis_res)
        if not tgt_block:
            warn(basis_id, "", f"Target ligand '{basis_res}' not found in basis {basis_id}.")
            continue

        tgt_mol = pdb_block_to_mol(tgt_block)
        if tgt_mol is None:
            warn(basis_id, "", f"Basis ligand parse failed for {basis_id} (PDB parse).")
            continue
        else:
            set_resname_all_atoms(tgt_mol, resname=basis_res, chain="Z", resnum=1)
            std_block = Chem.MolToPDBBlock(tgt_mol)

            standardized = combine_protein_and_ligand(basis_txt, std_block, drop_resname=basis_res)

            outp_std = os.path.join(standard_dir, f"{safe_root}_{basis_id}_{basis_res}.pdb")
            write_text(outp_std, standardized)

        tgt_mol = Chem.AddHs(tgt_mol, addCoords=True)

        # Replace with off-target ligands
        for idx, r in enumerate(rows):
            ttype = norm_ttype(r.get("target_type",""))
            if ttype != "off_target":
                continue
            if idx not in row_lig_blocks:
                warn(basis_id, "", "Off-target ligand PDB for this row is missing (skipped).")
                continue

            off_resname = row_lig_resname.get(idx, "LIG")
            off_block   = row_lig_blocks[idx]
            off_mol = pdb_block_to_mol(off_block)
            if off_mol is None:
                warn(basis_id, off_resname, "Off-target ligand parse failed for this row.")
                continue
            off_mol = Chem.AddHs(off_mol, addCoords=True)

            try:
                aligned = align_to_pose(off_mol, tgt_mol)
            except Exception as e:
                warn(basis_id, off_resname, f"Align failed: {e}")
                continue

            set_resname_all_atoms(aligned, resname=off_resname, chain="Z", resnum=1)
            aligned_block = Chem.MolToPDBBlock(aligned)

            combined = combine_protein_and_ligand(basis_txt, aligned_block, drop_resname=basis_res)
            outp = os.path.join(replaced_dir, f"{safe_root}_{basis_id}_{off_resname}.pdb")
            write_text(outp, combined)

            summary_rows.append({
                "protein_key": base_key,
                "key_mutation": mut_part,
                "basis_pdb_id": basis_id,
                "off_target_ligand_id": off_resname,
                "status": "ok",
                "message": "",
                "output_file": outp,
                "placement_basis": "target",
                "off_target_pdb_created": "yes"
            })

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    mkdir(args.out_dir)
    df = pd.read_csv(args.input_csv)

    # required columns
    req = ["protein_key","target_type","complex_pdb_id","ligand_smiles","pdb_ligand_id","key_mutation"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    summary_rows: List[dict] = []

    # Group by protein_key + key_mutation
    for (protein_key, key_mutation), gdf in df.groupby(["protein_key","key_mutation"], sort=False):
        try:
            process_group(gdf, protein_key, args.out_dir, summary_rows)
        except Exception as e:
            summary_rows.append({
                "protein_key": protein_key,
                "key_mutation": key_mutation,
                "status": "error",
                "message": str(e),
                "output_file": ""
            })
            print(f"[ERROR] {protein_key}_{key_mutation}: {e}", file=sys.stderr)
            traceback.print_exc()

    out_summary = os.path.join(args.out_dir, "summary.csv")
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    print(f"✅ Summary saved: {out_summary}")

if __name__ == "__main__":
    main()
