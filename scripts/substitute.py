import argparse, os
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def parse_xyz(line): return float(line[30:38]), float(line[38:46]), float(line[46:54])
def set_xyz(line,x,y,z): return f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
def get_res_fields(line):
    return line[17:20].strip(), line[21].strip(), int(line[22:26])
def set_res_fields(line,resn,chain,resid):
    line=f"{line[:17]}{resn:>3s}{line[20:]}"; line=f"{line[:21]}{chain:1s}{line[22:]}"; return f"{line[:22]}{resid:4d}{line[26:]}"
def get_serial(line): 
    try: return int(line[6:11])
    except: return 0
def set_serial(line,serial): return f"{line[:6]}{serial:5d}{line[11:]}"
def get_atom_name(line): return line[12:16]
def set_atom_name(line,name4): return f"{line[:12]}{name4[:4]:<4s}{line[16:]}"
def com_pdb_block(lines):
    pts=[parse_xyz(l) for l in lines if l.startswith(("ATOM  ","HETATM"))]
    if not pts: return (0.0,0.0,0.0)
    a=np.array(pts); m=a.mean(axis=0); return (float(m[0]),float(m[1]),float(m[2]))

def rdkit_from_mol2(path):
    m=Chem.MolFromMol2File(path,removeHs=False,sanitize=False)
    if m is None or m.GetNumAtoms()==0: 
        m=Chem.MolFromMol2File(path,removeHs=False)
    if m is None or m.GetNumAtoms()==0: 
        raise SystemExit("failed to read mol2")
    if m.GetNumConformers()==0:
        m=Chem.AddHs(m,addCoords=True)
        AllChem.EmbedMolecule(m,AllChem.ETKDGv3())
    else:
        try: 
            m=Chem.AddHs(m,addCoords=True)
        except: 
            pass
    return m

def het_lines_from_rdkit(m,resn,chain,resid,start_serial):
    conf=m.GetConformer()
    het=[]; idx2serial={}
    for i,a in enumerate(m.GetAtoms()):
        x,y,z=conf.GetAtomPosition(i).x,conf.GetAtomPosition(i).y,conf.GetAtomPosition(i).z
        elem=a.GetSymbol()
        nm=a.GetProp("_TriposAtomName") if a.HasProp("_TriposAtomName") else (a.GetProp("_Name") if a.HasProp("_Name") else f"{elem}{i+1}")
        nm=(nm[:4]).ljust(4)
        line=f"HETATM{start_serial:5d} {nm:<4s}{resn:>4s} {chain:1s}{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}"
        het.append(line); idx2serial[i]=start_serial; start_serial+=1
    return het, idx2serial

def conect_from_rdkit(m,idx2serial):
    out=[]
    for b in m.GetBonds():
        a1=idx2serial[b.GetBeginAtomIdx()]; a2=idx2serial[b.GetEndAtomIdx()]
        out.append(f"CONECT{a1:5d}{a2:5d}"); out.append(f"CONECT{a2:5d}{a1:5d}")
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--complex_pdb",required=True)
    ap.add_argument("--new_ligand",required=True)
    ap.add_argument("--old_resname",required=True)
    ap.add_argument("--old_chain",required=True)
    ap.add_argument("--old_resid",type=int,required=True)
    ap.add_argument("--out_pdb",required=True)
    ap.add_argument("--out_resname",default=None)
    args=ap.parse_args()

    with open(args.complex_pdb,"r") as f: lines=[l.rstrip("\n") for l in f]
    keep=[]; old_block=[]; max_serial=0
    for l in lines:
        if l.startswith(("ATOM  ","HETATM")):
            max_serial=max(max_serial,get_serial(l))
            resn,chn,resi=get_res_fields(l)
            if l.startswith("HETATM") and (resn,chn,resi)==(args.old_resname,args.old_chain,args.old_resid): 
                old_block.append(l); continue
        if not l.startswith(("CONECT","MASTER")): keep.append(l)
    if not old_block: raise SystemExit("old ligand not found")
    com_old=com_pdb_block(old_block)

    m=rdkit_from_mol2(args.new_ligand)
    het, idx2serial = het_lines_from_rdkit(m, args.out_resname or args.old_resname, args.old_chain, args.old_resid, max_serial+1)
    com_new=com_pdb_block(het)
    dx,dy,dz=com_old[0]-com_new[0], com_old[1]-com_new[1], com_old[2]-com_new[2]
    placed=[set_xyz(l, *(np.array(parse_xyz(l))+np.array([dx,dy,dz]))) for l in het]
    out=[l for l in keep if not l.startswith("END")]
    out.extend(placed)
    out.extend(conect_from_rdkit(m, idx2serial))
    out.append("END")
    with open(args.out_pdb,"w") as f: f.write("\n".join(out)+"\n")

if __name__=="__main__":
    main()
