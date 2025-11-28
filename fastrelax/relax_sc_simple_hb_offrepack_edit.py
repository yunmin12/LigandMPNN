import os
import sys
import glob
import numpy as np
from collections import defaultdict
import pandas as pd

import pyrosetta
from pyrosetta.rosetta.protocols.simple_moves import *
import pyrosetta.rosetta.protocols.rosetta_scripts as rosetta_scripts
import tempfile
import os

# from xml_relax_after_ligMPNN_PTM import XML_BSITE_REPACK_MIN_BETA
from xml_relax_after_ligMPNN_PTM_offrepack import XML_BSITE_REPACK_MIN_BETA
from gen_prot_lig_dist_cst2 import extract_dist_cst_from_pdb_use_allatm,CST_STDERR
from get_pock_res_by_dist_lig import PocketPDB

def repack_pose(work_pose, nres, packable_res, xml_in, tag, cst_fn):
    """Run the RosettaScripts XML and return the modified pose and filter scores."""
    repack_protocol = xml_in.format(packable_res, cst_fn)
    # Build mover from XML and apply in-place to the pose
    parser = rosetta_scripts.RosettaScriptsParser()
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as tmp_xml:
        tmp_xml.write(repack_protocol)
        tmp_xml_path = tmp_xml.name
    try:
        protocol_mover = parser.generate_mover(tmp_xml_path)
        protocol_mover.apply(work_pose)
    finally:
        try:
            os.remove(tmp_xml_path)
        except OSError:
            pass

    # Evaluate filters explicitly to collect scores
    xml_obj = rosetta_scripts.XmlObjects.create_from_string(repack_protocol)
    filter_names = ["fa_rep", "ddg", "cms", "res_totalscore", "totalscore"]
    scores = {}
    for fname in filter_names:
        try:
            fobj = xml_obj.get_filter(fname)
            scores[fname] = fobj.report_sm(work_pose)
        except Exception as exc:
            print(f"[WARN] Failed to evaluate filter {fname}: {exc}")
    return work_pose, scores

class RelaxScore:
    def __init__(self,pdbfn,target_hb_atms,repack_res=[],param_fn=None,seed=12345):
        self.pdbfn = pdbfn.strip()
        self.target_hb_atms = target_hb_atms.split(',')
        self.param_fn = param_fn
        self.tag = self.pdbfn.split('/')[-1].split('.pdb')[0]
        self.output_dir = '/'.join(self.pdbfn.split('/')[:-1])
        self.seed = seed
        self.cst_fn = f'{self.output_dir}/{self.tag}_{self.seed}.cst'
        self.repack_res = repack_res
        self.repack_all = False
        if (len(self.repack_res) == 0):
            self.repack_all = True
    def bsite_repack_min(self,pose_work):
        self.nres = pose_work.total_residue()
        #Repack all if not any residue is assigned
        if self.repack_all and len(self.repack_res) == 0:
            self.repack_res = []
            for ires in range(1,self.nres):
                self.repack_res.append(ires)
            #
        repackable_res_str = ','.join(['%d'%resno for resno in self.repack_res])
        pose_work, filter_scores = repack_pose(
            pose_work,
            self.nres,
            repackable_res_str,
            XML_BSITE_REPACK_MIN_BETA,
            self.tag,
            self.cst_fn,
        )
        out_pdb = f'{self.output_dir}/{self.tag}_betarelax_{self.seed}.pdb'
        return pose_work, out_pdb, filter_scores
    def calc_hb(self,pose_work, verbose=False):
        print(f"[INFO] [RelaxScore] calc_hb for tag={self.tag}")
        full_pose = pose_work
        #
        hbond_set = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
        full_pose.update_residue_neighbors()
        pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(full_pose,False,hbond_set)
        #
        lig_res = full_pose.residue(self.nres)
        lig_atm_hb = defaultdict(list)

        for lig_atmName in self.target_hb_atms:
            lig_query = lig_atmName.strip() #추가
            if verbose:
                print(f"[DEBUG] Checking H-bonds for ligand atom: {lig_query}")
            atm_idx = lig_res.atom_index(lig_query) #lig_atmName -> lig_query 수정
            atm_id = pyrosetta.rosetta.core.id.AtomID(atm_idx,self.nres)
            found_hbs = hbond_set.atom_hbonds(atm_id)
            #
            if (len(found_hbs) == 0):
                if verbose: # 추가
                    print(f'[DEBUG] [HB] {lig_query} : 0 (no hbonds)')
                continue
            for hb in found_hbs:
                don_resNo = hb.don_res()
                don_atmName = full_pose.residue(don_resNo).atom_name(hb.don_hatm())
                acc_resNo = hb.acc_res()
                acc_atmName = full_pose.residue(acc_resNo).atom_name(hb.acc_atm())
                #
                hb_atm = {don_resNo:don_atmName,acc_resNo:acc_atmName}
                #
                hb_res_pair = [don_resNo,acc_resNo]

                for i_res,resno in enumerate(hb_res_pair):
                    if resno == self.nres:
                        other_resno = hb_res_pair[1-i_res]
                        #if it is intra lig hb, continue
                        if other_resno == resno:
                            continue
                        #to remove gaps in atom names from pose
                        lig_atm_hb[hb_atm[resno].strip()].append((other_resno,hb_atm[other_resno]))
        #
        hb_sc = {}
        for lig_atmName in self.target_hb_atms:
            lig_query = lig_atmName.strip() # 추가
            if lig_query not in list(lig_atm_hb.keys()): # lig_atmName -> lig_query
                hb_sc['%s_hbond'%lig_query] = 0 # lig_atmName -> lig_query
                hb_sc['%s_hbond_pairs'%lig_query] ="" # lig_atmName -> lig_query
                if verbose: # 추가
                    print('[HBOND] %s : 0 (no partners)'%lig_query)
                continue
            else:
                tmp = []
                for hb in lig_atm_hb[lig_query]: # lig_atmName -> lig_query
                    if hb not in tmp:
                        tmp.append(hb)
                hb_sc['%s_hbond'%lig_query] = len(tmp) # lig_atmName -> lig_query

                # 추가
                #resno:ATOM 형태로 문자열 만들기
                pair_strs =[]
                for resno, atmname in tmp:
                    pair_strs.append("%s:%s" %(resno, atmname.strip()))
                hb_sc['%s_hbond_pairs'%lig_query] = (";".join(pair_strs))

                if verbose:
                    print("[HBOND] %s: %d (%s)" % (
                        lig_query,
                        len(tmp),
                        hb_sc["%s_hbond_pairs" % lig_query],
                    ))
        return hb_sc


def main():
    pdblist = sys.argv[1]
    ligname = sys.argv[2]
    lig_parm_fn = sys.argv[3] #ligand rosetta params file name
    target_hbatm_names = sys.argv[4] # kept for CLI compatibility; unused when HB calc is off
    dump_pdb = False
    seed = 12345
    if len(sys.argv) > 5:
        if sys.argv[5] == 'dump_pdb':
            dump_pdb = True
            if len(sys.argv) > 6:
                seed = int(sys.argv[6])
        else:
            seed = int(sys.argv[5])
    print("[INFO] ================================")
    print(f"[INFO] PDB list file : {pdblist}")
    print(f"[INFO] Ligand name   : {ligname}")
    print(f"[INFO] Params file   : {lig_parm_fn}")
    print(f"[INFO] Target HB atms: {target_hbatm_names}")
    print(f"[INFO] dump_pdb      : {dump_pdb}")
    print(f"[INFO] seed          : {seed}")
    print("[INFO] ================================")

    #
    df_s = []
    with open(pdblist) as fp:
        for line in fp:
            pyrosetta.init(
                '-beta -gen_potential -in:file:native %s -extra_res_fa %s -run:jran %d'
                % (line.strip(), lig_parm_fn, seed)
            )
            tmp_pdb = PocketPDB(line.strip(),ligname)
            repack_res = tmp_pdb.find_close_lig_contact_lig()
            #
            pdb_score = RelaxScore(line.strip(),target_hbatm_names,repack_res=repack_res,seed=seed)
            #
            csts = extract_dist_cst_from_pdb_use_allatm(line.strip(), ligname)
            fout = open(pdb_score.cst_fn,'w')
            fout.write('%s\n'%('\n'.join(csts)))
            fout.close()
            pose_init = pyrosetta.pose_from_pdb(line.strip())
            pose_work = pose_init.clone()
            pose_work, out_pdb, filter_scores = pdb_score.bsite_repack_min(pose_work)
            sc = {}
            sc.update(filter_scores)
            if dump_pdb:
                pose_work.dump_pdb(f'{out_pdb}')
            sc['tag'] = pdb_score.tag
            print(f"[DEBUG] Finished scoring tag={pdb_score.tag}")
            #
            df_s.append(pd.DataFrame.from_records([sc]))
    #
    sc_df = pd.DataFrame()
    if (len(df_s) > 0):
        sc_df = pd.concat(df_s,ignore_index=True)
    else:
        print("[WARN] No scores collected (df_s is empty).")

    #
    out_csv = './%s_betagen_edit.csv'%pdblist.split('/')[-1]
    sc_df.to_csv(out_csv, index=False)
    print(f'[INFO] worte {out_csv}')
    return sc_df

if __name__ == '__main__':
    main()
