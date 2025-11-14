import os
import sys
import glob
import numpy as np
from collections import defaultdict
import pandas as pd

import pyrosetta
from pyrosetta.rosetta.protocols.simple_moves import *
import pyrosetta.rosetta.protocols.rosetta_scripts as rosetta_scripts

import pyrosetta.distributed.io as io
import pyrosetta.distributed.packed_pose as packed_pose
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts
import pyrosetta.distributed.tasks.score as score

from xml_relax_after_ligMPNN_PTM import XML_BSITE_REPACK_MIN_BETA
from gen_prot_lig_dist_cst2 import extract_dist_cst_from_pdb_use_allatm,CST_STDERR
from get_pock_res_by_dist_lig import PocketPDB

def repack_pose(work_pose, nres, packable_res, xml_in, tag, cst_fn):
    #cst_fn is global var.
    repack_protocol = xml_in.format(packable_res,cst_fn)
#    repack_protocol = xml_in.format(packable_res,nres)
#    pyrosetta.rosetta.core.pose.setPoseExtraScore(work_pose,"tag",'%s'%tag)
    return rosetta_scripts.SingleoutputRosettaScriptsTask(repack_protocol), work_pose

class RelaxScore:
    def __init__(self,pdbfn,target_hb_atms,repack_res=[],param_fn=None):
        self.pdbfn = pdbfn.strip()
        self.target_hb_atms = target_hb_atms.split(',')
        self.param_fn = param_fn
        self.tag = self.pdbfn.split('/')[-1].split('.pdb')[0]
        self.output_dir = '/'.join(self.pdbfn.split('/')[:-1])
        self.cst_fn = f'{self.output_dir}/{self.tag}.cst'
        self.repack_res = repack_res
        self.repack_all = False
        if (len(self.repack_res) == 0):
            self.repack_all = True
    def bsite_repack_min(self,pose_work):
        self.nres = len(pose_work)
        #Repack all if not any residue is assigned
        if self.repack_all and len(self.repack_res) == 0:
            self.repack_res = []
            for ires in range(1,self.nres):
                self.repack_res.append(ires)
            #
        repackable_res_str = ','.join(['%d'%resno for resno in self.repack_res])
        xml_obj, pose_work = repack_pose(pose_work,self.nres,repackable_res_str,XML_BSITE_REPACK_MIN_BETA,self.tag,self.cst_fn)
        xml_obj.setup()
        pose_work = xml_obj(pose_work)
        out_pdb = '%s/%s_betarelax.pdb'%(self.output_dir,self.tag)
        return pose_work, out_pdb
    def calc_hb(self,pose_work):
        full_pose = packed_pose.to_pose(pose_work)
        #
        hbond_set = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
        full_pose.update_residue_neighbors()
        pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(full_pose,False,hbond_set)
        #
        lig_res = full_pose.residue(self.nres)
        lig_atm_hb = defaultdict(list)
        for lig_atmName in self.target_hb_atms:
            atm_idx = lig_res.atom_index(lig_atmName)
            atm_id = pyrosetta.rosetta.core.id.AtomID(atm_idx,self.nres)
            found_hbs = hbond_set.atom_hbonds(atm_id)
            #
            if (len(found_hbs) == 0):
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
            if lig_atmName not in list(lig_atm_hb.keys()):
                hb_sc['%s_hbond'%lig_atmName] = 0
            else:
                tmp = []
                for hb in lig_atm_hb[lig_atmName]:
                    if hb not in tmp:
                        tmp.append(hb)
                hb_sc['%s_hbond'%lig_atmName] = len(tmp)
        return hb_sc


def main():
    pdblist = sys.argv[1]
    ligname = sys.argv[2]
    lig_parm_fn = sys.argv[3] #ligand rosetta params file name
    target_hbatm_names = sys.argv[4] #target ligand atom names to calculate hbond
    dump_pdb = False
    if len(sys.argv) > 5 and sys.argv[5] == 'dump_pdb':
        dump_pdb = True
    #
    df_s = []
    with open(pdblist) as fp:
        for line in fp:
            pyrosetta.init('-beta -gen_potential -in:file:native %s -extra_res_fa %s'%(line.strip(),lig_parm_fn))
            tmp_pdb = PocketPDB(line.strip(),ligname)
            repack_res = tmp_pdb.find_close_lig_contact_lig()
            #
            pdb_score = RelaxScore(line.strip(),target_hbatm_names,repack_res=repack_res)
            #
            csts = extract_dist_cst_from_pdb_use_allatm(line.strip(), ligname)
            fout = open(pdb_score.cst_fn,'w')
            fout.write('%s\n'%('\n'.join(csts)))
            fout.close()
            pose_init = pyrosetta.pose_from_pdb(line.strip())
            pose_work = pose_init.clone()
            pose_work, out_pdb = pdb_score.bsite_repack_min(pose_work)
            sc = pose_work.scores
            hb_d = pdb_score.calc_hb(pose_work)
            sc.update(hb_d)
            if dump_pdb:
                packed_pose.to_pose(pose_work).dump_pdb(f'{out_pdb}')
            sc['tag'] = pdb_score.tag
            #
            df_s.append(pd.DataFrame.from_records([sc]))
    #
    sc_df = pd.DataFrame()
    if (len(df_s) > 0):
        sc_df = pd.concat(df_s,ignore_index=True)
        hbatm_s = target_hbatm_names.split(',')
        hb_tags = []
        for atmname in hbatm_s:
             hb_tags.append('%s_hbond'%atmname)
        sc_df['lig_hb_sum'] = sc_df[hb_tags].sum(axis=1)
    #
    sc_df.to_csv('./%s_betagen.csv'%pdblist.split('/')[-1])
    return sc_df

if __name__ == '__main__':
    main()
