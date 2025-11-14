#!/usr/bin/env python
import os
import sys
from collections import defaultdict
#from pathlib import Path
#script_path = str(Path(__file__).resolve().parent)
#sys.path.insert(0,script_path+'/../_lib')
from libPDB import Atom,dist_two_coords
CUTOFF = 8.0

class PocketPDB:
    def __init__(self,pdb,ligname):
        self.pdb_fn = pdb.strip()
        self.ligname = ligname.strip()
    def find_close_lig_contact_lig(self,d_cutoff=None):
        if d_cutoff == None:
            cutoff_use = CUTOFF
        else:
            cutoff_use = d_cutoff
        #
        contact_resno = []
        #
        prot_res_to_coords = defaultdict(list)
        with open(self.pdb_fn) as fp:
            lig_coords = []
            for line in fp:
                if line.startswith('ATOM'):
                    atm = Atom(line.strip())
                    if atm.chainID == 'A':
                        prot_res_to_coords[atm.resNo].append(atm.coord)
                elif line.startswith('HETATM'):
                    atm = Atom(line.strip())
                    if atm.resName == self.ligname:
                        lig_coords.append(atm.coord)
        #
        for resno in list(prot_res_to_coords.keys()):
            prot_R_s = prot_res_to_coords[resno]
            status = False
            for prot_R in prot_R_s:
                for lig_R in lig_coords:
                    d = dist_two_coords(prot_R,lig_R)
                    if d <= cutoff_use:
                        status = True
                        break
                if status:
                    break
            if status:
                contact_resno.append(resno)
        return contact_resno

                
                
                    
        
