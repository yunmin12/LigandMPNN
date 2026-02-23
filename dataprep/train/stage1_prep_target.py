"""
Stage 1: Prepare target PDB - extract ligand, standardize, and apply sequence blurring
Based on extract_target_ligands.py

Note: Run first, before Stage 2 (ligand saving)
Skips already prepared PDBs based on actual file existence
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select, Structure, Model, Chain, Atom
from Bio.PDB.Polypeptide import is_aa
import requests
import pandas as pd
from typing import Optional, Tuple
from collections import Counter
import numpy as np

from urllib3 import Retry
from utils.status_tracker import StatusTracker
    

from pathlib import Path
from typing import Optional, Iterable, Tuple
import gzip
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import io
from Bio.PDB.MMCIFParser import MMCIFParser
from typing import Optional, Tuple
import gzip
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

WATER_RESNAMES = {'HOH', 'WAT', 'H2O', 'DOD'}
METAL_ION_RESNAMES = {
    'NA','K','CL','CA','MG','ZN','FE','CU','MN','CO','NI',
    'CD','HG','PB','SR','CS','BA','TI','V','CR','MO','SE','W','RB','YB','AL'
}
BACKBONE_ATOMS = {"N", "CA", "C", "O"}


class BackboneOnlySelect(Select):
    """Select backbone atoms + CB, keep HETATM"""
    def __init__(self):
        self.n_protein_atoms_kept = 0
        self.n_protein_atoms_removed = 0
        self.n_hetero_atoms_kept = 0
        
    def accept_atom(self, atom):
        parent_res = atom.get_parent()
        hetflag = parent_res.id[0]
        
        # Protein residues - keep backbone + CB
        if hetflag == " ":
            if is_aa(parent_res, standard=True):
                # Keep backbone + CB for Alanine
                if atom.name in BACKBONE_ATOMS or atom.name == "CB":
                    self.n_protein_atoms_kept += 1
                    return True
                else:
                    self.n_protein_atoms_removed += 1
                    return False
        
        # HETATM - keep all (ligand)
        self.n_hetero_atoms_kept += 1
        return True


class TargetPreparator:
    def __init__(self, base_dir: str, status_tracker: StatusTracker):
        self.base_dir = Path(base_dir)
        self.status_tracker = status_tracker
        
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.parser = PDBParser(QUIET=True)
        self.io = PDBIO()
        
        # Track skipped entries
        self.n_skipped_existing = 0
        
        # Track auto-selected ligands for verification
        self.auto_selected_ligands = []

    def _make_session(self) -> requests.Session:
        s = requests.Session()

        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        s.mount("https://", adapter)
        s.mount("http://", adapter)

        s.headers.update({
            "User-Agent": "pdb-downloader/1.0 (+https://example.org; contact: you)",
            "Accept": "*/*",
        })
        return s

    def _looks_like_structure_file(self, head_bytes: bytes) -> bool:
        head = head_bytes.lstrip()[:200].upper()
        if head.startswith(b"<!DOCTYPE HTML") or head.startswith(b"<HTML"):
            return False
        if head.startswith((b"HEADER", b"ATOM", b"HETATM", b"MODEL", b"REMARK")):
            return True
        if head.startswith(b"DATA_"):
            return True
        return True

    def _download_to_file(
        self,
        session: requests.Session,
        url: str,
        out_path: Path,
        timeout: Tuple[float, float] = (10.0, 60.0),
    ) -> bool:
        with session.get(url, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                return False

            first = next(r.iter_content(chunk_size=8192), b"")
            if not first:
                return False

            # gz가 아니면 첫 청크로 HTML 여부를 쉽게 거를 수 있음
            is_gz = url.lower().endswith(".gz") or r.headers.get("Content-Encoding", "").lower() == "gzip"

            tmp = out_path.with_suffix(out_path.suffix + ".part")
            tmp.parent.mkdir(parents=True, exist_ok=True)

            if is_gz:
                # gz는 first가 "압축된 바이트"라서 HTML 검사 의미가 없음.
                # 대신 압축을 풀어본 뒤에 헤더 검사.
                compressed = first + r.raw.read()
                try:
                    decompressed = gzip.decompress(compressed)
                except Exception:
                    tmp.unlink(missing_ok=True)
                    return False

                if not decompressed or not self._looks_like_structure_file(decompressed[:200]):
                    tmp.unlink(missing_ok=True)
                    return False

                with open(tmp, "wb") as f_out:
                    f_out.write(decompressed)

            else:
                if not self._looks_like_structure_file(first):
                    tmp.unlink(missing_ok=True)
                    return False

                with open(tmp, "wb") as f_out:
                    f_out.write(first)
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f_out.write(chunk)

            # 최소 크기 체크
            if tmp.stat().st_size < 200:
                tmp.unlink(missing_ok=True)
                return False

            # 최종 HTML 방어
            head = tmp.open("rb").read(200).lstrip().upper()
            if head.startswith((b"<!DOCTYPE HTML", b"<HTML")):
                tmp.unlink(missing_ok=True)
                return False

            tmp.replace(out_path)
            return True

    def download_structure(self, pdb_id: str, out_dir: Path) -> Optional[Path]:
        """
        PDB 또는 CIF를 받아서 out_dir에 저장하고,
        실제로 저장된 파일 경로를 반환.
        """
        pid = pdb_id.strip().lower()
        pidU = pid.upper()

        session = self._make_session()

        out_pdb = out_dir / f"{pidU}.pdb"
        out_cif = out_dir / f"{pidU}.cif"

        candidates = [
            # RCSB
            (f"https://files.rcsb.org/download/{pidU}.pdb", out_pdb),
            (f"https://files.rcsb.org/download/{pidU}.pdb.gz", out_pdb),
            (f"https://files.rcsb.org/download/{pidU}.cif", out_cif),
            (f"https://files.rcsb.org/download/{pidU}.cif.gz", out_cif),

            # PDBe
            (f"https://www.ebi.ac.uk/pdbe/entry-files/download/{pid}_updated.cif", out_cif),
            (f"https://www.ebi.ac.uk/pdbe/entry-files/download/{pid}.cif", out_cif),

            # wwPDB NextGen
            (f"https://files-nextgen.wwpdb.org/download/{pidU}.cif", out_cif),
            (f"https://files-nextgen.wwpdb.org/download/{pidU}.cif.gz", out_cif),
        ]

        # PDBj divided archive (ent.gz)
        subdir = pid[1:3]
        candidates.append(
            (f"https://ftp.pdbj.org/pub/pdb/data/structures/divided/pdb/{subdir}/pdb{pid}.ent.gz", out_pdb)
        )

        for url, out_path in candidates:
            try:
                if self._download_to_file(session, url, out_path):
                    return out_path
            except Exception:
                pass
            time.sleep(0.1)

        return None

    def download_pdb(self, pdb_id: str, out_dir: Path = None) -> Optional[Path]:
        """
        process_row()가 기대하는 인터페이스:
        - 리턴은 "PDB 파일 경로" (Path)
        - temp 파일은 out_dir 아래에 만들고
        - finally에서 temp_pdb.unlink()로 지워도 되게 구성
        """
        if out_dir is None:
            out_dir = self.base_dir
        # process_row finally에서 지우니까 "항상 PDB"로 맞춰주자
        temp_pdb = out_dir / f"{pdb_id.upper()}_temp.pdb"
        temp_cif = out_dir / f"{pdb_id.upper()}_temp.cif"

        # 1) 먼저 구조 파일(PDB or CIF)을 받아온다
        got = self.download_structure(pdb_id, out_dir)
        if got is None or not got.exists():
            return None

        # 2) 이미 PDB면 temp_pdb로 이름만 맞춰서 리턴
        if got.suffix.lower() in [".pdb", ".ent"]:
            if got != temp_pdb:
                try:
                    got.replace(temp_pdb)
                except Exception:
                    # replace가 안 되면 copy로
                    temp_pdb.write_bytes(got.read_bytes())
            return temp_pdb

        # 3) CIF면 PDB로 변환
        if got.suffix.lower() == ".cif":
            # 이름을 temp_cif로 맞추기(정리 편하게)
            if got != temp_cif:
                try:
                    got.replace(temp_cif)
                except Exception:
                    temp_cif.write_bytes(got.read_bytes())
            try:
                parser = MMCIFParser(QUIET=True)
                structure = parser.get_structure("tmp", str(temp_cif))
                self.io.set_structure(structure)
                self.io.save(str(temp_pdb))
                return temp_pdb
            except Exception as e:
                logger.error(f"mmCIF→PDB conversion failed for {pdb_id}: {e}")
                return None
            finally:
                if temp_cif.exists():
                    temp_cif.unlink(missing_ok=True)

        return None
    
    def is_metal_ion(self, residue) -> bool:
        """Check if residue is a metal ion - EXACT COPY FROM ORIGINAL"""
        atoms = list(residue.get_atoms())
        if len(atoms) == 1:
            atom = atoms[0]
            return atom.element.strip().upper() in METAL_ION_RESNAMES
        return False
    
    def find_ligand_in_pdb(self, pdb_file: Path, het_id: Optional[str] = None, 
                          target_complex_id: str = None, pdb_id: str = None):
        """Find ligand location in PDB.
        
        If het_id is provided, find that specific ligand.
        If het_id is None, automatically select the largest organic ligand
        (excluding water and metal ions).
        """
        try:
            structure = self.parser.get_structure('complex', pdb_file)
            
            # Collect all HET residues
            all_het_residues = []
            ligand_candidates = []  # (chain_id, resid, resname, n_atoms)
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0].startswith('H'):  # HETATM
                            resname = residue.resname
                            all_het_residues.append(resname)
                            
                            # If het_id specified, look for exact match
                            if het_id and resname == het_id:
                                return chain.id, residue.id[1], residue.resname, None
                            
                            # If het_id not specified, collect ligand candidates
                            if not het_id:
                                # Skip water and metal ions
                                if resname in WATER_RESNAMES or resname in METAL_ION_RESNAMES:
                                    continue
                                if self.is_metal_ion(residue):
                                    continue
                                
                                # Count atoms (larger = more likely to be the drug-like ligand)
                                n_atoms = len(list(residue.get_atoms()))
                                if n_atoms > 5:  # Minimum size filter
                                    ligand_candidates.append((
                                        chain.id, residue.id[1], resname, n_atoms
                                    ))
            
            # If het_id was specified but not found
            if het_id:
                unique_hets = set(all_het_residues)
                logger.debug(f"Expected {het_id}, found HET residues: {unique_hets}")
                return None, None, None, None
            
            # If het_id was not specified, select largest ligand
            if ligand_candidates:
                # Sort by number of atoms (descending)
                ligand_candidates.sort(key=lambda x: x[3], reverse=True)
                best = ligand_candidates[0]
                logger.info(f"Auto-selected ligand: {best[2]} (chain {best[0]}, {best[3]} atoms)")
                if len(ligand_candidates) > 1:
                    logger.debug(f"  Other candidates: {[(c[2], c[3]) for c in ligand_candidates[1:4]]}")
                
                # Record auto-selection for verification log
                if target_complex_id and pdb_id:
                    self.auto_selected_ligands.append({
                        'target_complex_id': target_complex_id,
                        'pdb_id': pdb_id,
                        'selected_resname': best[2],
                        'selected_chain': best[0],
                        'selected_resid': best[1],
                        'selected_n_atoms': best[3],
                        'other_candidates': '; '.join([f"{c[2]}(chain {c[0]}, {c[3]} atoms)" 
                                                       for c in ligand_candidates[1:5]]) if len(ligand_candidates) > 1 else 'none',
                        'n_candidates': len(ligand_candidates),
                        'all_hetatms': '; '.join(sorted(set(all_het_residues)))
                    })
                
                return best[0], best[1], best[2], ligand_candidates
            
            logger.debug(f"No suitable ligand found. HET residues: {set(all_het_residues)}")
            return None, None, None, None
            
        except Exception as e:
            logger.error(f"Error parsing PDB: {e}")
            return None, None, None, None
    
    def extract_and_standardize(self, pdb_file: Path, chain_id: str, 
                                resid: int, resname: str, output_file: Path) -> bool:
        """
        Extract and standardize ligand - EXACT COPY FROM ORIGINAL
        1. Find target ligand in pdb
        2. Renumber ligand to Chain Z, resnum 1
        3. Remove other HETATM/water
        4. Save standardized pdb
        """
        try:
            structure = self.parser.get_structure('complex', pdb_file)
            
            ligand_found = False
            for model in structure:
                target_chain = None
                target_res = None

                # 1) Find target ligand residue
                for chain in model:
                    if chain.id != chain_id:
                        continue
                    for residue in chain:
                        if (residue.id[1] == resid and 
                            residue.resname == resname and
                            residue.id[0].startswith('H')): # HETATM
                            target_chain = chain
                            target_res = residue
                            ligand_found = True
                            break
                    if target_res is not None:
                        break
                if target_res is None:
                    continue

                # 2) Detach ligand from original chain
                orig_id = target_res.id
                het_field = orig_id[0]
                target_chain.detach_child(orig_id)

                # 3) Create chain Z
                if model.has_id('Z'):
                    chain_z = model['Z']
                else:
                    chain_z = Chain.Chain('Z')
                    model.add(chain_z)

                new_id = (het_field, 1, ' ')  # Standardize to resid 1
                if chain_z.has_id(new_id):
                    chain_z.detach_child(new_id)
                
                # 4) Add ligand to chain Z with standardized residue ID
                new_res = target_res.copy()
                new_res.id = new_id
                chain_z.add(new_res)

                # 5) Remove all other HETATM/water residues
                for chain in list(model):
                    for residue in list(chain):
                        resname_check = residue.resname
                        hetero_flag = residue.id[0]

                        # retain target ligand only (now in Chain Z, ResID 1)
                        if chain.id == 'Z' and residue.id[1] == 1:
                            continue
                        
                        # remove water molecules
                        if resname_check in WATER_RESNAMES:
                            chain.detach_child(residue.id)
                            continue
                        
                        # remove metal ions (by residue name)
                        if resname_check in METAL_ION_RESNAMES:
                            chain.detach_child(residue.id)
                            continue
                        
                        # remove all other HETATM records (hetero_flag is not blank)
                        if hetero_flag.strip():
                            chain.detach_child(residue.id)
                            continue
                
                ligand_found = True
                break

            if not ligand_found:
                logger.warning(f"Ligand {resname} not found at chain {chain_id}, resid {resid}")
                return False
            
            # Save standardized complex pdb
            self.io.set_structure(structure)
            self.io.save(str(output_file))
            
            logger.debug(f"Standardized ligand {resname} → Chain Z, ResID 1")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract/standardize ligand: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _create_virtual_cb(self, residue) -> Optional[Atom.Atom]:
        """Create virtual CB atom for residues that don't have one (e.g., GLY)"""
        try:
            # Need N, CA, C to calculate CB position
            if not (residue.has_id('N') and residue.has_id('CA') and residue.has_id('C')):
                return None
            
            # Get coordinates as numpy arrays
            n_coord = residue['N'].get_coord()
            ca_coord = residue['CA'].get_coord()
            c_coord = residue['C'].get_coord()
            
            # Calculate vectors
            # b: CA -> N direction
            b = n_coord - ca_coord
            # c_dir: CA -> C direction
            c_dir = c_coord - ca_coord
            
            # Normalize vectors
            b = b / np.linalg.norm(b)
            c_dir = c_dir / np.linalg.norm(c_dir)
            
            # CB position: tetrahedral geometry
            # Approximate method: CB is ~109.5° from both CA-N and CA-C
            # Use cross product to get perpendicular direction
            cross = np.cross(c_dir, b)
            cross = cross / np.linalg.norm(cross)
            
            # CB direction: combination of -b (away from N) and cross product
            # This creates approximate tetrahedral geometry
            cb_direction = -b + cross
            cb_direction = cb_direction / np.linalg.norm(cb_direction)
            
            # CB is ~1.54 Å from CA (standard C-C bond length)
            cb_coord = ca_coord + cb_direction * 1.54
            
            # Create CB atom
            cb_atom = Atom.Atom(
                name='CB',
                coord=cb_coord,
                bfactor=20.0,
                occupancy=1.0,
                altloc=' ',
                fullname=' CB ',
                serial_number=0,
                element='C'
            )
            
            return cb_atom
            
        except Exception as e:
            logger.warning(f"Failed to create virtual CB for {residue.get_parent().id}:{residue.id[1]}: {e}")
            return None
    
    def apply_sequence_blurring(self, input_pdb: Path, output_pdb: Path) -> bool:
        """Apply sequence blurring: convert all residues to ALA, keep backbone + CB"""
        try:
            structure = self.parser.get_structure('input', input_pdb)
            
            # First pass: Change all protein residues to ALA and add virtual CB if missing
            n_virtual_cb_added = 0
            for model in structure:
                for chain in model:
                    for residue in list(chain):
                        hetflag = residue.id[0]
                        # Only modify standard protein residues
                        if hetflag == " " and is_aa(residue, standard=True):
                            # Change to ALA
                            residue.resname = "ALA"
                            
                            # Add virtual CB if missing (e.g., GLY)
                            if not residue.has_id('CB'):
                                virtual_cb = self._create_virtual_cb(residue)
                                if virtual_cb:
                                    residue.add(virtual_cb)
                                    n_virtual_cb_added += 1
                                    logger.debug(f"Added virtual CB to {chain.id}:{residue.id[1]}")
            
            # Second pass: Save with selector (keeps backbone + CB, removes side-chains)
            selector = BackboneOnlySelect()
            self.io.set_structure(structure)
            self.io.save(str(output_pdb), selector)
            
            logger.info(f"Sequence blurring completed: {output_pdb}")
            logger.info(f"  Residues → ALA conversion")
            logger.info(f"  Virtual CB atoms added: {n_virtual_cb_added}")
            logger.info(f"  Protein atoms kept: {selector.n_protein_atoms_kept}")
            logger.info(f"  Protein atoms removed: {selector.n_protein_atoms_removed}")
            logger.info(f"  HETATM atoms kept: {selector.n_hetero_atoms_kept}")
            
            return True
            
        except Exception as e:
            logger.error(f"Sequence blurring failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def process_row(self, row: pd.Series, target_complex_id: str, uniprot_id: str) -> Tuple[bool, dict]:
        """
        Process one row: extract target ligand, standardize, and apply sequence blurring
        
        File naming: {base_dir}/{uniprot_id}/target/{target_complex_id}_std.pdb
                     {base_dir}/{uniprot_id}/target/{target_complex_id}_blurred.pdb
        
        Strategy:
        1. Check if both std and blurred PDBs already exist → skip completely
        2. Check if std exists → skip to blurring
        3. If not, check if PDB ID available (HET ID optional)
        4. If available: download and extract
        5. If not available: skip (return False silently)
        6. Apply sequence blurring (if std exists)
        """
        temp_pdb = None
        std_pdb = None
        
        try:
            # Create per-protein target directory
            protein_target_dir = self.base_dir / uniprot_id / "target"
            protein_target_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if output files already exist (skip if both present)
            expected_std = protein_target_dir / f"{target_complex_id}_std.pdb"
            expected_blurred = protein_target_dir / f"{target_complex_id}_blurred.pdb"
            
            if expected_std.exists() and expected_blurred.exists():
                # Both files exist and are valid
                if expected_std.stat().st_size > 100 and expected_blurred.stat().st_size > 100:
                    logger.info(f"{target_complex_id}: ✓ SKIPPED - Both std and blurred PDBs already exist")
                    self.n_skipped_existing += 1
                    self.status_tracker.mark_status(target_complex_id, '1_prep_target', 'success')
                    return True, {
                        'target_std_path': str(expected_std),
                        'target_blurred_path': str(expected_blurred),
                        'target_complex_id': target_complex_id,
                        'skipped': True
                    }
            
            # Check if target_std already exists (but blurred might not)
            target_std_path = row.get('target_std_path')
            
            if expected_std.exists() and expected_std.stat().st_size > 100:
                # Use existing std file
                std_pdb = expected_std
                logger.info(f"{target_complex_id}: Found existing std PDB, proceeding to blurring")
            elif pd.notna(target_std_path) and target_std_path:
                std_pdb = Path(target_std_path)
                if std_pdb.exists():
                    logger.info(f"{target_complex_id}: Using target_std from CSV, proceeding to blurring")
                else:
                    logger.debug(f"{target_complex_id}: target_std_path provided but file not found, will re-extract")
                    std_pdb = None
            else:
                # Need to download and extract
                pdb_id = row.get('valid_pdb_id')
                het_id = row.get('ligand_het_id')
                
                # Check if we have PDB ID (HET ID is now optional)
                if pd.isna(pdb_id) or str(pdb_id).strip().upper() == 'NAN':
                    logger.debug(f"{target_complex_id}: Skipping (no valid_pdb_id)")
                    return False, {'target_complex_id': target_complex_id}
                
                pdb_id = str(pdb_id).strip().upper()
                
                # HET ID is optional - if missing, will auto-select largest ligand
                if pd.notna(het_id) and str(het_id).strip().upper() != 'NAN':
                    het_id = str(het_id).strip()
                else:
                    het_id = None
                    logger.info(f"{target_complex_id}: No HET ID specified, will auto-select ligand")
                
                het_info = f"HET: {het_id}" if het_id else "auto-select ligand"
                logger.info(f"{target_complex_id}: Downloading PDB {pdb_id} ({het_info})")
                
                # Download PDB to per-protein directory
                temp_pdb = self.download_structure(pdb_id, protein_target_dir)
                if not temp_pdb or not temp_pdb.exists():
                    logger.debug(f"{target_complex_id}: Failed to download PDB {pdb_id}")
                    return False, {'target_complex_id': target_complex_id}
                
                # Find ligand in PDB (pass IDs for auto-selection logging)
                chain_id, resid, resname, candidates = self.find_ligand_in_pdb(
                    temp_pdb, het_id, complex_id=target_complex_id, pdb_id=pdb_id
                )
                
                if not chain_id:
                    het_msg = f"Ligand {het_id}" if het_id else "Suitable ligand"
                    logger.debug(f"{target_complex_id}: {het_msg} not found in PDB {pdb_id}")
                    return False, {'target_complex_id': target_complex_id}
                
                logger.debug(f"{target_complex_id}: Found ligand at Chain {chain_id}, ResID {resid}")
                
                # Extract and standardize
                std_pdb = protein_target_dir / f"{target_complex_id}_std.pdb"
                
                success = self.extract_and_standardize(temp_pdb, chain_id, resid, resname, std_pdb)
                if not success:
                    logger.debug(f"{target_complex_id}: Failed to extract/standardize")
                    return False, {'target_complex_id': target_complex_id}
            
            # Apply sequence blurring (always do this if std_pdb exists)
            if not std_pdb or not std_pdb.exists():
                return False, {'target_complex_id': target_complex_id}
            
            blurred_pdb = protein_target_dir / f"{target_complex_id}_blurred.pdb"
            logger.info(f"{target_complex_id}: Applying sequence blurring")
            
            success = self.apply_sequence_blurring(std_pdb, blurred_pdb)
            if not success:
                logger.debug(f"{target_complex_id}: Failed to apply sequence blurring")
                return False, {'target_complex_id': target_complex_id}
            
            logger.info(f"{target_complex_id}: ✓ Success")
            self.status_tracker.mark_status(target_complex_id, '1_prep_target', 'success')
            
            return True, {
                'target_std_path': str(std_pdb),
                'target_blurred_path': str(blurred_pdb),
                'target_complex_id': target_complex_id
            }
            
        except Exception as e:
            logger.error(f"{target_complex_id}: Unexpected error - {e}")
            self.status_tracker.mark_status(target_complex_id, '1_prep_target', 'failed', str(e))
            return False, {'target_complex_id': target_complex_id}
            
        finally:
            # Clean up temp file
            if temp_pdb and temp_pdb.exists():
                try:
                    temp_pdb.unlink()
                except Exception:
                    pass


def main():
    import argparse
    
    p = argparse.ArgumentParser(description="Stage 1: Prepare target PDB (run first)")
    p.add_argument("--csv", required=True, help="Input CSV with PDB information")
    p.add_argument("--base_dir", required=True, help="Base output directory (per-protein subdirs will be created)")
    p.add_argument("--status_dir", required=True, help="Status tracking directory")
    args = p.parse_args()
    
    # Initialize
    status_tracker = StatusTracker(Path(args.status_dir))
    preparator = TargetPreparator(args.base_dir, status_tracker)
    
    # Load CSV
    df = pd.read_csv(args.csv)
    logger.info(f"Loaded {len(df)} total entries")
    
    # ============================================================================
    # INDEXING POLICY (v2 - Pre-indexed CSV)
    # ============================================================================
    # Complex IDs are now assigned by create_index.py before running pipeline.
    # Input CSV must have 'target_complex_id' and 'offtarget_complex_id' columns.
    #
    # Naming format:
    # - Target complex:  {uniprot_id}_tar_{idx:06d}
    # - Off-target:      {uniprot_id}_off_{idx:06d}
    # - Pair (stage3+):  {uniprot_id}_tar_{tar_idx:06d}_off_{off_idx:06d}
    #
    # Run create_index.py first to generate indexed CSV from raw input.
    # ============================================================================
    
    # Validate required columns
    required_cols = ['uniprot_id', 'target_type', 'target_complex_id', 'offtarget_complex_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Required columns missing from input CSV: {missing_cols}")
        logger.error("Please run create_index.py first to generate indexed CSV")
        sys.exit(1)
    
    # Separate target and off-target entries
    target_df = df[df['target_type'] == 'target'].copy()
    off_target_df = df[df['target_type'] != 'target'].copy()
    
    # Calculate statistics from pre-indexed data
    n_proteins = df['uniprot_id'].nunique()
    n_targets_assigned = target_df['target_complex_id'].notna().sum()
    n_offtargets_assigned = off_target_df['offtarget_complex_id'].notna().sum()
    
    logger.info(f"Found {n_proteins} unique proteins (by UniProt ID)")
    logger.info(f"Processing {len(target_df)} target entries ({n_targets_assigned} with IDs)")
    logger.info(f"Passing through {len(off_target_df)} off-target entries ({n_offtargets_assigned} with IDs)")
    
    if len(target_df) == 0:
        logger.error("No target entries found!")
        sys.exit(1)
    
    # Print pre-processing statistics (BEFORE processing)
    print(f"\n{'='*70}")
    print(f"=== STAGE 1 INPUT ANALYSIS ===")
    print(f"{'='*70}")
    print(f"Unique proteins:         {n_proteins}")
    print(f"Target entries:          {len(target_df)}")
    print(f"Off-target entries:      {len(off_target_df)}")
    
    # Check PDB IDs
    has_pdb_id = target_df['valid_pdb_id'].apply(
        lambda x: pd.notna(x) and str(x).strip().upper() != 'NAN'
    ).sum()
    print(f"Rows with valid_pdb_id:  {has_pdb_id:>6} / {len(target_df)}")
    
    # Check HET IDs
    if 'ligand_het_id' in target_df.columns:
        has_het_id = target_df['ligand_het_id'].apply(
            lambda x: pd.notna(x) and str(x).strip().upper() != 'NAN'
        ).sum()
        print(f"Rows with ligand_het_id: {has_het_id:>6} / {len(target_df)}")
    
    # Check processable
    processable = target_df.apply(
        lambda row: (pd.notna(row.get('valid_pdb_id')) and str(row.get('valid_pdb_id')).strip().upper() != 'NAN') or
                    (pd.notna(row.get('target_std_path')) and str(row.get('target_std_path')).strip() != '' and 
                     Path(str(row.get('target_std_path'))).exists()),
        axis=1
    ).sum()
    print(f"Processable rows:        {processable:>6} / {len(target_df)} (has PDB ID or target_std)")
    print(f"{'='*70}\n")
    
    # Process target rows
    updated_target_rows = []
    success_count = 0
    failed_count = 0
    
    for idx, row in target_df.iterrows():
        target_complex_id = row['target_complex_id']
        uniprot_id = str(row['uniprot_id']).strip()
        if uniprot_id.upper() == 'NAN' or not uniprot_id:
            uniprot_id = 'UNKNOWN'
        
        success, updated_data = preparator.process_row(row, target_complex_id, uniprot_id)
        
        if success:
            row_dict = row.to_dict()
            row_dict.update(updated_data)
            updated_target_rows.append(row_dict)
            success_count += 1
        else:
            failed_count += 1
    
    # Pass through off-target rows unchanged (with their assigned offtarget_complex_id)
    off_target_rows = []
    for idx, row in off_target_df.iterrows():
        offtarget_complex_id = row['offtarget_complex_id']
        
        # Mark as success (pass-through)
        status_tracker.mark_status(offtarget_complex_id, '1_prep_target', 'success', 'off_target_passthrough')
        
        off_target_rows.append(row.to_dict())
    
    # Combine target and off-target rows
    all_rows = updated_target_rows + off_target_rows
    
    # Save updated CSV
    output_csv = Path(args.base_dir) / "stage1_output.csv"
    if all_rows:
        pd.DataFrame(all_rows).to_csv(output_csv, index=False)
        logger.info(f"Saved {len(updated_target_rows)} processed target entries")
        logger.info(f"Saved {len(off_target_rows)} pass-through off-target entries")
    else:
        logger.error("No entries to save!")
        pd.DataFrame(columns=['target_complex_id', 'target_std_path', 'target_blurred_path']).to_csv(output_csv, index=False)
    
    # Save auto-selected ligands log for verification
    if preparator.auto_selected_ligands:
        auto_log_csv = Path(args.base_dir) / "stage1_auto_selected_ligands.csv"
        auto_df = pd.DataFrame(preparator.auto_selected_ligands)
        auto_df.to_csv(auto_log_csv, index=False)
        logger.info(f"\n{'='*70}")
        logger.info(f"Auto-selected ligands log: {auto_log_csv}")
        logger.info(f"  Total auto-selections: {len(preparator.auto_selected_ligands)}")
        logger.info(f"  Please review this file to verify ligand selections!")
        logger.info(f"{'='*70}")
    
    # Print summary
    print(f"{'='*70}")
    print(f"=== STAGE 1 SUMMARY ===")
    print(f"{'='*70}")
    print(f"Unique proteins:       {n_proteins}")
    print(f"Target entries:        {len(target_df)}")
    print(f"  Skipped (exists):    {preparator.n_skipped_existing}")
    print(f"  Success (new):       {success_count - preparator.n_skipped_existing}")
    print(f"  Total success:       {success_count}")
    print(f"  Failed:              {failed_count}")
    print(f"  Success rate:        {100*success_count/len(target_df) if len(target_df) > 0 else 0:.1f}%")
    print(f"Off-target entries:    {len(off_target_df)} (passed through)")
    print(f"Auto-selected ligands: {len(preparator.auto_selected_ligands)}")
    print(f"Total output rows:     {len(all_rows)}")
    print(f"Output CSV:            {output_csv}")
    if preparator.auto_selected_ligands:
        auto_log_csv = Path(args.base_dir) / "stage1_auto_selected_ligands.csv"
        print(f"Auto-selection log:    {auto_log_csv}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()