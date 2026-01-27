"""
Stage 2: Prepare target PDB - extract ligand, standardize, and apply sequence blurring
Based on extract_target_ligands.py
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
    def __init__(self, output_std_dir: str, output_blurred_dir: str, status_tracker: StatusTracker):
        self.output_std_dir = Path(output_std_dir)
        self.output_blurred_dir = Path(output_blurred_dir)
        self.status_tracker = status_tracker
        
        self.output_std_dir.mkdir(parents=True, exist_ok=True)
        self.output_blurred_dir.mkdir(parents=True, exist_ok=True)
        
        self.parser = PDBParser(QUIET=True)
        self.io = PDBIO()

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

    def download_pdb(self, pdb_id: str) -> Optional[Path]:
        """
        process_row()가 기대하는 인터페이스:
        - 리턴은 "PDB 파일 경로" (Path)
        - temp 파일은 self.output_std_dir 아래에 만들고
        - finally에서 temp_pdb.unlink()로 지워도 되게 구성
        """
        # process_row finally에서 지우니까 "항상 PDB"로 맞춰주자
        temp_pdb = self.output_std_dir / f"{pdb_id.upper()}_temp.pdb"
        temp_cif = self.output_std_dir / f"{pdb_id.upper()}_temp.cif"

        # 1) 먼저 구조 파일(PDB or CIF)을 받아온다
        got = self.download_structure(pdb_id, self.output_std_dir)
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
    
    def find_ligand_in_pdb(self, pdb_file: Path, het_id: str):
        """Find ligand location in PDB - EXACT COPY FROM ORIGINAL"""
        try:
            structure = self.parser.get_structure('complex', pdb_file)
            
            # Collect all HET residues for debugging
            all_het_residues = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0].startswith('H'):  # HETATM
                            all_het_residues.append(residue.resname)
                            if residue.resname == het_id:
                                return chain.id, residue.id[1], residue.resname
            
            # Log what was found vs what was expected
            unique_hets = set(all_het_residues)
            logger.debug(f"Expected {het_id}, found HET residues: {unique_hets}")
            
            return None, None, None
        except Exception as e:
            logger.error(f"Error parsing PDB: {e}")
            return None, None, None
    
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
    
    def process_row(self, row: pd.Series, complex_id: str) -> Tuple[bool, dict]:
        """
        Process one row: extract target ligand, standardize, and apply sequence blurring
        
        Strategy:
        1. Check if target_std_path already exists → skip to blurring
        2. If not, check if PDB ID + HET ID available
        3. If available: download and extract
        4. If not available: skip (return False silently)
        5. Apply sequence blurring (if std exists)
        """
        temp_pdb = None
        std_pdb = None
        
        try:
            # Check if target_std already exists
            target_std_path = row.get('target_std_path')
            
            if pd.notna(target_std_path) and target_std_path:
                std_pdb = Path(target_std_path)
                if std_pdb.exists():
                    logger.info(f"{complex_id}: target_std exists, skipping extraction → blurring")
                else:
                    logger.debug(f"{complex_id}: target_std_path provided but file not found, skipping")
                    return False, {'complex_id': complex_id}
            else:
                # Need to download and extract
                pdb_id = row.get('valid_pdb_id')
                het_id = row.get('ligand_het_id')
                
                # Check if we have both PDB ID and HET ID
                if pd.isna(pdb_id) or str(pdb_id).strip().upper() == 'NAN':
                    logger.debug(f"{complex_id}: Skipping (no valid_pdb_id)")
                    return False, {'complex_id': complex_id}
                
                if pd.isna(het_id) or str(het_id).strip().upper() == 'NAN':
                    logger.debug(f"{complex_id}: Skipping (no ligand_het_id)")
                    return False, {'complex_id': complex_id}
                
                pdb_id = str(pdb_id).strip().upper()
                het_id = str(het_id).strip()
                
                logger.info(f"{complex_id}: Downloading PDB {pdb_id} (HET: {het_id})")
                
                # Download PDB
                temp_pdb = self.download_structure(pdb_id, self.output_std_dir)
                if not temp_pdb or not temp_pdb.exists():
                    logger.debug(f"{complex_id}: Failed to download PDB {pdb_id}")
                    return False, {'complex_id': complex_id}
                
                # Find ligand in PDB
                chain_id, resid, resname = self.find_ligand_in_pdb(temp_pdb, het_id)
                
                if not chain_id:
                    logger.debug(f"{complex_id}: Ligand {het_id} not found in PDB {pdb_id}")
                    return False, {'complex_id': complex_id}
                
                logger.debug(f"{complex_id}: Found ligand at Chain {chain_id}, ResID {resid}")
                
                # Extract and standardize
                std_pdb = self.output_std_dir / f"{complex_id}.pdb"
                
                success = self.extract_and_standardize(temp_pdb, chain_id, resid, resname, std_pdb)
                if not success:
                    logger.debug(f"{complex_id}: Failed to extract/standardize")
                    return False, {'complex_id': complex_id}
            
            # Apply sequence blurring (always do this if std_pdb exists)
            if not std_pdb:
                return False, {'complex_id': complex_id}
            
            blurred_pdb = self.output_blurred_dir / f"{complex_id}.pdb"
            logger.info(f"{complex_id}: Applying sequence blurring")
            
            success = self.apply_sequence_blurring(std_pdb, blurred_pdb)
            if not success:
                logger.debug(f"{complex_id}: Failed to apply sequence blurring")
                return False, {'complex_id': complex_id}
            
            logger.info(f"{complex_id}: ✓ Success")
            self.status_tracker.mark_status(complex_id, '2_prep_target', 'success')
            
            return True, {
                'target_std_path': str(std_pdb),
                'target_blurred_path': str(blurred_pdb),
                'complex_id': complex_id
            }
            
        except Exception as e:
            logger.error(f"{complex_id}: Unexpected error - {e}")
            self.status_tracker.mark_status(complex_id, '2_prep_target', 'failed', str(e))
            return False, {'complex_id': complex_id}
            
        finally:
            # Clean up temp file
            if temp_pdb and temp_pdb.exists():
                try:
                    temp_pdb.unlink()
                except Exception:
                    pass


def main():
    import argparse
    
    p = argparse.ArgumentParser(description="Stage 2: Prepare target PDB")
    p.add_argument("--csv", required=True, help="Input CSV (output from stage 1)")
    p.add_argument("--output_std_dir", required=True, help="Output directory for standardized PDBs")
    p.add_argument("--output_blurred_dir", required=True, help="Output directory for blurred PDBs")
    p.add_argument("--status_dir", required=True, help="Status tracking directory")
    args = p.parse_args()
    
    # Initialize
    status_tracker = StatusTracker(Path(args.status_dir))
    preparator = TargetPreparator(args.output_std_dir, args.output_blurred_dir, status_tracker)
    
    # Load CSV
    df = pd.read_csv(args.csv)
    logger.info(f"Loaded {len(df)} total entries")
    
    # Separate target and off-target entries
    target_df = df[df['target_type'] == 'target'].copy()
    off_target_df = df[df['target_type'] != 'target'].copy()
    
    logger.info(f"Processing {len(target_df)} target entries")
    logger.info(f"Passing through {len(off_target_df)} off-target entries unchanged")
    
    if len(target_df) == 0:
        logger.error("No target entries found!")
        sys.exit(1)
    
    # Print pre-processing statistics (BEFORE processing)
    print(f"\n{'='*70}")
    print(f"=== STAGE 2 INPUT ANALYSIS ===")
    print(f"{'='*70}")
    
    # Check PDB IDs
    has_pdb_id = target_df['valid_pdb_id'].apply(
        lambda x: pd.notna(x) and str(x).strip().upper() != 'NAN'
    ).sum()
    print(f"Rows with valid_pdb_id:  {has_pdb_id:>6} / {len(target_df)}")
    
    # Check HET IDs
    has_het_id = target_df['ligand_het_id'].apply(
        lambda x: pd.notna(x) and str(x).strip().upper() != 'NAN'
    ).sum()
    print(f"Rows with ligand_het_id: {has_het_id:>6} / {len(target_df)}")
    
    # Check both
    both = target_df.apply(
        lambda row: (pd.notna(row['valid_pdb_id']) and str(row['valid_pdb_id']).strip().upper() != 'NAN' and
                     pd.notna(row['ligand_het_id']) and str(row['ligand_het_id']).strip().upper() != 'NAN'),
        axis=1
    ).sum()
    print(f"Rows with BOTH:          {both:>6} / {len(target_df)}")
    print(f"{'='*70}\n")
    
    # Process target rows
    updated_target_rows = []
    success_count = 0
    failed_count = 0
    
    for idx, row in target_df.iterrows():
        complex_id = row.get('complex_id', f"complex_{idx:06d}")
        
        success, updated_data = preparator.process_row(row, complex_id)
        
        if success:
            row_dict = row.to_dict()
            row_dict.update(updated_data)
            updated_target_rows.append(row_dict)
            success_count += 1
        else:
            failed_count += 1
    
    # Pass through off-target rows unchanged
    off_target_rows = []
    for idx, row in off_target_df.iterrows():
        complex_id = row.get('complex_id', f"complex_{idx:06d}")
        
        # Mark as success (pass-through)
        status_tracker.mark_status(complex_id, '2_prep_target', 'success', 'off_target_passthrough')
        
        off_target_rows.append(row.to_dict())
    
    # Combine target and off-target rows
    all_rows = updated_target_rows + off_target_rows
    
    # Save updated CSV
    output_csv = Path(args.output_std_dir).parent / "stage2_output.csv"
    if all_rows:
        pd.DataFrame(all_rows).to_csv(output_csv, index=False)
        logger.info(f"Saved {len(updated_target_rows)} processed target entries")
        logger.info(f"Saved {len(off_target_rows)} pass-through off-target entries")
    else:
        logger.error("No entries to save!")
        # Still create empty CSV with proper columns
        pd.DataFrame(columns=['complex_id', 'target_std_path', 'target_blurred_path']).to_csv(output_csv, index=False)
    
    # Print summary
    print(f"{'='*70}")
    print(f"=== STAGE 2 SUMMARY ===")
    print(f"{'='*70}")
    print(f"Target entries:        {len(target_df)}")
    print(f"  Success:             {success_count}")
    print(f"  Failed:              {failed_count}")
    print(f"  Success rate:        {100*success_count/len(target_df) if len(target_df) > 0 else 0:.1f}%")
    print(f"Off-target entries:    {len(off_target_df)} (passed through)")
    print(f"Total output rows:     {len(all_rows)}")
    print(f"Output CSV:            {output_csv}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()