"""
Generate Boltz-2 YAML input files from LigandMPNN results.
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse
from Bio.PDB import PDBParser, PDBIO, Select
from Bio import SeqIO
from rdkit import Chem

# Configuration
p = argparse.ArgumentParser(description="Boltz-2 YAML Input Generator")
p.add_argument("--assay", dest="assay", required=True, choices=["Kd", "IC50"],
                help="Assay type to process (Kd or IC50)")
args = p.parse_args()

ASSAY_TYPE = args.assay
BASE_DIR = Path(f"/scratch/yunmin/data/graph/lmpnn/bdb_pdb/{ASSAY_TYPE}")
LMPNN_IN_DIR = BASE_DIR / "lmpnn_in"
LMPNN_OUT_DIR = BASE_DIR / "lmpnn_out"
RESULTS_CSV = BASE_DIR / f"eval/bindingdb_{ASSAY_TYPE}_all_lmpnn_metrics.csv"
OUTPUT_DIR = BASE_DIR / "boltz2"
CSV_DIR = OUTPUT_DIR / "csvs"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
CSV_DIR.mkdir(exist_ok=True, parents=True)

BOLTZ2_ATOM_LIMIT = 56
SKIP_NATIVE = True
N_SEQUENCES_PER_COMPLEX = 10

def count_heavy_atoms(smiles: str) -> int:
    """Count heavy (non-hydrogen) atoms in molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return mol.GetNumHeavyAtoms()
    except:
        return 0


def parse_sequences_from_fasta(fasta_path: Path, skip_first: bool = True) -> List[Tuple[str, str]]:
    """Parse all sequences from a single FASTA file."""
    sequences = []
    try:
        for idx, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
            if skip_first and idx == 0:
                continue
            sequences.append((record.id, str(record.seq)))
    except Exception as e:
        print(f"Error parsing {fasta_path}: {e}")
        return []
    return sequences


def parse_pdb_chains(pdb_path: Path) -> Dict:
    """Parse PDB to identify protein and ligand chains."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    protein_chains = []
    ligand_info = defaultdict(list)
    
    STANDARD_AA = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                   'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                   'THR', 'TRP', 'TYR', 'VAL'}
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            has_protein = False
            
            for residue in chain:
                resname = residue.resname.strip()
                
                if resname in STANDARD_AA:
                    has_protein = True
                elif residue.id[0].strip():
                    if resname not in ['HOH', 'WAT']:
                        ligand_info[chain_id].append(resname)
            
            if has_protein:
                protein_chains.append(chain_id)
    
    return {
        'protein_chains': protein_chains,
        'ligand_info': dict(ligand_info)
    }


def find_binding_chain(pdb_path: Path, ligand_chain: str) -> str:
    """Find protein chain closest to ligand."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    ligand_atoms = []
    protein_chain_atoms = defaultdict(list)
    
    STANDARD_AA = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                   'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                   'THR', 'TRP', 'TYR', 'VAL'}
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if chain.id == ligand_chain and residue.id[0].strip():
                    ligand_atoms.extend(list(residue.get_atoms()))
                elif residue.resname in STANDARD_AA:
                    protein_chain_atoms[chain.id].extend(list(residue.get_atoms()))
    
    if not ligand_atoms:
        return list(protein_chain_atoms.keys())[0] if protein_chain_atoms else "A"
    
    min_distances = {}
    for chain_id, atoms in protein_chain_atoms.items():
        distances = []
        for lig_atom in ligand_atoms[:10]:
            for prot_atom in atoms[::5]:
                distances.append(lig_atom - prot_atom)
        if distances:
            min_distances[chain_id] = min(distances)
    
    if min_distances:
        return min(min_distances.items(), key=lambda x: x[1])[0]
    return "A"


class LigandExtractor(Select):
    """Select ligand residues for extraction."""
    def __init__(self, ligand_chain_id, ligand_resname, target_ligand_id):
        self.ligand_chain_id = ligand_chain_id
        self.ligand_resname = ligand_resname
        self.target_ligand_id = target_ligand_id
        self.new_chain_id = "Z"
        
    def accept_residue(self, residue):
        # Accept all protein residues from all chains
        STANDARD_AA = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                       'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                       'THR', 'TRP', 'TYR', 'VAL'}
        
        if residue.resname in STANDARD_AA:
            return True
        
        # Accept target ligand from specified chain
        if (residue.get_parent().id == self.ligand_chain_id and 
            residue.resname == self.ligand_resname and
            residue.id[0].strip()):
            return True
            
        return False


def extract_ligand_to_chain_z(pdb_path: Path, ligand_chain_id: str, ligand_resname: str, output_dir: Path) -> Path:
    """Extract ligand from protein chain and move to chain Z."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    # Check if ligand is on a protein chain (has both protein and ligand residues)
    STANDARD_AA = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                   'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                   'THR', 'TRP', 'TYR', 'VAL'}
    
    has_protein_on_ligand_chain = False
    for model in structure:
        for chain in model:
            if chain.id == ligand_chain_id:
                for residue in chain:
                    if residue.resname in STANDARD_AA:
                        has_protein_on_ligand_chain = True
                        break
    
    # If ligand is already on its own chain, no need to extract
    if not has_protein_on_ligand_chain:
        return pdb_path
    
    # Create new structure with ligand moved to chain Z
    new_structure = structure.copy()
    
    # Find and move ligand residues to chain Z
    for model in new_structure:
        # Create new chain Z if it doesn't exist
        if "Z" not in [c.id for c in model]:
            from Bio.PDB.Chain import Chain
            new_chain = Chain("Z")
            model.add(new_chain)
        
        target_chain = None
        ligand_residues_to_move = []
        
        for chain in model:
            if chain.id == ligand_chain_id:
                target_chain = chain
                for residue in list(chain):
                    if residue.resname == ligand_resname and residue.id[0].strip():
                        ligand_residues_to_move.append(residue)
        
        # Move ligand residues to chain Z
        if target_chain and ligand_residues_to_move:
            z_chain = model["Z"]
            for residue in ligand_residues_to_move:
                residue_copy = residue.copy()
                target_chain.detach_child(residue.id)
                z_chain.add(residue_copy)
    
    # Save modified PDB
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"{pdb_path.stem}_chainZ.pdb"
    
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(str(output_path))
    
    return output_path


def generate_boltz2_inputs():
    """Generate Boltz-2 YAML files and manifest."""
    
    print("="*70)
    print("BOLTZ-2 INPUT GENERATION")
    print("="*70)
    
    print(f"\n1. Loading results from {RESULTS_CSV}")
    results_df = pd.read_csv(RESULTS_CSV)
    print(f"   Loaded {len(results_df)} entries")
    
    yaml_dir = OUTPUT_DIR / "inputs"
    yaml_dir.mkdir(exist_ok=True, parents=True)
    
    batch_list = []
    failed_records = []
    
    print(f"\n2. Processing protein-ligand complexes...")
    
    for idx, row in results_df.iterrows():
        uniprot_id = row['uniprot_id']
        lig_id = row['ligand_het_id']
        target_type = 'tar' if row['type'] == 'target' else 'off'
        smiles = row['ligand_smiles']
        
        error_record = {
            'uniprot_id': uniprot_id,
            'lig_id': lig_id,
            'target_type': target_type,
            'smiles': smiles,
            'error': None,
            'status': 'processing'
        }
        
        n_atoms = count_heavy_atoms(smiles)
        exceeds_limit = n_atoms > BOLTZ2_ATOM_LIMIT
        
        if n_atoms == 0:
            error_record['error'] = 'invalid_smiles'
            error_record['status'] = 'failed'
            failed_records.append(error_record)
            continue
        
        # Find PDB path
        input_complex_dir = LMPNN_IN_DIR / uniprot_id / target_type / lig_id
        pdb_path_option1 = input_complex_dir / f"{lig_id}.pdb"
        backbone_dir = LMPNN_OUT_DIR / uniprot_id / target_type / lig_id / "backbones"
        
        pdb_path = None
        if pdb_path_option1.exists():
            pdb_path = pdb_path_option1
        elif backbone_dir.exists():
            backbone_files = list(backbone_dir.glob("*_1.pdb"))
            if backbone_files:
                pdb_path = backbone_files[0]
        
        if pdb_path is None or not pdb_path.exists():
            error_record['error'] = f'pdb_not_found'
            error_record['status'] = 'failed'
            failed_records.append(error_record)
            continue
        
        # Find sequence file
        seq_dir = LMPNN_OUT_DIR / uniprot_id / target_type / lig_id / "seqs"
        if not seq_dir.exists():
            error_record['error'] = f'seqs_dir_not_found'
            error_record['status'] = 'failed'
            failed_records.append(error_record)
            continue
        
        fasta_files = list(seq_dir.glob("*.fa")) + list(seq_dir.glob("*.fasta"))
        
        if len(fasta_files) == 0:
            error_record['error'] = 'no_fasta_file_found'
            error_record['status'] = 'failed'
            failed_records.append(error_record)
            continue
        
        fasta_path = fasta_files[0]
        sequences = parse_sequences_from_fasta(fasta_path, skip_first=SKIP_NATIVE)
        
        if len(sequences) == 0:
            error_record['error'] = f'no_sequences_in_fasta'
            error_record['status'] = 'failed'
            failed_records.append(error_record)
            continue
        
        sequences = sequences[:N_SEQUENCES_PER_COMPLEX]
        
        print(f"   {uniprot_id}/{target_type}/{lig_id}: {len(sequences)} sequences")
        
        # Parse PDB
        try:
            chain_info = parse_pdb_chains(pdb_path)
            protein_chains = chain_info['protein_chains']
            ligand_info = chain_info['ligand_info']
            
            ligand_chain = None
            for chain_id, residues in ligand_info.items():
                if lig_id in residues:
                    ligand_chain = chain_id
                    break
            
            if not ligand_chain:
                ligand_chain = list(ligand_info.keys())[0] if ligand_info else "B"
            
            if len(protein_chains) == 0:
                error_record['error'] = 'no_protein_chains_found'
                error_record['status'] = 'failed'
                failed_records.append(error_record)
                continue
            
            # Extract ligand to chain Z if it's on a protein chain
            modified_pdb_dir = OUTPUT_DIR / "modified_pdbs" / uniprot_id / target_type / lig_id
            modified_pdb_path = extract_ligand_to_chain_z(pdb_path, ligand_chain, lig_id, modified_pdb_dir)
            
            # If ligand was extracted to chain Z, update ligand_chain
            if modified_pdb_path != pdb_path:
                ligand_chain = "Z"
                print(f"      → Extracted ligand to chain Z")
            
            binder_chain = find_binding_chain(modified_pdb_path, ligand_chain)
            
        except Exception as e:
            error_record['error'] = f'pdb_parsing_error: {str(e)}'
            error_record['status'] = 'failed'
            failed_records.append(error_record)
            continue
        
        # Process each designed sequence
        for seq_idx, (seq_id, sequence) in enumerate(sequences):
            try:
                if len(sequence) == 0:
                    continue
                
                yaml_content = {
                    'version': 1,
                    'sequences': [
                        {
                            'protein': {
                                'id': binder_chain,
                                'sequence': sequence,
                                'msa': 'empty'
                            }
                        },
                        {
                            'ligand': {
                                'id': ligand_chain,
                                'ccd': lig_id
                            }
                        }
                    ],
                    'properties': [
                        {
                            'affinity': {
                                'binder': ligand_chain
                            }
                        }
                    ]
                }
                
                # Save YAML with proper indentation
                yaml_filename = f"{uniprot_id}_{target_type}_{lig_id}_seq{seq_idx:02d}.yaml"
                yaml_path = yaml_dir / yaml_filename
                
                # ✅ Use indent=2 to ensure proper formatting
                with open(yaml_path, 'w') as f:
                    yaml.dump(yaml_content, f, 
                             default_flow_style=False, 
                             sort_keys=False,
                             indent=2)  # Ensures proper 2-space indentation
                
                # Add to manifest
                batch_list.append({
                    'yaml_file': yaml_filename,
                    'uniprot_id': uniprot_id,
                    'lig_id': lig_id,
                    'target_type': target_type,
                    'seq_idx': seq_idx,
                    'seq_id': seq_id,
                    'seq_length': len(sequence),
                    'n_atoms': n_atoms,
                    'exceeds_boltz2_limit': exceeds_limit,
                    'boltz2_reliable': not exceeds_limit,
                    'binder_chain': binder_chain,
                    'ligand_chain': ligand_chain,
                    'status': 'success'
                })
                
            except Exception as e:
                error_record_copy = error_record.copy()
                error_record_copy['error'] = f'yaml_generation_error: {str(e)}'
                error_record_copy['status'] = 'failed'
                error_record_copy['seq_idx'] = seq_idx
                failed_records.append(error_record_copy)
    
    # Save results
    print(f"\n3. Saving manifest...")
    manifest_df = pd.DataFrame(batch_list)
    manifest_path = CSV_DIR / "boltz2_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    
    if failed_records:
        failed_df = pd.DataFrame(failed_records)
        failed_path = CSV_DIR / "failed_ligands.csv"
        failed_df.to_csv(failed_path, index=False)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Generated {len(batch_list)} YAML files")
    print(f"   Manifest: {manifest_path}")
    if failed_records:
        print(f"⚠️  {len(failed_records)} failures")
    print(f"{'='*70}")
    
    return manifest_df, failed_records


if __name__ == "__main__":
    manifest, failures = generate_boltz2_inputs()
    print(f"\n✅ Complete!")