"""
Superimpose FastRelax ligand params pdb onto Vina docked ligand pdbs.
"""
import sys
import os
import re
from collections import OrderedDict

def extract_element_from_atom_name(atom_name):
    """
    Extract element name from atom name by removing trailing digits.
    Examples:
        C1 -> C
        C21 -> C
        H12 -> H
        Cl3 -> Cl
        BR2 -> BR
        N -> N
    """
    # Remove all trailing digits
    element = re.sub(r'\d+$', '', atom_name)
    return element

def parse_pdb_coords(pdb_file):
    """
    Parse PDB file and return OrderedDict of {(element, element_idx): {data}}.
    Maintains the order of atoms as they appear in the file.
    Key is (element, idx_within_element) to handle multiple atoms of same type.
    Data contains: atom_name (original), element, coords
    """
    coords = OrderedDict()
    element_counts = {}  # Track count of each element type
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # PDB format: columns 13-16 for atom name, 31-38, 39-46, 47-54 for x,y,z
                atom_name = line[12:16].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                element = extract_element_from_atom_name(atom_name)
                
                # Track index for this element type
                if element not in element_counts:
                    element_counts[element] = 0
                element_idx = element_counts[element]
                element_counts[element] += 1
                
                # Use (element, idx) as key
                key = (element, element_idx)
                coords[key] = {
                    'atom_name': atom_name,
                    'element': element,
                    'coords': [x, y, z]
                }
    return coords

def separate_hydrogen_nonhydrogen(coords_dict):
    """
    Separate atoms into non-H and H atoms while preserving order.
    Returns (non_h_list, h_list) where each is list of ((element, idx), data).
    """
    non_h = []
    h_atoms = []
    
    for key, data in coords_dict.items():
        element = data['element']
        if element.upper().startswith('H'):
            h_atoms.append((key, data))
        else:
            non_h.append((key, data))
    
    return non_h, h_atoms

def create_atom_mapping(vina_coords, params_coords):
    """
    Create mapping between vina atoms and params atoms.
    Vina: H atoms can be anywhere, atom names without numbers (C, O, H, Cl, N)
    Params: Non-H first then all H at end, atom names with numbers (C1, O2, H12)
    
    Match by element type and index while maintaining order within non-H and H groups.
    
    Returns dict: {params_key: vina_coords}
    """
    # Separate both into non-H and H
    vina_non_h, vina_h = separate_hydrogen_nonhydrogen(vina_coords)
    params_non_h, params_h = separate_hydrogen_nonhydrogen(params_coords)
    
    # Build mapping
    mapping = OrderedDict()
    
    # Map non-H atoms (should be in same order, match by element)
    if len(vina_non_h) != len(params_non_h):
        print(f"[WARN] Non-H atom count mismatch: vina={len(vina_non_h)}, params={len(params_non_h)}")
    
    for (vina_key, vina_data), (params_key, params_data) in zip(vina_non_h, params_non_h):
        vina_elem = vina_data['element']
        params_elem = params_data['element']
        vina_coords = vina_data['coords']
        vina_atom_name = vina_data['atom_name']
        params_atom_name = params_data['atom_name']
        
        if vina_elem.upper() != params_elem.upper():
            print(f"[WARN] Element mismatch: vina {vina_atom_name}({vina_elem}) != params {params_atom_name}({params_elem})")
        
        # Map params atom to vina coordinates
        mapping[params_key] = vina_coords
        print(f"[DEBUG] Mapped non-H: vina {vina_atom_name} -> params {params_atom_name}")
    
    # Map H atoms (should be in same order, match by element)
    if len(vina_h) != len(params_h):
        print(f"[WARN] H atom count mismatch: vina={len(vina_h)}, params={len(params_h)}")
    
    for (vina_key, vina_data), (params_key, params_data) in zip(vina_h, params_h):
        vina_elem = vina_data['element']
        params_elem = params_data['element']
        vina_coords = vina_data['coords']
        vina_atom_name = vina_data['atom_name']
        params_atom_name = params_data['atom_name']
        
        if not (vina_elem.upper().startswith('H') and params_elem.upper().startswith('H')):
            print(f"[WARN] H atom mismatch: vina {vina_atom_name}({vina_elem}) != params {params_atom_name}({params_elem})")
        
        # Map params atom to vina coordinates
        mapping[params_key] = vina_coords
        print(f"[DEBUG] Mapped H: vina {vina_atom_name} -> params {params_atom_name}")
    
    return mapping

def write_aligned_pdb(params_pdb, params_coords, coord_mapping, output_pdb):
    """
    Write new PDB file with coordinates from coord_mapping.
    Uses params_coords to map keys back to original atom names.
    """
    with open(params_pdb, 'r') as fin, open(output_pdb, 'w') as fout:
        for line in fin:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_name = line[12:16].strip()
                
                # Find the key for this atom in params_coords
                found = False
                for key, data in params_coords.items():
                    if data['atom_name'] == atom_name:
                        if key in coord_mapping:
                            coords = coord_mapping[key]
                            # Keep everything except coordinates (columns 31-54)
                            new_line = (
                                line[:30] +
                                f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}" +
                                line[54:]
                            )
                            fout.write(new_line)
                            found = True
                            break
                
                if not found:
                    print(f"[WARN] Atom {atom_name} not found in mapping")
                    fout.write(line)
            else:
                fout.write(line)

def extract_seed_idx(filename):
    """
    Extract seed index from filename like 'seed0_ligand.pdb', 'seed4_ligand.pdb', etc.
    Returns the index as string, or None if pattern not found.
    """
    match = re.search(r'seed(\d+)_ligand\.pdb', filename)
    if match:
        return match.group(1)
    return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python superimpose_ligands.py <params_pdb> <vina_pdb_0> [vina_pdb_1] ...")
        print("Example: python superimpose_ligands.py ligand_params.pdb seed0_ligand.pdb seed1_ligand.pdb seed2_ligand.pdb seed3_ligand.pdb seed4_ligand.pdb")
        sys.exit(1)
    
    params_pdb = sys.argv[1]
    vina_pdbs = sys.argv[2:]
    
    print(f"[INFO] Params PDB: {params_pdb}")
    print(f"[INFO] Vina PDBs: {vina_pdbs}")
    
    # Parse params_pdb once
    params_coords = parse_pdb_coords(params_pdb)
    print(f"[INFO] Params PDB has {len(params_coords)} atoms")
    
    # Process each vina_pdb
    for vina_pdb in vina_pdbs:
        print(f"\n[INFO] Processing {vina_pdb}")
        
        # Extract seed index from filename
        seed_idx = extract_seed_idx(os.path.basename(vina_pdb))
        if seed_idx is None:
            print(f"[WARN] Could not extract seed index from {vina_pdb}, skipping")
            continue
        
        # Parse vina coordinates
        vina_coords = parse_pdb_coords(vina_pdb)
        print(f"[INFO] Vina PDB has {len(vina_coords)} atoms")
        
        # Create mapping
        coord_mapping = create_atom_mapping(vina_coords, params_coords)
        
        # Generate output filename using seed index
        base_name = os.path.splitext(params_pdb)[0]
        output_dir = os.path.dirname(vina_pdbs[0])
        # output_pdb = os.path.join(output_dir, f"{base_name}_aligned_seed{seed_idx}.pdb")
        output_pdb = os.path.join(output_dir, f"seed{seed_idx}_ligand_aligned.pdb")
        
        # Write aligned PDB
        write_aligned_pdb(params_pdb, params_coords, coord_mapping, output_pdb)
        print(f"[INFO] Wrote {output_pdb}")
    
    print("\n[INFO] All alignments complete!")

if __name__ == '__main__':
    main()
