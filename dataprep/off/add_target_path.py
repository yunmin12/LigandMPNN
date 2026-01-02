import pandas as pd
from pathlib import Path

csv_path = "/scratch/yunmin/data/graph/lmpnn/bdb_pdb/off_inputs/ligand_sdf/bindingdb_count_set3_with_ligand_paths.csv"
# ligand_dir = Path("/scratch/yunmin/data/graph/lmpnn/bdb_pdb/off_inputs/target_ligands")
ligand_dir = [
    Path("/scratch/yunmin/data/graph/lmpnn/bdb_pdb/Kd/lmpnn_in"),
    Path("/scratch/yunmin/data/graph/lmpnn/bdb_pdb/IC50/lmpnn_in")
    ]

df = pd.read_csv(csv_path)

found = 0
unfound = 0

def find_target_ligand_path(row):
    global found, unfound
    # file_name = f"{row['valid_pdb_id']}_{row['ligand_het_id']}_target.pdb"
    file_name = f"{row['valid_pdb_id']}.pdb"
    file_paths = []
    
    file_path = ligand_dir[0] / f"{row['uniprot_id']}" / "tar" / f"{row['ligand_het_id']}"/ file_name
    file_paths.append(file_path)
    file_path = ligand_dir[1] / f"{row['uniprot_id']}" / "tar" / f"{row['ligand_het_id']}"/ file_name
    file_paths.append(file_path)
    
    for file_path in file_paths:
        if file_path.exists():
            print("Found:", file_path)
            found += 1
            return str(file_path) 
        else:
            # print("Not found:", file_path)
            unfound += 1
            return None

df['target_ligand_path'] = df.apply(find_target_ligand_path, axis=1)

df.to_csv(csv_path, index=False)
print("target_ligand_path column added and CSV updated.")
print(f"Total found: {found}, Total not found: {unfound}")