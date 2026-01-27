import argparse
import os
from posixpath import basename
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def compute_mol_features(smiles):
    """Compute physicochemical properties and fingerprints"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Physicochemical properties
        features = {
            'MW': Descriptors.MolWt(mol),
            'cLogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'Charge': Chem.GetFormalCharge(mol),
            # Additional features
            'NumRings': Descriptors.RingCount(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        }
        
        # ECFP4 fingerprint (Morgan fingerprint, radius=2)
        gen = GetMorganGenerator(radius=2, fpSize=2048)
        fp_ecfp4 = gen.GetFingerprint(mol)
        
        # MACCS keys
        fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
        
        # Scaffold
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
        except:
            scaffold_smiles = None
        
        return {
            'mol': mol,
            'features': features,
            'fp_ecfp4': fp_ecfp4,
            'fp_maccs': fp_maccs,
            'scaffold': scaffold_smiles
        }
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return None


def compute_tanimoto_similarity(fp1, fp2):
    """Compute Tanimoto similarity between two fingerprints"""
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def compute_scaffold_similarity(scaffold1, scaffold2):
    """Binary scaffold match (1 if same, 0 if different)"""
    if scaffold1 is None or scaffold2 is None:
        return 0.0
    return 1.0 if scaffold1 == scaffold2 else 0.0

def compute_physicochemical_similarity(features1, features2):
    """Compute normalized euclidean distance between physicochemical properties"""
    # Extract feature vectors
    keys = ['MW', 'cLogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Charge', 'NumRings', 'NumAromaticRings']
    vec1 = np.array([features1[k] for k in keys])
    vec2 = np.array([features2[k] for k in keys])
    
    # Normalize using StandardScaler (fit on both vectors)
    scaler = StandardScaler()
    vecs = np.vstack([vec1, vec2])
    vecs_scaled = scaler.fit_transform(vecs)
    
    # Compute Euclidean distance
    distance = np.linalg.norm(vecs_scaled[0] - vecs_scaled[1])
    
    # Convert to similarity (inverse exponential decay)
    similarity = np.exp(-distance)
    return similarity


def assign_difficulty_bucket(group):
    """Assign difficulty bucket based on composite similarity"""
    group = group.sort_values('similarity_composite', ascending=True).copy()
    n = len(group)
    
    # Easy: lowest 30% similarity (least similar to target)
    # Medium: middle 40%
    # Hard: top 30% similarity (most similar to target, hardest to discriminate)
    
    easy_cutoff = int(n * 0.30)
    hard_cutoff = int(n * 0.70)
    
    group['difficulty_bucket'] = 'medium'
    group.iloc[:easy_cutoff, group.columns.get_loc('difficulty_bucket')] = 'easy'
    group.iloc[hard_cutoff:, group.columns.get_loc('difficulty_bucket')] = 'hard'
    
    return group


def main():
    p = argparse.ArgumentParser(
        description='Compute ligand similarities and assign difficulty buckets'
    )
    p.add_argument(
        '--set',
        type=str,
        required=True,
        choices=['train', 'val', 'test', 'train_std', 'val_std', 'test_std'],
        help='Dataset split (train/val/test/train_std/val_std/test_std)'
    )
    args = p.parse_args()
    set_type = args.set
    
    output_dir = f"/scratch/yunmin/data/graph/train/example/{set_type}"
    # csv_path = f"/scratch/yunmin/data/graph/train/identity50/csvs/06_sample_example/{set_type}_set_sampled_20targets_100offtargets.csv"
    csv_path = f"/scratch/yunmin/data/graph/train/example/{set_type}/{set_type}_set_sampled_after_dataprep.csv"
    total_df = pd.read_csv(csv_path)
    print("Loaded representative sequences dataset")
    basename = os.path.splitext(os.path.basename(csv_path))[0]

    print("\n1️⃣ Computing molecular features and fingerprints...")

    # Load filtered dataset (>100 off-targets, >20 targets)
    print(f"\nProcessing dataset: {len(total_df):,} protein-ligand pairs")

    # Compute features for all ligands
    print("Computing molecular features for all ligands...")
    ligand_data = {}

    unique_smiles = total_df['ligand_smiles'].unique()
    print(f"  Total unique ligands: {len(unique_smiles):,}")

    for i, smiles in enumerate(unique_smiles):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1:,}/{len(unique_smiles):,} ligands...")
        
        mol_data = compute_mol_features(smiles)
        if mol_data:
            ligand_data[smiles] = mol_data

    print(f"✅ Successfully processed {len(ligand_data):,}/{len(unique_smiles):,} ligands")

    # Add features to dataframe
    total_df['mol_valid'] = total_df['ligand_smiles'].isin(ligand_data.keys())
    print(f"  Valid molecules: {total_df['mol_valid'].sum():,}")


    # Process each protein-target pair
    print("\n2️⃣ Computing similarities for each protein-target pair...")

    protein_target_scores = []

    # Get unique proteins
    unique_proteins = total_df[
        ['uniprot_id', 'sequence', 'cluster_id']
    ].drop_duplicates()

    print(f"Processing {len(unique_proteins):,} unique proteins...")

    for protein_idx, protein_row in tqdm(unique_proteins.iterrows(), 
                                        total=len(unique_proteins),
                                        desc="Processing proteins"):
        uniprot_id = protein_row['uniprot_id']
        sequence = protein_row['sequence']
        cluster_id = protein_row['cluster_id']
        
        # Get all ligands for this protein
        protein_df = total_df[
            (total_df['uniprot_id'] == uniprot_id) &
            (total_df['sequence'] == sequence) &
            (total_df['mol_valid'] == True)
        ].copy()
        
        # Get target and off-target ligands
        target_df = protein_df[protein_df['target_type'] == 'target']
        off_target_df = protein_df[protein_df['target_type'] == 'off_target']
        
        if len(target_df) == 0 or len(off_target_df) == 0:
            continue
        
        # Get unique target ligands
        target_smiles_list = target_df['ligand_smiles'].unique()
        
        # For EACH target ligand, compute similarity of ALL off-targets to it
        for target_smiles in target_smiles_list:
            target_data = ligand_data[target_smiles]
            
            # Store similarities for this specific target
            target_off_target_scores = []
            
            for _, off_row in off_target_df.iterrows():
                off_smiles = off_row['ligand_smiles']
                off_data = ligand_data[off_smiles]
                
                # Skip if off-target is same as target
                if off_smiles == target_smiles:
                    continue
                
                # 1. 2D Structural similarity (ECFP4 Tanimoto)
                sim_2d = compute_tanimoto_similarity(
                    off_data['fp_ecfp4'], 
                    target_data['fp_ecfp4']
                )
                
                # 2. Scaffold similarity (Murcko)
                sim_scaffold = compute_scaffold_similarity(
                    off_data['scaffold'],
                    target_data['scaffold']
                )
                
                # 3. Physicochemical similarity
                sim_physchem = compute_physicochemical_similarity(
                    off_data['features'],
                    target_data['features']
                )
                
                # Compute composite similarity score
                composite_score = (
                    0.4 * sim_2d +
                    0.3 * sim_scaffold +
                    0.3 * sim_physchem
                )
                
                target_off_target_scores.append({
                    'uniprot_id': uniprot_id,
                    'sequence': sequence,
                    'cluster_id': cluster_id,
                    'target_smiles': target_smiles,  # not 'ligand_smiles', but 'target_smiles'
                    'off_target_smiles': off_smiles,  # not 'ligand_smiles', but 'off_target_smiles'
                    'similarity_2d': sim_2d,
                    'similarity_scaffold': sim_scaffold,
                    'similarity_physchem': sim_physchem,
                    'similarity_composite': composite_score,
                })
            
            # Add to main list
            protein_target_scores.extend(target_off_target_scores)

    # Convert to DataFrame
    similarity_df = pd.DataFrame(protein_target_scores)
    print(f"\n✅ Computed similarities for {len(similarity_df):,} protein-target-offtarget combinations")
    print(f"   Unique proteins: {similarity_df['uniprot_id'].nunique():,}")
    print(f"   Unique targets: {similarity_df['target_smiles'].nunique():,}")
    print(f"   Unique off-targets: {similarity_df['off_target_smiles'].nunique():,}")

    similarity_df.to_csv(f"{output_dir}/{basename}_similarity.csv", index=False)
    print(f"\n💾 Saved similarity scores to '{basename}_similarity.csv'")


    print("\n3️⃣ Assigning difficulty buckets...")

    # Group by protein and assign buckets
    similarity_df_bucketed = (
        similarity_df
        .groupby(['uniprot_id', 'sequence'], group_keys=False)
        .apply(assign_difficulty_bucket)
        .reset_index(drop=True)
    )

    print(f"\n📊 Difficulty bucket distribution:")
    bucket_counts = similarity_df_bucketed['difficulty_bucket'].value_counts()
    for bucket, count in bucket_counts.items():
        print(f"  - {bucket.capitalize()}: {count:,} ({count/len(similarity_df_bucketed)*100:.1f}%)")

    # Check per-protein distribution
    print(f"\n📊 Per-protein bucket statistics:")
    bucket_stats = (
        similarity_df_bucketed
        .groupby(['uniprot_id', 'sequence'])['difficulty_bucket']
        .value_counts()
        .unstack(fill_value=0)
    )
    print(f"  Mean easy per protein: {bucket_stats['easy'].mean():.1f}")
    print(f"  Mean medium per protein: {bucket_stats['medium'].mean():.1f}")
    print(f"  Mean hard per protein: {bucket_stats['hard'].mean():.1f}")


    print("\n4️⃣ Merging with original dataset...")
    # Merge similarity scores back to total_df
    total_df_copy = total_df.copy()
    total_df_copy = total_df_copy.rename(columns={'ligand_smiles': 'off_target_smiles'})
    total_df_bucketed = total_df_copy.merge(
        similarity_df_bucketed[['uniprot_id', 'sequence', 'target_smiles', 'off_target_smiles', 
                                'similarity_2d', 'similarity_scaffold', 
                                'similarity_physchem', 'similarity_composite', 
                                'difficulty_bucket']],
        on=['uniprot_id', 'sequence', 'off_target_smiles'],
        how='left'
    )

    # reorder_col = [
    #     # protein and sequence information
    #     'protein_key', 'uniprot_id', 'target_name', 'target_source', 'sequence', 'sequence_length', 'cluster_id', 
    #     # ligand information
    #     'ligand_inchikey', 'target_type', 'target_smiles', 'off_target_smiles', 'ligand_name', 'ligand_het_id', 
    #     # experimental data (pdb structure and assay)
    #     'complex_pdb_id', 'valid_pdb_id', 'validation_status', 
    #     'representative_assay_nM', 'assay_type', 'original_assay_nM', 
    #     'curation_source', 'doi', 'data_source', 
    #     # similarity and bucket assignment
    #     'similarity_2d', 'similarity_scaffold', 'similarity_physchem', 'similarity_composite', 'difficulty_bucket',
    #     # molecular features used to similarity calculation
    #     'MW', 'cLogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Charge', 'NumRings', 'NumAromaticRings', 'scaffold', 'mol_valid', 
    #     # SAIR columns
    #     'entry_id (SAIR)', 'index (SAIR)', 'pIC50 (SAIR)', 'family (SAIR)', 'description (SAIR)', 
    #     'vina_score_min (SAIR)', 'number_clashes (SAIR)', 'internal_energy (SAIR)', 
    #     'confidence_score (SAIR)', 'complex_plddt (SAIR)', 'complex_iplddt (SAIR)', 'ptm (SAIR)', 'iptm (SAIR)', 
    #     ]
    reorder_col = [
        # protein and sequence information
        'protein_key', 'uniprot_id', 'target_name', 'target_source', 'sequence', 'sequence_length', 
        # ligand information
        'ligand_inchikey', 'target_type', 'target_smiles', 'off_target_smiles', 'ligand_name', 'ligand_het_id',  'difficulty_bucket',
        # path information
        'cluster_id', 'config_id', 'target_complex_id', 'target_blurred_path', 'offtarget_ligand_path', 'offtarget_pose_path', 'config_file',
        # experimental data (pdb structure and assay)
        'complex_pdb_id', 'valid_pdb_id', 'validation_status', 
        'representative_assay_nM', 'assay_type', 'original_assay_nM', 
        'curation_source', 'doi', 'data_source', 
        # similarity and bucket assignment
        'similarity_2d', 'similarity_scaffold', 'similarity_physchem', 'similarity_composite',
        # molecular features used to similarity calculation
        'MW', 'cLogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Charge', 'NumRings', 'NumAromaticRings', 'scaffold', 'mol_valid', 
        # SAIR columns
        'entry_id (SAIR)', 'index (SAIR)', 'pIC50 (SAIR)', 'family (SAIR)', 
        'description (SAIR)', 'vina_score_min (SAIR)', 'number_clashes (SAIR)', 'internal_energy (SAIR)', 
        'confidence_score (SAIR)', 'complex_plddt (SAIR)', 'complex_iplddt (SAIR)', 'ptm (SAIR)', 'iptm (SAIR)', 
        ]
    total_df_bucketed = total_df_bucketed[reorder_col]

    # Fill NaN for targets in off_target_smiles
    mask_target = total_df_bucketed["target_type"] == "target"
    total_df_bucketed.loc[mask_target, "off_target_smiles"] = pd.NA

    # Fill missing target_smiles from target_smiles of off-target entries
    keys = ["uniprot_id", "sequence", "cluster_id"]
    df = total_df_bucketed
    off_target_src = (
        df["target_smiles"]
        .where(df["target_type"].eq("off_target"))
        .groupby([df[k] for k in keys])
        .transform("first")
    )
    mask_fill = df["target_type"].eq("target") & df["target_smiles"].isna()
    df.loc[mask_fill, "target_smiles"] = off_target_src[mask_fill]
    total_df_bucketed = df

    # Assign "target" to the difficulty_bucket for target entries
    total_df_bucketed['difficulty_bucket'] = (
        total_df_bucketed['difficulty_bucket']
        .fillna('target')
    )

    print(f"✅ Merged similarity data")
    print(f"  Total rows: {len(total_df_bucketed):,}")
    print(f"  Rows with bucket: {(total_df_bucketed['difficulty_bucket'] != 'target').sum():,}")

    print("\n5️⃣ Saving results...")

    # Save bucketed dataset
    total_df_bucketed.to_csv(
        f"{output_dir}/{basename}_bucketed.csv",
        index=False
    )
    print(f"✅ Saved: {output_dir}/{basename}_bucketed.csv")

    # Save similarity statistics
    similarity_df_bucketed.to_csv(
        f"{output_dir}/{basename}_similarity_stat.csv",
        index=False
    )
    print(f"✅ Saved: {output_dir}/{basename}_similarity_stat.csv")

if __name__ == "__main__":
    main()