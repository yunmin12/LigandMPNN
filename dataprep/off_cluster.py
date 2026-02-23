import argparse
import os
from posixpath import basename
import shutil
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
import time
import pickle
import gc
warnings.filterwarnings('ignore')

def check_disk_space(path, min_gb=10):
    """Check if there's enough disk space available"""
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    if free_gb < min_gb:
        print(f"\n⚠️  WARNING: Low disk space! Only {free_gb:.1f} GB free")
        print(f"   Please free up space or change output directory")
        if free_gb < 1:
            raise OSError(f"Critically low disk space ({free_gb:.2f} GB). Aborting to prevent data corruption.")
    return free_gb

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
        '--mode',
        type=str,
        default='target_vs_offtarget',
        choices=['target_vs_offtarget', 'all_pairwise'],
        help='Similarity computation mode: target_vs_offtarget (default) or all_pairwise'
    )
    p.add_argument(
        '--similarity_methods',
        type=str,
        nargs='+',
        default=['2d', 'scaffold', 'physchem'],
        choices=['2d', 'scaffold', 'physchem'],
        help='Similarity methods to compute (default: all). Options: 2d (Tanimoto), scaffold, physchem'
    )
    p.add_argument(
        '--benchmark',
        action='store_true',
        help='Run a quick benchmark on first 100 ligand pairs to estimate computation time'
    )
    p.add_argument(
        '--chunk_size',
        type=int,
        default=100,
        help='Number of proteins to process before saving intermediate results (default: 100)'
    )
    p.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous checkpoint if available'
    )
    p.add_argument(
        '--max_ligands',
        type=int,
        default=0,
        help='Max ligands per protein for pairwise comparison (0=no limit). '
             'Proteins exceeding this are randomly subsampled. Recommended: 2000-5000'
    )
    p.add_argument(
        '--write_batch_size',
        type=int,
        default=100000,
        help='Number of pairwise results to buffer before flushing to disk (default: 100000)'
    )
    # p.add_argument(
    #     '--set',
    #     type=str,
    #     required=True,
    #     choices=['train', 'val', 'test', 'train_std', 'val_std', 'test_std'],
    #     help='Dataset split (train/val/test/train_std/val_std/test_std)'
    # )
    args = p.parse_args()
    mode = args.mode
    similarity_methods = args.similarity_methods
    run_benchmark = args.benchmark
    chunk_size = args.chunk_size
    resume = args.resume
    max_ligands = args.max_ligands
    write_batch_size = args.write_batch_size
    # set_type = args.set
    
    # output_dir = f"/scratch/yunmin/data/graph/train/example/{set_type}"
    # csv_path = f"/scratch/yunmin/data/graph/train/identity50/csvs/06_sample_example/{set_type}_set_sampled_20targets_100offtargets.csv"
    # csv_path = f"/scratch/yunmin/data/graph/train/example/{set_type}/{set_type}_set_sampled_after_dataprep.csv"
    # output_dir = "/scratch/yunmin/data/graph/train/example/eval_outputs/similarity"
    # csv_path = "/scratch/yunmin/data/graph/train/example/eval_outputs/similarity/bdb_df_ligand_features.csv"
    output_dir = "/scratch/yunmin/data/graph/train/example/eval_outputs/similarity"
    csv_path = "/scratch/yunmin/data/graph/train/example/eval_outputs/similarity/bdb_df_ligand_features.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check initial disk space
    print("\n💾 Checking disk space...")
    free_gb = check_disk_space(output_dir, min_gb=20)
    print(f"   Available: {free_gb:.1f} GB (recommended: >20 GB for large datasets)")

    total_df = pd.read_csv(csv_path)
    print("Loaded representative sequences dataset")
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Validate required columns
    required_cols = ['uniprot_id', 'sequence', 'ligand_smiles']
    missing_cols = [col for col in required_cols if col not in total_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")
    
    if len(total_df) == 0:
        raise ValueError("Input CSV is empty!")

    print("\n1️⃣ Computing molecular features and fingerprints...")

    # Load filtered dataset (>100 off-targets, >20 targets)
    print(f"\nProcessing dataset: {len(total_df):,} protein-ligand pairs")

    # Get unique SMILES - this is always needed
    unique_smiles = total_df['ligand_smiles'].unique()
    print(f"  Total unique ligands in dataset: {len(unique_smiles):,}")

    # Check for existing ligand_data checkpoint
    ligand_data_checkpoint = f"{output_dir}/{basename}_ligand_data.pkl"
    
    if resume and os.path.exists(ligand_data_checkpoint):
        print(f"\n🔄 Found checkpoint: {ligand_data_checkpoint}")
        print("   Loading pre-computed ligand data...")
        try:
            with open(ligand_data_checkpoint, 'rb') as f:
                ligand_data = pickle.load(f)
            print(f"✅ Loaded {len(ligand_data):,} pre-computed ligands from checkpoint")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print("   Will recompute from scratch...")
            ligand_data = None
    else:
        ligand_data = None
    
    # If no checkpoint or failed to load, compute features
    if ligand_data is None:
        # Check if features are already computed
        required_feature_cols = ['MW', 'cLogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 
                                  'Charge', 'NumRings', 'NumAromaticRings', 'scaffold']
        features_exist = all(col in total_df.columns for col in required_feature_cols)
    
        if features_exist:
            print("✅ Molecular features already exist in CSV, skipping feature computation!")
            print("   Computing fingerprints only (required for similarity calculation)...")
            
            ligand_data = {}
            print(f"  Processing {len(unique_smiles):,} unique ligands...")
            
            # Create a lookup dict for features from DataFrame
            feature_lookup = {}
            for _, row in total_df.drop_duplicates('ligand_smiles').iterrows():
                smiles = row['ligand_smiles']
                feature_lookup[smiles] = {
                    'MW': row['MW'],
                    'cLogP': row['cLogP'],
                    'TPSA': row['TPSA'],
                    'HBD': row['HBD'],
                    'HBA': row['HBA'],
                    'RotBonds': row['RotBonds'],
                    'Charge': row['Charge'],
                    'NumRings': row['NumRings'],
                    'NumAromaticRings': row['NumAromaticRings'],
                }
            
            for i, smiles in enumerate(unique_smiles):
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i+1:,}/{len(unique_smiles):,} ligands...")
                    # Save checkpoint every 1000 ligands
                    with open(ligand_data_checkpoint, 'wb') as f:
                        pickle.dump(ligand_data, f)
                
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    
                    # Get features from lookup
                    features = feature_lookup.get(smiles)
                    if features is None:
                        continue
                    
                    # Compute fingerprints (required for similarity)
                    gen = GetMorganGenerator(radius=2, fpSize=2048)
                    fp_ecfp4 = gen.GetFingerprint(mol)
                    fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
                    
                    # Get scaffold from DataFrame
                    scaffold_smiles = total_df[total_df['ligand_smiles'] == smiles]['scaffold'].iloc[0]
                    if pd.isna(scaffold_smiles):
                        scaffold_smiles = None
                    
                    entry = {}
                    if '2d' in similarity_methods:
                        entry['fp_ecfp4'] = fp_ecfp4
                    if 'scaffold' in similarity_methods:
                        entry['scaffold'] = scaffold_smiles
                    if 'physchem' in similarity_methods:
                        entry['features'] = features
                    ligand_data[smiles] = entry
                except Exception as e:
                    print(f"Error processing {smiles}: {e}")
                    continue
        else:
            print("Computing molecular features for all ligands (features not found in CSV)...")
            
            ligand_data = {}
            print(f"  Processing {len(unique_smiles):,} unique ligands...")

            for i, smiles in enumerate(unique_smiles):
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i+1:,}/{len(unique_smiles):,} ligands...")
                    # Save checkpoint every 1000 ligands
                    with open(ligand_data_checkpoint, 'wb') as f:
                        pickle.dump(ligand_data, f)
                
                mol_data = compute_mol_features(smiles)
                if mol_data:
                    entry = {}
                    if '2d' in similarity_methods:
                        entry['fp_ecfp4'] = mol_data['fp_ecfp4']
                    if 'scaffold' in similarity_methods:
                        entry['scaffold'] = mol_data['scaffold']
                    if 'physchem' in similarity_methods:
                        entry['features'] = mol_data['features']
                    ligand_data[smiles] = entry
        
        # Save final checkpoint
        print(f"\n💾 Saving ligand data checkpoint: {ligand_data_checkpoint}")
        with open(ligand_data_checkpoint, 'wb') as f:
            pickle.dump(ligand_data, f)

    # Report processing results
    n_processed = len(ligand_data)
    n_total = len(unique_smiles)
    n_failed = n_total - n_processed
    print(f"\n✅ Ligand processing complete:")
    print(f"   Processed: {n_processed:,}/{n_total:,} ligands ({n_processed/n_total*100:.1f}%)")
    if n_failed > 0:
        print(f"   Failed: {n_failed:,} ligands")
    
    if n_processed == 0:
        raise ValueError("No valid ligands were processed! Check your input data and SMILES format.")

    # Add features to dataframe
    total_df['mol_valid'] = total_df['ligand_smiles'].isin(ligand_data.keys())
    print(f"   Valid molecules in dataframe: {total_df['mol_valid'].sum():,}")
    
    if total_df['mol_valid'].sum() == 0:
        raise ValueError("No valid molecules found in dataframe after processing!")

    # Quick benchmark if requested
    if run_benchmark:
        print("\n⏱️ Running benchmark on 100 ligand pairs...")
        benchmark_ligands = list(ligand_data.keys())[:10]  # 10 ligands = 45 pairs
        
        times = {'2d': [], 'scaffold': [], 'physchem': []}
        
        for i, smiles1 in enumerate(benchmark_ligands):
            for j, smiles2 in enumerate(benchmark_ligands):
                if i >= j:
                    continue
                    
                data1 = ligand_data[smiles1]
                data2 = ligand_data[smiles2]
                
                # Benchmark Tanimoto
                if '2d' in similarity_methods:
                    t0 = time.perf_counter()
                    _ = compute_tanimoto_similarity(data1['fp_ecfp4'], data2['fp_ecfp4'])
                    times['2d'].append(time.perf_counter() - t0)
                
                # Benchmark Scaffold
                if 'scaffold' in similarity_methods:
                    t0 = time.perf_counter()
                    _ = compute_scaffold_similarity(data1['scaffold'], data2['scaffold'])
                    times['scaffold'].append(time.perf_counter() - t0)
                
                # Benchmark Physicochemical
                if 'physchem' in similarity_methods:
                    t0 = time.perf_counter()
                    _ = compute_physicochemical_similarity(data1['features'], data2['features'])
                    times['physchem'].append(time.perf_counter() - t0)
        
        print("\n📊 Benchmark results (average time per pair):")
        for method, time_list in times.items():
            if time_list:
                avg_time = np.mean(time_list) * 1000  # convert to ms
                print(f"  {method:12s}: {avg_time:.4f} ms")
        
        # Estimate total time
        total_pairs = total_df['mol_valid'].sum()
        if mode == 'all_pairwise':
            # Rough estimate: assume average ligands per protein
            unique_proteins_count = total_df[['uniprot_id', 'sequence']].drop_duplicates().shape[0]
            avg_ligands_per_protein = total_pairs / unique_proteins_count
            estimated_pairs = unique_proteins_count * (avg_ligands_per_protein * (avg_ligands_per_protein - 1) / 2)
        else:
            # target_vs_offtarget: harder to estimate, use conservative estimate
            estimated_pairs = total_pairs  # very rough estimate
        
        total_time_per_pair = sum(np.mean(times[m]) for m in times if times[m])
        estimated_total_time = estimated_pairs * total_time_per_pair
        
        print(f"\n⏳ Estimated total computation:")
        print(f"  Estimated pairs: {estimated_pairs:,.0f}")
        print(f"  Time per pair: {total_time_per_pair*1000:.4f} ms")
        print(f"  Total time: {estimated_total_time/60:.1f} minutes ({estimated_total_time/3600:.1f} hours)")
        
        proceed = input("\nContinue with full computation? (y/n): ")
        if proceed.lower() != 'y':
            print("Aborted.")
            return


    # Process each protein-target pair
    if mode == 'target_vs_offtarget':
        print("\n2️⃣ Computing similarities for each protein-target pair (target vs off-target mode)...")
    else:
        print("\n2️⃣ Computing similarities for each protein (all pairwise mode)...")

    protein_target_scores = []

    # Get unique proteins
    unique_proteins = total_df[
        ['uniprot_id', 'sequence',
          #'cluster_id'
          ]
    ].drop_duplicates()

    print(f"Processing {len(unique_proteins):,} unique proteins...")
    print(f"Chunk size: {chunk_size} proteins per batch")
    print(f"Total chunks: {int(np.ceil(len(unique_proteins) / chunk_size))}")

    # Temporary file for incremental results
    temp_output_file = f"{output_dir}/{basename}_{mode}_similarity_temp.csv"
    use_compression = mode == 'all_pairwise'  # Compress all_pairwise output
    
    if use_compression:
        temp_output_file += '.gz'
        print(f"   Using gzip compression for {mode} mode (saves ~90% disk space)")
    
    # Check if resuming from previous run
    processed_proteins = set()
    if resume and os.path.exists(temp_output_file):
        print(f"\n🔄 Found existing similarity temp file: {temp_output_file}")
        try:
            existing_df = pd.read_csv(temp_output_file)
            processed_proteins = set(existing_df['uniprot_id'].unique())
            print(f"   Already processed {len(processed_proteins)} proteins")
            print(f"   Continuing from where we left off...")
            chunk_num = 1
            header_written = True
        except Exception as e:
            print(f"⚠️  Failed to read temp file: {e}")
            print("   Starting fresh...")
            os.remove(temp_output_file)
            chunk_num = 0
            header_written = False
    else:
        if os.path.exists(temp_output_file):
            os.remove(temp_output_file)
        chunk_num = 0
        header_written = False

    for protein_idx, protein_row in tqdm(unique_proteins.iterrows(), 
                                        total=len(unique_proteins),
                                        desc="Processing proteins"):
        uniprot_id = protein_row['uniprot_id']
        sequence = protein_row['sequence']
        # cluster_id = protein_row['cluster_id']
        
        # Skip already processed proteins when resuming
        if resume and uniprot_id in processed_proteins:
            continue
        
        # Get all ligands for this protein
        protein_df = total_df[
            (total_df['uniprot_id'] == uniprot_id) &
            (total_df['sequence'] == sequence) &
            (total_df['mol_valid'] == True)
        ].copy()
        
        if mode == 'target_vs_offtarget':
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
                    
                    score_dict = {
                        'uniprot_id': uniprot_id,
                        'sequence': sequence,
                        # 'cluster_id': cluster_id,
                        'target_smiles': target_smiles,
                        'off_target_smiles': off_smiles,
                    }
                    
                    # 1. 2D Structural similarity (ECFP4 Tanimoto)
                    if '2d' in similarity_methods:
                        sim_2d = compute_tanimoto_similarity(
                            off_data['fp_ecfp4'], 
                            target_data['fp_ecfp4']
                        )
                        score_dict['similarity_2d'] = sim_2d
                    else:
                        sim_2d = 0
                    
                    # 2. Scaffold similarity (Murcko)
                    if 'scaffold' in similarity_methods:
                        sim_scaffold = compute_scaffold_similarity(
                            off_data['scaffold'],
                            target_data['scaffold']
                        )
                        score_dict['similarity_scaffold'] = sim_scaffold
                    else:
                        sim_scaffold = 0
                    
                    # 3. Physicochemical similarity
                    if 'physchem' in similarity_methods:
                        sim_physchem = compute_physicochemical_similarity(
                            off_data['features'],
                            target_data['features']
                        )
                        score_dict['similarity_physchem'] = sim_physchem
                    else:
                        sim_physchem = 0
                    
                    # Compute composite similarity score (weighted average of computed methods)
                    weights = {'2d': 0.4, 'scaffold': 0.3, 'physchem': 0.3}
                    computed_methods = [m for m in similarity_methods if m in weights]
                    if computed_methods:
                        total_weight = sum(weights[m] for m in computed_methods)
                        composite_score = sum(
                            weights[m] / total_weight * 
                            (sim_2d if m == '2d' else sim_scaffold if m == 'scaffold' else sim_physchem)
                            for m in computed_methods
                        )
                    else:
                        composite_score = 0
                    
                    score_dict['similarity_composite'] = composite_score
                    target_off_target_scores.append(score_dict)
                
                # Add to main list
                protein_target_scores.extend(target_off_target_scores)
            
            # Save checkpoint every chunk_size proteins to avoid memory issues
            if (protein_idx + 1) % chunk_size == 0 or protein_idx == len(unique_proteins) - 1:
                if protein_target_scores:
                    # Check disk space before writing
                    free_gb = check_disk_space(output_dir, min_gb=5)
                    
                    chunk_df = pd.DataFrame(protein_target_scores)
                    # Append to temp file
                    chunk_df.to_csv(
                        temp_output_file, 
                        mode='a', 
                        header=not header_written, 
                        index=False,
                        compression='gzip' if use_compression else 'infer'
                    )
                    header_written = True
                    chunk_num += 1
                    print(f"\n  💾 Saved chunk {chunk_num} ({len(protein_target_scores):,} pairs) | {free_gb:.1f} GB free")
                    # Clear memory
                    protein_target_scores = []
                    del chunk_df
        
        else:  # all_pairwise mode
            # Get all unique ligands for this protein (filter to those in ligand_data)
            all_ligands = [s for s in protein_df['ligand_smiles'].unique() if s in ligand_data]
            
            if len(all_ligands) < 2:
                continue
            
            # Subsample if too many ligands to avoid O(N^2) memory explosion
            n_ligands = len(all_ligands)
            if max_ligands > 0 and n_ligands > max_ligands:
                np.random.seed(42)  # reproducible
                all_ligands = list(np.random.choice(all_ligands, size=max_ligands, replace=False))
                n_ligands = len(all_ligands)
                print(f"\n  ⚠️  {uniprot_id}: subsampled {n_ligands} ligands (was {len(protein_df['ligand_smiles'].unique())})")
            
            total_pairs = n_ligands * (n_ligands - 1) // 2
            if total_pairs > 1000000:
                print(f"\n  ℹ️  {uniprot_id}: {n_ligands} ligands -> {total_pairs:,} pairs (streaming to disk)")
            
            # Use BulkTanimotoSimilarity for 2d-only mode (much faster)
            use_bulk_tanimoto = ('2d' in similarity_methods and 
                                 len(similarity_methods) == 1)
            
            # Stream results to disk in batches to avoid memory explosion
            batch_buffer = []
            pairs_written = 0
            
            if use_bulk_tanimoto:
                # Vectorized approach: for each ligand, compute similarity to all subsequent ligands at once
                fps = [ligand_data[s]['fp_ecfp4'] for s in all_ligands]
                
                for i in range(len(all_ligands)):
                    if i + 1 >= len(all_ligands):
                        break
                    
                    # BulkTanimotoSimilarity computes similarity of fps[i] vs list of fps
                    bulk_sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
                    
                    for k, sim_2d in enumerate(bulk_sims):
                        j = i + 1 + k
                        batch_buffer.append({
                            'uniprot_id': uniprot_id,
                            'sequence': sequence,
                            'ligand_smiles_1': all_ligands[i],
                            'ligand_smiles_2': all_ligands[j],
                            'similarity_2d': sim_2d,
                            'similarity_composite': sim_2d,
                        })
                        
                        # Flush batch to disk when buffer is full
                        if len(batch_buffer) >= write_batch_size:
                            check_disk_space(output_dir, min_gb=5)
                            chunk_df = pd.DataFrame(batch_buffer)
                            chunk_df.to_csv(
                                temp_output_file,
                                mode='a',
                                header=not header_written,
                                index=False
                            )
                            header_written = True
                            pairs_written += len(batch_buffer)
                            batch_buffer = []
                            del chunk_df
            else:
                # General approach for multiple similarity methods
                for i, smiles1 in enumerate(all_ligands):
                    data1 = ligand_data[smiles1]
                    
                    for j in range(i + 1, len(all_ligands)):
                        smiles2 = all_ligands[j]
                        data2 = ligand_data[smiles2]
                        
                        score_dict = {
                            'uniprot_id': uniprot_id,
                            'sequence': sequence,
                            'ligand_smiles_1': smiles1,
                            'ligand_smiles_2': smiles2,
                        }
                        
                        sim_2d = sim_scaffold = sim_physchem = 0
                        
                        if '2d' in similarity_methods:
                            sim_2d = compute_tanimoto_similarity(
                                data1['fp_ecfp4'], data2['fp_ecfp4']
                            )
                            score_dict['similarity_2d'] = sim_2d
                        
                        if 'scaffold' in similarity_methods:
                            sim_scaffold = compute_scaffold_similarity(
                                data1['scaffold'], data2['scaffold']
                            )
                            score_dict['similarity_scaffold'] = sim_scaffold
                        
                        if 'physchem' in similarity_methods:
                            sim_physchem = compute_physicochemical_similarity(
                                data1['features'], data2['features']
                            )
                            score_dict['similarity_physchem'] = sim_physchem
                        
                        weights = {'2d': 0.4, 'scaffold': 0.3, 'physchem': 0.3}
                        computed_methods = [m for m in similarity_methods if m in weights]
                        if computed_methods:
                            total_weight = sum(weights[m] for m in computed_methods)
                            composite_score = sum(
                                weights[m] / total_weight * 
                                (sim_2d if m == '2d' else sim_scaffold if m == 'scaffold' else sim_physchem)
                                for m in computed_methods
                            )
                        else:
                            composite_score = 0
                        
                        score_dict['similarity_composite'] = composite_score
                        batch_buffer.append(score_dict)
                        
                        # Flush batch to disk when buffer is full
                        if len(batch_buffer) >= write_batch_size:
                            check_disk_space(output_dir, min_gb=5)
                            chunk_df = pd.DataFrame(batch_buffer)
                            chunk_df.to_csv(
                                temp_output_file,
                                mode='a',
                                header=not header_written,
                                index=False,
                                compression='gzip' if use_compression else 'infer'
                            )
                            header_written = True
                            pairs_written += len(batch_buffer)
                            batch_buffer = []
                            del chunk_df
            
            # Flush remaining buffer for this protein
            if batch_buffer:
                check_disk_space(output_dir, min_gb=5)
                chunk_df = pd.DataFrame(batch_buffer)
                chunk_df.to_csv(
                    temp_output_file,
                    mode='a',
                    header=not header_written,
                    index=False,
                    compression='gzip' if use_compression else 'infer'
                )
                header_written = True
                pairs_written += len(batch_buffer)
                del chunk_df
                batch_buffer = []
            
            if total_pairs > 1000000:
                print(f"    ✅ Wrote {pairs_written:,} pairs for {uniprot_id}")
            
            # Explicit garbage collection after large proteins
            gc.collect()

    # Read all results from temp file
    print(f"\n📖 Reading results from temporary file...")
    
    if mode == 'all_pairwise':
        # For all_pairwise mode, the temp file may be very large.
        # Just rename it instead of reading into memory and rewriting.
        final_output = f"{output_dir}/{basename}_{mode}_similarity.csv"
        if use_compression:
            final_output += '.gz'
        
        # Count rows without loading entire file
        print(f"   Counting results (file-based, not loading into memory)...")
        import gzip
        open_func = gzip.open if use_compression else open
        with open_func(temp_output_file, 'rt') as f:
            total_rows = sum(1 for _ in f) - 1  # subtract header
        
        file_size_gb = os.path.getsize(temp_output_file) / (1024**3)
        print(f"\n✅ Computed similarities for {total_rows:,} ligand pairs ({file_size_gb:.1f} GB)")
        
        # Rename temp file to final output (no copying needed!)
        shutil.move(temp_output_file, final_output)
        print(f"💾 Renamed temp file to: {final_output}")
        
        # For all_pairwise mode, skip bucket assignment and merge
        print("\n3️⃣ Skipping difficulty bucket assignment (not applicable in all_pairwise mode)")
        print("\n4️⃣ Skipping merge with original dataset (all_pairwise mode outputs pairwise comparisons only)")
        print(f"\n✅ Done! Results saved to: {final_output}")
        return
    
    # For target_vs_offtarget mode, we need to read the file for bucket assignment
    print(f"   Loading similarity data for bucket assignment...")

    # Check file size before loading
    file_size_gb = os.path.getsize(temp_output_file) / (1024**3)
    if file_size_gb > 10:
        print(f"   ⚠️  Large file detected: {file_size_gb:.1f} GB")
        print(f"   This may take several minutes to load into memory...")
    
    similarity_df = pd.read_csv(temp_output_file)
    print(f"   ✅ Loaded {len(similarity_df):,} rows ({file_size_gb:.1f} GB)")
    os.remove(temp_output_file)  # Clean up temp file

    # Only assign buckets in target_vs_offtarget mode
    if mode == 'target_vs_offtarget':
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


    if mode == 'target_vs_offtarget':
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
        # reorder_col = [
        #     # protein and sequence information
        #     'protein_key', 'uniprot_id', 'target_name', 'target_source', 'sequence', 'sequence_length', 
        #     # ligand information
        #     'ligand_inchikey', 'target_type', 'target_smiles', 'off_target_smiles', 'ligand_name', 'ligand_het_id',  'difficulty_bucket',
        #     # path information
        #     'cluster_id', 'config_id', 'target_complex_id', 'target_blurred_path', 'offtarget_ligand_path', 'offtarget_pose_path', 'config_file',
        #     # experimental data (pdb structure and assay)
        #     'complex_pdb_id', 'valid_pdb_id', 'validation_status', 
        #     'representative_assay_nM', 'assay_type', 'original_assay_nM', 
        #     'curation_source', 'doi', 'data_source', 
        #     # similarity and bucket assignment
        #     'similarity_2d', 'similarity_scaffold', 'similarity_physchem', 'similarity_composite',
        #     # molecular features used to similarity calculation
        #     'MW', 'cLogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Charge', 'NumRings', 'NumAromaticRings', 'scaffold', 'mol_valid', 
        #     # SAIR columns
        #     'entry_id (SAIR)', 'index (SAIR)', 'pIC50 (SAIR)', 'family (SAIR)', 
        #     'description (SAIR)', 'vina_score_min (SAIR)', 'number_clashes (SAIR)', 'internal_energy (SAIR)', 
        #     'confidence_score (SAIR)', 'complex_plddt (SAIR)', 'complex_iplddt (SAIR)', 'ptm (SAIR)', 'iptm (SAIR)', 
        #     ]
        reorder_col = [
            # key columns
            'uniprot_id', 'ligand_inchikey', 'sequence', 'assay_type', 
            # protein and ligand information
            'protein_name', 'ligand_name', 'ligand_smiles', 
            # experimental data
            'complex_pdb_id', 'ligand_het_id', 'median_aff_nM', 'median_pX', 
            'mol_valid', 'MW', 'cLogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Charge', 'NumRings', 'NumAromaticRings', 'scaffold',
            # similarity and bucket assignment
            'similarity_2d', 'similarity_scaffold', 'similarity_physchem', 'similarity_composite', 'difficulty_bucket'
        ]
        total_df_bucketed = total_df_bucketed[reorder_col]

        # Fill NaN for targets in off_target_smiles
        mask_target = total_df_bucketed["target_type"] == "target"
        total_df_bucketed.loc[mask_target, "off_target_smiles"] = pd.NA

        # Fill missing target_smiles from target_smiles of off-target entries
        # keys = ["uniprot_id", "sequence", "cluster_id"]
        keys = ["uniprot_id", "sequence"]
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

    if mode == 'target_vs_offtarget':
        # Save bucketed dataset
        total_df_bucketed.to_csv(
            f"{output_dir}/{basename}_bucketed.csv",
            index=False
        )
        print(f"✅ Saved: {output_dir}/{basename}_bucketed.csv")

        # Save similarity statistics
        similarity_df_bucketed.to_csv(
            f"{output_dir}/{basename}_{mode}_similarity_stat.csv",
            index=False
        )
        print(f"✅ Saved: {output_dir}/{basename}_{mode}_similarity_stat.csv")
    else:
        print(f"✅ Similarity results already saved for {mode} mode")

if __name__ == "__main__":
    main()