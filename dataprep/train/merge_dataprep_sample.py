"""
Merge sampled CSV with stage6_output.csv to keep only pairs that remain after data preparation.
"""

import pandas as pd
import argparse
import os
from pathlib import Path


def merge_csvs(original_csv_path, stage6_csv_path, stage2_csv_path, output_csv_path):
    """
    Merge original sampled CSV with stage6_output.csv
    
    Args:
        original_csv_path: Path to original sampled CSV (e.g., test_set_sampled_20targets_100offtargets.csv)
        stage6_csv_path: Path to stage6_output.csv
        stage2_csv_path: Path to stage2_output.csv (needed for mapping)
        output_csv_path: Path to output merged CSV
    """
    print(f"Reading original CSV: {original_csv_path}")
    df_original = pd.read_csv(original_csv_path)
    print(f"  Original rows: {len(df_original)}")
    print(f"  Original columns: {list(df_original.columns)}")
    
    print(f"\nReading stage2 CSV: {stage2_csv_path}")
    df_stage2 = pd.read_csv(stage2_csv_path)
    print(f"  Stage2 rows: {len(df_stage2)}")
    print(f"  Stage2 columns: {list(df_stage2.columns)}")
    
    print(f"\nReading stage6 CSV: {stage6_csv_path}")
    df_stage6 = pd.read_csv(stage6_csv_path)
    print(f"  Stage6 rows: {len(df_stage6)}")
    print(f"  Stage6 columns: {list(df_stage6.columns)}")
    
    print(f"\nReading stage6 CSV: {stage6_csv_path}")
    df_stage6 = pd.read_csv(stage6_csv_path)
    print(f"  Stage6 rows: {len(df_stage6)}")
    print(f"  Stage6 columns: {list(df_stage6.columns)}")
    
    # Strategy:
    # 1. stage2 has all the original data with complex_id assigned
    # 2. stage6 has only the pairs that survived to stage6
    # 3. We match original CSV off-targets with stage2 by ligand_inchikey
    # 4. Then filter by stage6's offtarget_complex_ids
    # Note: protein_key format differs between original/stage2 (e.g., "O75173_Homo sapiens") 
    #       and stage6 (e.g., "O75173"), so we need to normalize
    
    # Normalize protein_key to just the uniprot ID (first part before underscore)
    df_stage6['protein_key_short'] = df_stage6['protein_key'].str.split('_').str[0]
    df_stage2['protein_key_short'] = df_stage2['protein_key'].str.split('_').str[0]
    df_original['protein_key_short'] = df_original['protein_key'].str.split('_').str[0]
    
    # ===== Process OFF-TARGET rows =====
    df_stage2_off = df_stage2[df_stage2['target_type'] == 'off_target'].copy()
    print(f"\nStage2 off-target rows: {len(df_stage2_off)}")
    
    df_original_off = df_original[df_original['target_type'] == 'off_target'].copy()
    print(f"Original off-target rows: {len(df_original_off)}")
    
    # Create a mapping from ligand_inchikey to complex_id for off-targets
    df_stage2_off['merge_key'] = df_stage2_off['protein_key_short'] + '_' + df_stage2_off['ligand_inchikey']
    inchikey_to_complex = df_stage2_off.set_index('merge_key')['complex_id'].to_dict()
    
    print(f"\nCreated off-target mapping for {len(inchikey_to_complex)} protein-ligand pairs")
    print("Sample off-target mappings:")
    for i, (k, v) in enumerate(inchikey_to_complex.items()):
        if i >= 3:
            break
        print(f"  {k[:60]}... -> {v}")
    
    # Add complex_id to original off-target rows
    df_original_off['merge_key'] = df_original_off['protein_key_short'] + '_' + df_original_off['ligand_inchikey']
    df_original_off['offtarget_complex_id'] = df_original_off['merge_key'].map(inchikey_to_complex)
    
    # Get the set of off-target complex_ids that made it to stage6
    stage6_offtarget_ids = set(df_stage6['offtarget_complex_id'].unique())
    print(f"\nStage6 has {len(stage6_offtarget_ids)} unique off-target complex_ids")
    
    # Filter original off-targets to only those in stage6
    df_filtered_off = df_original_off[df_original_off['offtarget_complex_id'].isin(stage6_offtarget_ids)].copy()
    print(f"Filtered down to {len(df_filtered_off)} off-target rows")
    
    # Merge off-targets with stage6 to add the additional columns
    df_merged_off = df_filtered_off.merge(
        df_stage6[['protein_key_short', 'offtarget_complex_id', 'config_id', 'target_complex_id', 
                   'target_blurred_path', 'offtarget_ligand_path', 'offtarget_pose_path', 'config_file']],
        on=['protein_key_short', 'offtarget_complex_id'],
        how='inner'
    )
    df_merged_off = df_merged_off.drop(columns=['merge_key', 'offtarget_complex_id', 'protein_key_short'])
    
    # ===== Process TARGET rows =====
    df_stage2_target = df_stage2[df_stage2['target_type'] == 'target'].copy()
    print(f"\nStage2 target rows: {len(df_stage2_target)}")
    
    df_original_target = df_original[df_original['target_type'] == 'target'].copy()
    print(f"Original target rows: {len(df_original_target)}")
    
    # Create a mapping from complex_id for targets
    complex_to_inchikey = df_stage2_target.set_index('complex_id')['ligand_inchikey'].to_dict()
    
    # Get the set of target complex_ids that made it to stage6
    stage6_target_ids = set(df_stage6['target_complex_id'].unique())
    print(f"\nStage6 has {len(stage6_target_ids)} unique target complex_ids")
    
    # Map target complex_ids to their inchikeys
    target_inchikeys = {complex_to_inchikey[cid] for cid in stage6_target_ids if cid in complex_to_inchikey}
    print(f"Mapped to {len(target_inchikeys)} unique target ligand inchikeys")
    
    # Filter original target rows by ligand_inchikey
    df_filtered_target = df_original_target[df_original_target['ligand_inchikey'].isin(target_inchikeys)].copy()
    print(f"Filtered down to {len(df_filtered_target)} target rows")
    
    # Add empty columns to target rows to match off-target structure
    # Target rows don't have config_id, offtarget_ligand_path, config_file
    df_filtered_target['config_id'] = None
    df_filtered_target['target_complex_id'] = df_filtered_target['ligand_inchikey'].map(
        {v: k for k, v in complex_to_inchikey.items()}
    )
    df_filtered_target['target_blurred_path'] = None
    df_filtered_target['offtarget_ligand_path'] = None
    df_filtered_target['offtarget_pose_path'] = None
    df_filtered_target['config_file'] = None
    df_filtered_target = df_filtered_target.drop(columns=['protein_key_short'])
    
    # ===== Combine target and off-target rows =====
    df_merged = pd.concat([df_filtered_target, df_merged_off], ignore_index=True)
    print(f"\nCombined result:")
    print(f"  Target rows: {len(df_filtered_target)}")
    print(f"  Off-target rows: {len(df_merged_off)}")
    print(f"  Total rows: {len(df_merged)}")
    
    print(f"\nMerged result:")
    print(f"  Rows after merge: {len(df_merged)}")
    print(f"  Columns: {df_merged.columns.tolist()}")
    
    # Save the result
    df_merged.to_csv(output_csv_path, index=False)
    print(f"\nSaved merged CSV to: {output_csv_path}")
    print(f"Final row count: {len(df_merged)}")
    
    return df_merged


def main():
    parser = argparse.ArgumentParser(
        description='Merge sampled CSV with stage6_output.csv'
    )
    parser.add_argument(
        '--set',
        type=str,
        required=True,
        choices=['train', 'val', 'test', 'train_std', 'val_std', 'test_std'],
        help='Dataset split (train/val/test/train_std/val_std/test_std)'
    )
    parser.add_argument(
        '--original-dir',
        type=str,
        default='/scratch/yunmin/data/graph/train/identity50/csvs/06_sample_example',
        help='Directory containing original sampled CSVs'
    )
    parser.add_argument(
        '--stage6-dir',
        type=str,
        default='/scratch/yunmin/data/graph/train/example',
        help='Directory containing stage6_output.csv files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as stage6-dir/{set})'
    )
    
    args = parser.parse_args()
    
    # Construct file paths
    if args.set.endswith('_std'):
        set_type_short = args.set.replace('_std', '')
    else:
        set_type_short = args.set
    original_csv = os.path.join(
        args.original_dir,
        f"{set_type_short}_set_sampled_20targets_100offtargets.csv"
    )
    stage2_csv = os.path.join(
        args.stage6_dir,
        args.set,
        "stage2_output.csv"
    )
    stage6_csv = os.path.join(
        args.stage6_dir,
        args.set,
        "stage6_output.csv"
    )
    
    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join(args.stage6_dir, args.set)
    else:
        output_dir = args.output_dir
    
    output_csv = os.path.join(
        output_dir,
        f"{args.set}_set_sampled_after_dataprep.csv"
    )
    
    # Check if input files exist
    if not os.path.exists(original_csv):
        raise FileNotFoundError(f"Original CSV not found: {original_csv}")
    if not os.path.exists(stage2_csv):
        raise FileNotFoundError(f"Stage2 CSV not found: {stage2_csv}")
    if not os.path.exists(stage6_csv):
        raise FileNotFoundError(f"Stage6 CSV not found: {stage6_csv}")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge CSVs
    merge_csvs(original_csv, stage6_csv, stage2_csv, output_csv)


if __name__ == '__main__':
    main()
