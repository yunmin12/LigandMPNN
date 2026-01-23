"""
Stage 3: Generate config files
Based on generate_configs.py

Strategy:
- Group by protein_key
- For each target in the protein:
  - Match with all off-targets for that protein
  - Generate one config per target-offtarget pair
"""
import os
import sys
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from utils.status_tracker import StatusTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigGenerator:
    def __init__(self, config_template_path: str, output_configs_dir: str, 
                 output_runs_dir: str, status_tracker: StatusTracker):
        self.config_template_path = Path(config_template_path)
        self.output_configs_dir = Path(output_configs_dir)
        self.output_runs_dir = Path(output_runs_dir)
        self.status_tracker = status_tracker
        
        self.output_configs_dir.mkdir(parents=True, exist_ok=True)
        self.output_runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Load template
        with open(self.config_template_path) as f:
            self.template = yaml.safe_load(f)
    
    def generate_config(self, target_row: pd.Series, offtarget_row: pd.Series, 
                       config_id: str) -> Optional[Dict[str, Any]]:
        """
        Generate config dict for one target-offtarget pair
        
        Args:
            target_row: Row with target_type == 'target'
            offtarget_row: Row with target_type == 'off_target'
            config_id: Unique ID for this config (e.g., "protein_001_target_A_off_B")
        """
        config = self.template.copy()
        
        # Target complex - ligand standardized to chain Z, resnum 1
        config['target_pdb'] = str(target_row['target_blurred_path'])
        config['target_ligand_chain'] = 'Z'
        config['target_ligand_resid'] = 1
        config['target_ligand_resname'] = str(target_row['ligand_het_id'])
        
        # Off-target ligand
        config['offtarget_ligand'] = str(offtarget_row['ligand_path'])
        # config['offtarget_ligand_resname'] = str(offtarget_row['ligand_het_id'])
        
        # Output directory
        config['output_dir'] = str(self.output_runs_dir / config_id)
        
        # Metadata
        config['metadata'] = {
            'config_id': config_id,
            'protein_key': str(target_row.get('protein_key', 'N/A')),
            'target_pdb_id': str(target_row.get('valid_pdb_id', 'N/A')),
            'target_complex_id': str(target_row.get('complex_id', 'N/A')),
            'target_het_id': str(target_row['ligand_het_id']),
            'offtarget_complex_id': str(offtarget_row.get('complex_id', 'N/A')),
            # 'offtarget_het_id': str(offtarget_row['ligand_het_id']),
            'target_name': str(target_row.get('target_name', 'N/A')),
        }
        
        return config
    
    def process_protein_group(self, protein_key: str, group_df: pd.DataFrame) -> Tuple[int, int]:
        """
        Process one protein group: generate configs for all target-offtarget pairs
        
        Returns:
            (success_count, failed_count)
        """
        # Separate target and off-target rows
        target_rows = group_df[group_df['target_type'] == 'target']
        offtarget_rows = group_df[group_df['target_type'] == 'off_target']
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing protein: {protein_key}")
        logger.info(f"  Targets: {len(target_rows)}")
        logger.info(f"  Off-targets: {len(offtarget_rows)}")
        logger.info(f"  Total configs to generate: {len(target_rows) * len(offtarget_rows)}")
        logger.info(f"{'='*70}")
        
        if len(target_rows) == 0:
            logger.warning(f"No target rows for protein {protein_key}")
            return 0, 0
        
        if len(offtarget_rows) == 0:
            logger.warning(f"No off-target rows for protein {protein_key}")
            return 0, 0
        
        success_count = 0
        failed_count = 0
        
        # Generate config for each target-offtarget pair
        for target_idx, target_row in target_rows.iterrows():
            target_complex_id = target_row.get('complex_id', f'target_{target_idx}')
            
            # Check if target has required fields
            if pd.isna(target_row.get('target_blurred_path')):
                logger.warning(f"Target {target_complex_id}: Missing target_blurred_path, skipping")
                failed_count += len(offtarget_rows)
                continue
            
            if pd.isna(target_row.get('ligand_het_id')):
                logger.warning(f"Target {target_complex_id}: Missing ligand_het_id, skipping")
                failed_count += len(offtarget_rows)
                continue
            
            for offtarget_idx, offtarget_row in offtarget_rows.iterrows():
                offtarget_complex_id = offtarget_row.get('complex_id', f'offtarget_{offtarget_idx}')
                
                # Generate unique config ID
                config_id = f"{protein_key}_{target_complex_id[8:]}_{offtarget_complex_id[8:]}"
                
                try:
                    # Check required fields for off-target
                    if pd.isna(offtarget_row.get('ligand_path')):
                        raise ValueError(f"Missing ligand_path")
                    
                    # Generate config
                    config = self.generate_config(target_row, offtarget_row, config_id)
                    if not config:
                        raise ValueError("Failed to generate config")
                    
                    # Save config
                    config_file = self.output_configs_dir / f"{config_id}.yaml"
                    with open(config_file, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    
                    logger.debug(f"Generated config: {config_file.name}")
                    
                    # Mark success
                    self.status_tracker.mark_status(config_id, '3_generate_config', 'success')
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"{config_id}: Failed - {e}")
                    self.status_tracker.mark_status(config_id, '3_generate_config', 'failed', str(e))
                    failed_count += 1
        
        logger.info(f"Protein {protein_key}: {success_count} configs generated, {failed_count} failed")
        return success_count, failed_count
    
    def process_all(self, df: pd.DataFrame) -> Tuple[int, int, List[Dict]]:
        """
        Process all proteins in DataFrame
        
        Returns:
            (total_success, total_failed, config_metadata_list)
        """
        # Check for protein_key column
        if 'uniprot_id' not in df.columns or 'sequence' not in df.columns:
            logger.error("Missing 'uniprot_id' or 'sequence' column in DataFrame")
            raise ValueError("DataFrame must have 'uniprot_id' and 'sequence' columns for grouping")
        
        total_success = 0
        total_failed = 0
        config_metadata_list = []
        
        # Group by protein_key
        grouped = df.groupby(['uniprot_id', 'sequence'])
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Found {len(grouped)} unique proteins")
        logger.info(f"{'='*70}\n")
        
        for protein_key, group_df in grouped:
            protein_key = protein_key[0]  # Use uniprot_id as protein_key
            success, failed = self.process_protein_group(protein_key, group_df)
            total_success += success
            total_failed += failed
            
            # Collect metadata for output CSV (optional)
            for target_row in group_df[group_df['target_type'] == 'target'].iterrows():
                for offtarget_row in group_df[group_df['target_type'] == 'off_target'].iterrows():
                    target_idx, target_data = target_row
                    offtarget_idx, offtarget_data = offtarget_row
                    
                    config_id = f"{protein_key}_{target_data.get('complex_id')[8:]}_{offtarget_data.get('complex_id')[8:]}"
                    
                    config_metadata_list.append({
                        'config_id': config_id,
                        'protein_key': protein_key,
                        'target_complex_id': target_data.get('complex_id'),
                        'target_pdb_id': target_data.get('valid_pdb_id'),
                        'target_het_id': target_data.get('ligand_het_id'),
                        'target_blurred_path': target_data.get('target_blurred_path'),
                        'offtarget_complex_id': offtarget_data.get('complex_id'),
                        # 'offtarget_het_id': offtarget_data.get('ligand_het_id'),
                        'ligand_path': offtarget_data.get('ligand_path'),
                        'config_file': str(self.output_configs_dir / f"{config_id}.yaml"),
                    })
        
        return total_success, total_failed, config_metadata_list


def main():
    import argparse

    p = argparse.ArgumentParser(description="Stage 3: Generate Vina config files")
    p.add_argument("--csv", required=True, help="Input CSV from Stage 2")
    p.add_argument("--template", required=True, help="Config template YAML")
    p.add_argument("--output_configs_dir", required=True, help="Output configs directory")
    p.add_argument("--output_runs_dir", required=True, help="Output runs directory")
    p.add_argument("--status_dir", required=True, help="Status tracking directory")
    args = p.parse_args()
    
    # Check if input CSV exists and has data
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"Input CSV not found: {csv_path}")
        logger.error("Stage 2 may have failed. Check stage2_output.csv")
        sys.exit(1)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        logger.error(f"Input CSV is empty: {csv_path}")
        logger.error("All Stage 2 entries may have failed")
        sys.exit(1)
    
    logger.info(f"Loaded {len(df)} entries from {csv_path}")
    
    # Print input statistics
    target_count = (df['target_type'] == 'target').sum()
    offtarget_count = (df['target_type'] == 'off_target').sum()
    
    print(f"\n{'='*70}")
    print(f"=== STAGE 3 INPUT ANALYSIS ===")
    print(f"{'='*70}")
    print(f"Total entries:     {len(df)}")
    print(f"  Target:          {target_count}")
    print(f"  Off-target:      {offtarget_count}")
    
    if 'protein_key' in df.columns:
        n_proteins = df['protein_key'].nunique()
        print(f"Unique proteins:   {n_proteins}")
        print(f"Expected configs:  {target_count * offtarget_count // n_proteins if n_proteins > 0 else 0} (approx)")
    print(f"{'='*70}\n")
    
    # Check required columns
    required_cols = ['complex_id', 'target_type', 'protein_key']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in CSV: {missing_cols}")
        logger.error(f"Available columns: {list(df.columns)}")
        sys.exit(1)
 
    # Initialize
    status_tracker = StatusTracker(Path(args.status_dir))
    generator = ConfigGenerator(
        args.template,
        args.output_configs_dir,
        args.output_runs_dir,
        status_tracker
    )
    
    # Process all proteins
    try:
        success_count, failed_count, config_metadata = generator.process_all(df)
    except Exception as e:
        logger.error(f"Fatal error during processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Save config metadata CSV
    output_csv = Path(args.output_configs_dir).parent / "stage3_output.csv"
    if config_metadata:
        pd.DataFrame(config_metadata).to_csv(output_csv, index=False)
        logger.info(f"Saved {len(config_metadata)} config metadata entries to {output_csv}")
    else:
        logger.warning("No configs generated, creating empty CSV")
        pd.DataFrame(columns=['config_id', 'protein_key']).to_csv(output_csv, index=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"=== STAGE 3 SUMMARY ===")
    print(f"{'='*70}")
    print(f"Configs generated:  {success_count}")
    print(f"Failed:             {failed_count}")
    print(f"Success rate:       {100*success_count/(success_count+failed_count) if (success_count+failed_count) > 0 else 0:.1f}%")
    print(f"Output CSV:         {output_csv}")
    print(f"Config files:       {args.output_configs_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
