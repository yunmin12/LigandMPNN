#!/usr/bin/env python3
"""
Generate individual config.yaml files for each complex in the CSV
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigGenerator:
    def __init__(self, csv_path: str, config_template_path: str, output_configs_dir: str, output_runs_dir: str):
        """
        Args:
            csv_path: Path to bindingdb_with_all_ligand_paths.csv
            config_template_path: Path to template config.yaml
            output_configs_dir: Where to save generated configs
            output_runs_dir: Where docking runs will output
        """
        self.csv_path = Path(csv_path)
        self.config_template_path = Path(config_template_path)
        self.output_configs_dir = Path(output_configs_dir)
        self.output_runs_dir = Path(output_runs_dir)
        
        self.output_configs_dir.mkdir(parents=True, exist_ok=True)
        self.output_runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Load template
        with open(self.config_template_path) as f:
            self.template = yaml.safe_load(f)

        self._offtarget_lookup: Dict[Tuple[str, str], str] = {}

    def generate_config(self, row: pd.Series, complex_id: str, off_path: str, off_het: str) -> Dict[str, Any]:
        """Generate config dict for one complex"""
        if not off_het or (isinstance(off_het, float) and pd.isna(off_het)):
            raise ValueError(f"Invalid off_het value: {off_het}")
        
        config = self.template.copy()
        
        # Target complex - ligand standardized to chain Z, resnum 1
        config['target_pdb'] = str(row['target_ligand_path'])
        config['target_ligand_chain'] = 'Z'  # Standardized
        config['target_ligand_resid'] = 1   # Standardized
        config['target_ligand_resname'] = str(row['ligand_het_id'])
        config['off_target_ligand_resname'] = str(off_het)

        
        # Off-target ligand from our downloaded files
        config['offtarget_ligand'] = str(off_path)
        
        config['output_dir'] = str(self.output_runs_dir / complex_id)
        
        # Metadata
        config['metadata'] = {
            'complex_id': complex_id,
            'pdb_id': str(row.get('valid_pdb_id', 'N/A')),
            'protein_key': str(row.get('protein_key', 'N/A')),
            'target_het_id': str(row['ligand_het_id']),
            'offtarget_het_id': str(off_het),
            'target_name': str(row.get('target_name', 'N/A')),
            'affinity_value': float(row.get('affinity_value', 0)) if pd.notna(row.get('affinity_value')) else None,
            'assay_type': str(row.get('assay_type', 'Kd'))
        }
        
        return config
    
    def process_csv(self, skip_failed: bool = True):
        """Generate all configs from CSV"""
        logger.info(f"Reading CSV from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        initial_total = len(df)
        
        # Check for required columns
        required_cols = [
            'protein_key', 
            'ligand_het_id', 'off_target_ids',
            'valid_pdb_id',
            'target_ligand_path', 'offtarget_ligand_path', 
            ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self._build_offtarget_lookup(df)

        df_targets = df[df['target_ligand_path'].notna()].copy()
        logger.info(f"Targets with target_ligand_path: {len(df_targets)}/{len(df)}")
        if len(df_targets) == 0:
            logger.error("No valid complexes remaining after filtering!")
            return
        logger.info(f"Generating configs for {len(df_targets)} complexes")
        
        summary_rows: List[Dict[str, Any]] = []

        generated_count = 0
        skipped_target_rows = 0
        skipped_pairs = 0
        
        for idx, row in df_targets.iterrows():
            pairs = self._resolve_all_offtargets_for_target_row(row)

            if not pairs:
                skipped_target_rows += 1
                msg = f"No matching off-targets for target idx={idx} (protein_key={row.get('protein_key')}, ligand_het_id={row.get('ligand_het_id')})"
                if not skip_failed:
                    logger.warning(msg)
                    continue
                raise ValueError(msg)

            for j, (off_path, off_het) in enumerate(pairs):
                complex_id = f"complex_{idx:04d}_off{j+1:02d}"
                try:
                    config = self.generate_config(row, complex_id, off_path, off_het)

                    config_file = self.output_configs_dir / f"{complex_id}.yaml"
                    with open(config_file, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    
                    summary_rows.append({
                        'config_id': complex_id,
                        'config_path': str(config_file),
                        'output_dir': config['output_dir'],
                        'protein_key': str(row.get('protein_key', 'N/A')),
                        'target_row_idx': int(idx),
                        'pdb_id': str(row.get('valid_pdb_id', 'N/A')),

                        'target_het_id': str(row.get('ligand_het_id', '')),
                        'target_ligand_path': str(row.get('target_ligand_path', '')),

                        'offtarget_het_id': str(off_het),
                        'offtarget_ligand_path': str(off_path),

                        'target_name': str(row.get('target_name', 'N/A')),
                        'assay_type': str(row.get('assay_type', 'Kd')),
                        'affinity_value': float(row.get('affinity_value', 0)) if pd.notna(row.get('affinity_value')) else None,
                    })

                    generated_count += 1
                    if generated_count % 100 == 0:
                        logger.info(f"Generated {generated_count} configs...")
                
                except Exception as e:
                    logger.error(f"Failed to generate config for {complex_id}: {e}")
                    skipped_pairs += 1
                    if not skip_failed:
                        raise e
        
        summary_df = pd.DataFrame(summary_rows)
        summary_csv_path = self.output_configs_dir / "summary_configs.csv"
        summary_df.to_csv(summary_csv_path, index=False)

        logger.info(f"\n=== Summary ===")
        logger.info(f"Generated: {generated_count} configs")
        logger.info(f"Skipped target rows: {skipped_target_rows} rows (no off-target match)")
        logger.info(f"Skipped pairs: {skipped_pairs} pairs (errors while writing)")
        logger.info(f"Configs saved to: {self.output_configs_dir}")
        logger.info(f"Summary CSV saved to: {summary_csv_path}")

    # off-target handling methods
    @staticmethod
    def _parse_offtarget_ids(raw: Any) -> List[str]:
        """Read off_target_ids column and split + strip by ;"""
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return []
        parts = [p.strip() for p in str(raw).split(';')]
        return [p for p in parts if p]
    
    def _build_offtarget_lookup(self, df: pd.DataFrame):
        """Generate lookup of (protein_key, ligand_het_id) -> offtarget_ligand_path"""
        lookup: Dict[Tuple[str, str], str] = {}
        for _, r in df.iterrows():
            pk = r.get('protein_key', None)
            het = r.get('ligand_het_id', None)
            path = r.get('offtarget_ligand_path', None)
            if pd.isna(pk) or pd.isna(het) or pd.isna(path):
                continue

            key = (str(pk), str(het).strip())
            if key not in lookup:
                lookup[key] = str(path)

        self._offtarget_lookup = lookup
        logger.info(f"Built offtarget lookup: {len(self._offtarget_lookup)} entries")

    def _resolve_all_offtargets_for_target_row(self, row: pd.Series) -> List[Tuple[str, str]]:
        """
        Read off_target_ids from target row, 
        return those existing in same protein_key: [(off_path, off_het), ...]
        """
        pk = row.get('protein_key', None)
        if pk is None or (isinstance(pk, float) and pd.isna(pk)):
            return []
        pk = str(pk)

        candidates = self._parse_offtarget_ids(row.get('off_target_ids', None))
        results: List[Tuple[str, str]] = []
        seen_hets = set()

        for het in candidates:
            if het in seen_hets:
                continue
            seen_hets.add(het)

            key = (pk, het)
            off_path = self._offtarget_lookup.get(key)
            if off_path and het:  # Ensure both off_path and het are valid
                results.append((off_path, het))

        return results
    
if __name__ == "__main__":
    # Paths
    # CSV_PATH = "/scratch/yunmin/data/graph/lmpnn/bdb_pdb/off_inputs/ligand_sdf/bindingdb_count_set3_with_ligand_paths_updated.csv"
    TEMPLATE_PATH = "/home/yunmin/proj/LigandMPNN/dataprep/off/config.yaml"
    # OUTPUT_CONFIGS_DIR = "/scratch/yunmin/data/graph/lmpnn/bdb_pdb/off_inputs/configs"
    # OUTPUT_RUNS_DIR = "/scratch/yunmin/data/graph/lmpnn/bdb_pdb/off_inputs/runs"

    placeholder = "test"
    CSV_PATH = f"/scratch/yunmin/data/graph/train/identity50/csvs/06_sample_example/{placeholder}_set_sampled_20targets_100offtargets.csv"
    TEMPLATE_PATH = "/home/yunmin/proj/LigandMPNN/dataprep/train/config.yaml"
    OUTPUT_CONFIGS_DIR = f"/scratch/yunmin/data/graph/train/example/{placeholder}/configs"
    OUTPUT_RUNS_DIR = f"/scratch/yunmin/data/graph/train/example/{placeholder}/runs"
    
    generator = ConfigGenerator(
        csv_path=CSV_PATH,
        config_template_path=TEMPLATE_PATH,
        output_configs_dir=OUTPUT_CONFIGS_DIR,
        output_runs_dir=OUTPUT_RUNS_DIR
    )
    
    generator.process_csv(skip_failed=True)
