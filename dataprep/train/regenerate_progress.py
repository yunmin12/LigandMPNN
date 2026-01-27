#!/usr/bin/env python3
"""
Regenerate pipeline_progress.csv from status directory
"""
import pandas as pd
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def regenerate_progress(base_dir: Path, stage1_csv: Path = None):
    """Regenerate pipeline_progress.csv from status directory"""
    
    status_dir = base_dir / 'status'
    if not status_dir.exists():
        logger.error(f"Status directory not found: {status_dir}")
        return
    
    # Get metadata from stage1 CSV if available
    metadata_map = {}
    if stage1_csv and stage1_csv.exists():
        try:
            df = pd.read_csv(stage1_csv)
            if 'complex_id' in df.columns:
                for _, row in df.iterrows():
                    complex_id = row['complex_id']
                    metadata_map[complex_id] = {
                        'protein_key': row.get('protein_key', 'N/A'),
                        'target_het_id': row.get('ligand_het_id', 'N/A'),
                        'offtarget_het_id': row.get('offtarget_het_id', 'N/A'),
                    }
                logger.info(f"Loaded metadata for {len(metadata_map)} complexes from {stage1_csv}")
        except Exception as e:
            logger.warning(f"Could not read stage1 CSV: {e}")
    
    # Get all complex IDs from status directory
    complex_ids = []
    for item in status_dir.iterdir():
        if item.is_dir():
            complex_ids.append(item.name)
    
    complex_ids.sort()
    logger.info(f"Found {len(complex_ids)} complexes in status directory")
    
    # Define stages
    STAGES = [
        '1_save_ligands',
        '2_prep_target',
        '3_generate_config',
        '4_vina_prep',
        '5_vina_dock',
        '6_vina_post'
    ]
    
    progress_records = []
    
    for complex_id in complex_ids:
        # Get metadata if available
        if complex_id in metadata_map:
            meta = metadata_map[complex_id]
        else:
            meta = {
                'protein_key': 'N/A',
                'target_het_id': 'N/A',
                'offtarget_het_id': 'N/A',
            }
        
        record = {
            'complex_id': complex_id,
            'protein_key': meta['protein_key'],
            'target_het_id': meta['target_het_id'],
            'offtarget_het_id': meta['offtarget_het_id'],
        }
        
        complex_status_dir = status_dir / complex_id
        
        # Check status for each stage
        for stage in STAGES:
            status = None
            error = None
            
            if complex_status_dir.exists():
                # Check for status files
                for status_type in ['success', 'failed', 'skip']:
                    status_file = complex_status_dir / f"{stage}.{status_type}"
                    if status_file.exists():
                        status = status_type
                        
                        # Read error message if failed
                        if status_type == 'failed':
                            try:
                                content = status_file.read_text()
                                for line in content.split('\n'):
                                    if line.startswith('Error:'):
                                        error = line.replace('Error:', '').strip()
                                        break
                            except:
                                error = 'Unknown error'
                        break
            
            record[stage] = status or 'pending'
            
            # Add error column if failed
            if status == 'failed' and error:
                record[f'{stage}_error'] = error
        
        progress_records.append(record)
    
    # Create progress DataFrame
    progress_df = pd.DataFrame(progress_records)
    
    # Save progress CSV
    progress_csv = base_dir / 'pipeline_progress.csv'
    progress_df.to_csv(progress_csv, index=False)
    logger.info(f"✓ Saved progress report to {progress_csv}")
    
    # Print summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE PROGRESS SUMMARY")
    logger.info("=" * 70)
    
    for stage in STAGES:
        counts = progress_df[stage].value_counts().to_dict()
        total = len(progress_df)
        
        logger.info(f"\n{stage}:")
        logger.info(f"  ✓ Success: {counts.get('success', 0):4d} ({counts.get('success', 0)/total*100:5.1f}%)")
        logger.info(f"  ✗ Failed:  {counts.get('failed', 0):4d} ({counts.get('failed', 0)/total*100:5.1f}%)")
        logger.info(f"  ⊘ Skipped: {counts.get('skip', 0):4d} ({counts.get('skip', 0)/total*100:5.1f}%)")
        logger.info(f"  ⋯ Pending: {counts.get('pending', 0):4d} ({counts.get('pending', 0)/total*100:5.1f}%)")
    
    logger.info("\n" + "=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regenerate pipeline_progress.csv')
    parser.add_argument('--base_dir', required=True, help='Base directory with status folder')
    parser.add_argument('--stage1_csv', help='Optional path to stage1_output.csv for metadata')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    stage1_csv = Path(args.stage1_csv) if args.stage1_csv else base_dir / 'stage1_output.csv'
    
    # stage1_csv is optional now
    if not stage1_csv.exists():
        logger.warning(f"Stage1 CSV not found: {stage1_csv}, will proceed without metadata")
        stage1_csv = None
    
    regenerate_progress(base_dir, stage1_csv)
