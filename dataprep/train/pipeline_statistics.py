#!/usr/bin/env python3
"""
Generate comprehensive statistics from pipeline execution
"""
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def analyze_stage_outputs(base_dir: Path):
    """Analyze all stage output CSV files"""
    
    stats = {
        'stage_counts': {},
        'failure_reasons': {},
        'final_dataset': {}
    }
    
    # Define stages
    stage_files = [
        'stage1_output.csv',
        'stage2_output.csv', 
        'stage3_output.csv',
        'stage4_output.csv',
        'stage5_output.csv',
        'stage6_output.csv'
    ]
    
    logger.info("\n" + "=" * 70)
    logger.info("STAGE-BY-STAGE ANALYSIS")
    logger.info("=" * 70)
    
    prev_count = None
    
    for i, stage_file in enumerate(stage_files, 1):
        stage_path = base_dir / stage_file
        
        if not stage_path.exists():
            logger.warning(f"Stage {i} output not found: {stage_path}")
            continue
        
        try:
            df = pd.read_csv(stage_path)
            count = len(df)
            
            stats['stage_counts'][f'stage{i}'] = count
            
            if prev_count is not None:
                lost = prev_count - count
                survival_rate = (count / prev_count * 100) if prev_count > 0 else 0
                logger.info(f"\nStage {i} ({stage_file}):")
                logger.info(f"  Output: {count:5d} complexes")
                logger.info(f"  Lost:   {lost:5d} complexes ({100-survival_rate:.1f}%)")
                logger.info(f"  Rate:   {survival_rate:5.1f}% survival")
            else:
                logger.info(f"\nStage {i} ({stage_file}):")
                logger.info(f"  Initial: {count:5d} complexes")
            
            prev_count = count
            
        except Exception as e:
            logger.error(f"Error reading {stage_file}: {e}")
    
    return stats


def analyze_failures(base_dir: Path):
    """Analyze failure reasons from status directory"""
    
    status_dir = base_dir / 'status'
    if not status_dir.exists():
        logger.warning(f"Status directory not found: {status_dir}")
        return {}
    
    STAGES = [
        '1_save_ligands',
        '2_prep_target',
        '3_generate_config',
        '4_vina_prep',
        '5_vina_dock',
        '6_vina_post'
    ]
    
    failure_stats = defaultdict(lambda: defaultdict(int))
    failure_examples = defaultdict(lambda: defaultdict(list))
    
    logger.info("\n" + "=" * 70)
    logger.info("FAILURE ANALYSIS")
    logger.info("=" * 70)
    
    for complex_dir in status_dir.iterdir():
        if not complex_dir.is_dir():
            continue
        
        complex_id = complex_dir.name
        
        for stage in STAGES:
            failed_file = complex_dir / f"{stage}.failed"
            
            if failed_file.exists():
                # Read error message
                try:
                    content = failed_file.read_text()
                    error_msg = "Unknown error"
                    
                    for line in content.split('\n'):
                        if line.startswith('Error:'):
                            error_msg = line.replace('Error:', '').strip()
                            break
                    
                    # Categorize error
                    error_category = categorize_error(error_msg)
                    failure_stats[stage][error_category] += 1
                    
                    # Keep examples (max 3 per category)
                    if len(failure_examples[stage][error_category]) < 3:
                        failure_examples[stage][error_category].append({
                            'complex_id': complex_id,
                            'error': error_msg[:100]  # Truncate long errors
                        })
                    
                except Exception as e:
                    logger.warning(f"Could not read {failed_file}: {e}")
    
    # Print failure analysis
    for stage in STAGES:
        if stage in failure_stats:
            stage_failures = failure_stats[stage]
            total_failures = sum(stage_failures.values())
            
            logger.info(f"\n{stage}: {total_failures} total failures")
            
            # Sort by frequency
            sorted_failures = sorted(stage_failures.items(), key=lambda x: x[1], reverse=True)
            
            for error_cat, count in sorted_failures[:5]:  # Top 5 error types
                pct = (count / total_failures * 100) if total_failures > 0 else 0
                logger.info(f"  • {error_cat}: {count} ({pct:.1f}%)")
                
                # Show examples
                if stage in failure_examples and error_cat in failure_examples[stage]:
                    for ex in failure_examples[stage][error_cat][:2]:  # Show 2 examples
                        logger.info(f"    - {ex['complex_id']}: {ex['error']}")
    
    return dict(failure_stats)


def categorize_error(error_msg: str) -> str:
    """Categorize error message into common types"""
    error_lower = error_msg.lower()
    
    # Common error patterns
    if 'file not found' in error_lower or 'does not exist' in error_lower:
        return 'File Not Found'
    elif 'timeout' in error_lower:
        return 'Timeout'
    elif 'vina' in error_lower and 'failed' in error_lower:
        return 'Vina Execution Failed'
    elif 'empty' in error_lower or 'no poses' in error_lower:
        return 'No Poses Generated'
    elif 'pdbqt' in error_lower:
        return 'PDBQT Conversion Error'
    elif 'parsing' in error_lower or 'parse' in error_lower:
        return 'Parsing Error'
    elif 'permission' in error_lower:
        return 'Permission Error'
    elif 'memory' in error_lower:
        return 'Memory Error'
    else:
        return 'Other Error'


def analyze_final_dataset(base_dir: Path):
    """Analyze final dataset statistics"""
    
    # Use stage6 as final output
    final_csv = base_dir / 'stage6_output.csv'
    
    if not final_csv.exists():
        logger.warning(f"Final output not found: {final_csv}")
        return {}
    
    try:
        df = pd.read_csv(final_csv)
        
        logger.info("\n" + "=" * 70)
        logger.info("FINAL DATASET STATISTICS")
        logger.info("=" * 70)
        
        # Total complexes
        total_complexes = len(df)
        logger.info(f"\nTotal complexes: {total_complexes}")
        
        # Unique proteins
        unique_proteins = 0
        if 'protein_key' in df.columns:
            unique_proteins = df['protein_key'].nunique()
            logger.info(f"Unique proteins: {unique_proteins}")
            
            # Top proteins
            top_proteins = df['protein_key'].value_counts().head(10)
            logger.info("\nTop 10 proteins by config count:")
            for protein, count in top_proteins.items():
                logger.info(f"  {protein}: {count}")
        
        # Count unique target and off-target complexes
        unique_targets = 0
        unique_offtargets = 0
        
        if 'target_complex_id' in df.columns:
            unique_targets = df['target_complex_id'].nunique()
            logger.info(f"\nUnique target complexes: {unique_targets}")
        
        if 'offtarget_complex_id' in df.columns:
            unique_offtargets = df['offtarget_complex_id'].nunique()
            logger.info(f"Unique off-target complexes: {unique_offtargets}")
        
        # Count unique target ligands
        if 'target_het_id' in df.columns:
            unique_target_ligands = df['target_het_id'].nunique()
            logger.info(f"Unique target ligands: {unique_target_ligands}")
        
        # Count poses (from runs directory - look in final/ subdirectories)
        runs_dir = base_dir / 'runs'
        total_poses = 0
        configs_with_poses = 0
        
        if runs_dir.exists():
            for config_dir in runs_dir.iterdir():
                if config_dir.is_dir():
                    # Count PDB files in final/ subdirectory
                    final_dir = config_dir / 'final'
                    if final_dir.exists():
                        pdb_files = list(final_dir.glob('*.pdb'))
                        if len(pdb_files) > 0:
                            configs_with_poses += 1
                            total_poses += len(pdb_files)
        
        logger.info(f"\nPose generation:")
        logger.info(f"  Configs with poses: {configs_with_poses} / {total_complexes}")
        logger.info(f"  Total poses: {total_poses}")
        
        if configs_with_poses > 0:
            avg_poses = total_poses / configs_with_poses
            logger.info(f"  Average poses per config: {avg_poses:.1f}")
        
        stats = {
            'total_complexes': total_complexes,
            'unique_proteins': unique_proteins,
            'unique_targets': unique_targets,
            'unique_offtargets': unique_offtargets,
            'configs_with_poses': configs_with_poses,
            'total_poses': total_poses,
            'avg_poses_per_config': total_poses / configs_with_poses if configs_with_poses > 0 else 0
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error analyzing final dataset: {e}")
        return {}


def save_statistics_json(base_dir: Path, stats: dict):
    """Save statistics to JSON file"""
    
    stats_file = base_dir / 'pipeline_statistics.json'
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\n✓ Statistics saved to {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate pipeline statistics')
    parser.add_argument('--base_dir', required=True, help='Base directory with pipeline outputs')
    
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    
    if not base_dir.exists():
        logger.error(f"Base directory not found: {base_dir}")
        return
    
    # Collect all statistics
    all_stats = {}
    
    # 1. Stage-by-stage analysis
    stage_stats = analyze_stage_outputs(base_dir)
    all_stats.update(stage_stats)
    
    # 2. Failure analysis
    failure_stats = analyze_failures(base_dir)
    all_stats['failure_reasons'] = failure_stats
    
    # 3. Final dataset analysis
    final_stats = analyze_final_dataset(base_dir)
    all_stats['final_dataset'] = final_stats
    
    # 4. Save to JSON
    save_statistics_json(base_dir, all_stats)
    
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
