#!/usr/bin/env python3
"""
Generate a human-readable summary report from pipeline execution
"""
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
import json


def generate_report(base_dir: Path):
    """Generate comprehensive summary report"""
    
    # Read pipeline progress
    progress_csv = base_dir / 'pipeline_progress.csv'
    stats_json = base_dir / 'pipeline_statistics.json'
    
    if not progress_csv.exists():
        print(f"Error: {progress_csv} not found. Run regenerate_progress.py first.")
        return
    
    if not stats_json.exists():
        print(f"Error: {stats_json} not found. Run pipeline_statistics.py first.")
        return
    
    # Load data
    progress_df = pd.read_csv(progress_csv)
    with open(stats_json) as f:
        stats = json.load(f)
    
    # Generate report
    report_path = base_dir / 'PIPELINE_SUMMARY_REPORT.txt'
    
    with open(report_path, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("LIGANDMPNN TRAINING DATA PIPELINE - EXECUTION SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base Directory: {base_dir}\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall Pipeline Flow
        f.write("1. OVERALL PIPELINE FLOW\n")
        f.write("-" * 80 + "\n")
        
        stages = ['stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6']
        stage_names = [
            'Save Ligands',
            'Prepare Target',
            'Generate Config',
            'Vina Prep',
            'Vina Dock',
            'Vina Post'
        ]
        
        initial = stats['stage_counts'].get('stage1', 0)
        final = stats['stage_counts'].get('stage6', 0)
        
        f.write(f"Initial complexes: {initial:>6,}\n")
        
        for i, (stage, name) in enumerate(zip(stages, stage_names)):
            count = stats['stage_counts'].get(stage, 0)
            if i > 0:
                prev_count = stats['stage_counts'].get(stages[i-1], 0)
                lost = prev_count - count
                survival = (count / prev_count * 100) if prev_count > 0 else 0
                f.write(f"  └─> Stage {i+1}: {name:<20} {count:>6,} complexes "
                       f"(-{lost:>5,}, {survival:>5.1f}% survival)\n")
            else:
                f.write(f"  └─> Stage {i+1}: {name:<20} {count:>6,} complexes\n")
        
        overall_survival = (final / initial * 100) if initial > 0 else 0
        f.write(f"\nFinal complexes:   {final:>6,}\n")
        f.write(f"Overall survival:  {overall_survival:>6.1f}%\n")
        f.write(f"Total loss:        {initial - final:>6,}\n\n")
        
        # Stage-by-Stage Status
        f.write("2. STAGE-BY-STAGE STATUS\n")
        f.write("-" * 80 + "\n")
        
        stage_cols = [
            '1_save_ligands',
            '2_prep_target',
            '3_generate_config',
            '4_vina_prep',
            '5_vina_dock',
            '6_vina_post'
        ]
        
        total = len(progress_df)
        
        for stage_col, name in zip(stage_cols, stage_names):
            if stage_col in progress_df.columns:
                counts = progress_df[stage_col].value_counts()
                
                f.write(f"\n{name} ({stage_col}):\n")
                f.write(f"  ✓ Success: {counts.get('success', 0):>6,} ({counts.get('success', 0)/total*100:>5.1f}%)\n")
                f.write(f"  ✗ Failed:  {counts.get('failed', 0):>6,} ({counts.get('failed', 0)/total*100:>5.1f}%)\n")
                f.write(f"  ⊘ Skipped: {counts.get('skip', 0):>6,} ({counts.get('skip', 0)/total*100:>5.1f}%)\n")
                f.write(f"  ⋯ Pending: {counts.get('pending', 0):>6,} ({counts.get('pending', 0)/total*100:>5.1f}%)\n")
        
        # Failure Analysis
        f.write("\n\n3. FAILURE ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
        if 'failure_reasons' in stats:
            for stage, failures in stats['failure_reasons'].items():
                total_failures = sum(failures.values())
                f.write(f"\n{stage}: {total_failures:,} total failures\n")
                
                sorted_failures = sorted(failures.items(), key=lambda x: x[1], reverse=True)
                for error_type, count in sorted_failures:
                    pct = (count / total_failures * 100) if total_failures > 0 else 0
                    f.write(f"  • {error_type:<30} {count:>6,} ({pct:>5.1f}%)\n")
        
        # Final Dataset Statistics
        f.write("\n\n4. FINAL DATASET STATISTICS\n")
        f.write("-" * 80 + "\n")
        
        final_stats = stats.get('final_dataset', {})
        
        f.write(f"\nDataset Composition:\n")
        f.write(f"  Total configs:           {final_stats.get('total_complexes', 0):>6,}\n")
        f.write(f"  Unique proteins:         {final_stats.get('unique_proteins', 0):>6,}\n")
        f.write(f"  Unique target complexes: {final_stats.get('unique_targets', 0):>6,}\n")
        f.write(f"  Unique off-targets:      {final_stats.get('unique_offtargets', 0):>6,}\n")
        
        f.write(f"\nPose Generation:\n")
        f.write(f"  Configs with poses:      {final_stats.get('configs_with_poses', 0):>6,}\n")
        f.write(f"  Total poses generated:   {final_stats.get('total_poses', 0):>6,}\n")
        f.write(f"  Avg poses per config:    {final_stats.get('avg_poses_per_config', 0):>6.1f}\n")
        
        # Top Proteins
        f.write("\n\n5. TOP PROTEINS BY CONFIG COUNT\n")
        f.write("-" * 80 + "\n\n")
        
        # Read stage6 for protein distribution
        stage6_csv = base_dir / 'stage6_output.csv'
        if stage6_csv.exists():
            df6 = pd.read_csv(stage6_csv)
            if 'protein_key' in df6.columns:
                protein_counts = df6['protein_key'].value_counts().head(15)
                
                for i, (protein, count) in enumerate(protein_counts.items(), 1):
                    pct = (count / len(df6) * 100) if len(df6) > 0 else 0
                    f.write(f"  {i:>2}. {protein:<20} {count:>4} configs ({pct:>5.1f}%)\n")
        
        # Recommendations
        f.write("\n\n6. RECOMMENDATIONS & NEXT STEPS\n")
        f.write("-" * 80 + "\n\n")
        
        # Calculate some metrics for recommendations
        stage1_success = progress_df['1_save_ligands'].value_counts().get('success', 0)
        stage1_failed = progress_df['1_save_ligands'].value_counts().get('failed', 0)
        stage4_failed = progress_df['4_vina_prep'].value_counts().get('failed', 0)
        
        if stage1_failed > stage1_success * 0.5:
            f.write("⚠ High failure rate in Stage 1 (Save Ligands):\n")
            f.write("   - Review ligand source availability (PDB/PubChem)\n")
            f.write("   - Check network connectivity for downloads\n\n")
        
        if stage4_failed > 100:
            f.write("⚠ Significant failures in Stage 4 (Vina Prep):\n")
            f.write("   - Review PDBQT conversion process\n")
            f.write("   - Check for missing off-target ligand paths\n\n")
        
        if final_stats.get('configs_with_poses', 0) == final_stats.get('total_complexes', 0):
            f.write("✓ All final configs have generated poses - Pipeline successful!\n\n")
        
        f.write("Next Steps:\n")
        f.write("  1. Review pipeline_progress.csv for detailed complex-level status\n")
        f.write("  2. Investigate failed cases using status directory logs\n")
        f.write("  3. Re-run failed stages if needed with updated configurations\n")
        f.write("  4. Proceed to model training with final dataset\n")
        
        # Footer
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n✓ Summary report saved to: {report_path}")
    print(f"\nQuick Summary:")
    print(f"  Initial: {initial:,} → Final: {final:,} ({overall_survival:.1f}% survival)")
    print(f"  Total poses: {final_stats.get('total_poses', 0):,}")
    print(f"  Unique proteins: {final_stats.get('unique_proteins', 0)}")


def main():
    parser = argparse.ArgumentParser(description='Generate summary report')
    parser.add_argument('--base_dir', required=True, help='Base directory with pipeline outputs')
    
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return
    
    generate_report(base_dir)


if __name__ == '__main__':
    main()
