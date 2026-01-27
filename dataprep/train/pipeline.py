"""
Complete Training Data Preparation Pipeline

Stages:
1. Save ligands (target + off-target)
2. Prepare target PDB (extract, standardize, sequence blur)
3. Generate Vina configs
4. Vina preparation (PDBQT conversion)
5. Vina docking
6. Vina post-processing
"""
import os
import sys
import logging
import argparse
import yaml
from pathlib import Path
import pandas as pd
from utils.status_tracker import StatusTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self, base_dir: str, csv_path: str, config_template: str = None):
        self.base_dir = Path(base_dir)
        self.csv_path = Path(csv_path)
        self.config_template = Path(config_template) if config_template else None
        
        # Create directory structure
        self.dirs = {
            'ligands': self.base_dir / 'ligands',
            'target_std': self.base_dir / 'target_std',
            'target_blurred': self.base_dir / 'target_blurred',
            'configs': self.base_dir / 'configs',
            'runs': self.base_dir / 'runs',
            'status': self.base_dir / 'status',
        }
        
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        
        # Initialize status tracker
        self.status_tracker = StatusTracker(self.dirs['status'])
        
        # CSV path for each stage
        self.stage_csvs = {
            0: self.csv_path,  # Input CSV
            1: self.base_dir / 'stage1_output.csv',
            2: self.base_dir / 'stage2_output.csv',
            3: self.base_dir / 'stage3_output.csv',
            4: self.base_dir / 'stage4_output.csv',
            5: self.base_dir / 'stage5_output.csv',
            6: self.base_dir / 'stage6_output.csv',
        }
    
    def run_stage1(self):
        """Stage 1: Save ligand files"""
        logger.info("=" * 60)
        logger.info("STAGE 1: Save Ligand Files")
        logger.info("=" * 60)
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / 'stage1_save_ligands.py'),
            '--csv', str(self.stage_csvs[0]),
            '--output_dir', str(self.dirs['ligands']),
            '--status_dir', str(self.dirs['status']),
        ]
        
        import subprocess
        # Real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Stage 1 failed with return code {process.returncode}")
            return False
        
        return True
    
    def run_stage2(self):
        """Stage 2: Prepare target PDB"""
        logger.info("=" * 60)
        logger.info("STAGE 2: Prepare Target PDB")
        logger.info("=" * 60)
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / 'stage2_prep_target.py'),
            '--csv', str(self.stage_csvs[1]),
            '--output_std_dir', str(self.dirs['target_std']),
            '--output_blurred_dir', str(self.dirs['target_blurred']),
            '--status_dir', str(self.dirs['status']),
        ]
        
        import subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Stage 2 failed with return code {process.returncode}")
            return False
        
        return True
    
    def run_stage3(self):
        """Stage 3: Generate Vina configs"""
        logger.info("=" * 60)
        logger.info("STAGE 3: Generate Vina Configs")
        logger.info("=" * 60)
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / 'stage3_generate_configs.py'),
            '--csv', str(self.stage_csvs[2]),
            '--template', str(self.config_template) if self.config_template else str(Path(__file__).parent / 'config.yaml'),
            '--output_configs_dir', str(self.dirs['configs']),
            '--output_runs_dir', str(self.dirs['runs']),
            '--status_dir', str(self.dirs['status']),
        ]
        
        import subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Stage 3 failed with return code {process.returncode}")
            return False
        
        return True
    
    def run_stage4(self, use_gpu: bool = False):
        """Stage 4: Vina preparation"""
        logger.info("=" * 60)
        logger.info("STAGE 4: Vina Preparation")
        logger.info("=" * 60)
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / 'stage4_vina_prep_gpu.py'),
            '--csv', str(self.stage_csvs[3]),
            '--runs_dir', str(self.dirs['runs']),
            '--status_dir', str(self.dirs['status']),
        ]
        
        if use_gpu:
            cmd.append('--use_gpu')
        
        import subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Stage 4 failed with return code {process.returncode}")
            return False
        
        return True
    
    def run_stage5(self, use_gpu: bool = False):
        """Stage 5: Vina docking (GPU accelerated)"""
        logger.info("=" * 60)
        logger.info("STAGE 5: Vina Docking (GPU accelerated)")
        logger.info("=" * 60)
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / 'stage5_vina_dock_gpu.py'),
            '--csv', str(self.stage_csvs[4]),
            '--runs_dir', str(self.dirs['runs']),
            '--status_dir', str(self.dirs['status']),
        ]
        
        if use_gpu:
            cmd.append('--use_gpu')
        
        import subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Stage 5 failed with return code {process.returncode}")
            return False
        
        return True
    
    def run_stage6(self):
        """Stage 6: Vina post-processing"""
        logger.info("=" * 60)
        logger.info("STAGE 6: Vina Post-processing")
        logger.info("=" * 60)
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / 'stage6_vina_post.py'),
            '--csv', str(self.stage_csvs[5]),
            '--runs_dir', str(self.dirs['runs']),
            '--status_dir', str(self.dirs['status']),
        ]
        
        import subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Stage 6 failed with return code {process.returncode}")
            return False
        
        return True
    
    def generate_progress_report(self):
        """Generate comprehensive progress report CSV"""
        logger.info("=" * 60)
        logger.info("Generating Progress Report")
        logger.info("=" * 60)
        
        # Load original CSV
        df = pd.read_csv(self.csv_path)
        
        # Get all complex IDs from status directory
        status_dirs = [d for d in self.dirs['status'].iterdir() if d.is_dir()]
        
        progress_records = []
        
        for idx, row in df.iterrows():
            complex_id = row.get('complex_id', f"complex_{idx:04d}")
            
            record = {
                'complex_id': complex_id,
                'protein_key': row.get('protein_key', 'N/A'),
                'target_het_id': row.get('target_het_id', 'N/A'),
                'offtarget_het_id': row.get('ligand_het_id', 'N/A'),
            }
            
            # Check status for each stage
            for stage in self.status_tracker.STAGES:
                status = self.status_tracker.check_status(complex_id, stage)
                record[stage] = status or 'pending'
                
                # Add error message if failed
                if status == 'failed':
                    error = self.status_tracker.read_error(complex_id, stage)
                    record[f'{stage}_error'] = error
            
            progress_records.append(record)
        
        # Create progress DataFrame
        progress_df = pd.DataFrame(progress_records)
        
        # Save progress CSV
        progress_csv = self.base_dir / 'pipeline_progress.csv'
        progress_df.to_csv(progress_csv, index=False)
        logger.info(f"Saved progress report to {progress_csv}")
        
        # Print summary statistics
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        
        for stage in self.status_tracker.STAGES:
            counts = progress_df[stage].value_counts().to_dict()
            total = len(progress_df)
            
            logger.info(f"\n{stage}:")
            logger.info(f"  ✓ Success: {counts.get('success', 0)} ({counts.get('success', 0)/total*100:.1f}%)")
            logger.info(f"  ✗ Failed:  {counts.get('failed', 0)} ({counts.get('failed', 0)/total*100:.1f}%)")
            logger.info(f"  ⊘ Skipped: {counts.get('skip', 0)} ({counts.get('skip', 0)/total*100:.1f}%)")
            logger.info(f"  ⋯ Pending: {counts.get('pending', 0)} ({counts.get('pending', 0)/total*100:.1f}%)")
        
        # Overall success rate (all stages completed)
        all_success = (progress_df[self.status_tracker.STAGES] == 'success').all(axis=1).sum()
        logger.info(f"\n{'='*60}")
        logger.info(f"Overall Success: {all_success}/{total} ({all_success/total*100:.1f}%)")
        logger.info(f"{'='*60}\n")
        
        return progress_df
    
    def run(self, start_stage: int = 1, end_stage: int = 6, use_gpu: bool = False):
        """Run pipeline from start_stage to end_stage"""
        logger.info("=" * 60)
        logger.info("Training Data Preparation Pipeline")
        logger.info(f"Base Directory: {self.base_dir}")
        logger.info(f"Input CSV: {self.csv_path}")
        logger.info(f"Stages: {start_stage} to {end_stage}")
        logger.info(f"Use GPU: {use_gpu}")
        logger.info("=" * 60)
        
        stage_funcs = {
            1: self.run_stage1,
            2: self.run_stage2,
            3: self.run_stage3,
            4: lambda: self.run_stage4(use_gpu=use_gpu),
            5: lambda: self.run_stage5(use_gpu=use_gpu),
            6: self.run_stage6,
        }
        
        for stage_num in range(start_stage, end_stage + 1):
            if stage_num in stage_funcs:
                success = stage_funcs[stage_num]()
                if not success:
                    logger.error(f"Stage {stage_num} failed. Stopping pipeline.")
                    break
        
        # Generate final progress report
        self.generate_progress_report()


def main():
    parser = argparse.ArgumentParser(
        description='Training Data Preparation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Run all stages
        python pipeline.py --csv input.csv --base_dir /scratch/data/runs
        
        # Run specific stages with GPU
        python pipeline.py --csv input.csv --base_dir /scratch/data/runs --start 4 --end 5 --use_gpu
        
        # Generate progress report only
        python pipeline.py --csv input.csv --base_dir /scratch/data/runs --report-only
                """
    )
    
    parser.add_argument('--csv', required=True, help='Input CSV with protein-target-offtarget pairs')
    parser.add_argument('--base_dir', required=True, help='Base output directory')
    parser.add_argument('--template', help='Config template YAML file (optional)')
    parser.add_argument('--start', type=int, default=1, help='Start stage (default: 1)')
    parser.add_argument('--end', type=int, default=6, help='End stage (default: 6)')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU acceleration (Uni-Dock) for docking')
    parser.add_argument('--report-only', action='store_true', help='Only generate progress report')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        base_dir=args.base_dir,
        csv_path=args.csv,
        config_template=args.template
    )
    
    # Run pipeline or report
    if args.report_only:
        pipeline.generate_progress_report()
    else:
        pipeline.run(start_stage=args.start, end_stage=args.end, use_gpu=args.use_gpu)


if __name__ == '__main__':
    main()