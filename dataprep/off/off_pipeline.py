"""
Pipeline Controller: Orchestrate the entire off-target docking pipeline
"""

import os
import sys
import yaml
import argparse
import subprocess
import logging
from pathlib import Path
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_file):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_step(script_name, config_file, step_name, additional_args=None):
    """
    Run a pipeline step
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {step_name}")
    logger.info(f"{'='*60}")
    
    cmd = ['python', script_name, config_file]
    if additional_args:
        cmd.extend(additional_args)
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        logger.info(f"✓ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {step_name} failed with exit code {e.returncode}")
        return False


def check_vina_completion(config):
    """Check if all Vina docking jobs are complete"""
    output_dir = Path(config['output_dir'])
    docking_dir = output_dir / 'docking'
    
    n_seeds = config['docking']['n_seeds']
    
    for seed in range(n_seeds):
        output_file = docking_dir / f'vina_out_seed{seed}.pdbqt'
        if not output_file.exists():
            return False
    
    return True


def check_relaxation_completion(config):
    """Check if all FastRelax jobs are complete"""
    output_dir = Path(config['output_dir'])
    relaxed_dir = output_dir / 'relaxed'
    prep_csv = relaxed_dir / 'fastrelax_prepared.csv'
    
    if not prep_csv.exists():
        return False
    
    import pandas as pd
    df = pd.read_csv(prep_csv)
    
    for idx, row in df.iterrows():
        pose_name = row['pose_name']
        output_pdb = relaxed_dir / f"{pose_name}_0001.pdb"
        if not output_pdb.exists():
            return False
    
    return True


def print_manual_instructions(step, config):
    """Print manual instructions for server steps"""
    output_dir = Path(config['output_dir'])
    
    if step == 'vina':
        docking_dir = output_dir / 'docking'
        logger.info("\n" + "="*60)
        logger.info("MANUAL STEP: Submit Vina jobs to server")
        logger.info("="*60)
        logger.info("\nSubmit these jobs:")
        for seed in range(config['docking']['n_seeds']):
            script = docking_dir / f'vina_seed{seed}.sh'
            logger.info(f"  sbatch {script}")
        
        logger.info("\nWait for all jobs to complete, then run:")
        logger.info(f"  python pipeline_controller.py {sys.argv[1]} --resume vina_postprocess")
    
    elif step == 'fastrelax':
        relaxed_dir = output_dir / 'relaxed'
        prep_csv = relaxed_dir / 'fastrelax_prepared.csv'
        
        logger.info("\n" + "="*60)
        logger.info("MANUAL STEP: Submit FastRelax jobs to server")
        logger.info("="*60)
        
        import pandas as pd
        df = pd.read_csv(prep_csv)
        
        logger.info(f"\nSubmit {len(df)} jobs:")
        for idx, row in df.iterrows():
            script = row['slurm_script']
            logger.info(f"  sbatch {script}")
        
        logger.info("\nWait for all jobs to complete, then continue with analysis.ipynb")


def main():
    parser = argparse.ArgumentParser(description='Control the off-target docking pipeline')
    parser.add_argument('config', help='Configuration YAML file')
    parser.add_argument('--mode', choices=['auto', 'step', 'check'], default='step',
                       help='Pipeline mode: auto (run all), step (manual control), check (check status)')
    parser.add_argument('--resume', choices=['prep', 'vina_postprocess', 'fastrelax_prep', 'analysis'],
                       help='Resume from specific step')
    parser.add_argument('--skip-vina', action='store_true',
                       help='Skip Vina docking (use existing results)')
    parser.add_argument('--skip-relax', action='store_true',
                       help='Skip FastRelax (use existing results)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    logger.info("="*60)
    logger.info("OFF-TARGET DOCKING PIPELINE CONTROLLER")
    logger.info("="*60)
    logger.info(f"\nConfiguration: {args.config}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output directory: {config['output_dir']}")
    
    # Determine starting step
    if args.resume:
        start_step = args.resume
    else:
        start_step = 'prep'
    
    steps_completed = []
    
    # Define pipeline steps
    pipeline = [
        ('prep', 'vina_prep.py', 'Vina Preparation'),
        ('vina', None, 'Vina Docking (Server)'),
        ('vina_postprocess', 'vina_postprocess.py', 'Vina Post-processing'),
        ('analysis1', None, 'Analysis Part 1 (Jupyter)'),
        ('fastrelax_prep', 'fastrelax_prep.py', 'FastRelax Preparation'),
        ('fastrelax', None, 'FastRelax (Server)'),
        ('analysis2', None, 'Analysis Part 2 (Jupyter)'),
    ]
    
    # Check mode
    if args.mode == 'check':
        logger.info("\n" + "="*60)
        logger.info("CHECKING PIPELINE STATUS")
        logger.info("="*60)
        
        # Check each step
        output_dir = Path(config['output_dir'])
        
        # 1. Vina prep
        receptor_pdbqt = output_dir / 'prepared' / 'receptor.pdbqt'
        if receptor_pdbqt.exists():
            logger.info("✓ Vina preparation complete")
        else:
            logger.info("✗ Vina preparation not complete")
        
        # 2. Vina docking
        if check_vina_completion(config):
            logger.info("✓ Vina docking complete")
        else:
            logger.info("✗ Vina docking not complete")
        
        # 3. Vina postprocess
        filtered_csv = output_dir / 'filtered' / 'poses_filtered.csv'
        if filtered_csv.exists():
            logger.info("✓ Vina post-processing complete")
        else:
            logger.info("✗ Vina post-processing not complete")
        
        # 4. FastRelax prep
        prep_csv = output_dir / 'relaxed' / 'fastrelax_prepared.csv'
        if prep_csv.exists():
            logger.info("✓ FastRelax preparation complete")
        else:
            logger.info("✗ FastRelax preparation not complete")
        
        # 5. FastRelax
        if prep_csv.exists() and check_relaxation_completion(config):
            logger.info("✓ FastRelax complete")
        else:
            logger.info("✗ FastRelax not complete")
        
        return
    
    # Run pipeline
    step_index = [s[0] for s in pipeline].index(start_step) if args.resume else 0
    
    for i, (step_name, script, description) in enumerate(pipeline[step_index:], start=step_index):
        
        # Skip steps if requested
        if step_name in ['vina', 'vina_postprocess'] and args.skip_vina:
            logger.info(f"\nSkipping {description} (--skip-vina)")
            continue
        
        if step_name in ['fastrelax_prep', 'fastrelax'] and args.skip_relax:
            logger.info(f"\nSkipping {description} (--skip-relax)")
            continue
        
        # Run step
        if script:
            # Python script step
            success = run_step(script, args.config, description)
            if not success:
                logger.error(f"\nPipeline stopped due to error in {description}")
                return
        else:
            # Manual/server step
            if step_name == 'vina':
                print_manual_instructions('vina', config)
                
                if args.mode == 'step':
                    logger.info("\nPipeline paused. Resume after Vina completion.")
                    return
                elif args.mode == 'auto':
                    logger.info("\nWaiting for Vina jobs to complete...")
                    while not check_vina_completion(config):
                        time.sleep(60)
                    logger.info("✓ Vina jobs completed")
            
            elif step_name == 'fastrelax':
                print_manual_instructions('fastrelax', config)
                
                if args.mode == 'step':
                    logger.info("\nPipeline paused. Continue with analysis.ipynb after FastRelax completion.")
                    return
                elif args.mode == 'auto':
                    logger.info("\nWaiting for FastRelax jobs to complete...")
                    while not check_relaxation_completion(config):
                        time.sleep(60)
                    logger.info("✓ FastRelax jobs completed")
            
            elif step_name in ['analysis1', 'analysis2']:
                logger.info(f"\n{'='*60}")
                logger.info(f"MANUAL STEP: {description}")
                logger.info(f"{'='*60}")
                logger.info("\nOpen and run analysis.ipynb in Jupyter")
                
                if args.mode == 'step':
                    logger.info("\nPipeline paused. Continue after analysis.")
                    return
        
        steps_completed.append(step_name)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"\nCompleted steps: {', '.join(steps_completed)}")
    logger.info("\nResults available in:")
    logger.info(f"  {config['output_dir']}/final/")


if __name__ == '__main__':
    main()