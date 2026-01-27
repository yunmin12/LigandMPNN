"""
Stage 5: Molecular Docking (GPU accelerated)
Run docking using Uni-Dock (GPU) or Vina (CPU)

UniDock batch strategy:
- Group configs by (uniprot_id, target_id) 
- One receptor.pdbqt per group
- Multiple ligands via ligand_index file
- YAML config for each seed
"""
import os
import sys
import logging
import argparse
import subprocess
import json
import yaml
from pathlib import Path
from collections import defaultdict
import pandas as pd
from typing import Optional, Dict, List, Tuple
from utils.status_tracker import StatusTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DockingRunner:
    def __init__(self, status_tracker: StatusTracker, runs_dir: Path,
                 use_gpu: bool = True, gpu_id: int = 0, cpu_threads: int = 4):
        self.status_tracker = status_tracker
        self.runs_dir = runs_dir
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.cpu_threads = cpu_threads
        
        # Detect docking engine
        if use_gpu:
            try:
                result = subprocess.run(['unidock', '--version'], 
                                       capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    logger.info(f"Using Uni-Dock on GPU {gpu_id}")
                    self.docking_engine = 'unidock'
                else:
                    raise FileNotFoundError
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.warning("Uni-Dock not found, falling back to Vina")
                self.use_gpu = False
                self.docking_engine = 'vina'
        else:
            self.docking_engine = 'vina'
            logger.info(f"Using AutoDock Vina (CPU, {cpu_threads} threads)")
    
    def group_configs_by_receptor(self, df: pd.DataFrame) -> Dict[Tuple[str, str], List[dict]]:
        """
        Group configs by (uniprot_id, target_id) to share same receptor
        
        Returns:
            Dict mapping (uniprot_id, target_id) to list of config info
        """
        groups = defaultdict(list)
        
        for idx, row in df.iterrows():
            config_id = row.get('config_id', f"config_{idx:06d}")
            
            # Parse config_id: {uniprot_id}_{target_id}_{off_id}
            parts = config_id.split('_')
            if len(parts) >= 3:
                uniprot_id = parts[0]
                target_id = parts[1]
                off_id = parts[2]
            else:
                logger.warning(f"Invalid config_id format: {config_id}")
                continue
            
            # Check if can run this stage
            if not self.status_tracker.can_run_stage(config_id, '5_vina_dock'):
                continue
            
            groups[(uniprot_id, target_id)].append({
                'config_id': config_id,
                'uniprot_id': uniprot_id,
                'target_id': target_id,
                'off_id': off_id,
                'row': row
            })
        
        logger.info(f"Grouped {len(df)} configs into {len(groups)} receptor groups")
        for (uniprot_id, target_id), configs in groups.items():
            logger.info(f"  {uniprot_id}_{target_id}: {len(configs)} ligands")
        
        return groups
    
    def _parse_vina_config(self, config_file: Path) -> dict:
        """Parse Vina-format config file into a dictionary"""
        config = {}
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        return config
    
    def run_unidock_group(self, uniprot_id: str, target_id: str, 
                          configs: List[dict]) -> Tuple[int, int]:
        """
        Run UniDock for a group of ligands sharing the same receptor
        
        Returns:
            (success_count, total_count)
        """
        try:
            # Use first config as reference for receptor and docking parameters
            ref_config = configs[0]
            ref_config_id = ref_config['config_id']
            ref_run_dir = self.runs_dir / ref_config_id
            ref_prepared_dir = ref_run_dir / 'prepared'
            
            # Get receptor path
            receptor_pdbqt = ref_prepared_dir / 'receptor.pdbqt'
            if not receptor_pdbqt.exists():
                raise ValueError(f"Receptor not found: {receptor_pdbqt}")
            
            # Parse docking parameters from vina config
            vina_config_file = ref_prepared_dir / 'vina_config_seed0.txt'
            if not vina_config_file.exists():
                raise ValueError(f"Vina config not found: {vina_config_file}")
            
            vina_params = self._parse_vina_config(vina_config_file)
            
            # Create group working directory
            group_dir = self.runs_dir / f"_unidock_batch_{uniprot_id}_{target_id}"
            group_dir.mkdir(parents=True, exist_ok=True)
            
            # Create ligand_index file (list of all ligand paths)
            ligand_index_file = group_dir / 'ligand_index.txt'
            config_map = {}  # Map ligand path to config info
            
            with open(ligand_index_file, 'w') as f:
                for config in configs:
                    config_id = config['config_id']
                    run_dir = self.runs_dir / config_id
                    ligand_pdbqt = run_dir / 'prepared' / 'off_ligand.pdbqt'
                    
                    if not ligand_pdbqt.exists():
                        logger.warning(f"Ligand not found: {ligand_pdbqt}")
                        continue
                    
                    f.write(f"{ligand_pdbqt}\n")
                    config_map[str(ligand_pdbqt)] = config
            
            logger.info(f"\n{'='*70}")
            logger.info(f"UniDock batch: {uniprot_id}_{target_id}")
            logger.info(f"  Receptor: {receptor_pdbqt}")
            logger.info(f"  Ligands: {len(config_map)}")
            logger.info(f"{'='*70}")
            
            # Get number of seeds from vina configs
            seed_files = sorted(ref_prepared_dir.glob('vina_config_seed*.txt'))
            n_seeds = len(seed_files)
            
            # Run docking for each seed
            success_count = 0
            total_expected = len(config_map) * n_seeds
            
            for seed_idx, seed_file in enumerate(seed_files):
                seed = seed_file.stem.split('seed')[1]
                
                # Read seed-specific parameters if different
                seed_params = self._parse_vina_config(seed_file)
                
                # Create output directory for this seed
                seed_out_dir = group_dir / f'seed{seed}'
                seed_out_dir.mkdir(exist_ok=True)
                
                # Create Vina-style config for UniDock (key = value format, NOT YAML!)
                config_file = group_dir / f'unidock_seed{seed}.txt'
                with open(config_file, 'w') as f:
                    f.write(f"receptor = {receptor_pdbqt}\n")
                    f.write(f"ligand_index = {ligand_index_file}\n")
                    f.write(f"center_x = {seed_params.get('center_x', vina_params.get('center_x', 0))}\n")
                    f.write(f"center_y = {seed_params.get('center_y', vina_params.get('center_y', 0))}\n")
                    f.write(f"center_z = {seed_params.get('center_z', vina_params.get('center_z', 0))}\n")
                    f.write(f"size_x = {seed_params.get('size_x', vina_params.get('size_x', 20))}\n")
                    f.write(f"size_y = {seed_params.get('size_y', vina_params.get('size_y', 20))}\n")
                    f.write(f"size_z = {seed_params.get('size_z', vina_params.get('size_z', 20))}\n")
                    f.write(f"exhaustiveness = {seed_params.get('exhaustiveness', vina_params.get('exhaustiveness', 32))}\n")
                    f.write(f"num_modes = {seed_params.get('num_modes', vina_params.get('num_modes', 5))}\n")
                    f.write(f"seed = {seed}\n")
                    f.write(f"dir = {seed_out_dir}\n")
                    f.write(f"search_mode = balance\n")
                
                # Run UniDock
                log_file = group_dir / f'unidock_seed{seed}.log'
                
                cmd = [
                    'unidock',
                    '--config', str(config_file)
                ]
                
                logger.info(f"Running UniDock seed {seed} ({seed_idx+1}/{n_seeds})...")
                logger.info(f"  Command: {' '.join(cmd)}")
                
                with open(log_file, 'w') as f:
                    result = subprocess.run(
                        cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        timeout=1800,  # 30 min timeout for batch
                        cwd=str(group_dir)
                    )
                
                # Check outputs and distribute to individual config directories
                if result.returncode == 0:
                    logger.info(f"  UniDock seed {seed} completed, distributing results...")
                    
                    # UniDock outputs: {ligand_basename}_out.pdbqt in seed{N}/ directory
                    for ligand_path_str, config in config_map.items():
                        ligand_path = Path(ligand_path_str)
                        ligand_basename = ligand_path.stem  # e.g., "off_ligand"
                        
                        # Find output file
                        output_file = seed_out_dir / f'{ligand_basename}_out.pdbqt'
                        
                        if output_file.exists():
                            # Copy to config's docking directory
                            config_id = config['config_id']
                            dest_dir = self.runs_dir / config_id / 'docking'
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            
                            dest_file = dest_dir / f'out_seed{seed}.pdbqt'
                            import shutil
                            shutil.copy2(output_file, dest_file)
                            
                            # Create log file indicator
                            dest_log = dest_dir / f'unidock_seed{seed}.log'
                            dest_log.write_text(f"Docking completed (batch mode)\nSee: {log_file}\n")
                            
                            success_count += 1
                        else:
                            logger.warning(f"  Output not found: {output_file}")
                    
                    completed = success_count
                    logger.info(f"  Progress: {completed}/{total_expected} ({100*completed/total_expected:.1f}%)")
                else:
                    logger.error(f"  UniDock seed {seed} failed (return code: {result.returncode})")
                    # Log stderr if available
                    try:
                        with open(log_file, 'r') as lf:
                            log_content = lf.read()
                            if log_content:
                                logger.error(f"  Log: {log_content[:500]}")
                    except:
                        pass
            
            # Cleanup batch directory
            import shutil
            shutil.rmtree(group_dir, ignore_errors=True)
            
            return success_count, total_expected
            
        except subprocess.TimeoutExpired:
            logger.error("UniDock timed out")
            return 0, len(configs) * 3
        except Exception as e:
            logger.error(f"UniDock group failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0, len(configs) * 3
    
    def run_vina_sequential(self, prepared_dir: Path) -> bool:
        """Run Vina docking sequentially (CPU)"""
        try:
            config_files = sorted(prepared_dir.glob('vina_config_seed*.txt'))
            if not config_files:
                raise ValueError("No Vina config files found")
            
            docking_dir = prepared_dir.parent / 'docking'
            success_count = 0
            
            for config_file in config_files:
                seed = config_file.stem.split('seed')[1]
                log_file = docking_dir / f'vina_seed{seed}.log'
                
                cmd = [
                    'vina',
                    '--config', str(config_file),
                    '--cpu', str(self.cpu_threads)
                ]
                
                with open(log_file, 'w') as f:
                    result = subprocess.run(
                        cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        timeout=600
                    )
                
                if result.returncode == 0:
                    success_count += 1
                else:
                    logger.warning(f"Vina seed {seed} failed")
            
            if success_count == 0:
                raise ValueError("All Vina runs failed")
            
            logger.info(f"Docking completed: {success_count}/{len(config_files)} successful")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Vina timed out")
            return False
        except Exception as e:
            logger.error(f"Vina failed: {e}")
            return False
    
    def process_single_config(self, row: pd.Series, config_id: str,
                             current_num: int = None, total_num: int = None) -> bool:
        """Run docking for one config (fallback for Vina or individual processing)"""
        # Check if can run this stage
        if not self.status_tracker.can_run_stage(config_id, '5_vina_dock'):
            self.status_tracker.mark_status(config_id, '5_vina_dock', 'skip',
                                           'Previous stage not completed')
            return False
        
        try:
            run_dir = self.runs_dir / config_id
            prepared_dir = run_dir / 'prepared'
            docking_dir = run_dir / 'docking'
            
            if not prepared_dir.exists():
                raise ValueError(f"Prepared directory not found: {prepared_dir}")
            
            # Format progress indicator
            progress_str = ""
            if current_num is not None and total_num is not None:
                progress_str = f" ({current_num}/{total_num})"
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Docking {config_id}{progress_str}")
            logger.info(f"{'='*70}")
            
            # Run Vina sequential
            success = self.run_vina_sequential(prepared_dir)
            
            if not success:
                raise ValueError("Docking failed")
            
            logger.info(f"✓ {config_id}: Docking completed")
            self.status_tracker.mark_status(config_id, '5_vina_dock', 'success')
            return True
            
        except Exception as e:
            logger.error(f"✗ {config_id}: Docking failed: {e}")
            self.status_tracker.mark_status(config_id, '5_vina_dock', 'failed', str(e))
            return False
    
    def process_all_configs(self, df: pd.DataFrame) -> Tuple[int, int]:
        """
        Process all configs using batch mode for UniDock or sequential for Vina
        
        Returns:
            (success_count, total_count)
        """
        if self.docking_engine == 'unidock':
            # Use batch processing
            logger.info("Using UniDock batch processing mode")
            groups = self.group_configs_by_receptor(df)
            
            total_success = 0
            total_count = 0
            group_num = 0
            
            for (uniprot_id, target_id), configs in groups.items():
                group_num += 1
                logger.info(f"\n{'='*70}")
                logger.info(f"Processing receptor group {group_num}/{len(groups)}: {uniprot_id}_{target_id}")
                logger.info(f"  {len(configs)} ligands")
                logger.info(f"{'='*70}")
                
                success, total = self.run_unidock_group(uniprot_id, target_id, configs)
                total_success += success
                total_count += total
                
                # Mark status for each config
                for config in configs:
                    config_id = config['config_id']
                    # Check if all seeds completed for this config
                    docking_dir = self.runs_dir / config_id / 'docking'
                    n_outputs = len(list(docking_dir.glob('out_seed*.pdbqt'))) if docking_dir.exists() else 0
                    
                    if n_outputs > 0:
                        self.status_tracker.mark_status(config_id, '5_vina_dock', 'success')
                    else:
                        self.status_tracker.mark_status(config_id, '5_vina_dock', 'failed', 
                                                       'No docking outputs created')
                
                logger.info(f"Group {group_num} completed: {success}/{total} successful")
                logger.info(f"Overall progress: {total_success}/{total_count} ({100*total_success/total_count:.1f}%)")
            
            return total_success, total_count
            
        else:
            # Sequential processing for Vina
            logger.info("Using Vina sequential processing mode")
            success_count = 0
            total_configs = len(df)
            
            for idx, row in df.iterrows():
                config_id = row.get('config_id', f"config_{idx:06d}")
                current_num = idx + 1
                
                if self.process_single_config(row, config_id, current_num, total_configs):
                    success_count += 1
            
            return success_count, total_configs


def main():
    p = argparse.ArgumentParser(description="Stage 5: Molecular Docking")
    p.add_argument("--csv", required=True, help="Input CSV (output from stage 4)")
    p.add_argument("--runs_dir", required=True, help="Runs directory")
    p.add_argument("--status_dir", required=True, help="Status tracking directory")
    p.add_argument("--use_gpu", action='store_true', help="Use GPU acceleration (Uni-Dock)")
    p.add_argument("--gpu_id", type=int, default=0, help="GPU device ID (default: 0)")
    p.add_argument("--cpu_threads", type=int, default=4, help="CPU threads for Vina (default: 4)")
    args = p.parse_args()
    
    # Initialize
    status_tracker = StatusTracker(Path(args.status_dir))
    runner = DockingRunner(
        status_tracker,
        runs_dir=Path(args.runs_dir),
        use_gpu=args.use_gpu,
        gpu_id=args.gpu_id,
        cpu_threads=args.cpu_threads
    )
    
    # Load CSV
    df = pd.read_csv(args.csv)
    logger.info(f"Loaded {len(df)} configs from {args.csv}")
    
    # Process all configs (batch or sequential)
    success_count, total_dockings = runner.process_all_configs(df)
    
    # Collect successful configs for output CSV
    updated_rows = []
    for idx, row in df.iterrows():
        config_id = row.get('config_id', f"config_{idx:06d}")
        
        # Check if this config has successful docking outputs
        docking_dir = Path(args.runs_dir) / config_id / 'docking'
        n_outputs = len(list(docking_dir.glob('out_seed*.pdbqt'))) if docking_dir.exists() else 0
        
        if n_outputs > 0:
            row_dict = row.to_dict()
            updated_rows.append(row_dict)
    
    # Save updated CSV
    output_csv = Path(args.runs_dir).parent / "stage5_output.csv"
    pd.DataFrame(updated_rows).to_csv(output_csv, index=False)
    logger.info(f"Saved {len(updated_rows)} successful configs to {output_csv}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"=== STAGE 5 SUMMARY ===")
    print(f"{'='*70}")
    print(f"Input configs:    {len(df)}")
    print(f"Successful dockings: {success_count}/{total_dockings}")
    print(f"Configs with outputs: {len(updated_rows)}")
    print(f"Success rate:     {100*len(updated_rows)/len(df) if len(df) > 0 else 0:.1f}%")
    print(f"Output CSV:       {output_csv}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
