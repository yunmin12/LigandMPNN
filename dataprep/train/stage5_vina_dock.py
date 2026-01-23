"""
Stage 5: Run Vina Docking
Execute AutoDock Vina for all prepared configs
"""
import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
import pandas as pd
from typing import Tuple
from utils.status_tracker import StatusTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VinaDocking:
    def __init__(self, status_tracker: StatusTracker, vina_path: str = 'vina'):
        self.status_tracker = status_tracker
        self.vina_path = vina_path
    
    def validate_pdbqt(self, pdbqt_file: Path, is_receptor: bool = True) -> Tuple[bool, str]:
        """
        Validate PDBQT file
        Returns: (is_valid, error_message)
        """
        if not pdbqt_file.exists():
            return False, f"File not found: {pdbqt_file}"
        
        if pdbqt_file.stat().st_size == 0:
            return False, f"File is empty: {pdbqt_file}"
        
        try:
            with open(pdbqt_file) as f:
                content = f.read()
            
            # For receptor: should NOT have ROOT/BRANCH/TORSDOF
            if is_receptor:
                if any(tag in content for tag in ['ROOT', 'BRANCH', 'TORSDOF']):
                    return False, "Receptor contains ligand tags (ROOT/BRANCH/TORSDOF)"
            
            # Check for invalid atom types
            invalid_atoms = ['Og', 'Ce', 'Sg', 'Ho']
            for line in content.split('\n'):
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_type = line.rstrip()[-2:].strip()
                    if atom_type in invalid_atoms:
                        return False, f"Invalid atom type found: {atom_type}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Error reading file: {e}"
    
    def run_vina_seed(self, config_file: Path, seed: int, docking_dir: Path) -> bool:
        """
        Run Vina for a single seed
        """
        try:
            output_pdbqt = docking_dir / f'out_seed{seed}.pdbqt'
            log_file = docking_dir / f'vina_seed{seed}.log'
            
            cmd = [
                self.vina_path,
                '--config', str(config_file)
            ]
            
            logger.info(f"  Running Vina seed {seed}...")
            
            with open(log_file, 'w') as log_f:
                result = subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    timeout=600  # 10 minutes timeout
                )
            
            if result.returncode != 0:
                raise ValueError(f"Vina exited with code {result.returncode}")
            
            # Check output file
            if not output_pdbqt.exists():
                raise ValueError("Output PDBQT not created")
            
            if output_pdbqt.stat().st_size == 0:
                raise ValueError("Output PDBQT is empty")
            
            logger.info(f"  ✓ Seed {seed} completed")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"  ✗ Seed {seed} timed out")
            return False
        except Exception as e:
            logger.error(f"  ✗ Seed {seed} failed: {e}")
            return False
    
    def process_config(self, config_id: str, runs_dir: Path, n_seeds: int = 5,
                      current_num: int = None, total_num: int = None) -> bool:
        """
        Process one config: run Vina for all seeds
        
        Args:
            config_id: Config ID
            runs_dir: Runs directory
            n_seeds: Number of seeds
            current_num: Current config number (for progress tracking)
            total_num: Total configs (for progress tracking)
        """
        # Check if can run this stage
        if not self.status_tracker.can_run_stage(config_id, '5_vina_dock'):
            self.status_tracker.mark_status(config_id, '5_vina_dock', 'skip',
                                           'Previous stage not completed')
            return False
        
        try:
            run_dir = runs_dir / config_id
            prepared_dir = run_dir / 'prepared'
            docking_dir = run_dir / 'docking'
            
            if not prepared_dir.exists():
                raise ValueError(f"Prepared directory not found: {prepared_dir}")
            
            docking_dir.mkdir(parents=True, exist_ok=True)
            
            # Format progress indicator
            progress_str = ""
            if current_num is not None and total_num is not None:
                progress_str = f" ({current_num}/{total_num})"
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing {config_id}{progress_str}")
            logger.info(f"{'='*70}")
            
            # Validate PDBQT files once
            receptor_pdbqt = prepared_dir / 'receptor.pdbqt'
            ligand_pdbqt = prepared_dir / 'off_ligand.pdbqt'
            
            valid, error = self.validate_pdbqt(receptor_pdbqt, is_receptor=True)
            if not valid:
                raise ValueError(f"Invalid receptor PDBQT: {error}")
            
            valid, error = self.validate_pdbqt(ligand_pdbqt, is_receptor=False)
            if not valid:
                raise ValueError(f"Invalid ligand PDBQT: {error}")
            
            logger.info(f"🚀 Starting Vina docking jobs")
            
            # Run Vina for each seed - ALL must succeed
            success_count = 0
            failed_seeds = []
            
            for seed in range(n_seeds):
                config_file = prepared_dir / f'vina_config_seed{seed}.txt'
                
                if not config_file.exists():
                    logger.warning(f"  🌱 Seed {seed}: Config file not found: {config_file}")
                    failed_seeds.append(seed)
                    continue
                
                logger.info(f"  🌱 Seed {seed}: Starting Vina docking")
                
                if self.run_vina_seed(config_file, seed, docking_dir):
                    success_count += 1
                else:
                    failed_seeds.append(seed)
            
            # ALL seeds must succeed (strict policy, matching run_vina_off.sh)
            if success_count != n_seeds:
                raise ValueError(
                    f"Vina docking incomplete: {success_count}/{n_seeds} seeds succeeded. "
                    f"Failed seeds: {failed_seeds}"
                )
            
            logger.info(f"✅ All Vina docking jobs completed")
            logger.info(f"✓ {config_id}: Vina docking completed ({success_count}/{n_seeds} seeds)")
            self.status_tracker.mark_status(config_id, '5_vina_dock', 'success')
            return True
            
        except Exception as e:
            logger.error(f"✗ {config_id}: Vina docking failed: {e}")
            self.status_tracker.mark_status(config_id, '5_vina_dock', 'failed', str(e))
            return False


def main():    
    p = argparse.ArgumentParser(description="Stage 5: Vina Docking")
    p.add_argument("--csv", required=True, help="Input CSV (output from stage 4)")
    p.add_argument("--runs_dir", required=True, help="Runs directory")
    p.add_argument("--status_dir", required=True, help="Status tracking directory")
    p.add_argument("--vina_path", default="vina", help="Path to Vina executable")
    p.add_argument("--n_seeds", type=int, default=5, help="Number of seeds")
    args = p.parse_args()
    
    # Initialize
    status_tracker = StatusTracker(Path(args.status_dir))
    docker = VinaDocking(status_tracker, vina_path=args.vina_path)
    
    # Load CSV
    df = pd.read_csv(args.csv)
    logger.info(f"Processing {len(df)} configs")
    
    # Process each config
    updated_rows = []
    success_count = 0
    failed_count = 0
    total_configs = len(df)
    
    for idx, row in df.iterrows():
        config_id = row.get('config_id', f"config_{idx:06d}")
        current_num = idx + 1
        
        if docker.process_config(config_id, Path(args.runs_dir), n_seeds=args.n_seeds,
                                current_num=current_num, total_num=total_configs):
            # Only add successful rows to output CSV
            row_dict = row.to_dict()
            updated_rows.append(row_dict)
            success_count += 1
        else:
            failed_count += 1
    
    # Save updated CSV (only successful rows)
    output_csv = Path(args.runs_dir).parent / "stage5_output.csv"
    pd.DataFrame(updated_rows).to_csv(output_csv, index=False)
    logger.info(f"Saved {success_count} successful entries to {output_csv}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"=== STAGE 5 SUMMARY ===")
    print(f"{'='*70}")
    print(f"Total processed:  {len(df)}")
    print(f"Success:          {success_count}")
    print(f"Failed:           {failed_count}")
    print(f"Success rate:     {100*success_count/len(df) if len(df) > 0 else 0:.1f}%")
    print(f"Output CSV:       {output_csv}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
