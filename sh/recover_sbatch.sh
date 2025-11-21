#!/bin/bash
#SBATCH -J placer
#SBATCH -c 4
#SBATCH -q normal
#SBATCH -p cpu
#SBATCH --mem=16G

# source /home/yunmin/miniforge3/bin/activate ligandmpnn_env
echo "===== Script Contents ====="
echo "submitted script: $0"
cat "$0"
echo "==========================="

source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
set -euo pipefail

BASE_DIR="/scratch/yunmin/data/db/v2/P00519_T315I"

python /home/yunmin/proj/LigandMPNN/dataprep/recover_v2.py --base_dir $BASE_DIR --ligand-chain Z --ligand-resname "AQ4" --mode off
###############################################
slurm_end $SLURM_CHANNEL_ID
