#!/bin/bash
#SBATCH -J prep_unidock
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Training Data Preparation Pipeline Runner (Uni-Dock GPU)

source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
set -e

# Activate unidock environment
source ~/opt/miniconda3/etc/profile.d/conda.sh
conda activate unidock

# Configuration
set="test"  # Options: train, val, test
CSV_PATH="/scratch/yunmin/data/graph/train/identity50/csvs/06_sample_example/${set}_set_sampled_20targets_100offtargets.csv"
BASE_DIR="/scratch/yunmin/data/graph/train/example/${set}_std"

echo "=========================================="
echo "Training Data Preparation Pipeline (Uni-Dock GPU)"
echo "=========================================="
echo "CSV: $CSV_PATH"
echo "Output: $BASE_DIR"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Check unidock
which unidock || echo "WARNING: unidock not found"
unidock --version || echo "WARNING: unidock version check failed"

# Run pipeline
python /home/yunmin/proj/LigandMPNN/dataprep/train/pipeline.py \
    --csv "$CSV_PATH" \
    --base_dir "$BASE_DIR" \
    --start 4 \
    --end 6 \
    --use_gpu

echo "=========================================="
echo "Pipeline completed!"
echo "=========================================="
echo "Check progress report at: ${BASE_DIR}/pipeline_progress.csv"
echo "=========================================="
###############################################
slurm_end $SLURM_CHANNEL_ID
