#!/bin/bash
#SBATCH -J prep
#SBATCH -p cpu
#SBATCH --mem=32G
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Training Data Preparation Pipeline Runner

source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
set -e

# Configuration
set="test"  # Options: train, val, test
CSV_PATH="/scratch/yunmin/data/graph/train/identity50/csvs/06_sample_example/${set}_set_sampled_20targets_100offtargets.csv"
BASE_DIR="/scratch/yunmin/data/graph/train/example/${set}"

echo "=========================================="
echo "Training Data Preparation Pipeline"
echo "=========================================="
echo "CSV: $CSV_PATH"
echo "Output: $BASE_DIR"
echo "=========================================="

# Run pipeline
python /home/yunmin/proj/LigandMPNN/dataprep/train/pipeline.py \
    --csv "$CSV_PATH" \
    --base_dir "$BASE_DIR" \
    --start 5 \
    --end 5

echo "=========================================="
echo "Pipeline completed!"
echo "=========================================="
echo "Check progress report at: ${BASE_DIR}/pipeline_progress.csv"
echo "=========================================="
###############################################
slurm_end $SLURM_CHANNEL_ID
