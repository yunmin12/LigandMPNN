#!/bin/bash
#SBATCH -J pipeline
#SBATCH -p cpu
#SBATCH --mem=32G
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################

set="train" # train, val, or test
BASE_DIR=/scratch/yunmin/data/graph/train/example/$set && \
  python /home/yunmin/proj/LigandMPNN/dataprep/train/regenerate_progress.py --base_dir $BASE_DIR && \
  python /home/yunmin/proj/LigandMPNN/dataprep/train/pipeline_statistics.py --base_dir $BASE_DIR && \
  python /home/yunmin/proj/LigandMPNN/dataprep/train/generate_summary_report.py --base_dir $BASE_DIR

###############################################
slurm_end $SLURM_CHANNEL_ID
