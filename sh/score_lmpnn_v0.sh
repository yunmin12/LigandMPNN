#!/bin/bash
#SBATCH -J lmpnn
#SBATCH -c 4
#SBATCH -q normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# source /home/yunmin/miniforge3/bin/activate ligandmpnn_env
echo "===== Script Contents ====="
echo "submitted script: $0"
cat "$0"
echo "==========================="

# ver.0 vanila LigandMPNN (only target) 
source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
set -euo pipefail

BASE_DIR="${BASE_DIR:-$HOME/proj/data/db/v2}"
OUT_DIR_SUFFIX=${1:-lmpnn_out_1028_v0}
OUT_DIR="$BASE_DIR/$OUT_DIR_SUFFIX"

shopt -s nullglob
for MUT_DIR in "$OUT_DIR"/*; do
    PRT_MUT="$(basename "$MUT_DIR")"
    for PREFIX_DIR in "$MUT_DIR"/*; do
      PREFIX="$(basename "$PREFIX_DIR")"
      echo "📁 Current directory: $PREFIX_DIR"
      SCORE_DIR="$PREFIX_DIR/scores"
      mkdir -p $SCORE_DIR
      
      python /home/yunmin/proj/LigandMPNN/score.py \
        --seed 111 \
        --model_type "ligand_mpnn" \
        --pdb_path $(ls $BASE_DIR/lmpnn_in_tar/$PRT_MUT/${PREFIX}/*.pdb | head -n 1) \
        --out_folder "$SCORE_DIR" \
        --number_of_batches 4 \
        --batch_size 1 \
        --single_aa_score 1 \
        --use_sequence 1 \
        "${OFF_ARGS[@]}"
    done
done
###############################################
slurm_end $SLURM_CHANNEL_ID
