#!/bin/bash
#SBATCH -J lmpnn
#SBATCH -c 4
#SBATCH -q normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# source /home/yunmin/miniforge3/bin/activate ligandmpnn_env

source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
set -euo pipefail

BASE=/home/yunmin/proj/data/db/v2
IN_OFF_DIR="${BASE}/lmpnn_in_off"
OUT_DIR=${1:-$BASE/lmpnn_out_1028_v2}

shopt -s nullglob
for MUT_DIR in "$OUT_DIR"/*; do
    PRT_MUT="$(basename "$MUT_DIR")"
    for PREFIX_DIR in "$MUT_DIR"/*; do
      PREFIX="$(basename "$PREFIX_DIR")"
      for LIG_DIR in "$PREFIX_DIR"/*; do
        OFF_TGT="${LIG_DIR##*_}"
        echo "📁 Current directory: $PREFIX/$OFF_TGT"
        SCORE_DIR="$LIG_DIR/scores"
        mkdir -p $SCORE_DIR
        OFF_DIR="${IN_OFF_DIR}/${PRT_MUT}/${PREFI/X}"
        OFFS=( "$OFF_DIR"/*.pdb )
        OFF_ARGS=()
        for f in "${OFFS[@]}"; do OFF_ARGS+=( --offtarget_pdb_path "$f" ); done
        python /home/yunmin/proj/LigandMPNN/score.py \
          --seed 111 \
          --model_type "ligand_mpnn" \
          --pdb_path $(ls $PREFIX_DIR/backbones/*.pdb | head -n 1) \
          --out_folder "$SCORE_DIR" \
          --number_of_batches 4 \
          --batch_size 1 \
          --single_aa_score 1 \
          --use_sequence 1 \
          "${OFF_ARGS[@]}"
      done
  done
done
###############################################
slurm_end $SLURM_CHANNEL_ID
