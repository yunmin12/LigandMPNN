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

BASE=/scratch/yunmin/data/db/v2
IN_OFF_DIR="${BASE}/lmpnn_in_off"
OUT_DIR_SUFFIX=${1:-lmpnn_out_1031_v2}
OUT_DIR="$BASE/$OUT_DIR_SUFFIX"

shopt -s nullglob
for MUT_DIR in "$OUT_DIR"/*; do
    PRT_MUT="$(basename "$MUT_DIR")"
    for PREFIX_DIR in "$MUT_DIR"/*; do
      PREFIX="$(basename "$PREFIX_DIR")"
      if [ "$PREFIX" != "$2" ]; then
        continue
      fi
      for LIG_DIR in "$PREFIX_DIR"/*; do
        OFF_TGT="$(basename "$LIG_DIR")"
        echo "📁 Current directory: $LIG_DIR"
        SCORE_DIR="${LIG_DIR}/off_scores"
        mkdir -p $SCORE_DIR
        OFF_DIR="${IN_OFF_DIR}/${PRT_MUT}/${PREFIX%_*}_${OFF_TGT}"
        OFFS=( "$OFF_DIR"/*.pdb )
        OFF_ARGS=()
        for f in "${OFFS[@]}"; do
          echo $f
          python /home/yunmin/proj/LigandMPNN/score.py \
            --seed 111 \
            --model_type "ligand_mpnn" \
            --pdb_path "$f" \
            --out_folder "$SCORE_DIR" \
            --number_of_batches 4 \
            --batch_size 1 \
            --single_aa_score 1 \
            --use_sequence 1;
        done    
      done
    done
done
###############################################
slurm_end $SLURM_CHANNEL_ID
