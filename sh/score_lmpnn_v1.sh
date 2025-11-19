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
OUT_DIR=${1:-lmpnn_out_1028_v1}

shopt -s nullglob
for MUT_DIR in "${BASE}/${OUT_DIR}"/*; do
    PRT_MUT="$(basename "$MUT_DIR")"
    for PREFIX_DIR in "$MUT_DIR"/*; do
      PREFIX="$(basename "$PREFIX_DIR")"
      echo "📁 Current directory: $PREFIX_DIR"
      SCORE_DIR="$PREFIX_DIR/scores_v2"
      mkdir -p $SCORE_DIR
      OFF_DIR="${IN_OFF_DIR}/${PRT_MUT}/${PREFIX}"
      OFFS=( "$OFF_DIR"/*.pdb )
      OFF_ARGS=()
      for f in "${OFFS[@]}"; do OFF_ARGS+=( --offtarget_pdb_path "$f" ); done
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
