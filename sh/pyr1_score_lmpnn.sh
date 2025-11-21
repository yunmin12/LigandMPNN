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

BASE_DIR="${BASE_DIR:-/scratch/yunmin/data/db/PYR1}"
OUT_DIR_SUFFIX=${1:-lmpnn_out_1028_v0}
OUT_DIR="$BASE_DIR/$OUT_DIR_SUFFIX"

shopt -s nullglob
for MODE_DIR in "$OUT_DIR"/*; do
  MODE_BASE="$(basename "$MODE_DIR")"
  MODE_BASE="${MODE_BASE##*_}"
  echo "📁 Current mode: ${MODE_BASE}"
  # lmpnn_tar, lmpnn_off, lmpnn_mod
  SCORE_DIR="$MODE_DIR/scores"
  mkdir -p $SCORE_DIR
  for TGT in "$MODE_DIR"/seqs/*.fa; do
    TGT_BASE="$(basename "$TGT")"
    TGT_BASE="${TGT_BASE%.fa}"
    echo "➡️ Current target: $TGT_BASE"
    if [ "$MODE_BASE" == "tar" ] || [ "$MODE_BASE" == "mod" ]; then
      PDB_PATH="$BASE_DIR"/recover_tar/"$TGT_BASE".pdb
    elif [ "$MODE_BASE" == "off" ]; then
      PDB_PATH="$BASE_DIR"/recover_off/"$TGT_BASE".pdb
    fi
    OFF_ARGS=()
    if [ "$MODE_BASE" == "mod" ]; then
      OFF_DIR="$BASE_DIR/recover_off"
      OFFS=( "$OFF_DIR"/*.pdb )
      for off in "${OFFS[@]}"; do OFF_ARGS+=( --offtarget_pdb_path "$off" ); done
    fi
    python /home/yunmin/proj/LigandMPNN/score.py \
      --seed 111 \
      --model_type "ligand_mpnn" \
      --pdb_path $PDB_PATH \
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
