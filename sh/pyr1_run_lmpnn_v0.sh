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

# ver.0 vanila LigandMPNN (only target and only off-target) 
source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
set -euo pipefail

BASE_DIR="${BASE_DIR:-/scratch/yunmin/data/db/v2/fastrelax/P00533_L858R}"
OUT_PRT_DIR="${1:-lmpnn_out_1104_v0}"
OUT_ROOT="${BASE_DIR}/${OUT_PRT_DIR}"
mkdir -p "$OUT_ROOT"
MODE="$2"

SEED=111
NB=4
PACK_SC=1
PACKS_PER=1
PACK_WITH_LIG=1
TEMP=0.3
NEG_ENABLE=0
NEG_WEIGHT=0
NEG_RES=""

FAILED=()

echo "[INFO] BASE_DIR=$BASE_DIR"
OUT_DIR="${OUT_ROOT}/lmpnn_${MODE}"
mkdir -p "$OUT_DIR"

shopt -s nullglob
for TGT in $BASE_DIR/lmpnn_in_${MODE}/*.pdb; do
  echo "[RUN] ${TGT}"
  if ! python /home/yunmin/proj/LigandMPNN/run.py \
    --seed "$SEED" \
    --model_type "ligand_mpnn" \
    --pdb_path "$TGT" \
    --out_folder "$OUT_DIR" \
    --number_of_batches "$NB" \
    --pack_side_chains "$PACK_SC" \
    --number_of_packs_per_design "$PACKS_PER" \
    --pack_with_ligand_context "$PACK_WITH_LIG" \
    --temperature "$TEMP"; then
    echo "[FAIL] $(basename "$TGT")"
    FAILED+=( "$(basename "$TGT")" )
  fi
done
shopt -u nullglob

echo "===== SUMMARY (v0) ====="
if (( ${#FAILED[@]} )); then
  echo "[FAILED COUNT] ${#FAILED[@]}"
  printf '%s\n' "${FAILED[@]}"
else
  echo "[OK] all pairs finished"
fi
###############################################
slurm_end $SLURM_CHANNEL_ID
