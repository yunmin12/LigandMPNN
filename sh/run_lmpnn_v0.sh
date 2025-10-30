#!/bin/bash
#SBATCH -J lmpnn
#SBATCH -c 4
#SBATCH -q normal
#SBATCH -p cpu
#SBATCH --mem=16G

# source /home/yunmin/miniforge3/bin/activate ligandmpnn_env
export CUDA_LAUNCH_BLOCKING=1
export TORCH_BACKTRACE=1

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
OUT_ROOT="${BASE_DIR}/lmpnn_out_1029_v0"
mkdir -p "$OUT_ROOT"

SEED=111
NB=4
PACK_SC=1
PACKS_PER=1
PACK_WITH_LIG=1
TEMP="$1"
NEG_ENABLE=0
NEG_WEIGHT=0
NEG_RES=""

FAILED=()

echo "[INFO] BASE_DIR=$BASE_DIR"

shopt -s nullglob
for TGT in "${BASE_DIR}"/*/cleaned_tar/*.pdb; do
  PRT_MUT="$(basename "$(dirname "$(dirname "$TGT")")")"
  PREFIX="$(basename "$TGT" .pdb)"
  PDBID="${PREFIX%%_*}"
  
  OUT_DIR="${OUT_ROOT}/${PRT_MUT}/${PREFIX}"
  mkdir -p "$OUT_DIR"

  echo "[RUN] mut=${PRT_MUT} pdbid=${PDBID} prefix=${PREFIX} -> ${OUT_DIR}"  
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
    echo "[FAIL] ${PRT_MUT}/${PREFIX}/$(basename "$TGT")"
    FAILED+=( "${PRT_MUT}/${PREFIX}/$(basename "$TGT")" )
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
