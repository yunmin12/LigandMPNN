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

# run LigandMPNN ver.2
# each recovered target ensemble -> matched off-target ensemble
source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
set -euo pipefail

BASE_DIR="${BASE_DIR:-/scratch/yunmin/data/db/v2/fastrelax/P00533_L858R}"
OUT_PRT_DIR="${1:-lmpnn_out_1104_v2}"
OUT_ROOT="${BASE_DIR}/${OUT_PRT_DIR}"
mkdir -p "$OUT_ROOT"

SEED=111
NB=4
PACK_SC=1
PACKS_PER=1
PACK_WITH_LIG=1
TEMP=0.3
NEG_ENABLE=1
TAR_WEIGHT=1.0
NEG_WEIGHT="$2"
NEG_RES=""
AUTO_POCKET=0

FAILED=()

echo "[INFO] BASE_DIR=$BASE_DIR"

shopt -s nullglob
OUT_DIR="${OUT_ROOT}/lmpnn_mod"
mkdir -p "$OUT_DIR"
for TGT in "${BASE_DIR}"/lmpnn_in_tar/*.pdb; do
  OFF_DIR="${BASE_DIR}"/lmpnn_in_off
  OFFS=( "$OFF_DIR"/*.pdb )
  if (( ${#OFFS[@]} == 0 )); then
    echo "[WARN] no off-target ensemble for target: "$(basename $TGT)""
    FAILED+=( "$(basename $TGT)" )
    continue
  fi
      
  OFF_ARGS=()
  for off in "${OFFS[@]}"; do OFF_ARGS+=( --offtarget_pdb_path "$off" ); done
  echo "➡️ [RUN] target=$(basename "$TGT")"
  if ! python /home/yunmin/proj/LigandMPNN/run.py \
    --seed "$SEED" \
    --model_type "ligand_mpnn" \
    --pdb_path "$TGT" \
    --out_folder "$OUT_DIR" \
    --number_of_batches "$NB" \
    --pack_side_chains "$PACK_SC" \
    --number_of_packs_per_design "$PACKS_PER" \
    --pack_with_ligand_context "$PACK_WITH_LIG" \
    --temperature "$TEMP" \
    --negative_enable "$NEG_ENABLE" \
    --target_logit_weight "$TAR_WEIGHT" \
    --off_target_logit_weight "$NEG_WEIGHT" \
    --negative_residues "$NEG_RES" \
    --auto_pocket "$AUTO_POCKET" \
    "${OFF_ARGS[@]}"; then
    echo "[FAIL] $(basename "$TGT")"
    FAILED+=( "$(basename "$TGT")" )
  fi
done
shopt -u nullglob

echo "===== SUMMARY (v2 structured) ====="
if (( ${#FAILED[@]} )); then
  echo "[FAILED COUNT] ${#FAILED[@]}"
  printf '%s\n' "${FAILED[@]}"
else
  echo "[OK] all targets finished"
fi
