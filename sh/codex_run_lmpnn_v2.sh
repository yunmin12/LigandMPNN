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

BASE_DIR="${BASE_DIR:-$HOME/proj/data/db/v2}"
IN_TAR_DIR="${BASE_DIR}/lmpnn_in_tar"
IN_OFF_DIR="${BASE_DIR}/lmpnn_in_off"
OUT_DIR="${1:-lmpnn_out_1104_v2}"
OUT_ROOT="${BASE_DIR}/${OUT_DIR}"
mkdir -p "$OUT_ROOT"

TGT_NUM="$3"

SEED=111
NB=4
PACK_SC=1
PACKS_PER=1
PACK_WITH_LIG=1
TEMP=0.3
NEG_ENABLE=1
TAR_WEIGHT=1.0
NEG_WEIGHT="$2"
NEG_RES="158"
AUTO_POCKET=0

FAILED=()

echo "[INFO] BASE_DIR=$BASE_DIR"

shopt -s nullglob
TGT_DIR_LIST=(
  "$BASE_DIR/lmpnn_in_tar/P00519_T315I/P00519_T315I_4WA9_AXI"
  "$BASE_DIR/lmpnn_in_tar/P00533_L858R/P00533_L858R_5X2C_7XR"
)
echo $TGT_NUM
for TGT_DIR in "${TGT_DIR_LIST[$TGT_NUM]}"; do
  PREFIX="$(basename "$TGT_DIR")"
  echo $PREFIX
  TGT_FILES=( "$TGT_DIR"/*.pdb )
  (( ${#TGT_FILES[@]} )) || continue

  for TGT in ${TGT_FILES[@]}; do
    PATTERN="${PREFIX%_*}"
    OFF_DIRS=( ${IN_OFF_DIR}/${PREFIX%_*_*}/*"$PATTERN"* )
    (( ${#OFF_DIRS[@]} )) || continue
    for OFF_DIR in ${OFF_DIRS[@]}; do
      OFF_PREFIX="$(basename "$OFF_DIR")"
      OFFS=( "$OFF_DIR"/*.pdb )
      if (( ${#OFFS[@]} == 0 )); then
        echo "[WARN] no off-target ensemble for target: $PREFIX / off-target: $OFF_PREFIX"
       FAILED+=( "$PREFIX/$OFF_PREFIX" )
        continue
      fi
        
      OUT_DIR="${OUT_ROOT}/${PREFIX%_*_*}/${PREFIX}/${OFF_PREFIX##*_}"
      mkdir -p "$OUT_DIR"

      OFF_ARGS=()
      for off in "${OFFS[@]}"; do OFF_ARGS+=( --offtarget_pdb_path "$off" ); done

      echo "[RUN] MUT=${PREFIX%_*_*}, PREFIX=${PREFIX}, target=$(basename "$TGT"), off-target=${OFF_PREFIX##*_}"
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
        echo "[FAIL] ${PREFIX%_*_*}/${PREFIX}/$(basename "$TGT")"
        FAILED+=( "${PREFIX%_*_*}/${PREFIX}/$(basename "$TGT")" )
      fi
    done
  done
done
shopt -u nullglob

echo "===== SUMMARY (v2 structured) ====="
if (( ${#FAILED[@]} )); then
  echo "[FAILED COUNT] ${#FAILED[@]}"
  printf '%s\n' "${FAILED[@]}"
else
  echo "[OK] all targets finished"
fi
