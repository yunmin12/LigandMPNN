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
OUT_ROOT="${BASE_DIR}/lmpnn_out_1031_v2"
mkdir -p "$OUT_ROOT"

SEED=111
NB=10
PACK_SC=1
PACKS_PER=1
PACK_WITH_LIG=1
TEMP="$1"
NEG_ENABLE=1
NEG_WEIGHT="$2"
NEG_RES=""

FAILED=()

echo "[INFO] BASE_DIR=$BASE_DIR"

shopt -s nullglob
for MUT_DIR in "$IN_TAR_DIR"/*; do
  PRT_MUT="$(basename "$MUT_DIR")"
  if [ $PRT_MUT != "$3" ]; then
    continue
  fi
  for PREFIX_DIR in "$MUT_DIR"/*; do
    PREFIX="$(basename "$PREFIX_DIR")"
    if [[ -n "${4:-}"  && ${PREFIX##*_} != "$4" ]]; then
      continue
    fi
    TGT_FILES=( "$PREFIX_DIR"/*.pdb )
    (( ${#TGT_FILES[@]} )) || continue

    for TGT in ${TGT_FILES[@]}; do
      PATTERN="${PREFIX%_*}"
      OFF_DIRS=( ${IN_OFF_DIR}/${PRT_MUT}/*"$PATTERN"* )
      (( ${#OFF_DIRS[@]} )) || continue
      for OFF_DIR in ${OFF_DIRS[@]}; do
        OFF_PREFIX="$(basename "$OFF_DIR")"
        OFFS=( "$OFF_DIR"/*.pdb )
        if (( ${#OFFS[@]} == 0 )); then
          echo "[WARN] no off-target ensemble for target: $PREFIX / off-target: $OFF_PREFIX"
          FAILED+=( "$PREFIX/$OFF_PREFIX" )
          continue
        fi
        
        OUT_DIR="${OUT_ROOT}/${PRT_MUT}/${PREFIX}/${OFF_PREFIX##*_}"
        mkdir -p "$OUT_DIR"

        OFF_ARGS=()
        for off in "${OFFS[@]}"; do OFF_ARGS+=( --offtarget_pdb_path "$off" ); done

        echo "[RUN] MUT=${PRT_MUT}, PREFIX=${PREFIX}, target=$(basename "$TGT"), off-target=${OFF_PREFIX##*_}"
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
          --negative_weight "$NEG_WEIGHT" \
          --negative_residues "$NEG_RES" \
          "${OFF_ARGS[@]}"; then
          echo "[FAIL] ${PRT_MUT}/${PREFIX}/$(basename "$TGT")"
          FAILED+=( "${PRT_MUT}/${PREFIX}/$(basename "$TGT")" )
        fi
      done
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
