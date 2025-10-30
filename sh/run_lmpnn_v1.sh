#!/bin/bash
#SBATCH -J lmpnn
#SBATCH -c 4
#SBATCH -q normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# source /home/yunmin/miniforge3/bin/activate ligandmpnn_env
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_BACKTRACE=1

echo "===== Script Contents ====="
echo "submitted script: $0"
cat "$0"
echo "==========================="

# ver.1 target complex -> off-target ensemble
source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
set -euo pipefail

BASE_DIR="${BASE_DIR:-$HOME/proj/data/db/v2}"
IN_OFF_DIR="$BASE_DIR/lmpnn_in_off"
OUT_ROOT="${BASE_DIR}/lmpnn_out_1029_v1"
mkdir -p "$OUT_ROOT"

SEED=111
NB=4
PACK_SC=0
PACKS_PER=0
PACK_WITH_LIG=0
TEMP="$1"
NEG_ENABLE=1
NEG_WEIGHT="$2"
NEG_RES=""

FAILED=()

echo "[INFO] BASE_DIR=$BASE_DIR"

shopt -s nullglob
for TGT in "$BASE_DIR"/*/cleaned_tar/*.pdb; do
  PRT_MUT="$(basename "$(dirname "$(dirname "$TGT")")")"
  PRT_MUT_PDB="$(basename "$TGT" .pdb)"

  OFF_BASE="${BASE_DIR}/lmpnn_in_off/${PRT_MUT}"
  [[ -d "$OFF_BASE" ]] || { echo "[WARN] no lmpnn_in_off/$PRT_MUT"; continue; }

  MATCH_DIRS=( $(find "$OFF_BASE" -maxdepth 1 -type d -name "${PRT_MUT_PDB%_*}_*") )
  if (( ${#MATCH_DIRS[@]} == 0 )); then
    echo "[WARN] no off-target folder for $OFF_BASE/${PRT_MUT_PDB%_*}_*"
    FAILED+=( "${PRT_MUT_PDB%_*}" )
    continue
  fi
    
  for PREFIX_DIR in "${MATCH_DIRS[@]}"; do
    PREFIX="$(basename "$PREFIX_DIR")"
    OFFS=( "$PREFIX_DIR"/*.pdb )
    (( ${#OFFS[@]} )) || continue
 
    OUT_DIR="${OUT_ROOT}/${PRT_MUT}/${PREFIX}"
    mkdir -p "$OUT_DIR"

    OFF_ARGS=()
    for f in "${OFFS[@]}"; do OFF_ARGS+=( --offtarget_pdb_path "$f" ); done

    echo "[RUN] MUT=${PRT_MUT}, pdbid=${PRT_MUT_PDB%*_*_}, prefix=${PREFIX}, off_n=${#OFFS[@]}"

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
      echo "[FAIL] ${PREFIX}"
      FAILED+=( "${PREFIX}" )
    fi
  done
done
shopt -u nullglob

echo "===== SUMMARY (v1 structured) ====="
if (( ${#FAILED[@]} )); then
  echo "[FAILED COUNT] ${#FAILED[@]}"
  printf '%s\n' "${FAILED[@]}"
else
  echo "[OK] all pairs finished"
fi
###############################################
slurm_end $SLURM_CHANNEL_ID
