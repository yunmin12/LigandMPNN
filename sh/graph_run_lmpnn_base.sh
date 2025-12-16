#!/bin/bash
#SBATCH -J lmpnn
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
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

# LigandMPNN Baseline
source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
set -euo pipefail

BASE="/scratch/yunmin/data/graph/lmpnn/bdb_pdb"
ASSAY_TYPE="$1"
BASE_DIR="${BASE}/${ASSAY_TYPE}"
OUT_ROOT="${BASE_DIR}/lmpnn_out"

SEED=111
NB=10
PACK_SC=1
PACKS_PER=1
PACK_WITH_LIG=1
TEMP=0.3
NEG_ENABLE=0
NEG_WEIGHT=0
NEG_RES=""

FAILED=()

echo "[INFO] BASE_DIR=$BASE_DIR"

TOTAL_PROTEINS=$(find "$BASE_DIR/lmpnn_in" -maxdepth 1 -type d | tail -n +2 | wc -l)
CURRENT_PROTEIN=0
echo "[INFO] Total proteins to process: $TOTAL_PROTEINS"

shopt -s nullglob
for PROTEIN_DIR in $BASE_DIR/lmpnn_in/*; do
  PROTEIN=$(basename ${PROTEIN_DIR})
  CURRENT_PROTEIN=$((CURRENT_PROTEIN + 1))
  echo "📁 Protein dir: ${PROTEIN} ($CURRENT_PROTEIN/$TOTAL_PROTEINS)"
  for TAR_DIR in $PROTEIN_DIR/*; do
    TAR=$(basename ${TAR_DIR})
    for LIG_DIR in $TAR_DIR/*; do
      LIG=$(basename ${LIG_DIR})
      OUT_DIR="${OUT_ROOT}/$PROTEIN/$TAR/$LIG"
      mkdir -p "$OUT_DIR"

      echo "➡️ [RUN] ${TAR} ligand: ${LIG}"
      for TGT in ${LIG_DIR}/*.pdb; do
        PDB_NAME=$(basename "$TGT")
        echo "  Processing: $PDB_NAME"
        echo "=== Processing PDB: $PDB_NAME (${PROTEIN}/${TAR}/${LIG}) ===" >&2
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
        echo "❌ [FAIL] $(basename "$TGT")"
        FAILED+=( "$(basename "$TGT")" )
        fi
      done
    done
  done
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
