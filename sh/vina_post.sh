#!/bin/bash
#SBATCH -J vina
#SBATCH -p cpu
#SBATCH --mem=32G
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
set -euo pipefail

CONFIG_NUM="$1"
CONFIG_DIR="/scratch/yunmin/data/graph/lmpnn/bdb_pdb/off_inputs/configs_$CONFIG_NUM"

SUCCESS=0
FAILED=0
CNT=0

shopt -s nullglob
configs=("$CONFIG_DIR"/*.yaml)
shopt -u nullglob

TOTAL=${#configs[@]}
echo "Total $TOTAL config(s) in $CONFIG_DIR"

for CONFIG_FILE in "${configs[@]}"; do
    # if python /home/yunmin/proj/LigandMPNN/dataprep/off/vina_post.py \
    if python /home/yunmin/proj/LigandMPNN/dataprep/off/vina_select.py \
      "$CONFIG_FILE"; then
    CNT=$((CNT + 1))
    SUCCESS=$((SUCCESS + 1))
    echo "Vina prepared: $(basename "$CONFIG_FILE") ($CNT/$TOTAL)"
  else 
    FAILED=$(($FAILED + 1))
    echo "FAILED: $(basename "$CONFIG_FILE")" >&2
  fi
done

echo "Done. TOTAL=$TOTAL SUCCESS=$SUCCESS FAILED=$FAILED"
###############################################
slurm_end $SLURM_CHANNEL_ID
