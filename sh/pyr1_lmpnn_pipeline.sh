#!/bin/bash
#SBATCH -J lmpnn
#SBATCH -c 4
#SBATCH -q normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

BASE_DIR="/scratch/yunmin/data/db/PYR1"
OUT_DIR_PREFIX="$1"
OUT_DIR="${BASE_DIR}/${OUT_DIR_PREFIX}"
SH_DIR="/home/yunmin/proj/LigandMPNN/sh"

set -e

echo "🚀 Launching PYR1 pipeline..."

echo "STEP 1: lmpnn_v0 (tar)"
jid1=$(sbatch "$SH_DIR/pyr1_run_lmpnn_v0.sh" "$OUT_DIR_PREFIX" tar | awk '{print $4}')
echo "  ➤ Submitted job $jid1"

echo "STEP 2: lmpnn_v0 (off) AFTER step1"
jid2=$(sbatch --dependency=afterok:${jid1} \
        "$SH_DIR/pyr1_run_lmpnn_v0.sh" "$OUT_DIR_PREFIX" off | awk '{print $4}')
echo "  ➤ Submitted job $jid2"

echo "STEP 3: lmpnn_v2 AFTER step2"
jid3=$(sbatch --dependency=afterok:${jid2} \
        "$SH_DIR/pyr1_run_lmpnn_v2.sh" "$OUT_DIR_PREFIX" 10.0 0 "" | awk '{print $4}')
echo "  ➤ Submitted job $jid3"

#sbatch $SH_DIR/pyr1_run_lmpnn_v2.sh "${OUT_DIR_PREFIX}" 2.0 1 ""
#sbatch $SH_DIR/pyr1_run_lmpnn_v2.sh "${OUT_DIR_PREFIX}" 10.0 0 "["A64", "A165", "A166"]"

echo "STEP 4: score_lmpnn AFTER step3"
jid4=$(sbatch --dependency=afterok:${jid3} \
        "$SH_DIR/pyr1_score_lmpnn.sh" "$OUT_DIR_PREFIX" | awk '{print $4}')
echo "  ➤ Submitted job $jid4"

echo "STEP 5: vis AFTER score"
jid5=$(sbatch --dependency=afterok:${jid4} \
        "$SH_DIR/pyr1_vis.sh" "$OUT_DIR_PREFIX" | awk '{print $4}')
echo "  ➤ Submitted job $jid5"

echo "🎉 All jobs submitted with dependencies."
echo "Pipeline:"
echo "  $jid1 → $jid2 → $jid3 → $jid4 → $jid5"

