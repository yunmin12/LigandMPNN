#!/user/bin/env bash
set -euo pipefail

BASE_DIR=$HOME/proj/data/db/v2
SH_DIR=$HOME/proj/LigandMPNN/sh

# run_lmpnn_v2
jobids=""
for i in {0..3}; do
  jid=$(sbatch $SH_DIR/codex_run_lmpnn_v2.sh lmpnn_out_1105_v2 2.0 ${i} | awk '{print $4}')
  jobids="${jobids}:$jid"
done
jobids=${jobids#:}

# score_lmpnn_v2
sbatch --dependency=afterok:$jobids $SH_DIR/score_lmpnn_v2.sh lmpnn_out_1105_v2
sbatch --dependency=afterok:$jobids $SH_DIR/score_lmpnn_v2_off.sh lmpnn_out_1105_v2 "P00519_T315I_4WA9_AXI"
sbatch --dependency=afterok:$jobids $SH_DIR/score_lmpnn_v2_off.sh lmpnn_out_1105_v2 "P00533_L858R_5X2C_7XR"

echo "✅ Done!"
