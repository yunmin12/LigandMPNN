BASE="/scratch/yunmin/data/db/PYR1"
IN_DIR="$BASE/lmpnn_out_1119"

echo "🎯 PYR1 -> target: WI5, off-target: A8S"
python /home/yunmin/proj/LigandMPNN/eval/compare_specificity.py \
  --orig_tar_scores ${IN_DIR}/lmpnn_tar/scores \
  --mod_tar_scores ${IN_DIR}/lmpnn_mod/scores \
  --orig_off_scores ${IN_DIR}/lmpnn_off/scores \
  --target_pdb ${BASE}/cleaned_tar/PYR1_WI5.pdb \
  --key_mutations A:K64Q A:F165A A:A166I \
  --output_dir ${IN_DIR}/plots

