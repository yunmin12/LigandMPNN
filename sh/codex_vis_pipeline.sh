# visualization
OFF_LIST=( 1N1 AQ4 BAX STI YJA ZD6 )
for OFF in "${OFF_LIST[@]}"; do
  python /home/yunmin/proj/LigandMPNN/eval/compare_specificity.py \
    --orig_scores ~/proj/data/db/v2/lmpnn_out_1105_v0/P00519_T315I/P00519_T315I_4WA9_AXI/scores \
    --mod_target_scores ~/proj/data/db/v2/lmpnn_out_1105_v2/P00519_T315I/P00519_T315I_4WA9_AXI/${OFF}/scores \
    --mod_off_scores ~/proj/data/db/v2/lmpnn_out_1105_v2/P00519_T315I/P00519_T315I_4WA9_AXI/${OFF}/off_scores \
    --target_pdb ~/proj/data/db/v2/P00519_T315I/cleaned_tar/P00519_T315I_4WA9_AXI.pdb \
    --key_mutations B:T315I \
    --output_dir ~/proj/data/db/v2/lmpnn_out_1105/P00519_T315I_4WA9_AXI;
done

python /home/yunmin/proj/LigandMPNN/eval/compare_specificity.py \
  --orig_scores ~/proj/data/db/v2/lmpnn_out_1105_v0/P00533_L858R/P00533_L858R_5X2C_7XR/scores \
  --mod_target_scores ~/proj/data/db/v2/lmpnn_out_1105_v2/P00533_L858R/P00533_L858R_5X2C_7XR/7XO/scores \
  --mod_off_scores ~/proj/data/db/v2/lmpnn_out_1105_v2/P00533_L858R/P00533_L858R_5X2C_7XR/7XO/off_scores \
  --target_pdb ~/proj/data/db/v2/P00533_L858R/cleaned_tar/P00533_L858R_5X2C_7XR.pdb \
  --key_mutations A:L858R \
  --output_dir ~/proj/data/db/v2/lmpnn_out_1105/P00533_L858R_5X2C_7XR

echo "✅ Done!"
