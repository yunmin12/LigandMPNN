BASE="/scratch/yunmin/data/db/v2"
IN_DIR="$1"

for OFF_DIR in $BASE/${IN_DIR}_v2/P00519_T315I/P00519_T315I_4WA9_AXI/*/; do
  OFF=$( basename ${OFF_DIR} )
  echo 🎯 "P00519_T15I, $OFF"
  python /home/yunmin/proj/LigandMPNN/eval/compare_specificity.py \
    --orig_scores /scratch/yunmin/data/db/v2/${IN_DIR}_v0/P00519_T315I/P00519_T315I_4WA9_AXI/scores \
    --mod_target_scores ${OFF_DIR}/scores \
    --mod_off_scores ${OFF_DIR}/off_scores \
    --target_pdb /scratch/yunmin/data/db/v2/P00519_T315I/cleaned_tar/P00519_T315I_4WA9_AXI.pdb \
    --key_mutations B:T315I \
    --output_dir /scratch/yunmin/data/db/v2/${IN_DIR}/P00519_T315I_4WA9_AXI/${OFF}
done

echo "🎯 P00533_L858R, 7XO"
python /home/yunmin/proj/LigandMPNN/eval/compare_specificity.py \
  --orig_scores /scratch/yunmin/data/db/v2/${IN_DIR}_v0/P00533_L858R/P00533_L858R_5X2C_7XR/scores \
  --mod_target_scores /scratch/yunmin/data/db/v2/${IN_DIR}_v2/P00533_L858R/P00533_L858R_5X2C_7XR/7XO/scores \
  --mod_off_scores /scratch/yunmin/data/db/v2/${IN_DIR}_v2/P00533_L858R/P00533_L858R_5X2C_7XR/7XO/off_scores \
  --target_pdb /scratch/yunmin/data/db/v2/P00533_L858R/cleaned_tar/P00533_L858R_5X2C_7XR.pdb \
  --key_mutations A:L858R \
  --output_dir /scratch/yunmin/data/db/v2/${IN_DIR}/P00533_L858R_5X2C_7XR/7XO

