temp="$1"
neg_weight="$2"

sbatch ~/proj/LigandMPNN/sh/run_lmpnn_v0.sh $temp
echo "run_lmpnn_v0.sh $temp"
sbatch ~/proj/LigandMPNN/sh/run_lmpnn_v1.sh $temp $neg_weight
echo "run_lmpnn_v1.sh $temp $neg_weight"
sbatch ~/proj/LigandMPNN/sh/run_lmpnn_v2.sh $temp $neg_weight P00519_T315I SKE
sbatch ~/proj/LigandMPNN/sh/run_lmpnn_v2.sh $temp $neg_weight P00519_T315I AXI
sbatch ~/proj/LigandMPNN/sh/run_lmpnn_v2.sh $temp $neg_weight P00533_G719C
sbatch ~/proj/LigandMPNN/sh/run_lmpnn_v2.sh $temp $neg_weight P00533_L858R
sbatch ~/proj/LigandMPNN/sh/run_lmpnn_v2.sh $temp $neg_weight P52732_D130V
echo "run_lmpnn_v1.sh $temp $neg_weight"

sbatch ~/proj/LigandMPNN/sh/score_lmpnn_v0.sh /scratch/yunmin/data/db/v2/lmpnn_out_1031_v0
sbatch ~/proj/LigandMPNN/sh/score_lmpnn_v2.sh /scratch/yunmin/data/db/v2/lmpnn_out_1031_v1
sbatch ~/proj/LigandMPNN/sh/score_lmpnn_v2.sh /scratch/yunmin/data/db/v2/lmpnn_out_1031_v2
echo "score_lmpnn_v0, v1, v2.sh"

sbatch ~/proj/LigandMPNN/sh/score_vis_match.sh lmpnn_out_1031 P00519_T315I_4WA9_AXI T315I
sbatch ~/proj/LigandMPNN/sh/score_vis_match.sh lmpnn_out_1031 P00533_L858R_5X2C_7XR L858R
echo "score_vis_match.sh"


