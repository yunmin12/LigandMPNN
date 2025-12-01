#!/bin/bash
#SBATCH -J frx
#SBATCH -c 4
#SBATCH -q normal
#SBATCH -p cpu
#SBATCH --mem=16G
#SBATCH --array=0-3

source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
SEEDS=(111 222 333 444)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "🌱 Current seed: $SEED"
OUTLOG="frx.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERRLOG="frx.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

srun python /home/yunmin/proj/LigandMPNN/fastrelax/relax_sc_simple_hb_offrepack_edit.py \
  /scratch/yunmin/data/db/PYR1/frx_tar_wt/pdb_list_tar_wt.txt \
  WI5 \
  /scratch/yunmin/data/db/PYR1/inputs/lig_params/WI5.params \
  "" \
  dump_pdb \
  "$SEED" \
  > "${OUTLOG}" 2> "${ERRLOG}"
###############################################
slurm_end $SLURM_CHANNEL_ID
