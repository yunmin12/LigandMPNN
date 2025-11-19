#!/bin/bash
#SBATCH -J bdb
#SBATCH -c 4
#SBATCH -q normal
#SBATCH -p cpu
#SBATCH --mem=16G

# source /home/yunmin/miniforge3/bin/activate ligandmpnn_env

source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
python /home/yunmin/proj/LigandMPNN/dataprep/bindingdb_v2.py \
  --out /scratch/yunmin/data/db/15_eval_set_1015_ensure_pdb_same_mut.csv \
  --input articles --ensure_pdb True --same_mutation True
###############################################
slurm_end $SLURM_CHANNEL_ID
