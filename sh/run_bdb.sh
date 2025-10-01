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
python /home/yunmin/proj/LigandMPNN/scripts/bindingdb_prep.py \
--in /home/yunmin/proj/data/db/BindingDB_BindingDB_Articles.tsv \
--out /home/yunmin/proj/data/db/benchmarking_set.csv
###############################################
slurm_end $SLURM_CHANNEL_ID
