#!/bin/bash
#SBATCH -J lig
#SBATCH -p cpu
#SBATCH --mem=32G
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################

python /home/yunmin/proj/LigandMPNN/dataprep/off/save_ligand.py

###############################################
slurm_end $SLURM_CHANNEL_ID
