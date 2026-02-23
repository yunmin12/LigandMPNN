#!/bin/bash
#SBATCH -J cluster
#SBATCH -p cpu
#SBATCH --mem=32G
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
set -euo pipefail

python -u /home/yunmin/proj/LigandMPNN/dataprep/off_cluster.py \
    --mode all_pairwise \
    --similarity_methods 2d \
    --chunk_size 50 \
    --resume \
    --max_ligands 1000 \
    --write_batch_size 100000

###############################################
slurm_end $SLURM_CHANNEL_ID
