#!/bin/bash
#SBATCH -J rfdaa
#SBATCH -c 4
#SBATCH -q normal
#SBATCH -p gpu
#SBATCH --mem 16G
#SBATCH --gres=gpu:1
# source /home/yunmin/miniforge3/bin/activate ligandmpnn_env
source /apps/.slurmrc
slurm_start D09CXEGQUT0
###############################################
outdir="$1"
pdb="$2"
contig="$3"
lig="$4"
module load rfdiffusion
rfdiffusion \
    inference.deterministic=False \
    diffuser.T=30 \
    inference.output_prefix=${outdir}/output \
    inference.input_pdb=/scratch/yunmin/data/rfdaa/${outdir}/${pdb}_trimmed.pdb \
    contigmap.contigs=[\'${contig}-${contig}\'] \
    inference.ligand=${lig} \
    inference.num_designs=5 \
    inference.design_startnum=0
###############################################
slurm_end D09CXEGQUT0
