#!/bin/bash
#SBATCH -J lmpnn
#SBATCH -c 4
#SBATCH -q normal
#SBATCH -p cpu
#SBATCH --mem=16G

# source /home/yunmin/miniforge3/bin/activate ligandmpnn_env

source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
protein="$1"
BASE=/home/yunmin/proj/data/lmpnn
#IMG=/home/yunmin/proj/LigandMPNN/mlfold.sif
INPUT="$BASE/$protein/inputs"
OUT="$BASE/$protein/outputs/0916"
#CKPT="/home/yunmin/proj/LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt"

mkdir -p "$OUT"
find "$INPUT" -type f | while IFS= read -r file; do
	echo "file: $file"
	module load ligandmpnn
	ligandmpnn \
    	--seed 111 \
    	--model_type "ligand_mpnn" \
    	--pdb_path "$file" \
    	--out_folder "$OUT" \
    	--number_of_batches 4 \
    	--pack_side_chains 1 \
    	--number_of_packs_per_design 1 \
    	--pack_with_ligand_context 1 \
    	--temperature 0.2;
done
###############################################
slurm_end $SLURM_CHANNEL_ID
