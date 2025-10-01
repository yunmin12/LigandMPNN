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
BASE=/home/yunmin/proj/data/bench
INPUT="$BASE/inputs/trimmed_pdb"
OUT="$BASE/outputs/1001"

mkdir -p "$OUT"
find "$INPUT" -type f -name "trimmed*" | while IFS= read -r file; do
    echo "📁 file: $file"
    #python /home/yunmin/proj/LigandMPNN/run.py \
    module load ligandmpnn
    ligandmpnn \
        --seed 111 \
        --model_type "ligand_mpnn" \
        --pdb_path "$file" \
        --out_folder "$OUT" \
        --number_of_batches 4 \
        --batch_size 4 \
        --save_stats 1  \
        --pack_side_chains 1 \
        --number_of_packs_per_design 1 \
        --pack_with_ligand_context 1 \
        --temperature 0.2;
done
###############################################
slurm_end $SLURM_CHANNEL_ID
