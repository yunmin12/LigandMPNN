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
#protein="$1"
BASE=/home/yunmin/proj/data/bench
INPUT="$BASE/outputs/1001/packed"
OUT="$BASE/outputs/1001/scores"

mkdir -p "$OUT"
find "$INPUT" -type f |  while IFS= read -r file; do
    echo "📁 file: $file"
    python /home/yunmin/proj/LigandMPNN/score.py \
        --seed 111 \
        --model_type "ligand_mpnn" \
        --pdb_path "$file" \
        --out_folder "$OUT" \
        --number_of_batches 10 \
        --batch_size 1 \
        --single_aa_score 1 \
        --use_sequence 1;
done
###############################################
slurm_end $SLURM_CHANNEL_ID
