#!/bin/bash
#SBATCH -J placer
#SBATCH -c 4
#SBATCH -q normal
#SBATCH -p cpu
#SBATCH --mem=16G

#source /home/yunmin/miniforge3/bin/activate ligandmpnn_env

source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
set -euo pipefail

base_dir="/scratch/yunmin/data/db/v2"
mode="${1:?mode required: tar|off}"

shopt -s nullglob
for parent_dir in "$base_dir"/*/; do
    case "$mode" in
        tar) in_dir="$parent_dir/cleaned_tar"; out_dir="$parent_dir/placer_tar" ;;
        off) in_dir="$parent_dir/cleaned_off"; out_dir="$parent_dir/placer_off" ;;
        *) echo "mode must be target or off" ?&2; exit 1 ;;
    esac
    
    [[ -d "$in_dir" ]] || { echo "Skip $parent_dir (no replaced)"; continue; }
    mkdir -p "$out_dir"

    pdbs=("$in_dir"/*.pdb)
    (( ${#pdbs[@]} )) || { echo "No pdb in $in_dir"; continue; }

    echo "=== Directory: ${parent_dir%/} ==="
    for f in "${pdbs[@]}"; do
        fname=$(basename "$f" .pdb)
        pdb_id=$(echo "$fname" | cut -d'_' -f3)
        ligand_id=$(echo "$fname" | cut -d'_' -f4)
        echo "$f | ligand: $ligand_id"
        module load placer 
        placer \
            --ifile "$f" \
		    --odir "$out_dir/$ligand_id/$pdb_id" \
		    --rerank prmsd \
		    -n 50 \
            --predict_ligand "$ligand_id"
            # --corruption_centers B-427-A8S-O7 B-427-A8S-C8 B-427-A8S-C9 \
    done
done
###############################################
slurm_end $SLURM_CHANNEL_ID
