#!/bin/bash
#SBATCH -J placer
#SBATCH -c 4
#SBATCH -q normal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nice=10000

#source /home/yunmin/miniforge3/bin/activate ligandmpnn_env

source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
set -euo pipefail

base_dir="/scratch/yunmin/data/db/v2"
mode="${1:?mode required: tar|off}"

shopt -s nullglob
for parent_dir in "$base_dir"/*/; do
    parent=$( basename ${parent_dir} )
    if [ "${parent}" != "P00519_T315I" ] && [ "${parent}" !=  "P00533_L858R" ]; then
      continue
    fi
    case "$mode" in
        tar) in_dir="$parent_dir/cleaned_tar"; out_dir="$parent_dir/placer_tar_large" ;;
        off) in_dir="$parent_dir/cleaned_off"; out_dir="$parent_dir/placer_off_large" ;;
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
        if [ "${fname%_*}" != "P00519_T315I_4WA9" ] && [ "${fname%_*}" != "P00533_L858R_5X2C" ]; then
          continue
        fi
        echo "$f | ligand: $ligand_id"
        module load placer 
        placer \
            --ifile "$f" \
		    --odir "$out_dir/${parent}_${pdb_id}_${ligand_id}" \
		    --rerank prmsd \
		    -n 500 \
            --predict_ligand "$ligand_id"
    done
done
###############################################
slurm_end $SLURM_CHANNEL_ID
