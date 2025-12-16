#!/bin/bash
#SBATCH -J boltz
#SBATCH -p gpu
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o boltz-%j.out
#SBATCH -e boltz-%j.err

source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
ASSAY_TYPE="$1" # Kd, IC50
YAML_NUM="$2" # 1-4
INPUT_YAML="/scratch/yunmin/data/graph/lmpnn/bdb_pdb/${ASSAY_TYPE}/boltz2/inputs_yaml/yaml${YAML_NUM}"
OUT_DIR="/scratch/yunmin/data/graph/lmpnn/bdb_pdb/${ASSAY_TYPE}/boltz2/outputs"

module load boltz

SUCCESS_COUNT=0
FAIL_COUNT=0

TOTAL_YAML=$(find "${INPUT_YAML}" -maxdepth 1 -type f -name "*.yaml" | wc -l)
CURRENT_YAML=0

shopt -s nullglob
for yaml_file in "${INPUT_YAML}"/*.yaml; do
  CURRENT_YAML=$((CURRENT_YAML + 1))
  if [ -f "$yaml_file" ]; then
    echo "==================================================" | tee /dev/stderr
    echo "Processing: $(basename $yaml_file) ($CURRENT_YAML/$TOTAL_YAML)" | tee /dev/stderr
    echo "==================================================" | tee /dev/stderr
    
    if boltz predict \
      "$yaml_file" \
      --out_dir "${OUT_DIR}" \
      --recycling_steps 3 \
      --sampling_steps 50 \
      --diffusion_samples 3 \
      --sampling_steps_affinity 200 \
      --diffusion_samples_affinity 5 \
      --use_potentials \
      --output_format pdb; then
      echo "✅ Completed: $(basename $yaml_file)" | tee /dev/stderr
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      echo "❌ Failed: $(basename $yaml_file)" | tee /dev/stderr
      FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo "" | tee /dev/stderr
  fi
done
shopt -u nullglob

echo "==================================================" | tee /dev/stderr
echo "Summary: ${SUCCESS_COUNT} succeeded, ${FAIL_COUNT} failed" | tee /dev/stderr
echo "==================================================" | tee /dev/stderr

###############################################
slurm_end $SLURM_CHANNEL_ID
