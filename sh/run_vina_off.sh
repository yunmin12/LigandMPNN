#!/bin/bash
#SBATCH -J vina
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH -c 4
#SBATCH -q normal
#SBATCH -p cpu
#SBATCH --mem=32G

source /apps/.slurmrc
slurm_start $SLURM_CHANNEL_ID
###############################################
set -euo pipefail

BASE_DIR="/scratch/yunmin/data/graph/lmpnn/bdb_pdb/off_inputs/runs"
COMPLEX_NUM="$1"

shopt -s nullglob
complexes=("$BASE_DIR"/*/)

cnt_complexes=0
total_complexes=$(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "TOTAL $total_complexes complexes prepared"

# stats
total_seeds=0
success_seeds=0
fail_seeds=0

valid_complex=0
invalid_complex=0

valid_list="$BASE_DIR/valid_complexes_off${COMPLEX_NUM}.txt"
: > "$valid_list"


for COMP_DIR in "$BASE_DIR"/*off$COMPLEX_NUM/; do
  cnt_complexes=$((cnt_complexes + 1))
  complex_all_ok=1
  echo "========================================" >&2
  echo "Current complex: $(basename $COMP_DIR) ($cnt_complexes/$total_complexes)" >&2
  echo "========================================" >&2
  
  PREP_DIR="$COMP_DIR/prepared"
  echo "🚀 Starting Vina docking jobs in $PREP_DIR" 
  for SEED in {0..2}; do
    echo "🌱 Starting Vina docking seed $SEED"
    config_file="$PREP_DIR/vina_config_seed$SEED.txt"
    echo "   Config: $config_file"
    
    # 1. Config existence
    [[ -f "$config_file" ]] || { echo "❌ Missing config: $config_file"; continue; } >&2
    
    # 2. Receptor/ligand pdbqt existence
    if [[ ! -s "$PREP_DIR/receptor.pdbqt" || ! -s "$PREP_DIR/offtarget_ligand.pdbqt" ]]; then
      echo "❌ Missing PDBQT files. Skipping." >&2
      continue 
    fi

    # 3. Filter ROOT in receptor
    if grep -qE '^(ROOT|BRANCH|TORSDOF)' "$PREP_DIR/receptor.pdbqt"; then
      echo "❌ Invalid receptor PDBQT (ligand tags found). Skipping." >&2
      continue
    fi

    # 4. Filter atom type error (e.g. Og/Ce/Sg/Ho)
    if grep -qE ' (Og|Ce|Sg|Ho)$' "$PREP_DIR/receptor.pdbqt" "$PREP_DIR/offtarget_ligand.pdbqt"; then
      echo "❌ Invalid atom types in PDBQT. Skipping." >&2
      continue
    fi

    if vina --config $config_file; then
      echo "✅ Vina docking seed $SEED completed" >&2
      success_seeds=$((success_seeds + 1))
    else
      echo "❌ Vina docking $(basename $config_file) failed" >&2
      fail_seeds=$((fail_seeds + 1))
      complex_all_ok=0
    fi
  done
  echo "✅ All Vina docking jobs completed"
  
  if [[ $complex_all_ok -eq 1 ]]; then
    valid_complex=$((valid_complex + 1))
    echo "$(basename "$COMP_DIR")" >> "$valid_list"
  else
    invalid_complex=$((invalid_complex + 1))
  fi
done
shopt -u nullglob

# save stats
seed_ok_pct=0
complex_ok_pct=0
if [[ $total_seeds -gt 0 ]]; then
  seed_ok_pct=$(( success_seeds * 100 / total_seeds ))
fi
if [[ $total_complexes -gt 0 ]]; then
  complex_ok_pct=$(( valid_complex * 100 / total_complexes))
fi

echo "========================================" >&2
echo "SUMMARY" >&2
echo "Complex: total=$total_complexes  valid(all seeds ok)=$valid_complex  invalid=$invalid_complex  valid%=${complex_ok_pct}%" >&2
echo "Seeds:   total=$total_seeds  success=$success_seeds  fail=$fail_seeds  ok%=${seed_ok_pct}%" >&2
echo "Valid complex list saved to: $valid_list" >&2
echo "========================================" >&2

###############################################
slurm_end $SLURM_CHANNEL_ID
