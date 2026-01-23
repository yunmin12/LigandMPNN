#!/bin/bash

# Resume pipeline for failed cases only

set -e

BASE_DIR="$1"
STAGE="$2"

if [ -z "$BASE_DIR" ] || [ -z "$STAGE" ]; then
    echo "Usage: $0 <base_dir> <stage_number>"
    echo ""
    echo "Example: $0 /scratch/data/train_example 2"
    echo ""
    echo "Stages:"
    echo "  1 - Save ligands"
    echo "  2 - Prepare target PDB"
    echo "  3 - Generate configs"
    echo "  4 - Vina preparation"
    exit 1
fi

STATUS_DIR="${BASE_DIR}/status"
CSV_PATH="${BASE_DIR}/stage$((STAGE-1))_output.csv"

if [ $STAGE -eq 1 ]; then
    # For stage 1, we need to find the original input CSV
    echo "Error: For stage 1, please specify the original CSV path"
    exit 1
fi

if [ ! -f "$CSV_PATH" ]; then
    echo "Error: Input CSV not found: $CSV_PATH"
    exit 1
fi

echo "=========================================="
echo "Resuming Pipeline - Stage $STAGE"
echo "Base Directory: $BASE_DIR"
echo "Input CSV: $CSV_PATH"
echo "=========================================="

# Count failed cases
stage_names=(
    ""
    "1_save_ligands"
    "2_prep_target"
    "3_generate_config"
    "4_vina_prep"
)

stage_name="${stage_names[$STAGE]}"
failed_count=$(find "$STATUS_DIR" -name "${stage_name}.failed" 2>/dev/null | wc -l)

echo "Found $failed_count failed cases for stage: $stage_name"
echo ""

if [ $failed_count -eq 0 ]; then
    echo "No failed cases found. Nothing to do."
    exit 0
fi

read -p "Do you want to retry these failed cases? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Remove failed status files to allow retry
echo "Removing failed status files..."
find "$STATUS_DIR" -name "${stage_name}.failed" -delete
echo "Done. Failed statuses cleared."

# Re-run the specific stage
TEMPLATE="${BASE_DIR}/../config.yaml"  # Adjust path as needed

python /home/yunmin/proj/LigandMPNN/dataprep/train/pipeline.py \
    --csv "$CSV_PATH" \
    --base_dir "$BASE_DIR" \
    --template "$TEMPLATE" \
    --start "$STAGE" \
    --end "$STAGE"

echo "=========================================="
echo "Retry completed!"
echo "Check progress with: ./check_progress.sh $BASE_DIR"
echo "=========================================="
