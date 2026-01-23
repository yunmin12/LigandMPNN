#!/bin/bash

# Quick progress checker

BASE_DIR="$1"

if [ -z "$BASE_DIR" ]; then
    echo "Usage: $0 <base_dir>"
    exit 1
fi

STATUS_DIR="${BASE_DIR}/status"

if [ ! -d "$STATUS_DIR" ]; then
    echo "Error: Status directory not found: $STATUS_DIR"
    exit 1
fi

echo "=========================================="
echo "Pipeline Progress Summary"
echo "Base Directory: $BASE_DIR"
echo "=========================================="
echo ""

stages=(
    "1_save_ligands"
    "2_prep_target"
    "3_generate_config"
    "4_vina_prep"
    "5_vina_dock"
    "6_vina_post"
)

for stage in "${stages[@]}"; do
    success=$(find "$STATUS_DIR" -name "${stage}.success" 2>/dev/null | wc -l)
    failed=$(find "$STATUS_DIR" -name "${stage}.failed" 2>/dev/null | wc -l)
    skip=$(find "$STATUS_DIR" -name "${stage}.skip" 2>/dev/null | wc -l)
    total=$((success + failed + skip))
    
    if [ $total -gt 0 ]; then
        success_pct=$(echo "scale=1; $success * 100 / $total" | bc)
    else
        success_pct="0.0"
    fi
    
    echo "Stage: $stage"
    echo "  ✓ Success: $success (${success_pct}%)"
    echo "  ✗ Failed:  $failed"
    echo "  ⊘ Skipped: $skip"
    echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
done

echo ""
echo "=========================================="
echo "Detailed Progress CSV:"
if [ -f "${BASE_DIR}/pipeline_progress.csv" ]; then
    echo "  ${BASE_DIR}/pipeline_progress.csv"
    
    # Show summary from CSV
    total_rows=$(tail -n +2 "${BASE_DIR}/pipeline_progress.csv" | wc -l)
    echo ""
    echo "Total complexes: $total_rows"
else
    echo "  Not yet generated. Run pipeline with --report-only to generate."
fi
echo "=========================================="
