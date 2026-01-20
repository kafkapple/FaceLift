#!/bin/bash
# Preprocessing Methods Comparison Report Runner
# ===============================================
#
# Usage:
#   ./run_preprocessing_comparison.sh [sample_idx]
#
# Examples:
#   ./run_preprocessing_comparison.sh       # Use default sample 100
#   ./run_preprocessing_comparison.sh 200   # Use sample 200
#
# Created: 2026-01-19

set -e

# Configuration
BASE_DIR="/home/joon/data/preprocessed/FaceLift_mouse"
OUTPUT_DIR="/home/joon/dev/FaceLift/docs/reports"
SCRIPT_DIR="/home/joon/dev/FaceLift/mouse_extensions/scripts"

# Default datasets to compare
DATASETS="D6-1 D7 D7_5 D7_5b"

# Sample index (default: 100)
SAMPLE_IDX=${1:-100}

# Report name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_NAME="preprocessing_comparison_${TIMESTAMP}"

echo "============================================================"
echo "Preprocessing Methods Comparison Report Generator"
echo "============================================================"
echo "Base directory: ${BASE_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Datasets: ${DATASETS}"
echo "Sample index: ${SAMPLE_IDX}"
echo "Report name: ${REPORT_NAME}"
echo ""

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate facelift

# Create output directory if not exists
mkdir -p "${OUTPUT_DIR}"

# Run comparison script
cd /home/joon/dev/FaceLift
python "${SCRIPT_DIR}/compare_preprocessing_methods.py" \
    --base-dir "${BASE_DIR}" \
    --datasets ${DATASETS} \
    --output-dir "${OUTPUT_DIR}" \
    --sample-idx "${SAMPLE_IDX}" \
    --report-name "${REPORT_NAME}"

echo ""
echo "============================================================"
echo "Report generated successfully!"
echo "============================================================"
echo ""
echo "Output files:"
echo "  - Report: ${OUTPUT_DIR}/${REPORT_NAME}_report.md"
echo "  - 6-View: ${OUTPUT_DIR}/${REPORT_NAME}_6view_comparison.png"
echo "  - Coverage: ${OUTPUT_DIR}/${REPORT_NAME}_coverage.png"
echo ""
echo "To view the report:"
echo "  cat ${OUTPUT_DIR}/${REPORT_NAME}_report.md"
echo ""
echo "To add new datasets, edit the DATASETS variable in this script."
