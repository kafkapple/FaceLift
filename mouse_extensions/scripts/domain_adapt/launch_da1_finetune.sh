#!/bin/bash
# DA1 GS-LRM Fine-tuning Launcher
# Waits for datagen to complete, then launches fine-tuning.
#
# Usage: CUDA_VISIBLE_DEVICES=6 bash launch_da1_finetune.sh

set -e

MVDIFF_DIR="$HOME/data/preprocessed/FaceLift_mouse/M5_mvdiff"
EXPECTED_FRAMES=2880
FACELIFT_DIR="$HOME/dev/FaceLift"
PYTHON="$HOME/anaconda3/envs/facelift/bin/python"
TXT_FILE="$MVDIFF_DIR/data_mouse_t2_train_mvdiff.txt"

echo "=== DA1 Fine-tune Launcher ==="
echo "Waiting for datagen to complete ($EXPECTED_FRAMES frames)..."

while true; do
    DONE=$(find "$MVDIFF_DIR" -name "cam_005.png" 2>/dev/null | wc -l)
    echo "[$(date +%H:%M)] $DONE / $EXPECTED_FRAMES frames complete"

    if [ "$DONE" -ge "$EXPECTED_FRAMES" ]; then
        echo "Datagen complete!"
        break
    fi

    sleep 300  # check every 5 minutes
done

# Verify txt file exists and has correct count
if [ ! -f "$TXT_FILE" ]; then
    echo "WARNING: txt file not found, generating..."
    find "$MVDIFF_DIR" -maxdepth 1 -type d -name '[0-9]*' | sort > "$TXT_FILE"
fi

TXT_LINES=$(wc -l < "$TXT_FILE")
echo "Dataset txt: $TXT_FILE ($TXT_LINES lines)"

# Launch GS-LRM fine-tuning
echo ""
echo "=== Launching GS-LRM DA1 Fine-tuning ==="
echo "Config: domain_adapt_E2_v1.yaml"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

cd "$FACELIFT_DIR"
$PYTHON train_gslrm.py \
    -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/domain_adapt_E2_v1.yaml \
    2>&1 | tee /tmp/da1_finetune.log
