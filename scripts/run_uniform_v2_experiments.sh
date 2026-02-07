#!/bin/bash
# Unified View Ablation v2.0 - Experiment Runner
# Created: 2026-02-06
# Usage: ./scripts/run_uniform_v2_experiments.sh [phase1|phase2|all]

cd /home/joon/dev/FaceLift
mkdir -p logs

TORCHRUN="/home/joon/anaconda3/envs/facelift/bin/torchrun --standalone --nproc_per_node=1"
BASE_CFG="configs/mouse/uniform/base_uniform_v2.yaml"

run_experiment() {
    local gpu=$1
    local name=$2
    local cfg=$3
    echo "Starting $name on GPU $gpu..."
    export CUDA_VISIBLE_DEVICES=$gpu && $TORCHRUN train_gslrm.py -b $BASE_CFG -e $cfg > logs/uniform_v2_$name.log 2>&1 &
    echo "  PID: $!"
}

phase1() {
    echo "=== Phase 1: baseline, 1view, 2view, 3view ==="
    run_experiment 4 baseline configs/mouse/uniform/baseline_v2.yaml
    run_experiment 5 1view configs/mouse/uniform/1view_v2.yaml
    run_experiment 6 2view configs/mouse/uniform/2view_v2.yaml
    run_experiment 7 3view configs/mouse/uniform/3view_v2.yaml
    echo ""
    echo "Phase 1 started. Monitor with: tail -f logs/uniform_v2_*.log"
}

phase2() {
    echo "=== Phase 2: 5view, 6view ==="
    run_experiment 4 5view configs/mouse/uniform/5view_v2.yaml
    run_experiment 5 6view configs/mouse/uniform/6view_v2.yaml
    echo ""
    echo "Phase 2 started. Monitor with: tail -f logs/uniform_v2_*.log"
}

case "${1:-phase1}" in
    phase1)
        phase1
        echo ""
        echo "After baseline completes (~2min), run: $0 phase2"
        ;;
    phase2)
        phase2
        ;;
    all)
        phase1
        echo ""
        echo "Waiting for baseline to complete (monitoring for ~3 min)..."
        sleep 180
        phase2
        ;;
    *)
        echo "Usage: $0 [phase1|phase2|all]"
        exit 1
        ;;
esac

echo ""
echo "Check progress:"
echo "  WandB: https://wandb.ai/FaceLift-Mouse"
echo "  Logs: tail -f logs/uniform_v2_*.log"
echo "  GPU: gpustat -i 5"
