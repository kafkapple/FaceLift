#!/bin/bash
# D6 All Experiments (2026-01-18)
# ========================================
# 3 Datasets x 6 Experiments = 18 Total
#
# Datasets:
#   D6-1: No-crop (원본 보존, 기하학 완벽)
#   D6-2: Virtual relocation (실험적)
#   D6-3: Crop + 정확 PP (D4 개선)
#
# Experiments:
#   E1_baseline: Paper baseline (no mask)
#   E2_gtmask: GT mask
#   E3_alpha: Alpha mask (safe, threshold 0.3)
#   E4_5v_alpha: 5 views + alpha (D4 best)
#   E6_5v_opacity: 5 views + alpha + opacity reg
#   E7_6v_alpha: 6 views + alpha

cd /home/joon/dev/FaceLift

# ============================================================
# D6-1: No-Crop (원본 보존, 기하학 완벽) - GPU 0,1
# ============================================================

# D6-1 E1: Paper baseline
CUDA_VISIBLE_DEVICES=0 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_1_E1_baseline.yaml > logs/d6_1_e1_baseline.log 2>&1 &

# D6-1 E2: GT mask
CUDA_VISIBLE_DEVICES=0 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_1_E2_gtmask.yaml > logs/d6_1_e2_gtmask.log 2>&1 &

# D6-1 E3: Alpha safe
CUDA_VISIBLE_DEVICES=0 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_1_E3_alpha_safe.yaml > logs/d6_1_e3_alpha_safe.log 2>&1 &

# D6-1 E4: 5v alpha
CUDA_VISIBLE_DEVICES=1 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_1_E4_5v_alpha.yaml > logs/d6_1_e4_5v_alpha.log 2>&1 &

# D6-1 E6: 5v opacity
CUDA_VISIBLE_DEVICES=1 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_1_E6_5v_opacity.yaml > logs/d6_1_e6_5v_opacity.log 2>&1 &

# D6-1 E7: 6v alpha
CUDA_VISIBLE_DEVICES=1 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_1_E7_6v_alpha.yaml > logs/d6_1_e7_6v_alpha.log 2>&1 &

# ============================================================
# D6-2: Virtual Relocation (실험적) - GPU 2,3
# ============================================================

# D6-2 E1: Paper baseline
CUDA_VISIBLE_DEVICES=2 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_2_E1_baseline.yaml > logs/d6_2_e1_baseline.log 2>&1 &

# D6-2 E2: GT mask
CUDA_VISIBLE_DEVICES=2 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_2_E2_gtmask.yaml > logs/d6_2_e2_gtmask.log 2>&1 &

# D6-2 E3: Alpha safe
CUDA_VISIBLE_DEVICES=2 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_2_E3_alpha_safe.yaml > logs/d6_2_e3_alpha_safe.log 2>&1 &

# D6-2 E4: 5v alpha
CUDA_VISIBLE_DEVICES=3 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_2_E4_5v_alpha.yaml > logs/d6_2_e4_5v_alpha.log 2>&1 &

# D6-2 E6: 5v opacity
CUDA_VISIBLE_DEVICES=3 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_2_E6_5v_opacity.yaml > logs/d6_2_e6_5v_opacity.log 2>&1 &

# D6-2 E7: 6v alpha
CUDA_VISIBLE_DEVICES=3 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_2_E7_6v_alpha.yaml > logs/d6_2_e7_6v_alpha.log 2>&1 &

# ============================================================
# D6-3: Crop + 정확 PP (권장) - GPU 4,5
# ============================================================

# D6-3 E1: Paper baseline
CUDA_VISIBLE_DEVICES=4 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_3_E1_baseline.yaml > logs/d6_3_e1_baseline.log 2>&1 &

# D6-3 E2: GT mask
CUDA_VISIBLE_DEVICES=4 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_3_E2_gtmask.yaml > logs/d6_3_e2_gtmask.log 2>&1 &

# D6-3 E3: Alpha safe
CUDA_VISIBLE_DEVICES=4 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_3_E3_alpha.yaml > logs/d6_3_e3_alpha.log 2>&1 &

# D6-3 E4: 5v alpha
CUDA_VISIBLE_DEVICES=5 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_3_E4_5v_alpha.yaml > logs/d6_3_e4_5v_alpha.log 2>&1 &

# D6-3 E6: 5v opacity
CUDA_VISIBLE_DEVICES=5 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_3_E6_5v_opacity.yaml > logs/d6_3_e6_5v_opacity.log 2>&1 &

# D6-3 E7: 6v alpha
CUDA_VISIBLE_DEVICES=5 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_3_E7_6v_alpha.yaml > logs/d6_3_e7_6v_alpha.log 2>&1 &

echo "Started 18 D6 experiments"
