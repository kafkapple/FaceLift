#!/bin/bash
# D6-3 Priority Experiments (2026-01-18)
# ========================================
# D6-3: Crop + 정확 PP (권장 데이터셋)
#
# 우선순위:
# 1. E4_5v_alpha - D4 best 설정 재현
# 2. E6_5v_opacity - + opacity regularization
# 3. E1_baseline - Paper baseline (no mask)
# 4. E2_gtmask - GT mask 비교
# 5. E3_alpha_safe - 안전한 alpha (threshold 0.3)
# 6. E7_6v_alpha - 전체 뷰 테스트

cd /home/joon/dev/FaceLift

# ============================================
# Priority 1: E4_5v_alpha (★ D4 best 설정)
# ============================================
CUDA_VISIBLE_DEVICES=0 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_3_E4_5v_alpha.yaml > logs/d6_3_e4_5v_alpha.log 2>&1 &

# ============================================
# Priority 2: E6_5v_opacity (+ opacity reg)
# ============================================
CUDA_VISIBLE_DEVICES=1 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_3_E6_5v_opacity.yaml > logs/d6_3_e6_5v_opacity.log 2>&1 &

# ============================================
# Priority 3: E1_baseline (Paper baseline)
# ============================================
CUDA_VISIBLE_DEVICES=2 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_3_E1_baseline.yaml > logs/d6_3_e1_baseline.log 2>&1 &

# ============================================
# Priority 4: E2_gtmask (GT mask)
# ============================================
CUDA_VISIBLE_DEVICES=3 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_3_E2_gtmask.yaml > logs/d6_3_e2_gtmask.log 2>&1 &

# ============================================
# Priority 5: E3_alpha_safe (안전한 alpha)
# ============================================
CUDA_VISIBLE_DEVICES=4 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_3_E3_alpha.yaml > logs/d6_3_e3_alpha.log 2>&1 &

# ============================================
# Priority 6: E7_6v_alpha (전체 뷰)
# ============================================
CUDA_VISIBLE_DEVICES=5 nohup torchrun \
    --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_3_E7_6v_alpha.yaml > logs/d6_3_e7_6v_alpha.log 2>&1 &

echo Started 6 D6-3 priority experiments
