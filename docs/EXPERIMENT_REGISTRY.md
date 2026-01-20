# FaceLift Mouse Experiment Registry (SSOT)

> **Single Source of Truth** for all experiment configurations
> Last Updated: 2026-01-20

---

## 1. Dataset Registry (D_)

| ID | Description | PP | fx/fy | Split | Samples | Status |
|----|-------------|-----|-------|-------|---------|--------|
| D4 | triangulation + force_256 | 256 (bug) | 549/557 | random | 3240/360 | ⛔ Deprecated |
| D7 | PP-centered shift | 256 | 549/549 | random 9:1 | 3237/360 | ○ 빠른 실험 |
| **D7_1** | individual scale | 256 | **549/549** | random 9:1 | 3237/360 | **★ 권장** |
| D7_2 | average scale | 256 | ~548/550 | random 9:1 | 3237/360 | ○ 이미지 품질 |
| **D7_t** | D7 temporal split | 256 | 549/549 | **1:1:1** | 1222/1187/1188 | **★ 일반화** |

### Dataset 선택 가이드
- **빠른 실험**: D7_1 (기하학 완벽)
- **일반화 검증**: D7_t (temporal split)
- **최종 평가**: D7_t test set

---

## 2. Experiment Series (E_)

### E1: Baseline (논문 재현)

| ID | Views | Random | Mask | Description |
|----|-------|--------|------|-------------|
| E1_1_paper_random | 4 | ✅ | none | 원본 논문 설정 |
| E1_2_paper_fixed | 4 | ❌ | none | View 고정 (대조군) |

### E2: Mask Mode Ablation

| ID | Views | Mask | Description |
|----|-------|------|-------------|
| E2_1_rgb_mask | 4 | rgb_pred | RGB 거리 기반 |
| E2_2_gt_mask | 4 | gt | GT mask 사용 (Oracle) |
| E2_3_alpha_mask | 4 | alpha | Rendered alpha |
| E2_4_4v_alpha_random | 4 | alpha + random | Alpha + random view |

### E3: View Count Ablation

| ID | Views | Mask | Random | Description |
|----|-------|------|--------|-------------|
| E3_2_5v_alpha | 5 | alpha | ❌ | 5 input views |
| E3_3_5v_alpha_random | 5 | alpha | ✅ | 5v + random |
| E3_4_6v_alpha_random | 6 | alpha | ✅ | 6v + random |

### E4: Alpha Loss Addition

| ID | Views | α Loss | Description |
|----|-------|--------|-------------|
| E4_2_5v_alpha_loss | 5 | 0.1 | Alpha mask + BCE loss |
| E4_3_5v_alpha_loss_random | 5 | 0.1 + random | + random view |
| E4_4_6v_alpha_loss_random | 6 | 0.1 + random | 6v + loss + random |

### E5: Alpha Parameter Tuning ★ NEW

| ID | Views | Threshold | α Loss | Opacity Reg | Description |
|----|-------|-----------|--------|-------------|-------------|
| E5_1_5v_alpha_random | 5 | 0.5 | 0.0 | - | Random baseline |
| **E5_2_5v_alpha_conservative** | 5 | **0.7** | **0.3** | - | 배경 침범 방지 |
| **E5_3_5v_alpha_aggressive** | 5 | 0.5 | **0.5** | **0.01** | 형상 우선 |
| **E5_4_4v_alpha_conservative** | 4 | 0.7 | 0.3 | - | 4v conservative |

---

## 3. Full Config Matrix

### D7_1 Configs

| Config | Views | Random | Mask | Threshold | α Loss | Reg |
|--------|-------|--------|------|-----------|--------|-----|
| D7_1_E1_1_paper_random | 4 | ✅ | none | - | - | - |
| D7_1_E1_2_paper_fixed | 4 | ❌ | none | - | - | - | (대조군)
| D7_1_E2_2_gt_mask | 4 | ❌ | gt | - | - | - |
| D7_1_E2_3_alpha_mask | 4 | ❌ | alpha | 0.5 | - | - |
| D7_1_E3_2_5v_alpha | 5 | ❌ | alpha | 0.5 | - | - |
| D7_1_E4_2_5v_alpha_loss | 5 | ❌ | alpha | 0.5 | 0.1 | - |
| D7_1_E5_1_5v_alpha_random | 5 | ✅ | alpha | 0.5 | - | - |
| **D7_1_E5_2_5v_alpha_conservative** | 5 | ❌ | alpha | **0.7** | **0.3** | - |
| **D7_1_E5_4_4v_alpha_conservative** | 4 | ❌ | alpha | **0.7** | **0.3** | - |

### D7_t Configs

| Config | Views | Random | Mask | Threshold | α Loss | Reg |
|--------|-------|--------|------|-----------|--------|-----|
| D7_t_E1_1_paper_random | 4 | ✅ | none | - | - | - |
| D7_t_E1_2_paper_fixed | 4 | ❌ | none | - | - | - | (대조군)
| D7_t_E2_1_rgb_mask | 4 | ❌ | rgb_pred | - | - | - |
| D7_t_E2_2_gt_mask | 4 | ❌ | gt | - | - | - |
| D7_t_E2_3_alpha_mask | 4 | ❌ | alpha | 0.5 | - | - |
| D7_t_E2_4_4v_alpha_random | 4 | ✅ | alpha | 0.5 | - | - |
| D7_t_E3_2_5v_alpha | 5 | ❌ | alpha | 0.5 | - | - |
| D7_t_E3_3_5v_alpha_random | 5 | ✅ | alpha | 0.5 | - | - |
| D7_t_E3_4_6v_alpha_random | 6 | ✅ | alpha | 0.5 | - | - |
| D7_t_E4_2_5v_alpha_loss | 5 | ❌ | alpha | 0.5 | 0.1 | - |
| D7_t_E4_3_5v_alpha_loss_random | 5 | ✅ | alpha | 0.5 | 0.1 | - |
| D7_t_E4_4_6v_alpha_loss_random | 6 | ✅ | alpha | 0.5 | 0.1 | - |
| D7_t_E5_1_5v_alpha_random | 5 | ✅ | alpha | 0.5 | - | - |
| **D7_t_E5_2_5v_alpha_conservative** | 5 | ❌ | alpha | **0.7** | **0.3** | - |
| **D7_t_E5_3_5v_alpha_aggressive** | 5 | ❌ | alpha | 0.5 | **0.5** | **0.01** |
| **D7_t_E5_4_4v_alpha_conservative** | 4 | ❌ | alpha | **0.7** | **0.3** | - |

---

## 4. Hypothesis & Control Pairs

### H1: Random View Selection
| Experiment | Control | Variable | Expected |
|------------|---------|----------|----------|
| E1_1 (random) | E1_2 (fixed) | random_view_selection | E1_1 > E1_2 |

### H2: Mask Mode
| Experiment | Control | Variable | Expected |
|------------|---------|----------|----------|
| E2_3 (alpha) | E2_2 (gt) | mask_mode | E2_2 ≥ E2_3 (Oracle) |
| E2_3 (alpha) | E2_1 (rgb) | mask_mode | E2_3 > E2_1 |

### H3: View Count
| Experiment | Control | Variable | Expected |
|------------|---------|----------|----------|
| E3_2 (5v) | E2_3 (4v) | num_input_views | E3_2 > E2_3 |
| E5_2 (5v) | E5_4 (4v) | num_input_views | E5_2 > E5_4 |

### H4: Alpha Loss
| Experiment | Control | Variable | Expected |
|------------|---------|----------|----------|
| E4_2 (loss) | E3_2 (no loss) | alpha_loss_weight | E4_2 ≥ E3_2 |

### H5: Alpha Tuning (배경 침범 해결)
| Experiment | Control | Variable | Expected |
|------------|---------|----------|----------|
| **E5_2** (conservative) | E4_2 | threshold 0.7, loss 0.3 | 배경 침범 ↓ |
| **E5_3** (aggressive) | E4_2 | loss 0.5, opacity_reg | floater ↓ |

---

## 5. Execution Commands

### 새 모듈화 CLI (권장)
```bash
# Format: train_gslrm.py -d {DATASET} -e {EXPERIMENT}
torchrun --standalone --nproc_per_node=1 train_gslrm.py -d D7_1 -e E5_2_5v_alpha_conservative
```

### 백그라운드 실행 템플릿
```bash
CUDA_VISIBLE_DEVICES={GPU} nohup torchrun --standalone --nproc_per_node=1 \
train_gslrm.py -d {DATASET} -e {EXPERIMENT} \
> logs/{dataset}_{experiment}.log 2>&1 &
```

### Priority 1: Alpha Tuning (즉시 실행)
```bash
# Conservative (배경 침범 방지)
CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nproc_per_node=1 \
train_gslrm.py -d D7_1 -e E5_2_5v_alpha_conservative > logs/d7_1_e5_2.log 2>&1 &

# Aggressive (형상 우선)
CUDA_VISIBLE_DEVICES=1 nohup torchrun --standalone --nproc_per_node=1 \
train_gslrm.py -d D7_1 -e E5_3_5v_alpha_aggressive > logs/d7_1_e5_3.log 2>&1 &

# 4v Conservative
CUDA_VISIBLE_DEVICES=2 nohup torchrun --standalone --nproc_per_node=1 \
train_gslrm.py -d D7_1 -e E5_4_4v_alpha_conservative > logs/d7_1_e5_4.log 2>&1 &
```

### Priority 2: D7_t 검증
```bash
CUDA_VISIBLE_DEVICES=3 nohup torchrun --standalone --nproc_per_node=1 \
train_gslrm.py -d D7_t -e E5_2_5v_alpha_conservative > logs/d7_t_e5_2.log 2>&1 &
```

---

## 6. Success Metrics

| Metric | Baseline (E4_2) | Target (E5_2) | Meaning |
|--------|-----------------|---------------|---------|
| mask_iou | 0.06-0.08 | **0.5+** | 형상 정확도 |
| val/psnr | ~19 | **20+** | 렌더링 품질 |
| train/val 일치 | ❌ | ✅ | 과적합 방지 |

### Qualitative
- Supervision 이미지: 전경/배경 분리
- Novel view: floater 없음
- Alpha 분포: Input/Novel 일관성

---

## 7. File Locations

```
/home/joon/dev/FaceLift/
├── configs/mouse/
│   ├── D7_1_E*.yaml       # D7_1 실험 설정
│   └── D7_t_E*.yaml       # D7_t 실험 설정
├── checkpoints/gslrm/     # 체크포인트
├── logs/                  # 학습 로그
├── experiments/validation/ # Validation 결과
└── docs/
    └── EXPERIMENT_REGISTRY.md  # ★ THIS FILE (SSOT)
```

---

## 8. Quick Reference

### Dataset 선택
```
빠른 실험 → D7_1
일반화 검증 → D7_t
```

### Experiment 선택
```
Baseline → E1_1 (논문 원본), E1_2 (대조군)
Mask 비교 → E2_2 vs E2_3
View 비교 → E5_4 (4v) vs E5_2 (5v)
배경 침범 → E5_2 (conservative)
형상 우선 → E5_3 (aggressive)
```

### Alpha Tuning Quick Reference
| 문제 | threshold | α loss | opacity_reg |
|------|-----------|--------|-------------|
| 배경→전경 | **↑ 0.7** | **↑ 0.3** | - |
| Floater | - | **↑ 0.5** | **↑ 0.01** |
| 가장자리 손실 | **↓ 0.5** | 유지 | - |

---

*SSOT: Single Source of Truth*
*Maintained by: Claude Code | FaceLift Mouse Project*
*Last Updated: 2026-01-20*
