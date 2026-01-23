> **Navigation**: [← MoC](../00_MoC_INDEX.md) | [Practical](../) | [Theory](../theory/)

# FaceLift Mouse Extension - Quick Reference

> **Last Updated**: 2026-01-24
> **Full Documentation**: [EXPERIMENT_REGISTRY](./experiments/EXPERIMENT_REGISTRY.md)

---

## Quick Start

### P0: 문헌 기반 마스크 실험 (권장) ⭐

```bash
cd /home/joon/dev/FaceLift

# GT Mask + Alpha Supervision (LGM + Pose Splatter)
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/mask_exp/D7_mask_E1_gt_alpha_sup.yaml
```

### P1: Paper Baseline (빠른 테스트)

```bash
# 500 steps (~10분)
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E_quick_alpha
```

---

## 1. Training Commands (정리)

### 1.1 기본 형식

```bash
# 형식 1: Dataset + Experiment 조합
CUDA_VISIBLE_DEVICES={N} torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d {Dataset} -e {Experiment}

# 형식 2: 단일 Config 파일
CUDA_VISIBLE_DEVICES={N} torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config {config_path}

# Background 실행
CUDA_VISIBLE_DEVICES={N} nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d {Dataset} -e {Experiment} \
    > logs/{dataset}_{experiment}.log 2>&1 &
```

### 1.2 권장 실험 명령어

| Priority | 명령어 | 목적 |
|----------|--------|------|
| **P0** | `-d D7_1 --config configs/mouse/mask_exp/D7_mask_E1_gt_alpha_sup.yaml` | 문헌 기반 ⭐ |
| **P1** | `-d D7_1 -e E1_1_paper_random` | Paper baseline |
| P2 | `-d D8 -e E1_1_paper_random` | Homography 검증 |
| P3 | `-d v13 -e E1_1_paper_random` | Legacy 비교 |
| P4 | `-d D4 -e E1_1_paper_random` | PP=256 강제 비교 |

### 1.3 Dataset 비교 실험 (동시 실행)

```bash
# GPU 4: D7_1 (표준)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E1_1_paper_random \
    > logs/d7_1_e1_1_paper_random.log 2>&1 &

# GPU 5: D8 (Homography)
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E1_1_paper_random \
    > logs/d8_e1_1_paper_random.log 2>&1 &

# GPU 6: v13 (Legacy 비교)
CUDA_VISIBLE_DEVICES=6 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d v13 -e E1_1_paper_random \
    > logs/v13_e1_1_paper_random.log 2>&1 &

# GPU 7: D4 (PP=256 강제 비교)
CUDA_VISIBLE_DEVICES=7 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D4 -e E1_1_paper_random \
    > logs/d4_e1_1_paper_random.log 2>&1 &
```

### 1.4 Mask 실험 시리즈

```bash
# E6: 문헌 기반 마스크 (configs/mouse/mask_exp/)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/mask_exp/D7_mask_E1_gt_alpha_sup.yaml \
    > logs/d7_mask_e1_gt_alpha_sup.log 2>&1 &

# E2: GT vs Alpha 비교
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E2_1_gt_mask_random \
    > logs/d7_1_e2_1_gt_mask_random.log 2>&1 &

# E5: Alpha Threshold 0.7
CUDA_VISIBLE_DEVICES=6 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E5_6_alpha_thresh_07 \
    > logs/d7_1_e5_6_alpha_thresh_07.log 2>&1 &
```

---

## 2. Dataset Quick Reference

### 2.1 권장 순위

| 순위 | Dataset | 용도 | PP | 상태 |
|------|---------|------|-----|------|
| **P0** ⭐ | **D7_1** | 표준 (기하학 정확) | 256 shift | ✅ 검증됨 |
| **P1** | **D8** | 최고 정밀도 (Homography) | 256 shift | ✅ 검증됨 |
| P2 | v13 | Legacy 비교용 | 256 강제 | ⚠️ Ghosting |
| P3 | D4 | PP=256 강제 비교용 | 256 강제 | ⚠️ Ray error |

### 2.2 Dataset 카테고리

```
Cat.1: LEGACY (기하학 무시)
└── v13: Simple resize, PP=256 강제 → Ray error ~5-13°

Cat.2: OBJECT-CENTERED (Crop 기반)
└── D4: Triangulation + Crop, PP=256 강제 → 뷰 불일치

Cat.3: PP-CENTERED SHIFT (Affine) ★
├── D7_1: individual scale → ~0° error ★ 표준
└── D7_2: average scale → ~0.2° error

Cat.4: PRECISION HOMOGRAPHY (Skew 보정) ★★
├── D8: H = K_target @ K^-1 → ~0° error ★★ 권장
└── D8_1: D8 + 1.3x zoom → cx/cy varies
```

→ 상세: [PREPROCESSING_REGISTRY](./datasets/PREPROCESSING_REGISTRY.md)

---

## 3. Experiment Quick Reference

### 3.1 실험 시리즈 개요

| Series | 변수 | 핵심 Config | 설명 |
|--------|------|-------------|------|
| **E1** | Paper Baseline | E1_1_paper_random | No mask, 4v, random |
| **E2** | Mask Mode | E2_1_gt_mask_random | GT vs Alpha |
| **E3** | View Count | E3_2_5v_alpha | 4v/5v/6v |
| **E4** | Alpha Tuning | E4_2_alpha_conservative | threshold, loss, reg |
| **E5** | Loss Ablation | E5_6_alpha_thresh_07 | alpha_loss, thresh |
| **E6** | Literature Mask ⭐ | D7_mask_E1_gt_alpha_sup | LGM+Pose Splatter |

### 3.2 문헌 기반 마스크 (E6) ⭐

| Config | Mode | Alpha Loss | 문헌 | Priority |
|--------|------|------------|------|----------|
| **D7_mask_E1_gt_alpha_sup** | gt | 0.1 MSE | LGM+Pose Splatter | **P0** ⭐ |
| D7_mask_E2_composite | composite | 0.05 MSE | Nerfstudio | P1 |
| D7_mask_E3_bg_penalty | none+bg | 0.1+0.5 | Obj-Centric 2DGS | P2 |
| D7_mask_E4_alpha_sup_only | none | 0.1 MSE | LGM | P3 |

```yaml
# E6 권장 설정 (D7_mask_E1_gt_alpha_sup)
losses:
  mask_mode: gt              # GT mask로 RGB loss 제한
  normalize_by_mask: true    # Pose Splatter: 작은 전경 필수
  alpha_loss_weight: 0.1     # LGM: alpha supervision
  alpha_loss_type: mse       # BCE보다 안정적
```

→ 상세: [EXPERIMENT_REGISTRY](./experiments/EXPERIMENT_REGISTRY.md)

---

## 4. Analysis Tools

### 4.1 Alpha Threshold 비교

```bash
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_alpha_thresholds \
    --checkpoint checkpoints/gslrm/{exp_name}/ckpt_step_500.pt \
    --config configs/mouse/{config}.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --output_dir alpha_threshold_comparison/{exp_name} \
    --thresholds 0.3 0.5 0.7 0.9 0.99
```

### 4.2 Rendered Alpha 분석

```bash
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_rendered_alpha \
    --checkpoint checkpoints/gslrm/{exp_name}/ckpt_step_500.pt \
    --config configs/mouse/{config}.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --output_dir alpha_analysis/{exp_name} \
    --sample_idx 0
```

### 4.3 체크포인트 확인

```bash
# 최근 1시간 내 생성된 체크포인트
find checkpoints -name '*.pt' -mmin -60 | sort

# 특정 실험
ls -la checkpoints/gslrm/D7_1_*/
```

---

## 5. GPU Reference

| GPU Index | Model | 호환 | 권장 |
|-----------|-------|------|------|
| 0-3 | RTX PRO 6000 Blackwell | ❌ | - |
| **4-7** | **RTX A6000** | **✅** | **사용** |

---

## 6. Config 파일 위치

```
configs/
├── datasets/                    # -d 플래그
│   ├── v13.yaml                 # Legacy
│   ├── D4.yaml                  # PP=256 강제
│   ├── D7_1.yaml                # ★ 표준
│   ├── D8.yaml                  # ★★ Homography
│   └── ...
│
├── experiments/                 # -e 플래그
│   ├── E1_*.yaml               # Paper baseline
│   ├── E2_*.yaml               # Mask mode
│   ├── E3_*.yaml               # View count
│   ├── E4_*.yaml               # Alpha tuning
│   ├── E5_*.yaml               # Loss ablation
│   └── E_quick_alpha.yaml      # Quick test
│
└── mouse/mask_exp/              # --config 플래그 (E6)
    ├── D7_mask_E1_gt_alpha_sup.yaml  # ★ P0
    └── ...
```

---

## 7. Troubleshooting

| 문제 | 원인 | 해결 |
|------|------|------|
| CUDA sm_120 오류 | Blackwell GPU | GPU 4-7 사용 |
| PSNR ~3 | 카메라 미정규화 | D7_1 또는 D8 사용 |
| fg_coverage > 0.3 | Alpha mask 확산 | mask_mode=gt 사용 |
| 체크포인트 못찾음 | 경로 다름 | `find checkpoints` |

---

## 8. Related Documents

- [EXPERIMENT_REGISTRY](./experiments/EXPERIMENT_REGISTRY.md) - 전체 실험 목록
- [PREPROCESSING_REGISTRY](./datasets/PREPROCESSING_REGISTRY.md) - 데이터셋 상세
- [Mask_Literature_Review](../theory/mask/Mask_Literature_Review.md) - 마스크 문헌 조사
- [Mask_Experiment_Priority](./experiments/Mask_Experiment_Priority.md) - 마스크 실험 우선순위

---

*Quick Reference v2.0 | 2026-01-24*
