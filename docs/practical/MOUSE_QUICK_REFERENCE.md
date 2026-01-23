> **Navigation**: [← MoC](../00_MoC_INDEX.md) | [Practical](../) | [Theory](../theory/)

# FaceLift Mouse Extension - Quick Reference

> **Last Updated**: 2026-01-24 (v3.0)
> **Full Documentation**: [EXPERIMENT_REGISTRY](./experiments/EXPERIMENT_REGISTRY.md)

---

## Naming Convention v2.0 ★

```
{Dataset}_E{MaskMode}[_{SubNum}][_fixed]

E0: none (baseline)       │ E3: alpha_sup_only (LGM)
E1: gt (GT mask only)     │ E4: bg_penalty
E2: gt_alpha_sup ★ 권장   │ E5: composite
```

**핵심**: Random view selection이 기본 (suffix 없음), `_fixed`는 고정 뷰 시에만 사용

---

## Quick Start

### P0: 권장 실험 (GT + Alpha Supervision) ⭐

```bash
cd /home/joon/dev/FaceLift

# E2: GT Mask + Alpha Supervision (LGM + Pose Splatter)
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E2.yaml
```

### All Mask Modes (한눈에)

| GPU | Config | 명령어 |
|-----|--------|--------|
| 4 | `D7_1_E0.yaml` | Baseline (no mask) |
| 5 | **`D7_1_E2.yaml`** ⭐ | **GT + Alpha (권장)** |
| 6 | `D7_1_E3.yaml` | Alpha Only (LGM) |
| 7 | `D7_1_E4.yaml` | BG Penalty |

---

## 1. Training Commands

### 1.1 기본 형식

```bash
# 단일 Config 파일 (권장)
CUDA_VISIBLE_DEVICES={N} torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/{config}.yaml

# Background 실행
CUDA_VISIBLE_DEVICES={N} nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/{config}.yaml \
    > logs/{config}.log 2>&1 &
```

### 1.2 권장 실험 우선순위

| Priority | Config | 목적 | 명령어 |
|----------|--------|------|--------|
| **P0** ⭐ | `D7_1_E2.yaml` | GT + Alpha Supervision | `--config configs/mouse/D7_1_E2.yaml` |
| P1 | `D7_1_E5.yaml` | Composite Mask | `--config configs/mouse/D7_1_E5.yaml` |
| P2 | `D7_1_E4.yaml` | Background Penalty | `--config configs/mouse/D7_1_E4.yaml` |
| P3 | `D7_1_E3.yaml` | Alpha Only (LGM) | `--config configs/mouse/D7_1_E3.yaml` |
| P4 | `D7_1_E1.yaml` | GT Mask Only | `--config configs/mouse/D7_1_E1.yaml` |
| P5 | `D7_1_E0.yaml` | Baseline (no mask) | `--config configs/mouse/D7_1_E0.yaml` |

### 1.3 전체 마스크 비교 실험 (동시 실행)

```bash
cd /home/joon/dev/FaceLift

# GPU 4: E0 (Baseline)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E0.yaml \
    > logs/D7_1_E0.log 2>&1 &

# GPU 5: E2 (GT + Alpha) ★ 권장
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E2.yaml \
    > logs/D7_1_E2.log 2>&1 &

# GPU 6: E3 (Alpha Only)
CUDA_VISIBLE_DEVICES=6 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E3.yaml \
    > logs/D7_1_E3.log 2>&1 &

# GPU 7: E4 (BG Penalty)
CUDA_VISIBLE_DEVICES=7 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E4.yaml \
    > logs/D7_1_E4.log 2>&1 &
```

### 1.4 Variants (Fixed Views, 4 Views)

```bash
# Fixed View Selection
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E2_fixed.yaml

# 4 Views (Random)
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E2_4v.yaml
```

---

## 2. Dataset Quick Reference

### 2.1 권장 순위

| 순위 | Dataset | 용도 | 상태 |
|------|---------|------|------|
| **P0** ⭐ | **D7_1** | 표준 (기하학 정확) | ✅ 검증됨 |
| P1 | D8 | 최고 정밀도 (Homography) | ✅ 검증됨 |
| P2 | v13 | Legacy 비교용 | ⚠️ Ghosting |

---

## 3. E0-E5 상세

| Exp | Mode | 핵심 설정 | 문헌 |
|-----|------|-----------|------|
| **E0** | none | `mask_mode: none` | Paper |
| **E1** | gt | `mask_mode: gt` | - |
| **E2** ⭐ | gt + α | `mask_mode: gt, alpha_loss: 0.1` | LGM + Pose Splatter |
| **E3** | α only | `mask_mode: none, alpha_loss: 0.1` | LGM |
| **E4** | bg | `bg_loss: 0.5` | Obj-Centric 2DGS |
| **E5** | composite | `mask_mode: composite` | Nerfstudio |

### E2 권장 설정 (configs/experiments/E2_gt_alpha_sup.yaml)

```yaml
training:
  losses:
    mask_mode: gt              # GT mask로 RGB loss 제한
    normalize_by_mask: true    # Pose Splatter: 작은 전경 필수
    alpha_loss_weight: 0.1     # LGM: alpha supervision
    alpha_loss_type: mse       # BCE보다 안정적
```

---

## 4. Analysis Tools

### 4.1 Alpha Threshold 비교

```bash
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_alpha_thresholds \
    --checkpoint checkpoints/gslrm/D7_1_E2/ckpt_step_500.pt \
    --config configs/mouse/D7_1_E2.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --output_dir experiments/analysis/D7_1_E2
```

### 4.2 체크포인트 확인

```bash
# 최근 1시간 내 생성된 체크포인트
find checkpoints -name "*.pt" -mmin -60 | sort

# 특정 실험
ls -la checkpoints/gslrm/D7_1_E2/
```

---

## 5. Config 파일 위치

```
configs/
├── experiments/              # 실험 정의
│   ├── E0_baseline.yaml      # No mask
│   ├── E1_gt.yaml           # GT only
│   ├── E2_gt_alpha_sup.yaml ★ # GT + Alpha (권장)
│   ├── E3_alpha_sup_only.yaml # Alpha only
│   ├── E4_bg_penalty.yaml    # BG penalty
│   └── E5_composite.yaml     # Composite
│
├── datasets/                # 데이터셋 정의
│   ├── D7_1.yaml           ★ # 표준
│   ├── D8.yaml              # Homography
│   └── v13.yaml             # Legacy
│
└── mouse/                   # Combined configs
    ├── D7_1_E0.yaml         # Baseline
    ├── D7_1_E2.yaml        ★ # 권장
    ├── D7_1_E2_fixed.yaml   # Fixed views
    ├── D7_1_E2_4v.yaml      # 4 views
    └── ...
```

---

## 6. GPU Reference

| GPU | Model | 호환 | 권장 |
|-----|-------|------|------|
| 0-3 | RTX PRO 6000 Blackwell | ❌ | - |
| **4-7** | **RTX A6000** | **✅** | **사용** |

---

## 7. Troubleshooting

| 문제 | 원인 | 해결 |
|------|------|------|
| CUDA sm_120 오류 | Blackwell GPU | GPU 4-7 사용 |
| PSNR ~3 | 카메라 미정규화 | D7_1 또는 D8 사용 |
| fg_coverage > 0.3 | Alpha mask 확산 | mask_mode=gt 사용 |

---

## 8. Related Documents

- [EXPERIMENT_REGISTRY](./experiments/EXPERIMENT_REGISTRY.md) - 전체 실험 목록
- [EXPERIMENT_NAMING_CONVENTION](./experiments/EXPERIMENT_NAMING_CONVENTION.md) - 네이밍 규칙 상세
- [PREPROCESSING_REGISTRY](./datasets/PREPROCESSING_REGISTRY.md) - 데이터셋 상세
- [VSCode_Debug_Mask_Guide](../tutorials/VSCode_Debug_Mask_Guide.md) - 디버깅 가이드

---

*Quick Reference v3.0 | 2026-01-24*
