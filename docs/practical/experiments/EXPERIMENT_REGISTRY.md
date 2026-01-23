> **Navigation**: [← MoC](../../00_MoC_INDEX.md) | [Practical](../) | [Experiments](./)

# FaceLift Experiment Registry

> **Version**: 3.0.0 (2026-01-24)
> **Single Source of Truth** for all experiment configurations
> **New Schema**: E0-E5 maps directly to mask modes

---

## Naming Convention v2.0

```
{Dataset}_E{MaskMode}[_{SubNum}][_fixed]
    │       │           │         │
    │       │           │         └── _fixed: 고정 뷰 선택 (Random이 기본)
    │       │           └── 선택적 서브넘버 (4v, 5v 등)
    │       └── E0-E5: 마스크 모드 (아래 표 참조)
    └── D7_1, D8, v13 등

예시:
  D7_1_E2        = D7.1 데이터 + GT+Alpha (권장) + Random views
  D7_1_E2_fixed  = D7.1 데이터 + GT+Alpha + Fixed views
  D7_1_E2_4v     = D7.1 데이터 + GT+Alpha + 4 views
```

### E0-E5 마스크 모드 매핑 ★

| Exp | Mask Mode | Alpha Loss | BG Loss | 문헌 | Priority |
|-----|-----------|------------|---------|------|----------|
| **E0** | none | - | - | Paper Baseline | P5 |
| **E1** | gt | - | - | GT Mask Only | P4 |
| **E2** ⭐ | **gt + α_sup** | **0.1 MSE** | - | **LGM + Pose Splatter** | **P0** |
| **E3** | α_sup only | 0.1 MSE | - | LGM | P3 |
| **E4** | bg_penalty | 0.1 MSE | 0.5 | Object-Centric 2DGS | P2 |
| **E5** | composite | 0.05 MSE | - | Nerfstudio | P1 |

**핵심 원칙**:
- **Random view selection이 기본** (suffix 없음)
- `_fixed` suffix는 고정 뷰 선택 시에만 사용
- `_4v`, `_5v` 등 뷰 수 지정 가능

---

## Quick Reference

### 권장 실험 조합

| Priority | 명령어 | 목적 |
|----------|--------|------|
| **P0** ⭐ | `--config configs/mouse/D7_1_E2.yaml` | GT + Alpha Supervision |
| **P1** | `--config configs/mouse/D7_1_E5.yaml` | Composite Mask |
| **P2** | `--config configs/mouse/D7_1_E4.yaml` | Background Penalty |
| **P3** | `--config configs/mouse/D7_1_E3.yaml` | Alpha Supervision Only |
| **P4** | `--config configs/mouse/D7_1_E1.yaml` | GT Mask Only |
| **P5** | `--config configs/mouse/D7_1_E0.yaml` | No Mask (Baseline) |

---

## Config Files

### Experiment Configs (configs/experiments/)

| File | Mode | 설명 |
|------|------|------|
| `E0_baseline.yaml` | none | 마스크 없음 (Paper baseline) |
| `E1_gt.yaml` | gt | GT 마스크만 |
| **`E2_gt_alpha_sup.yaml`** ⭐ | gt + α | **GT + Alpha Supervision (권장)** |
| `E3_alpha_sup_only.yaml` | α only | Alpha Supervision만 (LGM) |
| `E4_bg_penalty.yaml` | bg | Background Penalty |
| `E5_composite.yaml` | composite | Composite Mask |

### Combined Configs (configs/mouse/)

| File | Dataset | Experiment | Views | Selection |
|------|---------|------------|-------|-----------|
| `D7_1_E0.yaml` | D7_1 | Baseline | 5 | Random |
| `D7_1_E1.yaml` | D7_1 | GT Only | 5 | Random |
| **`D7_1_E2.yaml`** ⭐ | D7_1 | **GT + Alpha** | 5 | **Random** |
| `D7_1_E2_fixed.yaml` | D7_1 | GT + Alpha | 5 | Fixed |
| `D7_1_E2_4v.yaml` | D7_1 | GT + Alpha | 4 | Random |
| `D7_1_E3.yaml` | D7_1 | Alpha Only | 5 | Random |
| `D7_1_E4.yaml` | D7_1 | BG Penalty | 5 | Random |
| `D7_1_E5.yaml` | D7_1 | Composite | 5 | Random |

---

## Experiment Details

### E0: Baseline (No Mask)

**용도**: Paper 설정 재현, 마스크 효과 비교 기준

```yaml
losses:
  mask_mode: none
  alpha_loss_weight: 0.0
  bg_loss_weight: 0.0
```

### E1: GT Mask Only

**용도**: 마스킹만 적용, alpha loss 없음

```yaml
losses:
  mask_mode: gt
  normalize_by_mask: true
  alpha_loss_weight: 0.0
```

### E2: GT + Alpha Supervision ⭐ (권장)

**문헌**: LGM (ECCV 2024) + Pose Splatter (NeurIPS 2025)

```yaml
losses:
  mask_mode: gt
  normalize_by_mask: true    # Pose Splatter: 작은 전경 필수
  alpha_loss_weight: 0.1     # LGM: alpha supervision
  alpha_loss_type: mse       # BCE보다 안정적
```

**왜 E2가 권장인가?**
1. LGM: "α_sup enables faster convergence of the shape"
2. Pose Splatter: normalize_by_mask로 작은 객체 편향 방지
3. GT 마스크로 안정적인 RGB loss 계산

### E3: Alpha Supervision Only (LGM)

**문헌**: LGM (ECCV 2024)

```yaml
losses:
  mask_mode: none           # RGB loss on full image
  alpha_loss_weight: 0.1    # Shape supervision만
```

**용도**: RGB 마스킹 없이 alpha supervision만의 효과 측정

### E4: Background Penalty

**문헌**: Object-Centric 2DGS

```yaml
losses:
  mask_mode: none
  alpha_loss_weight: 0.1
  bg_loss_weight: 0.5      # Background penalty
```

**용도**: 배경 억제를 통한 전경 집중

### E5: Composite Mask

**문헌**: Nerfstudio

```yaml
losses:
  mask_mode: composite      # α * rendered + (1-α) * bg
  normalize_by_mask: false  # 전체 이미지 정규화
  alpha_loss_weight: 0.05
```

**용도**: 더 부드러운 전경-배경 블렌딩

---

## Legacy Config Migration

### 기존 → 새 명명 규칙

| Legacy Name | New Name | 비고 |
|-------------|----------|------|
| `D7_mask_E1_gt_alpha_sup.yaml` | `D7_1_E2.yaml` | ★ 권장 |
| `D7_mask_E0_baseline.yaml` | `D7_1_E0.yaml` | |
| `D7_mask_E2_composite.yaml` | `D7_1_E5.yaml` | E5로 이동 |
| `D7_mask_E3_bg_penalty.yaml` | `D7_1_E4.yaml` | E4로 이동 |
| `D7_mask_E4_alpha_sup_only.yaml` | `D7_1_E3.yaml` | E3로 이동 |
| `E1_1_paper_random.yaml` | `E0_baseline.yaml` | |
| `E2_1_gt_mask_random.yaml` | `E1_gt.yaml` | |

### 아카이브된 설정

기존 세부 넘버링 configs (`E1_1`, `E2_1`, `E4_7` 등)은 `configs/experiments/_archive/`로 이동

---

## Running Experiments

### Basic Commands

```bash
cd /home/joon/dev/FaceLift

# P0: 권장 (GT + Alpha Supervision)
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E2.yaml

# P0 with Fixed Views
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E2_fixed.yaml

# Background Execution
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E2.yaml \
    > logs/D7_1_E2.log 2>&1 &
```

### Full Comparison (All Mask Modes)

```bash
# GPU 4: E0 (Baseline)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E0.yaml \
    > logs/D7_1_E0.log 2>&1 &

# GPU 5: E2 (GT + Alpha) ★
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

---

## Related Documents

- [MOUSE_QUICK_REFERENCE](../MOUSE_QUICK_REFERENCE.md) - 명령어 빠른 참조
- [EXPERIMENT_NAMING_CONVENTION](./EXPERIMENT_NAMING_CONVENTION.md) - 네이밍 규칙 상세
- [Mask_Literature_Review](../../theory/mask/Mask_Literature_Review.md) - 마스크 문헌 조사
- [PREPROCESSING_REGISTRY](../datasets/PREPROCESSING_REGISTRY.md) - 데이터셋 레지스트리

---

*Experiment Registry v3.0.0 | 2026-01-24*
