# FaceLift Config System

> Last Updated: 2026-01-23

## Quick Start

```bash
# 권장 실험 실행
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E2_gt_alpha.yaml

# Config 검증
python scripts/validate_config.py --all
```

## Directory Structure

```
configs/
├── README.md              # 이 문서
├── gslrm.yaml             # Base config (원본, 수정 금지)
├── gslrm_mouse.yaml       # Mouse 실험용 base (modular mode)
│
├── base/                  # Modular mode base configs
│   └── gslrm_mouse.yaml
│
├── datasets/              # 데이터셋 정의
│   ├── D7_1.yaml          # ⭐ 권장 (Individual scale)
│   ├── D8.yaml            # Homography
│   └── ...
│
├── experiments/           # 실험 정의 (마스크 모드별)
│   ├── E0_none.yaml       # Baseline
│   ├── E1_gt.yaml         # GT mask only
│   ├── E2_gt_alpha.yaml   # ⭐ 권장 (GT + alpha)
│   └── ...
│
└── mouse/                 # Combined configs (standalone)
    ├── D7_1_E2_gt_alpha.yaml  # ⭐ 권장
    └── ...
```

## Config Modes

### 1. Standalone Mode (권장)

단일 파일에 모든 설정 포함. 실행이 간단하고 명확함.

```bash
torchrun ... --config configs/mouse/D7_1_E2_gt_alpha.yaml
```

### 2. Modular Mode

Base + Dataset + Experiment 조합. 실험 조합이 많을 때 유용.

```bash
torchrun ... --base configs/base/gslrm_mouse.yaml \
             --dataset configs/datasets/D7_1.yaml \
             --experiment configs/experiments/E2_gt_alpha.yaml
```

## ⚠️ 중요: Config 구조 규칙

```yaml
# ✅ 올바른 구조
model:
  num_views: 6           # 여기!
  num_input_views: 5     # 여기!
  image_tokenizer: ...

training:
  dataset:
    data_list: ...       # 데이터 경로만
    random_view_selection: true
  losses:
    mask_mode: gt
    alpha_loss_weight: 0.1

# ❌ 잘못된 구조 (이전 실수)
training:
  dataset:
    num_views: 6         # 여기 넣으면 안됨!
    num_input_views: 5   # 여기 넣으면 안됨!
```

## Validation

새 config 작성 후 반드시 검증:

```bash
python scripts/validate_config.py configs/mouse/my_new_config.yaml
```

## See Also

- [CONFIG_SCHEMA.md](../docs/practical/config/CONFIG_SCHEMA.md) - 상세 스키마
- [EXPERIMENT_NAMING_CONVENTION.md](../docs/practical/experiments/EXPERIMENT_NAMING_CONVENTION.md) - 네이밍 규칙
