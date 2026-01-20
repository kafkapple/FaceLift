# Modular Config System

모듈화된 설정 시스템으로, 데이터셋과 실험 설정을 독립적으로 관리하고 자유롭게 조합할 수 있습니다.

---

## 개요

기존 방식은 데이터셋 × 실험 조합마다 개별 설정 파일이 필요했습니다:
- 7 데이터셋 × 8 실험 = 56개 파일 (90% 중복)

모듈화 방식은 3-Layer 구조로 중복을 제거합니다:
- Base (1) + Dataset (N) + Experiment (M) = 1 + N + M 파일

---

## 디렉토리 구조

```
configs/
├── base/
│   └── gslrm_mouse.yaml       # 공통 설정 (모델, 옵티마이저 등)
│
├── datasets/
│   ├── D7_1.yaml              # D7_1 데이터셋 경로
│   ├── D7_2.yaml              # D7_2 데이터셋 경로
│   └── D7_t.yaml              # D7_t (temporal split) 경로
│
├── experiments/
│   ├── E1_1_paper_random.yaml # 4v, random, no mask
│   ├── E1_2_paper_fixed.yaml  # 4v, fixed, no mask
│   ├── E2_2_gt_mask.yaml      # 4v, gt mask
│   ├── E2_3_alpha_mask.yaml   # 4v, alpha mask
│   ├── E3_2_5v_alpha.yaml     # 5v, alpha mask
│   ├── E4_2_5v_alpha_loss.yaml# 5v, alpha mask + alpha loss
│   └── E5_1_5v_alpha_random.yaml # 5v, random, alpha mask
│
└── mouse/                      # Legacy 단일 설정 (호환용)
```

---

## 사용법

### 모듈화 모드 (권장)

```bash
# 기본 사용법
python train_gslrm.py --dataset D7_1 --experiment E3_2_5v_alpha

# 단축 옵션
python train_gslrm.py -d D7_1 -e E3_2_5v_alpha

# torchrun으로 실행
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_gslrm.py \
    -d D7_1 -e E3_2_5v_alpha
```

### Legacy 모드 (호환)

```bash
# 기존 단일 설정 파일 사용
python train_gslrm.py --config configs/mouse/D7_1_E3_2_5v_alpha.yaml
```

### Override 옵션

두 모드 모두 `--set` 옵션으로 설정 오버라이드 가능:

```bash
# 학습률 변경
python train_gslrm.py -d D7_1 -e E3_2_5v_alpha \
    --set training.optimizer.lr 5e-7

# 배치 크기 변경
python train_gslrm.py -d D7_1 -e E3_2_5v_alpha \
    --set training.dataloader.batch_size_per_gpu 4
```

---

## Config 계층

### 1. Base Config (`configs/base/gslrm_mouse.yaml`)

모든 실험에 공통인 설정:
- 모델 아키텍처 (transformer, gaussians)
- 런타임 설정 (AMP, TF32)
- 옵티마이저 기본값
- 스케줄러 기본값

### 2. Dataset Config (`configs/datasets/`)

데이터셋별 경로:
- training.dataset.dataset_path
- validation.dataset_path

### 3. Experiment Config (`configs/experiments/`)

실험별 파라미터:
- num_views, num_input_views
- random_view_selection
- mask_mode, alpha_loss_weight
- 기타 실험 변수

---

## 자동 생성되는 값

모듈화 모드에서 다음 값들이 자동 생성됩니다:

| 설정 | 자동 생성 값 |
|------|-------------|
| `checkpoint_dir` | `checkpoints/gslrm/{dataset}_{experiment}` |
| `wandb.group` | `{dataset}` |
| `wandb.exp_name` | `{dataset}_{experiment}` |

---

## 실험 스키마

### E1: Baseline (H1)
| ID | Views | Selection | Mask | Description |
|----|-------|-----------|------|-------------|
| E1_1 | 4 | random | none | Paper baseline |
| E1_2 | 4 | fixed | none | Fixed control |

### E2: Mask Supervision (H2)
| ID | Views | Selection | Mask | Description |
|----|-------|-----------|------|-------------|
| E2_2 | 4 | fixed | gt | GT mask |
| E2_3 | 4 | fixed | alpha | Rendered alpha |

### E3: More Views (H3)
| ID | Views | Selection | Mask | Description |
|----|-------|-----------|------|-------------|
| E3_2 | 5 | fixed | alpha | 5-view alpha |

### E4: Alpha Loss (H4)
| ID | Views | Selection | Mask | Alpha Loss | Description |
|----|-------|-----------|------|------------|-------------|
| E4_2 | 5 | fixed | alpha | 0.1 | Alpha supervision |

### E5: Combined Best (H5)
| ID | Views | Selection | Mask | Description |
|----|-------|-----------|------|-------------|
| E5_1 | 5 | random | alpha | Best combination |

---

## 비교 그룹

| 가설 | 비교 그룹 | 목적 |
|------|----------|------|
| H1 | E1_1 vs E1_2 | Random vs Fixed selection |
| H2 | E1_2 vs E2_2 vs E2_3 | No mask vs GT vs Alpha |
| H3 | E2_3 vs E3_2 | 4-view vs 5-view |
| H4 | E3_2 vs E4_2 | With/without alpha loss |
| H5 | E3_2 vs E5_1 | Fixed vs Random (5-view) |

---

## 새 데이터셋/실험 추가

### 새 데이터셋 추가

```yaml
# configs/datasets/D8_new.yaml
_name: D8_new
_description: "New dataset description"
_split_type: random

training:
  dataset:
    dataset_path: /path/to/D8_new/data_mouse_train.txt

validation:
  dataset_path: /path/to/D8_new/data_mouse_val.txt
```

### 새 실험 추가

```yaml
# configs/experiments/E6_1_new_feature.yaml
_name: E6_1_new_feature
_description: "New experiment with feature X"
_hypothesis: H6
_comparison_group: [E5_1, E6_1]

training:
  dataset:
    num_views: 6
    num_input_views: 5
    random_view_selection: false

  losses:
    new_feature_enabled: true
    new_feature_weight: 0.1
```

---

## 예제: 전체 실험 매트릭스 실행

```bash
#!/bin/bash
# run_all_experiments.sh

DATASETS=(D7_1 D7_2 D7_t)
EXPERIMENTS=(E1_1_paper_random E3_2_5v_alpha E4_2_5v_alpha_loss)

for dataset in "${DATASETS[@]}"; do
    for exp in "${EXPERIMENTS[@]}"; do
        echo "Running: ${dataset} + ${exp}"
        CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
            train_gslrm.py -d "${dataset}" -e "${exp}" &
    done
done
```

---

*Created: 2026-01-20 | Config Modularization v1.0*
