# FaceLift Config System

> Last Updated: 2026-02-09

## Quick Start

```bash
# Mode 1: Legacy (단일 파일)
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E2_gt_alpha.yaml

# Mode 2: Modular (권장 — base + dataset + experiment)
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e E0_1_facelift

# Mode 3: Flexible (custom base + experiment)
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -b configs/base/gslrm_mouse.yaml -e E0_1_facelift
```

## Directory Structure

```
configs/
├── README.md              # 이 문서
├── gslrm.yaml             # Base config (원본, 수정 금지)
│
├── base/                  # Base configs (modular/flexible mode)
│   └── gslrm_mouse.yaml   # Mouse 실험용 base
│
├── datasets/              # Dataset 정의 (modular mode)
│   ├── D7_1.yaml          # Individual scale
│   ├── M5t2.yaml          # Temporal 80:10:10 split
│   └── ...
│
├── experiments/           # Experiment 정의 (modular/flexible mode)
│   ├── E0_1_facelift.yaml # FaceLift baseline
│   ├── E1_2_alpha.yaml    # GT + alpha loss
│   ├── E_debug.yaml       # Quick debug run
│   └── ...
│
└── mouse/                 # Combined configs (legacy mode)
    ├── D7_1_E2_gt_alpha.yaml
    └── ...
```

## Config Modes

`train_gslrm.py`는 3가지 config 로딩 모드를 지원합니다.
**상호 배타적** — 하나만 선택해야 합니다.

### Mode 1: Legacy (`--config` / `-c`)

단일 YAML 파일에 모든 설정 포함. 실행이 단순하지만 조합이 늘면 파일 폭발.

```bash
train_gslrm.py --config configs/mouse/D7_1_E2_gt_alpha.yaml
```

- **장점**: 한 파일에 전체 설정이 보임
- **단점**: dataset × experiment 조합마다 파일 필요
- **사용처**: `configs/mouse/` 디렉토리, temporal training script (`train_temporal_gslrm.py`)

### Mode 2: Modular (`-d` + `-e`) — 권장

Base + Dataset + Experiment 3단계 merge. OmegaConf merge 순서: `base ← dataset ← experiment`

```bash
train_gslrm.py -d M5t2 -e E0_1_facelift
```

- **Merge**: `configs/base/gslrm_mouse.yaml` ← `configs/datasets/M5t2.yaml` ← `configs/experiments/E0_1_facelift.yaml`
- **Auto-generated**: checkpoint dir = `checkpoints/gslrm/M5t2_E0_1_facelift`, wandb group = `M5t2`
- **장점**: dataset과 experiment 자유 조합, 중복 최소
- **단점**: merge 결과를 머릿속에서 조합해야 함

### Mode 3: Flexible (`-b` + `-e`)

Custom base + Experiment 2단계 merge. Dataset layer를 건너뜀.

```bash
train_gslrm.py -b configs/base/gslrm_mouse.yaml -e E0_1_facelift
```

- **Merge**: `{custom_base}` ← `configs/experiments/{experiment}.yaml`
- **Auto-generated**: checkpoint dir = `checkpoints/gslrm/{base_name}_{exp_name}`
- **장점**: 비표준 base config 사용 가능 (e.g., 이미 dataset이 내장된 combined config)
- **주의**: `-b`와 `-d`는 동시 사용 불가

### CLI Override (`--set` / `-s`)

모든 모드에서 개별 값 오버라이드 가능:

```bash
train_gslrm.py -d M5t2 -e E0_1_facelift \
    --set training.schedule.max_fwdbwd_passes 400 \
    --set validation.val_every 200
```

### Mode 선택 가이드

| 상황 | 권장 모드 |
|------|-----------|
| 새 실험 (dataset + experiment 조합) | Mode 2: Modular |
| 기존 standalone config 실행 | Mode 1: Legacy |
| 특수 base config 필요 | Mode 3: Flexible |
| temporal training (`train_temporal_gslrm.py`) | Mode 1: Legacy |
| 빠른 디버그 | Mode 2 + `--set` override |

## Config 구조 규칙

### YAML 키 계층 (OmegaConf merge 기준)

```yaml
# ✅ 올바른 구조 — base/dataset/experiment 모두 동일 계층
model:
  num_views: 6
  num_input_views: 5

training:
  dataset:
    dataset_path: /path/to/split.txt    # 데이터셋 config
  losses:
    mask_mode: gt                        # 실험 config
    alpha_loss_weight: 0.1
  schedule:
    max_fwdbwd_passes: 30000
  optimizer:
    lr: 1.0e-6

validation:
  dataset_path: /path/to/val_split.txt

# ❌ 잘못된 구조 (플랫)
dataset:                                  # training.dataset 아님!
  dataset_path: ...
losses:                                   # training.losses 아님!
  mask_mode: gt
```

### Dataset Config 예시 (`configs/datasets/M5t2.yaml`)

```yaml
training:
  dataset:
    dataset_path: ~/data/.../data_mouse_t2_train.txt

validation:
  dataset_path: ~/data/.../data_mouse_t2_val.txt

test:
  dataset_path: ~/data/.../data_mouse_t2_test.txt
```

### Experiment Config 예시 (`configs/experiments/E0_1_facelift.yaml`)

```yaml
model:
  num_views: 6
  num_input_views: 4

training:
  dataset:
    random_view_selection: true
  losses:
    mask_mode: none
    alpha_loss_weight: 0.0
```

## 참고

- **Temporal training**: 별도 스크립트 (`mouse_extensions/scripts/train_temporal_gslrm.py`) 사용, Legacy mode만 지원
- **Inference**: `run_e2e_inference.py`, `gslrm_pipeline.py` 등 별도 argparse
- **Config 검증**: merge 후 `--set` override로 개별 값 확인 가능
