# Config Modularization Proposal

## 현재 문제점

```
configs/mouse/
├── D7_1_E1_1_paper_random.yaml
├── D7_1_E1_2_paper_fixed.yaml
├── D7_1_E2_2_gt_mask.yaml
├── D7_1_E2_3_alpha_mask.yaml
├── D7_1_E3_2_5v_alpha.yaml
├── D7_1_E4_2_5v_alpha_loss.yaml
├── D7_1_E5_1_5v_alpha_random.yaml
├── D7_t_E1_1_paper_random.yaml   # D7_1 복사 + 경로만 변경
├── D7_t_E1_2_paper_fixed.yaml    # D7_1 복사 + 경로만 변경
├── D7_t_E2_2_gt_mask.yaml        # ...
└── ... (N datasets × M experiments = N×M 파일)
```

**문제**: 7개 데이터셋 × 8개 실험 = 56개 파일 (90% 중복)

---

## 제안: 3-Layer Config 구조

```
configs/
├── base/
│   └── gslrm_base.yaml           # 공통 설정 (model, optimizer, runtime)
│
├── datasets/
│   ├── D7_1.yaml                 # 데이터셋 경로만
│   ├── D7_2.yaml
│   └── D7_t.yaml
│
├── experiments/
│   ├── E1_1_paper_random.yaml    # 실험 파라미터만
│   ├── E1_2_paper_fixed.yaml
│   ├── E2_2_gt_mask.yaml
│   ├── E2_3_alpha_mask.yaml
│   ├── E3_2_5v_alpha.yaml
│   ├── E4_2_5v_alpha_loss.yaml
│   └── E5_1_5v_alpha_random.yaml
│
└── generated/                     # 런타임 생성 또는 캐시
    └── D7_1_E3_2.yaml
```

---

## 구현 방안 비교

### Option A: OmegaConf Merge (권장)

**구현**:
```python
# train_gslrm.py 수정
from omegaconf import OmegaConf

def load_modular_config(dataset: str, experiment: str) -> DictConfig:
    base = OmegaConf.load("configs/base/gslrm_base.yaml")
    dataset_cfg = OmegaConf.load(f"configs/datasets/{dataset}.yaml")
    exp_cfg = OmegaConf.load(f"configs/experiments/{experiment}.yaml")

    # Merge: base <- dataset <- experiment
    merged = OmegaConf.merge(base, dataset_cfg, exp_cfg)

    # Auto-generate paths
    merged.training.checkpointing.checkpoint_dir = f"checkpoints/gslrm/{dataset}_{experiment}"
    merged.training.logging.wandb.group = dataset
    merged.training.logging.wandb.exp_name = f"{dataset}_{experiment}"

    return merged
```

**CLI 사용**:
```bash
python train_gslrm.py --dataset D7_1 --experiment E3_2_5v_alpha

# 또는 기존 방식 호환
python train_gslrm.py --config configs/mouse/D7_1_E3_2_5v_alpha.yaml
```

**장점**:
- OmegaConf 이미 프로젝트에 존재
- 코드 변경 최소화
- 기존 단일 config 방식과 호환 가능

**단점**:
- train_gslrm.py 수정 필요
- CLI 인터페이스 변경

---

### Option B: Hydra 도입

**구현**:
```yaml
# configs/config.yaml
defaults:
  - base: gslrm_base
  - dataset: D7_1
  - experiment: E3_2_5v_alpha

# Auto path generation via interpolation
checkpoint_dir: checkpoints/gslrm/${dataset}_${experiment}
```

**CLI 사용**:
```bash
python train_gslrm.py dataset=D7_1 experiment=E3_2_5v_alpha
python train_gslrm.py dataset=D7_t experiment=E4_2 --multirun  # 자동 sweep
```

**장점**:
- 가장 강력한 config 관리
- Multi-run, sweep 지원
- CLI override 자연스러움

**단점**:
- 새 dependency 추가
- 학습 곡선 존재
- 기존 코드 대폭 수정

---

### Option C: CLI Override 확장 (최소 변경)

**현재 지원**:
```bash
python train_gslrm.py --config base.yaml \
    --set training.dataset.dataset_path /path/to/D7_1/train.txt \
    --set training.losses.mask_mode alpha
```

**확장 제안**:
```bash
# 프리셋 플래그 추가
python train_gslrm.py --config configs/base/gslrm_base.yaml \
    --dataset D7_1 \
    --experiment E3_2
```

**장점**:
- 코드 변경 최소
- 기존 방식 완전 호환

**단점**:
- 긴 명령어
- 실수 가능성

---

## 권장: Option A (OmegaConf Merge)

### 이유
1. **이미 사용 중**: train_diffusion.py에서 OmegaConf 사용
2. **점진적 마이그레이션**: 기존 단일 config도 계속 지원
3. **적절한 복잡도**: Hydra보다 간단, CLI override보다 편리

### 구현 단계

**Phase 1: 파일 구조 생성** (1시간)
```
configs/
├── base/gslrm_base.yaml
├── datasets/{D7_1, D7_2, D7_t}.yaml
└── experiments/{E1_1, E2_2, ...}.yaml
```

**Phase 2: 로더 수정** (30분)
```python
# train_gslrm.py에 추가
def load_modular_config(dataset, experiment):
    ...
```

**Phase 3: CLI 확장** (30분)
```python
parser.add_argument("--dataset", "-d", type=str)
parser.add_argument("--experiment", "-e", type=str)
```

---

## 파일 구조 예시

### configs/base/gslrm_base.yaml
```yaml
# 모든 실험에 공통
model:
  class_name: gslrm.model.gslrm.GSLRM
  image_tokenizer:
    image_size: 512
    patch_size: 8
    in_channels: 9
  transformer:
    d: 1024
    d_head: 64
    n_layer: 24
  gaussians:
    n_gaussians: 2
    sh_degree: 0
  # ... (나머지 모델 설정)

training:
  runtime:
    use_tf32: true
    use_amp: true
    amp_dtype: bf16
  dataloader:
    batch_size_per_gpu: 2
    num_workers: 8
  optimizer:
    lr: 1.0e-06
    beta1: 0.9
    beta2: 0.95
    weight_decay: 0.05
  schedule:
    num_epochs: 50000
    warmup: 500
  # ... (나머지 공통 설정)

mouse:
  use_mouse_dataset: true
  normalize_to_z_up: true
  # ...
```

### configs/datasets/D7_1.yaml
```yaml
# D7_1 데이터셋 전용
_name: D7_1
_description: "Individual scale, geometrically correct"

training:
  dataset:
    dataset_path: /home/joon/data/preprocessed/FaceLift_mouse/D7_1/data_mouse_train.txt

validation:
  dataset_path: /home/joon/data/preprocessed/FaceLift_mouse/D7_1/data_mouse_val.txt
```

### configs/experiments/E3_2_5v_alpha.yaml
```yaml
# E3.2: 5-View Alpha Mask
_name: E3_2_5v_alpha
_description: "5 input views with alpha mask"
_hypothesis: "H3: More views improve quality"

training:
  dataset:
    random_view_selection: false
    num_views: 6
    num_input_views: 5
  losses:
    masked_pixelalign_loss: true
    masked_l2_loss: true
    masked_ssim_loss: true
    background_loss_weight: 0.0
    mask_mode: alpha
    alpha_mask_threshold: 0.5
    alpha_loss_weight: 0.0
```

---

## 장단점 요약

| 측면 | 현재 (단일 파일) | 제안 (모듈화) |
|------|-----------------|--------------|
| **파일 수** | N×M (56개) | N+M+1 (16개) |
| **중복** | 90% | 0% |
| **수정 용이성** | 전체 파일 수정 | 해당 레이어만 |
| **실험 추가** | 모든 데이터셋 복사 | 1개 파일만 |
| **데이터셋 추가** | 모든 실험 복사 | 1개 파일만 |
| **학습 곡선** | 없음 | 약간 |
| **호환성** | - | 기존 방식 지원 |

---

## 마이그레이션 전략

1. **기존 configs 보존**: `configs/mouse/_legacy/`로 이동
2. **점진적 적용**: 신규 실험부터 모듈화 방식 사용
3. **문서화**: `docs/CONFIG_GUIDE.md` 작성
4. **스크립트**: 기존 config → 모듈화 변환 스크립트

---

*Proposal Date: 2026-01-20*
