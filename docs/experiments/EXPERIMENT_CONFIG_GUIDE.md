# Experiment Config Guide

> Consolidated from CONFIG_SCHEMA, CONFIG_MODULAR, EXPERIMENT_SCHEMA | 2026-01-28
> 3개 문서를 하나로 통합한 실험 설정 종합 가이드

---

## 1. Modular Config System

### 개요

기존 방식은 데이터셋 × 실험 조합마다 개별 설정 파일이 필요했습니다:
- 7 데이터셋 × 8 실험 = 56개 파일 (90% 중복)

모듈화 방식은 **Base + Dataset + Experiment** 3-Layer 구조로 중복을 제거합니다:
- Base (1) + Dataset (N) + Experiment (M) = **1 + N + M 파일**

### 디렉토리 구조

```
configs/
├── base/
│   └── gslrm_mouse.yaml       # 공통 설정 (모델, 옵티마이저 등)
│
├── datasets/
│   ├── D7_1.yaml              # D7_1 데이터셋 경로
│   ├── D7_2.yaml              # D7_2 데이터셋 경로
│   ├── D7_t.yaml              # D7_t (temporal split) 경로
│   └── M5.yaml                # M5 (현재 권장 데이터셋)
│
├── experiments/
│   ├── E0_1_facelift.yaml     # FaceLift Baseline
│   ├── E1_2_alpha.yaml        # GT Mask + alpha
│   ├── E2_1_alpha.yaml        # Alpha Only
│   └── ...
│
└── mouse/                      # Legacy 단일 설정 (호환용)
```

### 사용법

#### 모듈화 모드 (권장)

```bash
# 기본 사용법
python train_gslrm.py --dataset M5 --experiment E0_1_facelift

# 단축 옵션
python train_gslrm.py -d M5 -e E0_1_facelift

# torchrun으로 실행
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5 -e E0_1_facelift
```

#### Legacy 모드 (호환)

```bash
# 기존 단일 설정 파일 사용
python train_gslrm.py --config configs/mouse/D7_1_E3_2_5v_alpha.yaml
```

**⚠️ Legacy 주의**: `--config` 모드는 완전한 구조가 필수. Modular 모드는 base 자동 merge로 구조 실수 방지.

#### Override 옵션

두 모드 모두 `--set` 옵션으로 설정 오버라이드 가능:

```bash
# 학습률 변경
python train_gslrm.py -d M5 -e E0_1_facelift \
    --set training.optimizer.lr 5e-7

# 배치 크기 변경
python train_gslrm.py -d M5 -e E0_1_facelift \
    --set training.dataloader.batch_size_per_gpu 4
```

### Config 계층

| Layer | 위치 | 내용 |
|-------|------|------|
| **Base** | `configs/base/gslrm_mouse.yaml` | 모델 아키텍처, 런타임(AMP, TF32), 옵티마이저/스케줄러 기본값 |
| **Dataset** | `configs/datasets/` | training/validation 데이터 경로 |
| **Experiment** | `configs/experiments/` | 뷰 수, 마스크 모드, alpha loss 등 실험 변수 |

### 자동 생성되는 값

모듈화 모드에서 다음 값들이 자동 생성됩니다:

| 설정 | 자동 생성 값 |
|------|-------------|
| `checkpoint_dir` | `checkpoints/gslrm/{dataset}_{experiment}` |
| `wandb.group` | `{dataset}` |
| `wandb.exp_name` | `{dataset}_{experiment}` |

### 새 데이터셋 추가

```yaml
# configs/datasets/M5.yaml
_name: M5
_description: "M-Series v5 dataset"
_split_type: random

training:
  dataset:
    dataset_path: /path/to/M5/data_mouse_train.txt

validation:
  dataset_path: /path/to/M5/data_mouse_val.txt
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

### 전체 실험 매트릭스 실행 예시

```bash
#\!/bin/bash
DATASETS=(M5 D7_1 D7_2)
EXPERIMENTS=(E0_1_facelift E1_2_alpha E2_1_alpha)

for dataset in "${DATASETS[@]}"; do
    for exp in "${EXPERIMENTS[@]}"; do
        echo "Running: ${dataset} + ${exp}"
        CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
            train_gslrm.py -d "${dataset}" -e "${exp}" &
    done
done
```

---

## 2. Config Schema Reference

FaceLift config는 YAML 형식이며, 크게 4개 섹션으로 구성됩니다.

```yaml
model:       # 모델 아키텍처 및 뷰 설정
training:    # 학습 설정 (데이터, 손실, 옵티마이저)
validation:  # 검증 설정
wandb:       # 로깅 설정
```

### 2.1 model 섹션

**⭐ 뷰 설정은 반드시 여기에\!**

```yaml
model:
  # ⭐ 뷰 설정 (필수)
  num_views: 6              # 전체 뷰 수 (타겟 렌더링용)
  num_input_views: 5        # 입력 뷰 수 (인코딩용)
  
  # 이미지 토크나이저
  image_tokenizer:
    image_size: 512         # 입력 이미지 크기
    patch_size: 8           # 패치 크기
  
  # Transformer
  transformer:
    d: 1024                 # 임베딩 차원
    n_layer: 24             # 레이어 수
    n_head: 16              # 어텐션 헤드 수
  
  # Gaussian 설정
  gaussians:
    n_gaussians: 2          # 픽셀당 Gaussian 수
    sh_degree: 0            # Spherical Harmonics 차수
  
  # 기타
  hard_pixelalign: true     # 픽셀 정렬 강제
  clip_xyz: true            # XYZ 클리핑
```

### 2.2 training 섹션

#### dataset

```yaml
training:
  dataset:
    data_list: /path/to/data_mouse_train.txt  # 학습 데이터 목록 (절대경로\!)
    image_size: 512                           # 이미지 크기
    random_view_selection: true               # 랜덤 뷰 선택 (기본: true)
    target_has_input: false                   # 타겟에 입력 뷰 포함 여부
    
    # ❌ 여기에 넣지 말 것\!
    # num_views, num_input_views → model 섹션으로
```

#### losses

```yaml
training:
  losses:
    # 기본 손실
    l2_loss_weight: 1.0           # RGB L2 손실
    perceptual_loss_weight: 0.5   # LPIPS 손실
    ssim_loss_weight: 0.0         # SSIM 손실
    
    # ⭐ 마스크 설정 (핵심)
    mask_mode: gt                 # none, gt, alpha, composite
    normalize_by_mask: true       # 마스크 픽셀 수로 정규화
    
    # Alpha 손실
    alpha_loss_weight: 0.1        # Alpha supervision 가중치
    alpha_loss_type: mse          # mse, bce, dice, focal
    
    # 배경 손실
    bg_loss_weight: 0.0           # 배경 페널티 (E4용)
    
    # 기타
    l2_warmup_steps: 500          # L2 손실 warmup
```

**mask_mode 옵션:**

| 값 | 설명 | 사용 예 |
|----|------|---------|
| `none` | 마스크 사용 안함 | E0 (baseline) |
| `gt` | GT 마스크로 L2/perceptual 계산 | E1, E2 |
| `alpha` | 렌더링된 alpha로 마스크 | (권장하지 않음 - 악순환 위험) |
| `composite` | 배경 합성 | E5 |

#### optimizer & scheduler

```yaml
training:
  optimizer:
    lr: 1.0e-06               # 학습률
    weight_decay: 0.05        # 가중치 감쇠
    grad_clip_norm: 50.0      # Gradient 클리핑
  
  scheduler:
    warmup: 500               # Warmup 스텝
    max_fwdbwd_passes: 15000  # 최대 학습 스텝
```

#### 기타

```yaml
training:
  amp:
    use_amp: true             # Mixed precision
    amp_dtype: bf16           # bf16 또는 fp16
  
  checkpoint:
    checkpoint_every: 100     # 체크포인트 저장 주기
    keep_last_n: 5            # 최근 N개만 유지
  
  logging:
    vis_every: 100            # 시각화 주기
    val_every: 200            # 검증 주기
  
  batch:
    batch_size_per_gpu: 2     # 배치 크기
    num_workers: 4            # DataLoader 워커 수
```

### 2.3 validation 섹션

```yaml
validation:
  data_list: /path/to/data_mouse_val.txt  # 검증 데이터 목록
```

### 2.4 wandb 섹션

```yaml
wandb:
  project: FaceLift           # 프로젝트 이름
  name: M5_E0_1_facelift      # 실험 이름 (파일명과 일치 권장)
  tags: [M5, E0, facelift]    # 태그
```

### 검증 체크리스트

새 config 작성 시 확인:

- [ ] `model.num_views`, `model.num_input_views` 있는가?
- [ ] `training.dataset`에 num_views/num_input_views **없는가?**
- [ ] `training.dataset.data_list` **절대경로**인가?
- [ ] `wandb.name`이 파일명과 일치하는가?
- [ ] `python scripts/validate_config.py` 통과하는가?
- [ ] Pretrained checkpoint 경로 포함되어 있는가? (`checkpoints/gslrm/ckpt_0000000000021125.pt`)

### 템플릿

```bash
# 기존 config 복사 후 수정
cp configs/experiments/E0_1_facelift.yaml configs/experiments/my_new.yaml
# 수정 후 검증
python scripts/validate_config.py configs/experiments/my_new.yaml
```

---

## 3. Experiment Naming & Categories

### 3.1 명명 규칙 (Naming Convention)

#### 기본 패턴

```
E{카테고리}_{번호}_{이름}                     - 기본 설정
E{카테고리}_{번호}_{서브번호}_{이름}_{변형}    - 변형 설정
```

#### 구성 요소

| 요소 | 설명 | 예시 |
|------|------|------|
| **카테고리** | 0, 1, 2, ... | E**0**, E**1**, E**2** |
| **번호** | 카테고리 내 순번 | E0_**1**, E0_**2** |
| **이름** | 설정 식별자 | E0_1_**facelift**, E1_2_**alpha** |
| **서브번호** | 변형 순번 | E0_1_**1**, E1_2_**3** |
| **변형** | 변형 키워드 | E0_1_1_facelift_**alpha** |

#### 예시

```
E0_1_facelift                    # 카테고리 0, 1번, FaceLift 기본
E0_1_1_facelift_alpha            # E0_1의 1번 변형: +alpha
E0_1_2_facelift_fixed            # E0_1의 2번 변형: +fixed
E0_1_3_facelift_alpha_fixed      # E0_1의 3번 변형: +alpha+fixed

E1_2_alpha                       # 카테고리 1, 2번, alpha 설정
E1_2_1_alpha_3v                  # E1_2의 1번 변형: 3 view
E1_2_2_alpha_5v                  # E1_2의 2번 변형: 5 view
E1_2_3_alpha_fixed               # E1_2의 3번 변형: fixed view
E1_2_4_alpha_overfit             # E1_2의 4번 변형: overfit test
```

### 3.2 카테고리 체계 (Category System)

| 카테고리 | mask_mode | alpha | 설명 | 상태 |
|----------|-----------|-------|------|------|
| **E0** | none | 0.0 or 0.1 | FaceLift Baseline | ⭐ 권장 |
| **E1** | gt | 0.0~1.0 | GT Mask 기반 | 활성 |
| **E2** | none | 0.1 | Alpha Only | 실험적 |

#### E0: FaceLift Baseline

- **핵심**: mask_mode=none (전체 이미지 학습)
- **출처**: FaceLift/GS-LRM 논문 원본
- **권장**: Val PSNR 가장 높음 (D7_1: 20.9)

#### E1: GT Mask

- **핵심**: mask_mode=gt (foreground만 학습)
- **출처**: Pose Splatter + LGM
- **주의**: 과적합 경향 (Gap +10.3)

#### E2: Alpha Only

- **핵심**: mask_mode=none + alpha supervision만
- **출처**: LGM 변형
- **상태**: 실험적

### 3.3 전체 Config 목록

#### E0: FaceLift Baseline

| ID | alpha | view | 설명 |
|----|-------|------|------|
| **E0_1_facelift** | 0.0 | random | 논문 원본 ⭐ |
| E0_1_1_facelift_alpha | 0.1 | random | +alpha |
| E0_1_2_facelift_fixed | 0.0 | fixed | +fixed |
| E0_1_3_facelift_alpha_fixed | 0.1 | fixed | +alpha+fixed |
| E0_2_mouse | 0.0 | random | Mouse-adapted (lr 조정) |

#### E1: GT Mask

| ID | alpha | 변형 | 설명 |
|----|-------|------|------|
| E1_1_base | 0.0 | - | GT mask만 |
| **E1_2_alpha** | 0.1 | - | GT mask + alpha |
| E1_2_1_alpha_3v | 0.1 | 3v | 3 view input |
| E1_2_2_alpha_5v | 0.1 | 5v | 5 view input |
| E1_2_3_alpha_fixed | 0.1 | fixed | fixed view |
| E1_2_4_alpha_overfit | 0.1 | overfit | 1 sample test |
| E1_3_lgm | 1.0 | - | LGM style (강한 alpha) |

#### E2: Alpha Only

| ID | alpha | 설명 |
|----|-------|------|
| E2_1_alpha | 0.1 | mask 없이 alpha만 |

### 3.4 변형 키워드 (Variant Keywords)

| 키워드 | 의미 | 적용 |
|--------|------|------|
| **alpha** | alpha_loss_weight > 0 | +alpha supervision |
| **fixed** | random_view_selection: false | 고정 뷰 선택 |
| **3v/5v** | num_input_views: 3/5 | view 수 변경 |
| **overfit** | 1 sample, validation off | overfit 테스트 |
| **lgm** | alpha_loss_weight: 1.0 | LGM 스타일 |

### 3.5 비교 그룹

| 가설 | 비교 그룹 | 목적 |
|------|----------|------|
| H1 | E1_1 vs E1_2 | Random vs Fixed selection |
| H2 | E1_2 vs E2_2 vs E2_3 | No mask vs GT vs Alpha |
| H3 | E2_3 vs E3_2 | 4-view vs 5-view |
| H4 | E3_2 vs E4_2 | With/without alpha loss |
| H5 | E3_2 vs E5_1 | Fixed vs Random (5-view) |

### 3.6 신규 실험 추가 가이드

#### 기본 설정 추가

```bash
# 1. 다음 번호 확인
ls configs/experiments/E{N}_*.yaml

# 2. 파일 생성: E{N}_{다음번호}_{이름}.yaml
```

#### 변형 설정 추가

```bash
# 1. 기존 변형 번호 확인
ls configs/experiments/E{N}_{M}_*.yaml

# 2. 파일 생성: E{N}_{M}_{다음서브번호}_{이름}_{변형}.yaml
```

#### 체크리스트

- [ ] 헤더 주석에 ID와 설명 포함
- [ ] model 섹션 포함 (num_views, num_input_views)
- [ ] training.dataset.random_view_selection 명시
- [ ] training.losses 섹션 완전히 명시
- [ ] EXPERIMENT_REGISTRY.md 업데이트

### 3.7 Config 파일 포맷

#### 헤더 템플릿

```yaml
# =============================================================================
# {ID}: {짧은 설명}
# =============================================================================
# {상세 설명}
# - 주요 설정1
# - 주요 설정2
```

#### 필수 섹션

```yaml
model:
  num_views: 6
  num_input_views: 4

training:
  dataset:
    random_view_selection: true/false

  losses:
    mask_mode: none/gt
    alpha_loss_weight: 0.0/0.1/1.0
    bg_loss_weight: 0.0
```

#### 선택 섹션

```yaml
training:
  optimizer:           # lr 조정 필요시
  runtime:             # grad_clip 필요시
  schedule:            # max_steps 조정시

validation:
  enabled: true/false  # overfit 테스트시 false
```

---

## 4. Related Docs

| 문서 | 내용 |
|------|------|
| [EXPERIMENT_REGISTRY](./EXPERIMENT_REGISTRY.md) | 실험 목록 및 실행 명령어 |
| [HYPOTHESIS_EXPERIMENT_PLAN](./HYPOTHESIS_EXPERIMENT_PLAN.md) | 가설 검증 계획 |
| [TRAINING_LOGGING_GUIDE](./TRAINING_LOGGING_GUIDE.md) | 학습/로깅 가이드 |
| [PREPROCESSING_REGISTRY](../datasets/PREPROCESSING_REGISTRY.md) | 전처리 SSOT |
| [INDEX](../INDEX.md) | 전체 문서 허브 |

> **Note**: MVDiffusion config는 별도 시스템 (`configs/mvdiffusion/`)으로 관리됩니다. 본 가이드는 GS-LRM 실험 설정에 한정됩니다.

---

*Consolidated: 2026-01-28 | Sources: CONFIG_SCHEMA (2026-01-23), CONFIG_MODULAR (2026-01-20), EXPERIMENT_SCHEMA (2026-01-25)*
