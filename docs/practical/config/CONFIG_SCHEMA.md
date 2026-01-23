# Config Schema Reference

> Last Updated: 2026-01-23

## Overview

FaceLift config는 YAML 형식이며, 크게 4개 섹션으로 구성됩니다.

```yaml
model:       # 모델 아키텍처 및 뷰 설정
training:    # 학습 설정 (데이터, 손실, 옵티마이저)
validation:  # 검증 설정
wandb:       # 로깅 설정
```

---

## 1. model 섹션

**뷰 설정은 반드시 여기에!**

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

---

## 2. training 섹션

### 2.1 dataset

```yaml
training:
  dataset:
    data_list: /path/to/data_mouse_train.txt  # 학습 데이터 목록 (절대경로)
    image_size: 512                           # 이미지 크기
    random_view_selection: true               # 랜덤 뷰 선택 (기본: true)
    target_has_input: false                   # 타겟에 입력 뷰 포함 여부
    
    # ❌ 여기에 넣지 말 것!
    # num_views, num_input_views → model 섹션으로
```

### 2.2 losses

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
| none | 마스크 사용 안함 | E0 (baseline) |
| gt | GT 마스크로 L2/perceptual 계산 | E1, E2 |
| alpha | 렌더링된 alpha로 마스크 | (권장하지 않음) |
| composite | 배경 합성 | E5 |

### 2.3 optimizer & scheduler

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

### 2.4 기타

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

---

## 3. validation 섹션

```yaml
validation:
  data_list: /path/to/data_mouse_val.txt  # 검증 데이터 목록
```

---

## 4. wandb 섹션

```yaml
wandb:
  project: FaceLift           # 프로젝트 이름
  name: D7_1_E2_gt_alpha      # 실험 이름 (파일명과 일치 권장)
  tags: [D7_1, E2, gt_alpha]  # 태그
```

---

## 검증 체크리스트

새 config 작성 시 확인:

- [ ] `model.num_views`, `model.num_input_views` 있는가?
- [ ] `training.dataset`에 num_views/num_input_views 없는가?
- [ ] `training.dataset.data_list` 절대경로인가?
- [ ] `wandb.name`이 파일명과 일치하는가?
- [ ] `python scripts/validate_config.py` 통과하는가?

---

## 템플릿

새 config 작성 시 복사해서 사용:

```bash
cp configs/mouse/D7_1_E2_gt_alpha.yaml configs/mouse/my_new_config.yaml
# 수정 후
python scripts/validate_config.py configs/mouse/my_new_config.yaml
```
