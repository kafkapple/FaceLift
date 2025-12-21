# Mouse-FaceLift: 2단계 파이프라인 파인튜닝 전략 및 실험 설계

---
date: 2025-12-10
context_name: "2_Research"
tags: [ai-assisted, mouse-reconstruction, multi-view, 3d-reconstruction, fine-tuning, mvdiffusion, gslrm]
project: FaceLift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

## 1. 제목 (Title)

**생쥐 6-View 영상 데이터를 활용한 FaceLift 2단계 파이프라인 파인튜닝 전략 수립**

---

## 2. 날짜 (Date)

2025-12-10

---

## 3. 연구 주제 (Research Topic)

FaceLift 모델의 2단계 파이프라인(MVDiffusion + GSLRM)을 생쥐 6-view 동기화 영상 데이터셋에 적응시키기 위한 파인튜닝 전략 수립 및 실험 설계.

### 가설
- 인간 얼굴 합성 데이터로 학습된 FaceLift 모델을 실제 생쥐 데이터로 파인튜닝하면, 도메인 갭에도 불구하고 single-view → multi-view → 3D 재구성 성능을 달성할 수 있다.
- 두 단계(MVDiffusion, GSLRM)를 독립적으로 파인튜닝하여 각 모듈의 적응 효과를 개별 검증할 수 있다.

---

## 4. 핵심 목표 (Key Objective)

1. **Stage 1 (MVDiffusion)**: 단일 뷰 입력 → 6개 다중 뷰 이미지 생성 모델 파인튜닝
2. **Stage 2 (GSLRM)**: 6개 뷰 이미지 → 3D Gaussian Splatting 재구성 모델 파인튜닝
3. **End-to-End 파이프라인**: 단일 이미지 입력 → 3D 모델 출력 완성

---

## 5. 배경 및 동기 (Background & Motivation)

### 5.1 FaceLift 원본 모델
- **출처**: Adobe Research, [FaceLift: Single Image to 3D Head](https://arxiv.org/abs/2412.17812)
- **원래 목적**: 인간 얼굴 단일 이미지 → 3D 재구성
- **구조**:
  - MVDiffusion: Stable UnCLIP 기반 multi-view 이미지 생성
  - GSLRM: Transformer 기반 3D Gaussian Splatting 예측

### 5.2 생쥐 데이터셋 특성
- **데이터 위치**: `/home/joon/data/markerless_mouse_1_nerf/`
- **구성**: 6개 동기화 카메라 (0.mp4 ~ 5.mp4)
- **메타데이터**: 마스크 (simpleclick_undist/), 카메라 캘리브레이션 (new_cam.pkl)
- **특징**:
  - 실제 촬영 데이터 (합성 데이터 대비 노이즈, 조명 변화 존재)
  - 제한된 데이터 규모 (~2000 프레임)
  - 생쥐 털 텍스처 (인간 피부 대비 복잡)

### 5.3 해결해야 할 문제
1. **도메인 갭**: 인간 얼굴 → 생쥐 (형태, 텍스처, 크기 차이)
2. **데이터 규모**: 합성 대규모 데이터 → 실제 소규모 데이터
3. **카메라 좌표계**: MAMMAL 형식 → FaceLift 형식 변환

---

## 6. 방법론 (Methodology)

### 6.1 전체 파이프라인 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Mouse-FaceLift 전체 파이프라인                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Single Image ──► MVDiffusion (Stage 1) ──► 6 Views ──► GSLRM (Stage 2) ──► 3D PLY │
│       ↓                   ↓                    ↓              ↓                    │
│   512x512 RGBA      30k steps fine-tune    512x512 x6    20k steps fine-tune      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Stage 1: MVDiffusion 파인튜닝

#### 설정 파일: `configs/mouse_mvdiffusion.yaml`

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `pretrained_model_name_or_path` | `checkpoints/mvdiffusion/pipeckpts` | FaceLift pretrained |
| `max_train_steps` | 30,000 | 제한된 데이터에 맞게 조정 |
| `learning_rate` | 5e-5 | 파인튜닝용 낮은 LR |
| `train_batch_size` | 4 | 메모리 고려 |
| `gradient_accumulation_steps` | 4 | Effective batch = 16 |
| `n_views` | 6 | 출력 뷰 수 |
| `reference_view_idx` | 0 | 입력 뷰 (cam_000) |

#### 데이터 증강 전략
```yaml
augmentation: true
aug_brightness: [0.9, 1.1]
aug_contrast: [0.9, 1.1]
aug_hflip: false  # 카메라 포즈 조정 필요하여 비활성화
```

#### 학습 전략
- **Classifier-Free Guidance**: 활성화 (`condition_drop_rate: 0.05`)
- **EMA 모델**: 활성화 (`use_ema: true`)
- **min-SNR 가중치**: `snr_gamma: 5.0`

### 6.3 Stage 2: GSLRM 파인튜닝

#### 설정 파일: `configs/mouse_config_finetune.yaml`

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `resume_ckpt` | `checkpoints/gslrm/ckpt_0000000000021125.pt` | FaceLift pretrained |
| `max_fwdbwd_passes` | 20,000 | 파인튜닝 스텝 |
| `learning_rate` | 2e-5 | MVDiffusion 대비 더 낮음 |
| `warmup` | 100 | 짧은 워밍업 |
| `reset_training_state` | true | 스텝 카운터 리셋 |

#### 손실 함수 구성
```yaml
losses:
  l2_loss_weight: 1.0        # 픽셀 MSE
  lpips_loss_weight: 0.5     # 지각적 유사도 (VGG)
  perceptual_loss_weight: 0.5
  ssim_loss_weight: 0.2      # 구조적 유사도
  pixelalign_loss_weight: 0.0  # 비활성화
  pointsdist_loss_weight: 0.0  # 비활성화
```

#### 모델 아키텍처 (변경 없음)
```yaml
model:
  transformer:
    d: 1024        # 히든 차원
    d_head: 64     # 어텐션 헤드 차원
    n_layer: 24    # 트랜스포머 레이어
  gaussians:
    n_gaussians: 2  # 실제 12288개
    sh_degree: 0
```

### 6.4 데이터 파이프라인

#### 전처리 스크립트: `scripts/process_mouse_data.py`
```bash
python scripts/process_mouse_data.py \
    --video_dir /home/joon/data/markerless_mouse_1_nerf/videos_undist \
    --meta_dir /home/joon/data/markerless_mouse_1_nerf \
    --output_dir data_mouse \
    --num_samples 2000 \
    --image_size 512
```

#### 출력 구조
```
data_mouse/
├── data_mouse_train.txt    # 학습 샘플 경로 목록 (80%)
├── data_mouse_val.txt      # 검증 샘플 경로 목록 (20%)
├── sample_000000/
│   ├── images/
│   │   ├── cam_000.png     # 512x512 RGBA
│   │   └── cam_00{1-5}.png
│   └── opencv_cameras.json # 카메라 파라미터
└── ...
```

#### 카메라 좌표 변환
```python
# MAMMAL format: K @ (R @ X + T)
# FaceLift format: w2c = [R|T] (4x4 matrix)

def convert_mammal_to_facelift(R, T, K):
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = T.flatten()
    return {
        "w2c": w2c.tolist(),
        "fx": K[0, 0], "fy": K[1, 1],
        "cx": K[0, 2], "cy": K[1, 2]
    }
```

---

## 7. 주요 결과 (Key Findings/Results)

### 7.1 구현 완료 항목

| 구성요소 | 파일 | 상태 |
|---------|------|------|
| 데이터 전처리 | `scripts/process_mouse_data.py` | 완료 |
| MVDiffusion Dataset | `mvdiffusion/data/mouse_dataset.py` | 완료 |
| GSLRM Dataset | `gslrm/data/mouse_dataset.py` | 완료 |
| MVDiffusion 학습 | `train_diffusion.py` | 완료 |
| GSLRM 학습 | `train_mouse.py` | 완료 |
| 추론 | `inference_mouse.py` | 완료 |
| 환경 설정 | `setup_mouse_env.sh` | 완료 |

### 7.2 Config 파일 비교

| Config | 용도 | Steps | Warmup | LR | Batch |
|--------|------|-------|--------|-----|-------|
| `mouse_mvdiffusion.yaml` | MVDiffusion fine-tune | 30k | 100 | 5e-5 | 4×4=16 |
| `mouse_config_finetune.yaml` | GSLRM fine-tune | 20k | 100 | 2e-5 | 2 |
| `mouse_config_debug.yaml` | 빠른 테스트 | 1k | 50 | 1e-4 | 4 |

### 7.3 학습 명령어

#### Stage 1: MVDiffusion
```bash
# Single GPU
python train_diffusion.py --config configs/mouse_mvdiffusion.yaml

# Multi GPU (권장)
accelerate launch --num_processes 4 \
    train_diffusion.py --config configs/mouse_mvdiffusion.yaml
```

#### Stage 2: GSLRM
```bash
# Overfitting 테스트 (필수)
python train_mouse.py --config configs/mouse_config_finetune.yaml --overfit 10

# Multi GPU (권장)
torchrun --nproc_per_node 4 --nnodes 1 \
    --rdzv_id ${RANDOM} --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    train_mouse.py --config configs/mouse_config_finetune.yaml
```

### 7.4 Wandb 로깅 구성

| 프로젝트 | 실험명 | 그룹 |
|---------|--------|------|
| `mouse_mvdiffusion` | `mouse_mvdiff_v1` | `mouse_facelift` |
| `mouse_facelift` | `mouse_6view_finetune` | `mouse` |

### 7.5 체크포인트 관리
- **MVDiffusion**: `checkpoints/experiments/train/mouse_mvdiffusion/pipeckpts/`
- **GSLRM**: `checkpoints/gslrm/mouse_finetune/`
- **보관 제한**: 최근 3개만 유지 (`checkpoints_total_limit: 3`)

---

## 8. 분석 및 논의 (Analysis & Discussion)

### 8.1 파인튜닝 vs Scratch 학습 결정

**선택: FaceLift Pretrained에서 파인튜닝**

이유:
1. **Transfer Learning 효과**: 기하학적 사전 지식 활용
2. **수렴 속도**: 제한된 데이터에서 더 빠른 수렴 기대
3. **일반화**: 과적합 방지 효과

대안 검토:
- Scratch 학습: 도메인 갭이 크면 오히려 방해될 수 있음
- Objaverse pretrained: 일반 3D 객체 사전 학습, 더 범용적

### 8.2 Stage별 학습률 설계 근거

| Stage | LR | 근거 |
|-------|-----|------|
| MVDiffusion | 5e-5 | 생성 모델, 다양한 출력 필요 |
| GSLRM | 2e-5 | 재구성 모델, 더 정밀한 조정 필요 |

### 8.3 데이터 증강 전략

**활성화된 증강**:
- Brightness jitter: [0.9, 1.1]
- Contrast jitter: [0.9, 1.1]
- Background color: white/black/gray 랜덤

**비활성화된 증강**:
- Horizontal flip: 카메라 포즈 조정 필요
- Rotation: 동일 이유

### 8.4 검증 전략

1. **Overfitting Test** (필수):
   - 10개 샘플로 loss → 0 수렴 확인
   - 코드 정확성 검증

2. **정기 검증**:
   - MVDiffusion: 500 steps마다
   - GSLRM: 500 steps마다

3. **평가 메트릭**:
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity)
   - LPIPS (Learned Perceptual)

---

## 9. 미결 과제 (Open Questions)

### 9.1 현재 한계점
1. **데이터 규모**: ~2000 샘플로 일반화 성능 불확실
2. **도메인 갭 정량화**: 인간 얼굴 → 생쥐 전이 효과 측정 필요
3. **카메라 캘리브레이션 정확도**: 변환 과정의 오차 영향

### 9.2 추가 탐색 필요 사항
1. **Multi-mouse 일반화**: 다른 생쥐 데이터에서 성능 검증
2. **View Selection**: 최적 reference view 탐색 (현재 cam_000 고정)
3. **Loss 가중치 튜닝**: LPIPS vs L2 비율 최적화
4. **Inference 속도 최적화**: 실시간 처리 가능성

### 9.3 다음 실험 계획
- [ ] Overfitting test 완료
- [ ] MVDiffusion 30k steps 학습
- [ ] GSLRM 20k steps 학습
- [ ] Hold-out 검증 세트 평가
- [ ] 정성적 시각화 (turntable video)

---

## 10. 참고 자료 (References)

1. **FaceLift Paper**: [arXiv:2412.17812](https://arxiv.org/abs/2412.17812)
2. **프로젝트 저장소**: `/home/joon/dev/FaceLift`
3. **관련 문서**:
   - `docs/guides/mouse_facelift_usage.md`: 사용 가이드
   - `docs/reports/241204_mouse_facelift_implementation_plan.md`: 초기 구현 계획
   - `docs/reports/241208_camera_parameters_analysis.md`: 카메라 파라미터 분석

---

## 부록: 주요 파일 참조

### 학습 스크립트
| 파일 | 설명 |
|------|------|
| `train_diffusion.py` | MVDiffusion 학습 (Stage 1) |
| `train_mouse.py` | GSLRM 학습 (Stage 2) |
| `inference_mouse.py` | 통합 추론 |

### 데이터셋 클래스
| 파일 | 클래스 | 용도 |
|------|--------|------|
| `mvdiffusion/data/mouse_dataset.py` | `MouseMVDiffusionDataset` | MVDiffusion 학습 |
| `gslrm/data/mouse_dataset.py` | `MouseViewDataset` | GSLRM 학습 |

### Config 파일
| 파일 | 용도 |
|------|------|
| `configs/mouse_mvdiffusion.yaml` | MVDiffusion 설정 |
| `configs/mouse_config_finetune.yaml` | GSLRM Fine-tune 설정 |
| `configs/mouse_config_debug.yaml` | 디버그용 설정 |
