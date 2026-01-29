# E2E Inference Guide (추론 가이드)

> **Updated**: 2026-01-29
> **관련 문서**: [EXPERIMENT_QUICKSTART](./EXPERIMENT_QUICKSTART.md) | [PREPROCESSING_REGISTRY](../datasets/PREPROCESSING_REGISTRY.md)

---

## 1. 개요

FaceLift E2E 추론 파이프라인:

```
Input Image → [MVDiffusion] → 6-view Images → [GS-LRM] → 3D Gaussians → Render
     또는
6-view Sample Directory → [GS-LRM] → 3D Gaussians → Render
```

---

## 2. 추론 스크립트

| 스크립트 | 용도 | 입력 |
|----------|------|------|
| `simple_temporal.py` | 시간에 따른 영상 생성 | 전처리된 데이터셋 (프레임 범위) |
| `run_e2e_inference.py` | 단일 이미지 → 3D | 이미지 1장 또는 6-view 샘플 |
| `render_from_checkpoint.py` | 체크포인트에서 직접 렌더링 | 샘플 디렉토리 |

---

## 3. Temporal Video 생성 (권장)

### 3.1 기본 명령어

```bash
cd /home/joon/dev/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh && conda activate facelift

# GPU 지정 + 실험 이름으로 체크포인트 자동 탐색
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.simple_temporal \
    --checkpoint M5_E0_1_facelift \
    --config configs/base/gslrm_mouse.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --start_frame 0 \
    --end_frame 100 \
    --frame_step 1 \
    --num_views 36 \
    --resolution 384 \
    --fps 24 \
    --output_dir outputs/temporal_M5
```

### 3.2 체크포인트 자동 탐색

`--checkpoint` 옵션은 다양한 형식을 지원합니다:

| 입력 형식 | 예시 | 동작 |
|----------|------|------|
| **실험 이름** | `M5_E0_1_facelift` | `checkpoints/gslrm/M5_E0_1_facelift/`에서 최신 찾기 |
| **pretrained** | `pretrained` | 원본 `ckpt_0000000000021125.pt` 사용 |
| **전체 경로** | `checkpoints/.../ckpt_*.pt` | 해당 파일 직접 사용 |
| **디렉토리** | `checkpoints/gslrm/M5_E0_1_facelift/` | 디렉토리 내 최신 찾기 |

**자동 탐색 우선순위**:
1. `best.pt` (있으면 우선)
2. 가장 높은 step의 `ckpt_*.pt`

```bash
# 실험 이름만 지정 (자동으로 최신 checkpoint)
--checkpoint M5_E0_1_facelift

# pretrained 모델 사용
--checkpoint pretrained

# 특정 파일 지정
--checkpoint checkpoints/gslrm/M5_E0_1_facelift/ckpt_0000000000006300.pt
```

### 3.3 GPU 지정

학습과 동일하게 `CUDA_VISIBLE_DEVICES`로 GPU 선택:

```bash
# GPU 4번 사용
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.simple_temporal ...

# GPU 5번 사용
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.inference.simple_temporal ...
```

### 3.4 주요 옵션

| 옵션 | 설명 | 기본값 | 권장 |
|------|------|--------|------|
| `--checkpoint` | 체크포인트 (자동 탐색 지원) | 필수 | 실험 이름 |
| `--start_frame` | 시작 프레임 (샘플 인덱스) | 첫 번째 | 0 |
| `--end_frame` | 끝 프레임 (exclusive) | 마지막 | 100 |
| `--frame_step` | 프레임 간격 | 1 | **1** (전처리 이미 5x 샘플링됨) |
| `--split` | Split 파일 경로 | None | test set 평가 시 |
| `--num_views` | 360° 회전 분할 수 | 36 | 36 (10° 간격) |
| `--resolution` | 렌더링 해상도 | 384 | 384 또는 512 |
| `--fps` | 출력 영상 FPS | 24 | 24 |
| `--fixed_angles` | 고정 각도 (복수 가능) | [0] | [0, 90, 180, 270] |
| `--elevation` | 카메라 고도각 | 20.0 | 20.0 |
| `--radius` | 카메라 거리 | 2.7 | 2.7 |

### 3.5 프레임 범위 이해

**중요**: 전처리 시 이미 `frame_jump=5` 적용됨

| 원본 데이터 | 전처리 후 | 추론 설정 | 결과 |
|-------------|-----------|-----------|------|
| ~18,000 프레임 | 3,600 샘플 | `--frame_step 1` | 3,600 샘플 전체 ✅ |
| ~18,000 프레임 | 3,600 샘플 | `--frame_step 5` | 720 샘플 (⚠️ 1/25 원본) |

```bash
# 전체 데이터셋
--frame_step 1

# 처음 100 샘플만 (빠른 테스트)
--start_frame 0 --end_frame 100 --frame_step 1

# 매 10번째 샘플 (미리보기)
--frame_step 10
```

### 3.6 Split 파일 사용

```bash
# Test set 평가
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.simple_temporal \
    --checkpoint M5t_E0_1_facelift \
    --config configs/base/gslrm_mouse.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5t \
    --split /home/joon/data/preprocessed/FaceLift_mouse/M5t/data_mouse_test.txt \
    --output_dir outputs/temporal_M5t_test
```

---

## 4. E2E 추론 (단일 이미지 → 3D)

### 4.1 6-view 샘플에서 직접 렌더링

```bash
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --sample_dir /home/joon/data/preprocessed/FaceLift_mouse/M5/000531 \
    --gslrm_checkpoint M5_E0_1_facelift \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/e2e_test
```

### 4.2 단일 이미지 입력 (MVDiffusion 필요)

```bash
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --input_image /path/to/single_image.png \
    --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse_finetuned \
    --gslrm_checkpoint M5_E0_1_facelift \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --camera_json configs/cameras/m5_fixed.json \
    --output_dir outputs/e2e_single
```

### 4.3 배치 처리

```bash
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --gslrm_checkpoint M5_E0_1_facelift \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/e2e_batch
```

---

## 5. 출력 구조

```
outputs/temporal_M5/
├── turntable_rotating.mp4      # 360° 회전 영상 (시간순)
├── turntable_fixed_0.mp4       # 고정 각도 0° 영상
├── turntable_fixed_90.mp4      # 고정 각도 90° 영상 (지정 시)
├── input_grid.mp4              # 입력 6-view 영상
├── combined_grid.mp4           # 입력 + 출력 결합
└── frames/                     # 개별 프레임 (선택)
    ├── 000000/
    │   ├── turntable.png       # 회전 strip
    │   └── input_views.png     # 입력 뷰
    └── ...
```

---

## 6. GPU 메모리 및 성능

| 설정 | GPU 메모리 | 프레임당 시간 |
|------|-----------|---------------|
| resolution=384, num_views=36 | ~8GB | ~2초 |
| resolution=512, num_views=60 | ~12GB | ~4초 |
| resolution=384, num_views=120 | ~10GB | ~5초 |

**권장**: A6000 (48GB) 기준, 배치 처리 가능

---

## 7. 체크포인트 위치

| 모델 | 경로 | `--checkpoint` 값 |
|------|------|---------------------|
| GS-LRM pretrained | `checkpoints/gslrm/ckpt_0000000000021125.pt` | `pretrained` |
| M5 + E0_1 | `checkpoints/gslrm/M5_E0_1_facelift/` | `M5_E0_1_facelift` |
| M5 + E1_2 | `checkpoints/gslrm/M5_E1_2_alpha/` | `M5_E1_2_alpha` |
| M5h_2 + E0_1 | `checkpoints/gslrm/M5h_2_E0_1_facelift/` | `M5h_2_E0_1_facelift` |
| MVDiffusion base | `checkpoints/mvdiffusion/pipeckpts/` | - |

---

## 8. 문제 해결

### Checkpoint not found

```bash
# 사용 가능한 체크포인트 확인
ls checkpoints/gslrm/

# 실험 이름으로 자동 탐색
--checkpoint M5_E0_1_facelift
```

### CUDA OOM

```bash
# 해상도 낮추기
--resolution 256

# 뷰 수 줄이기
--num_views 24
```

### 느린 렌더링

```bash
# 프레임 간격 늘리기 (미리보기용)
--frame_step 10

# 뷰 수 줄이기
--num_views 18
```

### 검은 화면 / 깨진 렌더링

- 체크포인트-config 불일치 확인
- 데이터셋 정규화 확인 (fx=549, trans=2.7)
- `--radius 2.7` 확인

---

## 9. Quick Reference

### Temporal Video (가장 많이 사용)

```bash
# 기본 (100 프레임 테스트)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.simple_temporal \
    --checkpoint M5_E0_1_facelift \
    --config configs/base/gslrm_mouse.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --start_frame 0 --end_frame 100 --frame_step 1 \
    --output_dir outputs/temporal_test

# 전체 데이터셋
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.simple_temporal \
    --checkpoint M5_E0_1_facelift \
    --config configs/base/gslrm_mouse.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --frame_step 1 \
    --output_dir outputs/temporal_full

# Test split 평가 (Pose Splatter 비교용)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.simple_temporal \
    --checkpoint M5t_E0_1_facelift \
    --config configs/base/gslrm_mouse.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5t \
    --split data_mouse_test.txt \
    --output_dir outputs/temporal_M5t_test

# Pretrained 모델로 테스트
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.simple_temporal \
    --checkpoint pretrained \
    --config configs/base/gslrm_mouse.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --start_frame 0 --end_frame 10 --frame_step 1 \
    --output_dir outputs/temporal_pretrained_test
```

---

## 10. 관련 문서

| 문서 | 내용 |
|------|------|
| [EXPERIMENT_QUICKSTART](./EXPERIMENT_QUICKSTART.md) | 학습 명령어 |
| [PREPROCESSING_REGISTRY](../datasets/PREPROCESSING_REGISTRY.md) | 데이터셋 SSOT |
| [TRAINING_LOGGING_GUIDE](./TRAINING_LOGGING_GUIDE.md) | WandB 메트릭 |

---

*FaceLift E2E Inference Guide v1.1 | 2026-01-29*
