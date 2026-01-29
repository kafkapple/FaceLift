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

*FaceLift E2E Inference Guide v1.2 | 2026-01-29*

---

## 11. Gaussian Export & Rerun 지원 (v1.2 추가)

### 11.1 새로운 Export 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--save_gaussian` | Gaussian .ply/.npz 파일 저장 | False |
| `--save_rerun` | Rerun .rrd 파일 저장 (인터랙티브 3D 뷰어) | False |
| `--save_first_only` | 첫 프레임만 export (빠른 테스트) | False |
| `--rotation_speed` | 회전 속도 (0.5=절반, 1.0=정상) | **0.5** |

### 11.2 회전 속도 조절

기본값 `--rotation_speed 0.5`로 절반 속도 (프레임 2배 보간):

```bash
# 절반 속도 (기본값 - 권장)
--rotation_speed 0.5   # 36 views → 72 frames

# 정상 속도
--rotation_speed 1.0   # 36 views → 36 frames

# 1/4 속도 (느리게)
--rotation_speed 0.25  # 36 views → 144 frames
```

### 11.3 Gaussian 파일 저장

```bash
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.simple_temporal \
    --checkpoint M5_E0_1_facelift \
    --config configs/base/gslrm_mouse.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --start_frame 0 --end_frame 10 \
    --save_gaussian \
    --output_dir outputs/with_gaussians
```

**출력 구조**:
```
outputs/with_gaussians/
├── gaussians/
│   ├── frame_000000.ply    # GS Viewer 호환
│   ├── frame_000000.npz    # Lightweight (Rerun용)
│   ├── frame_000001.ply
│   └── ...
├── turntable_first.mp4
└── ...
```

### 11.4 Rerun 인터랙티브 뷰어

```bash
# Rerun .rrd 파일 생성
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.simple_temporal \
    --checkpoint M5_E0_1_facelift \
    --config configs/base/gslrm_mouse.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --start_frame 0 --end_frame 10 \
    --save_gaussian --save_rerun \
    --output_dir outputs/with_rerun

# Rerun 뷰어로 열기
rerun outputs/with_rerun/rerun/sequence.rrd

# 웹 뷰어 (SSH 환경)
rerun --web-viewer outputs/with_rerun/rerun/sequence.rrd
```

**Rerun 기능**:
- 3D 포인트 클라우드 시각화
- 타임라인으로 프레임 간 이동
- 색상/투명도 별도 레이어
- 마우스로 회전/줌/팬

### 11.5 빠른 첫 프레임 테스트

```bash
# 첫 프레임만 Gaussian/Rerun 저장 (빠름)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.simple_temporal \
    --checkpoint M5_E0_1_facelift \
    --config configs/base/gslrm_mouse.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --start_frame 0 --end_frame 100 \
    --save_gaussian --save_rerun --save_first_only \
    --output_dir outputs/first_frame_test
```

### 11.6 전체 옵션 예제

```bash
# 모든 기능 활성화
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.simple_temporal \
    --checkpoint M5_E0_1_facelift \
    --config configs/base/gslrm_mouse.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --start_frame 0 --end_frame 100 \
    --num_views 36 \
    --resolution 384 \
    --fps 24 \
    --rotation_speed 0.5 \
    --fixed_angles 0 90 180 270 \
    --save_gaussian \
    --save_rerun \
    --output_dir outputs/full_export
```

### 11.7 업데이트된 출력 구조

```
outputs/full_export/
├── turntable_first.mp4         # 첫 프레임 360° (절반 속도)
├── time_fixed.mp4              # 고정 각도 0°
├── time_fixed_angle90.mp4      # 고정 각도 90° (지정 시)
├── time_fixed_angle180.mp4     # 고정 각도 180°
├── time_fixed_angle270.mp4     # 고정 각도 270°
├── time_rotating.mp4           # 시간에 따라 회전
├── full_all.mp4                # 전체 (T×V 프레임)
├── grid_first.jpg              # 첫 프레임 그리드
├── grid_6view.mp4              # 입력 6-view 영상
├── gaussians/                  # Gaussian 파일
│   ├── frame_000000.ply
│   ├── frame_000000.npz
│   └── ...
└── rerun/                      # Rerun 파일
    └── sequence.rrd            # 타임라인 시퀀스
```

---

*Updated: 2026-01-29 | Added Gaussian export & Rerun support*
