# E2E Inference Guide (추론 가이드)

> **Updated**: 2026-01-29 (v2.0 - 구조 개편)
> **관련 문서**: [EXPERIMENT_QUICKSTART](./EXPERIMENT_QUICKSTART.md) | [PREPROCESSING_REGISTRY](../datasets/PREPROCESSING_REGISTRY.md)

---

## 1. 개요

### 1.1 파이프라인

```
Path 1 (6-view):  6-view Sample → [GS-LRM] → 3D Gaussians → Render
Path 2 (1-view):  1-view Image → [MVDiffusion] → 6-view → [GS-LRM] → 3D Gaussians → Render
```

### 1.2 스크립트 비교

| 스크립트 | 입력 | MVDiffusion | 출력 | 용도 |
|----------|------|-------------|------|------|
| `run_e2e_inference.py` | 1-view 또는 6-view | ✅ 지원 | Gaussian, 그리드 | 3D 재구성, 평가 |
| `simple_temporal.py` | 6-view 데이터셋 | ❌ | 영상 (.mp4) | 시각화, 데모 |

---

## 2. Quick Start

### 2.1 권장 체크포인트

| 모델 | 경로 | 자동 탐색 값 |
|------|------|-------------|
| **GS-LRM** | `checkpoints/gslrm/M5t_E0_1_facelift/` | `M5t_E0_1_facelift` |
| **MVDiffusion** | `checkpoints/mvdiffusion/mouse_M5/` | `mouse_M5` |
| GS-LRM (pretrained) | `checkpoints/gslrm/ckpt_0000000000021125.pt` | `pretrained` |

### 2.2 단일 샘플 (6-view → 3D)

```bash
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --sample_dir /home/joon/data/preprocessed/FaceLift_mouse/M5/000531 \
    --gslrm_checkpoint M5t_E0_1_facelift \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/sample_531
```

### 2.3 단일 샘플 (1-view → MVDiffusion → 3D)

```bash
# 샘플의 view 0 사용
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --sample_dir /home/joon/data/preprocessed/FaceLift_mouse/M5/000531 \
    --input_view_idx 0 \
    --mvdiffusion_checkpoint mouse_M5 \
    --gslrm_checkpoint M5t_E0_1_facelift \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/sample_531_1view

# 외부 이미지 사용
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --input_image /path/to/image.png \
    --mvdiffusion_checkpoint mouse_M5 \
    --gslrm_checkpoint M5t_E0_1_facelift \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/from_image
```

### 2.4 배치 처리

```bash
# 전체 데이터셋 (6-view)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --gslrm_checkpoint M5t_E0_1_facelift \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/batch_full

# 프레임 범위 지정
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --start_frame 0 --end_frame 100 --frame_step 1 \
    --gslrm_checkpoint M5t_E0_1_facelift \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/batch_0_100

# 1-view 배치 (MVDiffusion 포함)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --input_view_idx 0 \
    --start_frame 0 --end_frame 100 \
    --mvdiffusion_checkpoint mouse_M5 \
    --gslrm_checkpoint M5t_E0_1_facelift \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/batch_1view
```

### 2.5 연속 영상 생성

```bash
# 기본 (100 프레임 테스트)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.simple_temporal \
    --checkpoint M5t_E0_1_facelift \
    --config configs/base/gslrm_mouse.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --start_frame 0 --end_frame 100 \
    --output_dir outputs/temporal_test

# Test split 평가
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.simple_temporal \
    --checkpoint M5t_E0_1_facelift \
    --config configs/base/gslrm_mouse.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5t \
    --split data_mouse_test.txt \
    --output_dir outputs/temporal_test_split

# 다중 고정 각도
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.simple_temporal \
    --checkpoint M5t_E0_1_facelift \
    --config configs/base/gslrm_mouse.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --start_frame 0 --end_frame 100 \
    --fixed_angles 0 9 18 27 \
    --output_dir outputs/temporal_4angles
```

---

## 3. 상세 옵션

### 3.1 프레임 범위

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--start_frame` | 시작 인덱스 | None (처음) |
| `--end_frame` | 종료 인덱스 (exclusive) | None (끝) |
| `--frame_step` | 간격 | 1 |

**참고**: 전처리 시 이미 `frame_jump=5` 적용됨. `--frame_step 1`이 원본 5프레임 간격.

### 3.2 MVDiffusion 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--num_steps` | Diffusion 스텝 수 | 50 |
| `--guidance_scale` | CFG 스케일 | 3.0 |
| `--seed` | 랜덤 시드 | 42 |
| `--image_size` | 출력 해상도 | 512 |

### 3.3 렌더링 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--num_views` | 360° 분할 수 | 36 (10° 간격) |
| `--resolution` | 렌더링 해상도 | 384 |
| `--elevation` | 카메라 고도각 | 20.0 |
| `--radius` | 카메라 거리 | 2.7 |
| `--fps` | 영상 FPS | 24 |
| `--rotation_speed` | 회전 속도 배율 | 0.5 (절반) |
| `--fixed_angles` | 고정 각도 (뷰 인덱스) | [0] |

### 3.4 Export 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--save_gaussian` / `--no_gaussian` | Gaussian .ply/.npz 저장 | ✅ 저장 |
| `--save_rerun` / `--no_rerun` | Rerun .rrd 저장 | ✅ 저장 |
| `--save_first_only` | 첫 프레임만 export | False |

```bash
# Rerun 뷰어로 열기
rerun outputs/temporal_test/rerun/sequence.rrd

# 웹 뷰어 (SSH 환경)
rerun --web-viewer outputs/temporal_test/rerun/sequence.rrd
```

### 3.5 체크포인트 자동 탐색

**GS-LRM** (`--gslrm_checkpoint`, `--checkpoint`):

| 입력 | 예시 | 동작 |
|------|------|------|
| 실험 이름 | `M5t_E0_1_facelift` | `checkpoints/gslrm/{name}/` 탐색 |
| `pretrained` | `pretrained` | 원본 체크포인트 |
| 디렉토리 | `checkpoints/gslrm/M5t_E0_1/` | best.pt 또는 최신 ckpt 탐색 |
| 파일 | `path/to/ckpt.pt` | 직접 사용 |

**MVDiffusion** (`--mvdiffusion_checkpoint`):

| 입력 | 예시 | 동작 |
|------|------|------|
| 실험 이름 | `mouse_M5` | `checkpoints/mvdiffusion/{name}/` 탐색 |
| 체크포인트 | `mouse_M5/checkpoint-3500` | 해당 checkpoint 사용 |
| 전체 경로 | `/path/to/checkpoint-*` | 직접 사용 |

**탐색 우선순위**: `best.pt` > `best_psnr.pt` > 최신 `ckpt_*.pt` (GS-LRM), `unet_ema/` > `unet/` (MVDiffusion)

---

## 4. 출력 구조

### 4.1 run_e2e_inference 출력

```
outputs/batch_1view/
├── 000000/                      # 샘플별 폴더
│   ├── generated_views/         # MVDiffusion 생성 (1-view 입력 시)
│   │   ├── view_00.png
│   │   └── ...
│   ├── gaussians.ply            # 3D Gaussian
│   ├── gaussians.rrd            # Rerun 뷰어 파일
│   ├── comparison_grid.jpg      # GT vs Rendered
│   └── turntable_grid.jpg       # 다각도 렌더링
├── 000001/
└── ...
```

### 4.2 simple_temporal 출력

```
outputs/temporal_test/
├── turntable_first.mp4          # 첫 프레임 360° (절반 속도)
├── time_fixed.mp4               # 시간 변화 (고정 각도 0°)
├── time_fixed_angle9.mp4        # 고정 각도 90° (지정 시)
├── time_rotating.mp4            # 시간 + 회전 동시
├── full_all.mp4                 # 전체 (T × V 프레임)
├── grid_first.jpg               # Turntable 그리드 + 각도 레이블
├── grid_6view.mp4               # 입력 6-view 영상
├── gaussians/                   # Gaussian 파일
│   ├── frame_000000.ply
│   ├── frame_000000.npz
│   └── ...
└── rerun/
    └── sequence.rrd             # 타임라인 시퀀스
```

### 4.3 Grid 레이블

모든 grid 이미지에 정보 레이블 자동 추가:
- **Turntable**: `Views 0-5 | 0 deg - 50 deg | Elev: 20 deg`
- **Input**: `Input Cameras 0-2`

---

## 5. 성능 & 문제 해결

### 5.1 GPU 메모리

| 설정 | GPU 메모리 | 프레임당 시간 |
|------|-----------|---------------|
| resolution=384, num_views=36 | ~8GB | ~2초 |
| resolution=512, num_views=60 | ~12GB | ~4초 |
| resolution=384, num_views=120 | ~10GB | ~5초 |

### 5.2 문제 해결

| 문제 | 해결 |
|------|------|
| **Checkpoint not found** | `ls checkpoints/gslrm/` 확인, 실험 이름 정확히 입력 |
| **CUDA OOM** | `--resolution 256`, `--num_views 24` |
| **느린 렌더링** | `--frame_step 10`, `--num_views 18` |
| **검은 화면** | 체크포인트-config 일치 확인, `--radius 2.7` |
| **Ghosting** | 데이터셋 정규화 확인 (fx=549, trans=2.7) |

---

## 6. 참조

### 6.1 관련 문서

| 문서 | 내용 |
|------|------|
| [EXPERIMENT_QUICKSTART](./EXPERIMENT_QUICKSTART.md) | 학습 명령어 |
| [PREPROCESSING_REGISTRY](../datasets/PREPROCESSING_REGISTRY.md) | 데이터셋 SSOT |
| [M5_SERIES_SPEC](../datasets/M5_SERIES_SPEC.md) | M5 시리즈 상세 |

### 6.2 명령어 요약

| 목적 | 핵심 인자 |
|------|----------|
| 단일 샘플 (6-view) | `--sample_dir` |
| 단일 샘플 (1-view) | `--sample_dir --input_view_idx N --mvdiffusion_checkpoint` |
| 배치 (6-view) | `--data_dir` |
| 배치 (1-view) | `--data_dir --input_view_idx N --mvdiffusion_checkpoint` |
| 프레임 범위 | `--start_frame X --end_frame Y --frame_step Z` |
| 연속 영상 | `simple_temporal.py --checkpoint --data_dir` |

---

*FaceLift E2E Inference Guide v2.0 | 2026-01-29*
