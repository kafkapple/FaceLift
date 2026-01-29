# E2E Inference Guide (추론 가이드)

> **Updated**: 2026-01-29 (v3.0 - 통합 파이프라인)
> **관련 문서**: [EXPERIMENT_QUICKSTART](./EXPERIMENT_QUICKSTART.md) | [PREPROCESSING_REGISTRY](../datasets/PREPROCESSING_REGISTRY.md)

---

## 1. 개요

### 1.1 통합 파이프라인

```
python -m mouse_extensions.inference.run --config configs/inference/default.yaml
```

```
Input (1-view or 6-view)
        │
        ▼
┌───────────────┐
│  MVDiffusion  │ ← view_idx 설정 시 자동 활성화
│  (optional)   │
└───────┬───────┘
        │ 6-view
        ▼
┌───────────────┐
│    GS-LRM     │
└───────┬───────┘
        │ Gaussians
        ▼
┌───────────────┐
│   Renderer    │ → turntable, fixed angle
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Exporter    │ → .ply, .rrd, .mp4, .jpg
└───────────────┘
```

### 1.2 권장 체크포인트

| 모델 | 자동 탐색 값 |
|------|-------------|
| **GS-LRM** | `M5t_E0_1_facelift` |
| **MVDiffusion** | `mouse_M5` |
| GS-LRM (pretrained) | `pretrained` |

---

## 2. Quick Start

### 2.1 Config 파일 사용 (권장)

```bash
# 기본 설정으로 실행
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.inference.run \
    --config configs/inference/default.yaml

# 1-view 모드 프리셋
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.inference.run \
    --config configs/inference/m5_1view.yaml
```

### 2.2 CLI 인자로 오버라이드

```bash
# 배치 처리 (6-view → 3D)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.inference.run \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --checkpoint M5t_E0_1_facelift \
    --start_frame 0 --end_frame 100 \
    --output_dir outputs/batch_test

# 1-view 모드 (MVDiffusion + GS-LRM)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.inference.run \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --view_idx 0 \
    --mvdiffusion_checkpoint mouse_M5 \
    --checkpoint M5t_E0_1_facelift \
    --start_frame 0 --end_frame 100 \
    --output_dir outputs/batch_1view

# 단일 샘플
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.inference.run \
    --sample_dir /home/joon/data/preprocessed/FaceLift_mouse/M5/000531 \
    --checkpoint M5t_E0_1_facelift \
    --output_dir outputs/sample_531
```

### 2.3 Dotlist 오버라이드

```bash
# Config + 부분 오버라이드
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.inference.run \
    --config configs/inference/default.yaml \
    input.data_dir=/path/to/data \
    input.frame_range.end=50 \
    output.video.fps=30
```

---

## 3. Config 구조

```yaml
# configs/inference/default.yaml

input:
  mode: batch                   # single_image | single_sample | batch
  data_dir: null
  sample_dir: null
  image_path: null
  view_idx: null                # null=6-view, 0-5=1-view+MVDiffusion
  frame_range:
    start: null
    end: null
    step: 1

pipeline:
  mvdiffusion:
    enabled: auto               # auto | true | false
    checkpoint: mouse_M5
    num_steps: 50
    guidance_scale: 3.0
  gslrm:
    checkpoint: M5t_E0_1_facelift
    config: configs/base/gslrm_mouse.yaml

rendering:
  resolution: 384
  num_views: 36
  elevation: 20.0
  radius: 2.7
  fixed_angles: [0]

output:
  dir: outputs/inference
  gaussian: true
  rerun: true
  video:
    enabled: true
    fps: 24
    types: [turntable, time_fixed, time_rotating, grid_6view]
  grid:
    enabled: true
    cols: 6
```

### 3.1 Config 상속

```yaml
# configs/inference/m5_1view.yaml
_base_: default.yaml

input:
  view_idx: 0
  frame_range:
    end: 100

pipeline:
  mvdiffusion:
    enabled: true
```

---

## 4. CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--config` | Config 파일 경로 | `configs/inference/default.yaml` |
| `--data_dir` | 배치 모드 데이터 경로 | - |
| `--sample_dir` | 단일 샘플 경로 | - |
| `--input_image` | 단일 이미지 경로 | - |
| `--view_idx` | 입력 뷰 인덱스 (0-5) | null (6-view) |
| `--start_frame` | 시작 프레임 | - |
| `--end_frame` | 끝 프레임 | - |
| `--frame_step` | 프레임 간격 | 1 |
| `--checkpoint` | GS-LRM 체크포인트 | - |
| `--mvdiffusion_checkpoint` | MVDiffusion 체크포인트 | - |
| `--output_dir` | 출력 디렉토리 | - |
| `--resolution` | 렌더링 해상도 | 384 |
| `--num_views` | Turntable 뷰 수 | 36 |
| `--fps` | 영상 FPS | 24 |
| `--no_video` | 영상 출력 비활성화 | - |
| `--no_gaussian` | Gaussian 저장 비활성화 | - |
| `--no_rerun` | Rerun 저장 비활성화 | - |

---

## 5. 출력 구조

```
outputs/inference/
├── turntable_first.mp4          # 첫 프레임 360°
├── time_fixed.mp4               # 시간 변화 (고정 각도)
├── time_rotating.mp4            # 시간 + 회전
├── full_all.mp4                 # 전체 (T × V)
├── grid_6view.mp4               # 입력 6-view 영상
├── grid_first.jpg               # Turntable 그리드
├── gaussians/
│   ├── frame_000000.ply
│   ├── frame_000001.ply
│   └── ...
└── rerun/
    └── sequence.rrd             # Rerun 뷰어 파일
```

---

## 6. 성능 & 문제 해결

### 6.1 GPU 메모리

| 설정 | GPU 메모리 | 프레임당 시간 |
|------|-----------|---------------|
| resolution=384, num_views=36 | ~8GB | ~2초 |
| resolution=512, num_views=60 | ~12GB | ~4초 |
| + MVDiffusion | +4GB | +3초 |

### 6.2 문제 해결

| 문제 | 해결 |
|------|------|
| **CUDA OOM** | `--resolution 256`, `--no_rerun` |
| **느린 처리** | `--frame_step 10`, `--num_views 18` |
| **Checkpoint not found** | `ls checkpoints/gslrm/` 확인 |
| **검은 화면** | `--radius 2.7`, 데이터셋 정규화 확인 |

---

## 7. 모듈 구조

```
mouse_extensions/inference/
├── modules/
│   ├── mvdiffusion.py      # 1-view → 6-view
│   ├── gslrm.py            # 6-view → Gaussians
│   ├── renderer.py         # Gaussians → images
│   └── exporter.py         # save outputs
├── unified_pipeline.py      # 파이프라인 오케스트레이션
├── run.py                   # CLI 진입점
└── checkpoint_utils.py      # 체크포인트 자동 탐색
```

---

## 8. 참조

| 문서 | 내용 |
|------|------|
| [EXPERIMENT_QUICKSTART](./EXPERIMENT_QUICKSTART.md) | 학습 명령어 |
| [PREPROCESSING_REGISTRY](../datasets/PREPROCESSING_REGISTRY.md) | 데이터셋 SSOT |
| [M5_SERIES_SPEC](../datasets/M5_SERIES_SPEC.md) | M5 시리즈 상세 |

---

*FaceLift E2E Inference Guide v3.0 | 2026-01-29*
