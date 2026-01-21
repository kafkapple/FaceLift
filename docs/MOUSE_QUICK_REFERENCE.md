# FaceLift Mouse Extension - Quick Reference

> **Last Updated**: 2026-01-21
> **Full Documentation**: Obsidian `30_Projects/_CODES/code_Face_Lift/docs/`

---

## 1. Training

```bash
# Basic training
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_E1.yaml

# Multi-GPU
torchrun --standalone --nproc_per_node=4 train_gslrm.py --config configs/mouse/D6_E1.yaml
```

---

## 2. Inference

```bash
# From 6-view sample
./scripts/run_inference.sh data/D6-3/sample_000100

# Single image with Zero123++
./scripts/run_inference.sh mouse.png --use_zero123pp

# Manual inference
python inference_mouse.py \
    --sample_dir data/D6-3/sample_000100 \
    --checkpoint checkpoints/gslrm/mouse/ \
    --output_dir outputs/inference/
```

---

## 3. Key Configuration

### 3.1 Turntable Visualization

```yaml
visualization:
  turntable:
    num_views: 64           # Grid views (8x8)
    resolution: 384
    elevation: 20
    radius: 2.7
    inference_views: 150    # Video frames
    fps: 30
```

### 3.2 Camera Exclusion (Ablation)

```yaml
training:
  dataset:
    exclude_camera_indices: [3]     # Exclude camera 3
    # include_camera_indices: [0,1,2,4,5]  # Or explicit include
```

### 3.3 Loss Configuration

```yaml
training:
  losses:
    l2_loss_weight: 1.0
    perceptual_loss_weight: 0.5
    mask_mode: "gt"              # none, gt, alpha
    alpha_loss_weight: 0.1       # Optional
```

---

## 4. Key Code Locations

| Feature | File | Line |
|---------|------|------|
| Projection Matrix | `gslrm/model/gaussians_renderer.py` | :287-306 |
| Loss Computation | `mouse_extensions/model/loss_extensions.py` | - |
| Turntable Render | `gslrm/model/gaussians_renderer.py` | :974 |
| Camera Exclusion | `gslrm/data/mouse_dataset.py` | :254-270 |

---

## 5. Preprocessing

```bash
# D6-3 preprocessing (recommended)
python -m mouse_extensions.preprocessing.preprocessor_d6 \
    --method D6-3 \
    --input_dir /path/to/raw \
    --output_dir /path/to/D6-3

# Verify
python -m mouse_extensions.preprocessing.format_validator /path/to/D6-3
```

---

## 6. Documentation Map

| Topic | Location |
|-------|----------|
| Full Theory & Analysis | Obsidian `260121_Comprehensive_Review_Plan.md` |
| Mask/Loss Guide | Obsidian `260121_Mask_Loss_Educational_Guide.md` |
| Paper Comparison | Obsidian `260121_Paper_Settings_Comparison.md` |
| Preprocessing Registry | `docs/PREPROCESSING_REGISTRY.md` |

---

*Quick Reference - See Obsidian for detailed documentation*

---

## 7. Camera Trajectory Visualization

### 7.1 Turntable 설정

```yaml
visualization:
  turntable:
    # Basic settings
    num_views: 64           # Total views in grid
    resolution: 384         # Render resolution
    elevation: 20           # Camera elevation (degrees)
    radius: 2.7             # Distance from origin
    grid_rows: 8
    grid_cols: 8
    
    # Trajectory mode (NEW)
    trajectory_mode: "turntable"  # turntable, spiral, figure8, arc, dataset_cameras
    elevation_end: 60             # End elevation for spiral/arc
    
    # Dataset camera views (NEW)
    include_dataset_views: true   # Include 6 dataset cameras in grid
    save_dataset_views: true      # Save separate dataset_views_{uid}.jpg
    
    # Inference video settings
    inference_views: 150
    fps: 30
```

### 7.2 Trajectory 모드

| Mode | 설명 |
|------|------|
| `turntable` | 고정 elevation, 360° 회전 (기본) |
| `spiral` | Elevation 점진 변화 + 1.5회전 |
| `figure8` | 8자 패턴 |
| `arc` | 고정 azimuth, elevation 변화 |
| `dataset_cameras` | 데이터셋 6개 카메라 간 보간 이동 |

### 7.3 Dataset Views 시각화

`include_dataset_views: true` 설정 시:
- Turntable grid 처음 6개 위치에 데이터셋 카메라 뷰 배치
- 각 뷰에 "Cam 0", "Cam 1", ... 오버레이
- gt_vs_pred 이미지와 직접 비교 가능

`save_dataset_views: true` 설정 시:
- `dataset_views_{uid}.jpg` 별도 저장
- 6개 카메라 렌더링만 가로 연결

### 7.4 Dataset Trajectory 영상

```python
from gslrm.model.gaussians_renderer import render_dataset_trajectory

# 6개 카메라 간 보간 이동 영상 생성
frames, segments = render_dataset_trajectory(
    gaussians,
    dataset_c2ws,      # [6, 4, 4]
    dataset_fxfycxcy,  # [6, 4]
    num_views=150,
    camera_order=[0, 1, 2, 3, 4, 5],  # 방문 순서
    loop=True,         # 처음으로 돌아감
    show_overlay=True  # "Cam 0 → Cam 1" 텍스트
)
# frames: [150, H, W, 3]
# segments: [(0, 25, 0, 1), (25, 50, 1, 2), ...]
```

---
