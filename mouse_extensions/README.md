# Mouse Extensions for FaceLift (v1.3.1)

Multi-model extensions for mouse data experiments.

## 📁 Module Structure

```
mouse_extensions/                   # 🐭 Mouse-specific extensions
├── __init__.py                     # Unified entry point
├── apply_patches.py                # Patch application script
├── PATCHES.md                      # Patch documentation
├── REFACTORING_GUIDE.md            # Refactoring how-to
├── README.md                       # This file
│
├── data/                           # 📊 Data Preprocessing
│   ├── __init__.py
│   └── preprocessing.py            # Camera normalization, backgrounds
│
├── model/                          # 🧠 Model Training Extensions
│   ├── __init__.py
│   ├── alpha_renderer.py           # Alpha channel rendering (diff_gauss)
│   ├── loss_extensions.py          # Mask computation, ghost metrics
│   └── visualization.py            # Training visualization
│
├── preprocessing/                  # 🔄 Data Pipeline Scripts
│   ├── __init__.py
│   ├── convert_unified.py          # Unified data converter
│   ├── run_pipeline.py             # Full preprocessing pipeline
│   └── split_dataset.py            # Train/val split generation
│
├── scripts/                        # 🛠️ Utility Scripts
│   ├── __init__.py
│   ├── preprocess.py               # Preprocessing entry point
│   └── validate_preprocessing.py   # Data validation
│
└── utils/                          # 🔧 Utilities
    ├── __init__.py
    ├── logging_utils.py            # WandB logging helpers
    └── visualize_masks.py          # Mask visualization tools
```

---

## 🔗 Integration with Original Code

### Plugin Points

```
FaceLift/
├── gslrm/
│   ├── data/
│   │   ├── dataset.py              # Original dataset (humans)
│   │   └── mouse_dataset.py        # ⬅️ Imports from mouse_extensions/data
│   │                               #    (556 lines, refactored v1.3.1)
│   └── model/
│       ├── gslrm.py                # ⬅️ Imports from mouse_extensions/model
│       └── gaussians_renderer.py   # ⬅️ Patched for alpha (PATCHES.md)
│
├── train_gslrm.py                  # ⬅️ Imports from mouse_extensions/utils
│
├── configs/
│   └── mouse/                      # Mouse experiment configs (7 experiments)
│       ├── gslrm_v15_reproduce.yaml
│       ├── gslrm_v48_v13_square.yaml
│       ├── gslrm_v49_no_mask.yaml
│       ├── gslrm_v54_paper.yaml
│       ├── gslrm_v55_paper_mask.yaml
│       ├── gslrm_v56_v13_nomask.yaml
│       └── gslrm_v57_6views_max.yaml
│
└── mouse_extensions/               # This module (~3000 lines)
```

### Import Flow

```python
# train_gslrm.py
from mouse_extensions.utils import get_experiment_info

# gslrm/data/mouse_dataset.py (REQUIRED - no fallback)
from mouse_extensions.data import (
    pil_to_np,
    normalize_camera_distance,
    normalize_camera_distance_with_intrinsics,
    normalize_cameras_to_y_up,
    normalize_cameras_to_z_up,
    get_bg_color,
    preprocess_cameras,
    PreprocessingConfig,
)

# gslrm/model/gslrm.py
from mouse_extensions.model import (
    compute_mask_from_config,
    compute_ghost_metrics,
    MaskType,
)
```

---

## 📦 Module Details

### data/ - Data Preprocessing

| Function | Description | Used By |
|----------|-------------|---------|
| `pil_to_np()` | PIL to numpy with RGBA support | mouse_dataset.py |
| `normalize_camera_distance()` | Scale cameras to target radius | mouse_dataset.py |
| `normalize_cameras_to_z_up()` | Align up direction to Z-axis | mouse_dataset.py |
| `preprocess_cameras()` | Full preprocessing pipeline | preprocessing scripts |
| `get_bg_color()` | Parse background color config | mouse_dataset.py |
| `PreprocessingConfig` | Preprocessing configuration dataclass | scripts |

### model/ - Model Training Extensions

| Function | Description | Used By |
|----------|-------------|---------|
| `compute_mask_from_config()` | Compute mask (GT/RGB/Alpha based) | gslrm.py |
| `compute_ghost_metrics()` | Ghosting analysis metrics | gslrm.py |
| `create_threshold_comparison()` | Threshold visualization | training |
| `MaskType`, `MaskConfig` | Mask configuration enums/classes | configs |
| `render_opencv_cam_with_alpha()` | Alpha-enabled rendering | gaussians_renderer.py |

### utils/ - Utilities

| Function | Description | Used By |
|----------|-------------|---------|
| `get_experiment_info()` | Extract WandB config info | train_gslrm.py |
| `get_wandb_log_dict()` | Training metrics log dict | train_gslrm.py |
| `get_validation_log_dict()` | Validation metrics log dict | train_gslrm.py |

---

## 🚀 Quick Start

```python
# Recommended: Direct submodule import
from mouse_extensions.data import preprocess_cameras, get_bg_color
from mouse_extensions.model import compute_mask_from_config, MaskType
from mouse_extensions.utils import get_experiment_info

# Legacy: Unified import (backward compatible)
from mouse_extensions import preprocess_cameras, compute_mask_from_config
```

---

## 📝 Config Integration

```yaml
# configs/mouse/gslrm_v*.yaml

mouse:
  use_mouse_dataset: true           # Enable MouseViewDataset
  normalize_cameras: false          # Camera normalization (already preprocessed)
  target_camera_distance: 0.0       # Target radius (0 = skip)
  normalize_to_z_up: true           # Z-up alignment
  auto_generate_mask: false         # Auto mask from RGB
  mask_threshold: 250               # Background threshold

training:
  dataset:
    random_view_selection: false    # Random vs fixed view selection
    background_color: white
  losses:
    masked_l2_loss: true            # Foreground-only L2 loss
    use_rendered_alpha_mask: false  # Use alpha for mask
    alpha_mask_threshold: 0.5       # Alpha threshold
```

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| **1.3.1** | **2025-01-15** | **Removed fallback functions from mouse_dataset.py (-307 lines)** |
| 1.3.0 | 2025-01-15 | Documentation update, integration mapping |
| 1.2.0 | 2025-01-14 | Submodule structure, root location |
| 1.1.0 | 2025-01-13 | Added preprocessing module |
| 1.0.0 | 2025-01-10 | Initial release |

---

## ✅ Refactoring Status

### Completed (v1.3.1)
- [x] **P0: Remove fallback duplication** - 307 lines removed from mouse_dataset.py
  - `pil_to_np`, `normalize_camera_distance`, `get_bg_color` etc.
  - mouse_extensions is now a **required** dependency

### Future (Optional)
- [ ] P1: Move MouseViewDataset to mouse_extensions/data/dataset.py
- [ ] P2: Centralize mouse config parsing

---

## 📊 Code Statistics

| Component | Lines | Description |
|-----------|-------|-------------|
| mouse_extensions/ | ~3000 | All mouse-specific code |
| gslrm/data/mouse_dataset.py | 556 | Dataset classes (refactored) |
| configs/mouse/ | 7 files | Experiment configurations |

**Total mouse-specific code**: ~3500 lines (well modularized)

---

## 🔧 Diagnostic & Solution Tools (2026-01-17)

### Principal Point Bug Fix

**문제**: v12/v13 데이터셋에 cx=cy=256 강제 설정 버그
- 실제 PP는 `_transform.scaled_cx/scaled_cy`에 기록
- 11-13° 광선 방향 오류 → 고스팅 아티팩트

### scripts/diagnostics/

| Script | Purpose |
|--------|---------|
| `validate_cameras.py` | 일반 카메라 검증 도구 |
| `diagnose_facelift_cameras.py` | FaceLift 전용 진단 (PP 버그 검출) |

```bash
# PP 버그 진단 실행
python mouse_extensions/scripts/diagnostics/diagnose_facelift_cameras.py \
    --dataset_path /path/to/data_mouse_train.txt \
    --max_samples 50
```

### scripts/solutions/

| Module | Purpose |
|--------|---------|
| `camera_normalization.py` | Per-sample 카메라 정규화 |
| `principal_point_correction.py` | PP 검증 및 보정 |
| `pp_correction_integration.py` | 데이터셋 로더 통합 모듈 |

### patches/

| Script | Target | Purpose |
|--------|--------|---------|
| `patch_mouse_dataset_pp_correction.py` | gslrm/data/mouse_dataset.py | PP 보정 기능 추가 |

**적용 방법**:
```bash
cd /home/joon/dev/FaceLift
python mouse_extensions/patches/patch_mouse_dataset_pp_correction.py
```

**Config 설정**:
```yaml
mouse:
  pp_correction: "crop"  # Options: "none", "actual_pp", "crop"
```

---

## 📁 코드 위치 원칙

### 왜 mouse_dataset.py는 gslrm/data/에 있는가?

| 코드 | 위치 | 이유 |
|------|------|------|
| MouseViewDataset | `gslrm/data/` | 학습 파이프라인 핵심 (DataLoader 호환) |
| 전처리 유틸 | `mouse_extensions/data/` | 재사용 가능한 함수들 |
| 패치 | `mouse_extensions/patches/` | 원본 최소 변경 원칙 |
| 진단 도구 | `mouse_extensions/scripts/diagnostics/` | 독립 실행 스크립트 |

**원칙**: 핵심 코드는 원본 위치 유지, 확장/패치는 mouse_extensions에 모듈화
