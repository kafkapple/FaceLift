# Mouse Extensions for FaceLift (v1.2.0)

Multi-model extensions for mouse data experiments.

## 📁 Structure

```
mouse_extensions/               # Project root level
├── __init__.py                 # Unified entry (re-exports all)
├── data/                       # 📊 Data Preprocessing
│   ├── __init__.py
│   └── preprocessing.py        # Camera normalization, backgrounds
├── model/                      # 🧠 Model Training
│   ├── __init__.py
│   ├── alpha_renderer.py       # Alpha channel rendering
│   ├── loss_extensions.py      # Mask computation, ghost metrics
│   └── visualization.py        # Training visualization
├── utils/                      # 🔧 Utilities
│   ├── __init__.py
│   └── logging_utils.py        # WandB logging
├── apply_patches.py            # Patch script
├── PATCHES.md                  # Patch documentation
├── REFACTORING_GUIDE.md        # Refactoring how-to
└── README.md
```

## 🚀 Quick Start

```python
# Recommended: Direct submodule import
from mouse_extensions.data import preprocess_cameras, get_bg_color
from mouse_extensions.model import compute_mask_from_config, MaskType
from mouse_extensions.utils import get_experiment_info

# Legacy: Unified import (backward compatible)
from mouse_extensions import preprocess_cameras, compute_mask_from_config
```

## 📦 Modules

### data/ - Data Preprocessing
| Function | Description |
|----------|-------------|
| `normalize_camera_distance()` | Scale cameras to target radius (2.7) |
| `normalize_cameras_to_z_up()` | Align up direction to Z-axis |
| `preprocess_cameras()` | Full preprocessing pipeline |
| `get_bg_color()` | Parse background color config |
| `PreprocessingConfig` | Preprocessing configuration class |

### model/ - Model Training
| Function | Description |
|----------|-------------|
| `compute_mask_from_config()` | Compute mask (GT/RGB/Alpha) |
| `compute_ghost_metrics()` | Ghosting analysis metrics |
| `create_threshold_comparison()` | Threshold visualization |
| `MaskType` | Mask type enum |
| `render_opencv_cam_with_alpha()` | Alpha-enabled rendering |

### utils/ - Utilities
| Function | Description |
|----------|-------------|
| `get_experiment_info()` | Extract WandB config info |
| `get_wandb_log_dict()` | Training metrics log dict |
| `get_validation_log_dict()` | Validation metrics log dict |

## 🎯 Used By

- `gslrm/` - GS-LRM model training
- `mvdiffusion/` - MVDiffusion model (planned)
- `train_gslrm.py` - Training script

## 📝 Config Example

```yaml
mouse:
  normalize_cameras: true
  target_camera_distance: 2.7
  normalize_to_z_up: true

training:
  dataset:
    random_view_selection: true
    background_color: white
  losses:
    use_rendered_alpha_mask: true
    alpha_mask_threshold: 0.5
```

## 🔄 Version History

| Version | Changes |
|---------|---------|
| 1.2.0 | Submodule structure, root location |
| 1.1.0 | Added preprocessing module |
| 1.0.0 | Initial release |
