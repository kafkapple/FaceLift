# Mouse Dataset Guide

> MouseViewDataset for FaceLift GS-LRM training with 6-view mouse images.
> **Updated**: 2026-02-11

---

## 1. Overview

`MouseViewDataset` handles multi-view mouse images with:
- 6 synchronized camera views (fixed order)
- Single input view for reconstruction (configurable)
- Optional data augmentation for limited real data
- Camera distance normalization, PP correction

## 2. Implementation Location

| Component | Location |
|-----------|----------|
| **Main implementation** | `mouse_extensions/data/mouse_dataset.py` (698 lines) |
| **Deprecation stub** | `gslrm/data/mouse_dataset.py` (redirects to above) |
| **Preprocessing utils** | `mouse_extensions/data/preprocessing.py` |
| **PP correction** | `mouse_extensions/scripts/solutions/pp_correction_integration.py` |

> **Note**: Original `gslrm/data/mouse_dataset.py` is now a 30-line compatibility shim.
> Always import from `mouse_extensions.data`:
> ```python
> from mouse_extensions.data import MouseViewDataset
> ```

## 3. Key Differences from RandomViewDataset

| Feature | RandomViewDataset | MouseViewDataset |
|---------|-------------------|------------------|
| Views | 32 (random sample) | **6** (fixed order) |
| View selection | Random each epoch | Fixed `[0,1,2,3,4,5]` |
| Camera distance | Uniform | **Normalized** (target=2.7) |
| Augmentation | None | Optional (brightness, contrast) |
| PP correction | None | Auto-applied if available |
| Background | Fixed | Configurable (white/black/random/three_choices) |

## 4. Configuration

### Dataset Config (in YAML)

```yaml
training:
  dataset:
    dataset_path: /path/to/data_mouse_train.txt
    image_size: 512
    num_views: 6
    num_input_views: 4          # or 1 for single-view mode
    random_view_selection: true  # random subset of 6 views
    target_has_input: false
    background_color: three_choices  # white, black, random, three_choices
    augmentation: false          # brightness/contrast augmentation
```

### Mouse-Specific Config

```yaml
mouse:
  use_mouse_dataset: true
  normalize_cameras: false
  target_camera_distance: 2.7
```

## 5. Data Structure

```
sample_XXXXXX/
├── images/
│   ├── cam_000.png    # View 0 (input view)
│   ├── cam_001.png    # View 1
│   ├── ...
│   └── cam_005.png    # View 5
└── opencv_cameras.json
```

### Camera JSON Format

```json
{
  "cam_000": {"w2c": [...], "K": [...]},
  "cam_001": {"w2c": [...], "K": [...]},
  ...
}
```

## 6. Usage

```python
from mouse_extensions.data import MouseViewDataset
from easydict import EasyDict as edict
import yaml

with open("configs/base/gslrm_mouse.yaml") as f:
    config = edict(yaml.safe_load(f))

dataset = MouseViewDataset(config, split="train")
sample = dataset[0]
# sample["image"]: [V, C, H, W]
# sample["c2w"]:   [V, 4, 4]
# sample["fxfycxcy"]: [V, 4]
```

## 7. Related Docs

| Doc | Content |
|-----|---------|
| [PREPROCESSING_REGISTRY](../datasets/PREPROCESSING_REGISTRY.md) | Dataset preprocessing SSOT |
| [EXPERIMENT_CONFIG_GUIDE](../experiments/EXPERIMENT_CONFIG_GUIDE.md) | Config schema reference |
| [M5_SERIES_SPEC](../datasets/M5_SERIES_SPEC.md) | M5 dataset details |

---

*Mouse Dataset Guide | Updated: 2026-02-11*
