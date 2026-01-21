# Mouse Extensions for FaceLift (v2.0.0)

Multi-model extensions for mouse data experiments.

**Updated**: 2026-01-21

---

## Module Structure

```
mouse_extensions/                      # ~4500 lines
├── __init__.py                        # Unified entry point
├── README.md                          # This file
│
├── preprocessing/                     # ★ Data Pipeline
│   ├── preprocess.py                  # Unified preprocessor (D7/D8)
│   ├── presets.py                     # Preset definitions
│   ├── camera_normalizer.py           # Camera normalization
│   ├── center_estimation.py           # 3D triangulation
│   ├── data_loader.py                 # Raw data loading
│   └── format_validator.py            # Format validation
│
├── model/                             # ★ Model Training
│   ├── loss_extensions.py             # Mask loss, ghost metrics (22KB)
│   ├── visualization_extensions.py    # Training visualization (20KB)
│   ├── alpha_renderer.py              # Alpha rendering
│   ├── gaussian_pruning.py            # Pruning utilities
│   └── gslrm_patches.py               # Model patches
│
├── scripts/                           # Utility Scripts
│   ├── diagnostics/                   # Camera validation
│   └── solutions/                     # PP correction
│
└── utils/                             # Utilities
    └── logging_utils.py               # WandB helpers
```

---

## Quick Start

### Preprocessing

```bash
# D8 (recommended)
python -m mouse_extensions.preprocessing.preprocess \
    --preset D8 --input-dir /path/to/raw --output-dir /path/to/D8

# D7.1 (standard)
python -m mouse_extensions.preprocessing.preprocess \
    --preset D7.1 --input-dir /path/to/raw --output-dir /path/to/D7_1
```

### Training

```bash
# Format: train_gslrm.py -d {dataset} -e {experiment}
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E4_3_alpha_aggressive
```

---

## Key Modules

### preprocessing/

| File | Purpose |
|------|---------|
| `preprocess.py` | Unified D7/D8 preprocessor |
| `presets.py` | D7.1, D7.2, D8 preset definitions |
| `camera_normalizer.py` | fx→549, trans→2.7 normalization |
| `center_estimation.py` | 3D triangulation for crop center |

### model/

| File | Purpose |
|------|---------|
| `loss_extensions.py` | Mask computation, alpha loss, ghost metrics |
| `visualization_extensions.py` | Training visual, mask overlay, error map |
| `gslrm_patches.py` | Model enhancement patches |

---

## Config System

```
configs/
├── base/gslrm_mouse.yaml       # Base config
├── datasets/                    # Dataset-specific
│   ├── D7_1.yaml, D7_2.yaml
│   ├── D8.yaml, D8_1.yaml
│   └── v13.yaml
└── experiments/                 # Experiment configs
    ├── E1_*  # Paper Baseline
    ├── E2_*  # Mask Mode
    ├── E3_*  # View Count
    ├── E4_*  # Alpha Tuning
    └── E5_*  # Loss Ablation
```

**Usage**: `train_gslrm.py -d {dataset} -e {experiment}`

See `configs/experiments/README.md` for full experiment matrix.

---

## Integration Points

```
gslrm/data/mouse_dataset.py
    └── from mouse_extensions.data import ...

gslrm/model/gslrm.py
    └── from mouse_extensions.model import ...

train_gslrm.py
    └── from mouse_extensions.utils import ...
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **2.0.0** | **2026-01-21** | Modular config, D7/D8 presets, experiment matrix |
| 1.3.1 | 2025-01-15 | Removed fallback functions |
| 1.2.0 | 2025-01-14 | Submodule structure |
| 1.0.0 | 2025-01-10 | Initial release |

---

## Related Documentation

- `docs/PREPROCESSING_REGISTRY.md` - Dataset registry (SSOT)
- `configs/experiments/README.md` - Experiment matrix
- `docs/CONFIG_MODULAR.md` - Config system guide
- `docs/ALPHA_MASK_COMPLETE_GUIDE.md` - Mask tuning guide

---

*FaceLift Mouse Project | v2.0.0 | 2026-01-21*
