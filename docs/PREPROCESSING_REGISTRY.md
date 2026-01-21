# FaceLift Mouse Preprocessing Registry

> **Version**: v3.1 (2026-01-21)
> **Single Source of Truth** for all preprocessing configurations

---

## Quick Reference

### Active Presets (D7 Paradigm)

| Preset | Scale Mode | fx/fy | Ray Error | Status |
|--------|------------|-------|-----------|--------|
| **D7.1** | individual | 549/549 (exact) | ~0° | **★ RECOMMENDED** |
| **D7.2** | average | ~548/~550 | ~0.2° | Alternative |
| **D7** | fx_only | 549/549 (fy forced) | ~0.4° | Current Production |

### Camera Parameter Consistency (2026-01-21 분석)

| 데이터셋 | fx/fy | cx/cy | w2c | 프레임 간 일관성 |
|----------|-------|-------|-----|-----------------|
| **D7_1** | 549/549 | **256/256 고정** | 동일 | ✅ **100% 동일** |
| D6-3 | 549/549 | 가변 (±100px) | 동일 | ⚠️ PP만 변동 |

**결론**: D7_1은 모든 프레임에서 카메라 파라미터 완전 동일 (고정 셋업)

### Deprecated Presets

| Preset | Issue | Ray Error |
|--------|-------|-----------|
| D1-D4 | PP=256 forced | 5-13° |
| D6-1/2/3 | Object-centered | varies |

---

## 1. System Architecture

### Unified Entry Point
```bash
python -m mouse_extensions.preprocessing.preprocess --preset D7.1 \
    --input-dir /path/to/raw \
    --output-dir /path/to/D7_1
```

### Modular Config System
```
configs/mouse/_modular/
├── base/
│   ├── model.yaml      # Fixed GS-LRM architecture
│   └── runtime.yaml    # Training runtime settings
├── schemas/
│   ├── 5v_alpha.yaml   # 5-view with alpha mask (★ recommended)
│   ├── 4v_alpha.yaml   # 4-view with alpha mask
│   ├── 5v_nomask.yaml  # 5-view no mask (stable)
│   ├── 6v_alpha.yaml   # 6-view with alpha mask
│   └── 4v_random.yaml  # 4-view random selection
├── datasets/
│   ├── D7.yaml         # Current production
│   ├── D7_t.yaml       # Temporal split
│   ├── D7_1.yaml       # Individual scale (recommended)
│   └── D7_2.yaml       # Average scale
└── generate_config.py  # Config generator
```

---

## 2. D7 Preprocessing Details

### 2.1 Common Pipeline
1. Load raw images and camera parameters
2. Compute scale factor to achieve fx=549
3. Shift image to center PP at (256, 256)
4. Normalize camera distance to 2.7
5. Save in FaceLift format

### 2.2 Scale Mode Differences

**D7 (fx_only)**:
```python
scale = target_fx / orig_fx
# fy becomes: orig_fy * scale ≈ 551.5
# But recorded as: fy = 549 (forced!)
```

**D7.1 (individual)** [★ RECOMMENDED]:
```python
scale_x = target_fx / orig_fx
scale_y = target_fy / orig_fy
# fx = 549 (exact)
# fy = 549 (exact)
# Anisotropic: ~0.6% vertical compression
```

**D7.2 (average)**:
```python
scale = (scale_x + scale_y) / 2
# fx ≈ 548.1
# fy ≈ 550.0
# Isotropic scaling
```

---

## 3. Preprocessing Commands

### D7.1 (Recommended)
```bash
cd /home/joon/dev/FaceLift
/home/joon/anaconda3/envs/facelift/bin/python \
    -m mouse_extensions.scripts.preprocess_D7_pp_centered \
    --data-dir /home/joon/data/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --camera-pkl /home/joon/data/markerless_mouse_1_nerf/new_cam.pkl \
    --frame-interval 5 \
    --scale-mode individual
```

### D7.2 (Alternative)
```bash
cd /home/joon/dev/FaceLift
/home/joon/anaconda3/envs/facelift/bin/python \
    -m mouse_extensions.scripts.preprocess_D7_pp_centered \
    --data-dir /home/joon/data/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D7_2 \
    --camera-pkl /home/joon/data/markerless_mouse_1_nerf/new_cam.pkl \
    --frame-interval 5 \
    --scale-mode average
```

---

## 4. Config Generation

### Generate Experiment Config
```bash
cd /home/joon/dev/FaceLift/configs/mouse/_modular

# List options
python generate_config.py --list

# Generate D7.1 + 5v_alpha
python generate_config.py --dataset D7_1 --schema 5v_alpha

# Output: configs/mouse/D7_1_E1_5v_alpha.yaml
```

### Available Schemas
| Schema | Views | Mask | Description |
|--------|-------|------|-------------|
| 5v_alpha | 5 | alpha (0.5) | ★ Recommended |
| 4v_alpha | 4 | alpha (0.5) | Fewer views |
| 5v_nomask | 5 | none | Stable baseline |
| 6v_alpha | 6 | alpha (0.5) | Maximum views |
| 4v_random | 4 | alpha (0.5) | Random selection |

---

## 5. Data Locations

### Preprocessed Data
```
/home/joon/data/preprocessed/FaceLift_mouse/
├── D7/        # 3238 train, 359 val
├── D7_t/      # 1222 train, 1187 val, 1188 test (temporal)
├── D7_1/      # [TO BE CREATED]
└── D7_2/      # [TO BE CREATED]
```

### Raw Data
```
/home/joon/data/markerless_mouse_1_nerf/
├── raw_videos/           # Original MP4s
├── simpleclick_undist/   # Mask videos
└── new_cam.pkl           # Camera parameters
```

---

## 6. Archive

### Archived Configs
```
configs/mouse/_archive/
├── legacy_gslrm_v/       # 36 files (gslrm_v15~v71)
├── deprecated_D1_D4/     # 38 files (D1~D4)
└── deprecated_D6/        # 30 files (D6-1/2/3)
```

### Archived Datasets
```
/home/joon/data/preprocessed/FaceLift_mouse/
├── D1_pp_centered/       # [DEPRECATED]
├── D2_correct_pp/        # [DEPRECATED]
├── D3*/                  # [DEPRECATED]
├── D4/                   # [DEPRECATED]
├── D6-*/                 # [DEPRECATED]
└── data_mouse_v1*/       # [DEPRECATED]
```

---

## 7. Changelog

### v3.0 (2026-01-20)
- Added D7.1, D7.2 presets
- Created modular config system
- Archived 104 legacy configs
- Unified preprocessing entry point

### v2.0 (2026-01-19)
- D7 PP-centered shift paradigm
- Temporal split variants

### v1.0 (2026-01-17)
- Initial D1-D4 object-centered approach

---

*Last updated: 2026-01-20*
