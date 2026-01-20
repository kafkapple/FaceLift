# D7 Preprocessing Commands

## Quick Reference

| Version | Scale Mode | fx/fy | Command |
|---------|------------|-------|---------|
| **D7** | fx_only | 549/549 (fy forced) | --scale-mode fx_only |
| **D7.1** | individual | 549/549 (exact) | --scale-mode individual |
| **D7.2** | average | ~548/~550 (approx) | --scale-mode average |

## D7.1 (Individual Scale - Recommended)

Geometrically correct: different scale_x and scale_y to achieve fx=fy=549 exactly.

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

## D7.2 (Average Scale)

Isotropic scaling with averaged scale factor. fx~548, fy~550.

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

## Post-Processing: Create Temporal Split (Optional)

After preprocessing, create temporal split symlinks:

```bash
# For D7.1 with temporal split
python -m mouse_extensions.scripts.create_temporal_split \
    --source /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --output /home/joon/data/preprocessed/FaceLift_mouse/D7_1_t
```

## Generate Experiment Configs

```bash
cd /home/joon/dev/FaceLift/configs/mouse/_modular

# Generate D7.1 config with 5v_alpha schema
python generate_config.py --dataset D7_1 --schema 5v_alpha

# Generate D7.2 config
python generate_config.py --dataset D7_2 --schema 5v_alpha
```

## Verification

After preprocessing, verify with:

```bash
python -m mouse_extensions.scripts.verify_D7_preprocessing \
    --dataset /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --name D7_1 \
    --output /home/joon/dev/FaceLift/mouse_extensions/reports/05_D7_verification/260120_D7_1_verification.md
```

---
*Updated: 2026-01-20*
