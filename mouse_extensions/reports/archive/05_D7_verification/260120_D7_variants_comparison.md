# D7 Variants Comparison Report

> **Generated**: 2026-01-20
> **Scope**: D7 vs D7.1 vs D7.2 preprocessing comparison

---

## Executive Summary

| Variant | Scale Mode | fx | fy | Ray Error | Status |
|---------|------------|-----|-----|-----------|--------|
| **D7** | fx_only | 549.0 | 549.0 (forced) | ~0.4deg | Production |
| **D7.1** | individual | 549.0 | 549.0 (exact) | ~0deg | **Recommended** |
| **D7.2** | average | ~547 | ~551 | ~0deg | Alternative |

---

## Detailed Analysis

### D7 (fx_only) - Current Production

**Scale Mode**: fx_only
- scale = target_fx / orig_fx
- Same scale applied to both x and y
- fy forced to 549, actual scaled value ~551.5

**Intrinsics**: fx=549.0, fy=549.0 (forced), cx=256.0, cy=256.0

**Ray Error**: ~0.4deg (LOW risk)

---

### D7.1 (individual) - RECOMMENDED

**Scale Mode**: individual
- scale_x = target_fx / orig_fx = 0.3404
- scale_y = target_fy / orig_fy = 0.3382
- Different scales for x and y directions

**Intrinsics**: fx=549.0 (exact), fy=549.0 (exact), cx=256.0, cy=256.0

**Ray Error**: ~0deg (geometrically correct)

**Pros**: Geometrically perfect, matches GS-LRM pretrained distribution
**Cons**: ~0.6% anisotropic scaling (visually negligible)

---

### D7.2 (average) - Alternative

**Scale Mode**: average
- scale_avg = (scale_x + scale_y) / 2 = 0.3393
- Isotropic scaling

**Intrinsics**: fx~547.3, fy~550.7 (varies), cx=256.0, cy=256.0

**Ray Error**: ~0deg (consistent transform)

**Pros**: Isotropic scaling (no shape distortion)
**Cons**: fx, fy not exactly 549

---

## Preprocessing Commands

D7.1 (Recommended):
  python -m mouse_extensions.scripts.preprocess_D7_pp_centered \
      --data-dir /home/joon/data/markerless_mouse_1_nerf \
      --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
      --camera-pkl /home/joon/data/markerless_mouse_1_nerf/new_cam.pkl \
      --frame-interval 5 \
      --scale-mode individual

D7.2 (Alternative):
  python -m mouse_extensions.scripts.preprocess_D7_pp_centered \
      --data-dir /home/joon/data/markerless_mouse_1_nerf \
      --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D7_2 \
      --camera-pkl /home/joon/data/markerless_mouse_1_nerf/new_cam.pkl \
      --frame-interval 5 \
      --scale-mode average

---

*Generated: 2026-01-20*
