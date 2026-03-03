# Visualization Code Refactoring Plan

> Created: 2026-03-03 | Status: In Progress

## 1. Background (Why)

FaceLift visualization code has grown organically through multiple phases:
`video_generator` → `unified_visualizer` → `turntable_renderer` + `render_vertical_rotation.py`.
This resulted in dead code accumulation (~25%), duplicated logic, and inconsistent output paths.

## 2. Completed (Phase 1 & 2)

### Phase 1: Dead Code Removal (`7b6ea0f`)

| Action | File | Lines |
|--------|------|------:|
| Delete | `utils/visualize_masks.py` | -166 |
| Archive | `evaluation/visualization.py` → `_archive/` | -384 |
| Archive | `reports/report_generator.py` → `_archive/` | -966 |
| Delete | `_archive/video_generator.py` | -259 |
| Extract | `gaussian_export.py` from `unified_visualizer.py` | +120 |
| Clean | `render_vertical_rotation.py` remove `load_test_sample()` | -42 |
| Sync | `visualization/__init__.py` `__all__` | +19 |
| **Total** | **5 files changed** | **-1,794** |

### Phase 2: Additional Cleanup (`6ff1bd7`)

| Action | File | Lines |
|--------|------|------:|
| Archive | `model/visualization.py` → `_archive/` | -257 |
| Delete | `_archive/FL_vs_PS_comparison.md` (md5 dup) | -55KB |
| Delete | `_archive/mvdiffusion_bottleneck_analysis.md` (md5 dup) | -11KB |
| Delete | `_archive/hp_preprocessing_ablation.md` (md5 dup) | -6KB |

## 3. Pending Tasks

### P1: Short-term (Duplication Removal)

| # | Task | Files | Impact |
|---|------|-------|--------|
| 1 | Extract `_compute_c2w()` in `render_vertical_rotation.py` | 1 | -15 lines dup |
| 2 | Remove `DEFAULT_TURNTABLE_CONFIG` dict, unify to `TurntableVideoConfig` dataclass | 2 | Config single source |
| 3 | Move `compute_alpha_metrics()` to `evaluation/metrics.py` | 2 | Module responsibility |
| 4 | Move `compute_pred_mask_for_visualization()` to `model/mask_utils.py` | 2 | Module responsibility |
| 5 | Update `CLEANUP_PLAN.md` (mentions non-existent `split_manager.py`) | 1 | Doc accuracy |

### P2: Medium-term (Structure)

| # | Task | Impact |
|---|------|--------|
| 1 | Split `turntable_config.py` (481 lines) → `constants.py` + `camera_interpolation.py` + `grid_utils.py` | SRP |
| 2 | Split `TurntableRenderer.render_all()` (109 lines) → individual public methods | Readability |
| 3 | Integrate `render_vertical_rotation.py` vertical arc → `TurntableRenderer.render_vertical_arc()` | Code reuse |
| 4 | Refactor `TurntableVideoConfig.from_config()` → dict-mapping loop | -30 lines boilerplate |

### P3: Long-term (Architecture)

| # | Task | Impact |
|---|------|--------|
| 1 | Extract trajectory code (446 lines) from `gaussians_renderer.py` → `visualization/` | Upstream separation |
| 2 | Centralized output path module (`mouse_extensions/utils/output_paths.py`) | Path consistency |
| 3 | Standardize all hardcoded font/path references | Portability |

## 4. Output Path Standardization

### Current Issues

**4 root paths scattered:**
```
1. /home/joon/dev/FaceLift/outputs/          (primary)
2. /home/joon/dev/FaceLift/experiments/       (validation)
3. /home/joon/outputs/                        (orphan!)
4. /node_data/joon/checkpoints/FaceLift/eval/ (mixed with checkpoints)
```

**File naming inconsistencies:**
- UID formats: `_0`, `_cam_000`, `_sample_0000` (3 variants)
- Frame zero-padding: 4/6/8 digits (3 variants)
- View index: `view_00` (2-digit) vs `cam_000` (3-digit)
- Grid names: 5 different patterns
- Temporal prefix: `time_` vs `temporal_` (2 variants)

### Proposed Standard

**Directory structure:**
```
outputs/
├── eval/{experiment_id}/samples/{frame:06d}/cam_{idx:02d}/
├── analysis/
├── comparison/
├── viz/
└── reports/
```

**Naming rules:**
| Target | Format |
|--------|--------|
| Frame number | `%06d` (003240) |
| Camera index | `%02d` (00-05) |
| UID in filename | Remove (directory provides uniqueness) |
| Turntable video | `turntable_orbit.mp4` |
| Temporal video | `temporal_{variant}.mp4` |

## 5. Development Workflow (Hybrid)

```
Local (~/dev/FaceLift)              gpu03 (/home/joon/dev/FaceLift)
┌───────────────────────┐           ┌──────────────────────────┐
│ Code Review (opus)    │           │ Training (GPU 4-7)       │
│ Architecture Analysis │  git push │ Inference                │
│ Refactoring           │ ───────→  │ E2E Evaluation           │
│ Documentation         │ ←───────  │ Experiment Execution     │
│                       │  git pull │                          │
└───────────────────────┘           └──────────────────────────┘
```

**Agent model assignment:**
- `opus`: Code review, architecture, education
- `sonnet`: SSH operations, build/deploy
- `haiku`: File search, grep, structure scan

## 6. Code Quality Issues Found

### Active Files Summary (post-cleanup)

| File | Lines | Issues |
|------|------:|--------|
| `turntable_renderer.py` | 663 | `render_all()` 109 lines, `from_config()` boilerplate |
| `turntable_config.py` | 481 | 7 responsibilities mixed (SRP violation) |
| `render_vertical_rotation.py` | ~295 | Doesn't use `TurntableRenderer`, independent pipeline |
| `inference_viz.py` | 124 | `save_multiview_turntable_grid()` 90% dup with vertical rotation |
| `error_annotation.py` | 157 | `compute_pred_mask_for_visualization()` misplaced |
| `alpha_visualization.py` | 170 | `compute_alpha_metrics()` misplaced |
| `gaussian_export.py` | ~120 | Newly extracted, clean |

### Hardcoded Paths (6 locations)

| File | Line | Path |
|------|------|------|
| `render_vertical_rotation.py` | 237, 308 | `/home/joon/dev/FaceLift/.../m5_cameras.json` |
| `cross_view_comparison.py` | 15, 18 | `Path("/home/joon/dev/FaceLift/outputs/...")` |
| `visualize_mask_comparison.py` | 201 | argparse default absolute path |
| `unified_report.py` | 52 | `Path('~/dev/FaceLift')` (no expanduser) |

---

*FaceLift Visualization Refactoring | v1.0 | 2026-03-03*
