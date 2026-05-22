# Cinematic v11/v9 Production Specification

> Version: **v13 (PLY mode)** | Updated: 2026-05-15 | SSOT for cinematic directing
> Code: `mouse_extensions/behavior/cinematic_sequence.py` | Configs: `configs/mouse/cinematic/`

## TL;DR

NeurIPS 2026 데모. GS-LRM 기반 Multi-view Mouse 3D Reconstruction 파이프라인 시각화.

| Preset | Resolution / FPS | Mode | Source |
|---|---|---|---|
| **`cinematic_v11`** | 768px / 20fps | live ckpt | production |
| **`cinematic_v11_FINAL_alpha03_16k`** | 768px / 20fps | **PLY (α=0.3, 16k)** | paper-consistent ✓ |
| `cinematic_v11_FINAL_alpha10_16k` | 768px / 20fps | PLY (α=1.0) | artifact demo |
| `cinematic_v11_FINAL_filtered_a067` | 768px / 20fps | PLY (view-filtered) | comparison |
| `cinematic_v11_paper` | 768px / 20fps | live ckpt | paper render |
| `cinematic_default` | 512px / 15fps | minimal | smoke / dev |

> Legacy v6/v8/v9/quick_test/compare/face_follow variants archived to `configs/mouse/cinematic/_archive/` (2026-05-22).

## Outputs

| 위치 | 용도 |
|---|---|
| `~/results/FaceLift/cinematic/{preset}/` | 최신 결과 (로컬) |
| `gpu03:/node_data/joon/cinematic_repro/{preset}/` | 생성 산출 + segment cache |

자동 파일명: `cinematic_{stem}_{res}px_{fps}fps_f{start}-{end}.mp4` (auto H.264 reencode via `_reencode_h264`).

## Segment Structure (production: 17 segments, ~99-101s @ v11)

### Act 1 — Input & Pipeline

| # | Type | Notes |
|---|---|---|
| 0-1 | `flow_gt_opener` ×2 | GT 6-cam mosaic → crop-zoom → single view |
| 2 | `flow_gt` | GT RGB single view |
| 3 | `flow_mask` | FG Segmentation (SAM2) |
| 4 | `gaussian_flash` | Frozen-time 360° Gaussian primitive orbit, `match_gt: true`, `pause_at_end: 1.0` |
| 5 | `flow_render` | GS-LRM 6-cam render, `crossfade: 0.4` |

### Act 2 — 3D Exploration

| # | Type | Notes |
|---|---|---|
| 6 | `freeze_orbit` | 360° orbit, frozen time, `match_gt: true` |
| 7 | `flow_novel` | Bottom view (-80°), SLERP from elev=20 |
| 8 / 8b | `flow_novel` | Bottom + 4 paw KP / Return (-80° → 25°) + ALL 22 KP |
| 9 | `flow_head_kp` | Body part reveal + KP overlay continuous |
| 9c / 9d | `face_follow` front / reverse | nose→neck axis, distance 1.2, smoothing_window 7, `kp_toggle`, `show_kp_labels` |
| 10 | `freeze_orbit` | Temporal Orbit, `use_prev_novel: true` (SLERP from face_follow) |

### Act 3 — Comparison & Novel Views

| # | Type | Notes |
|---|---|---|
| 11 | `flow_render_mosaic` | 6-cam render zoom + fade + static hold |
| 12 | `grid_novel_6views` | 3×2 extrapolated novel views grid |
| 13 | `grid_novel_dense` | 6×6 turntable grid (36 views) |

## Key Technical Decisions

| Decision | Code SSOT | Rationale |
|---|---|---|
| **Scene-centred orbit** | `compute_scene_center_from_gt` | LSQ ray intersection of GT cams → mouse-centric rotation (vs world origin) |
| **gaussian_flash fxfy interp** | `_seg_gaussian_flash` | Interpolate GT→orbit fxfy in transition phase (avoids projection drift) |
| **Frame range wrap-around** | yaml `frame_range` | v11=`0:3500`, v9=`500:2500` — covers demand 무 wrap |
| **KP overlay refinement** | `overlay_keypoints` | radius=1, bone_width=1, `LINE_8` (no AA bleed), depth-modulated, length sanity |
| **Per-segment crossfade** | yaml `crossfade:` field | overrides global 0.3s |
| **face_follow smoothing** | `smoothing_window: 7` | rolling mean of kp_3d → camera trajectory 안정 (overlay는 raw kp 유지) |
| **face_follow opacity_mask** | `reverse: true` default | vis_mask = GT 6 cams 학습 → reverse cam 영역 sparse 보존 |
| **PLY mode** | `load_gaussians_from_ply` + `infer_frame_from_ply` | ckpt-independent, paper-consistent (16k count-matched 사전 저장) |

## Model / Data SSOT

| Item | Value |
|---|---|
| **PLY (paper)** | `/node_data/joon/data/shared/FaceLift_mouse_6view/gaussians/M5t2_6view_alpha03_v3_maskcarve16k/` (3600 PLYs, alpha=0.3, 16k count-matched) |
| **Fallback ckpt** | `base_uniform_v2_6view_v2/best_psnr.pt` (24.49 PSNR, no alpha) |
| GS-LRM config | `configs/base/gslrm_mouse.yaml` |
| `num_input_views` | 6 |
| Data | `/home/joon/data/preprocessed/FaceLift_mouse/M5` |
| Keypoints | `~/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz` |
| Camera distance | 2.7 (normalized) |
| hfov | 50° |

> **CKPT 손실 사고 (2026-05-15)**: `M5t2_6view_alpha03_v3` symlink target 삭제. PLY mode 도입으로 시네마틱이 ckpt 가용성에서 독립.

## Config Inheritance

`_base.yaml` (24L) — model/camera/global 공통. Preset yaml `extends:` 체인으로 chain:

```
_base.yaml
  ├── cinematic_v11.yaml (production, 17 segs)
  │   ├── cinematic_v11_FINAL_alpha03_16k.yaml (PLY α=0.3)
  │   ├── cinematic_v11_FINAL_alpha10_16k.yaml (PLY α=1.0)
  │   └── cinematic_v11_FINAL_filtered_a067.yaml (PLY view-filtered)
  ├── cinematic_v11_paper.yaml (paper render, live ckpt)
  └── cinematic_default.yaml (minimal smoke)
```

Recursive `_load_yaml_with_extends` (cycle-guarded). Deep merge.

## Usage

```bash
# Paper preset (PLY mode, ckpt-independent, α=0.3 best)
CUDA_VISIBLE_DEVICES=N python -m mouse_extensions.behavior.cinematic_sequence \
    --config configs/mouse/cinematic/cinematic_v11_FINAL_alpha03_16k.yaml \
    --output-dir /node_data/joon/cinematic_repro/v11_alpha03 \
    --use-cache --save-segments

# Production (live ckpt)
... --config configs/mouse/cinematic/cinematic_v11.yaml ...

# Smoke / dev (512px, faster)
... --config configs/mouse/cinematic/cinematic_default.yaml ...
```

## Wall-time

| Preset | First run | Cached re-run |
|---|---|---|
| default (minimal, 512px, 15fps) | ~30 min | ~3 min |
| v11 / v11_FINAL_* (17 segs, 768px, 20fps) | ~2-3 hours | ~10-15 min |

Heaviest: `grid_novel_dense` (36-view turntable).

<details><summary>📚 Iteration history (v3 → v13)</summary>

- **v13 (2026-05-15)**: PLY mode (`load_gaussians_from_ply` + `_infer` dispatcher). Recursive `extends` (cycle-guarded). ckpt 손실 사고 → 시네마틱 decoupling.
- **v12 (2026-05-15)**: scene_center alignment (LSQ ray intersection) + `pause_at_end` 1.0s freeze. fxfy interpolation fix (gaussian_flash transition phase).
- **v11 (2026-05-14)**: KP refinement (`max(1,...)`, conditional outline, `LINE_8` for r=1, depth-modulated bone width, length sanity).
- **v10 (2026-05-14)**: face_follow `smoothing_window` (scipy uniform_filter1d) + `show_kp_labels`. Slow gaussian (1.3x).
- **v9 (2026-05-14)**: gaussian_flash camera_fix (`get_orbit_cameras` accepts `center`).
- **v8 (2026-05-14)**: gaussian orbit (frozen-time 360°), `kp_toggle`, `use_opacity_mask=true` for reverse, `show_kp` split.
- **v7 (2026-05-14)**: frame_range wrap-around fix (cursor `% len`), face_follow SLERP entry (`interpolate_cameras`).
- **v6 (2026-05-14)**: face_follow segment (`compute_face_camera_c2w`), KP `kp_radius/bone_width: 1`.
- **v5 (2026-05-14)**: gaussian_flash 신설, 1.3x slowdown, mp4v→H.264 auto-reencode (`_reencode_h264`).
- **v4 (2026-05-14)**: frame_step 제거 (5Hz → 15Hz unique content rate).
- **v3 (2026-05-14)**: SSIM=1.0 regression (refactor + `_base.yaml` extends bit-perfect).

</details>
