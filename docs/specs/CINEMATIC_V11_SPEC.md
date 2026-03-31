# Cinematic v11 Production Specification

> Version: v11c | Updated: 2026-03-31 | Base: v10d → v11 → v11b → v11c

## Overview

NeurIPS 2026 데모 영상. GS-LRM 기반 Multi-view Mouse 3D Reconstruction 파이프라인 시각화.

## Global Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 768px (test: 384px) | Square, production quality |
| FPS | 20 | Matches source data rate (100fps / 5-frame interval) |
| frame_step | 1 (global default) | Every frame used. No per-segment override. |
| Background | White (1,1,1) | Clean presentation |
| Frame range | 1500:1860 | 360 frames temporal coverage |
| Crossfade | 0.3s | Global default (overridden per segment) |
| Font scaling | Resolution-proportional | Reference: 512px. Auto-scales via `H / 512.0` |

## Segment Structure (13 segments, ~58.5s)

### Act 1: Input & Pipeline (0:00 – 0:12.5)

| # | Type | Duration | Transition | Content | Label |
|---|------|----------|-----------|---------|-------|
| 0 | flow_gt_opener | 3.0s | — | 3.0s 6-cam GT temporal mosaic | (none) |
| 1 | flow_gt_opener | 2.0s | 2.0s crop-zoom | — | (none) |
| 2 | flow_gt | 2.0s | — | 2.0s GT RGB single view | "GT RGB (Camera 0)" |
| 3 | flow_mask | 1.5s | — | 1.5s FG segmentation | "FG Segmentation (SAM2)" |
| 4 | flow_render | 4.0s | — | 4.0s GS-LRM splatting | "GS-LRM 6v Reconstruction" |

### Act 2: 3D Exploration (0:12.5 – 0:36.0)

| # | Type | Duration | Transition | Content | Label |
|---|------|----------|-----------|---------|-------|
| 5 | freeze_orbit | 9.0s | 0.5s elevation sweep | 8.5s 360° orbit (frozen time) | "360-deg Orbit (Frozen Time)" |
| 6 | flow_novel | 3.0s | 0.75s elevation sweep | 2.25s bottom view render | "Novel View (Bottom, -80deg)" |
| 7 | flow_novel | 3.0s | — | 3.0s bottom + paw KP overlay | "Novel View (Bottom) + Paw Keypoints" |
| 8 | flow_head_kp | 7.5s | 0.75s elevation sweep | 6.75s progressive body + 22 KP | "Body Parts + All Keypoints" |
| 9 | freeze_orbit | 7.5s | 0.5s transition | 7.0s temporal orbit | "Temporal Orbit (Time+Rotation)" |

### Act 3: Comparison & Novel Views (0:36.0 – 0:58.5)

| # | Type | Duration | Transition | Content | Label |
|---|------|----------|-----------|---------|-------|
| 10 | flow_render_mosaic | 5.0s | 1.25s zoom + 1.25s fade | **2.5s static 6-cam render grid** | "GS-LRM Reconstruction (6 Cameras)" |
| 11 | grid_novel_6views | 4.0s | 0.8s zoom-out | 3.2s extrapolated views | "Extrapolated Novel Views" |
| 12 | grid_novel_dense | 6.0s | 0.3s crossfade-in | 5.7s turntable grid | "Turntable Grid (36 Views)" |

## Key Technical Decisions

### frame_step = 1 (Global)
- **Why**: frame_step>1 causes frame repetition → 끊김. fps가 playback 속도를 제어.
- **Rule**: YAML에 frame_step 하드코딩 금지. 코드 기본값 1.

### Orbit Elevation: match_gt = true
- **Why**: GT 카메라(~35°) → orbit target(20°) elevation 차이가 vertical teleportation 유발.
- **Fix**: `match_gt: true` → GT 카메라의 실제 elevation 사용. Transition에서 elevation 변화 없음.

### Novel View Masking: opacity > 0.05
- **Why**: vis_mask는 GT 카메라(위에서 촬영)에서의 가시성만 고려. Bottom view에서 belly Gaussians 삭제됨.
- **Fix**: opacity threshold만으로 노이즈 제거. 전 방향 Gaussian 보존.

### Dense Grid: cell_resolution = 256
- **Why**: 768px grid에서 셀=128px. 768px로 렌더 후 128px downsample은 6× 낭비.
- **Fix**: 256px로 렌더 → 128px downsample. **fxfy를 res_scale로 스케일링** 필수.

### Font Scaling: Resolution-proportional
- **Why**: fontscale=0.65 고정 시 384px에서 과대, 768px에서 과소.
- **Fix**: `fontscale = 0.65 * (H / 512.0)`. Reference resolution = 512px.

## Render Mosaic Segment (Seg 10) — 3-Phase Design

```
Phase 1 (25%): Zoom-out from temporal orbit camera
  - SLERP: orbit_c2w → GT_c2w
  - Scale: 1.0 → 0.5

Phase 2 (25%): Crossfade
  - Blend: zoomed single-view → 6-cam render grid
  - smoothstep easing

Phase 3 (50%): Static hold ★
  - Show completed 6-cam render grid
  - Camera index labels (Cam 0–5)
  - Viewer absorption time: ~2.5s
```

## Extrapolated Novel 6 Views (Seg 11)

GT cameras: elevation 20-40°, azimuth 0-300° (6 views, 60° apart).

| Position | Label | Elev | Azim | Distance from GT |
|----------|-------|------|------|------------------|
| Top-Down | "Top-Down (80deg)" | +80° | 270° | +40° above GT max |
| High-Front | "High-Front (70deg)" | +70° | 270° | +30° above GT max |
| High-Side | "High-Side (60deg)" | +60° | 0° | +20° above GT max |
| Belly-Up | "Belly-Up (-85deg)" | -85° | 270° | -105° below GT min |
| Rear-Low | "Rear-Low (-40deg)" | -40° | 90° | -60° below GT min |
| Front-Low | "Front-Low (-30deg)" | -30° | 270° | -50° below GT min |

All rendered with **opacity > 0.05 mask** (not vis_mask).

## KP Visualization

| Parameter | Value | Reference |
|-----------|-------|-----------|
| kp_radius | 4 | Config override |
| bone_width | 2 | Default |
| depth_alpha | 0.25–0.80 | Config override |
| show_labels (bottom paw) | true | LP, RP, LF, RF |
| show_labels (body parts) | true | All 22 abbreviations |

## Duration Balance

| Act | Duration | Content | Ratio |
|-----|----------|---------|-------|
| Act 1 (Input) | 12.5s | 10.5s | 84% content |
| Act 2 (3D) | 23.5s | 20.5s | 87% content |
| Act 3 (Comparison) | 15.0s | 11.4s | 76% content |
| **Total** | **58.5s** | **42.4s** | **72% content** |

---

*SSOT for cinematic directing. Config: `configs/mouse/cinematic/cinematic_v11.yaml`*
