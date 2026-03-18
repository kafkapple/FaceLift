# Dataset Frame Indexing & Specification

FaceLift Mouse 프로젝트의 전체 데이터 파이프라인: 원본 → MAMMAL fitting → M5 전처리 → 모델 학습/평가.
**모든 프레임 인덱싱 작업의 필수 참조 문서.**

---

## 1. Data Pipeline Overview

```
DANNCE (Harvard, Dunn et al. 2021)
│  markerless_mouse_1: 6cam, 1152×1024, 100fps, 18,000 frames (180s)
│
▼  MAMMAL (An et al. 2023) — segment mask 추가, NeRF용 재가공
│  markerless_mouse_1_nerf/: 6cam, 512×512 video, ~18,000 frames
│
▼  MAMMAL Fitting (mesh + keypoints)
│  step_2_frame_{video_frame:06d}.obj (step=5 → 3,600 OBJ files)
│  keypoints_22_3d.npz: (3600, 22, 3)
│
▼  FaceLift M5 Preprocessing (center crop → 512×512 RGBA, camera normalization)
│  M5/000000 ~ M5/003599: 3,600 samples × 6 views × 512×512
│
▼  Model Training (GS-LRM + MVDiffusion)
   M5t2 temporal split: train(80%) / val(10%) / test(10%)
```

---

## 2. Frame Indexing Map (⚠️ CRITICAL)

### 2.1 Indexing Convention

| System | Index Range | Step | Total | Example |
|--------|:-----------:|:----:|:-----:|---------|
| **Video frame** | 0 ~ 17,995 | 5 | 3,600 | frame 500 |
| **M5 sample** | 0 ~ 3,599 | 1 | 3,600 | sample 100 |
| **MAMMAL OBJ** | 000000 ~ 017995 | 5 | 3,600 | `step_2_frame_000500.obj` |
| **Keypoints** | [0] ~ [3599] | 1 | 3,600 | `kp_all[100]` |

### 2.2 Conversion Formula

```python
# M5 sample index → MAMMAL OBJ file
def m5_to_obj_path(m5_idx: int) -> str:
    video_frame = m5_idx * 5
    return f"step_2_frame_{video_frame:06d}.obj"

# M5 sample index → Keypoint
def m5_to_keypoint(kp_all, m5_idx: int):
    return kp_all[m5_idx]  # Direct indexing (3600 entries)

# MAMMAL OBJ frame number → M5 sample index
def obj_frame_to_m5(obj_frame: int) -> int:
    assert obj_frame % 5 == 0, f"Invalid OBJ frame: {obj_frame}"
    return obj_frame // 5
```

### 2.3 Why Step=5?

MAMMAL fitting processes 100fps DANNCE video but outputs mesh at every 5th frame:
- Input: 18,000 video frames at 100fps (180 seconds)
- Output: 3,600 mesh+keypoints at 20fps effective
- OBJ naming preserves original video frame number (not sequential index)

### 2.4 Known Bug & Fix (2026-03-11)

**Bug**: `get_frame_obj_path(frame_idx)` used `f"step_2_frame_{frame_idx:06d}.obj"` (step=1)
**Symptom**: Frame N loaded mesh for video frame N instead of video frame N×5
**Impact**: Frame 0 correct (coincidence: 0×5=0), all others load wrong pose
**Fix**: Changed to `frame_idx * 5` → `f"step_2_frame_{frame_idx * 5:06d}.obj"`

---

## 3. Data Source Specifications

### 3.1 DANNCE Original (markerless_mouse_1)

| Item | Value |
|------|-------|
| **Source** | Harvard, Dunn et al. 2021 (Nature Methods) |
| **Cameras** | 6 synchronized cameras |
| **Resolution** | 1152 × 1024 |
| **FPS** | 100 |
| **Total Frames** | 18,000 (180 seconds) |
| **Subject** | Single freely-moving mouse in cylindrical arena |
| **GitHub** | [spoonsso/dannce](https://github.com/spoonsso/dannce) |

### 3.2 MAMMAL Processed (markerless_mouse_1_nerf)

| Item | Value |
|------|-------|
| **Source** | An et al. 2023 (Nature Communications) |
| **Processing** | Segment masks added, video format, NeRF preprocessing |
| **Resolution** | 512 × 512 (resized from 1152×1024) |
| **Video format** | 6 MP4 files (raw) + 6 MP4 files (masks) |
| **Camera params** | `new_cam.pkl` (original intrinsics/extrinsics) |
| **GitHub** | [anl13/MAMMAL_mouse](https://github.com/anl13/MAMMAL_mouse) |
| **Server path** | `/home/joon/data/raw/markerless_mouse_1_nerf/` |

### 3.3 MAMMAL Fitting Output

| Item | Value |
|------|-------|
| **Fitting config** | `v012345_kp22` — **최대 설정** (6 views × 22 keypoints) |
| **Config description** | `"6-view RGB with 22 keypoints (Original MAMMAL paper baseline)"` |
| **Server path** | `/home/joon/dev/MAMMAL_mouse/results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260126_025249/` |
| **Actual storage** | Symlink → `/home/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/` |
| **OBJ directory** | `obj/` (3,600 files, step=5 naming) |
| **Mesh topology** | 14,522 vertices, 28,800 faces (identical across all frames) |
| **Keypoints** | `keypoints_22_3d.npz` — shape (3600, 22, 3), units: mm |
| **Coordinate system** | MAMMAL world space (mm, cage-relative origin) |
| **Frame range** | 0 ~ 17,995 (step=5), complete coverage of all 18,000 video frames |
| **Optimization** | 3-step: pose/shape init → refinement → silhouette alignment (mask_step2=3000) |
| **Other fittings** | ❌ None — this is the **only** fitting result (no reduced-view variants) |

> **⚠️ Pseudo-Ground Truth**: MAMMAL fitting은 최대 데이터(6view, 22kp)를 사용하지만,
> 절대적 GT가 아닌 **pseudo-GT**입니다. Keypoint detection 오류, occlusion, model topology 한계가 있음.
> GS-LRM과의 비교 시 이 점을 인지하고 해석해야 합니다.

### 3.4 UV Texture Assets

| File | Spec | Path |
|------|------|------|
| **Template OBJ** | 14,522 V + 15,399 VT + 28,800 F | `exports/mouse_frame0_textured.obj` |
| **Texture map** | 512×512 RGB, multi-view back-projection | `exports/texture_final.png` |
| **Material** | Wavefront MTL (`map_Kd texture_final.png`) | `exports/mouse_frame0_textured.mtl` |
| **UV coords** | 15,399 entries (0-1 normalized) | `mouse_model/mouse_txt/textures.txt` |
| **Face→UV map** | 28,800 entries | `mouse_model/mouse_txt/faces_tex.txt` |

**UV Seam Expansion**: Template OBJ has 15,399 UV coords > 14,522 vertices.
Faces reference `v/vt` pairs. UV seams duplicate vertices → `uv_to_v` mapping needed:
```python
uv_to_v[uv_idx] = vert_idx  # Maps 15,399 UV indices → 14,522 vertex indices
expanded_verts = frame_verts[uv_to_v]  # [14522,3] → [15399,3]
```

### 3.5 M5 Preprocessed Data

| Item | Value |
|------|-------|
| **Server path** | `/home/joon/data/preprocessed/FaceLift_mouse/M5/` |
| **Total samples** | 3,600 |
| **Sample structure** | `{idx:06d}/images/cam_{view:03d}.png` + `opencv_cameras.json` |
| **Image resolution** | 512 × 512 RGBA |
| **Camera views** | 6 per sample |
| **Intrinsics (raw)** | fx=549, fy=549, cx=256, cy=256 |
| **Intrinsics (384)** | fx=411.75, fy=411.75, cx=192, cy=192 |
| **Normalization** | Batch Uniform (centroid→origin, avg dist→2.7) |
| **Paradigm** | `recentered_affine` |

### 3.6 M5t2 Split (Temporal, Canonical)

| Split | Range (M5 index) | Range (video frame) | Count | Ratio |
|-------|:-----------------:|:-------------------:|:-----:|:-----:|
| **Train** | 0 ~ 2,879 | 0 ~ 14,395 | 2,880 | 80% |
| **Val** | 2,880 ~ 3,239 | 14,400 ~ 16,195 | 360 | 10% |
| **Test** | 3,240 ~ 3,599 | 16,200 ~ 17,995 | 360 | 10% |

Split strategy: `temporal`, seed: 42

---

## 4. Temporal Resolution Summary

| Stage | FPS | Frame Count | Duration |
|-------|:---:|:-----------:|:--------:|
| DANNCE recording | 100 | 18,000 | 180s |
| MAMMAL fitting | 20 | 3,600 | 180s |
| M5 samples | 20 | 3,600 | 180s |

> **Note**: MAMMAL fitting step=5 on 100fps gives effective 20fps.
> M5 preprocessing uses frame_interval=5 on the same 18,000 frame count,
> yielding 3,600 samples that align 1:1 with MAMMAL fitting outputs.

---

## 5. Coordinate Transform Quick Reference

```python
# SSOT: mouse_extensions/coordinate_utils.py
from mouse_extensions.coordinate_utils import mammal_to_gslrm, gslrm_to_mammal

# M5_SCENE_CENTER = [59.672, 51.517, 107.099] mm (camera rig centroid)
# M5_DISTANCE_SCALE = 2.7 / 307.785 ≈ 0.008781

point_gslrm = mammal_to_gslrm(point_mm)   # MAMMAL mm → GS-LRM normalized
point_mm = gslrm_to_mammal(point_gslrm)   # GS-LRM normalized → MAMMAL mm
```

See [[COORDINATE_SYSTEMS]] for full details. Guards: `assert_gslrm_space()`, `assert_mammal_space()`.

---

## 6. Discontinuity Frames

Original video has recording gaps at:
```python
DISCONTINUITY_FRAMES = {5900, 11800, 17700}  # video frame numbers
# → M5 sample indices: {1180, 2360, 3540}
```
These frames are valid and included in training. Gap = temporal discontinuity only.

---

## 7. Verification Examples

```python
# Frame 100 verification
m5_idx = 100
video_frame = 100 * 5  # = 500
obj_file = "step_2_frame_000500.obj"  # ✓
kp_idx = 100  # kp_all[100] = keypoints for this frame

# Frame 3599 (last sample, test set)
m5_idx = 3599
video_frame = 3599 * 5  # = 17995
obj_file = "step_2_frame_017995.obj"  # ✓ = last OBJ file
```

---

## 8. Related Documents

- ↑ [[../docs/INDEX]] — Document hub
- ↔ [[COORDINATE_SYSTEMS]] — Coordinate transforms & scale
- ↔ [[CAMERA_CALIBRATION]] — Camera parameters & rendering pipeline
- ↔ [[../../docs/datasets/RAW_DATA]] — Raw data sources & sampling
- ↔ [[../../docs/datasets/M5_SERIES_SPEC]] — M5 normalization variants
- ↔ [[../../docs/datasets/PREPROCESSING_REGISTRY]] — Preprocessing presets

---

*FaceLift | Dataset Frame Indexing & Specification | Created: 2026-03-11*
*⚠️ Frame indexing 오류는 전체 파이프라인에 영향. 변경 전 반드시 이 문서 참조.*
