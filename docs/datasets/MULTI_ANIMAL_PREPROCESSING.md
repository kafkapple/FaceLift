# Multi-Animal Dataset Preprocessing Specification

s-DANNCE (social DANNCE) 등 다중 동물 데이터셋을 FaceLift GS-LRM 파이프라인에 등록하기 위한
전처리 스펙, 마스크 전략, 디렉토리 구조를 정의합니다.

---

## 1. Background & Motivation

### Why: 기존 파이프라인의 한계

FaceLift M5 파이프라인은 **단일 동물**(markerless_mouse_1, Harvard DANNCE) 전용으로 설계됨:
- RGBA PNG에 binary mask (0/255) 단일 foreground
- 인스턴스 구분 없음 — 전체 foreground가 하나의 객체
- s-DANNCE처럼 **2+ 동물이 동시에 존재**하는 데이터를 처리할 수 없음

### How: 2-File 전략 + 3D Keypoint 기반 Identity

기존 RGBA 호환성을 유지하면서 인스턴스별 마스크를 **별도 파일**로 추가하고,
s-DANNCE 3D 키포인트를 cross-view identity의 single source of truth로 활용합니다.

### What: 이 문서가 정의하는 것

1. GS-LRM 입력 스펙 요약 + Plucker ray 상세 (현재 파이프라인)
2. s-DANNCE 소스 데이터 스펙 (실제 데이터셋 기반)
3. 다중 동물용 마스크 저장 포맷 (2-File 전략)
4. 전처리 파이프라인 확장
5. 디렉토리 구조 & 메타데이터 스키마
6. 엣지 케이스 & 리스크 완화

---

## 2. Current Pipeline Specs (GS-LRM Input)

### 2.1 Model Input Tensors

| Tensor | Shape | Description |
|--------|-------|-------------|
| `image` | `[B, V, 9, 512, 512]` | Multi-view posed images (RGB 3ch + Plucker 6ch) |
| `c2w` | `[B, V, 4, 4]` | Camera-to-world matrices |
| `fxfycxcy` | `[B, V, 4]` | Intrinsics `[fx, fy, cx, cy]` |
| `index` | `[B, V, 2]` | `[view_idx, scene_idx]` |

- **V** = 6 views (4 encoder input, 6 target)
- **Resolution**: 512 x 512

#### C=9 Channel Breakdown: RGB (3) + Plucker Rays (6)

| Channel | Name | Formula | Value Range | Description |
|:-------:|------|---------|:-----------:|-------------|
| 0-2 | RGB | `pixel * 2.0 - 1.0` | **[-1, 1]** | Normalized image color |
| 3-5 | Moment **m** | `o × d` (cross product) | **[-2.7, 2.7]** | Plucker moment vector |
| 6-8 | Direction **d** | `normalize(unproject(u,v))` | **[-1, 1]** | Unit ray direction (world space) |

> **Code**: `gslrm/model/gslrm.py:973-1017` (`_create_posed_images_with_plucker`)
> **Ray generation**: `gslrm/model/transform_data.py:38-78` (`compute_camera_rays`)

#### Plucker Ray 계산 과정 (Per-Pixel)

```
Input: c2w (4x4), fxfycxcy = [fx, fy, cx, cy], pixel (u, v)

Step 1: Pinhole Unprojection (pixel → camera space)
    dx = (u + 0.5 - cx) / fx
    dy = (v + 0.5 - cy) / fy
    dz = 1.0
    d_cam = normalize([dx, dy, dz])

Step 2: World Space Rotation
    d_world = d_cam @ c2w[:3,:3].T    # camera → world
    d = normalize(d_world)             # unit direction vector

Step 3: Ray Origin
    o = c2w[:3, 3]                     # camera position (world)
    (same for all pixels in this view)

Step 4: Plucker Moment
    m = o × d                          # cross product

Step 5: Concatenation → C=9
    posed_image = cat([RGB_normalized, m, d], dim=channel)
```

#### Why Plucker Coordinates? (왜 6D인가)

**수학적 배경**: 3D 공간의 직선은 **4 자유도**(4-DOF)를 가짐.
Plucker 좌표는 이를 6D로 임베딩 (Grassmannian Gr(2,4) → P⁵).
`d·m = 0` 제약조건이 항상 성립하여 실제 5D 매니폴드 위에 존재.

**`(d, m)` vs `(o, d)` — 왜 moment를 쓰는가?**

| 표현 | 문제 | 결과 |
|------|------|------|
| `(o, d)` | 동일 ray에 대해 `o' = o + k*d`로 **무한히 많은** 표현 존재 | 네트워크가 불변성을 학습해야 함 |
| `(d, m)` | `m = o × d = (o+kd) × d` → **k에 무관** (유일한 표현) | Canonical, 학습 효율적 |

**기하학적 직관**:
- `d` = ray 방향 (unit vector)
- `|m|` = **원점에서 직선까지의 수직 거리** (`|m| = |o| × sin θ`)
- `m`의 방향 = 원점, 카메라, ray가 이루는 평면의 법선

**구체적 예시**: 카메라 위치 `o = [2.7, 0, 0]`, 원점 바라봄 (중심 픽셀)
```
d = [-1, 0, 0]      (원점을 향한 단위 벡터)
m = [2.7,0,0] × [-1,0,0] = [0, 0, 0]   (ray가 원점을 관통 → 거리=0)
```

비중심 픽셀 예시: 약간 위쪽을 바라보는 ray
```
d ≈ [-0.99, 0.1, 0]   (거의 -X 방향, 약간 +Y)
m = [2.7,0,0] × [-0.99,0.1,0] = [0, 0, 0.27]   (|m|≈0.27, 원점에서 0.27 거리)
```

#### Normalization 주의사항

| 대상 | Normalization | 비고 |
|------|:------------:|------|
| RGB (ch 0-2) | `*2.0 - 1.0` → [-1, 1] | 명시적 |
| d (ch 6-8) | L2 unit norm | `compute_camera_rays`에서 수행 |
| m (ch 3-5) | **없음** | `o × d` 그대로, scene scale에 비례 |

> ⚠️ moment `m`에 별도 normalization이 없으므로, 카메라 거리 정규화(avg=2.7)가
> 간접적으로 moment 값의 scale을 통제합니다. 새 데이터셋에서 동일한 카메라 정규화를
> 적용해야 Plucker 값 범위가 호환됩니다.

#### Summary: C=9 채널 구성

```
┌─────────┬─────────────┬─────────────┬───────────────────────────────────────────────────┐
│ Channel │    이름     │   값 범위   │                       계산                        │
├─────────┼─────────────┼─────────────┼───────────────────────────────────────────────────┤
│   0-2   │ RGB         │   [-1, 1]   │ pixel * 2.0 - 1.0                                 │
├─────────┼─────────────┼─────────────┼───────────────────────────────────────────────────┤
│   3-5   │ Moment m    │ [-2.7, 2.7] │ o × d (origin × direction)                        │
├─────────┼─────────────┼─────────────┼───────────────────────────────────────────────────┤
│   6-8   │ Direction d │   [-1, 1]   │ pinhole unprojection → world rotation → normalize │
└─────────┴─────────────┴─────────────┴───────────────────────────────────────────────────┘
```

#### Summary: 왜 6D Plucker인가?

- 3D 직선 = **4-DOF** → Grassmannian Gr(2,4)의 6D 임베딩 (redundant but canonical)
- `(o, d)` 표현은 동일 ray에 **무한한 표현**이 존재 (o를 ray 위 어디든 잡을 수 있음)
- `(d, m)` 표현은 **유일** — moment `m = o × d`는 o의 위치에 무관 (`m' = (o+kd) × d = m`)
- `|m|` = 원점에서 ray까지의 **수직 거리** (기하학적 의미가 명확)
- `d · m = 0` 항상 성립 (Plücker relation) — 6D 중 실제 5D 매니폴드
- 카메라 거리 정규화(2.7)가 moment 값 범위를 간접 통제 → **새 데이터셋에서도 동일 정규화 필수**

### 2.2 Current Mask Format (M5, Single Animal)

```
cam_000.png  →  RGBA, 512x512
                RGB = image pixels
                A   = binary foreground mask (0 or 255)
```

- Source: SimpleClick segmentation on undistorted videos
- Dataset loader: `mouse_extensions/data/mouse_dataset.py`
  - RGBA → alpha channel extracted as mask
  - `auto_generate_mask: false` (M5 uses pre-computed alpha)

### 2.3 Camera Format (Per-Frame JSON)

```json
{
  "frames": [
    {
      "w": 512, "h": 512,
      "fx": 549.0, "fy": 549.0, "cx": 256.0, "cy": 256.0,
      "w2c": [[4x4 matrix]],
      "file_path": "images/cam_000.png",
      "view_id": 0
    }
  ]
}
```

Normalization: centroid → origin, avg camera distance → 2.7.

---

## 3. s-DANNCE Source Data Specifications

> **Reference**: `sdannce-poc/docs/data/dataset_catalog.md` (joon server)

### 3.1 Dataset Overview

| Item | Value |
|------|-------|
| **Source** | Klibaite et al., Cell 188(8), 2025. DOI: 10.1016/j.cell.2025.01.044 |
| **Harvard Dataverse** | https://dataverse.harvard.edu/dataverse/socialDANNCE_data |
| **Total** | 17 datasets, 1,690 recordings (~197 GB) |
| **Species** | Rat (1,562 recordings), Mouse (128 recordings) |
| **Social recordings** | 1,342 (79.4%, dyadic = 2 animals) |
| **Lone recordings** | 348 (20.6%, single animal) |
| **TRIADS** | 205 recordings (3-animal interaction) |

### 3.2 Recording Specifications

| Parameter | Value |
|-----------|-------|
| **Cameras** | 6 synchronized (Basler a2A1920-160ucBAS) |
| **Resolution** | 1920 x 1200 |
| **FPS** | 50 fps |
| **Frames/recording** | 90,000 (standard, = 30 min) |
| **Codec** | H.264 MP4 (CRF 28) |
| **Per-camera size** | 125-236 MB |

### 3.3 3D Keypoint Data (Per-Animal)

| Field | Shape | Description |
|-------|-------|-------------|
| `pred` | `(90000, 3, 23)` | 3D keypoint coordinates (mm, world space) |
| `p_max` | `(90000, 23)` | Per-keypoint confidence (0-1) |
| `sampleID` | `(1, 90000)` | Frame indices |

- **23 keypoints** (rat): Snout, EarL/R, SpineF/M/L, TailBase, ShoulderL/R, ElbowL/R, WristL/R, HandL/R, HipL/R, KneeL/R, AnkleL/R, FootL/R
- **Coordinate ranges**: X=[-300,400], Y=[-274,424], Z=[-33,333] mm (cage ~70x70x36 cm)
- **NaN**: 0 (s-DANNCE always produces predictions)
- **File size**: ~117 MB per animal per session

### 3.4 Camera Calibration Format

| Field | Shape | Format | Description |
|-------|-------|--------|-------------|
| `K` | (3, 3) | **MATLAB transposed** | Intrinsic matrix |
| `r` | (3, 3) | Rotation matrix | World → camera rotation |
| `t` | (1, 3) | Translation vector | World → camera translation (mm) |
| `RDistort` | (1, 2) | (k1, k2) | Radial distortion |
| `TDistort` | (1, 2) | (p1, p2) | Tangential distortion |

**Typical intrinsics** (Basler): fx=2228-2293, fy=2229-2295, cx=940-978, cy=515-617

> ⚠️ **OpenCV 변환 필수**:
> ```python
> K_opencv = K_matlab.T              # MATLAB → OpenCV: transpose
> rvec, _ = cv2.Rodrigues(r_matrix)  # 3x3 → Rodrigues vector
> dist_coeffs = [k1, k2, p1, p2, 0] # Combine distortion
> ```

### 3.5 Per-Session Directory Structure

```
experiment_folder/                    # e.g., 2022_09_22_M3_M4 (~1.9 GB)
├── videos/
│   ├── Camera1/
│   │   ├── 0.mp4                    # Full recording (175-237 MB)
│   │   ├── frametimes.npy           # (2, 90000): indices + timestamps
│   │   └── metadata.tab
│   ├── Camera2/ ~ Camera6/
├── calibration/
│   ├── hires_cam1_params.mat ~ hires_cam6_params.mat
├── COM/predict01/
│   ├── instance0com3d.mat           # Animal 0 center-of-mass (2.9 MB)
│   └── instance1com3d.mat           # Animal 1 center-of-mass
├── SDANNCE/
│   ├── bsl0.5_FM_rat1/
│   │   ├── save_data_AVG0.mat       # Animal 1 keypoints (117 MB)
│   │   └── save_data_AVG.mat        # Averaged predictions
│   └── bsl0.5_FM_rat2/
│       └── (same structure)
├── io.yaml
└── sampleCAL_BG_dannce.mat
```

### 3.6 Key Differences from Current M5 Data

| Item | M5 (DANNCE mouse) | s-DANNCE |
|------|:------------------:|:--------:|
| Species | Mouse | **Rat** (+ some Mouse) |
| Animals | 1 | **2** (social) or 3 (triads) |
| Resolution | 1152 x 1024 | **1920 x 1200** |
| FPS | 100 | **50** |
| Frames | 18,000 | **90,000** |
| Keypoints | 22 (MAMMAL) | **23** (s-DANNCE rat skeleton) |
| Masks | SimpleClick video | **없음** ⚠️ (SAM 생성 필요) |
| Calibration | `new_cam.pkl` (Python) | **MATLAB .mat** (transpose 필요) |
| Keypoint source | MAMMAL fitting | **s-DANNCE prediction** |

### 3.7 Proposed Frame Sampling

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Source FPS | 50 fps | s-DANNCE 표준 |
| Frame step | 20 | 50/20 = 2.5 effective fps |
| Total frames | 4,500 | 90,000 / 20 |
| Train/Val/Test | 3,600 / 450 / 450 | 8:1:1 temporal split |
| Train range | frame 0-71,999 (0-24min) | |
| Val range | frame 72,000-80,999 (24-27min) | |
| Test range | frame 81,000-89,999 (27-30min) | |

---

## 4. Multi-Animal Mask Strategy (마스크 전략)

### 3.1 Design Decision: 2-File Approach

**기존 RGBA 이미지(RGB + combined mask)를 수정하지 않고**, 인스턴스 마스크를 별도 파일로 추가합니다.

```
# Before (M5, single animal)
{frame}/images/cam_000.png          # RGBA (RGB + binary alpha)

# After (multi-animal extension)
{frame}/images/cam_000.png          # RGBA (RGB + combined alpha)  ← UNCHANGED
{frame}/masks/cam_000_inst.png      # Grayscale, instance ID map   ← NEW
{frame}/instance_meta.json          # Instance metadata + keypoints ← NEW
```

**Rationale**:
- 기존 단일 동물 코드 **무수정** 호환 (alpha만 보면 동일)
- I/O 오버헤드 미미 (grayscale PNG, frame당 6개, ~5-10KB/each)
- 인스턴스 수 제한 없음 (pixel value 1-255 → 최대 255 인스턴스)

### 3.2 Mask File Specifications

#### Combined Alpha (기존 호환)

| Item | Spec |
|------|------|
| **File** | `{frame}/images/cam_{view:03d}.png` |
| **Channels** | RGBA (4ch) |
| **Alpha** | 0 = background, 255 = ANY foreground (union of all instances) |
| **Resolution** | 512 x 512 |
| **Encoding** | PNG, lossless |

#### Instance Mask (신규)

| Item | Spec |
|------|------|
| **File** | `{frame}/masks/cam_{view:03d}_inst.png` |
| **Channels** | Grayscale (1ch) |
| **Pixel values** | 0 = background, 1 = mouse_0, 2 = mouse_1, ... |
| **Resolution** | 512 x 512 |
| **Encoding** | PNG, lossless, `cv2.IMREAD_GRAYSCALE` |
| **Alignment** | Pixel-perfect aligned with corresponding RGBA image |

**Instance ID Convention**:
- ID는 **frame 내 고정**, cross-frame temporally consistent (s-DANNCE tracking ID 기반)
- ID=1이 항상 같은 개체 (s-DANNCE `animal_0` → ID=1, `animal_1` → ID=2)
- ID는 **모든 6개 뷰에서 동일** (cross-view consistency)

### 3.3 Cross-View Identity: 3D Keypoint Projection

s-DANNCE는 이미 cross-frame, cross-view consistent한 3D 키포인트를 제공합니다.
이것이 instance identity의 **유일한 ground truth source**입니다.

```
Per Frame:
  s-DANNCE output → 3D keypoints per animal (already tracked, consistent IDs)
       │
       ▼
  For each camera view:
    1. Project 3D keypoints → 2D using camera P = K @ [R|t]
    2. Match projected keypoints to 2D segmentation masks
    3. Assign s-DANNCE animal ID to matched mask → instance ID
       │
       ▼
  Result: Per-view instance mask with globally consistent IDs
```

**Matching Algorithm** (robust to calibration error):
```python
def assign_instance_ids(
    masks_2d: list[np.ndarray],       # per-view raw segmentation masks
    keypoints_3d: dict[int, np.ndarray],  # animal_id → (N_kp, 3)
    cameras: list[Camera],            # 6 cameras with P matrices
) -> list[np.ndarray]:
    """Assign instance IDs to 2D masks using 3D keypoint projection."""
    instance_masks = []
    for view_idx, (mask, cam) in enumerate(zip(masks_2d, cameras)):
        inst_map = np.zeros_like(mask, dtype=np.uint8)
        labeled_masks = label_connected_components(mask)  # separate blobs

        for animal_id, kp3d in keypoints_3d.items():
            kp2d = cam.project(kp3d)  # (N_kp, 2)
            # Vote: which blob contains the most projected keypoints?
            best_blob = find_best_matching_blob(labeled_masks, kp2d)
            if best_blob is not None:
                inst_map[labeled_masks == best_blob] = animal_id

        instance_masks.append(inst_map)
    return instance_masks
```

**Fallback** (keypoint projection fails due to occlusion):
- 투영된 키포인트가 어떤 blob에도 매칭되지 않으면 → 해당 뷰에서 해당 동물 ID=0 (invisible)
- Multi-view 특성상 한 뷰에서 가려져도 다른 뷰에서 보임

---

## 5. Metadata Schema

### 4.1 instance_meta.json (Per-Frame)

```json
{
  "frame_idx": 0,
  "num_instances": 2,
  "instances": {
    "1": {
      "sdannce_animal_id": "animal_0",
      "keypoints_3d": [[x, y, z], ...],
      "num_keypoints": 22,
      "visible_views": [0, 1, 2, 3, 4, 5],
      "bbox_2d": {
        "cam_000": [x1, y1, x2, y2],
        "cam_001": [x1, y1, x2, y2]
      }
    },
    "2": {
      "sdannce_animal_id": "animal_1",
      "keypoints_3d": [[x, y, z], ...],
      "num_keypoints": 22,
      "visible_views": [0, 1, 3, 4, 5],
      "bbox_2d": { ... }
    }
  },
  "occlusion_flags": {
    "cam_002": {"1_2": 0.35}
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `frame_idx` | int | M-series sample index (0-based) |
| `num_instances` | int | Number of animals in frame |
| `instances.{id}.keypoints_3d` | float[N][3] | s-DANNCE 3D keypoints (mm, world space) |
| `instances.{id}.visible_views` | int[] | Views where this animal has >50% keypoints visible |
| `instances.{id}.bbox_2d` | dict | Per-view 2D bounding boxes (after preprocessing transforms) |
| `occlusion_flags` | dict | Per-view pairwise IoU of instance masks (occlusion severity) |

### 4.2 dataset_info.json (Per-Dataset, Root Level)

```json
{
  "dataset_name": "social_mouse_1",
  "source": "s-DANNCE",
  "num_cameras": 6,
  "num_frames": 5000,
  "num_animals": 2,
  "animal_ids": ["animal_0", "animal_1"],
  "resolution": [512, 512],
  "fps_effective": 20,
  "keypoint_format": "s-DANNCE_22kp",
  "coordinate_system": "DANNCE_world_mm",
  "preprocessing": {
    "paradigm": "recentered_affine",
    "version": "M5_multi",
    "camera_normalization": "batch_uniform_2.7"
  },
  "splits": {
    "train": "data_train.txt",
    "val": "data_val.txt",
    "test": "data_test.txt"
  }
}
```

---

## 6. Directory Structure

### 5.1 Full Layout

```
/home/joon/data/preprocessed/FaceLift_mouse/{DATASET_NAME}/
├── dataset_info.json                    # Dataset-level metadata
├── data_train.txt                       # Split files (frame indices)
├── data_val.txt
├── data_test.txt
├── 000000/
│   ├── images/
│   │   ├── cam_000.png                  # RGBA 512x512 (combined alpha)
│   │   ├── cam_001.png
│   │   ├── cam_002.png
│   │   ├── cam_003.png
│   │   ├── cam_004.png
│   │   └── cam_005.png
│   ├── masks/                           # ← NEW: instance masks
│   │   ├── cam_000_inst.png             # Grayscale, pixel=instance_id
│   │   ├── cam_001_inst.png
│   │   ├── cam_002_inst.png
│   │   ├── cam_003_inst.png
│   │   ├── cam_004_inst.png
│   │   └── cam_005_inst.png
│   ├── opencv_cameras.json              # Camera parameters (same format)
│   └── instance_meta.json               # ← NEW: per-frame instance metadata
├── 000001/
│   └── ...
└── ...
```

### 5.2 Naming Convention

| Dataset | Name | Example |
|---------|------|---------|
| Single mouse (existing) | `M5` | `/FaceLift_mouse/M5/` |
| Social 2-mouse | `S{N}` | `/FaceLift_mouse/S1/` (S = Social) |
| Multi-species | `X{N}` | `/FaceLift_mouse/X1/` (X = Cross-species) |

Split naming: `{dataset}_t{variant}` → e.g., `S1_t1` (temporal split variant 1)

---

## 7. Preprocessing Pipeline Extension

### 6.1 Pipeline Overview

```
s-DANNCE Raw Data
│  videos (6 cam), calibration, 3D keypoints (per-animal tracked)
│
▼  Step 1: Frame Extraction
│  Extract synchronized frames from 6 camera videos
│
▼  Step 2: Camera Normalization
│  Apply M5-compatible preprocessing (affine, crop, normalize)
│  Same paradigm: recentered_affine, fx=549, dist=2.7
│
▼  Step 3: Instance Segmentation
│  Generate per-animal 2D masks for each view
│  Options: (a) SAM prompted by projected keypoints
│           (b) s-DANNCE provided masks (if available)
│           (c) Background subtraction + connected components
│
▼  Step 4: Cross-View Identity Assignment
│  Project 3D keypoints → 2D per view
│  Match 2D masks to projected keypoints → assign consistent IDs
│
▼  Step 5: Mask Composition & Storage
│  Combined alpha → RGBA image (backward compatible)
│  Instance map → grayscale PNG
│  Metadata → JSON
│
▼  Step 6: Validation
│  Cross-view consistency check
│  Mask-keypoint alignment verification
│  Coverage statistics
```

### 6.2 Step-by-Step Details

#### Step 1: Frame Extraction

s-DANNCE 원본 데이터 구조에 맞게 프레임 추출.
기존 `data_loader.py`의 `raw` 소스 타입을 확장하거나, 새로운 소스 타입 추가.

```python
# Actual s-DANNCE directory structure (Section 3.5 참조)
raw/{session_name}/                    # e.g., 2022_09_22_M3_M4
├── videos/
│   ├── Camera1/0.mp4 ~ Camera6/0.mp4 # 6 cameras, 1920x1200, 50fps
│   ├── Camera1/frametimes.npy         # (2, 90000): frame sync timestamps
├── calibration/
│   ├── hires_cam1_params.mat ~ hires_cam6_params.mat  # MATLAB format ⚠️
├── SDANNCE/
│   ├── bsl0.5_FM_rat1/save_data_AVG0.mat  # (90000, 3, 23) per animal
│   └── bsl0.5_FM_rat2/save_data_AVG0.mat
├── COM/predict01/
│   ├── instance0com3d.mat             # Center-of-mass trajectories
│   └── instance1com3d.mat
└── sampleCAL_BG_dannce.mat
# NOTE: 세그멘테이션 마스크는 제공되지 않음 → SAM2로 생성 필요
```

#### Step 2: Camera Normalization

**반드시 M5와 동일한 파라다임 적용** (recentered_affine):
1. Per-view affine: scale to target_fx=549, shift PP to (256, 256)
2. Crop to 512x512
3. Batch camera normalization: centroid → origin, avg distance → 2.7

> ⚠️ 동일한 정규화를 적용해야 pretrained GS-LRM 가중치와 호환됩니다.
> 카메라 배치가 크게 다른 경우 (e.g., 카메라 수, baseline) 별도 검증 필요.

#### Step 3: Instance Segmentation

**Option A — SAM + Keypoint Prompts** (권장, 마스크 미제공 시):
```python
from segment_anything import SamPredictor

for view_idx in range(6):
    for animal_id, kp3d in keypoints_3d.items():
        kp2d = camera[view_idx].project(kp3d)  # (N_kp, 2)
        # Use visible keypoints as point prompts
        visible = filter_in_frame(kp2d, w=512, h=512)
        mask = sam_predictor.predict(
            point_coords=visible,
            point_labels=np.ones(len(visible)),
        )
```

**Option B — s-DANNCE Provided Masks**:
s-DANNCE 파이프라인이 이미 per-animal mask를 제공하는 경우,
해당 마스크에 동일한 affine/crop 변환을 적용하면 됨.

**Option C — Background Subtraction + Separation**:
단일 foreground mask → 3D keypoint 투영으로 개체 분리.

#### Step 4: Cross-View Identity Assignment

Section 3.3 참조. s-DANNCE 3D 키포인트가 single source of truth.

#### Step 5: Mask Composition

```python
import cv2
import numpy as np

def compose_masks(
    rgb: np.ndarray,          # (512, 512, 3) uint8
    instance_masks: dict,     # {animal_id: (512, 512) bool}
) -> tuple[np.ndarray, np.ndarray]:
    """Create RGBA image and instance map."""
    # Combined alpha: union of all instances
    combined = np.zeros((512, 512), dtype=np.uint8)
    inst_map = np.zeros((512, 512), dtype=np.uint8)

    for animal_id, mask in instance_masks.items():
        combined[mask] = 255
        inst_map[mask] = animal_id  # 1, 2, ...

    # RGBA image (backward compatible)
    rgba = np.concatenate([rgb, combined[:, :, None]], axis=2)

    return rgba, inst_map

# Save
cv2.imwrite(f"{frame_dir}/images/cam_{v:03d}.png", rgba_bgra)
cv2.imwrite(f"{frame_dir}/masks/cam_{v:03d}_inst.png", inst_map)
```

#### Step 6: Validation

```python
def validate_frame(frame_dir: str) -> dict:
    """Cross-view consistency and alignment checks."""
    meta = json.load(open(f"{frame_dir}/instance_meta.json"))
    errors = []

    for view_idx in range(6):
        rgba = cv2.imread(f"{frame_dir}/images/cam_{view_idx:03d}.png", -1)
        inst = cv2.imread(f"{frame_dir}/masks/cam_{view_idx:03d}_inst.png", 0)

        # Check 1: Alpha matches instance union
        alpha_fg = (rgba[:, :, 3] > 0)
        inst_fg = (inst > 0)
        if not np.array_equal(alpha_fg, inst_fg):
            errors.append(f"cam_{view_idx}: alpha/instance mismatch")

        # Check 2: Instance IDs match metadata
        unique_ids = set(np.unique(inst)) - {0}
        expected_ids = set(int(k) for k in meta["instances"].keys())
        if not unique_ids.issubset(expected_ids):
            errors.append(f"cam_{view_idx}: unexpected IDs {unique_ids - expected_ids}")

    return {"valid": len(errors) == 0, "errors": errors}
```

---

## 8. Dataset Loader Extension

### 7.1 Backward Compatible Loading

```python
class MouseViewDataset:
    def __getitem__(self, idx):
        # Existing: load RGBA, extract alpha → works unchanged
        image = load_rgba(path)  # (512, 512, 4)
        rgb = image[:, :, :3]
        mask = image[:, :, 3]    # Combined foreground

        # NEW: optionally load instance mask
        if self.use_instance_masks:
            inst_path = path.replace("images/", "masks/").replace(".png", "_inst.png")
            inst_map = cv2.imread(inst_path, cv2.IMREAD_GRAYSCALE)

            if self.target_instance_id is not None:
                # Single-animal mode: isolate one instance
                mask = (inst_map == self.target_instance_id).astype(np.uint8) * 255
            # else: use combined mask (default, backward compatible)

        return rgb, mask, cameras, ...
```

### 7.2 Config Extension

```yaml
# configs/mouse/datasets/S1_t1.yaml
dataset:
  name: S1_t1
  type: social_mouse
  base_dir: /home/joon/data/preprocessed/FaceLift_mouse/S1
  num_animals: 2
  instance_masks: true            # Enable instance mask loading
  target_instance_id: null        # null = combined, 1 = mouse_0 only, 2 = mouse_1 only
  splits:
    train: data_train.txt
    val: data_val.txt
    test: data_test.txt
```

---

## 9. Edge Cases & Risk Mitigation

### 8.1 Occlusion (가장 큰 리스크)

| Scenario | Severity | Mitigation |
|----------|----------|------------|
| Partial overlap (IoU < 0.3) | Low | Instance mask에서 겹침 영역은 closer animal에 할당 (depth 추정 or 3D keypoint z-value) |
| Heavy overlap (IoU > 0.5) | Medium | `occlusion_flags`에 기록, 해당 프레임 학습 제외 옵션 |
| Full occlusion (한 뷰에서 완전 가림) | High | `visible_views` 필드로 추적, 해당 뷰에서 inst_id=0, 나머지 뷰에서 보완 |

**Multi-view 특성이 자연스러운 완화책**: 6개 뷰 중 1-2개에서 가려져도 나머지에서 해당 동물 관찰 가능.

### 8.2 Segmentation Quality

| Issue | Mitigation |
|-------|------------|
| SAM mask boundary noise | Morphological opening/closing (3x3 kernel) |
| Tail/limb fragmentation | Connected component → largest blob only, or convex hull |
| Background false positive | 3D keypoint bounding sphere → crop ROI first |

### 8.3 Camera Calibration Mismatch

s-DANNCE 카메라 캘리브레이션이 M5와 다를 수 있음:
- **해결**: 동일한 `camera_normalizer.py` 파이프라인 적용 (affine → normalize)
- **검증**: Ray error 계산 (< 1° 목표)
- **주의**: 카메라 수가 6개가 아닌 경우, `num_input_views` config 조정 필요

### 8.4 Identity Swap

| Scenario | Mitigation |
|----------|------------|
| Frame-to-frame swap | s-DANNCE tracking이 이미 해결 (temporal consistency) |
| s-DANNCE tracking failure | `instance_meta.json`에 confidence score 포함, 낮은 프레임 제외 |
| 매우 유사한 외모 | 3D keypoint이 유일한 단서 — appearance matching 불필요 |

---

## 10. Evaluation Modes

### 9.1 Single-Instance Eval (기존 호환)

```bash
# Isolate mouse_0 only → same as M5 single-animal eval
train_gslrm.py -d S1_t1 -e E0_1_facelift --set dataset.target_instance_id 1
```

- 선택된 인스턴스의 마스크만 사용
- 다른 동물은 배경으로 처리
- 기존 fair_comparison.py 그대로 사용 가능

### 9.2 Multi-Instance Eval (신규)

```bash
# Reconstruct all animals simultaneously
train_gslrm.py -d S1_t1 -e E_multi --set dataset.target_instance_id null
```

- Combined mask 사용 (모든 동물 = foreground)
- GS-LRM이 여러 동물을 하나의 scene으로 재구성
- Per-instance PSNR: instance mask로 영역 분리 후 개별 계산

### 9.3 Metrics Extension

| Metric | Single-Instance | Multi-Instance |
|--------|:-:|:-:|
| PSNR_gt (full mask) | O | O (combined) |
| PSNR_per_instance | - | O (per animal) |
| IoU | O | O (per animal + combined) |
| PSNR_intersection | O | O |
| Coverage | O | O (per animal) |
| Identity Consistency | - | O (cross-view ID check) |

---

## 11. Implementation Checklist

- [ ] s-DANNCE raw data 확보 및 gpu03 전송
- [ ] s-DANNCE 데이터 포맷 조사 (카메라, 키포인트, 마스크 유무)
- [ ] `data_loader.py`에 s-DANNCE 소스 타입 추가
- [ ] `preprocess.py`에 multi-animal 분기 추가
- [ ] Instance mask 생성 파이프라인 구현 (SAM or provided masks)
- [ ] Cross-view identity assignment 구현
- [ ] `instance_meta.json` 생성 로직
- [ ] `mouse_dataset.py`에 instance mask loading 추가
- [ ] Validation script 작성
- [ ] Config 템플릿 생성 (`configs/mouse/datasets/S1_t1.yaml`)
- [ ] 전처리 결과 시각화 (instance overlay, cross-view grid)
- [ ] Fair eval metrics extension (per-instance PSNR)

---

## 12. Related Documents

- ↑ [[../INDEX]] — Document hub
- ↔ [[PREPROCESSING_REGISTRY]] — Preprocessing presets & history
- ↔ [[M5_SERIES_SPEC]] — M5 normalization spec
- ↔ [[RAW_DATA]] — Raw data sources
- ↔ [[../../mouse_extensions/docs/DATASET_FRAME_INDEXING]] — Frame indexing & conversion
- ↔ [[../../mouse_extensions/docs/CAMERA_CALIBRATION]] — Camera parameters
- ↔ [[../../mouse_extensions/docs/COORDINATE_SYSTEMS]] — Coordinate transforms

---

*FaceLift | Multi-Animal Dataset Preprocessing Spec | Created: 2026-03-18*
*⚠️ s-DANNCE 실제 데이터 확보 후 포맷 세부사항 업데이트 필요*
