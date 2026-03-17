# MAMMAL Mesh Rendering Pipeline

> **Navigation**: [← INDEX](../INDEX.md) | [mesh_gs_pair_collection](../experiments/mesh_gs_pair_collection.md)
> **Purpose**: MAMMAL mesh fitting → UV texture → per-frame rendering 3단계 기술 문서
> **Updated**: 2026-03-13 | **Version**: v1.0

---

## 1. Why

GS-LRM은 novel view에서 Gaussian artifact가 발생한다. 이를 정량화하려면 동일 카메라 시점의 **pseudo-GT**가 필요하다.
MAMMAL은 multi-view video로부터 parametric mesh를 fitting하여, 임의 카메라 시점에서 렌더링 가능한 3D mesh를 제공한다.

이 문서는 MAMMAL 결과물이 FaceLift 파이프라인에서 어떻게 활용되는지를 3단계로 설명한다:
1. **Fitting Data** — MAMMAL이 어떻게 per-frame OBJ를 생성하는가
2. **UV Texture Map** — 텍스처가 어떻게 메시에 매핑되는가
3. **Per-frame Rendering** — 각 프레임을 novel view 카메라에서 렌더링하는 과정

---

## 2. Stage 1: Fitting Data

### 2.1 MAMMAL이란?

MAMMAL (Multi-view Articulated Motion with Motion Capture Learning)은 multi-view 비디오로부터 동물의 3D pose와 shape를 복원하는 parametric mesh fitting 시스템이다.

### 2.2 입출력

```
Input:  6-view synchronized video (100fps, 18,000 frames)
        └── /home/joon/data/raw/markerless_mouse_1_nerf/

Output: Per-frame OBJ meshes (step=5, 3,600 frames)
        └── /home/joon/dev/MAMMAL_mouse/results/fitting/
            markerless_mouse_1_nerf_v012345_kp22_20260126_025249/obj/
            ├── step_2_frame_000000.obj
            ├── step_2_frame_000005.obj
            └── ... (3,600 files)
```

### 2.3 Fitting Configuration

| 항목 | 값 | 설명 |
|------|---|------|
| **Config ID** | `v012345_kp22` | 6-view 전부 + 22 keypoint |
| **Camera views** | v0~v5 (6개) | 모든 GT 카메라 사용 |
| **Keypoints** | 22 | Full skeleton model |
| **Step size** | 5 | 100fps → 20fps (매 5번째 프레임) |

### 2.4 Mesh Topology (모든 프레임 동일)

| 항목 | 값 |
|------|---|
| **Vertices** | 14,522 |
| **Faces** | 28,800 triangles |
| **좌표계** | MAMMAL world space (mm 단위, cage 기준 원점) |
| **형태** | Articulated template — pose만 변화, topology 불변 |

### 2.5 Frame Indexing

M5 dataset index와 MAMMAL video frame의 관계:

```python
mammal_frame = m5_frame_idx * 5

# Example: M5 frame 137 → MAMMAL video frame 685
#          → step_2_frame_000685.obj
```

> **상세**: `mouse_extensions/docs/DATASET_FRAME_INDEXING.md`

---

## 3. Stage 2: UV Texture Map

### 3.1 Why UV Mapping이 필요한가

MAMMAL fitting은 **geometry만** 출력한다 (OBJ에 vertex position만 있음).
실제 마우스의 외형을 재현하려면 texture가 필요하다.

해결: 대표 프레임(frame 0)에서 **UV texture를 한 번 생성**하고, 모든 프레임에 동일한 UV를 적용한다.
이것이 가능한 이유는 **mesh topology가 모든 프레임에서 동일**하기 때문이다.

### 3.2 Texture 파일 위치

```
/home/joon/dev/MAMMAL_mouse/exports/
├── mouse_frame0_textured.obj    # Template mesh (UV 좌표 포함)
├── mouse_frame0_textured.mtl    # Material 정의 (texture 참조)
└── texture_final.png            # 512×512 RGB texture map
```

### 3.3 UV Expansion 문제

OBJ 파일의 vertex(`v`)와 UV 좌표(`vt`)는 1:1 대응이 아니다:

```
OBJ 원본:
  v   → 14,522개 (unique 3D positions)
  vt  → 15,399개 (UV coordinates)
  f   → 28,800개 (각 face가 v/vt 쌍 참조)
```

**Why**: UV seam(이음새) 경계에서 하나의 3D vertex가 서로 다른 UV 좌표를 가질 수 있다.
Trimesh로 로드하면 이런 vertex를 분리(expand)하여 15,399개가 된다.

### 3.4 Face-level Vertex Mapping (핵심 알고리즘)

Per-frame OBJ는 14,522개 vertex를 가지지만, template은 15,399개(expanded)이다.
이 불일치를 해결하는 매핑 알고리즘:

```python
# 1. Template OBJ에서 원본 vertex index 파싱
faces_v_orig = []  # [28800, 3] — 각 face의 원본 vertex index
with open(template_obj) as f:
    for line in f:
        if line.startswith('f '):
            parts = line.split()[1:]
            fv = [int(p.split('/')[0]) - 1 for p in parts]
            faces_v_orig.append(fv)

# 2. Expanded→Original 매핑 구축
#    trimesh.faces[i][j] (expanded, 15399 범위)
#    faces_v_orig[i][j]  (original, 14522 범위)
expanded_to_orig = np.full(15399, -1, dtype=np.int32)
for i in range(28800):
    for j in range(3):
        exp_idx = template_mesh.faces[i][j]
        orig_idx = faces_v_orig[i][j]
        expanded_to_orig[exp_idx] = orig_idx

# 3. Per-frame vertex를 expanded 공간으로 치환
mesh.vertices = frame_verts[expanded_to_orig]  # [15399, 3]
```

### 3.5 UV Bug Fix 히스토리

2026-03-11에 수정된 버그. 이전에는 vertex 순서가 잘못 매핑되어 mesh가 왜곡됨.

| 지표 | Before (buggy) | After (fixed) | Delta |
|------|:--------------:|:-------------:|:-----:|
| IoU(GT) | 0.665 | 0.754 | **+0.088** |
| PSNR_masked(GT) | 8.75 dB | 10.42 dB | **+1.67 dB** |

> **상세**: `mouse_extensions/docs/UV_TEXTURE_RENDERING_BUG.md`

---

## 4. Stage 3: Per-frame Rendering

### 4.1 전체 Flow

```
Frame Index (e.g., 137)
    ↓
[1] OBJ 로드: step_2_frame_000685.obj (14,522 vertices)
    ↓
[2] UV Texture 적용: _load_textured_mesh() → 15,399 vertices
    ↓
[3] 좌표 변환: mammal_to_facelift(vertices)
    ↓
[4] Pyrender Scene 구성: mesh + camera + lighting
    ↓
[5] Offscreen Render: 384×384 RGB image
    ↓
[6] 저장: pseudo_gt/{view_name}/{frame_idx:05d}.png
```

### 4.2 좌표 변환

MAMMAL mm 공간 → FaceLift normalized 공간:

```python
M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])  # mm
M5_DISTANCE_SCALE = 2.7 / 307.785  # ≈ 0.008781

def mammal_to_facelift(points_mm):
    return (points_mm - M5_SCENE_CENTER) * M5_DISTANCE_SCALE
```

> `M5_SCENE_CENTER`: 마우스 행동 케이지의 중심 좌표 (mm)
> `307.785mm`: 6개 GT 카메라에서 scene center까지의 **평균 거리** (mm 단위)
> `2.7`: FaceLift turntable 카메라 반경 (normalized 단위)
> `M5_DISTANCE_SCALE`: 실세계 카메라 거리(307.785mm) → normalized 반경(2.7) 비율

### 4.3 Novel View 카메라

4개의 고정 시점 (모든 프레임 동일):

| View | Elevation | Azimuth | Purpose |
|------|:---------:|:-------:|---------|
| `bottom` | -70° | 0° | Ventral (belly, paws) |
| `top` | +70° | 0° | Dorsal (back, ears) |
| `front_low` | -30° | 0° | Frontal (face, whiskers) |
| `side_low` | -30° | 90° | Lateral (profile, gait) |

**Intrinsics**: fx=fy=411.75 @ 384×384 (GT 548.99@512에서 스케일)
**Radius**: 2.7 (turntable distance)

### 4.4 Pyrender 렌더링

```python
def _render_pyrender(mesh, cam_params, renderer, use_texture):
    c2w_cv = np.array(cam_params["c2w"])

    # OpenCV→OpenGL 좌표 변환
    cv_to_gl = np.diag([1, -1, -1, 1])
    c2w_gl = c2w_cv @ cv_to_gl

    # Camera 생성
    camera = pyrender.IntrinsicsCamera(
        fx=fx, fy=fy, cx=cx, cy=cy,
        znear=0.01, zfar=100.0,
    )

    # Scene 구성
    scene = pyrender.Scene(bg_color=[1, 1, 1, 1])
    scene.add(mesh)
    scene.add(camera, pose=c2w_gl)

    # Lighting (textured mesh → 강한 조명 필요)
    if use_texture:
        ambient_light = [1.0, 1.0, 1.0]
        key_light_intensity = 5.0      # C57BL/6 마우스가 검은색
        fill_light_intensity = 3.0     # 4방향
    else:
        ambient_light = [0.3, 0.3, 0.3]
        key_light_intensity = 3.0

    color, depth = renderer.render(scene)
    return color  # [H, W, 3] uint8
```

**핵심 포인트**:
- **좌표 변환**: OpenCV(Y-down, Z-forward) → OpenGL(Y-up, Z-backward) via `diag([1,-1,-1,1])`
- **조명**: C57BL/6 마우스는 검은색이므로 texture 모드에서 강한 조명(5.0) 필요
- **Backend**: `PYOPENGL_PLATFORM=egl` (headless GPU 서버)

### 4.5 Output Scale

| Component | Count |
|-----------|:-----:|
| Total frames | 3,600 |
| Novel views per frame | 4 |
| **Total pseudo-GT images** | **14,400** |
| Resolution | 384×384 |

---

## 5. Code References

| Component | File | Line |
|-----------|------|:----:|
| `generate_mammal()` | `mouse_extensions/scripts/novel_view/collect_dataset.py` | 711-800 |
| `_load_textured_mesh()` | `mouse_extensions/scripts/eval/poc_mesh_gs_pairs.py` | 326-397 |
| `_render_pyrender()` | `mouse_extensions/scripts/novel_view/collect_dataset.py` | 671-708 |
| `get_frame_obj_path()` | `mouse_extensions/scripts/novel_view/collect_dataset.py` | 232-239 |
| `mammal_to_facelift()` | `mouse_extensions/scripts/novel_view/collect_dataset.py` | 121-123 |
| `generate_novel_cameras()` | `mouse_extensions/scripts/novel_view/collect_dataset.py` | 242-290 |

## 6. Related Documents

- ↑ [[../INDEX]] — Document hub
- ↔ [[../experiments/mesh_gs_pair_collection]] — Dataset collection pipeline
- ↔ [[../../mouse_extensions/docs/UV_TEXTURE_RENDERING_BUG]] — UV texture bug fix
- ↔ [[../../mouse_extensions/docs/COORDINATE_SYSTEMS]] — Coordinate transforms
- ↔ [[../../mouse_extensions/docs/DATASET_FRAME_INDEXING]] — Frame indexing
- ↔ [[../experiments/DIFIX_TRAINING_STRATEGY]] — DiFix 3D+ training (uses pseudo-GT)

---

*FaceLift | MAMMAL Mesh Rendering Pipeline | 2026-03-13*
