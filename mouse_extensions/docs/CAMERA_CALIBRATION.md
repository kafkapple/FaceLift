# Camera Calibration & Parameter Reference

FaceLift PoC 렌더링 파이프라인의 카메라 파라미터 체계. MAMMAL mesh와 GS-LRM을 동일 카메라로 렌더링하기 위한 핵심 참조 문서.

---

## 1. Camera Parameter Sources

### 1.1 GT Cameras (from GS-LRM preprocessing)

**원본**: `opencv_cameras.json` (M5 전처리 데이터)

```json
{
  "frames": [{
    "w2c": [[4x4 matrix]],
    "fx": 548.99, "fy": 548.99,
    "cx": 256.0, "cy": 256.0,
    "w": 512, "h": 512
  }]
}
```

**로딩 과정** (`gslrm_pipeline.py:load_sample_data()`):
1. `w2c` → `c2w = np.linalg.inv(w2c)` (World-to-Camera → Camera-to-World)
2. 해상도 변경 시 intrinsics 비례 스케일링:
   ```python
   scale = target_resolution / original_width  # e.g., 384/512 = 0.75
   fx_scaled = fx * scale  # 548.99 * 0.75 = 411.74
   fy_scaled = fy * scale
   cx_scaled = cx * scale  # 256.0 * 0.75 = 192.0
   cy_scaled = cy * scale
   ```

**M5 GT Camera Parameters (at 384×384)**:

| Parameter | Value | Note |
|-----------|-------|------|
| fx, fy | 411.75 | Square pixels, symmetric |
| cx, cy | 192.0 | Image center |
| Resolution | 384×384 | Rescaled from 512×512 |
| FoV (half) | ~25.1° | `arctan(192/411.75)` |
| # Views | 6 | Fixed camera rig |
| Camera distance | ~2.7 | Normalized (all frames identical) |

### 1.2 Novel Cameras (generated)

**생성 방식**: Spherical coordinates → OpenCV C2W matrix

```python
# Spherical to Cartesian (camera position)
x = radius * cos(elevation) * cos(azimuth)
y = radius * cos(elevation) * sin(azimuth)
z = radius * sin(elevation)
eye = center + [x, y, z]

# Look-at → C2W (OpenCV convention)
forward = normalize(target - eye)      # Z-forward (into screen)
right = normalize(cross(forward, up))  # X-right
down = cross(forward, right)           # Y-down
c2w = [[right, down, forward, eye], [0,0,0,1]]
```

**Novel View Definitions**:

| View | Elevation | Azimuth | Purpose |
|------|-----------|---------|---------|
| bottom | -70° | 0° | Unseen region (primary artifact area) |
| top | +70° | 0° | Overhead view |
| front_low | -30° | 0° | Front-facing low angle |
| side_low | -30° | 90° | Side-facing low angle |

**Intrinsics**: GT 카메라와 동일 (`fx = 548.99 * (resolution/512)`)

> **⚠️ 수정 이력 (2026-03-11)**: 초기에 `fx = resolution * 1.3 = 499.2` 사용 → GT와 21% focal length 불일치 발견 → GT 기반으로 수정. 심의에서 3개 모델 모두 이 문제 지적.

---

## 2. Coordinate Convention

### 2.1 Convention Map

```
                    OpenCV (GS-LRM native)          OpenGL (pyrender)
                    ─────────────────────          ──────────────────
    Y-axis          Down ↓                         Up ↑
    Z-axis          Forward → (into screen)        Backward ← (out of screen)
    X-axis          Right →                        Right →
    Handedness      Right-handed                   Right-handed
```

### 2.2 Conversion Formula

```python
# OpenCV C2W → OpenGL C2W (pyrender에서 사용)
cv_to_gl = np.diag([1, -1, -1, 1])  # Flip Y and Z
c2w_gl = c2w_cv @ cv_to_gl

# 의미: 카메라의 local Y축과 Z축을 반전
# - Y: down → up
# - Z: forward → backward
# - X: 유지 (right)
# - Translation: 유지 (world 좌표이므로)
```

**Why `@` (right-multiply)?**
- `c2w_cv`의 columns = camera axes in world space
- Right-multiplying `diag(1,-1,-1,1)` flips the Y/Z columns (camera axes) while keeping translation unchanged
- This is equivalent to: `c2w_gl[:3, 1] = -c2w_cv[:3, 1]` and `c2w_gl[:3, 2] = -c2w_cv[:3, 2]`

### 2.3 pyrender IntrinsicsCamera

```python
camera = pyrender.IntrinsicsCamera(
    fx=411.75, fy=411.75,   # Focal length in pixels
    cx=192.0, cy=192.0,     # Principal point in pixels
    znear=0.01, zfar=100.0, # Clipping planes
)
scene.add(camera, pose=c2w_gl)  # OpenGL convention C2W
```

**주의사항**:
- pyrender `cx/cy`는 **pixel 좌표** (not normalized)
- pyrender 내부에서 NDC projection matrix 자동 생성
- `znear`이 너무 크면 가까운 geometry가 잘림

---

## 3. Rendering Pipeline

### 3.1 GS-LRM Rendering (Phase 1 / facelift env)

```
opencv_cameras.json
    ↓ load_sample_data()
c2w_cv [4,4] + fxfycxcy [4]
    ↓ render_opencv_cam()
Rendered image [H,W,3]
```

`render_opencv_cam()`은 내부적으로 OpenCV convention을 직접 처리 → **변환 불필요**.

### 3.2 MAMMAL Mesh Rendering (Phase 2 / mammal_stable env)

```
camera_config.json (Phase 1에서 저장)
    ↓ c2w_cv from JSON
c2w_cv [4,4] + fxfycxcy [4]
    ↓ c2w_gl = c2w_cv @ diag(1,-1,-1,1)
c2w_gl [4,4]
    ↓ pyrender.IntrinsicsCamera(fx, fy, cx, cy)
    ↓ scene.add(camera, pose=c2w_gl)
    ↓ renderer.render(scene)
Rendered image [H,W,3]
```

### 3.3 Data Flow Diagram

```
Phase 1 (facelift env)                    Phase 2 (mammal_stable env)
━━━━━━━━━━━━━━━━━━━                    ━━━━━━━━━━━━━━━━━━━━━━━━━━

M5 Sample                                camera_config.json
├── opencv_cameras.json ──→ load ──→ ──→ ├── gt_cameras
│   (w2c + intrinsics)     c2w_cv        │   (c2w_cv + fxfycxcy)
│                            │           │        ↓
│                            ↓           │   pyrender (OpenGL)
│                     render_opencv_cam   │        ↓
│                            │           │   mammal_gt/*.png
│                            ↓           │
└── images/cam_*.png    gslrm_gt/*.png   ├── novel_cameras
                        gslrm_novel/*.png│   (c2w_cv + fxfycxcy)
                                         │        ↓
                                         │   pyrender (OpenGL)
                                         │        ↓
                                         └── mammal_novel/*.png
```

---

## 4. UV Texture Integration

### 4.1 Texture Assets

| File | Description |
|------|-------------|
| `exports/texture_final.png` | 512×512 RGB, multi-view 역투영으로 생성 |
| `exports/mouse_frame0_textured.obj` | UV coords + face-UV mapping 포함 |
| `exports/mouse_frame0_textured.mtl` | Material definition (`map_Kd texture_final.png`) |
| `mouse_model/mouse_txt/textures.txt` | 15,399 UV coordinates (0-1 normalized) |
| `mouse_model/mouse_txt/faces_tex.txt` | 28,800 face → UV index mapping |

### 4.2 Texture Loading Strategy

```python
# Template OBJ (UV topology) + per-frame OBJ (vertex positions)
template = trimesh.load("mouse_frame0_textured.obj", process=False)
frame_mesh = trimesh.load(f"step_2_frame_{idx:06d}.obj", process=False)

# UV topology is shared — only vertex positions change per frame
template.vertices = frame_mesh.vertices  # Swap positions
# template retains UV coords, faces_tex, material → pyrender.Mesh.from_trimesh()
```

### 4.3 Lighting for Textured Mesh

**⚠️ Baked Lighting 주의**: `texture_final.png`은 multi-view 역투영으로 생성되어 실제 조명 정보가 포함됨. pyrender에서 추가 조명을 강하게 넣으면 이중 조명 문제 발생.

```python
if use_texture:
    scene = pyrender.Scene(
        bg_color=[1,1,1,1],
        ambient_light=[0.5, 0.5, 0.5],  # Moderate ambient
    )
    # Gentle fill light only
    light = pyrender.DirectionalLight(intensity=1.5)
else:
    scene = pyrender.Scene(
        bg_color=[1,1,1,1],
        ambient_light=[0.3, 0.3, 0.3],
    )
    # Stronger lighting for flat material
    light = pyrender.DirectionalLight(intensity=3.0)
```

---

## 5. Verification Checklist

### Camera Alignment

- [x] GT cameras: w2c → c2w inversion 확인
- [x] GT intrinsics: 해상도 비례 스케일링 확인 (512→384: fx 548.99→411.75)
- [x] Novel intrinsics: GT focal length 사용 (resolution*1.3 → 수정)
- [x] OpenCV→OpenGL: `c2w_gl = c2w_cv @ diag(1,-1,-1,1)` 확인
- [ ] GT view overlay: MAMMAL render가 GT RGB의 생쥐 위치와 일치하는지 시각 확인
- [ ] Novel view: GS-LRM render와 MAMMAL render의 생쥐 크기/위치 일치 확인

### UV Texture

- [x] Template OBJ vertex count = frame OBJ vertex count (14,522)
- [x] texture_final.png: 512×512 RGB 확인
- [ ] Textured render: seam/stretching artifact 없는지 확인
- [ ] Baked lighting: 이중 조명 없는지 확인

---

## 6. Known Issues

| Date | Issue | Cause | Resolution |
|------|-------|-------|------------|
| 2026-03-11 | Novel view fx=499.2 vs GT fx=411.75 | `fx = resolution * 1.3` 근사값 사용 | GT focal length로 수정 |
| 2026-03-11 | pyrender NoSuchDisplayException | Headless server에서 X11 없음 | `PYOPENGL_PLATFORM=egl` 환경변수 |
| 2026-03-11 | `.numpy()` on grad tensor | `render_opencv_cam` 결과에 grad 있음 | `.detach()` 추가 |

---

## Related Documents

- `↑ MOC`: `mouse_extensions/docs/` (domain docs)
- `↔ Coordinate Systems`: `COORDINATE_SYSTEMS.md` (좌표계 변환 공식)
- `↔ PoC Script`: `scripts/eval/poc_mesh_gs_pairs.py`
- `↔ GS-LRM Pipeline`: `inference/gslrm_pipeline.py` (load_sample_data)
- `↔ Renderer`: `gslrm/model/gaussians_renderer.py:render_opencv_cam()`

---

*FaceLift | Camera Calibration Reference | Created: 2026-03-11*
