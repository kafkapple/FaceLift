# Camera Data Pipeline - FaceLift Mouse Extension

> **Created**: 2026-01-21
> **Purpose**: opencv_cameras.json → 렌더링까지 카메라 데이터 흐름 문서화

---

## 1. 전체 파이프라인 개요

```
opencv_cameras.json (W2C)
         │
         ▼
    Dataset 로딩
    (W2C → C2W 변환)
         │
    ┌────┴────┐
    │ Mouse   │  PP 보정, 카메라 정규화
    │ 특화    │  (mouse_dataset.py)
    └────┬────┘
         │
         ▼
    GSLRM Forward
    (C2W 전달)
         │
         ▼
    Gaussian Renderer
    (C2W → W2C, Projection Matrix)
         │
         ▼
    렌더링된 이미지
```

---

## 2. opencv_cameras.json 포맷

**원본 FaceLift 포맷** (pretrained 모델 기준):
```json
{
  "frames": [
    {
      "file_path": "images/000000.png",
      "w2c": [[...], [...], [...], [...]],  // 4x4 World-to-Camera
      "fx": 549.0,
      "fy": 549.0,
      "cx": 256.0,
      "cy": 256.0
    },
    ...
  ]
}
```

**핵심**: `w2c`는 **World-to-Camera** 변환 행렬 (OpenCV 컨벤션)

---

## 3. 원본 FaceLift 코드

### 3.1 Dataset 로딩 (`gslrm/data/dataset.py`)

**W2C 로딩** (line 211):
```python
data_json_path = os.path.join(self.all_data_paths[idx].strip(), "opencv_cameras.json")
with open(data_json_path, 'r') as f:
    data_json = json.load(f)
cameras = data_json["frames"]
```

**Intrinsics 추출** (line 269):
```python
intrinsics = np.array([camera["fx"], camera["fy"], camera["cx"], camera["cy"]])
intrinsics *= resize_ratio  # 리사이즈 시 비례 조정
```

**W2C → C2W 변환** (line 277):
```python
c2w = np.linalg.inv(np.array(camera["w2c"]))
```

### 3.2 Gaussian Renderer (`gslrm/model/gaussians_renderer.py`)

**Camera 클래스** (line 386):
```python
class Camera:
    def __init__(self, C2W, fxfycxcy, h, w):
        self.C2W = C2W.clone().float()
        self.W2C = self.C2W.inverse()  # C2W → W2C 역변환
```

**Projection Matrix** (line 397):
```python
def getProjectionMatrix(W, H, fx, fy, cx, cy, znear, zfar):
    P = torch.zeros(4, 4, device=fx.device)
    P[0, 0] = 2 * fx / W
    P[1, 1] = 2 * fy / H
    P[0, 2] = 2 * (cx / W) - 1
    P[1, 2] = 2 * (cy / H) - 1
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[3, 2] = 1.0
    P[2, 3] = -(2 * zfar * znear) / (zfar - znear)
    return P
```

**Full Transform** (line 408-415):
```python
self.world_view_transform = self.W2C.transpose(0, 1)
self.projection_matrix = getProjectionMatrix(...).transpose(0, 1)
self.full_proj_transform = self.world_view_transform @ self.projection_matrix
self.camera_center = self.C2W[:3, 3]
```

### 3.3 GSLRM 모델 (`gslrm/model/gslrm.py`)

**Renderer 선언** (line 79-170):
```python
class Renderer(nn.Module):
    def forward(self, xyz, features, scaling, rotation, opacity,
                height, width, C2W, fxfycxcy, deferred=True):
        if deferred:
            return deferred_gaussian_render(
                xyz, features, scaling, rotation, opacity,
                height, width, C2W, fxfycxcy, self.scaling_modifier
            )
```

**Sequential Rendering** (line 165):
```python
for j in range(v):
    result = render_opencv_cam(pc, height, width, C2W[i, j], fxfycxcy[i, j])
    renderings[i, j] = result["render"]
```

---

## 4. Mouse 특화 코드 (`gslrm/data/mouse_dataset.py`)

### 4.1 추가된 기능

| 기능 | 위치 | 설명 |
|------|------|------|
| PP 보정 | line 380-392 | Principal Point 정확한 계산 |
| 카메라 정규화 | line 448-450 | fx→549, distance→2.7 |
| 축 정규화 | line 440-442 | Z-up/Y-up 변환 |
| 카메라 제외 | line 254-270 | Ablation용 특정 카메라 제외 |

### 4.2 PP 보정 (`apply_pp_correction`)

```python
# line 380-392
if self.pp_correction_enabled:
    image, intrinsics = apply_pp_correction(
        image, intrinsics, center_x, center_y, target_size
    )
else:
    intrinsics = np.array([
        camera["fx"], camera["fy"], camera["cx"], camera["cy"]
    ])
    intrinsics *= resize_ratio
```

**왜 필요?**
- 원본 FaceLift: cx=cy=256 가정 (이미지 중앙)
- Mouse 데이터: crop 위치에 따라 cx, cy 가변
- 정확한 PP 없으면 ray 방향 오류 → ghosting

### 4.3 카메라 정규화

```python
# line 448-450
input_c2ws, input_fxfycxcy = normalize_camera_distance_with_intrinsics(
    input_c2ws, input_fxfycxcy, self.target_camera_distance
)
```

**정규화 타겟** (pretrained 모델 분포):
- `fx = fy = 549.0`
- `camera_distance ≈ 2.7`

### 4.4 카메라 제외 (Ablation)

```python
# line 254-270
if self.exclude_camera_indices:
    cameras = [c for i, c in enumerate(cameras) 
               if i not in self.exclude_camera_indices]
```

**Config 예시**:
```yaml
training:
  dataset:
    exclude_camera_indices: [3]  # 카메라 3 제외
```

---

## 5. 데이터 흐름 상세

### 5.1 좌표계 변환

```
World Space (3D)
     │
     │ W2C (world_view_transform)
     ▼
Camera Space (view coordinates)
     │
     │ Projection Matrix
     ▼
NDC (Normalized Device Coordinates)
     │
     │ Viewport Transform
     ▼
Screen Space (pixels)
```

### 5.2 행렬 관계

```python
# 저장: W2C (World-to-Camera)
# 로딩: C2W = inv(W2C)
# 렌더링: W2C = inv(C2W)  # 다시 역변환

# Full projection
P_full = W2C.T @ P_proj.T
point_screen = P_full @ point_world
```

### 5.3 Intrinsics 의미

```
fx, fy: Focal length (pixels)
cx, cy: Principal point (pixels)

fx = f * sx  (f: focal length mm, sx: pixel/mm)
```

**Projection**:
```
u = fx * (X/Z) + cx
v = fy * (Y/Z) + cy
```

---

## 6. 코드 위치 참조

| 기능 | 파일 | 라인 |
|------|------|------|
| W2C 로딩 (원본) | `gslrm/data/dataset.py` | 211 |
| Intrinsics (원본) | `gslrm/data/dataset.py` | 269 |
| C2W 변환 (원본) | `gslrm/data/dataset.py` | 277 |
| W2C 로딩 (Mouse) | `gslrm/data/mouse_dataset.py` | 300 |
| PP 보정 | `gslrm/data/mouse_dataset.py` | 380-392 |
| 카메라 정규화 | `gslrm/data/mouse_dataset.py` | 448-450 |
| Projection Matrix | `gslrm/model/gaussians_renderer.py` | 397-406 |
| Camera 클래스 | `gslrm/model/gaussians_renderer.py` | 377-420 |
| render_opencv_cam | `gslrm/model/gaussians_renderer.py` | 879-920 |
| Renderer.forward | `gslrm/model/gslrm.py` | 133-140 |

---

## 7. 원본 GitHub 참조

| 항목 | URL |
|------|-----|
| W2C 로딩 | [dataset.py#L211](https://github.com/weijielyu/FaceLift/blob/76af02634dc63bd53f22eceaa2cc0b5a11c7a8cd/gslrm/data/dataset.py#L211) |
| Intrinsics | [dataset.py#L269](https://github.com/weijielyu/FaceLift/blob/76af02634dc63bd53f22eceaa2cc0b5a11c7a8cd/gslrm/data/dataset.py#L269) |
| C2W 변환 | [dataset.py#L277](https://github.com/weijielyu/FaceLift/blob/76af02634dc63bd53f22eceaa2cc0b5a11c7a8cd/gslrm/data/dataset.py#L277) |
| Renderer 선언 | [gslrm.py#L585](https://github.com/weijielyu/FaceLift/blob/76af02634dc63bd53f22eceaa2cc0b5a11c7a8cd/gslrm/model/gslrm.py#L585) |
| 렌더링 호출 | [gslrm.py#L1096](https://github.com/weijielyu/FaceLift/blob/76af02634dc63bd53f22eceaa2cc0b5a11c7a8cd/gslrm/model/gslrm.py#L1096) |
| W2C 계산 | [gaussians_renderer.py#L272](https://github.com/weijielyu/FaceLift/blob/76af02634dc63bd53f22eceaa2cc0b5a11c7a8cd/gslrm/model/gaussians_renderer.py#L272) |

---

## 8. 요약

1. **opencv_cameras.json**: W2C 행렬 저장
2. **Dataset**: W2C → C2W 변환 후 반환
3. **Mouse 특화**: PP 보정, 카메라 정규화 추가
4. **GSLRM**: C2W를 Renderer에 전달
5. **Renderer**: C2W → W2C 역변환 후 Projection Matrix와 결합
6. **최종**: 3D Gaussian → 2D 이미지 렌더링

---

*Engram v1.0 | Created: 2026-01-21*
