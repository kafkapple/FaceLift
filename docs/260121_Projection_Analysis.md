# 260121 3D→2D Projection Analysis

> **Date**: 2026-01-21
> **Purpose**: GS-LRM에서 사용하는 3D→2D projection 공식 분석

---

## 1. Overview

GS-LRM은 3D Gaussian Splatting을 렌더링할 때 **OpenGL 스타일의 projection matrix**를 사용합니다.

---

## 2. Projection Pipeline

```
3D World Coordinates (x, y, z)
        │
        │ World-View Transform (W2C)
        ▼
Camera Coordinates (X, Y, Z)
        │
        │ Projection Matrix (P)
        ▼
Normalized Device Coordinates (NDC)
        │
        │ Viewport Transform
        ▼
Pixel Coordinates (u, v)
```

---

## 3. Projection Matrix Construction

**코드 위치**: `gslrm/model/gaussians_renderer.py:287-296`

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

### 3.1 Matrix Structure

$$
P = \begin{bmatrix}
\frac{2f_x}{W} & 0 & \frac{2c_x}{W} - 1 & 0 \\
0 & \frac{2f_y}{H} & \frac{2c_y}{H} - 1 & 0 \\
0 & 0 & -\frac{z_f + z_n}{z_f - z_n} & -\frac{2z_f z_n}{z_f - z_n} \\
0 & 0 & 1 & 0
\end{bmatrix}
$$

### 3.2 Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `W, H` | 이미지 너비, 높이 | 512 |
| `fx, fy` | 초점 거리 (pixels) | ~549 (normalized) |
| `cx, cy` | 주점 (principal point) | varies |
| `znear, zfar` | 클리핑 평면 | 0.01, 100.0 |

---

## 4. Full Transform Chain

**코드 위치**: `gslrm/model/gaussians_renderer.py:299-307`

```python
# World to Camera (extrinsic)
self.world_view_transform = self.W2C.transpose(0, 1)

# Camera to NDC (intrinsic + projection)
self.projection_matrix = getProjectionMatrix(...).transpose(0, 1)

# Combined: World to NDC
self.full_proj_transform = world_view_transform @ projection_matrix

# Camera center in world coordinates
self.camera_center = self.C2W[:3, 3]
```

---

## 5. Standard Formulation vs GS-LRM

### 5.1 Standard Pinhole Model

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
= K \cdot [R | t] \cdot \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

Where:
$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

### 5.2 GS-LRM (OpenGL Style)

GS-LRM은 OpenGL NDC 좌표계를 사용:
- NDC 범위: [-1, 1] × [-1, 1]
- Z 방향: -Z가 전방 (right-handed)

$$
\begin{bmatrix} x_{ndc} \\ y_{ndc} \\ z_{ndc} \\ w \end{bmatrix}
= P \cdot M_{view} \cdot \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
$$

Pixel 좌표:
$$
u = \frac{(x_{ndc}/w + 1) \cdot W}{2}, \quad
v = \frac{(y_{ndc}/w + 1) \cdot H}{2}
$$

---

## 6. Key Insights

### 6.1 Principal Point Handling

- `P[0,2] = 2*(cx/W) - 1`: cx를 NDC로 변환
- cx=W/2일 때 → P[0,2]=0 (중앙)
- cx≠W/2일 때 → off-center projection

**Bug in D4**: `cx=cy=256`으로 강제 → PP 오류 → Ghosting
**Fix in D6-3**: 실제 PP 값 사용

### 6.2 Focal Length Normalization

- Pretrained 모델: `fx≈549` 기대
- Mouse 원본: `fx≈844`
- 정규화 필수: `fx_norm = fx * (target_dist / actual_dist)`

---

## 7. References

- `gslrm/model/gaussians_renderer.py`: Camera, GaussianModel classes
- `mouse_extensions/preprocessing/camera_normalizer.py`: 정규화 모듈
- `docs/PREPROCESSING_REGISTRY.md`: 전처리 방식별 PP 처리

---

*Created: 2026-01-21*
