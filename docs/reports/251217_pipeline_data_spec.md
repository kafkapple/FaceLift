# FaceLift Pipeline Data Specification (생쥐 데이터용)

**날짜**: 2024-12-17
**목적**: MVDiffusion과 GS-LRM 각 단계별 입출력 데이터 스펙 정리

---

## 전체 파이프라인 구조

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         FaceLift Pipeline (Mouse Data)                          │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Stage 0    │ →   │   Stage 1    │ →   │   Stage 2    │ →   │   Output     │
│  Raw Data    │     │  MVDiffusion │     │   GS-LRM     │     │  3D Model    │
│  (6 views)   │     │ (1→6 views)  │     │ (6→3D GS)    │     │ (Gaussians)  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
     288×256              512×512             512×512            3D Points
     6 cameras          1 input view        6 target views       + Colors
```

---

## Stage 0: Raw Mouse Data (원본 데이터)

### 디렉토리 구조
```
data_mouse/sample_XXXXXX/
├── images/
│   ├── cam_000.png    # View 0
│   ├── cam_001.png    # View 1
│   ├── cam_002.png    # View 2
│   ├── cam_003.png    # View 3
│   ├── cam_004.png    # View 4
│   └── cam_005.png    # View 5
└── opencv_cameras.json
```

### 이미지 스펙

| 항목 | 스펙 | 생쥐 데이터 예시 |
|------|------|------------------|
| **해상도** | H × W | 256 × 288 (또는 512×512 리사이즈) |
| **채널** | C | 3 (RGB) 또는 4 (RGBA) |
| **포맷** | - | PNG |
| **뷰 수** | N | 6 |
| **배경** | - | 흰색 (배경 제거됨) |

### 카메라 파라미터 (opencv_cameras.json)

```json
{
  "frames": [
    {
      "w": 288,                          // 이미지 너비
      "h": 256,                          // 이미지 높이
      "fx": 408.1,                       // focal length x (pixels)
      "fy": 409.8,                       // focal length y (pixels)
      "cx": 144.0,                       // principal point x
      "cy": 128.0,                       // principal point y
      "w2c": [                           // World-to-Camera 4×4 행렬
        [r00, r01, r02, tx],
        [r10, r11, r12, ty],
        [r20, r21, r22, tz],
        [0,   0,   0,   1]
      ],
      "file_path": "images/cam_000.png",
      "view_id": 0
    },
    // ... 5 more cameras
  ]
}
```

### 좌표계

| 항목 | 원본 | 정규화 후 |
|------|------|-----------|
| **Up Direction** | ~[0, 0, 1] (Z-up) | [0, 1, 0] (Y-up) |
| **카메라 배치** | XY 평면 orbit | XZ 평면 orbit |

---

## Stage 1: MVDiffusion (Multi-View Diffusion)

### 역할
- **입력**: 단일 뷰 이미지 1장
- **출력**: 멀티뷰 이미지 6장 (다른 각도에서 본 예측)

### 입력 텐서 스펙

```python
# 입력 데이터 구조
input_batch = {
    "pixel_values": torch.Tensor,      # [B, 1, 3, H, W]
    "camera_param": torch.Tensor,      # [B, N_target, 12]
}
```

| 텐서 | 차원 | 생쥐 데이터 예시 | 설명 |
|------|------|------------------|------|
| `pixel_values` | [B, 1, 3, H, W] | [2, 1, 3, 512, 512] | 입력 이미지 (1장) |
| `camera_param` | [B, N, 12] | [2, 6, 12] | 타겟 카메라 파라미터 |

### camera_param 구성 (12차원)

```
camera_param[i] = [
    fx/W, fy/H, cx/W, cy/H,           # Intrinsics (4)
    r00, r01, r02,                     # Rotation row 0 (3)
    r10, r11, r12,                     # Rotation row 1 (3)
    elevation, azimuth                 # Spherical coords (2) - optional
]
```

### 출력 텐서 스펙

```python
# 출력 데이터 구조
output = {
    "images": torch.Tensor,            # [B, N, 3, H, W]
}
```

| 텐서 | 차원 | 생쥐 데이터 예시 | 설명 |
|------|------|------------------|------|
| `images` | [B, N, 3, H, W] | [2, 6, 3, 512, 512] | 생성된 6개 뷰 |

### 모델 아키텍처 요약

```
Input Image (512×512)
    │
    ▼
┌─────────────────────┐
│  Image Encoder      │  CLIP/VAE Encoder
│  (Latent: 64×64)    │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Cross-View         │  각 뷰 간 attention
│  Attention UNet     │  Stable Diffusion 기반
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Diffusion Process  │  T timesteps 반복
│  (Denoising)        │
└─────────────────────┘
    │
    ▼
Output: 6 × (512×512) Images
```

---

## Stage 2: GS-LRM (Gaussian Splatting Large Reconstruction Model)

### 역할
- **입력**: 멀티뷰 이미지 (1~6장) + 카메라 파라미터
- **출력**: 3D Gaussian Splatting 포인트 클라우드

### 입력 텐서 스펙

```python
# 입력 데이터 구조
input_batch = {
    "image": torch.Tensor,             # [B, N, C, H, W]
    "c2w": torch.Tensor,               # [B, N, 4, 4]
    "fxfycxcy": torch.Tensor,          # [B, N, 4]
    "bg_color": torch.Tensor,          # [B, 3]
    "index": torch.Tensor,             # [B, N, 2]
}
```

| 텐서 | 차원 | 생쥐 데이터 예시 | 설명 |
|------|------|------------------|------|
| `image` | [B, N, C, H, W] | [2, 6, 4, 512, 512] | 입력 이미지 (RGBA) |
| `c2w` | [B, N, 4, 4] | [2, 6, 4, 4] | Camera-to-World 행렬 (**Y-up 정규화됨**) |
| `fxfycxcy` | [B, N, 4] | [2, 6, 4] | [fx, fy, cx, cy] 픽셀 단위 |
| `bg_color` | [B, 3] | [2, 3] | 배경색 RGB [0-1] |
| `index` | [B, N, 2] | [2, 6, 2] | [view_id, scene_id] |

### 입력 텐서 상세

#### image (이미지 텐서)
```python
# Shape: [B, N, C, H, W]
# B = batch_size (2)
# N = num_views (6)
# C = channels (4 for RGBA, 3 for RGB)
# H, W = 512, 512

image.shape = [2, 6, 4, 512, 512]
image.dtype = torch.float32
image.range = [0.0, 1.0]  # 정규화됨
```

#### c2w (Camera-to-World 행렬)
```python
# Shape: [B, N, 4, 4]
# 4×4 변환 행렬

c2w = [
    [R(3×3) | t(3×1)]    # R: 회전, t: 위치
    [0 0 0  |   1   ]
]

# 카메라 위치: c2w[:3, 3]
# 카메라 forward: -c2w[:3, 2]
# 카메라 up: -c2w[:3, 1]
```

#### fxfycxcy (Intrinsics)
```python
# Shape: [B, N, 4]
# 픽셀 단위의 카메라 내부 파라미터

fxfycxcy = [fx, fy, cx, cy]

# 생쥐 데이터 예시 (512×512 리사이즈 후):
# fx ≈ 725.3  (408.1 × 512/288)
# fy ≈ 819.6  (409.8 × 512/256)
# cx = 256.0  (centered)
# cy = 256.0  (centered)
```

### 출력 텐서 스펙 (3D Gaussians)

```python
# 출력 데이터 구조
output = {
    "gaussians": GaussianModel,        # 3D Gaussian 객체
    "render": torch.Tensor,            # [B, N, 3, H, W] 렌더링 결과
}

# GaussianModel 내부 속성
gaussians = {
    "xyz": torch.Tensor,               # [M, 3] - 위치
    "features_dc": torch.Tensor,       # [M, 3] - RGB 색상
    "opacity": torch.Tensor,           # [M, 1] - 불투명도
    "scaling": torch.Tensor,           # [M, 3] - 크기 (x, y, z)
    "rotation": torch.Tensor,          # [M, 4] - 회전 (quaternion)
}
```

| 텐서 | 차원 | 생쥐 데이터 예시 | 설명 |
|------|------|------------------|------|
| `xyz` | [M, 3] | [~65536, 3] | Gaussian 중심 좌표 |
| `features_dc` | [M, 3] | [~65536, 3] | RGB 색상 [0-1] |
| `opacity` | [M, 1] | [~65536, 1] | 불투명도 [0-1] |
| `scaling` | [M, 3] | [~65536, 3] | 3D 크기 (log scale) |
| `rotation` | [M, 4] | [~65536, 4] | Quaternion [w, x, y, z] |

**M (Gaussian 개수) 계산:**
```
M = N_views × (H/patch_size) × (W/patch_size) × n_gaussians
  = 6 × (512/8) × (512/8) × 2
  = 6 × 64 × 64 × 2
  = 49,152
```

### 모델 아키텍처 요약

```
Input: 6 × (512×512) Images + Cameras
    │
    ▼
┌─────────────────────────┐
│  Image Tokenizer        │  Patchify: 8×8 patches
│  → 64×64 tokens/view    │  + Plucker ray embedding
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Transformer            │  24 layers
│  (d=1024, heads=16)     │  Cross-view attention
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Gaussian Decoder       │  Token → Gaussian params
│  (MLP heads)            │  xyz, rgb, opacity, scale, rot
└─────────────────────────┘
    │
    ▼
Output: ~50K 3D Gaussians
```

---

## Stage 3: Gaussian Rasterizer (렌더링)

### 역할
- **입력**: 3D Gaussians + 카메라 파라미터
- **출력**: 2D 렌더링 이미지

### 입력

```python
rasterizer(
    means3D=xyz,           # [M, 3] Gaussian 위치
    colors=features_dc,    # [M, 3] RGB 색상
    opacities=opacity,     # [M, 1] 불투명도
    scales=scaling,        # [M, 3] 크기
    rotations=rotation,    # [M, 4] 회전
    viewpoint_camera={
        "c2w": c2w,        # [4, 4] Camera-to-World
        "fxfycxcy": [...], # [4] Intrinsics
        "image_height": H,
        "image_width": W,
    }
)
```

### 출력

```python
rendered_image  # [H, W, 3] RGB 이미지 (float32, 0-1)
```

---

## 카메라 좌표계 변환

### 원본 → Y-up 정규화

```python
# 문제: 생쥐 데이터 up direction ≠ Y-up
# Pretrained GS-LRM은 Y-up 좌표계 가정

# 1. Up direction 추정 (PCA)
positions = [c2w[:3, 3] for c2w in cameras]
cov = centered_positions.T @ centered_positions
up_direction = smallest_eigenvector(cov)  # Orbit plane normal

# 2. 회전 행렬 계산 (Rodrigues)
rotation_axis = cross(up_direction, [0, 1, 0])
angle = arccos(dot(up_direction, [0, 1, 0]))
R_align = rodrigues(rotation_axis, angle)

# 3. 모든 카메라에 적용
for c2w in cameras:
    c2w_new[:3, :3] = R_align @ c2w[:3, :3]
    c2w_new[:3, 3] = R_align @ c2w[:3, 3]
```

### 변환 전후 비교

| 항목 | 변환 전 (Original) | 변환 후 (Normalized) |
|------|-------------------|---------------------|
| **Up Vector** | [0, 0, 1] | [0, 1, 0] |
| **카메라 평면** | XY plane | XZ plane |
| **좌표계** | Z-up | Y-up |

---

## 학습 설정 요약

### MVDiffusion (Stage 1)

| 항목 | 값 |
|------|-----|
| Batch size | 4 |
| Learning rate | 1e-5 |
| Image size | 512×512 |
| Timesteps | 1000 |
| Views | 6 |

### GS-LRM (Stage 2)

| 항목 | 값 | 비고 |
|------|-----|------|
| Batch size | 2 | GPU 메모리 제한 |
| Learning rate | **1e-6** | Gradient explosion 방지 |
| Image size | 512×512 | |
| Input views | 1 | 단일 뷰 입력 |
| Target views | 6 | 모든 뷰 supervision |
| Grad clip | 0.5 | 안정화 |
| Warmup | 500 steps | |

---

## 관련 파일

| 파일 | 설명 |
|------|------|
| `gslrm/data/mouse_dataset.py` | MouseViewDataset (카메라 정규화 포함) |
| `train_gslrm.py` | GS-LRM 학습 스크립트 |
| `configs/mouse_gslrm_lowlr.yaml` | 생쥐용 Low-LR 학습 설정 |
| `scripts/visualize_camera_normalization.py` | 카메라 정규화 시각화 |
