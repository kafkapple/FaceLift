---
date: 2025-12-20
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, gaussian-avatar, synthetic-data, implementation-plan]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# Gaussian Mouse Avatar 구현 계획

## 1. 개요

### 1.1 목표
단일 카메라 환경에서 학습된 모델을 다양한 카메라 배치에 일반화하기 위해:
1. **Gaussian Mouse Avatar** 생성
2. **합성 다중 뷰 데이터** 생성 (다양한 각도/거리)
3. **GS-LRM Fine-tuning**으로 일반화 성능 향상

### 1.2 참고 연구
| 논문/코드 | 핵심 기여 | 활용 포인트 |
|-----------|----------|-------------|
| **MoReMouse** (arXiv 2507.04258) | Gaussian mouse avatar + 합성 데이터 | 전체 파이프라인 컨셉 |
| **Pose Splatter** (arXiv 2505.18342) | 3D GS for animals, rotation-invariant | Gaussian 표현 방식 |
| **MAMMAL_mouse** | LBS skinning body model | 해부학적 정확성 |
| **pose-splatter** | Mesh→Gaussian, multi-view rendering | 구현 인프라 |

---

## 2. 품질 평가 모듈 (Quality Evaluation Module)

### 2.1 목적
> **핵심 질문**: Fine-tuning 없이 zero-shot으로 사용 가능한가?

MVDiffusion 출력 품질을 **정량적으로** 평가하여 fine-tune 필요성을 자동 판단

### 2.2 평가 메트릭

```python
# scripts/evaluate_mvdiff_quality.py

class MVDiffQualityEvaluator:
    """MVDiffusion 출력 품질 자동 평가기"""

    QUALITY_THRESHOLDS = {
        # Tier 1: View Consistency (가장 중요)
        'lpips_cross_view': {
            'excellent': 0.15,   # 뷰 간 일관성 우수
            'good': 0.25,
            'acceptable': 0.35,
            'threshold': 0.40    # 이 이상이면 fine-tune 필수
        },

        # Tier 2: Reference Fidelity
        'ref_psnr': {
            'excellent': 30.0,
            'good': 26.0,
            'acceptable': 22.0,
            'threshold': 20.0    # 이 이하면 fine-tune 필수
        },
        'ref_ssim': {
            'excellent': 0.95,
            'good': 0.90,
            'acceptable': 0.85,
            'threshold': 0.80
        },

        # Tier 3: Mask Quality
        'mask_iou': {
            'excellent': 0.95,
            'good': 0.90,
            'acceptable': 0.85,
            'threshold': 0.80
        },
        'mask_edge_sharpness': {
            'excellent': 0.70,
            'good': 0.55,
            'acceptable': 0.40,
            'threshold': 0.30
        },

        # Tier 4: Geometric Consistency (epipolar)
        'epipolar_error': {
            'excellent': 2.0,    # pixels
            'good': 5.0,
            'acceptable': 10.0,
            'threshold': 15.0
        }
    }
```

### 2.3 자동 판단 로직

```python
def evaluate_and_decide(self, generated_views, reference_view, camera_params):
    """
    Returns:
        decision: 'ready' | 'finetune_mvdiff' | 'finetune_both' | 'reprocess'
        metrics: dict of all computed metrics
        report: human-readable analysis
    """
    metrics = self.compute_all_metrics(generated_views, reference_view, camera_params)

    # Decision Tree
    critical_failures = []
    warnings = []

    # Check View Consistency (Critical)
    if metrics['lpips_cross_view'] > self.THRESHOLDS['lpips_cross_view']['threshold']:
        critical_failures.append('view_consistency')

    # Check Reference Fidelity
    if metrics['ref_psnr'] < self.THRESHOLDS['ref_psnr']['threshold']:
        critical_failures.append('reference_fidelity')

    # Check Mask Quality
    if metrics['mask_iou'] < self.THRESHOLDS['mask_iou']['threshold']:
        warnings.append('mask_quality')

    # Decision
    if len(critical_failures) >= 2:
        decision = 'finetune_both'  # MVDiffusion + GS-LRM 모두 fine-tune
    elif len(critical_failures) == 1:
        decision = 'finetune_mvdiff'  # MVDiffusion만 fine-tune
    elif len(warnings) > 0:
        decision = 'ready_with_caution'  # 사용 가능하지만 주의
    else:
        decision = 'ready'  # Zero-shot 사용 가능

    return decision, metrics, self.generate_report(metrics, decision)
```

### 2.4 배치 평가 스크립트

```bash
# 사용 예시
python scripts/evaluate_mvdiff_quality.py \
    --input_dir outputs/new_environment_test/ \
    --reference_idx 0 \
    --camera_params configs/camera_intrinsics.json \
    --output_report reports/quality_eval_new_env.json \
    --visualize
```

---

## 3. Gaussian Mouse Avatar 모듈 아키텍처

### 3.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Gaussian Mouse Avatar Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  MAMMAL Body    │───▶│  Gaussian       │───▶│  Differentiable │         │
│  │  Model (LBS)    │    │  Initializer    │    │  Renderer       │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                      │                      │                   │
│           ▼                      ▼                      ▼                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  Pose/Shape     │    │  Appearance     │    │  Camera         │         │
│  │  Parameters     │    │  Optimization   │    │  Controller     │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 모듈 구성

```
src/
├── mouse_avatar/
│   ├── __init__.py
│   ├── body_model.py          # MAMMAL body model wrapper
│   ├── gaussian_avatar.py     # Main avatar class
│   ├── appearance.py          # Texture/appearance optimization
│   ├── renderer.py            # Differentiable Gaussian renderer
│   └── camera.py              # Camera parameter controller
├── data_generation/
│   ├── __init__.py
│   ├── camera_sampler.py      # 카메라 위치/각도 샘플링
│   ├── pose_sampler.py        # 포즈 다양화
│   ├── synthetic_dataset.py   # 합성 데이터셋 생성기
│   └── facelift_formatter.py  # FaceLift 형식 변환
└── evaluation/
    ├── __init__.py
    ├── quality_metrics.py     # 품질 메트릭 계산
    └── auto_decision.py       # Fine-tune 필요성 자동 판단
```

---

## 4. 핵심 모듈 상세 설계

### 4.1 Body Model Wrapper (`body_model.py`)

```python
"""
MAMMAL body model을 래핑하여 Gaussian Avatar에서 사용
기반: /home/joon/dev/MAMMAL_mouse/bodymodel_th.py
"""

import torch
import torch.nn as nn
from pathlib import Path

class MouseBodyModel(nn.Module):
    """MAMMAL 기반 생쥐 body model wrapper"""

    def __init__(
        self,
        model_path: str = "assets/mouse_body_model.pkl",
        device: str = "cuda",
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # Load MAMMAL parameters
        params = self._load_model(model_path)

        # Template mesh
        self.register_buffer("v_template", params['vertices'])  # (V, 3)
        self.register_buffer("faces", params['faces'])           # (F, 3)

        # Skinning weights for LBS
        self.register_buffer("skinning_weights", params['skinning_weights'])  # (V, J)

        # Joint hierarchy
        self.register_buffer("t_pose_joints", params['t_pose_joints'])  # (J, 3)
        self.register_buffer("parents", params['parents'])               # (J,)

        # Number of joints (typically 23 for mouse)
        self.n_joints = self.t_pose_joints.shape[0]

    def forward(
        self,
        pose: torch.Tensor,      # (B, J, 3) euler angles
        shape: torch.Tensor = None,  # (B, S) shape parameters (optional)
        translation: torch.Tensor = None  # (B, 3)
    ) -> dict:
        """
        Forward kinematics with Linear Blend Skinning

        Returns:
            vertices: (B, V, 3) posed vertices
            joints: (B, J, 3) posed joint locations
            transforms: (B, J, 4, 4) joint transformation matrices
        """
        batch_size = pose.shape[0]

        # 1. Get base vertices (apply shape if provided)
        if shape is not None:
            vertices = self.v_template + self._apply_shape(shape)
        else:
            vertices = self.v_template.expand(batch_size, -1, -1)

        # 2. Compute joint transformations
        transforms = self._compute_transforms(pose)  # (B, J, 4, 4)

        # 3. Apply LBS
        posed_vertices = self._lbs(vertices, transforms)

        # 4. Apply translation
        if translation is not None:
            posed_vertices = posed_vertices + translation.unsqueeze(1)

        # 5. Get posed joint locations
        posed_joints = self._transform_joints(transforms)

        return {
            'vertices': posed_vertices,
            'joints': posed_joints,
            'transforms': transforms,
            'faces': self.faces
        }

    def _compute_transforms(self, pose: torch.Tensor) -> torch.Tensor:
        """Compute hierarchical joint transformations from euler angles"""
        # Implementation based on MAMMAL_mouse/bodymodel_th.py
        # euler_to_rot_mat + forward kinematics
        pass

    def _lbs(self, vertices: torch.Tensor, transforms: torch.Tensor) -> torch.Tensor:
        """Linear Blend Skinning"""
        # W: (V, J), T: (B, J, 4, 4), V: (B, V, 3)
        # Output: (B, V, 3)
        pass
```

### 4.2 Gaussian Avatar (`gaussian_avatar.py`)

```python
"""
Gaussian Mouse Avatar - 메시에서 3D Gaussian으로 변환 및 최적화
기반: /home/joon/dev/pose-splatter/src/modules/mesh_init/gaussian_init.py
참고: MoReMouse, Pose Splatter
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class GaussianParams:
    """3D Gaussian parameters"""
    means: torch.Tensor      # (N, 3) positions
    scales: torch.Tensor     # (N, 3) scale in xyz
    rotations: torch.Tensor  # (N, 4) quaternions
    colors: torch.Tensor     # (N, 3) RGB or (N, C) SH coefficients
    opacities: torch.Tensor  # (N, 1)

class GaussianMouseAvatar(nn.Module):
    """
    Gaussian representation of mouse avatar

    Features:
    - Mesh-guided Gaussian initialization
    - Pose-dependent deformation via LBS
    - Learnable appearance (per-vertex colors/SH)
    - Geodesic correspondence for temporal consistency (MoReMouse)
    """

    def __init__(
        self,
        body_model: MouseBodyModel,
        n_gaussians_per_vertex: int = 1,
        sh_degree: int = 0,  # 0=RGB, 1-3=spherical harmonics
        use_geodesic_embedding: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        self.body_model = body_model
        self.n_gaussians_per_vertex = n_gaussians_per_vertex
        self.device = device

        n_vertices = body_model.v_template.shape[0]
        n_gaussians = n_vertices * n_gaussians_per_vertex

        # Initialize Gaussian parameters
        self._init_gaussians(n_gaussians, sh_degree)

        # Geodesic correspondence embeddings (MoReMouse)
        if use_geodesic_embedding:
            self._init_geodesic_embeddings(n_vertices)

    def _init_gaussians(self, n_gaussians: int, sh_degree: int):
        """Initialize learnable Gaussian parameters"""

        # Scales (log space for stability)
        self.log_scales = nn.Parameter(
            torch.zeros(n_gaussians, 3) - 4.0  # exp(-4) ≈ 0.018
        )

        # Rotations (quaternions, initialized to identity)
        self.rotations = nn.Parameter(
            torch.tensor([[1., 0., 0., 0.]]).expand(n_gaussians, -1).clone()
        )

        # Colors (SH coefficients)
        if sh_degree == 0:
            self.colors = nn.Parameter(torch.rand(n_gaussians, 3))
        else:
            sh_dim = (sh_degree + 1) ** 2
            self.colors = nn.Parameter(torch.zeros(n_gaussians, sh_dim, 3))

        # Opacities (sigmoid space)
        self.raw_opacities = nn.Parameter(torch.zeros(n_gaussians, 1))

        # Local offsets from mesh vertices
        self.local_offsets = nn.Parameter(torch.zeros(n_gaussians, 3))

    def _init_geodesic_embeddings(self, n_vertices: int):
        """
        Geodesic correspondence embeddings (from MoReMouse)

        목적: 동적 영역(다리, 꼬리)에서도 시간적 일관성 유지
        방법: 각 정점에 고유한 geodesic 임베딩 할당
        """
        embedding_dim = 32
        self.geodesic_embeddings = nn.Parameter(
            torch.randn(n_vertices, embedding_dim) * 0.01
        )

        # Embedding을 Gaussian 속성에 매핑하는 MLP
        self.embedding_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # scale(3) + rotation(4) + color(3)
        )

    def forward(
        self,
        pose: torch.Tensor,
        shape: Optional[torch.Tensor] = None,
        translation: Optional[torch.Tensor] = None
    ) -> GaussianParams:
        """
        Generate posed Gaussian parameters

        Args:
            pose: (B, J, 3) joint rotations
            shape: (B, S) shape parameters
            translation: (B, 3) global translation

        Returns:
            GaussianParams for each batch element
        """
        # 1. Get posed mesh from body model
        body_output = self.body_model(pose, shape, translation)
        posed_vertices = body_output['vertices']  # (B, V, 3)
        transforms = body_output['transforms']     # (B, J, 4, 4)

        batch_size = posed_vertices.shape[0]

        # 2. Compute Gaussian means (vertices + local offsets)
        # Transform local offsets by joint transforms
        deformed_offsets = self._deform_offsets(transforms)
        means = posed_vertices + deformed_offsets

        # 3. Apply geodesic embeddings to refine parameters
        if hasattr(self, 'geodesic_embeddings'):
            param_deltas = self.embedding_mlp(self.geodesic_embeddings)
            scale_delta = param_deltas[:, :3]
            rotation_delta = param_deltas[:, 3:7]
            color_delta = param_deltas[:, 7:10]
        else:
            scale_delta = rotation_delta = color_delta = 0

        # 4. Compute final Gaussian parameters
        scales = torch.exp(self.log_scales + scale_delta)
        rotations = self._normalize_quaternions(self.rotations + rotation_delta)
        colors = torch.sigmoid(self.colors + color_delta)
        opacities = torch.sigmoid(self.raw_opacities)

        return GaussianParams(
            means=means,
            scales=scales.expand(batch_size, -1, -1),
            rotations=rotations.expand(batch_size, -1, -1),
            colors=colors.expand(batch_size, -1, -1),
            opacities=opacities.expand(batch_size, -1, -1)
        )

    def optimize_from_images(
        self,
        images: torch.Tensor,      # (N_views, H, W, 4) RGBA
        cameras: dict,              # camera parameters
        poses: torch.Tensor,        # (N_views, J, 3) if different per view
        n_iterations: int = 1000,
        lr: float = 1e-3
    ):
        """
        Optimize avatar appearance from multi-view images

        이 메서드는 실제 6-view 생쥐 데이터로 텍스처/외관을 최적화
        """
        optimizer = torch.optim.Adam([
            {'params': self.colors, 'lr': lr},
            {'params': self.log_scales, 'lr': lr * 0.1},
            {'params': self.raw_opacities, 'lr': lr},
            {'params': self.local_offsets, 'lr': lr * 0.01},
        ])

        for i in range(n_iterations):
            optimizer.zero_grad()

            total_loss = 0
            for view_idx in range(images.shape[0]):
                # Render
                gaussians = self.forward(poses[view_idx:view_idx+1])
                rendered = self.renderer(gaussians, cameras[view_idx])

                # Loss
                loss = self._compute_loss(rendered, images[view_idx])
                total_loss += loss

            total_loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Iter {i}: Loss = {total_loss.item():.4f}")
```

### 4.3 Camera Sampler (`camera_sampler.py`)

```python
"""
합성 데이터 생성을 위한 카메라 파라미터 샘플링
기반: /home/joon/dev/pose-splatter/scripts/visualization/render_32views_for_facelift.py
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class CameraConfig:
    """Camera sampling configuration"""
    # Azimuth (horizontal rotation around z-axis)
    azimuth_range: Tuple[float, float] = (0, 360)    # degrees
    azimuth_samples: int = 12                         # 30° intervals

    # Elevation (vertical angle from horizontal plane)
    elevation_range: Tuple[float, float] = (-30, 60)  # degrees
    elevation_samples: int = 4                        # 생쥐는 위에서 보는 경우가 많음

    # Distance (camera-object distance)
    distance_range: Tuple[float, float] = (0.8, 2.0)  # relative to default
    distance_samples: int = 3

    # Field of view
    fov_range: Tuple[float, float] = (30, 60)        # degrees
    fov_samples: int = 2

    @property
    def total_configurations(self) -> int:
        return (self.azimuth_samples *
                self.elevation_samples *
                self.distance_samples *
                self.fov_samples)

class CameraSampler:
    """
    다양한 카메라 배치 샘플링

    목적: GS-LRM이 다양한 카메라 환경에 일반화되도록
         충분히 다양한 viewpoint에서 합성 데이터 생성
    """

    # 기본 설정 (일반화에 충분한 다양성)
    DEFAULT_CONFIG = CameraConfig(
        azimuth_range=(0, 360),
        azimuth_samples=12,       # 30° 간격
        elevation_range=(-30, 60),
        elevation_samples=4,      # -30°, 0°, 30°, 60°
        distance_range=(0.8, 2.0),
        distance_samples=3,       # 0.8x, 1.2x, 2.0x
        fov_range=(30, 60),
        fov_samples=2
    )

    # FaceLift 호환 6-view 설정
    FACELIFT_CONFIG = CameraConfig(
        azimuth_range=(0, 360),
        azimuth_samples=6,        # 60° 간격 (FaceLift 6-view)
        elevation_range=(0, 30),
        elevation_samples=2,
        distance_range=(1.0, 1.5),
        distance_samples=2,
        fov_range=(45, 45),
        fov_samples=1
    )

    def __init__(self, config: CameraConfig = None, image_size: int = 512):
        self.config = config or self.DEFAULT_CONFIG
        self.image_size = image_size

    def sample_orbit_cameras(
        self,
        center: torch.Tensor = None,
        n_views: int = 6,
        elevation: float = 15.0,
        distance: float = 1.5
    ) -> List[dict]:
        """
        Orbit camera sampling (FaceLift 스타일)

        Args:
            center: (3,) object center
            n_views: number of views to generate
            elevation: camera elevation in degrees
            distance: camera distance from center

        Returns:
            List of camera parameter dicts
        """
        if center is None:
            center = torch.zeros(3)

        cameras = []
        for i in range(n_views):
            azimuth = (360.0 / n_views) * i

            # Spherical to Cartesian
            az_rad = np.radians(azimuth)
            el_rad = np.radians(elevation)

            x = distance * np.cos(el_rad) * np.cos(az_rad)
            y = distance * np.cos(el_rad) * np.sin(az_rad)
            z = distance * np.sin(el_rad)

            position = torch.tensor([x, y, z]) + center

            # Look at center
            forward = (center - position)
            forward = forward / forward.norm()

            # Construct view matrix
            up = torch.tensor([0., 0., 1.])
            right = torch.cross(forward, up)
            right = right / right.norm()
            up = torch.cross(right, forward)

            # Intrinsics (assuming perspective projection)
            fov = 45.0
            fx = fy = self.image_size / (2 * np.tan(np.radians(fov) / 2))
            cx = cy = self.image_size / 2

            cameras.append({
                'position': position,
                'forward': forward,
                'up': up,
                'right': right,
                'fov': fov,
                'fx': fx, 'fy': fy,
                'cx': cx, 'cy': cy,
                'width': self.image_size,
                'height': self.image_size,
                'azimuth': azimuth,
                'elevation': elevation,
                'distance': distance
            })

        return cameras

    def sample_diverse_cameras(
        self,
        center: torch.Tensor = None,
        n_cameras: int = 100,
        strategy: str = 'stratified'
    ) -> List[dict]:
        """
        다양한 카메라 배치 샘플링

        Args:
            center: object center
            n_cameras: total number of cameras
            strategy: 'grid', 'stratified', 'random'

        Returns:
            List of camera parameter dicts
        """
        if strategy == 'grid':
            return self._sample_grid(center)
        elif strategy == 'stratified':
            return self._sample_stratified(center, n_cameras)
        else:
            return self._sample_random(center, n_cameras)

    def _sample_stratified(self, center, n_cameras):
        """Stratified sampling for uniform coverage"""
        cameras = []

        # Divide sphere into regions and sample within each
        azimuths = np.linspace(
            self.config.azimuth_range[0],
            self.config.azimuth_range[1],
            self.config.azimuth_samples,
            endpoint=False
        )
        elevations = np.linspace(
            self.config.elevation_range[0],
            self.config.elevation_range[1],
            self.config.elevation_samples
        )
        distances = np.linspace(
            self.config.distance_range[0],
            self.config.distance_range[1],
            self.config.distance_samples
        )

        for az in azimuths:
            for el in elevations:
                for dist in distances:
                    # Add small random jitter for diversity
                    az_jitter = az + np.random.uniform(-5, 5)
                    el_jitter = el + np.random.uniform(-5, 5)
                    dist_jitter = dist + np.random.uniform(-0.05, 0.05)

                    cam = self._create_camera(
                        center, az_jitter, el_jitter, dist_jitter
                    )
                    cameras.append(cam)

        # Subsample if needed
        if len(cameras) > n_cameras:
            indices = np.random.choice(len(cameras), n_cameras, replace=False)
            cameras = [cameras[i] for i in indices]

        return cameras

    def _create_camera(self, center, azimuth, elevation, distance, fov=45.0):
        """Create single camera from spherical coordinates"""
        if center is None:
            center = torch.zeros(3)

        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)

        x = distance * np.cos(el_rad) * np.cos(az_rad)
        y = distance * np.cos(el_rad) * np.sin(az_rad)
        z = distance * np.sin(el_rad)

        position = torch.tensor([x, y, z], dtype=torch.float32) + center
        forward = (center - position)
        forward = forward / forward.norm()

        up = torch.tensor([0., 0., 1.])
        right = torch.cross(forward, up)
        if right.norm() < 1e-6:  # Handle degenerate case (looking straight up/down)
            up = torch.tensor([0., 1., 0.])
            right = torch.cross(forward, up)
        right = right / right.norm()
        up = torch.cross(right, forward)

        fx = fy = self.image_size / (2 * np.tan(np.radians(fov) / 2))

        return {
            'position': position,
            'forward': forward,
            'up': up,
            'right': right,
            'fov': fov,
            'fx': fx, 'fy': fy,
            'cx': self.image_size / 2,
            'cy': self.image_size / 2,
            'width': self.image_size,
            'height': self.image_size,
            'azimuth': azimuth,
            'elevation': elevation,
            'distance': distance
        }
```

### 4.4 Synthetic Dataset Generator (`synthetic_dataset.py`)

```python
"""
Gaussian Mouse Avatar로부터 합성 다중 뷰 데이터셋 생성
출력 형식: FaceLift 호환 (512x512 RGBA images)
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

class SyntheticDatasetGenerator:
    """
    합성 데이터셋 생성기

    입력: Gaussian Mouse Avatar + Camera Configurations
    출력: FaceLift 형식의 다중 뷰 데이터셋

    데이터 구조:
    synthetic_data/
    ├── sample_0000/
    │   ├── images/
    │   │   ├── 000.png  (512x512 RGBA)
    │   │   ├── 001.png
    │   │   └── ...
    │   └── meta.json
    ├── sample_0001/
    │   └── ...
    └── data_synthetic.txt  (list of all samples)
    """

    def __init__(
        self,
        avatar: 'GaussianMouseAvatar',
        camera_sampler: 'CameraSampler',
        renderer: 'GaussianRenderer',
        output_dir: str = "synthetic_data"
    ):
        self.avatar = avatar
        self.camera_sampler = camera_sampler
        self.renderer = renderer
        self.output_dir = Path(output_dir)

    def generate_dataset(
        self,
        n_samples: int = 1000,
        views_per_sample: int = 6,
        pose_variations: bool = True,
        camera_strategy: str = 'stratified',
        seed: int = 42
    ):
        """
        합성 데이터셋 생성

        Args:
            n_samples: 생성할 샘플 수
            views_per_sample: 샘플당 뷰 수 (FaceLift 호환 = 6)
            pose_variations: 포즈 다양화 여부
            camera_strategy: 카메라 샘플링 전략
            seed: 재현성을 위한 시드
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        sample_list = []

        for sample_idx in tqdm(range(n_samples), desc="Generating samples"):
            sample_dir = self.output_dir / f"sample_{sample_idx:04d}"
            sample_dir.mkdir(exist_ok=True)
            (sample_dir / "images").mkdir(exist_ok=True)

            # 1. Sample pose
            if pose_variations:
                pose = self._sample_pose()
            else:
                pose = self._get_default_pose()

            # 2. Sample cameras
            cameras = self.camera_sampler.sample_orbit_cameras(
                n_views=views_per_sample,
                elevation=np.random.uniform(0, 45),
                distance=np.random.uniform(1.0, 2.0)
            )

            # 3. Render views
            gaussians = self.avatar(pose.unsqueeze(0))

            camera_metas = []
            for view_idx, camera in enumerate(cameras):
                # Render
                rendered = self.renderer(gaussians, camera)

                # Save image
                image_path = sample_dir / "images" / f"{view_idx:03d}.png"
                self._save_rgba_image(rendered, image_path)

                camera_metas.append({
                    'view_idx': view_idx,
                    'azimuth': camera['azimuth'],
                    'elevation': camera['elevation'],
                    'distance': camera['distance'],
                    'fov': camera['fov']
                })

            # 4. Save metadata
            meta = {
                'sample_idx': sample_idx,
                'pose': pose.tolist(),
                'n_views': views_per_sample,
                'cameras': camera_metas
            }
            with open(sample_dir / "meta.json", 'w') as f:
                json.dump(meta, f, indent=2)

            sample_list.append(str(sample_dir))

        # Save sample list
        with open(self.output_dir / "data_synthetic.txt", 'w') as f:
            f.write('\n'.join(sample_list))

        print(f"Generated {n_samples} samples in {self.output_dir}")

        # Generate statistics
        self._generate_statistics()

    def _sample_pose(self):
        """Sample random but realistic mouse pose"""
        n_joints = self.avatar.body_model.n_joints

        # Base pose (slight variations)
        pose = torch.zeros(n_joints, 3)

        # Add realistic joint angle variations
        # Spine joints (0-4): small rotations
        pose[0:5] = torch.randn(5, 3) * 0.1

        # Limb joints (5-16): larger range of motion
        pose[5:17] = torch.randn(12, 3) * 0.3

        # Tail joints (17-22): whip-like motion
        for i in range(17, min(23, n_joints)):
            pose[i] = pose[i-1] * 0.8 + torch.randn(3) * 0.2

        return pose

    def _get_default_pose(self):
        """Get T-pose or resting pose"""
        n_joints = self.avatar.body_model.n_joints
        return torch.zeros(n_joints, 3)

    def _save_rgba_image(self, rendered: torch.Tensor, path: Path):
        """Save rendered tensor as RGBA PNG"""
        # rendered: (H, W, 4) or (4, H, W)
        if rendered.shape[0] == 4:
            rendered = rendered.permute(1, 2, 0)

        # Clamp and convert to uint8
        rendered = (rendered.clamp(0, 1) * 255).byte().cpu().numpy()

        Image.fromarray(rendered, 'RGBA').save(path)

    def _generate_statistics(self):
        """Generate dataset statistics report"""
        report = {
            'total_samples': len(list(self.output_dir.glob("sample_*"))),
            'views_per_sample': 6,
            'camera_coverage': self._compute_camera_coverage(),
            'pose_diversity': self._compute_pose_diversity()
        }

        with open(self.output_dir / "statistics.json", 'w') as f:
            json.dump(report, f, indent=2)

    def _compute_camera_coverage(self):
        """Compute statistics on camera angle coverage"""
        azimuths = []
        elevations = []
        distances = []

        for meta_file in self.output_dir.glob("*/meta.json"):
            with open(meta_file) as f:
                meta = json.load(f)
            for cam in meta['cameras']:
                azimuths.append(cam['azimuth'])
                elevations.append(cam['elevation'])
                distances.append(cam['distance'])

        return {
            'azimuth': {'min': min(azimuths), 'max': max(azimuths), 'mean': np.mean(azimuths)},
            'elevation': {'min': min(elevations), 'max': max(elevations), 'mean': np.mean(elevations)},
            'distance': {'min': min(distances), 'max': max(distances), 'mean': np.mean(distances)}
        }

    def _compute_pose_diversity(self):
        """Compute pose variation statistics"""
        # Implementation: analyze pose parameter distributions
        return {'status': 'computed'}
```

---

## 5. 합성 데이터 생성 계획

### 5.1 카메라 파라미터 범위

| 파라미터 | 범위 | 샘플 수 | 이유 |
|---------|------|--------|------|
| **Azimuth** | 0° - 360° | 12 (30° 간격) | 전체 회전 커버 |
| **Elevation** | -30° - 60° | 4 | 아래/위 다양한 각도 |
| **Distance** | 0.8x - 2.0x | 3 | 근거리/원거리 대응 |
| **FOV** | 30° - 60° | 2 | 다양한 렌즈 특성 |

**총 조합**: 12 × 4 × 3 × 2 = **288 카메라 구성**

### 5.2 포즈 다양화 전략

```python
POSE_VARIATION_STRATEGY = {
    # 정적 포즈 (reference)
    'static': {
        'n_samples': 100,
        'pose_noise': 0.0
    },

    # 일반적인 행동 포즈
    'natural': {
        'n_samples': 500,
        'behaviors': ['walking', 'grooming', 'exploring', 'resting'],
        'pose_noise': 0.2
    },

    # 극단적 포즈 (robustness)
    'extreme': {
        'n_samples': 200,
        'behaviors': ['stretching', 'rearing', 'turning'],
        'pose_noise': 0.4
    },

    # 연속 동작 시퀀스
    'sequence': {
        'n_samples': 200,
        'sequence_length': 10,
        'temporal_smoothness': 0.8
    }
}
```

### 5.3 예상 데이터셋 크기

| 항목 | 수량 | 계산 |
|------|------|------|
| 샘플 수 | 1,000 | 100 + 500 + 200 + 200 |
| 샘플당 뷰 | 6 | FaceLift 호환 |
| 총 이미지 | 6,000 | 1,000 × 6 |
| 이미지 크기 | 512×512×4 | RGBA |
| 디스크 용량 | ~6 GB | 1MB/image × 6,000 |
| 생성 시간 | ~2-3 시간 | GPU 렌더링 기준 |

---

## 6. 통합 파이프라인

### 6.1 전체 워크플로우

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Full Pipeline Workflow                                │
└─────────────────────────────────────────────────────────────────────────────┘

Phase 1: Avatar Construction (1회)
──────────────────────────────────
[실제 6-view 데이터] → [MAMMAL Body Model] → [Gaussian Avatar] → [Appearance 최적화]
      ↓                      ↓                     ↓                    ↓
  data_mouse_          mesh + skinning      initialized         optimized avatar
  pixel_based/                              gaussians           (텍스처 학습됨)


Phase 2: Synthetic Data Generation (1회)
────────────────────────────────────────
[Optimized Avatar] → [Camera Sampler] → [Gaussian Renderer] → [FaceLift Format]
       ↓                   ↓                    ↓                    ↓
   pose variations    288 camera configs    6000 images       synthetic_data/


Phase 3: GS-LRM Fine-tuning (선택적)
───────────────────────────────────
[Real Data] + [Synthetic Data] → [GS-LRM] → [Generalized Model]
                                               ↓
                                    다양한 카메라 환경 대응


Phase 4: New Environment Inference
──────────────────────────────────
[New Camera Setup] → [Quality Evaluator] → [Decision]
                            ↓
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
         Ready       Finetune MVDiff   Finetune Both
```

### 6.2 실행 스크립트

```bash
# 1. Avatar 구축 및 외관 최적화
python scripts/build_mouse_avatar.py \
    --body_model assets/mouse_body_model.pkl \
    --train_data data_mouse_pixel_based/ \
    --output_avatar checkpoints/mouse_avatar.pt \
    --optimize_iterations 5000

# 2. 합성 데이터 생성
python scripts/generate_synthetic_data.py \
    --avatar checkpoints/mouse_avatar.pt \
    --n_samples 1000 \
    --views_per_sample 6 \
    --camera_strategy stratified \
    --output_dir synthetic_data/

# 3. GS-LRM Fine-tuning (옵션)
python train_gslrm.py \
    --config configs/mouse_gslrm_with_synthetic.yaml \
    --real_data data_mouse_pixel_based/ \
    --synthetic_data synthetic_data/ \
    --synthetic_ratio 0.5

# 4. 새 환경 품질 평가
python scripts/evaluate_mvdiff_quality.py \
    --input outputs/new_environment_test/ \
    --output_report reports/quality_new_env.json
```

---

## 7. 예상 결과 및 한계

### 7.1 예상 성능 향상

| 시나리오 | Before (현재) | After (합성 데이터) |
|---------|---------------|---------------------|
| 같은 카메라, 다른 생쥐 | 80-90% | 90-95% |
| 같은 카메라, 다른 종 | 60-80% | 75-90% |
| **다른 카메라 배치** | 20-40% | **60-80%** |
| 다른 뷰 수 (4뷰) | 30-50% | 50-70% |

### 7.2 한계점

1. **Avatar 품질 의존성**: 합성 데이터 품질은 avatar 최적화 품질에 의존
2. **Domain Gap**: 합성 이미지 vs 실제 이미지 차이 존재
3. **포즈 분포**: 학습 데이터에 없는 극단적 포즈는 여전히 어려움
4. **텍스처 일반화**: 다른 모색(털 색깔)의 생쥐는 추가 학습 필요

### 7.3 Domain Gap 완화 전략

```python
# 합성 데이터에 적용할 augmentation
DOMAIN_RANDOMIZATION = {
    'lighting': {
        'intensity': (0.5, 1.5),
        'color_temp': (4000, 7000),  # Kelvin
        'shadows': True
    },
    'noise': {
        'gaussian': (0, 0.03),
        'salt_pepper': 0.01
    },
    'blur': {
        'motion': (0, 2),
        'defocus': (0, 1)
    },
    'color': {
        'brightness': (0.8, 1.2),
        'contrast': (0.9, 1.1),
        'saturation': (0.8, 1.2)
    }
}
```

---

## 8. 구현 우선순위 및 일정

### Phase 1: 기반 모듈 (우선)
- [ ] `MouseBodyModel` - MAMMAL wrapper
- [ ] `GaussianMouseAvatar` - 기본 구조
- [ ] `CameraSampler` - 카메라 샘플링

### Phase 2: 렌더링 & 최적화
- [ ] Gaussian Renderer 통합
- [ ] Avatar appearance 최적화
- [ ] 품질 검증

### Phase 3: 데이터 생성
- [ ] `SyntheticDatasetGenerator`
- [ ] Domain randomization
- [ ] FaceLift 형식 변환

### Phase 4: 통합 & 검증
- [ ] GS-LRM fine-tuning 실험
- [ ] 품질 평가 자동화
- [ ] 다양한 환경 테스트

---

## 9. 참고 코드 경로

| 모듈 | 참고 코드 | 위치 |
|------|----------|------|
| Body Model | MAMMAL | `/home/joon/dev/MAMMAL_mouse/bodymodel_th.py` |
| Gaussian Init | pose-splatter | `/home/joon/dev/pose-splatter/src/modules/mesh_init/gaussian_init.py` |
| Multi-view Render | pose-splatter | `/home/joon/dev/pose-splatter/scripts/visualization/render_32views_for_facelift.py` |
| FaceLift Format | FaceLift | `/home/joon/dev/FaceLift/mvdiffusion/data/` |

---

*Generated: 2025-12-20*
*Author: Claude Code (AI-assisted)*
