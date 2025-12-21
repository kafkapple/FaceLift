# Mouse 32뷰 합성 데이터 생성 구현 계획

## 1. 개요

**목표**: 6개 실제 마우스 뷰에서 32개 뷰 합성 데이터셋 생성

**핵심 기술**: Sparse-view 3DGS/2DGS 최적화 후 Novel View 렌더링

---

## 2. 기술 선택

### 2.1 라이브러리 비교

| 라이브러리 | 장점 | 단점 | Sparse View 지원 |
|------------|------|------|------------------|
| **[gsplat](https://github.com/nerfstudio-project/gsplat)** | CUDA 최적화, Nerfstudio 통합 | 기본 3DGS만 | 커스텀 필요 |
| **[2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting)** | 기하학 정확도 높음, 공식 구현 | 설치 복잡 | O (표면 기반) |
| **[InstantSplat](https://instantsplat.github.io/)** | 3DGS/2DGS 모두 지원, pose-free | 복잡한 의존성 | ◎ (설계 목적) |
| **[FreeSplatter](https://github.com/TencentARC/FreeSplatter)** | Feed-forward, 2DGS fine-tuned | 모델 크기 큼 | ◎ (feed-forward) |
| **[MVSGaussian](https://github.com/TQTQliu/MVSGaussian)** | Single forward pass | 특정 데이터셋 | O |

### 2.2 권장 선택

```
1순위: InstantSplat
  - Sparse view에 특화된 설계
  - 3DGS/2DGS 모두 지원
  - Self-supervised camera optimization

2순위: 2d-gaussian-splatting (공식)
  - 기하학적 정확도 최고
  - 마우스 표면 재구성에 적합

3순위: gsplat + 커스텀 정규화
  - 가장 유연한 구현
  - Depth regularization 직접 추가
```

---

## 3. Phase 1: 환경 구축 (Day 1)

### 3.1 기본 의존성 설치

```bash
# gpu05 서버에서 실행
ssh gpu05

# 새 conda 환경 생성
conda create -n gs_synthesis python=3.10 -y
conda activate gs_synthesis

# PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 기본 라이브러리
pip install numpy scipy pillow tqdm einops
pip install opencv-python imageio
```

### 3.2 gsplat 설치

```bash
# gsplat (CUDA 가속)
pip install gsplat

# 테스트
python -c "import gsplat; print(gsplat.__version__)"
```

### 3.3 2DGS 설치 (옵션)

```bash
# 2D Gaussian Splatting 공식 저장소
cd ~/dev
git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive
cd 2d-gaussian-splatting

# submodules 설치
pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn
```

### 3.4 InstantSplat 설치 (권장)

```bash
# InstantSplat
cd ~/dev
git clone https://github.com/NVlabs/InstantSplat.git
cd InstantSplat
pip install -r requirements.txt
```

---

## 4. Phase 2: Sparse 3DGS 구현 (Day 2-3)

### 4.1 디렉토리 구조 생성

```bash
cd ~/dev/FaceLift
mkdir -p scripts/synthesis
touch scripts/synthesis/__init__.py
```

### 4.2 핵심 모듈 구현

#### 4.2.1 설정 파일

```yaml
# scripts/synthesis/config.yaml
sparse_3dgs:
  # Gaussian 파라미터
  num_init_points: 5000      # 초기 포인트 수
  max_gaussians: 50000       # 최대 Gaussian 수

  # 최적화 설정
  iterations: 1500
  densify_until: 1000
  densify_interval: 100

  # Learning rates
  lr_position: 0.0016
  lr_feature: 0.0025
  lr_opacity: 0.05
  lr_scaling: 0.005
  lr_rotation: 0.001

  # Sparse view 정규화
  depth_weight: 0.5          # Depth consistency loss
  normal_weight: 0.1         # Normal smoothness loss
  tv_weight: 0.01            # Total variation

sparse_2dgs:
  # 2DGS 특수 설정
  num_init_points: 10000
  iterations: 2000

  # 표면 정규화
  depth_distortion_weight: 100
  normal_consistency_weight: 0.1

rendering:
  num_views: 32              # 생성할 뷰 수
  elevation: 20              # 카메라 elevation (도)
  distance: 2.7              # 카메라 거리
  image_size: 512

output:
  base_dir: "data_mouse_synthetic"
```

#### 4.2.2 카메라 유틸리티

```python
# scripts/synthesis/camera_utils.py
"""카메라 파라미터 생성 및 변환 유틸리티"""

import numpy as np
import torch


def look_at(eye, center, up):
    """
    Look-at 행렬 생성 (OpenCV 좌표계).

    Args:
        eye: 카메라 위치 [3]
        center: 타겟 위치 [3]
        up: 업 벡터 [3]

    Returns:
        c2w: Camera-to-world 행렬 [4, 4]
    """
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    # Z axis (forward, looking from eye to center)
    z = center - eye
    z = z / np.linalg.norm(z)

    # X axis (right)
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)

    # Y axis (down in OpenCV)
    y = np.cross(z, x)

    # Camera-to-world matrix
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = eye

    return c2w


def generate_camera_poses(
    num_views=32,
    elevation=20,
    distance=2.7,
    center=(0, 0, 0)
):
    """
    원형 배치로 카메라 poses 생성 (FaceLift 스타일).

    Args:
        num_views: 생성할 뷰 수
        elevation: 카메라 elevation 각도 (도)
        distance: 원점에서 카메라까지 거리
        center: 타겟 중심점

    Returns:
        c2ws: [num_views, 4, 4] Camera-to-world 행렬들
        azimuths: [num_views] 각 뷰의 azimuth 각도
    """
    c2ws = []
    azimuths = []

    elev_rad = np.radians(elevation)

    for i in range(num_views):
        azimuth = i * (360 / num_views)
        azim_rad = np.radians(azimuth)

        # Spherical to Cartesian
        x = distance * np.cos(elev_rad) * np.sin(azim_rad)
        y = distance * np.cos(elev_rad) * np.cos(azim_rad)
        z = distance * np.sin(elev_rad)

        eye = np.array([x, y, z])
        c2w = look_at(eye, center=center, up=[0, 0, 1])

        c2ws.append(c2w)
        azimuths.append(azimuth)

    return np.stack(c2ws, axis=0), np.array(azimuths)


def get_intrinsics(image_size=512, fov=49.1):
    """
    카메라 내부 파라미터 생성.

    Args:
        image_size: 이미지 크기
        fov: Field of view (도)

    Returns:
        intrinsics: [fx, fy, cx, cy]
    """
    focal = image_size / (2 * np.tan(np.radians(fov) / 2))
    cx = cy = image_size / 2
    return np.array([focal, focal, cx, cy], dtype=np.float32)
```

#### 4.2.3 Sparse 3DGS 최적화

```python
# scripts/synthesis/sparse_3dgs.py
"""
Sparse View 3D Gaussian Splatting 최적화.

6개 뷰에서 빠르게 3DGS를 최적화하여 대략적인 형태를 재구성.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image

try:
    import gsplat
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("Warning: gsplat not installed. Using fallback.")


class GaussianModel:
    """3D Gaussian Splatting 모델"""

    def __init__(self, num_points=5000, sh_degree=0, device="cuda"):
        self.device = device
        self.sh_degree = sh_degree

        # Gaussian 파라미터 (초기화)
        self.xyz = None           # [N, 3] 위치
        self.features = None      # [N, C] 색상 특성
        self.opacity = None       # [N, 1] 불투명도
        self.scaling = None       # [N, 3] 스케일
        self.rotation = None      # [N, 4] 회전 (quaternion)

    def init_from_random(self, num_points, spatial_extent=1.0):
        """랜덤 초기화"""
        self.xyz = torch.randn(num_points, 3, device=self.device) * spatial_extent * 0.5
        self.features = torch.rand(num_points, 3, device=self.device)
        self.opacity = torch.ones(num_points, 1, device=self.device) * 0.5
        self.scaling = torch.ones(num_points, 3, device=self.device) * 0.01
        self.rotation = torch.zeros(num_points, 4, device=self.device)
        self.rotation[:, 0] = 1.0  # Identity quaternion

        # Requires grad
        self.xyz.requires_grad_(True)
        self.features.requires_grad_(True)
        self.opacity.requires_grad_(True)
        self.scaling.requires_grad_(True)
        self.rotation.requires_grad_(True)

    def init_from_pointcloud(self, points, colors=None):
        """Point cloud에서 초기화"""
        self.xyz = torch.tensor(points, device=self.device, dtype=torch.float32)

        if colors is not None:
            self.features = torch.tensor(colors, device=self.device, dtype=torch.float32)
        else:
            self.features = torch.ones(len(points), 3, device=self.device) * 0.5

        self.opacity = torch.ones(len(points), 1, device=self.device) * 0.5
        self.scaling = torch.ones(len(points), 3, device=self.device) * 0.01
        self.rotation = torch.zeros(len(points), 4, device=self.device)
        self.rotation[:, 0] = 1.0

        self.xyz.requires_grad_(True)
        self.features.requires_grad_(True)
        self.opacity.requires_grad_(True)
        self.scaling.requires_grad_(True)
        self.rotation.requires_grad_(True)

    def get_params(self):
        """최적화 파라미터 반환"""
        return [self.xyz, self.features, self.opacity, self.scaling, self.rotation]

    @property
    def num_gaussians(self):
        return self.xyz.shape[0] if self.xyz is not None else 0


class Sparse3DGS:
    """
    6개 Sparse 뷰에서 빠른 3DGS 최적화.

    설계 원칙:
    - 적은 Gaussian 수 → 빠른 최적화
    - 짧은 iteration → 대략적 형태만 필요
    - 정규화 추가 → Sparse view artifact 감소
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 설정 로드
        self.num_init_points = config.get("num_init_points", 5000)
        self.max_gaussians = config.get("max_gaussians", 50000)
        self.iterations = config.get("iterations", 1500)

        # Learning rates
        self.lr_position = config.get("lr_position", 0.0016)
        self.lr_feature = config.get("lr_feature", 0.0025)
        self.lr_opacity = config.get("lr_opacity", 0.05)
        self.lr_scaling = config.get("lr_scaling", 0.005)
        self.lr_rotation = config.get("lr_rotation", 0.001)

        # 정규화 가중치
        self.depth_weight = config.get("depth_weight", 0.5)

        self.model = None

    def _create_optimizer(self):
        """파라미터별 learning rate로 optimizer 생성"""
        param_groups = [
            {"params": [self.model.xyz], "lr": self.lr_position, "name": "xyz"},
            {"params": [self.model.features], "lr": self.lr_feature, "name": "features"},
            {"params": [self.model.opacity], "lr": self.lr_opacity, "name": "opacity"},
            {"params": [self.model.scaling], "lr": self.lr_scaling, "name": "scaling"},
            {"params": [self.model.rotation], "lr": self.lr_rotation, "name": "rotation"},
        ]
        return torch.optim.Adam(param_groups)

    def _render(self, c2w, intrinsics, image_size):
        """
        Gaussian 렌더링.

        Args:
            c2w: [4, 4] Camera-to-world 행렬
            intrinsics: [4] fx, fy, cx, cy
            image_size: int

        Returns:
            rgb: [H, W, 3] 렌더링 이미지
            depth: [H, W] Depth map (optional)
        """
        if not GSPLAT_AVAILABLE:
            # Fallback: placeholder
            return torch.ones(image_size, image_size, 3, device=self.device) * 0.5, None

        # gsplat 렌더링 사용
        # (실제 구현은 gsplat API에 따라 조정 필요)
        w2c = torch.linalg.inv(c2w)

        # TODO: gsplat.rasterization 호출
        # rgb, alpha, info = gsplat.rasterization(...)

        raise NotImplementedError("gsplat 렌더링 구현 필요")

    def _compute_loss(self, rendered, target, depth_rendered=None, depth_target=None):
        """
        Loss 계산.

        Args:
            rendered: [H, W, 3] 렌더링 이미지
            target: [H, W, 3] GT 이미지

        Returns:
            loss: 총 loss
            loss_dict: 개별 loss 딕셔너리
        """
        # L1 Loss
        l1_loss = torch.abs(rendered - target).mean()

        # SSIM Loss (간략화)
        # TODO: 실제 SSIM 구현
        ssim_loss = torch.tensor(0.0, device=self.device)

        # Depth consistency (optional)
        depth_loss = torch.tensor(0.0, device=self.device)
        if depth_rendered is not None and depth_target is not None:
            depth_loss = torch.abs(depth_rendered - depth_target).mean()

        # 총 loss
        loss = l1_loss + 0.2 * (1 - ssim_loss) + self.depth_weight * depth_loss

        return loss, {
            "l1": l1_loss.item(),
            "ssim": ssim_loss.item(),
            "depth": depth_loss.item()
        }

    def optimize(self, images, c2ws, intrinsics):
        """
        6개 뷰에서 3DGS 최적화.

        Args:
            images: [6, H, W, 3] 입력 이미지 (numpy, 0-1)
            c2ws: [6, 4, 4] Camera-to-world 행렬 (numpy)
            intrinsics: [4] fx, fy, cx, cy (numpy)

        Returns:
            GaussianModel: 최적화된 모델
        """
        # Tensor 변환
        images = torch.tensor(images, device=self.device, dtype=torch.float32)
        c2ws = torch.tensor(c2ws, device=self.device, dtype=torch.float32)
        intrinsics = torch.tensor(intrinsics, device=self.device, dtype=torch.float32)

        num_views, H, W, _ = images.shape

        # 모델 초기화
        self.model = GaussianModel(device=self.device)
        self.model.init_from_random(self.num_init_points, spatial_extent=1.0)

        # Optimizer 생성
        optimizer = self._create_optimizer()

        # 최적화 루프
        pbar = tqdm(range(self.iterations), desc="Sparse 3DGS Optimization")
        for step in pbar:
            optimizer.zero_grad()

            total_loss = 0.0

            # 각 뷰에서 loss 계산
            for v in range(num_views):
                rendered, depth = self._render(c2ws[v], intrinsics, H)
                loss, loss_dict = self._compute_loss(rendered, images[v])
                total_loss += loss

            total_loss /= num_views
            total_loss.backward()
            optimizer.step()

            # 로깅
            if step % 100 == 0:
                pbar.set_postfix({"loss": total_loss.item()})

        return self.model

    def render_novel_view(self, c2w, intrinsics, image_size=512):
        """새로운 카메라 위치에서 렌더링"""
        c2w = torch.tensor(c2w, device=self.device, dtype=torch.float32)
        intrinsics = torch.tensor(intrinsics, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            rgb, _ = self._render(c2w, intrinsics, image_size)

        return rgb.cpu().numpy()


def test_sparse_3dgs():
    """기본 테스트"""
    config = {
        "num_init_points": 1000,
        "iterations": 100,
    }

    optimizer = Sparse3DGS(config)

    # 더미 데이터
    images = np.random.rand(6, 512, 512, 3).astype(np.float32)
    c2ws = np.stack([np.eye(4) for _ in range(6)]).astype(np.float32)
    intrinsics = np.array([512, 512, 256, 256], dtype=np.float32)

    print("Running sparse 3DGS optimization test...")
    # model = optimizer.optimize(images, c2ws, intrinsics)
    print("Test passed (skeleton only)")


if __name__ == "__main__":
    test_sparse_3dgs()
```

---

## 5. Phase 3: 2DGS 옵션 구현 (Day 4-5)

### 5.1 왜 2DGS가 Sparse View에 더 적합한가?

```
3DGS 문제점 (Sparse View):
- 3D 타원체가 빈 공간에 floater로 생성됨
- View 사이에서 inconsistency
- Depth ambiguity

2DGS 장점:
- 2D 디스크가 표면에 정렬됨
- Geometry-aware rendering
- Sparse view에서 더 안정적인 표면 재구성
```

### 5.2 Sparse2DGS 구현 개요

```python
# scripts/synthesis/sparse_2dgs.py
"""
Sparse View 2D Gaussian Splatting 최적화.

2D 디스크 기반으로 표면에 정렬된 Gaussian 사용.
기하학적 정확도가 중요한 경우 권장.
"""

class Gaussian2DModel:
    """2D Gaussian (Surface-aligned disk) 모델"""

    def __init__(self, num_points=10000, device="cuda"):
        self.device = device

        # 2DGS 파라미터
        self.xyz = None           # [N, 3] 위치
        self.features = None      # [N, C] 색상
        self.opacity = None       # [N, 1] 불투명도
        self.scaling = None       # [N, 2] 2D 스케일 (tangent plane)
        self.rotation = None      # [N, 4] 회전
        self.normal = None        # [N, 3] 표면 법선 (implicit from rotation)


class Sparse2DGS:
    """
    2DGS 기반 Sparse View 재구성.

    특징:
    - Depth distortion loss로 표면 정렬 강제
    - Normal consistency로 기하학적 정확도 향상
    - 3DGS보다 sparse view에서 안정적
    """

    def __init__(self, config):
        self.config = config

        # 2DGS 특수 설정
        self.depth_distortion_weight = config.get("depth_distortion_weight", 100)
        self.normal_consistency_weight = config.get("normal_consistency_weight", 0.1)

    def optimize(self, images, c2ws, intrinsics):
        """2DGS 최적화"""
        # 구현은 2d-gaussian-splatting 라이브러리 활용
        raise NotImplementedError("2DGS 구현 - 공식 repo 활용")
```

---

## 6. Phase 4: 데이터셋 생성 파이프라인 (Day 6-7)

### 6.1 배치 처리 스크립트

```python
# scripts/synthesis/batch_process.py
"""
전체 마우스 데이터셋에 대해 32뷰 합성 데이터 생성.

Usage:
    python scripts/synthesis/batch_process.py \
        --input_dir /path/to/mouse_6view \
        --output_dir /path/to/mouse_32view \
        --method 3dgs  # or 2dgs
"""

import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

from sparse_3dgs import Sparse3DGS
from sparse_2dgs import Sparse2DGS
from camera_utils import generate_camera_poses, get_intrinsics


def process_single_scene(
    input_dir,
    output_dir,
    method="3dgs",
    config=None
):
    """
    단일 scene 처리: 6뷰 → 32뷰.

    Args:
        input_dir: 6뷰 데이터 디렉토리 (opencv_cameras.json 포함)
        output_dir: 32뷰 출력 디렉토리
        method: "3dgs" or "2dgs"
        config: 최적화 설정
    """
    # 1. 입력 데이터 로드
    with open(os.path.join(input_dir, "opencv_cameras.json"), "r") as f:
        camera_data = json.load(f)

    frames = camera_data["frames"]
    images = []
    c2ws = []

    for frame in frames[:6]:  # 6개 뷰만 사용
        img_path = os.path.join(input_dir, frame["file_path"])
        img = np.array(Image.open(img_path).convert("RGB")) / 255.0
        images.append(img)

        w2c = np.array(frame["w2c"])
        c2w = np.linalg.inv(w2c)
        c2ws.append(c2w)

    images = np.stack(images, axis=0).astype(np.float32)
    c2ws = np.stack(c2ws, axis=0).astype(np.float32)

    intrinsics = np.array([
        frames[0]["fx"], frames[0]["fy"],
        frames[0]["cx"], frames[0]["cy"]
    ], dtype=np.float32)

    # 2. Sparse GS 최적화
    if method == "3dgs":
        optimizer = Sparse3DGS(config)
    else:
        optimizer = Sparse2DGS(config)

    model = optimizer.optimize(images, c2ws, intrinsics)

    # 3. 32개 뷰 렌더링
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    novel_c2ws, azimuths = generate_camera_poses(
        num_views=32,
        elevation=20,
        distance=2.7
    )

    output_frames = []
    for i, c2w in enumerate(tqdm(novel_c2ws, desc="Rendering")):
        # 렌더링
        rgb = optimizer.render_novel_view(c2w, intrinsics, image_size=512)

        # 저장
        img_path = f"images/cam_{i:03d}.png"
        Image.fromarray((rgb * 255).astype(np.uint8)).save(
            os.path.join(output_dir, img_path)
        )

        # 카메라 정보
        w2c = np.linalg.inv(c2w)
        output_frames.append({
            "w": 512, "h": 512,
            "fx": float(intrinsics[0]),
            "fy": float(intrinsics[1]),
            "cx": float(intrinsics[2]),
            "cy": float(intrinsics[3]),
            "w2c": w2c.tolist(),
            "file_path": img_path,
            "azimuth": float(azimuths[i])
        })

    # 4. JSON 저장
    with open(os.path.join(output_dir, "opencv_cameras.json"), "w") as f:
        json.dump({"frames": output_frames}, f, indent=2)

    print(f"Generated 32 views at {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="6-view dataset directory")
    parser.add_argument("--output_dir", required=True, help="32-view output directory")
    parser.add_argument("--method", choices=["3dgs", "2dgs"], default="3dgs")
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    config = {
        "iterations": args.iterations,
        "num_init_points": 10000,
    }

    # 모든 scene 처리
    input_scenes = sorted(Path(args.input_dir).glob("scene_*"))

    for scene_dir in tqdm(input_scenes, desc="Processing scenes"):
        scene_name = scene_dir.name
        output_scene_dir = os.path.join(args.output_dir, scene_name)

        if os.path.exists(output_scene_dir):
            print(f"Skipping {scene_name} (already exists)")
            continue

        try:
            process_single_scene(
                str(scene_dir),
                output_scene_dir,
                method=args.method,
                config=config
            )
        except Exception as e:
            print(f"Error processing {scene_name}: {e}")
            continue


if __name__ == "__main__":
    main()
```

---

## 7. 구현 체크리스트

### Week 1

- [ ] **Day 1**: 환경 구축
  - [ ] conda 환경 생성
  - [ ] gsplat 설치 및 테스트
  - [ ] 2DGS repo clone 및 설치

- [ ] **Day 2-3**: Sparse 3DGS 구현
  - [ ] camera_utils.py 완성
  - [ ] GaussianModel 클래스 완성
  - [ ] gsplat 렌더링 통합
  - [ ] 단일 scene 테스트

- [ ] **Day 4-5**: 2DGS 옵션 구현
  - [ ] 2d-gaussian-splatting 통합
  - [ ] Sparse2DGS 클래스 완성
  - [ ] 3DGS vs 2DGS 품질 비교

- [ ] **Day 6-7**: 배치 처리
  - [ ] batch_process.py 완성
  - [ ] 전체 데이터셋 처리
  - [ ] 품질 검증

### Week 2

- [ ] **Day 8-10**: GS-LRM 학습
  - [ ] Objaverse pretrained 체크포인트 확보
  - [ ] MixedDataset 구현
  - [ ] Fine-tuning 실험

- [ ] **Day 11-12**: 평가 및 문서화
  - [ ] 정량적 평가 (PSNR, SSIM)
  - [ ] 정성적 평가 (360° 비디오)
  - [ ] 보고서 작성

---

## 8. 예상 리소스

| 단계 | GPU 메모리 | 시간/scene | 총 예상 시간 |
|------|------------|-----------|--------------|
| Sparse 3DGS 최적화 | ~4GB | ~30초 | ~13시간 (1600 scenes) |
| Sparse 2DGS 최적화 | ~6GB | ~60초 | ~26시간 (1600 scenes) |
| 32뷰 렌더링 | ~2GB | ~5초 | ~2시간 |

**총 예상**: 배치 처리로 1-2일 내 완료 가능

---

## 9. 참고 자료

- [gsplat Documentation](https://docs.gsplat.studio/)
- [2D Gaussian Splatting (SIGGRAPH 2024)](https://github.com/hbb1/2d-gaussian-splatting)
- [InstantSplat](https://instantsplat.github.io/)
- [DNGaussian (CVPR 2024)](https://fictionarry.github.io/DNGaussian/) - Depth-regularized sparse view
