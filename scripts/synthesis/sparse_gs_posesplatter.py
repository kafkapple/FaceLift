"""
Sparse View Gaussian Splatting using PoseSplatter components.

6개 마우스 뷰에서 빠르게 3DGS/2DGS 최적화 후 32개 Novel View 렌더링.
PoseSplatter의 검증된 렌더러와 shape carving을 활용.

Usage:
    python scripts/synthesis/sparse_gs_posesplatter.py \
        --input_dir /path/to/mouse_6view_scene \
        --output_dir /path/to/output_32view \
        --mode 3d  # or 2d_gsplat

Requirements:
    - PoseSplatter 설치 (~/dev/pose-splatter)
    - gsplat 설치

Author: AI-assisted (Claude)
Date: 2024-12-17
"""

import sys
import os

# PoseSplatter 경로 추가
POSESPLATTER_PATH = os.path.expanduser("~/dev/pose-splatter")
if POSESPLATTER_PATH not in sys.path:
    sys.path.insert(0, POSESPLATTER_PATH)

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict

# PoseSplatter imports
try:
    from src.gaussian_renderer import create_renderer, GaussianRenderer
    from src.shape_carver import get_volume_torch, project_points_torch
    from src.shape_carving import create_3d_grid
    POSESPLATTER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PoseSplatter import failed: {e}")
    print(f"Make sure PoseSplatter is installed at {POSESPLATTER_PATH}")
    POSESPLATTER_AVAILABLE = False


class SparseGaussianModel(nn.Module):
    """
    Sparse View Gaussian Splatting 모델.

    PoseSplatter의 렌더러를 활용하여 6개 뷰에서 Gaussian 최적화.
    """

    def __init__(
        self,
        num_gaussians: int = 20000,
        device: str = "cuda",
        mode: str = "3d",  # "3d" or "2d_gsplat"
    ):
        super().__init__()
        self.device = device
        self.mode = mode
        self.num_gaussians = num_gaussians

        # Gaussian 파라미터 (14개/Gaussian)
        # [N, 14]: means(3) + log_scales(3) + quats(4) + colors(3) + opacity(1)
        self.gaussian_params = None

    def init_from_pointcloud(self, points: np.ndarray, colors: Optional[np.ndarray] = None):
        """
        Point cloud에서 Gaussian 초기화.

        Args:
            points: [N, 3] 3D 점 좌표
            colors: [N, 3] RGB 색상 (optional, 0-1 범위)
        """
        N = len(points)
        self.num_gaussians = N

        # 14개 파라미터로 초기화
        params = torch.zeros(N, 14, device=self.device, dtype=torch.float32)

        # Means (위치)
        params[:, 0:3] = torch.tensor(points, device=self.device, dtype=torch.float32)

        # Log scales (작은 초기값)
        params[:, 3:6] = -5.0  # exp(-5) ≈ 0.007

        # Quaternions (identity rotation)
        params[:, 6] = 1.0  # w
        params[:, 7:10] = 0.0  # x, y, z

        # Colors
        if colors is not None:
            params[:, 10:13] = torch.tensor(colors, device=self.device, dtype=torch.float32)
        else:
            params[:, 10:13] = 0.5  # 회색

        # Opacity (logit, sigmoid(0) = 0.5)
        params[:, 13] = 0.0

        self.gaussian_params = nn.Parameter(params)

    def init_random(self, num_points: int, spatial_extent: float = 0.1):
        """랜덤 초기화 (fallback)"""
        N = num_points
        self.num_gaussians = N

        params = torch.zeros(N, 14, device=self.device, dtype=torch.float32)

        # Random positions
        params[:, 0:3] = torch.randn(N, 3, device=self.device) * spatial_extent

        # Small scales
        params[:, 3:6] = -5.0

        # Identity quaternions
        params[:, 6] = 1.0

        # Random colors
        params[:, 10:13] = torch.rand(N, 3, device=self.device)

        # Mid opacity
        params[:, 13] = 0.0

        self.gaussian_params = nn.Parameter(params)

    def forward(self):
        """현재 Gaussian 파라미터 반환"""
        return self.gaussian_params


class SparseGSOptimizer:
    """
    6개 Sparse 뷰에서 Gaussian 최적화.

    PoseSplatter의 렌더러를 사용하여 최적화 수행.
    """

    def __init__(
        self,
        config: Dict,
        device: str = "cuda",
    ):
        self.config = config
        self.device = device

        # 설정
        self.iterations = config.get("iterations", 1000)
        self.lr = config.get("lr", 0.01)
        self.mode = config.get("mode", "3d")  # "3d" or "2d_gsplat"
        self.num_init_points = config.get("num_init_points", 20000)

        self.model = None
        self.renderer = None

    def _create_renderer(self, width: int, height: int):
        """PoseSplatter 렌더러 생성"""
        if not POSESPLATTER_AVAILABLE:
            raise RuntimeError("PoseSplatter not available")

        self.renderer = create_renderer(
            mode=self.mode,
            width=width,
            height=height,
            device=self.device,
        )
        # 흰색 배경
        self.renderer.set_background_color(torch.ones(3, device=self.device))

    def _init_from_visual_hull(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        Ks: np.ndarray,
        c2ws: np.ndarray,
        grid_size: int = 64,
        ell: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visual Hull을 사용하여 초기 포인트 클라우드 생성.

        Args:
            images: [V, H, W, 3] 이미지 (0-1 범위)
            masks: [V, H, W] 마스크 (0 or 1)
            Ks: [V, 3, 3] intrinsics
            c2ws: [V, 4, 4] camera-to-world

        Returns:
            points: [N, 3] 초기 포인트
            colors: [N, 3] 초기 색상
        """
        V, H, W, _ = images.shape

        # c2w → w2c (extrinsics)
        w2cs = np.linalg.inv(c2ws)

        # 3D 그리드 생성
        # volume_idx: 그리드 범위
        volume_idx = [[-ell/2, ell/2], [-ell/2, ell/2], [-ell/2, ell/2]]
        grid = create_3d_grid(ell, grid_size, volume_idx=volume_idx)
        grid = torch.tensor(grid, device=self.device, dtype=torch.float32)

        # 마스크를 이미지로 변환 (shape carving용)
        masks_img = torch.tensor(masks, device=self.device, dtype=torch.float32)
        masks_img = masks_img.unsqueeze(-1).repeat(1, 1, 1, 3)  # [V, H, W, 3]
        masks_img = masks_img.permute(0, 3, 1, 2)  # [V, 3, H, W]

        Ks_t = torch.tensor(Ks, device=self.device, dtype=torch.float32)
        w2cs_t = torch.tensor(w2cs, device=self.device, dtype=torch.float32)

        # Visual hull: 모든 뷰에서 마스크 내에 있는 voxel 찾기
        volume = get_volume_torch(masks_img, Ks_t, w2cs_t, grid)

        # 임계값 적용 (모든 뷰에서 보이는 voxel)
        occupancy = (volume[0] > 0.5).float()  # [n1, n2, n3]

        # Occupied voxel 좌표 추출
        occupied_idx = torch.nonzero(occupancy, as_tuple=False)  # [N, 3]

        if len(occupied_idx) == 0:
            print("Warning: No occupied voxels found, using random init")
            return None, None

        # Grid index → 3D 좌표
        n1, n2, n3 = grid.shape[:3]
        points = grid[occupied_idx[:, 0], occupied_idx[:, 1], occupied_idx[:, 2]]
        points = points.cpu().numpy()

        # 색상: 이미지에서 샘플링
        images_t = torch.tensor(images, device=self.device, dtype=torch.float32)
        images_t = images_t.permute(0, 3, 1, 2)  # [V, 3, H, W]

        color_volume = get_volume_torch(images_t, Ks_t, w2cs_t, grid)
        colors = color_volume[:, occupied_idx[:, 0], occupied_idx[:, 1], occupied_idx[:, 2]]
        colors = colors.permute(1, 0).cpu().numpy()  # [N, 3]

        # 포인트 수 제한
        if len(points) > self.num_init_points:
            idx = np.random.choice(len(points), self.num_init_points, replace=False)
            points = points[idx]
            colors = colors[idx]

        print(f"Visual hull: {len(points)} points")
        return points, colors

    def optimize(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        c2ws: np.ndarray,
        Ks: np.ndarray,
    ) -> SparseGaussianModel:
        """
        6개 뷰에서 Gaussian 최적화.

        Args:
            images: [V, H, W, 3] 이미지 (0-1 범위)
            masks: [V, H, W] 마스크
            c2ws: [V, 4, 4] camera-to-world
            Ks: [V, 3, 3] intrinsics

        Returns:
            최적화된 SparseGaussianModel
        """
        V, H, W, _ = images.shape

        # 렌더러 생성
        self._create_renderer(W, H)

        # Visual Hull로 초기화
        points, colors = self._init_from_visual_hull(images, masks, Ks, c2ws)

        # 모델 생성 및 초기화
        self.model = SparseGaussianModel(device=self.device, mode=self.mode)

        if points is not None:
            self.model.init_from_pointcloud(points, colors)
        else:
            self.model.init_random(self.num_init_points)

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # 텐서 변환
        images_t = torch.tensor(images, device=self.device, dtype=torch.float32)
        c2ws_t = torch.tensor(c2ws, device=self.device, dtype=torch.float32)
        Ks_t = torch.tensor(Ks, device=self.device, dtype=torch.float32)

        # w2c 계산
        w2cs_t = torch.linalg.inv(c2ws_t)

        # 최적화 루프
        pbar = tqdm(range(self.iterations), desc="Sparse GS Optimization")
        for step in pbar:
            optimizer.zero_grad()

            total_loss = 0.0
            gaussian_params = self.model()

            # 각 뷰에서 렌더링 및 loss 계산
            for v in range(V):
                rgb, alpha = self.renderer.render(
                    gaussian_params,
                    w2cs_t[v],  # viewmat
                    Ks_t[v],    # K
                )

                # L1 Loss
                target = images_t[v]
                l1_loss = torch.abs(rgb - target).mean()

                total_loss += l1_loss

            total_loss /= V
            total_loss.backward()
            optimizer.step()

            if step % 100 == 0:
                pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

        return self.model

    def render_novel_view(
        self,
        c2w: np.ndarray,
        K: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        새로운 카메라에서 렌더링.

        Args:
            c2w: [4, 4] camera-to-world
            K: [3, 3] intrinsics
            width, height: 이미지 크기

        Returns:
            rgb: [H, W, 3] 렌더링 이미지 (0-1 범위)
        """
        if self.model is None:
            raise RuntimeError("Model not optimized yet")

        # 렌더러 크기 확인
        if self.renderer.width != width or self.renderer.height != height:
            self._create_renderer(width, height)

        w2c = np.linalg.inv(c2w)

        w2c_t = torch.tensor(w2c, device=self.device, dtype=torch.float32)
        K_t = torch.tensor(K, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            gaussian_params = self.model()
            rgb, alpha = self.renderer.render(gaussian_params, w2c_t, K_t)

        return rgb.cpu().numpy()


def generate_camera_poses(
    num_views: int = 32,
    elevation: float = 20.0,
    distance: float = 2.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FaceLift 스타일 카메라 배치 생성.

    Args:
        num_views: 뷰 수
        elevation: elevation 각도 (도)
        distance: 카메라 거리

    Returns:
        c2ws: [N, 4, 4] camera-to-world
        azimuths: [N] azimuth 각도
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
        center = np.array([0, 0, 0])
        up = np.array([0, 0, 1])

        # Look-at matrix
        z_axis = center - eye
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(z_axis, up)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = x_axis
        c2w[:3, 1] = y_axis
        c2w[:3, 2] = z_axis
        c2w[:3, 3] = eye

        c2ws.append(c2w)
        azimuths.append(azimuth)

    return np.stack(c2ws, axis=0), np.array(azimuths)


def load_scene(input_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Scene 데이터 로드 (FaceLift 형식).

    Returns:
        images: [V, H, W, 3]
        masks: [V, H, W]
        c2ws: [V, 4, 4]
        Ks: [V, 3, 3]
    """
    camera_path = os.path.join(input_dir, "opencv_cameras.json")

    with open(camera_path, "r") as f:
        camera_data = json.load(f)

    frames = camera_data["frames"][:6]  # 6개 뷰만

    images = []
    c2ws = []
    Ks = []

    for frame in frames:
        # 이미지 로드
        img_path = os.path.join(input_dir, frame["file_path"])
        img = np.array(Image.open(img_path).convert("RGB")) / 255.0
        images.append(img)

        # 카메라 파라미터
        w2c = np.array(frame["w2c"])
        c2w = np.linalg.inv(w2c)
        c2ws.append(c2w)

        fx, fy = frame["fx"], frame["fy"]
        cx, cy = frame["cx"], frame["cy"]
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        Ks.append(K)

    images = np.stack(images, axis=0).astype(np.float32)
    c2ws = np.stack(c2ws, axis=0).astype(np.float32)
    Ks = np.stack(Ks, axis=0).astype(np.float32)

    # 마스크: 흰색(1.0) 배경 제거
    masks = (images.mean(axis=-1) < 0.95).astype(np.float32)

    return images, masks, c2ws, Ks


def save_scene(
    output_dir: str,
    rgb_images: List[np.ndarray],
    c2ws: np.ndarray,
    K: np.ndarray,
    azimuths: np.ndarray,
):
    """32뷰 scene 저장 (FaceLift 형식)"""
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    frames = []
    H, W = rgb_images[0].shape[:2]

    for i, (rgb, c2w, azimuth) in enumerate(zip(rgb_images, c2ws, azimuths)):
        # 이미지 저장
        img_path = f"images/cam_{i:03d}.png"
        Image.fromarray((rgb * 255).astype(np.uint8)).save(
            os.path.join(output_dir, img_path)
        )

        # 카메라 정보
        w2c = np.linalg.inv(c2w)
        frames.append({
            "w": W, "h": H,
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "w2c": w2c.tolist(),
            "file_path": img_path,
            "azimuth": float(azimuth),
        })

    # JSON 저장
    with open(os.path.join(output_dir, "opencv_cameras.json"), "w") as f:
        json.dump({"frames": frames}, f, indent=2)

    print(f"Saved {len(frames)} views to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sparse GS with PoseSplatter")
    parser.add_argument("--input_dir", required=True, help="6-view input directory")
    parser.add_argument("--output_dir", required=True, help="32-view output directory")
    parser.add_argument("--mode", choices=["3d", "2d_gsplat"], default="3d")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--num_views", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    if not POSESPLATTER_AVAILABLE:
        print("Error: PoseSplatter not available")
        sys.exit(1)

    # 데이터 로드
    print(f"Loading scene from {args.input_dir}")
    images, masks, c2ws, Ks = load_scene(args.input_dir)
    print(f"Loaded {len(images)} views, shape: {images.shape}")

    # 최적화
    config = {
        "iterations": args.iterations,
        "lr": args.lr,
        "mode": args.mode,
        "num_init_points": 20000,
    }

    optimizer = SparseGSOptimizer(config)
    model = optimizer.optimize(images, masks, c2ws, Ks)

    # Novel view 렌더링
    print(f"Rendering {args.num_views} novel views...")
    novel_c2ws, azimuths = generate_camera_poses(
        num_views=args.num_views,
        elevation=20,
        distance=2.7,
    )

    H, W = images.shape[1:3]
    K = Ks[0]  # 동일한 intrinsics 사용

    rgb_images = []
    for c2w in tqdm(novel_c2ws, desc="Rendering"):
        rgb = optimizer.render_novel_view(c2w, K, W, H)
        rgb_images.append(rgb)

    # 저장
    save_scene(args.output_dir, rgb_images, novel_c2ws, K, azimuths)
    print("Done!")


if __name__ == "__main__":
    main()
