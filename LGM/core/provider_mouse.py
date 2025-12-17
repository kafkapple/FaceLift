"""
Mouse Dataset for LGM Fine-tuning

Strategy: Use 4 views as input, remaining 2 views for supervision.

MVDiffusion generates 6 views at 60° intervals: [0°, 60°, 120°, 180°, 240°, 300°]
LGM expects 4 views at 90° intervals: [0°, 90°, 180°, 270°]

View selection for training:
- Input 4 views: [0, 1, 3, 5] → [0°, 60°, 180°, 300°] (closest to 90° intervals)
- Supervision 2 views: [2, 4] → [120°, 240°] (held out for loss)

This allows self-supervised fine-tuning without 3D GT.
"""

import os
import json
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image

from core.options import Options
from core.utils import get_rays

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class MouseDataset(Dataset):
    """
    Mouse multi-view dataset for LGM fine-tuning.

    Data structure expected:
        data_root/
        ├── sample_000000/
        │   ├── images/
        │   │   ├── cam_000.png  (view 0: 0°)
        │   │   ├── cam_001.png  (view 1: 60°)
        │   │   ├── cam_002.png  (view 2: 120°)
        │   │   ├── cam_003.png  (view 3: 180°)
        │   │   ├── cam_004.png  (view 4: 240°)
        │   │   └── cam_005.png  (view 5: 300°)
        │   └── opencv_cameras.json
        └── data_train.txt (list of sample directories)
    """

    def __init__(
        self,
        opt: Options,
        data_root: str = "data_mouse",
        split_file: str = "data_mouse_train.txt",
        training: bool = True,
        num_input_views: int = 4,
        num_output_views: int = 6,  # Total views for supervision
        input_view_indices: list = None,  # Which views to use as input
    ):
        self.opt = opt
        self.data_root = Path(data_root)
        self.training = training
        self.num_input_views = num_input_views
        self.num_output_views = num_output_views

        # Default: use views closest to 90° intervals
        # MVDiffusion [0°, 60°, 120°, 180°, 240°, 300°]
        # Select [0, 1, 3, 5] → [0°, 60°, 180°, 300°] as input
        self.input_view_indices = input_view_indices or [0, 1, 3, 5]

        # Load sample list
        split_path = self.data_root / split_file
        if split_path.exists():
            with open(split_path, 'r') as f:
                self.items = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # Fallback: find all sample directories
            self.items = sorted([
                str(d) for d in self.data_root.iterdir()
                if d.is_dir() and d.name.startswith('sample_')
            ])

        # Train/val split
        if training:
            self.items = self.items[:-max(1, len(self.items) // 10)]
        else:
            self.items = self.items[-max(1, len(self.items) // 10):]

        print(f"[MouseDataset] Loaded {len(self.items)} samples for {'train' if training else 'val'}")
        print(f"  Input views: {self.input_view_indices}")

        # Camera parameters (fixed for mouse setup)
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1

        # Pre-compute camera poses for 6 views at 60° intervals
        self._setup_camera_poses()

    def _setup_camera_poses(self):
        """Setup camera poses for 6 views at 60° intervals around the object."""
        self.cam_poses_template = []

        for i in range(6):
            azimuth = i * 60  # 0, 60, 120, 180, 240, 300
            elevation = 0  # Horizontal ring

            # Create camera pose (looking at origin)
            c2w = self._orbit_camera(elevation, azimuth, radius=self.opt.cam_radius)
            self.cam_poses_template.append(c2w)

        self.cam_poses_template = torch.stack(self.cam_poses_template, dim=0)  # [6, 4, 4]

    def _orbit_camera(self, elevation: float, azimuth: float, radius: float = 1.5):
        """Create camera-to-world matrix for orbit camera."""
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)

        # Camera position
        x = radius * np.cos(elevation) * np.sin(azimuth)
        y = radius * np.sin(elevation)
        z = radius * np.cos(elevation) * np.cos(azimuth)

        # Camera orientation (looking at origin)
        forward = -np.array([x, y, z])
        forward = forward / np.linalg.norm(forward)

        right = np.cross(np.array([0, 1, 0]), forward)
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)

        # Build c2w matrix
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = [x, y, z]

        return torch.from_numpy(c2w)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample_path = Path(self.items[idx])
        if not sample_path.is_absolute():
            sample_path = self.data_root / sample_path

        results = {}

        # Load all 6 views
        images = []
        masks = []

        for i in range(6):
            img_path = sample_path / "images" / f"cam_{i:03d}.png"

            if not img_path.exists():
                # Try alternative naming
                img_path = sample_path / "images" / f"view_{i:02d}.png"

            if img_path.exists():
                img = Image.open(img_path)

                # Handle RGBA
                if img.mode == 'RGBA':
                    img_np = np.array(img).astype(np.float32) / 255.0
                    rgb = img_np[..., :3]
                    alpha = img_np[..., 3:4]
                    # White background
                    rgb = rgb * alpha + (1 - alpha)
                    mask = alpha[..., 0]
                else:
                    img_np = np.array(img.convert('RGB')).astype(np.float32) / 255.0
                    rgb = img_np
                    mask = np.ones(rgb.shape[:2], dtype=np.float32)

                # To tensor [C, H, W]
                rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
                mask = torch.from_numpy(mask).float()

                images.append(rgb)
                masks.append(mask)
            else:
                print(f"[WARN] Image not found: {img_path}")
                # Placeholder
                images.append(torch.zeros(3, 512, 512))
                masks.append(torch.zeros(512, 512))

        images = torch.stack(images, dim=0)  # [6, 3, H, W]
        masks = torch.stack(masks, dim=0)    # [6, H, W]

        # Get camera poses
        cam_poses = self.cam_poses_template.clone()  # [6, 4, 4]

        # Normalize camera poses (transform first pose to canonical)
        transform = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, self.opt.cam_radius],
            [0, 0, 0, 1]
        ], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [6, 4, 4]

        # Select input views
        input_indices = self.input_view_indices
        images_input = images[input_indices].clone()  # [4, 3, H, W]
        cam_poses_input = cam_poses[input_indices].clone()  # [4, 4, 4]

        # Resize input images
        images_input = F.interpolate(
            images_input,
            size=(self.opt.input_size, self.opt.input_size),
            mode='bilinear',
            align_corners=False
        )

        # Normalize input images
        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # Build ray embeddings for input views
        rays_embeddings = []
        for i in range(len(input_indices)):
            rays_o, rays_d = get_rays(
                cam_poses_input[i],
                self.opt.input_size,
                self.opt.input_size,
                self.opt.fovy
            )
            rays_plucker = torch.cat([
                torch.cross(rays_o, rays_d, dim=-1),
                rays_d
            ], dim=-1)  # [H, W, 6]
            rays_embeddings.append(rays_plucker.permute(2, 0, 1))  # [6, H, W]

        rays_embeddings = torch.stack(rays_embeddings, dim=0)  # [4, 6, H, W]

        # Combine input images with ray embeddings
        results['input'] = torch.cat([images_input, rays_embeddings], dim=1)  # [4, 9, H, W]

        # Output images for supervision (all 6 views)
        results['images_output'] = F.interpolate(
            images,
            size=(self.opt.output_size, self.opt.output_size),
            mode='bilinear',
            align_corners=False
        )  # [6, 3, H, W]

        results['masks_output'] = F.interpolate(
            masks.unsqueeze(1),
            size=(self.opt.output_size, self.opt.output_size),
            mode='bilinear',
            align_corners=False
        )  # [6, 1, H, W]

        # Camera poses for rendering
        results['cam_poses'] = cam_poses  # [6, 4, 4]
        results['cam_view'] = torch.inverse(cam_poses).transpose(1, 2)  # [6, 4, 4]
        results['cam_view_proj'] = results['cam_view'] @ self.proj_matrix  # [6, 4, 4]
        results['cam_pos'] = -cam_poses[:, :3, 3]  # [6, 3]

        return results


if __name__ == "__main__":
    # Test dataset
    from core.options import Options

    opt = Options(
        input_size=256,
        output_size=512,
        fovy=49.1,
        znear=0.5,
        zfar=2.5,
        cam_radius=1.5,
    )

    dataset = MouseDataset(
        opt,
        data_root="data_mouse",
        split_file="data_mouse_train.txt",
        training=True,
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Input shape: {sample['input'].shape}")  # [4, 9, 256, 256]
        print(f"Output images shape: {sample['images_output'].shape}")  # [6, 3, 512, 512]
        print(f"Cam poses shape: {sample['cam_poses'].shape}")  # [6, 4, 4]
