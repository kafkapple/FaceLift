#!/usr/bin/env python3
"""
LGM 6-View Inference Script

Takes 6 pre-generated views and runs LGM to produce 3D Gaussian + 360° video.

Usage:
    python scripts/infer_lgm_6view.py \
        --input_dir data_mouse/sample_000000/images \
        --checkpoint checkpoints/lgm/mouse_6view_v7/best.pt \
        --output_dir outputs/lgm_6view_test
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "LGM"))

from kiui.cam import orbit_camera
from core.options import Options
from core.models_6view import LGM6View
from core.utils import get_rays

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_model(checkpoint_path, device="cuda"):
    opt = Options(
        input_size=256,
        up_channels=(1024, 1024, 512, 256, 128),
        up_attention=(True, True, True, False, False),
        splat_size=128,
        output_size=512,
        batch_size=1,
        num_views=6,
        num_input_views=6,
        fovy=49.1,
        znear=0.5,
        zfar=2.5,
        cam_radius=1.5,
    )
    
    model = LGM6View(opt, num_input_views=6)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt, strict=False)
    print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
    
    model = model.half().to(device)
    model.eval()
    return model, opt


def load_6views(input_dir, input_size=256):
    input_dir = Path(input_dir)
    images = []
    
    for i in range(6):
        img_path = None
        for pattern in [f"cam_{i:03d}.png", f"cam_{i:02d}.png", f"view_{i:02d}.png", f"{i:03d}.png"]:
            p = input_dir / pattern
            if p.exists():
                img_path = p
                break
        
        if img_path is None:
            raise FileNotFoundError(f"View {i} not found in {input_dir}")
        
        img = Image.open(img_path)
        
        if img.mode == 'RGBA':
            img_np = np.array(img).astype(np.float32) / 255.0
            rgb = img_np[..., :3]
            alpha = img_np[..., 3:4]
            rgb = rgb * alpha + (1 - alpha)  # White background
        else:
            rgb = np.array(img.convert('RGB')).astype(np.float32) / 255.0
        
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
        images.append(rgb)
    
    images = torch.stack(images, dim=0)  # [6, 3, H, W]
    images = F.interpolate(images, size=(input_size, input_size), mode='bilinear', align_corners=False)
    images = TF.normalize(images, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    
    return images


def prepare_cameras_and_input(images, opt):
    """Prepare cameras and input tensor with ray embeddings."""
    view_angles = [0, 60, 120, 180, 240, 300]
    
    # Build camera poses
    cam_poses = []
    for azi in view_angles:
        c2w = orbit_camera(0, azi, radius=opt.cam_radius, opengl=True)
        c2w = torch.from_numpy(c2w).float()
        cam_poses.append(c2w)
    cam_poses = torch.stack(cam_poses, dim=0)  # [6, 4, 4]
    
    # Normalize - make first camera canonical
    transform = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, opt.cam_radius],
        [0, 0, 0, 1]
    ], dtype=torch.float32) @ torch.inverse(cam_poses[0])
    cam_poses = transform.unsqueeze(0) @ cam_poses  # [1, 6, 4, 4]
    cam_poses = cam_poses.squeeze(0)  # Back to [6, 4, 4]
    
    # Build ray embeddings
    rays_embeddings = []
    for i in range(6):
        rays_o, rays_d = get_rays(cam_poses[i], opt.input_size, opt.input_size, opt.fovy)
        rays_plucker = torch.cat([
            torch.cross(rays_o, rays_d, dim=-1),
            rays_d
        ], dim=-1)  # [H, W, 6]
        rays_embeddings.append(rays_plucker.permute(2, 0, 1))  # [6, H, W]
    
    rays_embeddings = torch.stack(rays_embeddings, dim=0)  # [6, 6, H, W]
    
    # Combine: [6, 9, H, W]
    input_tensor = torch.cat([images, rays_embeddings], dim=1)
    
    return input_tensor.unsqueeze(0), cam_poses  # [1, 6, 9, H, W], [6, 4, 4]


def render_360_video(model, opt, gaussians, output_path, device, num_frames=120, fps=30):
    """Render 360° rotation video."""
    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1
    
    frames = []
    
    for i in tqdm(range(num_frames), desc="Rendering 360"):
        azi = i * (360 / num_frames)
        
        c2w = orbit_camera(0, azi, radius=opt.cam_radius, opengl=True)
        c2w = torch.from_numpy(c2w).float().unsqueeze(0).to(device)
        c2w[:, :3, 1:3] *= -1  # Invert for rendering
        
        cam_view = torch.inverse(c2w).transpose(1, 2)
        cam_view_proj = cam_view @ proj_matrix
        cam_pos = -c2w[:, :3, 3]
        
        with torch.no_grad():
            rendered = model.gs.render(
                gaussians,
                cam_view.unsqueeze(0),
                cam_view_proj.unsqueeze(0),
                cam_pos.unsqueeze(0),
                scale_modifier=1.0
            )
        
        img = rendered['image'].squeeze(1)  # [1, 3, H, W]
        img = img.permute(0, 2, 3, 1).cpu().numpy()[0]  # [H, W, 3]
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        frames.append(img)
    
    imageio.mimwrite(output_path, frames, fps=fps)
    print(f"[INFO] Saved video: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lgm/mouse_6view_v7/best.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/lgm_6view_test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    
    print("=" * 60)
    print("LGM 6-View Inference")
    print("=" * 60)
    
    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.input_dir).parent.name
    
    # Step 1: Load model
    print("\n[Step 1] Loading model...")
    model, opt = load_model(args.checkpoint, args.device)
    
    # Step 2: Load 6 views
    print("\n[Step 2] Loading 6 views...")
    images = load_6views(args.input_dir, opt.input_size)
    print(f"  Loaded: {images.shape}")
    
    # Step 3: Prepare input
    print("\n[Step 3] Preparing input...")
    input_tensor, cam_poses = prepare_cameras_and_input(images, opt)
    input_tensor = input_tensor.half().to(args.device)
    print(f"  Input shape: {input_tensor.shape}")
    
    # Step 4: Run inference
    print("\n[Step 4] Running LGM inference...")
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            gaussians = model.forward_gaussians(input_tensor)
    print(f"  Generated {gaussians.shape[1]} Gaussians")
    
    # Step 5: Save PLY
    print("\n[Step 5] Saving 3D Gaussian...")
    ply_path = output_dir / f"{name}.ply"
    model.gs.save_ply(gaussians, str(ply_path))
    print(f"  Saved: {ply_path}")
    
    # Step 6: Render 360° video
    print("\n[Step 6] Rendering 360° video...")
    video_path = output_dir / f"{name}_360.mp4"
    render_360_video(model, opt, gaussians, str(video_path), args.device, 
                     num_frames=args.num_frames, fps=args.fps)
    
    print("\n" + "=" * 60)
    print(f"Done! Results: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
