#!/usr/bin/env python3
"""
End-to-End Single Image to 3D Pipeline

Complete pipeline: Single Image → MVDiffusion → 6 Views → LGM → 3D Gaussian

Usage:
    python scripts/inference_single_to_3d.py \
        --input_image path/to/mouse_image.png \
        --output_dir outputs/3d_results \
        --mvdiffusion_ckpt checkpoints/mvdiffusion/mouse/best.pt \
        --lgm_ckpt checkpoints/lgm/mouse_6view/best.pt
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import imageio
import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "LGM"))

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_mvdiffusion_model(checkpoint_path, device="cuda"):
    """Load MVDiffusion model for single → multi-view generation."""
    from gslrm.model.module.mvdiffusion import MVDiffusionModule
    
    # Load config from checkpoint or use default
    model = MVDiffusionModule.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip",
        torch_dtype=torch.float16,
    )
    
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading MVDiffusion checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    return model


def load_lgm_model(checkpoint_path, device="cuda", num_views=6):
    """Load LGM 6-view model for multi-view → 3D."""
    from core.options import Options
    from core.models_6view import LGM6View
    
    opt = Options(
        input_size=256,
        up_channels=(1024, 1024, 512, 256, 128),
        up_attention=(True, True, True, False, False),
        splat_size=128,
        output_size=512,
        num_views=num_views,
        num_input_views=num_views,
        fovy=49.1,
        znear=0.5,
        zfar=2.5,
        cam_radius=1.5,
    )
    
    model = LGM6View(opt, num_input_views=num_views)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading LGM checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    else:
        # Try loading pretrained
        from safetensors.torch import load_file
        pretrained_path = "LGM/pretrained/model_fp16.safetensors"
        if Path(pretrained_path).exists():
            print(f"Loading pretrained LGM from {pretrained_path}")
            ckpt = load_file(pretrained_path, device='cpu')
            current_state = model.state_dict()
            for k, v in ckpt.items():
                if k in current_state and current_state[k].shape == v.shape:
                    current_state[k].copy_(v)
    
    model = model.half().to(device)
    model.eval()
    return model, opt


def preprocess_image(image_path, size=512):
    """Load and preprocess input image."""
    img = Image.open(image_path)
    
    # Handle RGBA
    if img.mode == 'RGBA':
        img_np = np.array(img).astype(np.float32) / 255.0
        rgb = img_np[..., :3]
        alpha = img_np[..., 3:4]
        rgb = rgb * alpha + (1 - alpha)  # White background
        img = Image.fromarray((rgb * 255).astype(np.uint8))
    else:
        img = img.convert('RGB')
    
    # Resize
    img = img.resize((size, size), Image.LANCZOS)
    
    return img


def generate_multiviews(mvdiffusion_model, input_image, device="cuda", num_views=6):
    """
    Generate multi-view images from single input using MVDiffusion.
    
    Returns: list of 6 PIL Images at 60° intervals
    """
    # This is a placeholder - actual implementation depends on your MVDiffusion setup
    # For now, return the input image repeated (for testing)
    
    print("[NOTE] Using placeholder MVDiffusion - replace with actual inference")
    return [input_image] * num_views


def prepare_lgm_input(images, input_size=256, device="cuda", opt=None):
    """
    Prepare 6-view images for LGM input with ray embeddings.
    
    Args:
        images: list of 6 PIL Images
        
    Returns:
        input_tensor: [1, 6, 9, H, W] tensor
    """
    from core.utils import get_rays
    from kiui.cam import orbit_camera
    
    # Convert images to tensors
    image_tensors = []
    for img in images:
        img = img.resize((input_size, input_size), Image.LANCZOS)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()  # [3, H, W]
        image_tensors.append(img_t)
    
    images_input = torch.stack(image_tensors, dim=0)  # [6, 3, H, W]
    
    # Normalize
    images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    
    # Build ray embeddings for 6 views at 60° intervals
    view_angles = [0, 60, 120, 180, 240, 300]
    cam_radius = opt.cam_radius if opt else 1.5
    fovy = opt.fovy if opt else 49.1
    
    cam_poses = []
    for azi in view_angles:
        pose = orbit_camera(0, azi, radius=cam_radius)
        cam_poses.append(torch.from_numpy(pose))
    cam_poses = torch.stack(cam_poses, dim=0)  # [6, 4, 4]
    
    # Normalize camera poses
    transform = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, cam_radius],
        [0, 0, 0, 1]
    ], dtype=torch.float32) @ torch.inverse(cam_poses[0])
    cam_poses = transform.unsqueeze(0) @ cam_poses
    
    # Compute ray embeddings
    rays_embeddings = []
    for i in range(6):
        rays_o, rays_d = get_rays(cam_poses[i], input_size, input_size, fovy)
        rays_plucker = torch.cat([
            torch.cross(rays_o, rays_d, dim=-1),
            rays_d
        ], dim=-1)  # [H, W, 6]
        rays_embeddings.append(rays_plucker.permute(2, 0, 1))  # [6, H, W]
    
    rays_embeddings = torch.stack(rays_embeddings, dim=0)  # [6, 6, H, W]
    
    # Combine: [6, 9, H, W]
    input_tensor = torch.cat([images_input, rays_embeddings], dim=1)
    
    # Add batch dim and move to device
    input_tensor = input_tensor.unsqueeze(0).half().to(device)
    
    return input_tensor


def render_360_video(model, opt, gaussians, output_path, device="cuda", num_frames=120, fps=30):
    """Render 360° rotation video."""
    from kiui.cam import orbit_camera
    
    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1

    images = []
    azimuth_step = 360 / num_frames

    for i in tqdm.tqdm(range(num_frames), desc="Rendering 360°"):
        azi = i * azimuth_step

        cam_poses = torch.from_numpy(
            orbit_camera(0, azi, radius=opt.cam_radius, opengl=True)
        ).unsqueeze(0).to(device)

        cam_poses[:, :3, 1:3] *= -1

        cam_view = torch.inverse(cam_poses).transpose(1, 2)
        cam_view_proj = cam_view @ proj_matrix
        cam_pos = - cam_poses[:, :3, 3]

        with torch.no_grad():
            image = model.gs.render(
                gaussians,
                cam_view.unsqueeze(0),
                cam_view_proj.unsqueeze(0),
                cam_pos.unsqueeze(0),
                scale_modifier=1
            )['image']

        img_np = (image.squeeze(1).permute(0, 2, 3, 1).float().cpu().numpy() * 255).astype(np.uint8)
        images.append(img_np)

    images = np.concatenate(images, axis=0)
    imageio.mimwrite(output_path, images, fps=fps)
    print(f"Saved video: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Single Image to 3D Pipeline")
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/single_to_3d")
    parser.add_argument("--mvdiffusion_ckpt", type=str, default=None)
    parser.add_argument("--lgm_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_mvdiffusion", action="store_true", 
                        help="Skip MVDiffusion, use input as view 0")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    name = Path(args.input_image).stem
    
    print("="*60)
    print("Single Image to 3D Pipeline")
    print("="*60)
    print(f"Input: {args.input_image}")
    print(f"Output: {output_dir}")
    
    # Step 1: Load input image
    print("\n[Step 1] Loading input image...")
    input_image = preprocess_image(args.input_image)
    input_image.save(output_dir / f"{name}_input.png")
    
    # Step 2: Generate multi-views (or skip)
    print("\n[Step 2] Generating multi-view images...")
    if args.skip_mvdiffusion:
        print("  Skipping MVDiffusion - using input image for all views")
        multiview_images = [input_image] * 6
    else:
        # TODO: Implement actual MVDiffusion inference
        print("  [TODO] MVDiffusion inference not implemented yet")
        print("  Using placeholder (input image repeated)")
        multiview_images = [input_image] * 6
    
    # Save multi-view images
    for i, img in enumerate(multiview_images):
        img.save(output_dir / f"{name}_view_{i:02d}.png")
    
    # Step 3: Load LGM model
    print("\n[Step 3] Loading LGM 6-view model...")
    lgm_model, opt = load_lgm_model(args.lgm_ckpt, args.device)
    
    # Step 4: Prepare input and run LGM
    print("\n[Step 4] Running LGM inference...")
    input_tensor = prepare_lgm_input(multiview_images, opt.input_size, args.device, opt)
    print(f"  Input shape: {input_tensor.shape}")
    
    with torch.no_grad():
        gaussians = lgm_model.forward_gaussians(input_tensor)
    
    print(f"  Generated {gaussians.shape[1]:,} Gaussians")
    
    # Step 5: Save 3D Gaussian
    print("\n[Step 5] Saving 3D Gaussian...")
    ply_path = output_dir / f"{name}.ply"
    lgm_model.gs.save_ply(gaussians, str(ply_path))
    print(f"  Saved: {ply_path}")
    
    # Step 6: Render video
    print("\n[Step 6] Rendering 360° video...")
    video_path = output_dir / f"{name}.mp4"
    render_360_video(lgm_model, opt, gaussians, str(video_path), args.device)
    
    print("\n" + "="*60)
    print(f"Done! Results saved to {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
