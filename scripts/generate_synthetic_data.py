#!/usr/bin/env python3
"""
Generate synthetic multi-view data using finetuned MVDiffusion model.

This script takes real mouse images and generates synthetic 6-view images
for training GS-LRM. The MVDiffusion model is loaded from a finetuned checkpoint.

Usage:
    python scripts/generate_synthetic_data.py \
        --input_dir data_mouse_centered \
        --output_dir data_mouse_synthetic \
        --mvdiff_checkpoint checkpoints/mvdiffusion/mouse_centered_real/checkpoint-2000 \
        --num_samples 100
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image
from einops import rearrange
import math

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline


# ============================================================================
# FaceLift 표준 카메라 설정
# ============================================================================
# CRITICAL: MVDiffusion이 생성한 이미지는 이 가상 카메라 설정을 기반으로 함
# - 거리: 2.7 (GS-LRM pretrained 기준)
# - 6개 뷰: 60도 간격 azimuth (0°, 60°, 120°, 180°, 240°, 300°)
# - elevation: 20도 고정
# - FOV: ~50도 (fx=fy=548.99)
# ============================================================================

def generate_facelift_standard_cameras(
    sample_id: str,
    num_views: int = 6,
    camera_distance: float = 2.7,
    elevation_deg: float = 20.0,
    image_size: int = 512
) -> dict:
    """
    Generate FaceLift standard camera configuration.

    This MUST be used for synthetic data generation because MVDiffusion
    generates images assuming these specific camera positions.

    Args:
        sample_id: Sample identifier
        num_views: Number of views (default: 6)
        camera_distance: Distance from origin (default: 2.7, FaceLift standard)
        elevation_deg: Elevation angle in degrees (default: 20)
        image_size: Image size (default: 512)

    Returns:
        Camera configuration dictionary in FaceLift format
    """
    # FaceLift 표준 intrinsics (FOV ~50 degrees)
    fov_deg = 50.0
    fov_rad = math.radians(fov_deg)
    fx = fy = image_size / (2 * math.tan(fov_rad / 2))
    cx = cy = image_size / 2.0

    elevation_rad = math.radians(elevation_deg)

    frames = []
    for i in range(num_views):
        # Azimuth: 균등 분포 (0°, 60°, 120°, 180°, 240°, 300°)
        azimuth_deg = i * (360.0 / num_views)
        azimuth_rad = math.radians(azimuth_deg)

        # Spherical to Cartesian (Z-up coordinate system)
        # Camera position looking at origin
        x = camera_distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        y = camera_distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        z = camera_distance * math.sin(elevation_rad)

        cam_pos = np.array([x, y, z])

        # Camera orientation: look at origin
        forward = -cam_pos / np.linalg.norm(cam_pos)  # pointing to origin
        up = np.array([0, 0, 1])  # Z-up
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Construct c2w (camera-to-world)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = right
        c2w[:3, 1] = -up  # OpenGL convention: -Y is up in camera space
        c2w[:3, 2] = -forward  # OpenGL convention: -Z is forward
        c2w[:3, 3] = cam_pos

        # Invert to get w2c (world-to-camera)
        w2c = np.linalg.inv(c2w)

        frame = {
            "w": image_size,
            "h": image_size,
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "w2c": w2c.tolist(),
            "file_path": f"images/cam_{i:03d}.png",
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
            "camera_distance": camera_distance
        }
        frames.append(frame)

    return {
        "id": sample_id,
        "camera_type": "facelift_standard_6view",
        "camera_distance": camera_distance,
        "elevation_deg": elevation_deg,
        "frames": frames
    }


def load_mvdiffusion_pipeline(
    checkpoint_path: str,
    base_model_path: str,
    prompt_embed_path: str,
    device: torch.device
):
    """Load MVDiffusion pipeline with finetuned UNet weights.

    Args:
        checkpoint_path: Path to finetuned checkpoint (e.g., checkpoint-2000)
        base_model_path: Path to base model (pipeckpts)
        prompt_embed_path: Path to prompt embeddings
        device: CUDA device

    Returns:
        pipeline, generator, prompt_embeddings
    """
    print(f"Loading MVDiffusion from: {base_model_path}")
    print(f"Loading finetuned UNet from: {checkpoint_path}")

    # Load base pipeline
    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
    )

    # Load finetuned UNet weights
    unet_path = os.path.join(checkpoint_path, "unet")
    if os.path.exists(unet_path):
        print(f"Loading finetuned UNet from: {unet_path}")

        # Check for safetensors or bin format
        safetensors_path = os.path.join(unet_path, "diffusion_pytorch_model.safetensors")
        bin_path = os.path.join(unet_path, "diffusion_pytorch_model.bin")

        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            unet_state_dict = load_file(safetensors_path)
            print("Loading from safetensors format")
        elif os.path.exists(bin_path):
            unet_state_dict = torch.load(bin_path, map_location="cpu")
            print("Loading from bin format")
        else:
            raise FileNotFoundError(f"No UNet weights found in {unet_path}")

        pipeline.unet.load_state_dict(unet_state_dict, strict=False)
        print("Finetuned UNet weights loaded successfully")
    else:
        # Try EMA weights
        unet_ema_path = os.path.join(checkpoint_path, "unet_ema")
        if os.path.exists(unet_ema_path):
            print(f"Loading EMA UNet from: {unet_ema_path}")

            safetensors_path = os.path.join(unet_ema_path, "diffusion_pytorch_model.safetensors")
            bin_path = os.path.join(unet_ema_path, "diffusion_pytorch_model.bin")

            if os.path.exists(safetensors_path):
                from safetensors.torch import load_file
                unet_state_dict = load_file(safetensors_path)
            elif os.path.exists(bin_path):
                unet_state_dict = torch.load(bin_path, map_location="cpu")
            else:
                raise FileNotFoundError(f"No UNet weights found in {unet_ema_path}")

            pipeline.unet.load_state_dict(unet_state_dict, strict=False)
            print("EMA UNet weights loaded successfully")
        else:
            print("Warning: No finetuned UNet found, using base model")

    pipeline.unet.enable_xformers_memory_efficient_attention()
    pipeline.to(device)

    # Load prompt embeddings
    print(f"Loading prompt embeddings from: {prompt_embed_path}")
    prompt_embeddings = torch.load(prompt_embed_path)

    generator = torch.Generator(device=device)

    return pipeline, generator, prompt_embeddings


def generate_multiview(
    pipeline,
    generator,
    prompt_embeddings,
    input_image: Image.Image,
    guidance_scale: float = 3.0,
    num_steps: int = 50,
    seed: int = 42,
) -> list:
    """Generate 6-view images from single input image.

    Args:
        pipeline: MVDiffusion pipeline
        generator: Random generator
        prompt_embeddings: Prompt embeddings
        input_image: Input image (512x512 RGBA)
        guidance_scale: CFG scale
        num_steps: Diffusion steps
        seed: Random seed

    Returns:
        List of 6 PIL images (views 0-5)
    """
    generator.manual_seed(seed)

    # Generate multi-view images
    mv_imgs = pipeline(
        input_image,
        None,
        prompt_embeds=prompt_embeddings,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
        num_inference_steps=num_steps,
        generator=generator,
        eta=1.0,
    ).images

    # Extract 6 views
    if len(mv_imgs) == 7:
        views = [mv_imgs[i] for i in [1, 2, 3, 4, 5, 6]]
    elif len(mv_imgs) == 6:
        views = [mv_imgs[i] for i in [0, 1, 2, 3, 4, 5]]
    else:
        raise ValueError(f"Unexpected number of views: {len(mv_imgs)}")

    return views


def process_sample(
    sample_dir: Path,
    output_dir: Path,
    pipeline,
    generator,
    prompt_embeddings,
    guidance_scale: float,
    num_steps: int,
    seed: int,
) -> bool:
    """Process a single sample and generate synthetic views.

    Args:
        sample_dir: Path to sample directory (contains images/cam_*.png)
        output_dir: Output directory for this sample
        pipeline: MVDiffusion pipeline
        generator: Random generator
        prompt_embeddings: Prompt embeddings
        guidance_scale: CFG scale
        num_steps: Diffusion steps
        seed: Random seed

    Returns:
        True if successful, False otherwise
    """
    # Find reference image (cam_000.png)
    images_dir = sample_dir / "images"
    ref_image_path = images_dir / "cam_000.png"

    if not ref_image_path.exists():
        print(f"Warning: No reference image found at {ref_image_path}")
        return False

    # Load reference image
    input_image = Image.open(ref_image_path)

    # Convert to RGB if needed (white background for alpha)
    if input_image.mode == "RGBA":
        # Create white background and composite
        background = Image.new("RGB", input_image.size, (255, 255, 255))
        background.paste(input_image, mask=input_image.split()[3])  # Use alpha as mask
        input_image = background
    elif input_image.mode != "RGB":
        input_image = input_image.convert("RGB")

    # Ensure 512x512
    if input_image.size != (512, 512):
        input_image = input_image.resize((512, 512), Image.LANCZOS)

    # Generate 6 views
    try:
        views = generate_multiview(
            pipeline, generator, prompt_embeddings,
            input_image, guidance_scale, num_steps, seed
        )
    except Exception as e:
        print(f"Error generating views for {sample_dir.name}: {e}")
        return False

    # Create output directory
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Save generated views
    for i, view in enumerate(views):
        view_path = output_images_dir / f"cam_{i:03d}.png"
        view.save(view_path)

    # CRITICAL: 원본 카메라 복사 (FaceLift 표준 카메라 생성 금지!)
    # ============================================================
    # MVDiffusion은 원본 마우스 카메라 배치로 학습되었으므로,
    # 생성된 이미지도 원본 카메라 배치에 맞춰져 있음.
    # FaceLift 표준 카메라(이상적 구형 배치)를 사용하면 이미지-카메라 불일치 발생
    # -> GS-LRM이 잘못된 Plucker ray를 계산 -> white prediction 유발
    # ============================================================
    cameras_src = sample_dir / "opencv_cameras.json"
    cameras_dst = output_dir / "opencv_cameras.json"

    if cameras_src.exists():
        # 원본 카메라 복사 (sample_id만 업데이트)
        with open(cameras_src, 'r') as f:
            camera_data = json.load(f)
        camera_data["id"] = output_dir.name
        with open(cameras_dst, 'w') as f:
            json.dump(camera_data, f, indent=4)
    else:
        print(f"Warning: No camera file found at {cameras_src}, skipping")

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic multi-view data")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory with real data (FaceLift format)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for synthetic data")
    parser.add_argument("--mvdiff_checkpoint", type=str, required=True,
                        help="Path to finetuned MVDiffusion checkpoint")
    parser.add_argument("--base_model_path", type=str,
                        default="checkpoints/mvdiffusion/pipeckpts",
                        help="Path to base MVDiffusion model")
    parser.add_argument("--prompt_embed_path", type=str,
                        default="mvdiffusion/data/mouse_prompt_embeds_6view_real/clr_embeds.pt",
                        help="Path to prompt embeddings")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to process (None = all)")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                        help="Guidance scale for diffusion")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Ratio of samples for training (rest for validation)")

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Find all samples
    sample_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name.startswith("sample_")
    ])

    if args.num_samples is not None:
        sample_dirs = sample_dirs[:args.num_samples]

    print(f"Found {len(sample_dirs)} samples to process")

    # Load pipeline
    pipeline, generator, prompt_embeddings = load_mvdiffusion_pipeline(
        args.mvdiff_checkpoint,
        args.base_model_path,
        args.prompt_embed_path,
        device
    )

    # Process samples
    successful = []
    failed = []

    for sample_dir in tqdm(sample_dirs, desc="Generating synthetic data"):
        sample_name = sample_dir.name
        sample_output_dir = output_dir / sample_name

        success = process_sample(
            sample_dir, sample_output_dir,
            pipeline, generator, prompt_embeddings,
            args.guidance_scale, args.num_steps, args.seed
        )

        if success:
            successful.append(sample_name)
        else:
            failed.append(sample_name)

    # Create train/val split
    np.random.seed(args.seed)
    np.random.shuffle(successful)

    n_train = int(len(successful) * args.train_ratio)
    train_samples = successful[:n_train]
    val_samples = successful[n_train:]

    # Write train list
    train_list_path = output_dir / "data_train.txt"
    with open(train_list_path, "w") as f:
        for sample in sorted(train_samples):
            f.write(f"{output_dir.name}/{sample}\n")

    # Write val list
    val_list_path = output_dir / "data_val.txt"
    with open(val_list_path, "w") as f:
        for sample in sorted(val_samples):
            f.write(f"{output_dir.name}/{sample}\n")

    print(f"\n=== Synthetic Data Generation Complete ===")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Train list: {train_list_path}")
    print(f"Val list: {val_list_path}")

    if failed:
        print(f"\nFailed samples: {failed[:10]}{'...' if len(failed) > 10 else ''}")


if __name__ == "__main__":
    main()
