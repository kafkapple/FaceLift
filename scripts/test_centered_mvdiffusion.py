#!/usr/bin/env python3
"""
Test MVDiffusion with center-aligned mouse images.

Method 1: Pre-process images to center-align BEFORE MVDiffusion inference.
This tests whether centering the input helps generate consistent multi-view outputs.

Compares:
1. Original (not centered) input
2. Center-aligned input using bounding box detection

Output: Grid images showing 6-view MVDiffusion outputs for comparison.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel


def find_mouse_bbox(image: np.ndarray, threshold: int = 240) -> tuple:
    """
    Find bounding box of mouse using simple background subtraction.
    Assumes white background (pixel values > threshold).

    Returns:
        (x_min, y_min, x_max, y_max) or None if not found
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Binary mask: mouse is darker than background
    mask = gray < threshold

    # Find contours
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    return (x_min, y_min, x_max, y_max)


def center_align_image(
    image: np.ndarray,
    target_size: int = 512,
    target_center: tuple = (256, 256),
    background_color: tuple = (255, 255, 255),
    scale_factor: float = 1.0
) -> tuple:
    """
    Center-align object in image.

    Args:
        image: Input image (H, W, C)
        target_size: Output image size
        target_center: Target center position
        background_color: Background fill color
        scale_factor: Scale factor for the object

    Returns:
        (aligned_image, crop_params)
    """
    bbox = find_mouse_bbox(image)
    if bbox is None:
        return image, {"error": "No object found"}

    x_min, y_min, x_max, y_max = bbox

    # Calculate object center and size
    obj_center_x = (x_min + x_max) / 2
    obj_center_y = (y_min + y_max) / 2
    obj_width = x_max - x_min
    obj_height = y_max - y_min
    obj_size = max(obj_width, obj_height)

    # Calculate scale to fit object nicely (with some padding)
    # Target: object takes ~60% of image
    target_obj_size = target_size * 0.6
    scale = (target_obj_size / obj_size) * scale_factor

    # Scale image
    h, w = image.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Calculate new object center after scaling
    scaled_center_x = obj_center_x * scale
    scaled_center_y = obj_center_y * scale

    # Create output image with background
    output = np.full((target_size, target_size, 3), background_color, dtype=np.uint8)

    # Calculate offset to center object
    offset_x = int(target_center[0] - scaled_center_x)
    offset_y = int(target_center[1] - scaled_center_y)

    # Calculate paste region
    src_x1 = max(0, -offset_x)
    src_y1 = max(0, -offset_y)
    src_x2 = min(new_w, target_size - offset_x)
    src_y2 = min(new_h, target_size - offset_y)

    dst_x1 = max(0, offset_x)
    dst_y1 = max(0, offset_y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # Paste
    output[dst_y1:dst_y2, dst_x1:dst_x2] = scaled_image[src_y1:src_y2, src_x1:src_x2]

    crop_params = {
        "original_bbox": bbox,
        "scale": scale,
        "offset": (offset_x, offset_y),
        "obj_center": (obj_center_x, obj_center_y),
        "obj_size": obj_size
    }

    return output, crop_params


def load_mvdiffusion_pipeline(pretrained_path: str, unet_checkpoint: str, device: str = "cuda"):
    """Load MVDiffusion pipeline with trained UNet."""
    print(f"Loading MVDiffusion pipeline from {pretrained_path}")

    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        pretrained_path,
        torch_dtype=torch.float16,
    )

    # Load trained UNet
    unet_path = os.path.join(unet_checkpoint, "unet")
    if os.path.exists(unet_path):
        print(f"Loading trained UNet from {unet_path}")
        trained_unet = UNetMV2DConditionModel.from_pretrained(
            unet_path,
            torch_dtype=torch.float16,
        )
        pipeline.unet = trained_unet
    else:
        print(f"Warning: UNet not found at {unet_path}, using pretrained")

    pipeline.to(device)

    if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Warning: Could not enable xformers: {e}")

    return pipeline


def generate_views(
    pipeline,
    input_image: Image.Image,
    prompt_embeds: torch.Tensor,
    num_views: int = 6,
    guidance_scale: float = 3.0,
    num_steps: int = 50,
    seed: int = 42,
    device: str = "cuda"
) -> torch.Tensor:
    """Generate multi-view images from single input."""

    input_tensor = TF.to_tensor(input_image).unsqueeze(0)
    input_tensor = input_tensor.repeat(num_views, 1, 1, 1).to(device)

    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        output = pipeline(
            input_tensor,
            None,
            prompt_embeds=prompt_embeds.to(device, dtype=torch.float16),
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
            output_type='pt',
            eta=1.0,
        )

    return output.images


def create_comparison_grid(
    original_input: np.ndarray,
    centered_input: np.ndarray,
    original_outputs: list,
    centered_outputs: list,
    title: str = ""
) -> Image.Image:
    """
    Create comparison grid showing:
    Row 1: Original input + 6 generated views
    Row 2: Centered input + 6 generated views
    """
    img_size = 256  # Display size
    n_cols = 7  # 1 input + 6 views
    n_rows = 2

    padding = 10
    label_height = 30

    grid_width = n_cols * img_size + (n_cols + 1) * padding
    grid_height = n_rows * img_size + (n_rows + 1) * padding + n_rows * label_height + 50  # Extra for title

    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Title
    if title:
        draw.text((grid_width // 2 - 150, 10), title, fill=(0, 0, 0), font=title_font)

    # Labels
    labels = ["Input", "View 0", "View 1", "View 2", "View 3", "View 4", "View 5"]
    row_labels = ["Original", "Centered"]

    y_start = 50

    for row in range(n_rows):
        y = y_start + row * (img_size + padding + label_height)

        # Row label
        draw.text((5, y + img_size // 2), row_labels[row], fill=(0, 0, 0), font=font)

        for col in range(n_cols):
            x = padding + col * (img_size + padding)

            # Column label (only first row)
            if row == 0:
                draw.text((x + img_size // 2 - 20, y - label_height), labels[col], fill=(0, 0, 0), font=font)

            # Get image
            if col == 0:
                img_array = original_input if row == 0 else centered_input
            else:
                outputs = original_outputs if row == 0 else centered_outputs
                img_array = outputs[col - 1]

            # Resize and paste
            img = Image.fromarray(img_array).resize((img_size, img_size), Image.LANCZOS)
            grid.paste(img, (x, y))

            # Draw border
            draw.rectangle([x, y, x + img_size, y + img_size], outline=(200, 200, 200), width=1)

    return grid


def main():
    parser = argparse.ArgumentParser(description="Test center-aligned MVDiffusion inference")

    parser.add_argument("--mvdiff_pretrained", type=str,
                        default="checkpoints/mvdiffusion/pipeckpts")
    parser.add_argument("--mvdiff_checkpoint", type=str,
                        default="checkpoints/mvdiffusion/mouse/mouse_embeds_6x_aug/checkpoint-4000")
    parser.add_argument("--prompt_embeds", type=str,
                        default="mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt")
    parser.add_argument("--input_dir", type=str,
                        default="data_mouse",
                        help="Input data directory")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/center_align_test",
                        help="Output directory for comparison grids")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    pipeline = load_mvdiffusion_pipeline(
        args.mvdiff_pretrained,
        args.mvdiff_checkpoint,
        args.device
    )

    # Load prompt embeddings
    prompt_embeds = torch.load(args.prompt_embeds, map_location=args.device)
    prompt_embeds = prompt_embeds.to(torch.float16)
    print(f"Prompt embeds shape: {prompt_embeds.shape}")

    # Get sample list
    input_dir = Path(args.input_dir)
    data_file = input_dir / "data_mouse_train.txt"

    if data_file.exists():
        with open(data_file, 'r') as f:
            sample_paths = [line.strip() for line in f if line.strip()][:args.num_samples]
    else:
        # Find sample directories
        sample_paths = sorted([str(p) for p in input_dir.glob("sample_*")])[:args.num_samples]

    print(f"\nTesting {len(sample_paths)} samples")
    print(f"Output directory: {output_dir}\n")

    for i, sample_path in enumerate(sample_paths):
        sample_path = Path(sample_path)
        sample_name = sample_path.name

        print(f"[{i+1}/{len(sample_paths)}] Processing {sample_name}...")

        # Find input image (cam_000.png)
        images_dir = sample_path / "images"
        if not images_dir.exists():
            images_dir = sample_path

        input_path = images_dir / "cam_000.png"
        if not input_path.exists():
            print(f"  Warning: {input_path} not found, skipping")
            continue

        # Load original image
        original_img = np.array(Image.open(input_path).convert('RGB'))

        # Center-align image
        centered_img, crop_params = center_align_image(
            original_img,
            target_size=512,
            target_center=(256, 256),  # Center of 512x512
            scale_factor=1.0
        )

        print(f"  Crop params: {crop_params}")

        # Generate views - Original
        original_pil = Image.fromarray(original_img)
        original_views = generate_views(
            pipeline, original_pil, prompt_embeds,
            seed=args.seed, device=args.device
        )
        original_outputs = []
        for v in range(6):
            view_img = original_views[v].permute(1, 2, 0).cpu().numpy()
            view_img = (view_img * 255).clip(0, 255).astype(np.uint8)
            original_outputs.append(view_img)

        # Generate views - Centered
        centered_pil = Image.fromarray(centered_img)
        centered_views = generate_views(
            pipeline, centered_pil, prompt_embeds,
            seed=args.seed, device=args.device
        )
        centered_outputs = []
        for v in range(6):
            view_img = centered_views[v].permute(1, 2, 0).cpu().numpy()
            view_img = (view_img * 255).clip(0, 255).astype(np.uint8)
            centered_outputs.append(view_img)

        # Create comparison grid
        grid = create_comparison_grid(
            original_img, centered_img,
            original_outputs, centered_outputs,
            title=f"Sample: {sample_name} - Original vs Center-Aligned"
        )

        # Save grid
        grid_path = output_dir / f"comparison_{sample_name}.png"
        grid.save(grid_path)
        print(f"  Saved: {grid_path}")

        # Also save individual views for centered
        centered_dir = output_dir / sample_name / "centered"
        centered_dir.mkdir(parents=True, exist_ok=True)

        centered_pil.save(centered_dir / "input.png")
        for v, view_img in enumerate(centered_outputs):
            Image.fromarray(view_img).save(centered_dir / f"view_{v:02d}.png")

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
