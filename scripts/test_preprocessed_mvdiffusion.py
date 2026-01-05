#!/usr/bin/env python3
"""
Test MVDiffusion with preprocessed (background-removed, centered) images.

This script:
1. Loads preprocessed images (from preprocess_mouse_for_mvdiffusion.py)
2. Runs MVDiffusion inference
3. Creates a grid visualization for comparison
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel


def load_mvdiffusion_pipeline(pretrained_path: str, unet_checkpoint: str, device: str = "cuda"):
    """Load MVDiffusion pipeline."""
    print(f"Loading MVDiffusion pipeline...")

    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        pretrained_path,
        torch_dtype=torch.float16,
    )

    unet_path = os.path.join(unet_checkpoint, "unet")
    if os.path.exists(unet_path):
        print(f"Loading trained UNet from {unet_path}")
        trained_unet = UNetMV2DConditionModel.from_pretrained(
            unet_path,
            torch_dtype=torch.float16,
        )
        pipeline.unet = trained_unet

    pipeline.to(device)

    if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except:
            pass

    return pipeline


def generate_views(pipeline, input_image, prompt_embeds, num_views=6,
                   guidance_scale=3.0, num_steps=50, seed=42, device="cuda"):
    """Generate multi-view images."""

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


def create_grid(input_img, output_images, title=""):
    """Create visualization grid."""
    img_size = 256
    n_cols = 7
    padding = 5

    grid_width = n_cols * img_size + (n_cols + 1) * padding
    grid_height = img_size + 2 * padding + 30

    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    # Title
    if title:
        draw.text((10, 5), title, fill=(0, 0, 0), font=font)

    y = 25

    # Input image
    x = padding
    img = input_img.resize((img_size, img_size), Image.LANCZOS)
    grid.paste(img, (x, y))
    draw.text((x + 5, y + img_size - 20), "Input", fill=(0, 0, 0), font=font)

    # Generated views
    for i, view_img in enumerate(output_images):
        x = padding + (i + 1) * (img_size + padding)
        img = view_img.resize((img_size, img_size), Image.LANCZOS)
        grid.paste(img, (x, y))
        draw.text((x + 5, y + img_size - 20), f"View {i}", fill=(0, 0, 0), font=font)

    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_dir", type=str, required=True,
                        help="Directory with preprocessed samples")
    parser.add_argument("--output_dir", type=str, default="outputs/preprocessed_mvdiff_test")
    parser.add_argument("--mvdiff_pretrained", type=str,
                        default="checkpoints/mvdiffusion/pipeckpts")
    parser.add_argument("--mvdiff_checkpoint", type=str,
                        default="checkpoints/mvdiffusion/mouse/facelift_prompt_6x_resumed/checkpoint-12500")
    parser.add_argument("--prompt_embeds", type=str,
                        default="mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    pipeline = load_mvdiffusion_pipeline(
        args.mvdiff_pretrained,
        args.mvdiff_checkpoint,
        args.device
    )

    # Load prompt embeddings
    prompt_embeds = torch.load(args.prompt_embeds, map_location=args.device, weights_only=False)
    prompt_embeds = prompt_embeds.to(torch.float16)
    print(f"Prompt embeds shape: {prompt_embeds.shape}")

    # Get sample list
    preprocessed_dir = Path(args.preprocessed_dir)
    sample_dirs = sorted(preprocessed_dir.glob("sample_*"))[:args.num_samples]

    print(f"\nTesting {len(sample_dirs)} samples from {preprocessed_dir}")
    print(f"Output: {output_dir}\n")

    for sample_dir in tqdm(sample_dirs, desc="Processing"):
        sample_name = sample_dir.name
        images_dir = sample_dir / "images"

        # Find input image (view 0)
        input_path = None
        for pattern in ["00.png", "cam_000.png"]:
            p = images_dir / pattern
            if p.exists():
                input_path = p
                break

        if not input_path:
            print(f"  Skipping {sample_name}: no input image")
            continue

        # Load input
        input_img = Image.open(input_path).convert('RGB')

        # Generate views
        views = generate_views(
            pipeline, input_img, prompt_embeds,
            seed=args.seed, device=args.device
        )

        # Convert to PIL images
        output_images = []
        for i in range(6):
            view_arr = views[i].permute(1, 2, 0).cpu().numpy()
            view_arr = (view_arr * 255).clip(0, 255).astype(np.uint8)
            output_images.append(Image.fromarray(view_arr))

        # Create grid
        grid = create_grid(input_img, output_images, title=f"Preprocessed: {sample_name}")
        grid.save(output_dir / f"grid_{sample_name}.png")

        # Save individual views
        views_dir = output_dir / sample_name
        views_dir.mkdir(exist_ok=True)
        input_img.save(views_dir / "input.png")
        for i, img in enumerate(output_images):
            img.save(views_dir / f"view_{i:02d}.png")

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
