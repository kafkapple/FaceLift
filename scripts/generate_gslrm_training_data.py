#!/usr/bin/env python3
"""
Generate GS-LRM Training Data from MVDiffusion

Phase 2 of the training pipeline:
1. Load trained MVDiffusion model
2. For each sample, use each of the 6 views as input (6x augmentation)
3. Generate synthetic 6-view images
4. Save with matching camera parameters

Usage:
    python scripts/generate_gslrm_training_data.py \
        --mvdiff_checkpoint checkpoints/mvdiffusion/mouse/mouse_embeds_6x_aug/checkpoint-XXXX \
        --input_data data_mouse/data_mouse_train.txt \
        --output_dir data_mouse_synthetic \
        --prompt_embeds mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt \
        --camera_json data_mouse/sample_000000/opencv_cameras.json
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# Local imports
from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel


def load_mvdiffusion_pipeline(
    pretrained_path: str,
    unet_checkpoint: str,
    device: str = "cuda"
):
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

    # Enable memory optimizations
    if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Warning: Could not enable xformers: {e}")

    return pipeline


def load_image(image_path: str, image_size: int = 512) -> Image.Image:
    """Load and preprocess image."""
    image = Image.open(image_path)

    if image.mode == 'RGBA':
        bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(bg, image).convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.LANCZOS)

    return image


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

    # Prepare input tensor
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

    return output.images  # [6, C, H, W]


def main():
    parser = argparse.ArgumentParser(description="Generate GS-LRM training data from MVDiffusion")

    # Paths
    parser.add_argument("--mvdiff_pretrained", type=str,
                        default="checkpoints/mvdiffusion/pipeckpts",
                        help="Pretrained MVDiffusion path")
    parser.add_argument("--mvdiff_checkpoint", type=str, required=True,
                        help="Trained MVDiffusion checkpoint (e.g., checkpoint-10000)")
    parser.add_argument("--input_data", type=str,
                        default="data_mouse/data_mouse_train.txt",
                        help="Input data list file")
    parser.add_argument("--output_dir", type=str,
                        default="data_mouse_synthetic",
                        help="Output directory for synthetic data")
    parser.add_argument("--prompt_embeds", type=str,
                        default="mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt",
                        help="Prompt embeddings path")
    parser.add_argument("--camera_json", type=str,
                        default="data_mouse/sample_000000/opencv_cameras.json",
                        help="Reference camera JSON (will be copied to each sample)")

    # Generation parameters
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # Options
    parser.add_argument("--augment_all_views", action="store_true", default=True,
                        help="Use all 6 views as input (6x augmentation)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip samples that already exist")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation split ratio")

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

    # Load reference camera (to be copied to all samples)
    with open(args.camera_json, 'r') as f:
        camera_data = json.load(f)

    # Load input data list
    with open(args.input_data, 'r') as f:
        sample_paths = [line.strip() for line in f if line.strip()]

    print(f"\n{'='*60}")
    print(f"Generating GS-LRM Training Data")
    print(f"{'='*60}")
    print(f"Input samples: {len(sample_paths)}")
    print(f"Augmentation: {'All 6 views' if args.augment_all_views else 'View 0 only'}")
    print(f"Expected output: {len(sample_paths) * (6 if args.augment_all_views else 1)} samples")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # Track generated samples
    train_samples = []
    val_samples = []
    sample_idx = 0

    for sample_path in tqdm(sample_paths, desc="Processing samples"):
        sample_path = Path(sample_path)
        images_dir = sample_path / "images"

        if not images_dir.exists():
            images_dir = sample_path

        # Determine which views to use as input
        if args.augment_all_views:
            input_views = list(range(args.num_views))
        else:
            input_views = [0]

        for ref_view_idx in input_views:
            # Output sample name
            out_sample_name = f"sample_{sample_idx:06d}"
            out_sample_dir = output_dir / out_sample_name
            out_images_dir = out_sample_dir / "images"

            # Skip if exists
            if args.skip_existing and out_sample_dir.exists():
                sample_idx += 1
                continue

            out_images_dir.mkdir(parents=True, exist_ok=True)

            # Load input image
            input_image_path = images_dir / f"cam_{ref_view_idx:03d}.png"
            if not input_image_path.exists():
                print(f"Warning: {input_image_path} not found, skipping")
                continue

            input_image = load_image(str(input_image_path), args.image_size)

            # Generate views
            generated_views = generate_views(
                pipeline,
                input_image,
                prompt_embeds,
                args.num_views,
                args.guidance_scale,
                args.num_steps,
                args.seed + sample_idx,  # Vary seed per sample
                args.device
            )

            # Save generated views
            for view_idx in range(args.num_views):
                view_img = generated_views[view_idx].permute(1, 2, 0).cpu().numpy()
                view_img = (view_img * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(view_img).save(out_images_dir / f"cam_{view_idx:03d}.png")

            # Copy camera parameters
            with open(out_sample_dir / "opencv_cameras.json", 'w') as f:
                json.dump(camera_data, f, indent=2)

            # Save metadata
            metadata = {
                "source_sample": str(sample_path),
                "reference_view": ref_view_idx,
                "generation_params": {
                    "guidance_scale": args.guidance_scale,
                    "num_steps": args.num_steps,
                    "seed": args.seed + sample_idx,
                }
            }
            with open(out_sample_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            # Add to train/val split
            if np.random.random() < args.val_ratio:
                val_samples.append(str(out_sample_dir))
            else:
                train_samples.append(str(out_sample_dir))

            sample_idx += 1

    # Save train/val splits
    with open(output_dir / "data_train.txt", 'w') as f:
        f.write("\n".join(train_samples))

    with open(output_dir / "data_val.txt", 'w') as f:
        f.write("\n".join(val_samples))

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"{'='*60}")
    print(f"Total samples generated: {sample_idx}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Train list: {output_dir / 'data_train.txt'}")
    print(f"Val list: {output_dir / 'data_val.txt'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
