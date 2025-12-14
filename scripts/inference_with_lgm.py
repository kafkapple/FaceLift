#!/usr/bin/env python3
"""
MVDiffusion + LGM Integration Script

Pipeline: Single Image → MVDiffusion (6 views) → LGM (4 views) → 3D Gaussian

LGM expects 4 views at 90° intervals (0, 90, 180, 270)
MVDiffusion generates 6 views at 60° intervals (0, 60, 120, 180, 240, 300)

View mapping strategy:
- LGM view 0 (0°)   ← MVDiffusion view 0 (0°)
- LGM view 1 (90°)  ← MVDiffusion view 1 (60°) - closest approximation
- LGM view 2 (180°) ← MVDiffusion view 3 (180°)
- LGM view 3 (270°) ← MVDiffusion view 5 (300°) - closest approximation

Usage:
    python scripts/inference_with_lgm.py \
        --input_image data_mouse/sample_000000/images/cam_000.png \
        --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse/facelift_prompt_6x/checkpoint-2000 \
        --lgm_checkpoint LGM/pretrained/model_fp16.safetensors \
        --output_dir outputs/lgm_inference
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_mvdiffusion_pipeline(checkpoint_path: str, device: str = "cuda"):
    """Load MVDiffusion pipeline with trained UNet."""
    from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
    from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel

    checkpoint_path = Path(checkpoint_path)
    unet_path = checkpoint_path / "unet"

    if unet_path.exists():
        # Training checkpoint: load base + replace UNet
        base_pipeline = "checkpoints/mvdiffusion/pipeckpts"
        print(f"Loading base MVDiffusion from {base_pipeline}...")
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            base_pipeline, torch_dtype=torch.float16
        )
        print(f"Loading trained UNet from {unet_path}...")
        trained_unet = UNetMV2DConditionModel.from_pretrained(
            str(unet_path), torch_dtype=torch.float16
        )
        pipe.unet = trained_unet
    else:
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            str(checkpoint_path), torch_dtype=torch.float16
        )

    pipe.to(device)
    if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe


def generate_6views_mvdiffusion(
    pipe,
    input_image: Image.Image,
    prompt_embed_path: str = "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt",
    num_steps: int = 25,
    guidance_scale: float = 3.0,
    seed: int = 42,
    device: str = "cuda"
) -> torch.Tensor:
    """Generate 6 views using MVDiffusion."""
    import torchvision.transforms.functional as TF

    # Preprocess input
    if input_image.mode == 'RGBA':
        bg = Image.new('RGBA', input_image.size, (255, 255, 255, 255))
        input_image = Image.alpha_composite(bg, input_image).convert('RGB')
    elif input_image.mode != 'RGB':
        input_image = input_image.convert('RGB')

    if input_image.size != (512, 512):
        input_image = input_image.resize((512, 512), Image.LANCZOS)

    # Load prompt embeddings
    if os.path.exists(prompt_embed_path):
        prompt_embeds = torch.load(prompt_embed_path, weights_only=False)
    else:
        print(f"Warning: Prompt embeddings not found at {prompt_embed_path}")
        prompt_embeds = torch.zeros(6, 77, 1024)
    prompt_embeds = prompt_embeds.to(device, dtype=torch.float16)

    # Prepare input tensor
    input_tensor = TF.to_tensor(input_image).unsqueeze(0).repeat(6, 1, 1, 1)
    input_tensor = input_tensor.to(device)

    # Generate
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"Generating 6 views with MVDiffusion ({num_steps} steps)...")

    output = pipe(
        image=input_tensor,
        prompt=[""] * 6,
        prompt_embeds=prompt_embeds,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type='pt'
    )

    return output.images  # [6, C, H, W]


def convert_6views_to_4views(views_6: torch.Tensor) -> torch.Tensor:
    """
    Convert MVDiffusion 6 views to LGM 4 views.

    MVDiffusion: [0°, 60°, 120°, 180°, 240°, 300°]
    LGM expects: [0°, 90°, 180°, 270°]

    Mapping:
    - 0° → 0° (exact)
    - 90° → 60° (closest, 30° off)
    - 180° → 180° (exact)
    - 270° → 300° (closest, 30° off)
    """
    # indices: MVDiffusion view indices to use
    indices = [0, 1, 3, 5]  # 0°, 60°, 180°, 300°

    views_4 = views_6[indices]  # [4, C, H, W]
    print(f"Converted 6 views → 4 views (indices: {indices})")

    return views_4


def prepare_lgm_input(
    views_4: torch.Tensor,
    input_size: int = 256,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Prepare 4 views for LGM input format.

    LGM expects:
    - 4 views at 256x256
    - Normalized with ImageNet stats
    - Concatenated with ray embeddings
    """
    IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # Resize to LGM input size
    views_4 = F.interpolate(
        views_4,
        size=(input_size, input_size),
        mode='bilinear',
        align_corners=False
    )

    # Normalize
    views_4 = (views_4 - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD

    return views_4  # [4, C, H, W]


def save_views(views: torch.Tensor, output_dir: str, prefix: str = "view"):
    """Save generated views as images."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, view in enumerate(views):
        img = (view.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(output_path / f"{prefix}_{i:02d}.png")

    print(f"Saved {len(views)} views to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MVDiffusion + LGM Inference")

    parser.add_argument("--input_image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--mvdiffusion_checkpoint", type=str, required=True,
                        help="Path to MVDiffusion checkpoint")
    parser.add_argument("--lgm_checkpoint", type=str, default=None,
                        help="Path to LGM checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/lgm_inference",
                        help="Output directory")
    parser.add_argument("--mvdiffusion_steps", type=int, default=25,
                        help="MVDiffusion inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                        help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    parser.add_argument("--skip_lgm", action="store_true",
                        help="Skip LGM inference (only generate views)")

    args = parser.parse_args()

    # Create output directory
    sample_name = Path(args.input_image).stem
    output_path = Path(args.output_dir) / sample_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load MVDiffusion
    print("\n" + "="*60)
    print("Step 1: Loading MVDiffusion pipeline")
    print("="*60)
    mvdiff_pipe = load_mvdiffusion_pipeline(args.mvdiffusion_checkpoint, args.device)

    # Step 2: Generate 6 views
    print("\n" + "="*60)
    print("Step 2: Generating 6 views with MVDiffusion")
    print("="*60)
    input_image = Image.open(args.input_image)
    views_6 = generate_6views_mvdiffusion(
        mvdiff_pipe,
        input_image,
        num_steps=args.mvdiffusion_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=args.device
    )

    # Save 6 views
    save_views(views_6, output_path / "mvdiffusion_6views", "view")

    # Step 3: Convert to 4 views for LGM
    print("\n" + "="*60)
    print("Step 3: Converting to 4 views for LGM")
    print("="*60)
    views_4 = convert_6views_to_4views(views_6)
    save_views(views_4, output_path / "lgm_4views", "view")

    # Step 4: LGM inference (if checkpoint provided)
    if args.lgm_checkpoint and not args.skip_lgm:
        print("\n" + "="*60)
        print("Step 4: Running LGM inference")
        print("="*60)

        try:
            # Check if LGM is available
            lgm_path = Path(args.lgm_checkpoint).parent.parent
            sys.path.insert(0, str(lgm_path))

            from core.options import Options
            from core.models import LGM

            # Prepare input
            lgm_input = prepare_lgm_input(views_4, input_size=256, device=args.device)

            # Load LGM model
            print(f"Loading LGM from {args.lgm_checkpoint}...")

            # This is a placeholder - actual LGM loading requires more setup
            # including ray embeddings and proper model initialization
            print("LGM inference requires additional setup.")
            print("Please run LGM inference manually:")
            print(f"  cd LGM && python infer.py big --resume {args.lgm_checkpoint} --test_path {output_path / 'lgm_4views'}")

        except ImportError as e:
            print(f"LGM not available: {e}")
            print("\nTo use LGM, install it first:")
            print("  git clone https://github.com/3DTopia/LGM")
            print("  cd LGM && pip install -r requirements.txt")
            print("  wget https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16.safetensors -P pretrained/")
    else:
        print("\n" + "="*60)
        print("Step 4: Skipping LGM (no checkpoint or --skip_lgm)")
        print("="*60)
        print(f"\nTo run LGM manually on generated views:")
        print(f"  cd LGM && python infer.py big --resume pretrained/model_fp16.safetensors --test_path {output_path / 'lgm_4views'}")

    print("\n" + "="*60)
    print(f"Done! Outputs saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
