#!/usr/bin/env python3
"""
Stage 1: MVDiffusion 단독 테스트
입력: Single-view 이미지 (512x512)
출력: 6-view 이미지

테스트 목적:
- MVDiffusion이 올바르게 multi-view를 생성하는지 확인
- 학습된 UNet vs Pretrained UNet 비교
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from einops import rearrange
from tqdm import tqdm

from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel


def load_pipeline(pretrained_path: str, unet_path: str = None, device: str = "cuda"):
    """Load MVDiffusion pipeline."""
    print(f"Loading pipeline from {pretrained_path}")

    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        pretrained_path,
        torch_dtype=torch.float16,
    )

    if unet_path and Path(unet_path).exists():
        print(f"Loading fine-tuned UNet from {unet_path}")
        trained_unet = UNetMV2DConditionModel.from_pretrained(
            unet_path,
            torch_dtype=torch.float16,
        )
        pipeline.unet = trained_unet
    else:
        print("Using pretrained UNet (no fine-tuning)")

    pipeline.to(device)

    if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except:
            pass

    return pipeline


def generate_multiview(
    pipeline,
    input_image: Image.Image,
    prompt_embeds: torch.Tensor,
    guidance_scale: float = 3.0,
    num_steps: int = 50,
    seed: int = 42,
    device: str = "cuda"
):
    """Generate 6 views from single input."""

    # Prepare input tensor
    input_tensor = TF.to_tensor(input_image).unsqueeze(0)  # [1, C, H, W]
    input_tensor = input_tensor.repeat(6, 1, 1, 1).to(device)  # [6, C, H, W]

    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        output = pipeline(
            input_tensor,
            None,
            prompt_embeds=prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
            output_type='pt',
            eta=1.0,
        )

    return output.images  # [6, C, H, W]


def main():
    parser = argparse.ArgumentParser(description="Stage 1: MVDiffusion Test")

    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/stage1_mvdiffusion")
    parser.add_argument("--pretrained", type=str, default="checkpoints/mvdiffusion/pipeckpts")
    parser.add_argument("--unet", type=str, default="checkpoints/mvdiffusion/mouse/original_facelift_embeds/checkpoint-6000/unet")
    parser.add_argument("--prompt_embeds", type=str, default="mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt")
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # Test mode
    parser.add_argument("--test_pretrained", action="store_true", help="Also test pretrained UNet")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_name = Path(args.input_image).stem

    print("="*60)
    print("Stage 1: MVDiffusion Test")
    print("="*60)

    # Load input image
    print(f"\n[1] Loading input image: {args.input_image}")
    image = Image.open(args.input_image)
    if image.mode == 'RGBA':
        bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(bg, image).convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    if image.size != (512, 512):
        image = image.resize((512, 512), Image.LANCZOS)

    # Load prompt embeddings
    print(f"[2] Loading prompt embeddings: {args.prompt_embeds}")
    prompt_embeds = torch.load(args.prompt_embeds, map_location=args.device, weights_only=False)
    prompt_embeds = prompt_embeds.to(torch.float16)
    print(f"    Shape: {prompt_embeds.shape}")

    results = {}

    # Test 1: Fine-tuned UNet (if available)
    if Path(args.unet).exists():
        print(f"\n[3a] Testing FINE-TUNED UNet: {args.unet}")

        pipeline = load_pipeline(args.pretrained, args.unet, args.device)
        views = generate_multiview(
            pipeline, image, prompt_embeds,
            args.guidance_scale, args.num_steps, args.seed, args.device
        )

        # Save results
        save_dir = output_dir / f"{sample_name}_finetuned"
        save_dir.mkdir(exist_ok=True)

        image.save(save_dir / "input.png")

        for i in range(views.shape[0]):
            view_img = (views[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(view_img).save(save_dir / f"view_{i:02d}.png")

        # Save grid
        grid = rearrange(views, "V C H W -> H (V W) C")
        grid_img = (grid.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(grid_img).save(save_dir / "multiview_grid.png")

        print(f"    Saved to: {save_dir}")
        results['finetuned'] = save_dir

        del pipeline
        torch.cuda.empty_cache()

    # Test 2: Pretrained UNet
    if args.test_pretrained:
        print(f"\n[3b] Testing PRETRAINED UNet (no fine-tuning)")

        pipeline = load_pipeline(args.pretrained, None, args.device)
        views = generate_multiview(
            pipeline, image, prompt_embeds,
            args.guidance_scale, args.num_steps, args.seed, args.device
        )

        # Save results
        save_dir = output_dir / f"{sample_name}_pretrained"
        save_dir.mkdir(exist_ok=True)

        image.save(save_dir / "input.png")

        for i in range(views.shape[0]):
            view_img = (views[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(view_img).save(save_dir / f"view_{i:02d}.png")

        grid = rearrange(views, "V C H W -> H (V W) C")
        grid_img = (grid.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(grid_img).save(save_dir / "multiview_grid.png")

        print(f"    Saved to: {save_dir}")
        results['pretrained'] = save_dir

    print("\n" + "="*60)
    print("Stage 1 Complete!")
    print("="*60)
    print("\nResults:")
    for name, path in results.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
