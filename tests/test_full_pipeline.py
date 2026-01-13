#!/usr/bin/env python3
"""
Full pipeline test: MVDiffusion (single-view → 6-view) → GS-LRM (6-view → 3D)

Usage:
    python test_full_pipeline.py --input_image path/to/mouse.png --output_dir outputs/test_pipeline
"""

import argparse
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from diffusers import DDIMScheduler
from easydict import EasyDict as edict
from einops import rearrange
from PIL import Image
from tqdm import tqdm

# Local imports
from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from gslrm.model.gaussians_renderer import render_turntable, imageseq2video


def load_mvdiffusion_pipeline(
    pretrained_path: str,
    trained_unet_path: str,
    device: str = "cuda"
):
    """
    Load MVDiffusion pipeline with trained UNet.

    Args:
        pretrained_path: Path to pretrained pipeline (FaceLift)
        trained_unet_path: Path to trained UNet checkpoint
        device: Device to use

    Returns:
        Loaded pipeline
    """
    print(f"Loading MVDiffusion pipeline from {pretrained_path}")

    # Load base pipeline
    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        pretrained_path,
        torch_dtype=torch.float16,
    )

    # Load trained UNet
    print(f"Loading trained UNet from {trained_unet_path}")
    from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel

    trained_unet = UNetMV2DConditionModel.from_pretrained(
        trained_unet_path,
        torch_dtype=torch.float16,
    )

    # Replace UNet
    pipeline.unet = trained_unet

    # Move to device
    pipeline.to(device)

    # Enable memory optimizations
    if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("  Enabled xformers attention")
        except Exception as e:
            print(f"  Warning: Could not enable xformers: {e}")

    return pipeline


def load_gslrm_model(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda"
):
    """Load GS-LRM model."""
    print(f"Loading GS-LRM config from {config_path}")
    with open(config_path, 'r') as f:
        config = edict(yaml.safe_load(f))

    # Dynamic model import
    module, class_name = config.model.class_name.rsplit(".", 1)
    GSLRM = importlib.import_module(module).__dict__[class_name]

    model = GSLRM(config).to(device)

    # Find latest checkpoint
    checkpoint_dir = Path(checkpoint_path)
    if checkpoint_dir.is_dir():
        checkpoint_files = list(checkpoint_dir.glob("ckpt_*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
        checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.stem.split("_")[-1]))
        latest_ckpt = checkpoint_files[-1]
    else:
        latest_ckpt = checkpoint_dir

    print(f"Loading GS-LRM checkpoint: {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)

    # Extract model state dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Handle DDP wrapped state dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Filter out loss calculator weights
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("loss_calculator.")}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, config


def load_camera_params(camera_json_path: str, device: str = "cuda"):
    """Load camera parameters for 6 views."""
    print(f"Loading camera params from {camera_json_path}")

    with open(camera_json_path, 'r') as f:
        camera_data = json.load(f)["frames"]

    # Use first 6 views
    num_views = min(6, len(camera_data))

    c2ws = []
    fxfycxcys = []

    for i in range(num_views):
        frame = camera_data[i]

        # Camera extrinsics (w2c -> c2w)
        w2c = np.array(frame["w2c"])
        c2w = np.linalg.inv(w2c)
        c2ws.append(c2w)

        # Camera intrinsics
        fx, fy = frame["fx"], frame["fy"]
        cx, cy = frame["cx"], frame["cy"]
        fxfycxcys.append([fx, fy, cx, cy])

    c2ws = torch.from_numpy(np.array(c2ws)).float().unsqueeze(0).to(device)
    fxfycxcys = torch.from_numpy(np.array(fxfycxcys)).float().unsqueeze(0).to(device)

    return c2ws, fxfycxcys


def preprocess_image(image_path: str, image_size: int = 512) -> Image.Image:
    """Load and preprocess input image."""
    image = Image.open(image_path)

    # Handle RGBA
    if image.mode == 'RGBA':
        bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(bg, image).convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.LANCZOS)

    return image


def main():
    parser = argparse.ArgumentParser(description="Full pipeline test: MVDiffusion → GS-LRM")

    # Input
    parser.add_argument("--input_image", type=str, required=True, help="Input image path")

    # Model paths
    parser.add_argument(
        "--mvdiff_pretrained", type=str,
        default="checkpoints/mvdiffusion/pipeckpts",
        help="Path to pretrained MVDiffusion pipeline"
    )
    parser.add_argument(
        "--mvdiff_unet", type=str,
        default="checkpoints/mvdiffusion/mouse/original_facelift_embeds/checkpoint-6000/unet",
        help="Path to trained UNet"
    )
    parser.add_argument(
        "--gslrm_checkpoint", type=str,
        default="checkpoints/gslrm/mouse_finetune",
        help="Path to GS-LRM checkpoint"
    )
    parser.add_argument(
        "--gslrm_config", type=str,
        default="configs/mouse_config.yaml",
        help="Path to GS-LRM config"
    )
    parser.add_argument(
        "--prompt_embeds", type=str,
        default="mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt",
        help="Path to prompt embeddings"
    )
    parser.add_argument(
        "--camera_json", type=str,
        default="data_mouse/sample_000000/opencv_cameras.json",
        help="Path to camera parameters (must match prompt_embeds)"
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/test_pipeline", help="Output directory")

    # Generation parameters
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="MVDiffusion guidance scale")
    parser.add_argument("--num_steps", type=int, default=50, help="MVDiffusion inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--image_size", type=int, default=512, help="Image size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory structure
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_name = Path(args.input_image).stem
    sample_dir = output_dir / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Stage-wise subdirectories
    stage1_dir = sample_dir / "stage1_mvdiffusion"
    stage2_dir = sample_dir / "stage2_gslrm"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Full Pipeline Test: MVDiffusion → GS-LRM")
    print(f"{'='*60}\n")

    # =========================================================================
    # Step 1: Load input image
    # =========================================================================
    print("Step 1: Loading input image...")
    input_image = preprocess_image(args.input_image, args.image_size)
    input_image.save(stage1_dir / "input.png")
    print(f"  Saved input image to {stage1_dir / 'input.png'}")

    # =========================================================================
    # Step 2: Load MVDiffusion pipeline
    # =========================================================================
    print("\nStep 2: Loading MVDiffusion pipeline...")
    mvdiff_pipeline = load_mvdiffusion_pipeline(
        args.mvdiff_pretrained,
        args.mvdiff_unet,
        args.device
    )

    # Load prompt embeddings
    prompt_embeds = torch.load(args.prompt_embeds, map_location=args.device)
    prompt_embeds = prompt_embeds.to(torch.float16)
    print(f"  Prompt embeds shape: {prompt_embeds.shape}")

    # =========================================================================
    # Step 3: Generate multi-view images with MVDiffusion
    # =========================================================================
    print("\nStep 3: Generating 6 views with MVDiffusion...")

    # Prepare input tensor
    import torchvision.transforms.functional as TF
    input_tensor = TF.to_tensor(input_image).unsqueeze(0)  # [1, C, H, W]
    input_tensor = input_tensor.repeat(6, 1, 1, 1).to(args.device)  # [6, C, H, W]

    # Generate
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    with torch.no_grad():
        output = mvdiff_pipeline(
            input_tensor,
            None,
            prompt_embeds=prompt_embeds,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            generator=generator,
            output_type='pt',
            eta=1.0,
        )

    generated_views = output.images  # [6, C, H, W]
    print(f"  Generated views shape: {generated_views.shape}")

    # Save generated views to stage1
    for i in range(generated_views.shape[0]):
        view_img = (generated_views[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(view_img).save(stage1_dir / f"view_{i:02d}.png")

    # Save grid
    grid = rearrange(generated_views, "V C H W -> H (V W) C")
    grid_img = (grid.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(grid_img).save(stage1_dir / "multiview_grid.png")
    print(f"  Saved generated views to {stage1_dir}")

    # =========================================================================
    # Step 4: Load GS-LRM model
    # =========================================================================
    print("\nStep 4: Loading GS-LRM model...")
    gslrm_model, gslrm_config = load_gslrm_model(
        args.gslrm_config,
        args.gslrm_checkpoint,
        args.device
    )

    # =========================================================================
    # Step 5: Load camera parameters
    # =========================================================================
    print("\nStep 5: Loading camera parameters...")
    c2ws, fxfycxcys = load_camera_params(args.camera_json, args.device)
    print(f"  C2W shape: {c2ws.shape}")
    print(f"  Intrinsics shape: {fxfycxcys.shape}")

    # =========================================================================
    # Step 6: Run GS-LRM inference
    # =========================================================================
    print("\nStep 6: Running GS-LRM inference...")

    # Prepare batch
    images = generated_views.unsqueeze(0)  # [1, 6, C, H, W]

    # Index tensor
    index = torch.stack([
        torch.arange(6).long(),  # view index
        torch.zeros(6).long(),   # scene index
    ], dim=-1).unsqueeze(0).to(args.device)

    batch = edict({
        "image": images,
        "c2w": c2ws,
        "fxfycxcy": fxfycxcys,
        "index": index,
    })

    with torch.no_grad(), torch.autocast(enabled=True, device_type="cuda", dtype=torch.float16):
        result = gslrm_model.forward(batch, create_visual=True, split_data=True)

    # =========================================================================
    # Step 7: Save outputs
    # =========================================================================
    print("\nStep 7: Saving outputs...")

    # Get Gaussian model and apply filters
    gaussians = result.gaussians[0]
    filtered_gaussians = gaussians.apply_all_filters(
        opacity_thres=0.04,
        scaling_thres=0.1,
        floater_thres=0.6,
        crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
        cam_origins=None,
        nearfar_percent=(0.0001, 1.0),
    )

    # Save PLY to stage2
    ply_path = stage2_dir / "gaussians.ply"
    filtered_gaussians.save_ply(str(ply_path))
    print(f"  Saved PLY: {ply_path}")

    # Save rendered views to stage2
    if result.render is not None:
        comp_image = result.render[0].detach()
        for i in range(comp_image.shape[0]):
            view_img = (comp_image[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(view_img).save(stage2_dir / f"render_view_{i:02d}.png")
        print(f"  Saved {comp_image.shape[0]} rendered views to {stage2_dir}")

    # Generate turntable video
    print("  Generating turntable video...")
    try:
        vis_image = render_turntable(
            filtered_gaussians,
            rendering_resolution=args.image_size,
            num_views=120,
        )
        vis_image = rearrange(vis_image, "h (v w) c -> v h w c", v=120)
        vis_image = np.ascontiguousarray(vis_image)

        video_path = stage2_dir / "turntable.mp4"
        imageseq2video(vis_image, str(video_path), fps=30)
        print(f"  Saved turntable video: {video_path}")
    except Exception as e:
        print(f"  Warning: Could not generate turntable: {e}")

    print(f"\n{'='*60}")
    print("Done! Outputs saved to:")
    print(f"  Stage 1 (MVDiffusion): {stage1_dir}")
    print(f"  Stage 2 (GS-LRM):      {stage2_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
