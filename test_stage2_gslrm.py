#!/usr/bin/env python3
"""
Stage 2: GS-LRM 단독 테스트
입력: 6-view 이미지 + 카메라 파라미터
출력: 3D Gaussian Splatting (PLY) + 렌더링

테스트 목적:
- GS-LRM이 올바르게 3D 재구성하는지 확인
- Pretrained vs Mouse-finetuned 비교
- 다른 카메라 파라미터의 영향 확인
"""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from PIL import Image
from einops import rearrange

from gslrm.model.gaussians_renderer import render_turntable, imageseq2video


def load_images(image_dir: Path, num_views: int = 6, device: str = "cuda"):
    """Load 6-view images."""
    images = []
    for i in range(num_views):
        # Try different naming conventions
        for pattern in [f"view_{i:02d}.png", f"cam_{i:03d}.png", f"{i:02d}.png"]:
            img_path = image_dir / pattern
            if img_path.exists():
                img = Image.open(img_path).convert('RGB')
                img = np.array(img) / 255.0
                images.append(img)
                break

    if len(images) != num_views:
        raise ValueError(f"Expected {num_views} images, found {len(images)}")

    # [N, H, W, C] -> [N, C, H, W]
    images = np.stack(images)
    images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    return images.unsqueeze(0).to(device)  # [1, N, C, H, W]


def load_camera_params(camera_json: str, num_views: int = 6, device: str = "cuda"):
    """Load camera parameters."""
    with open(camera_json, 'r') as f:
        camera_data = json.load(f)["frames"]

    c2ws = []
    fxfycxcys = []

    for i in range(min(num_views, len(camera_data))):
        frame = camera_data[i]
        w2c = np.array(frame["w2c"])
        c2w = np.linalg.inv(w2c)
        c2ws.append(c2w)
        fxfycxcys.append([frame["fx"], frame["fy"], frame["cx"], frame["cy"]])

    c2ws = torch.from_numpy(np.array(c2ws)).float().unsqueeze(0).to(device)
    fxfycxcys = torch.from_numpy(np.array(fxfycxcys)).float().unsqueeze(0).to(device)

    return c2ws, fxfycxcys


def load_gslrm_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """Load GS-LRM model."""
    print(f"Loading config: {config_path}")
    with open(config_path, 'r') as f:
        config = edict(yaml.safe_load(f))

    module, class_name = config.model.class_name.rsplit(".", 1)
    GSLRM = importlib.import_module(module).__dict__[class_name]

    model = GSLRM(config).to(device)

    # Load checkpoint
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_dir():
        ckpt_files = list(ckpt_path.glob("ckpt_*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint in {ckpt_path}")
        ckpt_path = sorted(ckpt_files, key=lambda x: int(x.stem.split("_")[-1]))[-1]

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("loss_calculator.")}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, config


def run_inference(model, images, c2ws, fxfycxcys, device: str = "cuda"):
    """Run GS-LRM inference."""
    num_views = images.shape[1]

    index = torch.stack([
        torch.arange(num_views).long(),
        torch.zeros(num_views).long(),
    ], dim=-1).unsqueeze(0).to(device)

    batch = edict({
        "image": images,
        "c2w": c2ws,
        "fxfycxcy": fxfycxcys,
        "index": index,
    })

    with torch.no_grad(), torch.autocast(enabled=True, device_type="cuda", dtype=torch.float16):
        result = model.forward(batch, create_visual=True, split_data=True)

    return result


def main():
    parser = argparse.ArgumentParser(description="Stage 2: GS-LRM Test")

    parser.add_argument("--image_dir", type=str, required=True, help="Directory with 6-view images")
    parser.add_argument("--camera_json", type=str, required=True, help="Camera parameters JSON")
    parser.add_argument("--output_dir", type=str, default="outputs/stage2_gslrm")
    parser.add_argument("--config", type=str, default="configs/mouse_config.yaml")

    # Checkpoints to test
    parser.add_argument("--ckpt_pretrained", type=str, default="checkpoints/gslrm/ckpt_0000000000021125.pt")
    parser.add_argument("--ckpt_finetuned", type=str, default="checkpoints/gslrm/mouse_finetune")

    parser.add_argument("--test_both", action="store_true", help="Test both pretrained and finetuned")
    parser.add_argument("--no_turntable", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = Path(args.image_dir)
    sample_name = image_dir.name

    print("="*60)
    print("Stage 2: GS-LRM Test")
    print("="*60)

    # Load images
    print(f"\n[1] Loading images from: {image_dir}")
    images = load_images(image_dir, device=args.device)
    print(f"    Shape: {images.shape}")

    # Load camera parameters
    print(f"\n[2] Loading camera params: {args.camera_json}")
    c2ws, fxfycxcys = load_camera_params(args.camera_json, device=args.device)
    print(f"    C2W shape: {c2ws.shape}")

    checkpoints_to_test = []

    # Test pretrained
    if Path(args.ckpt_pretrained).exists():
        checkpoints_to_test.append(("pretrained", args.ckpt_pretrained))

    # Test finetuned
    if args.test_both and Path(args.ckpt_finetuned).exists():
        checkpoints_to_test.append(("finetuned", args.ckpt_finetuned))

    results = {}

    for ckpt_name, ckpt_path in checkpoints_to_test:
        print(f"\n[3] Testing {ckpt_name.upper()}: {ckpt_path}")

        model, config = load_gslrm_model(args.config, ckpt_path, args.device)
        result = run_inference(model, images, c2ws, fxfycxcys, args.device)

        # Save results
        save_dir = output_dir / f"{sample_name}_{ckpt_name}"
        save_dir.mkdir(exist_ok=True)

        # Get gaussians and apply filters
        gaussians = result.gaussians[0]
        filtered_gaussians = gaussians.apply_all_filters(
            opacity_thres=0.04,
            scaling_thres=0.1,
            floater_thres=0.6,
            crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
        )

        # Save PLY
        ply_path = save_dir / "gaussians.ply"
        filtered_gaussians.save_ply(str(ply_path))
        print(f"    Saved PLY: {ply_path}")

        # Save rendered views
        if result.render is not None:
            comp_image = result.render[0].detach()
            for i in range(comp_image.shape[0]):
                view_img = (comp_image[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(view_img).save(save_dir / f"render_{i:02d}.png")
            print(f"    Saved {comp_image.shape[0]} render views")

        # Generate turntable
        if not args.no_turntable:
            print("    Generating turntable video...")
            try:
                vis_image = render_turntable(filtered_gaussians, rendering_resolution=512, num_views=120)
                vis_image = rearrange(vis_image, "h (v w) c -> v h w c", v=120)
                vis_image = np.ascontiguousarray(vis_image)
                video_path = save_dir / "turntable.mp4"
                imageseq2video(vis_image, str(video_path), fps=30)
                print(f"    Saved turntable: {video_path}")
            except Exception as e:
                print(f"    Warning: turntable failed: {e}")

        results[ckpt_name] = save_dir

        del model
        torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("Stage 2 Complete!")
    print("="*60)
    print("\nResults:")
    for name, path in results.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
