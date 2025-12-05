#!/usr/bin/env python3
# Copyright 2025 Adobe Inc.
# Modified for Mouse-FaceLift project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mouse-FaceLift Inference Script

Takes a single mouse image and outputs:
1. Rendered novel views from multiple angles
2. Optional 3D mesh export (if supported by model)

Usage:
    python inference_mouse.py \
        --input_dir examples/mouse/ \
        --output_dir outputs/mouse/ \
        --checkpoint checkpoints/gslrm/mouse/

    # Single image
    python inference_mouse.py \
        --input_image path/to/mouse.png \
        --output_dir outputs/ \
        --checkpoint checkpoints/gslrm/mouse/

Author: Claude Code (AI-assisted)
Date: 2024-12-04
"""

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from PIL import Image
from tqdm import tqdm


def load_config(config_path: str) -> edict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)


def load_model(config: edict, checkpoint_dir: str, device: str) -> torch.nn.Module:
    """
    Load trained model from checkpoint.

    Args:
        config: Model configuration
        checkpoint_dir: Directory containing checkpoint
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    # Dynamic model import
    module, class_name = config.model.class_name.rsplit(".", 1)
    GSLRM = importlib.import_module(module).__dict__[class_name]

    model = GSLRM(config).to(device)

    # Find latest checkpoint
    checkpoint_files = list(Path(checkpoint_dir).glob("ckpt_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    # Sort by step number and get latest
    checkpoint_files = sorted(
        checkpoint_files,
        key=lambda x: int(x.stem.split("_")[-1])
    )
    latest_ckpt = checkpoint_files[-1]

    print(f"Loading checkpoint: {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)

    # Extract model state dict from checkpoint
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        print(f"Loaded from training checkpoint (step {checkpoint.get('fwdbwd_pass_step', 'unknown')})")
    else:
        state_dict = checkpoint

    # Handle DDP wrapped state dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Filter out loss calculator weights (not needed for inference)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("loss_calculator.")}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def load_camera_params(camera_json_path: str) -> Dict:
    """Load camera parameters from JSON file."""
    with open(camera_json_path, 'r') as f:
        return json.load(f)


def get_default_cameras(num_views: int = 6, image_size: int = 512) -> Dict:
    """
    Generate default camera parameters for inference.

    Creates cameras arranged in a circle around the object.

    Args:
        num_views: Number of camera views
        image_size: Image size for intrinsics

    Returns:
        Camera parameters dict in FaceLift format
    """
    frames = []
    radius = 2.7
    fov_deg = 50
    fx = fy = 0.5 * image_size / np.tan(0.5 * np.deg2rad(fov_deg))
    cx = cy = image_size / 2

    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        cam_pos = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0.0
        ])

        # Camera orientation (looking at origin)
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        R = np.stack([right, -up, forward], axis=0)
        T = -R @ cam_pos.reshape(3, 1)

        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T.flatten()

        frame = {
            "w": image_size,
            "h": image_size,
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "w2c": w2c.tolist(),
            "file_path": f"images/cam_{i:03d}.png"
        }
        frames.append(frame)

    return {"frames": frames}


def preprocess_image(
    image_path: str,
    target_size: int = 512,
    bg_color: Tuple[int, int, int] = (255, 255, 255)
) -> torch.Tensor:
    """
    Preprocess input image for inference.

    Args:
        image_path: Path to input image
        target_size: Target image size
        bg_color: Background color for compositing

    Returns:
        Preprocessed image tensor [C, H, W]
    """
    image = Image.open(image_path)

    # Convert to RGBA if needed
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Resize to square
    w, h = image.size
    if w != h:
        # Center crop to square
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        image = image.crop((left, top, left + min_dim, top + min_dim))

    # Resize to target size
    if image.size[0] != target_size:
        image = image.resize((target_size, target_size), resample=Image.LANCZOS)

    # Convert to numpy
    image_np = np.array(image).astype(np.float32) / 255.0

    # Composite onto background if has alpha
    if image_np.shape[-1] == 4:
        alpha = image_np[..., 3:4]
        bg = np.array(bg_color, dtype=np.float32) / 255.0
        rgb = image_np[..., :3] * alpha + bg * (1 - alpha)
        image_np = np.concatenate([rgb, alpha[..., 0:1]], axis=-1)

    # To tensor [C, H, W]
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

    return image_tensor


def prepare_batch(
    image_tensor: torch.Tensor,
    cameras: Dict,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Prepare input batch for model inference.

    Args:
        image_tensor: Preprocessed image tensor [C, H, W]
        cameras: Camera parameters dict
        device: Device to move tensors to

    Returns:
        Batch dict ready for model input
    """
    frames = cameras["frames"]
    num_views = len(frames)

    # Replicate input image for all views (will be used as reference)
    images = image_tensor.unsqueeze(0).repeat(num_views, 1, 1, 1)  # [V, C, H, W]

    # Extract camera parameters
    c2ws = []
    fxfycxcys = []

    for frame in frames:
        w2c = np.array(frame["w2c"])
        c2w = np.linalg.inv(w2c)
        c2ws.append(c2w)

        intrinsics = np.array([
            frame["fx"], frame["fy"], frame["cx"], frame["cy"]
        ])
        fxfycxcys.append(intrinsics)

    c2ws = torch.from_numpy(np.array(c2ws)).float()  # [V, 4, 4]
    fxfycxcys = torch.from_numpy(np.array(fxfycxcys)).float()  # [V, 4]

    # Background color - must be [B, V, 3] to match data_splitter expectations
    bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    bg_color_batch = bg_color.unsqueeze(0).unsqueeze(0).repeat(1, num_views, 1)  # [1, V, 3]

    # View indices [B, V]
    index = torch.arange(num_views).unsqueeze(0)  # [1, V]

    batch = {
        "image": images.unsqueeze(0).to(device),  # [1, V, C, H, W]
        "c2w": c2ws.unsqueeze(0).to(device),  # [1, V, 4, 4]
        "fxfycxcy": fxfycxcys.unsqueeze(0).to(device),  # [1, V, 4]
        "bg_color": bg_color_batch.to(device),  # [1, V, 3]
        "index": index.to(device),  # [1, V]
    }

    return batch


def save_results(
    result: Dict,
    output_dir: str,
    image_name: str,
    save_video: bool = True
):
    """
    Save inference results.

    Args:
        result: Model output dict
        output_dir: Output directory
        image_name: Name for output files
        save_video: Whether to save as video
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save rendered views
    if "rendered_images" in result:
        rendered = result["rendered_images"]
        if isinstance(rendered, torch.Tensor):
            rendered = rendered.detach().cpu().numpy()

        # Save individual views
        for i, view in enumerate(rendered):
            if view.ndim == 3 and view.shape[0] in [3, 4]:
                view = view.transpose(1, 2, 0)
            view_uint8 = (view * 255).clip(0, 255).astype(np.uint8)

            if view_uint8.shape[-1] == 4:
                Image.fromarray(view_uint8, mode="RGBA").save(
                    os.path.join(output_dir, f"{image_name}_view_{i:02d}.png")
                )
            else:
                Image.fromarray(view_uint8[..., :3]).save(
                    os.path.join(output_dir, f"{image_name}_view_{i:02d}.png")
                )

        # Save as video if requested
        if save_video and len(rendered) > 1:
            try:
                import cv2
                video_path = os.path.join(output_dir, f"{image_name}_views.mp4")
                h, w = rendered[0].shape[:2] if rendered[0].ndim == 3 else rendered[0].shape[1:3]

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, 10.0, (w, h))

                for view in rendered:
                    if view.ndim == 3 and view.shape[0] in [3, 4]:
                        view = view.transpose(1, 2, 0)
                    frame = (view[..., :3] * 255).clip(0, 255).astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame)

                out.release()
                print(f"Saved video: {video_path}")
            except Exception as e:
                print(f"Warning: Could not save video: {e}")

    # Save gaussian parameters if available
    if "gaussians" in result:
        try:
            gaussians = result["gaussians"]
            # Extract serializable tensor data only
            gaussians_data = {}
            if hasattr(gaussians, '__dict__'):
                for k, v in gaussians.__dict__.items():
                    if isinstance(v, torch.Tensor):
                        gaussians_data[k] = v.detach().cpu()
            elif isinstance(gaussians, dict):
                for k, v in gaussians.items():
                    if isinstance(v, torch.Tensor):
                        gaussians_data[k] = v.detach().cpu()

            if gaussians_data:
                gaussians_path = os.path.join(output_dir, f"{image_name}_gaussians.pt")
                torch.save(gaussians_data, gaussians_path)
                print(f"Saved gaussians: {gaussians_path}")
        except Exception as e:
            print(f"Warning: Could not save gaussians: {e}")


def find_images(input_path: str) -> List[str]:
    """Find all image files in directory or return single image path."""
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        images = []
        for ext in image_extensions:
            images.extend(Path(input_path).glob(f"*{ext}"))
            images.extend(Path(input_path).glob(f"*{ext.upper()}"))
        return sorted([str(p) for p in images])

    return []


def main():
    parser = argparse.ArgumentParser(description="Mouse-FaceLift Inference")
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--input_image", type=str, default=None,
        help="Path to single input image"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/mouse/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/gslrm/mouse/",
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--config", type=str, default="configs/mouse_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--camera_json", type=str, default=None,
        help="Path to camera parameters JSON (optional)"
    )
    parser.add_argument(
        "--num_views", type=int, default=6,
        help="Number of output views"
    )
    parser.add_argument(
        "--image_size", type=int, default=512,
        help="Image size for inference"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on"
    )
    parser.add_argument(
        "--save_video", action="store_true",
        help="Save results as video"
    )
    args = parser.parse_args()

    # Determine input path
    input_path = args.input_image or args.input_dir
    if not input_path:
        parser.error("Either --input_dir or --input_image must be specified")

    # Find images
    image_paths = find_images(input_path)
    if not image_paths:
        print(f"No images found in {input_path}")
        return

    print(f"Found {len(image_paths)} images to process")

    # Load configuration
    config = load_config(args.config)

    # Override config with command line args
    config.model.image_tokenizer.image_size = args.image_size

    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load model
    print("Loading model...")
    model = load_model(config, args.checkpoint, args.device)
    print("Model loaded successfully")

    # Load or generate camera parameters
    if args.camera_json and os.path.exists(args.camera_json):
        cameras = load_camera_params(args.camera_json)
    else:
        cameras = get_default_cameras(args.num_views, args.image_size)
        print(f"Using default camera arrangement with {args.num_views} views")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each image
    for image_path in tqdm(image_paths, desc="Processing images"):
        image_name = Path(image_path).stem

        try:
            # Preprocess image
            image_tensor = preprocess_image(
                image_path, args.image_size
            )

            # Prepare batch
            batch = prepare_batch(image_tensor, cameras, args.device)

            # Run inference
            with torch.no_grad(), torch.autocast(
                enabled=True, device_type="cuda" if "cuda" in args.device else "cpu",
                dtype=torch.bfloat16
            ):
                result = model(batch, create_visual=True)

            # Save results
            save_results(
                result, args.output_dir, image_name,
                save_video=args.save_video
            )

            print(f"Processed: {image_name}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
