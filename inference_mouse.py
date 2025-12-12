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

Complete inference pipeline for mouse 3D reconstruction:
1. Load 6-view images from mouse dataset
2. Run GSLRM to generate Gaussian splats
3. Render novel views (turntable video)
4. Export PLY and OBJ mesh

Usage:
    # From 6-view sample folder
    python inference_mouse.py \
        --sample_dir data_mouse/sample_000000 \
        --checkpoint checkpoints/gslrm/mouse/ \
        --output_dir outputs/mouse_inference/

    # Single image with MVDiffusion (when trained)
    python inference_mouse.py \
        --input_image examples/mouse.png \
        --mvdiffusion_checkpoint checkpoints/mouse_mvdiffusion/ \
        --checkpoint checkpoints/gslrm/mouse/ \
        --output_dir outputs/

Author: Claude Code (AI-assisted)
Date: 2024-12-09
"""

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from einops import rearrange
from PIL import Image
from tqdm import tqdm

# Local imports
from gslrm.model.gaussians_renderer import render_turntable, imageseq2video

# Auto-download utility
def ensure_weights_available():
    """Check and download model weights if missing."""
    try:
        from scripts.download_weights import ensure_weights
        return ensure_weights(auto_download=True)
    except ImportError:
        # Fallback: manual check
        from pathlib import Path
        gslrm_path = Path("checkpoints/gslrm/ckpt_0000000000021125.pt")
        if not gslrm_path.exists():
            print("WARNING: Model weights not found!")
            print("Run: python scripts/download_weights.py")
            return False
        return True

# Constants
DEFAULT_IMG_SIZE = 512
DEFAULT_TURNTABLE_VIEWS = 120
DEFAULT_TURNTABLE_FPS = 30

# MVDiffusion camera configuration (6 fixed views)
# Based on FaceLift paper: evenly distributed around the object
MVDIFFUSION_AZIMUTHS = [0, 60, 120, 180, 240, 300]  # degrees
MVDIFFUSION_ELEVATION = 0  # degrees (frontal view plane)


def compute_mvdiffusion_cameras(
    image_size: int = 512,
    device: str = "cuda",
    camera_distance: float = 2.7
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute camera parameters for MVDiffusion 6-view setup.

    MVDiffusion generates 6 views evenly distributed around the object
    at the same elevation level.

    Args:
        image_size: Image size for computing intrinsics
        device: Device to create tensors on
        camera_distance: Distance from camera to origin

    Returns:
        Tuple of (c2w matrices [6, 4, 4], fxfycxcy [6, 4])
    """
    c2ws = []
    fxfycxcys = []

    # Approximate FOV (~50 degrees) -> focal length
    fov = 50.0
    focal = image_size / (2 * np.tan(np.radians(fov / 2)))
    cx, cy = image_size / 2, image_size / 2

    for azim in MVDIFFUSION_AZIMUTHS:
        azim_rad = np.radians(azim)
        elev_rad = np.radians(MVDIFFUSION_ELEVATION)

        # Camera position on sphere
        x = camera_distance * np.cos(elev_rad) * np.sin(azim_rad)
        y = camera_distance * np.sin(elev_rad)
        z = camera_distance * np.cos(elev_rad) * np.cos(azim_rad)

        cam_pos = np.array([x, y, z])

        # Look-at matrix (camera looks at origin)
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(np.array([0, 1, 0]), forward)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)

        # Camera-to-world matrix
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward  # OpenGL convention
        c2w[:3, 3] = cam_pos

        c2ws.append(c2w)
        fxfycxcys.append([focal, focal, cx, cy])

    c2ws = torch.from_numpy(np.array(c2ws)).float().to(device)
    fxfycxcys = torch.from_numpy(np.array(fxfycxcys)).float().to(device)

    return c2ws, fxfycxcys


def load_config(config_path: str) -> edict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)


def load_model(config: edict, checkpoint_dir: str, device: str) -> torch.nn.Module:
    """
    Load trained GSLRM model from checkpoint.

    Args:
        config: Model configuration
        checkpoint_dir: Directory containing checkpoint files
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    # Dynamic model import
    module, class_name = config.model.class_name.rsplit(".", 1)
    GSLRM = importlib.import_module(module).__dict__[class_name]

    model = GSLRM(config).to(device)

    # Find latest checkpoint
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_files = list(checkpoint_dir.glob("ckpt_*.pt"))
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


def load_sample_data(
    sample_dir: str,
    image_size: int = 512,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load 6-view images and camera parameters from a sample directory.

    Args:
        sample_dir: Path to sample directory containing images/ and opencv_cameras.json
        image_size: Target image size
        device: Device to load tensors on

    Returns:
        Tuple of (images, c2w, fxfycxcy, index)
    """
    sample_path = Path(sample_dir)

    # Load camera parameters
    camera_json_path = sample_path / "opencv_cameras.json"
    if not camera_json_path.exists():
        raise FileNotFoundError(f"Camera file not found: {camera_json_path}")

    with open(camera_json_path, 'r') as f:
        camera_data = json.load(f)

    frames = camera_data["frames"]
    num_views = len(frames)

    # Determine images directory
    images_dir = sample_path / "images"
    if not images_dir.exists():
        images_dir = sample_path

    # Load images and camera params
    images = []
    c2ws = []
    fxfycxcys = []

    for i, frame in enumerate(frames):
        # Load image
        image_path = images_dir / f"cam_{i:03d}.png"
        if not image_path.exists():
            image_path = images_dir / frame.get("file_path", f"cam_{i:03d}.png").split("/")[-1]

        image = Image.open(image_path)

        # Handle RGBA
        if image.mode == "RGBA":
            # Composite on white background
            bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(bg, image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if needed
        if image.size[0] != image_size:
            image = image.resize((image_size, image_size), Image.LANCZOS)

        # To tensor [C, H, W]
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        images.append(image_tensor)

        # Camera extrinsics (w2c -> c2w)
        w2c = np.array(frame["w2c"])
        c2w = np.linalg.inv(w2c)
        c2ws.append(c2w)

        # Camera intrinsics
        fx, fy = frame["fx"], frame["fy"]
        cx, cy = frame["cx"], frame["cy"]
        # Scale intrinsics if image was resized
        scale = image_size / frame.get("w", image_size)
        fxfycxcys.append([fx * scale, fy * scale, cx * scale, cy * scale])

    # Stack tensors
    images = torch.stack(images, dim=0).unsqueeze(0).to(device)  # [1, V, C, H, W]
    c2ws = torch.from_numpy(np.array(c2ws)).float().unsqueeze(0).to(device)  # [1, V, 4, 4]
    fxfycxcys = torch.from_numpy(np.array(fxfycxcys)).float().unsqueeze(0).to(device)  # [1, V, 4]

    # Create index tensor [B, V, 2] - (view_idx, scene_idx)
    # IMPORTANT: Order must match training data loader!
    # - First element (index[:,:,0]): view_idx - used by model for view identification
    # - Second element (index[:,:,-1]): scene_idx - used for scene identification
    index = torch.stack([
        torch.arange(num_views).long(),  # view index (FIRST!)
        torch.zeros(num_views).long(),   # scene index (all 0 for single sample)
    ], dim=-1).unsqueeze(0).to(device)  # [1, V, 2]

    print(f"Loaded {num_views} views from {sample_dir}")
    print(f"  Images shape: {images.shape}")
    print(f"  C2W shape: {c2ws.shape}")
    print(f"  Intrinsics shape: {fxfycxcys.shape}")
    print(f"  Index shape: {index.shape}")

    return images, c2ws, fxfycxcys, index


def run_inference(
    model: torch.nn.Module,
    images: torch.Tensor,
    c2ws: torch.Tensor,
    fxfycxcys: torch.Tensor,
    index: torch.Tensor,
    device: str = "cuda"
) -> edict:
    """
    Run GSLRM inference on multi-view images.

    Args:
        model: GSLRM model
        images: Input images [B, V, C, H, W]
        c2ws: Camera-to-world matrices [B, V, 4, 4]
        fxfycxcys: Camera intrinsics [B, V, 4]
        index: View indices [B, V, 2]
        device: Device

    Returns:
        Model output containing gaussians and rendered images
    """
    # Create batch
    batch = edict({
        "image": images,
        "c2w": c2ws,
        "fxfycxcy": fxfycxcys,
        "index": index,
    })

    # Run inference
    with torch.no_grad(), torch.autocast(
        enabled=True,
        device_type="cuda" if "cuda" in device else "cpu",
        dtype=torch.float16
    ):
        result = model.forward(batch, create_visual=True, split_data=True)

    return result


def save_outputs(
    result: edict,
    output_dir: str,
    sample_name: str,
    save_turntable: bool = True,
    save_mesh: bool = True,
    turntable_views: int = DEFAULT_TURNTABLE_VIEWS,
    turntable_fps: int = DEFAULT_TURNTABLE_FPS,
    image_size: int = DEFAULT_IMG_SIZE
):
    """
    Save inference outputs: PLY, OBJ, rendered views, turntable video.

    Args:
        result: Model output
        output_dir: Output directory
        sample_name: Name for output files
        save_turntable: Whether to generate turntable video
        save_mesh: Whether to save mesh files (PLY, OBJ)
        turntable_views: Number of turntable views
        turntable_fps: Turntable video FPS
        image_size: Image size for rendering
    """
    output_path = Path(output_dir) / sample_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving outputs to {output_path}")

    # Get Gaussian model
    gaussians = result.gaussians[0]

    # Apply filters to clean up Gaussians
    filtered_gaussians = gaussians.apply_all_filters(
        opacity_thres=0.04,
        scaling_thres=0.1,
        floater_thres=0.6,
        crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
        cam_origins=None,
        nearfar_percent=(0.0001, 1.0),
    )

    # Save PLY
    if save_mesh:
        ply_path = output_path / "gaussians.ply"
        filtered_gaussians.save_ply(str(ply_path))
        print(f"  Saved PLY: {ply_path}")

        # Convert PLY to OBJ (point cloud as vertices)
        try:
            obj_path = output_path / "mesh.obj"
            ply_to_obj(str(ply_path), str(obj_path))
            print(f"  Saved OBJ: {obj_path}")
        except Exception as e:
            print(f"  Warning: Could not save OBJ: {e}")

    # Save rendered views from training
    if result.render is not None:
        comp_image = result.render[0].detach()
        v = comp_image.size(0)

        # Save individual views
        for i in range(v):
            view_img = comp_image[i].permute(1, 2, 0).cpu().numpy()
            view_img = (view_img * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(view_img).save(output_path / f"render_view_{i:02d}.png")

        # Save grid view
        if v > 1:
            comp_image_grid = rearrange(comp_image, "v c h w -> h (v w) c")
            comp_image_grid = (comp_image_grid.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(comp_image_grid).save(output_path / "render_grid.png")

        print(f"  Saved {v} rendered views")

    # Generate turntable video
    if save_turntable:
        print(f"  Generating turntable video ({turntable_views} views)...")
        try:
            vis_image = render_turntable(
                filtered_gaussians,
                rendering_resolution=image_size,
                num_views=turntable_views,
            )
            vis_image = rearrange(vis_image, "h (v w) c -> v h w c", v=turntable_views)
            vis_image = np.ascontiguousarray(vis_image)

            video_path = output_path / "turntable.mp4"
            imageseq2video(vis_image, str(video_path), fps=turntable_fps)
            print(f"  Saved turntable video: {video_path}")
        except Exception as e:
            print(f"  Warning: Could not generate turntable: {e}")

    print(f"  Done saving outputs for {sample_name}")


def ply_to_obj(ply_path: str, obj_path: str):
    """
    Convert PLY point cloud to OBJ format.

    For Gaussian splats, we extract the point positions and colors
    and save as a colored point cloud in OBJ format.

    Args:
        ply_path: Path to input PLY file
        obj_path: Path to output OBJ file
    """
    try:
        # Try using Open3D for better mesh handling
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(ply_path)

        # Option 1: Save as point cloud OBJ
        o3d.io.write_point_cloud(obj_path.replace('.obj', '_points.ply'), pcd)

        # Option 2: Create mesh using Poisson reconstruction (if enough points)
        if len(pcd.points) > 1000:
            # Estimate normals if not present
            if not pcd.has_normals():
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                )
                pcd.orient_normals_consistent_tangent_plane(k=15)

            # Poisson surface reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9
            )

            # Remove low-density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.05)
            mesh.remove_vertices_by_mask(vertices_to_remove)

            # Save mesh
            o3d.io.write_triangle_mesh(obj_path, mesh)
            print(f"    Created mesh with {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        else:
            # Just save points as OBJ vertices
            save_points_as_obj(pcd, obj_path)

    except ImportError:
        # Fallback: manual PLY parsing and OBJ writing
        print("    Open3D not available, using fallback PLY->OBJ conversion")
        convert_ply_to_obj_simple(ply_path, obj_path)


def save_points_as_obj(pcd, obj_path: str):
    """Save Open3D point cloud as OBJ file with vertex colors."""
    import open3d as o3d

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    with open(obj_path, 'w') as f:
        f.write(f"# Point cloud converted from PLY\n")
        f.write(f"# {len(points)} points\n")

        for i, p in enumerate(points):
            if colors is not None:
                c = colors[i]
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
            else:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def convert_ply_to_obj_simple(ply_path: str, obj_path: str):
    """Simple PLY to OBJ conversion without Open3D."""
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    vertex = ply['vertex']

    x = vertex['x']
    y = vertex['y']
    z = vertex['z']

    # Try to get colors
    has_color = 'red' in vertex.data.dtype.names

    with open(obj_path, 'w') as f:
        f.write(f"# Converted from {ply_path}\n")
        f.write(f"# {len(x)} vertices\n")

        for i in range(len(x)):
            if has_color:
                r = vertex['red'][i] / 255.0
                g = vertex['green'][i] / 255.0
                b = vertex['blue'][i] / 255.0
                f.write(f"v {x[i]:.6f} {y[i]:.6f} {z[i]:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
            else:
                f.write(f"v {x[i]:.6f} {y[i]:.6f} {z[i]:.6f}\n")


def find_sample_dirs(data_dir: str) -> List[str]:
    """Find all sample directories in data directory."""
    data_path = Path(data_dir)
    samples = []

    # Look for directories with opencv_cameras.json
    for item in sorted(data_path.iterdir()):
        if item.is_dir():
            if (item / "opencv_cameras.json").exists():
                samples.append(str(item))
            elif (item / "images" / "cam_000.png").exists():
                samples.append(str(item))

    return samples


def main():
    parser = argparse.ArgumentParser(description="Mouse-FaceLift Inference")

    # Input options
    parser.add_argument(
        "--sample_dir", type=str, default=None,
        help="Path to sample directory with 6 views and opencv_cameras.json"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Directory containing multiple sample directories"
    )
    parser.add_argument(
        "--input_image", type=str, default=None,
        help="Single input image (requires MVDiffusion)"
    )

    # Model options
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/gslrm/mouse/",
        help="Path to GSLRM checkpoint directory"
    )
    parser.add_argument(
        "--config", type=str, default="configs/mouse_config.yaml",
        help="Path to model config"
    )
    parser.add_argument(
        "--mvdiffusion_checkpoint", type=str, default=None,
        help="Path to MVDiffusion checkpoint (for single image input)"
    )

    # Output options
    parser.add_argument(
        "--output_dir", type=str, default="outputs/mouse_inference/",
        help="Output directory"
    )
    parser.add_argument(
        "--save_turntable", action="store_true", default=True,
        help="Generate turntable video"
    )
    parser.add_argument(
        "--no_turntable", action="store_true",
        help="Skip turntable video generation"
    )
    parser.add_argument(
        "--save_mesh", action="store_true", default=True,
        help="Save PLY and OBJ files"
    )
    parser.add_argument(
        "--turntable_views", type=int, default=DEFAULT_TURNTABLE_VIEWS,
        help="Number of turntable views"
    )

    # Hardware options
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on"
    )
    parser.add_argument(
        "--image_size", type=int, default=DEFAULT_IMG_SIZE,
        help="Image size"
    )

    # Zero123++ options (for single-image input)
    parser.add_argument(
        "--use_zero123pp", action="store_true",
        help="Use Zero123++ for single-image to multi-view generation"
    )
    parser.add_argument(
        "--zero123pp_model", type=str, default=None,
        help="Zero123++ model ID (default: sudo-ai/zero123plus-v1.2)"
    )
    parser.add_argument(
        "--zero123pp_steps", type=int, default=75,
        help="Number of diffusion steps for Zero123++"
    )
    parser.add_argument(
        "--zero123pp_guidance", type=float, default=4.0,
        help="Guidance scale for Zero123++"
    )

    # MVDiffusion options (for single-image input with fine-tuned model)
    parser.add_argument(
        "--mvdiffusion_steps", type=int, default=50,
        help="Number of diffusion steps for MVDiffusion"
    )
    parser.add_argument(
        "--mvdiffusion_guidance", type=float, default=3.0,
        help="Guidance scale for MVDiffusion"
    )
    parser.add_argument(
        "--prompt_embed_path", type=str,
        default="mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt",
        help="Path to prompt embeddings for MVDiffusion"
    )

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--use_pretrained_facelift", action="store_true",
        help="Use original FaceLift weights (auto-downloads from HuggingFace)"
    )
    parser.add_argument(
        "--auto_download", action="store_true", default=True,
        help="Auto-download weights if missing"
    )

    args = parser.parse_args()

    # Auto-download weights if needed
    if args.auto_download or args.use_pretrained_facelift:
        if not ensure_weights_available():
            print("ERROR: Could not ensure weights are available")
            print("Try manual download: python scripts/download_weights.py")
            return

    # Use FaceLift pretrained weights if requested
    if args.use_pretrained_facelift:
        args.checkpoint = "checkpoints/gslrm"
        print("Using FaceLift pretrained weights")

    # Validate inputs
    if not args.sample_dir and not args.data_dir and not args.input_image:
        parser.error("Must specify --sample_dir, --data_dir, or --input_image")

    if args.input_image and not (args.use_zero123pp or args.zero123pp_model or args.mvdiffusion_checkpoint):
        parser.error("--input_image requires --use_zero123pp or --mvdiffusion_checkpoint")

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load config and model
    print("Loading configuration...")
    config = load_config(args.config)
    config.model.image_tokenizer.image_size = args.image_size

    print("Loading GSLRM model...")
    model = load_model(config, args.checkpoint, args.device)
    print("Model loaded successfully")

    # Collect samples to process
    samples = []
    if args.sample_dir:
        samples.append(args.sample_dir)
    elif args.data_dir:
        samples = find_sample_dirs(args.data_dir)
        print(f"Found {len(samples)} samples in {args.data_dir}")
    elif args.input_image:
        # Single image mode - use Zero123++ or MVDiffusion
        if args.use_zero123pp or args.zero123pp_model:
            print("Using Zero123++ for single-image to multi-view generation")
            try:
                from mvdiffusion.pipelines.zero123pp_pipeline import Zero123PlusPipeline

                # Load Zero123++
                zero123_model = args.zero123pp_model or "sudo-ai/zero123plus-v1.2"
                print(f"Loading Zero123++ from {zero123_model}...")
                zero123_pipe = Zero123PlusPipeline.from_pretrained(
                    zero123_model,
                    device=args.device
                )

                # Process single image
                sample_name = Path(args.input_image).stem
                output_path = Path(args.output_dir) / sample_name
                output_path.mkdir(parents=True, exist_ok=True)

                # Generate views
                print(f"Generating 6 views from {args.input_image}...")
                batch = zero123_pipe.generate_for_gslrm(
                    args.input_image,
                    output_size=args.image_size,
                    seed=args.seed,
                    num_inference_steps=args.zero123pp_steps,
                    guidance_scale=args.zero123pp_guidance
                )

                # Save generated views
                views, _ = zero123_pipe.generate_views(
                    args.input_image,
                    output_size=args.image_size,
                    seed=args.seed
                )
                zero123_pipe.save_views(views, str(output_path / "generated_views"))

                # Run GSLRM
                print("Running GSLRM inference...")
                result = run_inference(
                    model,
                    batch['image'],
                    batch['c2w'],
                    batch['fxfycxcy'],
                    batch['index'],
                    args.device
                )

                # Save outputs
                do_turntable = args.save_turntable and not args.no_turntable
                save_outputs(
                    result,
                    args.output_dir,
                    sample_name,
                    save_turntable=do_turntable,
                    save_mesh=args.save_mesh,
                    turntable_views=args.turntable_views,
                    image_size=args.image_size
                )

                print(f"Done! Output saved to {output_path}")
                return

            except ImportError as e:
                print(f"Error: Could not import Zero123++ pipeline: {e}")
                print("Please install required dependencies: pip install diffusers")
                return
            except Exception as e:
                print(f"Error with Zero123++: {e}")
                import traceback
                traceback.print_exc()
                return
        elif args.mvdiffusion_checkpoint:
            print("Using MVDiffusion for single-image to multi-view generation")
            try:
                from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
                from diffusers import DDIMScheduler
                from PIL import Image
                import torchvision.transforms.functional as TF

                # Load MVDiffusion pipeline
                print(f"Loading MVDiffusion from {args.mvdiffusion_checkpoint}...")
                mvdiff_pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
                    args.mvdiffusion_checkpoint,
                    torch_dtype=torch.float16
                )
                mvdiff_pipe.to(args.device)

                # Enable memory optimizations
                if hasattr(mvdiff_pipe, 'enable_xformers_memory_efficient_attention'):
                    try:
                        mvdiff_pipe.enable_xformers_memory_efficient_attention()
                        print("  Enabled xformers attention")
                    except Exception:
                        pass

                # Load and preprocess input image
                input_img = Image.open(args.input_image)
                if input_img.mode == 'RGBA':
                    bg = Image.new('RGBA', input_img.size, (255, 255, 255, 255))
                    input_img = Image.alpha_composite(bg, input_img).convert('RGB')
                elif input_img.mode != 'RGB':
                    input_img = input_img.convert('RGB')

                if input_img.size != (args.image_size, args.image_size):
                    input_img = input_img.resize((args.image_size, args.image_size), Image.LANCZOS)

                # Load prompt embeddings
                prompt_embed_path = getattr(args, 'prompt_embed_path', None) or \
                    "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt"
                if os.path.exists(prompt_embed_path):
                    prompt_embeds = torch.load(prompt_embed_path)
                else:
                    print(f"Warning: Prompt embeddings not found at {prompt_embed_path}")
                    prompt_embeds = torch.zeros(6, 77, 1024)

                prompt_embeds = prompt_embeds.to(args.device, dtype=torch.float16)

                # Replicate input image for all views
                input_tensor = TF.to_tensor(input_img).unsqueeze(0).repeat(6, 1, 1, 1)
                input_tensor = input_tensor.to(args.device)

                # Set seed
                generator = torch.Generator(device=args.device).manual_seed(args.seed)

                # Generate multi-view images
                print(f"Generating 6 views from {args.input_image}...")
                mvdiff_output = mvdiff_pipe(
                    image=input_tensor,
                    prompt=[""] * 6,
                    prompt_embeds=prompt_embeds,
                    num_inference_steps=args.mvdiffusion_steps,
                    guidance_scale=args.mvdiffusion_guidance,
                    generator=generator,
                    output_type='pt'
                )

                generated_views = mvdiff_output.images  # [6, C, H, W]

                # Save generated views
                sample_name = Path(args.input_image).stem
                output_path = Path(args.output_dir) / sample_name
                output_path.mkdir(parents=True, exist_ok=True)
                views_dir = output_path / "generated_views"
                views_dir.mkdir(exist_ok=True)

                for i, view in enumerate(generated_views):
                    view_img = (view.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(view_img).save(views_dir / f"view_{i:02d}.png")
                print(f"  Saved generated views to {views_dir}")

                # Prepare data for GSLRM (use fixed camera poses)
                # MVDiffusion uses similar camera setup to Zero123++
                c2ws, fxfycxcys = compute_mvdiffusion_cameras(args.image_size, args.device)

                # Stack views for GSLRM
                images = generated_views.unsqueeze(0)  # [1, 6, C, H, W]
                c2ws = c2ws.unsqueeze(0)  # [1, 6, 4, 4]
                fxfycxcys = fxfycxcys.unsqueeze(0)  # [1, 6, 4]

                # Index tensor
                index = torch.stack([
                    torch.zeros(6).long(),
                    torch.arange(6).long()
                ], dim=-1).unsqueeze(0).to(args.device)

                # Run GSLRM
                print("Running GSLRM inference...")
                result = run_inference(
                    model,
                    images,
                    c2ws,
                    fxfycxcys,
                    index,
                    args.device
                )

                # Save outputs
                do_turntable = args.save_turntable and not args.no_turntable
                save_outputs(
                    result,
                    args.output_dir,
                    sample_name,
                    save_turntable=do_turntable,
                    save_mesh=args.save_mesh,
                    turntable_views=args.turntable_views,
                    image_size=args.image_size
                )

                print(f"Done! Output saved to {output_path}")
                return

            except ImportError as e:
                print(f"Error: Could not import MVDiffusion pipeline: {e}")
                return
            except Exception as e:
                print(f"Error with MVDiffusion: {e}")
                import traceback
                traceback.print_exc()
                return
        else:
            print("Single image mode requires --use_zero123pp or --mvdiffusion_checkpoint")
            return

    if not samples:
        print("No samples found to process")
        return

    # Process each sample
    save_turntable = args.save_turntable and not args.no_turntable

    for sample_dir in tqdm(samples, desc="Processing samples"):
        sample_name = Path(sample_dir).name

        try:
            # Load data
            images, c2ws, fxfycxcys, index = load_sample_data(
                sample_dir, args.image_size, args.device
            )

            # Run inference
            result = run_inference(model, images, c2ws, fxfycxcys, index, args.device)

            # Save outputs
            save_outputs(
                result,
                args.output_dir,
                sample_name,
                save_turntable=save_turntable,
                save_mesh=args.save_mesh,
                turntable_views=args.turntable_views,
                image_size=args.image_size
            )

        except Exception as e:
            print(f"Error processing {sample_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
