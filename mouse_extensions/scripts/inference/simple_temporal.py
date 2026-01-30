#!/usr/bin/env python3
# DEPRECATED: Use unified pipeline instead:#   python -m mouse_extensions.inference.run --config configs/inference/default.yaml# This script is kept for backward compatibility.
"""
Simple Temporal Video - Enhanced with Gaussian Export and Rerun Support

Features:
- Turntable videos with configurable rotation speed
- Gaussian .ply and .npz export
- Rerun .rrd sequence for interactive 3D viewing
- Input grid video

Usage:
    python -m mouse_extensions.scripts.inference.simple_temporal \\
        --checkpoint M5_E0_1_facelift \\
        --config configs/base/gslrm_mouse.yaml \\
        --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \\
        --start_frame 0 --end_frame 100 \\
        --save_gaussian --save_rerun \\
        --rotation_speed 0.5 \\
        --output_dir outputs/temporal_M5

Author: FaceLift Team
Date: 2026-01-29
"""

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict as edict
from einops import rearrange
from omegaconf import OmegaConf
import cv2
from tqdm import tqdm
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gslrm.model.gaussians_renderer import render_turntable, imageseq2video


# ============================================================================
# Grid Label Helper Functions
# ============================================================================

def add_turntable_row_labels(
    grid_image: np.ndarray,
    num_views: int,
    grid_rows: int,
    grid_cols: int,
    row_height: int,
    label_height: int = 45,
    elevation: float = 20.0,
) -> np.ndarray:
    """
    Add angle info labels to turntable grid image.
    
    Args:
        grid_image: [H, W, 3] uint8 image
        num_views: Total number of views (360 degree)
        grid_rows: Number of rows in grid
        grid_cols: Number of columns in grid
        row_height: Height of each cell
        label_height: Height of label bar
        elevation: Camera elevation angle
    
    Returns:
        Image with label bars added
    """
    h, w = grid_image.shape[:2]
    
    # Calculate angle step
    angle_step = 360.0 / num_views
    
    # Create new image with label bars
    new_height = h + label_height * grid_rows
    result = np.zeros((new_height, w, 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for row_idx in range(grid_rows):
        # Position of this row's label bar
        label_y = row_idx * (row_height + label_height)
        # Position of this row's image data
        img_y = label_y + label_height
        
        # Copy image row
        src_start = row_idx * row_height
        src_end = src_start + row_height
        if src_end <= h:
            result[img_y:img_y + row_height, :] = grid_image[src_start:src_end, :]
        
        # Draw label bar (dark background)
        result[label_y:label_y + label_height, :] = (30, 30, 30)
        
        # Calculate angle range for this row
        start_view = row_idx * grid_cols
        end_view = min(start_view + grid_cols - 1, num_views - 1)
        start_angle = int(start_view * angle_step)
        end_angle = int(end_view * angle_step)
        
        # Line 1: Angle range
        angle_text = f"Views {start_view}-{end_view} | {start_angle} deg - {end_angle} deg"
        font_scale = 1.0
        (tw, th), _ = cv2.getTextSize(angle_text, font, font_scale, 2)
        text_y = label_y + th + 8
        cv2.putText(result, angle_text, (10, text_y), font, font_scale, (255, 255, 255), 2)
        
        # Line 2: Elevation info (smaller, on the right side)
        elev_text = f"Elev: {elevation:.0f} deg"
        (tw2, th2), _ = cv2.getTextSize(elev_text, font, 0.7, 1)
        cv2.putText(result, elev_text, (w - tw2 - 10, text_y), font, 0.7, (180, 180, 180), 1)
    
    return result


def add_input_view_labels(
    grid_image: np.ndarray,
    num_cams: int,
    grid_rows: int,
    grid_cols: int,
    row_height: int,
    label_height: int = 35,
) -> np.ndarray:
    """
    Add camera labels to input view grid.
    
    Args:
        grid_image: [H, W, 3] uint8
        num_cams: Number of cameras
        grid_rows, grid_cols: Grid dimensions
        row_height: Cell height
        label_height: Label bar height
    
    Returns:
        Labeled grid image
    """
    h, w = grid_image.shape[:2]
    
    # Create new image with label bars
    new_height = h + label_height * grid_rows
    result = np.zeros((new_height, w, 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for row_idx in range(grid_rows):
        label_y = row_idx * (row_height + label_height)
        img_y = label_y + label_height
        
        # Copy image row
        src_start = row_idx * row_height
        src_end = src_start + row_height
        if src_end <= h:
            result[img_y:img_y + row_height, :] = grid_image[src_start:src_end, :]
        
        # Draw label bar
        result[label_y:label_y + label_height, :] = (30, 30, 30)
        
        # Camera indices for this row
        start_cam = row_idx * grid_cols
        end_cam = min(start_cam + grid_cols - 1, num_cams - 1)
        
        cam_text = f"Input Cameras {start_cam}-{end_cam}"
        font_scale = 0.9
        (tw, th), _ = cv2.getTextSize(cam_text, font, font_scale, 2)
        text_y = label_y + th + 6
        cv2.putText(result, cam_text, (10, text_y), font, font_scale, (255, 255, 255), 2)
    
    return result




def find_checkpoint(checkpoint_input: str, base_dir: str = "checkpoints/gslrm") -> str:
    """
    Flexibly find checkpoint file.

    Args:
        checkpoint_input: Can be:
            - Full path to .pt file
            - Directory containing .pt files
            - Dataset/experiment name (e.g., "M5_E0_1_facelift")
            - "pretrained" or "base" for original checkpoint

    Returns:
        Path to the checkpoint file
    """
    base_path = Path(base_dir)
    input_path = Path(checkpoint_input)

    # Case 1: Exact file path exists
    if input_path.exists() and input_path.is_file():
        print(f"Using checkpoint: {input_path}")
        return str(input_path)

    # Case 2: "pretrained" or "base" -> original checkpoint
    if checkpoint_input.lower() in ["pretrained", "base", "original"]:
        pretrained = base_path / "ckpt_0000000000021125.pt"
        if pretrained.exists():
            print(f"Using pretrained checkpoint: {pretrained}")
            return str(pretrained)
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained}")

    # Case 3: Directory path or name
    search_dir = None
    if input_path.exists() and input_path.is_dir():
        search_dir = input_path
    elif (base_path / checkpoint_input).exists():
        search_dir = base_path / checkpoint_input

    if search_dir:
        # Find best.pt or best_psnr.pt first
        for best_name in ["best.pt", "best_psnr.pt"]:
            best_pt = search_dir / best_name
            if best_pt.exists():
                print(f"Using best checkpoint: {best_pt}")
                return str(best_pt)

        # Find all ckpt_*.pt files and get the latest
        pt_files = list(search_dir.glob("ckpt_*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No checkpoint files found in {search_dir}")

        def extract_step(p):
            match = re.search(r"ckpt_(\d+)\.pt", p.name)
            return int(match.group(1)) if match else 0

        pt_files.sort(key=extract_step, reverse=True)
        latest = pt_files[0]
        print(f"Using latest checkpoint: {latest} (step {extract_step(latest)})")
        return str(latest)

    # Case 4: Try as experiment name pattern
    matching_dirs = list(base_path.glob(f"{checkpoint_input}*"))
    if matching_dirs:
        matching_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return find_checkpoint(str(matching_dirs[0]), base_dir)

    # List available options
    available = [d.name for d in base_path.iterdir() if d.is_dir()]
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_input}. "
        f"Available: {', '.join(available[:5])}..."
    )


def load_model(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Load GS-LRM model."""
    from gslrm.model.gslrm import GSLRM

    # Auto-find checkpoint
    checkpoint_path = find_checkpoint(checkpoint_path)

    config = OmegaConf.load(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = GSLRM(config)
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    step = checkpoint.get("fwdbwd_pass_step", checkpoint.get("step", "?"))
    print(f"Model loaded. Step: {step}")
    return model, config


def load_sample(sample_dir: Path, device: str = "cuda") -> edict:
    """Load sample - minimal preprocessing."""
    import cv2
    from mouse_extensions.data import (
        normalize_camera_distance_with_intrinsics,
        normalize_cameras_to_z_up,
    )

    cam_path = sample_dir / "opencv_cameras.json"
    with open(cam_path) as f:
        cam_data = json.load(f)

    frames = cam_data["frames"]
    images, c2ws, fxfycxcy_list = [], [], []

    for i, frame_info in enumerate(frames):
        img_path = sample_dir / "images" / f"cam_{i:03d}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if img.shape[-1] == 4:
            img_rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = np.mean(img_rgb, axis=-1)
            alpha = (gray < 250).astype(np.float32)[..., None]

        img_rgb = img_rgb.astype(np.float32) / 255.0
        img_rgba = img_rgb * alpha + 1.0 * (1 - alpha)  # white bg
        images.append(img_rgba)

        w2c = np.array(frame_info["w2c"])
        c2ws.append(np.linalg.inv(w2c))
        fxfycxcy_list.append(
            [frame_info["fx"], frame_info["fy"], frame_info["cx"], frame_info["cy"]]
        )

    images = np.stack(images)
    c2ws = np.array(c2ws)
    fxfycxcy = np.array(fxfycxcy_list)

    # Normalize
    c2ws = normalize_cameras_to_z_up(c2ws)
    c2ws, fxfycxcy = normalize_camera_distance_with_intrinsics(c2ws, fxfycxcy, 2.7)

    # Keep raw images for input grid visualization
    raw_images = (images * 255).clip(0, 255).astype(np.uint8)

    images = torch.from_numpy(images).float().to(device)
    images = rearrange(images, "v h w c -> v c h w")

    return (
        edict(
            {
                "image": images.unsqueeze(0),
                "c2w": torch.from_numpy(c2ws).float().to(device).unsqueeze(0),
                "fxfycxcy": torch.from_numpy(fxfycxcy).float().to(device).unsqueeze(0),
                "index": torch.zeros(len(frames), 3, dtype=torch.long, device=device).unsqueeze(0),
            }
        ),
        raw_images,
    )


@torch.no_grad()
def process_frame(
    model,
    sample_dir: Path,
    device: str,
    resolution: int,
    num_views: int,
    elevation: float,
    radius: float,
    return_gaussians: bool = False,
):
    """Process single frame -> turntable frames and optionally Gaussians."""

    sample, raw_input_images = load_sample(sample_dir, device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(sample, create_visual=False, split_data=False)

    gaussians = output.gaussians[0]

    # Render turntable
    turntable_strip = render_turntable(
        gaussians,
        rendering_resolution=resolution,
        num_views=num_views,
        elevation=elevation,
        radius=radius,
        trajectory_mode="turntable",
    )
    # [H, V*W, C] → [V, H, W, C]
    h = turntable_strip.shape[0]
    w = turntable_strip.shape[1] // num_views
    frames = turntable_strip.reshape(h, num_views, w, 3)
    frames = np.transpose(frames, (1, 0, 2, 3))

    result = (frames, raw_input_images)

    if return_gaussians:
        # Return gaussians for export (clone to avoid memory issues after cleanup)
        result = (frames, raw_input_images, gaussians)
    else:
        del gaussians

    del sample, output
    gc.collect()
    torch.cuda.empty_cache()

    return result


def find_sample_dirs(data_dir: Path) -> list:
    """Auto-discover sample directories (numeric folder names), sorted."""
    dirs = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and d.name.isdigit():
            dirs.append((int(d.name), d))
    return dirs


def interpolate_frames_for_speed(frames: np.ndarray, speed_factor: float) -> np.ndarray:
    """
    Interpolate frames to adjust rotation speed.
    
    Args:
        frames: [N, H, W, C] array
        speed_factor: 0.5 = half speed (2x frames), 2.0 = double speed (0.5x frames)
    
    Returns:
        Interpolated frames
    """
    if speed_factor == 1.0:
        return frames
    
    num_original = frames.shape[0]
    num_target = int(num_original / speed_factor)
    
    if num_target <= 1:
        return frames[:1]
    
    indices = np.linspace(0, num_original - 1, num_target)
    new_frames = []
    
    for idx in indices:
        lower = int(np.floor(idx))
        upper = min(int(np.ceil(idx)), num_original - 1)
        t = idx - lower
        
        if lower == upper or t < 0.001:
            new_frames.append(frames[lower])
        else:
            blended = (1 - t) * frames[lower].astype(np.float32) + t * frames[upper].astype(np.float32)
            new_frames.append(blended.astype(np.uint8))
    
    return np.stack(new_frames)


def main():
    parser = argparse.ArgumentParser(
        description="Simple Temporal Video with Gaussian Export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (100 frames test)
  python -m mouse_extensions.scripts.inference.simple_temporal \\
      --checkpoint M5_E0_1_facelift \\
      --config configs/base/gslrm_mouse.yaml \\
      --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \\
      --start_frame 0 --end_frame 100

  # With Gaussian export and half-speed rotation
  python -m mouse_extensions.scripts.inference.simple_temporal \\
      --checkpoint M5_E0_1_facelift \\
      --config configs/base/gslrm_mouse.yaml \\
      --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \\
      --save_gaussian --save_rerun \\
      --rotation_speed 0.5
        """,
    )
    
    # Model and data
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint path, directory, or experiment name")
    parser.add_argument("--config", type=str, default="configs/base/gslrm_mouse.yaml",
                        help="Config file path (default: configs/base/gslrm_mouse.yaml)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Preprocessed dataset directory")
    
    # Frame selection
    parser.add_argument("--start_frame", type=int, default=None,
                        help="Start frame index (default: auto-detect)")
    parser.add_argument("--end_frame", type=int, default=None,
                        help="End frame index (default: auto-detect)")
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Frame step (default: 1)")
    parser.add_argument("--split", type=str, default=None,
                        help="Split file path (overrides start/end/step)")
    
    # Rendering settings
    parser.add_argument("--num_views", type=int, default=36,
                        help="360-degree subdivisions (default: 36 = 10° per view)")
    parser.add_argument("--resolution", type=int, default=384,
                        help="Rendering resolution (default: 384)")
    parser.add_argument("--elevation", type=float, default=20.0,
                        help="Camera elevation angle (default: 20.0)")
    parser.add_argument("--radius", type=float, default=2.7,
                        help="Camera distance (default: 2.7)")
    
    # Video settings
    parser.add_argument("--fps", type=int, default=24,
                        help="Video FPS (default: 24)")
    parser.add_argument("--rotation_speed", type=float, default=0.5,
                        help="Rotation speed factor: 0.5=half speed, 1.0=normal (default: 0.5)")
    parser.add_argument("--fixed_angles", type=int, nargs="+", default=[0],
                        help="Fixed angle views for time videos (default: [0])")
    
    # Export options (all enabled by default)
    parser.add_argument("--save_gaussian", action="store_true", default=True,
                        help="Save Gaussian .ply and .npz files (default: True)")
    parser.add_argument("--no_gaussian", action="store_true",
                        help="Disable Gaussian file saving")
    parser.add_argument("--save_rerun", action="store_true", default=True,
                        help="Save Rerun .rrd files for interactive viewing (default: True)")
    parser.add_argument("--no_rerun", action="store_true",
                        help="Disable Rerun .rrd file saving")
    parser.add_argument("--save_first_only", action="store_true",
                        help="Only save Gaussian/Rerun for first frame (faster)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/simple_temporal",
                        help="Output directory (default: outputs/simple_temporal)")
    
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for exports
    if args.save_gaussian:
        (output_dir / "gaussians").mkdir(exist_ok=True)
    if args.save_rerun:
        (output_dir / "rerun").mkdir(exist_ok=True)

    model, config = load_model(args.checkpoint, args.config, "cuda")
    data_dir = Path(args.data_dir)

    # Load frame indices
    if args.split:
        split_path = Path(args.split)
        if not split_path.exists():
            split_path = data_dir / args.split
        with open(split_path) as f:
            lines = [l.strip().rstrip("/") for l in f if l.strip()]
        frame_indices = sorted([int(Path(l).name) for l in lines])
        frame_indices = frame_indices[:: args.frame_step]
        print(f"Split file: {len(frame_indices)} frames (step {args.frame_step})")
    elif args.start_frame is None or args.end_frame is None:
        sample_dirs = find_sample_dirs(data_dir)
        if not sample_dirs:
            print(f"No numeric sample directories found in {data_dir}")
            return
        all_indices = [idx for idx, _ in sample_dirs]
        start = args.start_frame if args.start_frame is not None else all_indices[0]
        end = args.end_frame if args.end_frame is not None else all_indices[-1] + 1
        frame_indices = [idx for idx in all_indices if start <= idx < end]
        frame_indices = frame_indices[:: args.frame_step]
        print(
            f"Auto-detected {len(all_indices)} samples, using {len(frame_indices)} "
            f"(range {start}-{end}, step {args.frame_step})"
        )
    else:
        frame_indices = list(range(args.start_frame, args.end_frame, args.frame_step))
        print(f"Processing {len(frame_indices)} frames")

    # Handle --no_* flags
    if args.no_gaussian:
        args.save_gaussian = False
    if args.no_rerun:
        args.save_rerun = False
    
    # Check if we need to export Gaussians
    need_gaussians = args.save_gaussian or args.save_rerun
    
    # Import visualizer if needed
    visualizer = None
    if need_gaussians:
        from mouse_extensions.visualization.unified_visualizer import (
            GaussianExporter,
            RerunExporter,
        )

    all_turntables = []
    all_input_views = []
    all_gaussians = []  # Store Gaussians for sequence export

    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Frames")):
        sample_dir = data_dir / f"{frame_idx:06d}"
        if not sample_dir.exists():
            print(f"Skip: {sample_dir}")
            continue

        # Determine if we need Gaussians for this frame
        export_this_frame = need_gaussians and (not args.save_first_only or i == 0)
        
        result = process_frame(
            model,
            sample_dir,
            "cuda",
            args.resolution,
            args.num_views,
            args.elevation,
            args.radius,
            return_gaussians=export_this_frame,
        )

        if export_this_frame:
            frames, raw_inputs, gaussians = result
            
            # Export Gaussian immediately to free memory
            uid = f"frame_{frame_idx:06d}"
            
            if args.save_gaussian:
                npz_path = str(output_dir / "gaussians" / f"{uid}.npz")
                GaussianExporter.to_npz(gaussians, npz_path)
                
                ply_path = str(output_dir / "gaussians" / f"{uid}.ply")
                GaussianExporter.to_ply(gaussians, ply_path)
                
                if i == 0:
                    print(f"Saved: {uid}.ply, {uid}.npz")
            
            if args.save_rerun and (args.save_first_only or i == len(frame_indices) - 1):
                # For save_first_only, save single .rrd
                # Otherwise, we will create sequence .rrd at the end
                if args.save_first_only:
                    npz_path = str(output_dir / "gaussians" / f"{uid}.npz")
                    rrd_path = str(output_dir / "rerun" / f"{uid}.rrd")
                    if RerunExporter.is_available():
                        RerunExporter.single_frame_to_rrd(npz_path, rrd_path)
                        print(f"Saved: {uid}.rrd")
            
            del gaussians
            gc.collect()
            torch.cuda.empty_cache()
        else:
            frames, raw_inputs = result

        all_turntables.append(frames)
        all_input_views.append(raw_inputs)

    if not all_turntables:
        print("No frames!")
        return

    T = len(all_turntables)
    V = all_turntables[0].shape[0]
    H, W = all_turntables[0].shape[1:3]
    print(f"Collected: T={T}, V={V}, H={H}, W={W}")

    # Create Rerun sequence if not save_first_only
    if args.save_rerun and not args.save_first_only and RerunExporter.is_available():
        rrd_path = str(output_dir / "rerun" / "sequence.rrd")
        npz_dir = str(output_dir / "gaussians")
        if os.path.exists(npz_dir) and list(Path(npz_dir).glob("*.npz")):
            RerunExporter.sequence_to_rrd(npz_dir, rrd_path)
            print(f"Saved: sequence.rrd (timeline with {T} frames)")

    # Apply rotation speed factor
    speed_info = ""
    if args.rotation_speed != 1.0:
        speed_info = f" (speed={args.rotation_speed})"
        all_turntables = [
            interpolate_frames_for_speed(t, args.rotation_speed) for t in all_turntables
        ]
        V_new = all_turntables[0].shape[0]
        print(f"Rotation speed {args.rotation_speed}: {V} -> {V_new} views per frame")
        V = V_new

    # === Output 1: First frame 360° turntable ===
    imageseq2video(all_turntables[0], str(output_dir / "turntable_first.mp4"), fps=args.fps)
    print(f"Saved: turntable_first.mp4{speed_info}")

    # === Output 2: Fixed angle, time variation ===
    for angle_idx in args.fixed_angles:
        angle_idx = angle_idx % V
        fixed_frames = np.stack([t[angle_idx] for t in all_turntables])
        suffix = f"_angle{angle_idx}" if len(args.fixed_angles) > 1 else ""
        imageseq2video(fixed_frames, str(output_dir / f"time_fixed{suffix}.mp4"), fps=args.fps)
        print(f"Saved: time_fixed{suffix}.mp4 (angle={angle_idx}/{V})")

    # === Output 3: Rotating with time ===
    rotating_frames = []
    for t in range(T):
        angle = (t * V // T) % V
        rotating_frames.append(all_turntables[t][angle])
    rotating_frames = np.stack(rotating_frames)
    imageseq2video(rotating_frames, str(output_dir / "time_rotating.mp4"), fps=args.fps)
    print("Saved: time_rotating.mp4")

    # === Output 4: Full (all time × all angles) ===
    full_frames = np.concatenate(all_turntables, axis=0)
    imageseq2video(full_frames, str(output_dir / "full_all.mp4"), fps=args.fps)
    print(f"Saved: full_all.mp4 ({T}x{V}={T*V} frames)")

    # === Output 5: Grid image (first frame) with angle labels ===
    first = all_turntables[0]
    cols = 6
    rows = (V + cols - 1) // cols
    pad_count = rows * cols - V
    if pad_count > 0:
        padding = np.zeros((pad_count, H, W, 3), dtype=first.dtype)
        first = np.concatenate([first, padding], axis=0)
    grid = first.reshape(rows, cols, H, W, 3)
    grid = grid.transpose(0, 2, 1, 3, 4).reshape(rows * H, cols * W, 3)
    
    # Add angle labels to grid
    grid_labeled = add_turntable_row_labels(
        grid, 
        num_views=V,
        grid_rows=rows,
        grid_cols=cols,
        row_height=H,
        elevation=args.elevation,
    )
    Image.fromarray(grid_labeled).save(str(output_dir / "grid_first.jpg"))
    print(f"Saved: grid_first.jpg ({rows}x{cols} grid with angle labels)")

    # === Output 6: 6-camera input grid (first frame image) ===
    num_cams = all_input_views[0].shape[0]
    input_grid_cols = 3
    input_grid_rows = (num_cams + input_grid_cols - 1) // input_grid_cols
    ih, iw = all_input_views[0].shape[1:3]
    
    # Create input grid image (first frame)
    input_views = all_input_views[0]
    pad_n = input_grid_rows * input_grid_cols - num_cams
    if pad_n > 0:
        input_views = np.concatenate(
            [input_views, np.ones((pad_n, ih, iw, 3), dtype=input_views.dtype) * 255], axis=0
        )
    input_grid = input_views.reshape(input_grid_rows, input_grid_cols, ih, iw, 3)
    input_grid = input_grid.transpose(0, 2, 1, 3, 4).reshape(input_grid_rows * ih, input_grid_cols * iw, 3)
    
    # Add camera labels
    input_grid_labeled = add_input_view_labels(
        input_grid,
        num_cams=num_cams,
        grid_rows=input_grid_rows,
        grid_cols=input_grid_cols,
        row_height=ih,
    )
    Image.fromarray(input_grid_labeled).save(str(output_dir / "grid_input.jpg"))
    print(f"Saved: grid_input.jpg ({input_grid_rows}x{input_grid_cols} input cameras)")

    # === Output 7: 6-camera input grid video ===
    grid_video_frames = []
    for t in range(len(all_input_views)):
        views = all_input_views[t]
        pad_n = input_grid_rows * input_grid_cols - num_cams
        if pad_n > 0:
            views = np.concatenate(
                [views, np.ones((pad_n, ih, iw, 3), dtype=views.dtype) * 255], axis=0
            )
        grid = views.reshape(input_grid_rows, input_grid_cols, ih, iw, 3)
        grid = grid.transpose(0, 2, 1, 3, 4).reshape(input_grid_rows * ih, input_grid_cols * iw, 3)
        grid_video_frames.append(grid)

    grid_video_frames = np.stack(grid_video_frames)
    imageseq2video(grid_video_frames, str(output_dir / "grid_6view.mp4"), fps=args.fps)
    print(f"Saved: grid_6view.mp4 ({len(grid_video_frames)} frames, {input_grid_rows}x{input_grid_cols})")

    # Summary
    print(f"\n{'='*60}")
    print(f"Done! Output: {output_dir}")
    print(f"{'='*60}")
    print("Videos:")
    print(f"  - turntable_first.mp4  : First frame 360° rotation")
    print(f"  - time_fixed.mp4       : Fixed angle, time progression")
    print(f"  - time_rotating.mp4    : Rotating view with time")
    print(f"  - full_all.mp4         : All frames × all angles")
    print(f"  - grid_6view.mp4       : 6-camera input views")
    print(f"  - grid_first.jpg       : First frame turntable grid with angle labels")
    print(f"  - grid_input.jpg       : Input camera views grid")
    if args.save_gaussian:
        print("Gaussians:")
        print(f"  - gaussians/*.ply      : PLY files (GS viewer compatible)")
        print(f"  - gaussians/*.npz      : NPZ files (lightweight)")
    if args.save_rerun:
        print("Rerun:")
        print(f"  - rerun/*.rrd          : Interactive 3D viewer files")
        print(f"  View with: rerun {output_dir}/rerun/sequence.rrd")


if __name__ == "__main__":
    main()
