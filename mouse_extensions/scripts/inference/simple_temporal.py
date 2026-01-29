#!/usr/bin/env python3
"""
Simple Temporal Video - 검증된 render_turntable 사용

여러 프레임의 turntable을 순서대로 이어붙여 영상 생성.
복잡한 처리 없이 기존 검증된 코드만 사용.
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
from tqdm import tqdm
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gslrm.model.gaussians_renderer import render_turntable, imageseq2video


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
        # Find best.pt first
        best_pt = search_dir / "best.pt"
        if best_pt.exists():
            print(f"Using best checkpoint: {best_pt}")
            return str(best_pt)

        # Find all ckpt_*.pt files and get the latest
        pt_files = list(search_dir.glob("ckpt_*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No checkpoint files found in {search_dir}")

        def extract_step(p):
            match = re.search(r'ckpt_(\d+)\.pt', p.name)
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
    """Load sample - 최소한의 전처리만."""
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
        fxfycxcy_list.append([frame_info["fx"], frame_info["fy"], 
                             frame_info["cx"], frame_info["cy"]])

    images = np.stack(images)
    c2ws = np.array(c2ws)
    fxfycxcy = np.array(fxfycxcy_list)

    # 정규화
    c2ws = normalize_cameras_to_z_up(c2ws)
    c2ws, fxfycxcy = normalize_camera_distance_with_intrinsics(c2ws, fxfycxcy, 2.7)

    # Keep raw images (uint8, HWC, RGB) for input grid visualization
    raw_images = (images * 255).clip(0, 255).astype(np.uint8)  # [V, H, W, C]

    images = torch.from_numpy(images).float().to(device)
    images = rearrange(images, "v h w c -> v c h w")
    
    return edict({
        "image": images.unsqueeze(0),
        "c2w": torch.from_numpy(c2ws).float().to(device).unsqueeze(0),
        "fxfycxcy": torch.from_numpy(fxfycxcy).float().to(device).unsqueeze(0),
        "index": torch.zeros(len(frames), 3, dtype=torch.long, device=device).unsqueeze(0),
    }), raw_images


@torch.no_grad()
def process_frame(model, sample_dir: Path, device: str, resolution: int, 
                  num_views: int, elevation: float, radius: float) -> np.ndarray:
    """단일 프레임 처리 → turntable 프레임들 반환."""
    
    sample, raw_input_images = load_sample(sample_dir, device)
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(sample, create_visual=False, split_data=False)
    
    gaussians = output.gaussians[0]
    
    # 검증된 render_turntable 사용
    turntable_strip = render_turntable(
        gaussians,
        rendering_resolution=resolution,
        num_views=num_views,
        elevation=elevation,
        radius=radius,
        trajectory_mode="turntable"
    )
    # [H, V*W, C] → [V, H, W, C]
    h = turntable_strip.shape[0]
    w = turntable_strip.shape[1] // num_views
    frames = turntable_strip.reshape(h, num_views, w, 3)
    frames = np.transpose(frames, (1, 0, 2, 3))  # [V, H, W, C]
    
    del sample, output, gaussians
    gc.collect()
    torch.cuda.empty_cache()
    
    return frames, raw_input_images


def find_sample_dirs(data_dir: Path) -> list:
    """Auto-discover sample directories (numeric folder names), sorted."""
    dirs = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and d.name.isdigit():
            dirs.append((int(d.name), d))
    return dirs


def main():
    parser = argparse.ArgumentParser(description="Simple Temporal Video")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--start_frame", type=int, default=None,
                        help="Start frame index (default: auto-detect)")
    parser.add_argument("--end_frame", type=int, default=None,
                        help="End frame index (default: auto-detect)")
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=36, help="360-degree subdivisions")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--radius", type=float, default=2.7)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--fixed_angles", type=int, nargs="+", default=[0],
                        help="View angles for fixed-angle videos (default: [0])")
    parser.add_argument("--split", type=str, default=None,
                        help="Path to split file (e.g. data_mouse_val.txt). Overrides start/end/step.")
    parser.add_argument("--output_dir", type=str, default="outputs/simple_temporal")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config = load_model(args.checkpoint, args.config, "cuda")
    data_dir = Path(args.data_dir)

    # Load frame indices from split file or auto-discover
    if args.split:
        split_path = Path(args.split)
        if not split_path.exists():
            split_path = data_dir / args.split
        with open(split_path) as f:
            lines = [l.strip().rstrip("/") for l in f if l.strip()]
        frame_indices = sorted([int(Path(l).name) for l in lines])
        frame_indices = frame_indices[::args.frame_step]
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
        frame_indices = frame_indices[::args.frame_step]
        print(f"Auto-detected {len(all_indices)} samples, using {len(frame_indices)} "
              f"(range {start}-{end}, step {args.frame_step})")
    else:
        frame_indices = list(range(args.start_frame, args.end_frame, args.frame_step))
        print(f"Processing {len(frame_indices)} frames")

    all_turntables = []  # [T][V, H, W, C]
    all_input_views = []  # [T][V, H, W, C] raw input camera images
    
    for frame_idx in tqdm(frame_indices, desc="Frames"):
        sample_dir = data_dir / f"{frame_idx:06d}"
        if not sample_dir.exists():
            print(f"Skip: {sample_dir}")
            continue
        
        frames, raw_inputs = process_frame(
            model, sample_dir, "cuda",
            args.resolution, args.num_views, args.elevation, args.radius
        )
        all_turntables.append(frames)
        all_input_views.append(raw_inputs)

    if not all_turntables:
        print("No frames!")
        return

    T = len(all_turntables)
    V = all_turntables[0].shape[0]
    H, W = all_turntables[0].shape[1:3]
    print(f"Collected: T={T}, V={V}, H={H}, W={W}")

    # === 출력 1: 첫 프레임 360도 turntable ===
    imageseq2video(all_turntables[0], str(output_dir / "turntable_first.mp4"), fps=args.fps)
    print("Saved: turntable_first.mp4")

    # === 출력 2: 고정 각도, 시간 변화 (multi-angle) ===
    for angle_idx in args.fixed_angles:
        angle_idx = angle_idx % V
        fixed_frames = np.stack([t[angle_idx] for t in all_turntables])  # [T, H, W, C]
        suffix = f"_angle{angle_idx}" if len(args.fixed_angles) > 1 else ""
        imageseq2video(fixed_frames, str(output_dir / f"time_fixed{suffix}.mp4"), fps=args.fps)
        print(f"Saved: time_fixed{suffix}.mp4 (angle={angle_idx}/{V})")

    # === 출력 3: 시간에 따라 각도 회전 ===
    rotating_frames = []
    for t in range(T):
        angle = (t * V // T) % V
        rotating_frames.append(all_turntables[t][angle])
    rotating_frames = np.stack(rotating_frames)
    imageseq2video(rotating_frames, str(output_dir / "time_rotating.mp4"), fps=args.fps)
    print("Saved: time_rotating.mp4")

    # === 출력 4: 전체 (모든 시간 × 모든 각도) ===
    full_frames = np.concatenate(all_turntables, axis=0)  # [T*V, H, W, C]
    imageseq2video(full_frames, str(output_dir / "full_all.mp4"), fps=args.fps)
    print(f"Saved: full_all.mp4 ({T}x{V}={T*V} frames)")

    # === 출력 5: Grid 이미지 (첫 프레임) ===
    first = all_turntables[0]  # [V, H, W, C]
    cols = 6
    rows = (V + cols - 1) // cols
    # Pad if needed
    pad_count = rows * cols - V
    if pad_count > 0:
        padding = np.zeros((pad_count, H, W, 3), dtype=first.dtype)
        first = np.concatenate([first, padding], axis=0)
    grid = first.reshape(rows, cols, H, W, 3)
    grid = grid.transpose(0, 2, 1, 3, 4).reshape(rows * H, cols * W, 3)
    Image.fromarray(grid).save(str(output_dir / "grid_first.jpg"))
    print("Saved: grid_first.jpg")

    # === Output 6: 6-camera input grid video (2x3 layout) ===
    num_cams = all_input_views[0].shape[0]  # typically 6
    grid_cols = 3
    grid_rows = (num_cams + grid_cols - 1) // grid_cols  # 2 for 6 cameras
    ih, iw = all_input_views[0].shape[1:3]
    
    grid_video_frames = []
    for t in range(len(all_input_views)):
        views = all_input_views[t]  # [V, H, W, C]
        # Pad if num_cams < grid_rows * grid_cols
        pad_n = grid_rows * grid_cols - num_cams
        if pad_n > 0:
            views = np.concatenate([views, np.ones((pad_n, ih, iw, 3), dtype=views.dtype) * 255], axis=0)
        grid = views.reshape(grid_rows, grid_cols, ih, iw, 3)
        grid = grid.transpose(0, 2, 1, 3, 4).reshape(grid_rows * ih, grid_cols * iw, 3)
        grid_video_frames.append(grid)
    
    grid_video_frames = np.stack(grid_video_frames)  # [T, GH, GW, C]
    imageseq2video(grid_video_frames, str(output_dir / "grid_6view.mp4"), fps=args.fps)
    print(f"Saved: grid_6view.mp4 ({len(grid_video_frames)} frames, {grid_rows}x{grid_cols} layout)")

    print(f"\nDone! Output: {output_dir}")


if __name__ == "__main__":
    main()
