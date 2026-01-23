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


def load_model(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Load GS-LRM model."""
    from gslrm.model.gslrm import GSLRM

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

    images = torch.from_numpy(images).float().to(device)
    images = rearrange(images, "v h w c -> v c h w")
    
    return edict({
        "image": images.unsqueeze(0),
        "c2w": torch.from_numpy(c2ws).float().to(device).unsqueeze(0),
        "fxfycxcy": torch.from_numpy(fxfycxcy).float().to(device).unsqueeze(0),
        "index": torch.zeros(len(frames), 3, dtype=torch.long, device=device).unsqueeze(0),
    })


@torch.no_grad()
def process_frame(model, sample_dir: Path, device: str, resolution: int, 
                  num_views: int, elevation: float, radius: float) -> np.ndarray:
    """단일 프레임 처리 → turntable 프레임들 반환."""
    
    sample = load_sample(sample_dir, device)
    
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
    
    return frames


def main():
    parser = argparse.ArgumentParser(description="Simple Temporal Video")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=30)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=36, help="360도 분할 수")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--radius", type=float, default=2.7)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--output_dir", type=str, default="outputs/simple_temporal")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config = load_model(args.checkpoint, args.config, "cuda")
    data_dir = Path(args.data_dir)
    
    frame_indices = list(range(args.start_frame, args.end_frame, args.frame_step))
    print(f"Processing {len(frame_indices)} frames")

    all_turntables = []  # [T][V, H, W, C]
    
    for frame_idx in tqdm(frame_indices, desc="Frames"):
        sample_dir = data_dir / f"{frame_idx:06d}"
        if not sample_dir.exists():
            print(f"Skip: {sample_dir}")
            continue
        
        frames = process_frame(
            model, sample_dir, "cuda",
            args.resolution, args.num_views, args.elevation, args.radius
        )
        all_turntables.append(frames)

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

    # === 출력 2: 고정 각도, 시간 변화 ===
    fixed_angle_frames = np.stack([t[0] for t in all_turntables])  # [T, H, W, C]
    imageseq2video(fixed_angle_frames, str(output_dir / "time_fixed_angle.mp4"), fps=args.fps)
    print("Saved: time_fixed_angle.mp4")

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

    print(f"\nDone! Output: {output_dir}")


if __name__ == "__main__":
    main()
