#!/usr/bin/env python3
"""
Mouse 4D Reconstruction Inference Script

Fine-tuned GS-LRM 모델을 사용하여:
1. 연속 프레임에서 프레임별 3D Gaussian 생성
2. 특정 뷰(또는 여러 뷰)에서 렌더링
3. 4D 영상으로 저장

Usage:
    python inference_mouse_4d.py \
        --data_dir data_mouse/frame_*/  \
        --checkpoint checkpoints/gslrm/exp_c/ckpt_latest.pt \
        --output_dir outputs/mouse_4d \
        --render_views 0 1 2 3 4 5 \
        --num_frames 100
"""

import os
import sys
import argparse
import glob
import yaml
import importlib
from typing import List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from easydict import EasyDict as edict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gslrm.model.gaussians_renderer import render_opencv_cam, imageseq2video
from gslrm.model.utils_3dgs import GaussianModel


def load_gslrm_model(config_path: str, checkpoint_path: str, device: torch.device):
    """Load fine-tuned GS-LRM model."""
    with open(config_path, 'r') as f:
        config = edict(yaml.safe_load(f))

    # Import model class
    module_name, class_name = config.model.class_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    # Create model
    model = model_class(config.model)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle DDP wrapper
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    print(f"Model loaded successfully")
    return model, config


def load_frame_data(frame_dir: str, num_views: int = 6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load images and camera data for a single frame.

    Returns:
        images: [num_views, 3, H, W]
        c2ws: [num_views, 4, 4] - camera to world matrices
        intrinsics: [num_views, 4] - fx, fy, cx, cy
    """
    # Load images
    images = []
    for view_idx in range(num_views):
        img_path = os.path.join(frame_dir, 'images', f'{view_idx:02d}.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(frame_dir, 'images', f'{view_idx}.png')

        img = Image.open(img_path).convert('RGB')
        img = np.array(img).astype(np.float32) / 255.0
        images.append(img)

    images = np.stack(images, axis=0)  # [V, H, W, 3]
    images = torch.from_numpy(images).permute(0, 3, 1, 2)  # [V, 3, H, W]

    # Load cameras
    cameras_path = os.path.join(frame_dir, 'cameras.npz')
    if os.path.exists(cameras_path):
        cam_data = np.load(cameras_path)
        c2ws = torch.from_numpy(cam_data['c2w']).float()  # [V, 4, 4]
        intrinsics = torch.from_numpy(cam_data['intrinsics']).float()  # [V, 4]
    else:
        # Try individual camera files
        c2ws = []
        intrinsics = []
        for view_idx in range(num_views):
            cam_path = os.path.join(frame_dir, 'cameras', f'{view_idx:02d}.npz')
            cam = np.load(cam_path)
            c2ws.append(cam['c2w'])
            intrinsics.append(cam['intrinsics'])
        c2ws = torch.from_numpy(np.stack(c2ws, axis=0)).float()
        intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()

    return images, c2ws, intrinsics


def prepare_model_input(
    images: torch.Tensor,
    c2ws: torch.Tensor,
    intrinsics: torch.Tensor,
    input_view_indices: List[int],
    target_view_indices: List[int],
    device: torch.device
) -> dict:
    """
    Prepare input batch for GS-LRM model.

    Args:
        images: [num_views, 3, H, W]
        c2ws: [num_views, 4, 4]
        intrinsics: [num_views, 4]
        input_view_indices: which views to use as input
        target_view_indices: which views to render
    """
    # Select input views
    input_images = images[input_view_indices]  # [num_input, 3, H, W]
    input_c2ws = c2ws[input_view_indices]
    input_intrinsics = intrinsics[input_view_indices]

    # Select target views
    target_c2ws = c2ws[target_view_indices]
    target_intrinsics = intrinsics[target_view_indices]

    # Add batch dimension
    batch = {
        'input_images': input_images.unsqueeze(0).to(device),  # [1, num_input, 3, H, W]
        'input_c2ws': input_c2ws.unsqueeze(0).to(device),
        'input_intrinsics': input_intrinsics.unsqueeze(0).to(device),
        'target_c2ws': target_c2ws.unsqueeze(0).to(device),
        'target_intrinsics': target_intrinsics.unsqueeze(0).to(device),
    }

    return batch


@torch.no_grad()
def run_inference(
    model,
    images: torch.Tensor,
    c2ws: torch.Tensor,
    intrinsics: torch.Tensor,
    input_view_indices: List[int],
    render_view_idx: int,
    device: torch.device,
    image_size: int = 512
) -> np.ndarray:
    """
    Run GS-LRM inference and render from specified view.

    Returns:
        rendered_image: [H, W, 3] numpy array
    """
    # Prepare input
    target_view_indices = [render_view_idx]

    # Resize images if needed
    _, _, h, w = images.shape
    if h != image_size or w != image_size:
        import torch.nn.functional as F
        images = F.interpolate(images, size=(image_size, image_size), mode='bilinear', align_corners=False)

    # Model forward pass
    input_images = images[input_view_indices].unsqueeze(0).to(device)
    input_c2ws = c2ws[input_view_indices].unsqueeze(0).to(device)
    input_intrinsics = intrinsics[input_view_indices].unsqueeze(0).to(device)
    target_c2ws = c2ws[target_view_indices].unsqueeze(0).to(device)
    target_intrinsics = intrinsics[target_view_indices].unsqueeze(0).to(device)

    # Run model to get Gaussian parameters
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(
            input_images=input_images,
            input_c2ws=input_c2ws,
            input_Ks=input_intrinsics,
            target_c2ws=target_c2ws,
            target_Ks=target_intrinsics,
        )

    # Create Gaussian model and render
    gaussians = outputs.gaussians  # GaussianModel or similar

    # Render from target view
    rendered = render_opencv_cam(
        pc=gaussians,
        height=image_size,
        width=image_size,
        C2W=target_c2ws[0, 0],  # [4, 4]
        fxfycxcy=target_intrinsics[0, 0],  # [4]
        bg_color=(1.0, 1.0, 1.0)
    )

    # Convert to numpy
    rendered_image = rendered['image'].permute(1, 2, 0).cpu().numpy()
    rendered_image = (rendered_image * 255).clip(0, 255).astype(np.uint8)

    return rendered_image


def main():
    parser = argparse.ArgumentParser(description='Mouse 4D Reconstruction')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing frame_* subdirectories')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to fine-tuned GS-LRM checkpoint')
    parser.add_argument('--config', type=str, default='configs/mouse_gslrm_exp_c.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs/mouse_4d',
                        help='Output directory')
    parser.add_argument('--render_views', type=int, nargs='+', default=[0],
                        help='View indices to render (0-5)')
    parser.add_argument('--input_views', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Input view indices (default: 0-4, leaving one for eval)')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='Number of frames to process (default: all)')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Starting frame index')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output video FPS')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image size for processing')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, config = load_gslrm_model(args.config, args.checkpoint, device)

    # Find all frames
    frame_dirs = sorted(glob.glob(os.path.join(args.data_dir, 'frame_*')))
    if not frame_dirs:
        # Try direct frame format
        frame_dirs = sorted(glob.glob(os.path.join(args.data_dir, '*')))
        frame_dirs = [d for d in frame_dirs if os.path.isdir(d)]

    print(f"Found {len(frame_dirs)} frames")

    # Limit frames if specified
    if args.num_frames:
        frame_dirs = frame_dirs[args.start_frame:args.start_frame + args.num_frames]
    else:
        frame_dirs = frame_dirs[args.start_frame:]

    print(f"Processing {len(frame_dirs)} frames from index {args.start_frame}")

    # Process each render view separately
    for render_view in args.render_views:
        print(f"\n=== Rendering View {render_view} ===")

        view_output_dir = os.path.join(args.output_dir, f'view_{render_view:02d}')
        os.makedirs(view_output_dir, exist_ok=True)

        rendered_frames = []

        for frame_idx, frame_dir in enumerate(tqdm(frame_dirs, desc=f"View {render_view}")):
            try:
                # Load frame data
                images, c2ws, intrinsics = load_frame_data(frame_dir, num_views=6)

                # Run inference
                rendered = run_inference(
                    model=model,
                    images=images,
                    c2ws=c2ws,
                    intrinsics=intrinsics,
                    input_view_indices=args.input_views,
                    render_view_idx=render_view,
                    device=device,
                    image_size=args.image_size
                )

                # Save individual frame
                frame_path = os.path.join(view_output_dir, f'frame_{frame_idx:06d}.png')
                Image.fromarray(rendered).save(frame_path)

                rendered_frames.append(rendered)

            except Exception as e:
                print(f"Error processing {frame_dir}: {e}")
                continue

        # Create video
        if rendered_frames:
            video_path = os.path.join(args.output_dir, f'view_{render_view:02d}.mp4')
            imageseq2video(rendered_frames, video_path, fps=args.fps)
            print(f"Saved video: {video_path}")

    # Create combined video with all views side by side
    if len(args.render_views) > 1:
        print("\n=== Creating combined multi-view video ===")
        create_multiview_video(args.output_dir, args.render_views, args.fps)

    print(f"\nDone! Outputs saved to: {args.output_dir}")


def create_multiview_video(output_dir: str, render_views: List[int], fps: int):
    """Create a combined video showing all rendered views side by side."""
    import cv2

    # Get frame count from first view
    first_view_dir = os.path.join(output_dir, f'view_{render_views[0]:02d}')
    frame_files = sorted(glob.glob(os.path.join(first_view_dir, 'frame_*.png')))

    if not frame_files:
        print("No frames found for combined video")
        return

    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    h, w = first_frame.shape[:2]

    # Calculate grid layout
    n_views = len(render_views)
    if n_views <= 3:
        grid_cols, grid_rows = n_views, 1
    elif n_views <= 6:
        grid_cols, grid_rows = 3, 2
    else:
        grid_cols = int(np.ceil(np.sqrt(n_views)))
        grid_rows = int(np.ceil(n_views / grid_cols))

    # Create video writer
    combined_path = os.path.join(output_dir, 'combined_views.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(combined_path, fourcc, fps, (w * grid_cols, h * grid_rows))

    for frame_idx in tqdm(range(len(frame_files)), desc="Combining views"):
        grid = np.zeros((h * grid_rows, w * grid_cols, 3), dtype=np.uint8)

        for view_idx, render_view in enumerate(render_views):
            view_dir = os.path.join(output_dir, f'view_{render_view:02d}')
            frame_path = os.path.join(view_dir, f'frame_{frame_idx:06d}.png')

            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                row = view_idx // grid_cols
                col = view_idx % grid_cols
                grid[row*h:(row+1)*h, col*w:(col+1)*w] = frame

        out.write(grid)

    out.release()
    print(f"Saved combined video: {combined_path}")


if __name__ == '__main__':
    main()
