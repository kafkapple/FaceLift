#!/usr/bin/env python3
"""
FaceLift Frame-based Visualization with Rerun.io

Timeline = actual video frame sequence (not training iteration)
Each frame shows:
- 3D Gaussians
- 6 GT images (from original data)
- 6 Pred images (rendered from Gaussians)

Usage:
    python visualize_frames_rerun.py \
        --inference-dir outputs/sequential_frames \
        --data-root /home/joon/data/preprocessed/FaceLift_mouse/D6-1/val \
        --start 0 --end 20 --save
"""
import argparse
import numpy as np
import re
from pathlib import Path
from typing import Optional, List

try:
    import rerun as rr
    import rerun.blueprint as rrb
    BLUEPRINT_AVAILABLE = True
except ImportError:
    try:
        import rerun as rr
        BLUEPRINT_AVAILABLE = False
    except:
        print("Error: pip install rerun-sdk")
        exit(1)

try:
    from plyfile import PlyData
except ImportError:
    print("Error: pip install plyfile")
    exit(1)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available")


def load_gaussian_ply(ply_path: str) -> dict:
    """Load 3DGS PLY with correct attribute transformations."""
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)

    if "f_dc_0" in vertex.data.dtype.names:
        f_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)
        C0 = 0.28209479177387814
        colors = (0.5 + C0 * f_dc).clip(0, 1)
        colors = (colors * 255).astype(np.uint8)
    elif "red" in vertex.data.dtype.names:
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
    else:
        colors = np.ones((len(means), 3), dtype=np.uint8) * 128

    if "opacity" in vertex.data.dtype.names:
        opacity_logit = vertex["opacity"].astype(np.float32)
        opacities = 1.0 / (1.0 + np.exp(-opacity_logit))
    else:
        opacities = np.ones(len(means), dtype=np.float32)

    return {"means": means, "colors": colors, "opacities": opacities}


def load_gt_images(data_root: Path, frame_id: str, num_views: int = 6) -> List[np.ndarray]:
    """Load GT images from original data."""
    images = []
    sample_dir = data_root / frame_id
    images_dir = sample_dir / "images"
    
    if not images_dir.exists():
        return images
    
    for i in range(num_views):
        img_path = images_dir / f"cam_{i:03d}.png"
        if img_path.exists():
            img = np.array(Image.open(img_path))
            if len(img.shape) == 3 and img.shape[-1] == 4:
                img = img[:, :, :3]
            images.append(img)
    
    return images


def load_pred_images(inference_dir: Path, frame_id: str, num_views: int = 6) -> List[np.ndarray]:
    """Load rendered images from inference output."""
    images = []
    frame_dir = inference_dir / frame_id
    
    if not frame_dir.exists():
        return images
    
    for i in range(num_views):
        img_path = frame_dir / f"render_view_{i:02d}.png"
        if img_path.exists():
            img = np.array(Image.open(img_path))
            if len(img.shape) == 3 and img.shape[-1] == 4:
                img = img[:, :, :3]
            images.append(img)
    
    return images


def find_frames(inference_dir: Path, start: int, end: Optional[int]) -> List[str]:
    """Find frame directories in order."""
    frames = sorted([
        d.name for d in inference_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    ])
    
    result = []
    for f in frames:
        idx = int(f)
        if idx < start:
            continue
        if end is not None and idx >= end:
            continue
        result.append(f)
    
    return result


def setup_blueprint(num_views: int = 6):
    """Create Rerun blueprint for 3D + GT + Pred layout."""
    if not BLUEPRINT_AVAILABLE:
        return
    
    gt_views = [rrb.Spatial2DView(name=f"GT_{i}", origin=f"views/gt/{i}") for i in range(num_views)]
    pred_views = [rrb.Spatial2DView(name=f"Pred_{i}", origin=f"views/pred/{i}") for i in range(num_views)]
    
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(rrb.Spatial3DView(name="3D Gaussians", origin="world")),
            rrb.Horizontal(*gt_views),
            rrb.Horizontal(*pred_views),
            row_shares=[2, 1, 1],
        ),
        rrb.TimePanel(state="expanded"),
    )
    
    try:
        rr.send_blueprint(blueprint)
    except Exception as e:
        print(f"Blueprint error: {e}")


def main():
    parser = argparse.ArgumentParser(description="FaceLift Frame-based Rerun Visualization")
    
    parser.add_argument("--inference-dir", required=True, help="Inference output directory")
    parser.add_argument("--data-root", required=True, help="Original data root (e.g., .../val)")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=None, help="End frame index (exclusive)")
    parser.add_argument("--opacity-threshold", type=float, default=0.5, help="Opacity threshold")
    parser.add_argument("--point-radius", type=float, default=0.003, help="Point radius")
    parser.add_argument("--num-views", type=int, default=6, help="Number of views")
    parser.add_argument("--save", action="store_true", help="Save to .rrd")
    parser.add_argument("-o", "--output", type=str, help="Output path")
    
    args = parser.parse_args()
    
    inference_dir = Path(args.inference_dir)
    data_root = Path(args.data_root)
    
    print("=" * 60)
    print("FaceLift Frame-based Visualization")
    print("=" * 60)
    print(f"Inference dir: {inference_dir}")
    print(f"Data root: {data_root}")
    print(f"Frame range: {args.start} - {args.end}")
    print()
    
    frames = find_frames(inference_dir, args.start, args.end)
    if not frames:
        print("Error: No frames found")
        return
    
    print(f"Found {len(frames)} frames: {frames[0]} -> {frames[-1]}")
    print()
    
    save_path = args.output or (f"frames_{frames[0]}-{frames[-1]}.rrd" if args.save else None)
    
    rr.init("FaceLift Frames", recording_id="frames_viz")
    setup_blueprint(args.num_views)
    
    for frame_idx, frame_id in enumerate(frames):
        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("video_frame", sequence=int(frame_id))
        
        # Load Gaussians
        ply_path = inference_dir / frame_id / "gaussians.ply"
        if ply_path.exists():
            try:
                data = load_gaussian_ply(str(ply_path))
                mask = data["opacities"] > args.opacity_threshold
                means = data["means"][mask]
                colors = data["colors"][mask]
                opacities = data["opacities"][mask]
                
                alphas = (opacities * 255).clip(0, 255).astype(np.uint8)
                colors_rgba = np.concatenate([colors, alphas[:, None]], axis=1)
                radii = np.full(len(means), args.point_radius)
                
                rr.log("world/gaussians", rr.Points3D(positions=means, colors=colors_rgba, radii=radii))
                
                print(f"Frame {frame_id}: {len(means)} gaussians", end="")
            except Exception as e:
                print(f"Frame {frame_id}: PLY error - {e}")
                continue
        else:
            print(f"Frame {frame_id}: No PLY")
            continue
        
        # Load GT images
        gt_images = load_gt_images(data_root, frame_id, args.num_views)
        for i, img in enumerate(gt_images):
            rr.log(f"views/gt/{i}", rr.Image(img))
        
        # Load Pred images
        pred_images = load_pred_images(inference_dir, frame_id, args.num_views)
        for i, img in enumerate(pred_images):
            rr.log(f"views/pred/{i}", rr.Image(img))
        
        print(f" | GT:{len(gt_images)} Pred:{len(pred_images)}")
    
    if save_path:
        rr.save(save_path)
        print(f"\nSaved to: {save_path}")
    
    print(f"\nDone! View with: rerun {save_path or 'file.rrd'}")


if __name__ == "__main__":
    main()
