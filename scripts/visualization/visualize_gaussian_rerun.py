#!/usr/bin/env python3
"""
FaceLift Gaussian Visualization with Rerun.io

Each frame shows:
- 3D Gaussians point cloud
- N GT images (from original data)
- N Pred images (rendered from Gaussians)

Usage:
    python visualize_gaussian_rerun.py --exp-dir experiments/validation/D6_1_E4_5v_alpha \
        --data-root /home/joon/data/preprocessed/FaceLift_mouse/D6-1/val \
        --start 1 --end 2000 --step 200 --save
"""
import argparse
import numpy as np
import re
from pathlib import Path
from typing import Optional, List, Dict

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

    # Position (normalized to ~[-1, 1])
    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)

    # Color from SH (f_dc coefficients)
    if "f_dc_0" in vertex.data.dtype.names:
        f_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)
        C0 = 0.28209479177387814
        colors = (0.5 + C0 * f_dc).clip(0, 1)
        colors = (colors * 255).astype(np.uint8)
    elif "red" in vertex.data.dtype.names:
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
    else:
        colors = np.ones((len(means), 3), dtype=np.uint8) * 128

    # Opacity: stored as logit -> sigmoid
    if "opacity" in vertex.data.dtype.names:
        opacity_logit = vertex["opacity"].astype(np.float32)
        opacities = 1.0 / (1.0 + np.exp(-opacity_logit))
    else:
        opacities = np.ones(len(means), dtype=np.float32)

    # Scale: stored as log -> exp
    if "scale_0" in vertex.data.dtype.names:
        scale_log = np.stack([
            vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]
        ], axis=1).astype(np.float32)
        scales = np.exp(scale_log)
    else:
        scales = np.ones((len(means), 3), dtype=np.float32) * 0.01

    return {
        "means": means,
        "colors": colors,
        "opacities": opacities,
        "scales": scales,
    }


def load_gt_images(data_root: Path, sample_id: str, num_views: int = 6) -> List[np.ndarray]:
    """Load GT images from original data path."""
    images = []
    # sample_id format: "00000044" -> folder name "000044"
    folder_name = sample_id.lstrip("0").zfill(6) if sample_id.isdigit() else sample_id
    
    # Try different folder name formats
    possible_folders = [
        data_root / folder_name,
        data_root / sample_id,
        data_root / sample_id.lstrip("0").zfill(6),
    ]
    
    sample_dir = None
    for folder in possible_folders:
        if folder.exists():
            sample_dir = folder
            break
    
    if sample_dir is None:
        print(f"    Warning: GT folder not found for {sample_id}")
        return images
    
    images_dir = sample_dir / "images"
    if not images_dir.exists():
        print(f"    Warning: images folder not found in {sample_dir}")
        return images
    
    for i in range(num_views):
        img_path = images_dir / f"cam_{i:03d}.png"
        if img_path.exists():
            img = np.array(Image.open(img_path))
            # Remove alpha channel if present
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            images.append(img)
        else:
            print(f"    Warning: {img_path.name} not found")
    
    return images


def parse_pred_images(gt_vs_pred_path: Path, num_views: int) -> List[np.ndarray]:
    """Parse Pred images from gt_vs_pred.png (row 1 = Pred RGB)."""
    if not PIL_AVAILABLE or not gt_vs_pred_path.exists():
        return []
    
    img = np.array(Image.open(gt_vs_pred_path))
    h, w = img.shape[:2]
    
    num_rows = 5  # GT, Pred, GT+Mask, Pred+Mask, Error
    view_h = h // num_rows
    view_w = w // num_views
    
    pred_images = []
    row_idx = 1  # Pred RGB is row 1
    
    for view_idx in range(num_views):
        y1, y2 = row_idx * view_h, (row_idx + 1) * view_h
        x1, x2 = view_idx * view_w, (view_idx + 1) * view_w
        pred_images.append(img[y1:y2, x1:x2])
    
    return pred_images


def find_iterations(exp_dir: Path, start: int, end: Optional[int], step: int) -> List[Dict]:
    """Find all iterations in experiment directory."""
    results = []
    
    iter_dirs = sorted([
        d for d in exp_dir.iterdir()
        if d.is_dir() and d.name.startswith("iter_")
    ], key=lambda x: int(re.search(r"iter_(\d+)", x.name).group(1)))
    
    for iter_dir in iter_dirs:
        iteration = int(re.search(r"iter_(\d+)", iter_dir.name).group(1))
        
        if iteration < start:
            continue
        if end is not None and iteration > end:
            continue
        if (iteration - start) % step != 0:
            continue
        
        sample_dirs = sorted([d for d in iter_dir.iterdir() if d.is_dir()])
        if not sample_dirs:
            continue
        
        sample_dir = sample_dirs[0]
        ply_path = sample_dir / "gaussians.ply"
        gt_vs_pred_path = sample_dir / "gt_vs_pred.png"
        
        if ply_path.exists():
            results.append({
                "iteration": iteration,
                "ply_path": ply_path,
                "gt_vs_pred_path": gt_vs_pred_path if gt_vs_pred_path.exists() else None,
                "sample_id": sample_dir.name,
            })
    
    return results


def setup_blueprint(num_views: int):
    """Create Rerun blueprint: 3D + GT row + Pred row."""
    if not BLUEPRINT_AVAILABLE:
        return
    
    gt_views = [
        rrb.Spatial2DView(name=f"GT_{i}", origin=f"views/gt/{i}")
        for i in range(num_views)
    ]
    
    pred_views = [
        rrb.Spatial2DView(name=f"Pred_{i}", origin=f"views/pred/{i}")
        for i in range(num_views)
    ]
    
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="3D Gaussians", origin="world"),
            ),
            rrb.Horizontal(*gt_views),
            rrb.Horizontal(*pred_views),
            row_shares=[2, 1, 1],
        ),
        rrb.TimePanel(state="expanded"),
    )
    
    try:
        rr.send_blueprint(blueprint)
    except Exception as e:
        print(f"Warning: Blueprint error: {e}")


def visualize_experiment(
    exp_dir: str,
    data_root: Optional[str] = None,
    start: int = 0,
    end: Optional[int] = None,
    step: int = 1,
    opacity_threshold: float = 0.5,
    point_radius: float = 0.003,
    num_views: int = 6,
    save_path: Optional[str] = None,
):
    """Main visualization function."""
    exp_path = Path(exp_dir)
    exp_name = exp_path.name
    data_root_path = Path(data_root) if data_root else None
    
    print("=" * 60)
    print("FaceLift Gaussian Visualization")
    print("=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"Data root: {data_root}")
    print(f"Iteration range: {start} - {end} (step={step})")
    print(f"Opacity threshold: {opacity_threshold}")
    print(f"Point radius: {point_radius}")
    print(f"Num views: {num_views}")
    print()
    
    iterations = find_iterations(exp_path, start, end, step)
    if not iterations:
        print("Error: No iterations found")
        return
    
    print(f"Found {len(iterations)} frames")
    print()
    
    rr.init(f"FaceLift: {exp_name}", recording_id=f"{exp_name}")
    setup_blueprint(num_views)
    
    for frame_idx, item in enumerate(iterations):
        iteration = item["iteration"]
        ply_path = item["ply_path"]
        gt_vs_pred_path = item["gt_vs_pred_path"]
        sample_id = item["sample_id"]
        
        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("iteration", sequence=iteration)
        
        # Load Gaussians
        try:
            data = load_gaussian_ply(str(ply_path))
            means = data["means"]
            colors = data["colors"]
            opacities = data["opacities"]
            
            # Filter by opacity
            mask = opacities > opacity_threshold
            means_vis = means[mask]
            colors_vis = colors[mask]
            opacities_vis = opacities[mask]
            
            # RGBA with opacity
            alphas = (opacities_vis * 255).clip(0, 255).astype(np.uint8)
            colors_rgba = np.concatenate([colors_vis, alphas[:, None]], axis=1)
            
            # Fixed small radius for clean visualization
            radii = np.full(len(means_vis), point_radius)
            
            rr.log(
                "world/gaussians",
                rr.Points3D(positions=means_vis, colors=colors_rgba, radii=radii)
            )
            
            visible = len(means_vis)
            total = len(means)
            pct = 100 * visible / total if total > 0 else 0
            
            print(f"Frame {frame_idx:2d} (iter {iteration:5d}) | {visible:5d}/{total} ({pct:.1f}%) gaussians", end="")
            
        except Exception as e:
            print(f"Frame {frame_idx:2d} (iter {iteration:5d}) | Error: {e}")
            continue
        
        # Load GT images from original data
        gt_count = 0
        if data_root_path:
            gt_images = load_gt_images(data_root_path, sample_id, num_views)
            for i, img in enumerate(gt_images):
                rr.log(f"views/gt/{i}", rr.Image(img))
                gt_count += 1
        
        # Load Pred images from gt_vs_pred.png
        pred_count = 0
        if gt_vs_pred_path:
            pred_images = parse_pred_images(gt_vs_pred_path, num_views)
            for i, img in enumerate(pred_images):
                rr.log(f"views/pred/{i}", rr.Image(img))
                pred_count += 1
        
        print(f" | GT:{gt_count} Pred:{pred_count}")
    
    if save_path:
        rr.save(save_path)
        print(f"\nSaved to: {save_path}")
    
    print(f"\nDone! View with: rerun {save_path or 'file.rrd'}")


def main():
    parser = argparse.ArgumentParser(description="FaceLift Gaussian Rerun Visualization")
    
    parser.add_argument("--exp-dir", required=True, help="Experiment validation directory")
    parser.add_argument("--data-root", type=str, help="Root path to GT images (e.g., .../D6-1/val)")
    parser.add_argument("--start", type=int, default=0, help="Start iteration")
    parser.add_argument("--end", type=int, default=None, help="End iteration")
    parser.add_argument("--step", type=int, default=1, help="Iteration step")
    
    parser.add_argument("--opacity-threshold", type=float, default=0.5, help="Min opacity (default: 0.5)")
    parser.add_argument("--point-radius", type=float, default=0.003, help="Point radius for visualization")
    parser.add_argument("--num-views", type=int, default=6, help="Number of views")
    
    parser.add_argument("--save", action="store_true", help="Save to .rrd file")
    parser.add_argument("-o", "--output", type=str, help="Output path")
    
    args = parser.parse_args()
    
    save_path = None
    if args.save or args.output:
        save_path = args.output or f"{Path(args.exp_dir).name}_viz.rrd"
    
    visualize_experiment(
        exp_dir=args.exp_dir,
        data_root=args.data_root,
        start=args.start,
        end=args.end,
        step=args.step,
        opacity_threshold=args.opacity_threshold,
        point_radius=args.point_radius,
        num_views=args.num_views,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
