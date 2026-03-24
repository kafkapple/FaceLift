#!/usr/bin/env python3
"""
Multi-View Triangulation Evaluation with GS-LRM Novel Views.

For each test frame:
  1. GS-LRM predict() → Gaussians (1 forward pass)
  2. Generate N cameras via get_turntable_cameras()
  3. Render from each camera via render_opencv_cam()
  4. Project MAMMAL 3D GT → 2D on each rendered view
  5. Add Gaussian noise (σ = 0, 1, 2, 5 px) to simulate detection error
  6. Triangulate from N views via DLT
  7. Compare with MAMMAL GT → MPJPE

Also compares with Phase 1 DANNCE GT 6-view results to estimate
how much noise σ in DANNCE's actual detections.

Usage:
    # On gpu03 with CUDA
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.multiview_triangulation_eval \
        --config configs/mouse/base_uniform_v2.yaml \
        --checkpoint checkpoints/gslrm/6view_v2/best_psnr.pt \
        --data-dir ~/data/preprocessed/FaceLift_mouse/M5t2 \
        --mammal-3d <KP_22 from mouse_extensions.paths> \
        --output-dir outputs/triangulation_eval

Date: 2026-03-04
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def build_novel_view_cameras(
    num_views: int,
    elevation: float = 20.0,
    radius: float = 2.7,
    render_size: int = 384,
    center: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate turntable cameras and corresponding projection matrices.

    Uses GS-LRM's get_turntable_cameras() for consistency.

    Args:
        num_views: Number of views to generate
        elevation: Camera elevation in degrees
        radius: Distance from center
        render_size: Render width/height
        center: Orbit center point (default: origin)

    Returns:
        fxfycxcy: (N, 4) intrinsics
        c2ws: (N, 4, 4) camera-to-world matrices
    """
    from mouse_extensions.visualization import get_turntable_cameras

    w, h, nv, fxfycxcy, c2ws = get_turntable_cameras(
        hfov=50,
        num_views=num_views,
        w=render_size,
        h=render_size,
        radius=radius,
        elevation=elevation,
        trajectory_mode="turntable",
        clockwise=True,
        center=center,
    )
    return fxfycxcy, c2ws


def c2w_to_projection_matrix(
    c2w: np.ndarray,
    fxfycxcy: np.ndarray,
) -> np.ndarray:
    """Convert c2w + intrinsics to 3x4 projection matrix P = K @ [R|t].

    Args:
        c2w: (4, 4) camera-to-world matrix
        fxfycxcy: (4,) intrinsics [fx, fy, cx, cy]

    Returns:
        (3, 4) projection matrix
    """
    fx, fy, cx, cy = fxfycxcy
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)

    # w2c = inv(c2w)
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    t = w2c[:3, 3:]

    return K @ np.hstack([R, t])


def render_novel_views(
    gaussians,
    c2ws: np.ndarray,
    fxfycxcy: np.ndarray,
    render_size: int = 384,
    device: str = "cuda",
) -> List[np.ndarray]:
    """Render Gaussians from novel view cameras.

    Args:
        gaussians: GaussianModel from GS-LRM predict()
        c2ws: (N, 4, 4) camera poses
        fxfycxcy: (N, 4) intrinsics
        render_size: Output image size
        device: Torch device

    Returns:
        List of rendered images as numpy arrays (H, W, 3), float32 [0,1]
    """
    from mouse_extensions.visualization import render_opencv_cam

    rendered_images = []

    for i in range(len(c2ws)):
        c2w_t = torch.from_numpy(c2ws[i]).float().to(device)
        fxfycxcy_t = torch.from_numpy(fxfycxcy[i]).float().to(device)

        with torch.no_grad():
            result = render_opencv_cam(
                pc=gaussians,
                height=render_size,
                width=render_size,
                C2W=c2w_t,
                fxfycxcy=fxfycxcy_t,
                bg_color=(1.0, 1.0, 1.0),
            )

        # result["render"] is (3, H, W) or (1, H, W, 3)
        render = result["render"]
        if render.dim() == 4:
            render = render[0]  # (H, W, 3)
        if render.shape[0] == 3:
            render = render.permute(1, 2, 0)  # (H, W, 3)

        rendered_images.append(render.cpu().numpy())

    return rendered_images


def run_multiview_triangulation_eval(
    pipeline,
    data_dir: str,
    mammal_3d: np.ndarray,
    view_counts: List[int] = None,
    noise_levels: List[float] = None,
    frame_range: Tuple[int, int] = (3240, 3600),
    frame_step: int = 10,
    render_size: int = 384,
    device: str = "cuda",
) -> Dict:
    """Run the full multi-view triangulation evaluation.

    Args:
        pipeline: GSLRMInference instance
        data_dir: Path to preprocessed dataset (M5t2)
        mammal_3d: (3600, 22, 3) MAMMAL GT
        view_counts: [6, 9, 12, 24]
        noise_levels: [0, 1, 2, 5] in pixels
        frame_range: Test set frame range
        frame_step: Step between frames
        render_size: Rendering resolution
        device: Torch device

    Returns:
        Dict with results per view_count per noise_level
    """
    from mouse_extensions.analysis.triangulation_analysis import (
        triangulate_batch,
        compute_mpjpe,
        project_3d_to_2d,
    )
    from mouse_extensions.inference.gslrm_pipeline import load_sample_data

    if view_counts is None:
        view_counts = [6, 9, 12, 24]
    if noise_levels is None:
        noise_levels = [0.0, 1.0, 2.0, 5.0]

    max_views = max(view_counts)
    start_frame, end_frame = frame_range
    m5_frames = list(range(start_frame, end_frame, frame_step))

    # Generate cameras for max view count (subsets selected later)
    fxfycxcy_all, c2ws_all = build_novel_view_cameras(
        num_views=max_views,
        render_size=render_size,
    )

    # Build projection matrices for all cameras
    proj_matrices_all = np.zeros((max_views, 3, 4))
    for i in range(max_views):
        proj_matrices_all[i] = c2w_to_projection_matrix(c2ws_all[i], fxfycxcy_all[i])

    # Camera dicts for project_3d_to_2d
    camera_dicts = []
    for i in range(max_views):
        w2c = np.linalg.inv(c2ws_all[i])
        fx, fy, cx, cy = fxfycxcy_all[i]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        camera_dicts.append({"K": K, "R": w2c[:3, :3], "t": w2c[:3, 3:]})

    data_path = Path(data_dir)
    results = {}

    for n_views in view_counts:
        # Uniform selection from max_views cameras
        selected = np.linspace(0, max_views, n_views, endpoint=False, dtype=int).tolist()
        P_sel = proj_matrices_all[selected]

        results[n_views] = {}

        for sigma in noise_levels:
            all_pred = []
            all_gt = []

            for fi, m5_frame in enumerate(m5_frames):
                gt_3d = mammal_3d[m5_frame]  # (22, 3)
                n_joints = gt_3d.shape[0]

                # Project GT 3D → 2D on each novel view camera
                pts_2d = np.zeros((n_views, n_joints, 3))
                for vi, cam_idx in enumerate(selected):
                    cam = camera_dicts[cam_idx]
                    projected = project_3d_to_2d(gt_3d, cam["K"], cam["R"], cam["t"])
                    if sigma > 0:
                        noise = np.random.randn(*projected.shape) * sigma
                        projected = projected + noise
                    pts_2d[vi, :, :2] = projected
                    pts_2d[vi, :, 2] = 1.0  # perfect confidence (synthetic)

                pred_3d = triangulate_batch(pts_2d, P_sel)
                all_pred.append(pred_3d)
                all_gt.append(gt_3d)

                if fi == 0:
                    logger.info(
                        f"  Frame {m5_frame}: projected to {n_views} views, σ={sigma}"
                    )

            all_pred = np.array(all_pred)
            all_gt = np.array(all_gt)
            metrics = compute_mpjpe(all_pred, all_gt)

            results[n_views][sigma] = metrics
            logger.info(
                f"  {n_views} views, σ={sigma:.1f}px: MPJPE={metrics['mpjpe']:.4f}mm"
            )

    return results


def save_representative_views(
    pipeline,
    data_dir: str,
    mammal_3d: np.ndarray,
    output_dir: Path,
    frame_idx: int = 3240,
    num_views: int = 24,
    render_size: int = 384,
    device: str = "cuda",
):
    """Render and save representative views with keypoint overlay.

    Args:
        pipeline: GSLRMInference instance
        data_dir: Preprocessed dataset path
        mammal_3d: GT 3D keypoints
        output_dir: Where to save images
        frame_idx: M5 frame to visualize
        num_views: Number of views to render
        render_size: Image resolution
        device: Torch device
    """
    import matplotlib.pyplot as plt
    from mouse_extensions.inference.gslrm_pipeline import load_sample_data

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input data for this frame
    sample_dir = Path(data_dir) / f"frame_{frame_idx:06d}"
    if not sample_dir.exists():
        # Try alternative directory structures
        logger.warning(f"Sample dir not found: {sample_dir}. Skipping representative views.")
        return

    images, c2ws, fxfycxcys, index = load_sample_data(str(sample_dir), device=device)

    # GS-LRM predict
    result = pipeline.predict(images, c2ws, fxfycxcys, index)
    gaussians = result.gaussians[0]

    # Filter Gaussians
    filtered = gaussians.apply_all_filters(
        opacity_thres=0.04,
        scaling_thres=0.1,
        floater_thres=0.6,
        crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
    )

    # Generate novel view cameras
    fxfycxcy_novel, c2ws_novel = build_novel_view_cameras(
        num_views=num_views,
        render_size=render_size,
    )

    # Render
    rendered_imgs = render_novel_views(
        filtered, c2ws_novel, fxfycxcy_novel,
        render_size=render_size, device=device,
    )

    # Project GT 3D keypoints onto each view
    gt_3d = mammal_3d[frame_idx]  # (22, 3)

    # Create grid visualization
    cols = min(6, num_views)
    rows = (num_views + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for i in range(num_views):
        ax = axes[i]
        ax.imshow(np.clip(rendered_imgs[i], 0, 1))

        # Project keypoints
        w2c = np.linalg.inv(c2ws_novel[i])
        fx, fy, cx, cy = fxfycxcy_novel[i]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        from mouse_extensions.analysis.triangulation_analysis import project_3d_to_2d
        pts_2d = project_3d_to_2d(gt_3d, K, w2c[:3, :3], w2c[:3, 3:])

        # Filter visible points (within image bounds)
        in_bounds = (
            (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < render_size)
            & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < render_size)
        )

        ax.scatter(
            pts_2d[in_bounds, 0], pts_2d[in_bounds, 1],
            c="lime", s=8, marker="o", edgecolors="black", linewidths=0.3,
        )
        ax.set_title(f"View {i}", fontsize=8)
        ax.axis("off")

    # Hide unused axes
    for i in range(num_views, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"GS-LRM Novel Views (N={num_views}) + GT Keypoints", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "representative_views.png", dpi=150)
    plt.close()
    logger.info(f"Saved representative views: {output_dir / 'representative_views.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-view triangulation eval with GS-LRM novel views"
    )
    parser.add_argument("--config", type=str, required=True, help="GS-LRM config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="GS-LRM checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Preprocessed dataset dir")
    parser.add_argument("--mammal-3d", type=str, required=True, help="MAMMAL 3D keypoints npz")
    parser.add_argument("--output-dir", type=str, default="outputs/triangulation_eval")
    parser.add_argument("--view-counts", type=int, nargs="+", default=[6, 9, 12, 24])
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 1.0, 2.0, 5.0])
    parser.add_argument("--frame-range", type=int, nargs=2, default=[3240, 3600])
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--render-size", type=int, default=384)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-views", action="store_true", help="Save representative views")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load MAMMAL 3D
    from mouse_extensions.analysis.triangulation_analysis import (
        load_mammal_3d, plot_noise_experiment,
    )

    logger.info("Loading MAMMAL 3D keypoints...")
    mammal_3d = load_mammal_3d(args.mammal_3d)

    # Load GS-LRM pipeline
    logger.info("Loading GS-LRM model...")
    from mouse_extensions.inference.gslrm_pipeline import GSLRMInference

    pipeline = GSLRMInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Run triangulation eval
    logger.info("Running multi-view triangulation evaluation...")
    results = run_multiview_triangulation_eval(
        pipeline=pipeline,
        data_dir=args.data_dir,
        mammal_3d=mammal_3d,
        view_counts=args.view_counts,
        noise_levels=args.noise_levels,
        frame_range=tuple(args.frame_range),
        frame_step=args.frame_step,
        render_size=args.render_size,
        device=args.device,
    )

    # Save results JSON
    results_path = output_dir / "multiview_results.json"

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = {}
    for nv, noise_dict in results.items():
        serializable[str(nv)] = {}
        for sigma, metrics in noise_dict.items():
            serializable[str(nv)][str(sigma)] = {
                k: _convert(v) for k, v in metrics.items()
            }

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved results: {results_path}")

    # Plot
    plot_noise_experiment(results, output_dir / "accuracy_vs_views.png")

    # Save representative views
    if args.save_views:
        save_representative_views(
            pipeline=pipeline,
            data_dir=args.data_dir,
            mammal_3d=mammal_3d,
            output_dir=output_dir,
            frame_idx=args.frame_range[0],
            num_views=max(args.view_counts),
            render_size=args.render_size,
            device=args.device,
        )

    # Print summary
    print("\n" + "=" * 70)
    print("MULTI-VIEW TRIANGULATION EVAL (GS-LRM Novel Views)")
    print("=" * 70)
    for nv in sorted(results.keys()):
        for sigma in sorted(results[nv].keys()):
            m = results[nv][sigma]
            print(f"  {nv:3d} views, σ={sigma:4.1f}px: MPJPE={m['mpjpe']:.4f}mm")
    print("=" * 70)


if __name__ == "__main__":
    main()
