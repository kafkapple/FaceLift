"""Ablation comparison grid video generator.

Generates side-by-side grid videos for:
  --type view   : 1~6-view ablation (6 rows × 4 representative novel views)
  --type alpha  : alpha-weight ablation (4 rows × 4 representative novel views)
                  α ∈ {0.0, 0.3, 0.5, 1.0} using 4-view checkpoints

For 2×6 grid (Row1=GT, Row2=Novel, Cols=1v~6v), use view_ablation_2x6.py instead.

No intermediate PNGs — writes directly to VideoWriter.

Usage (run on gpu03):
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.ablation_comparison \\
        --type view \\
        --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \\
        --output-dir outputs/viz/comparison/mouse/view_ablation_v2

    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.ablation_comparison \\
        --type alpha \\
        --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \\
        --output-dir outputs/viz/comparison/mouse/alpha_ablation_v2
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# Shared render utilities
from mouse_extensions.scripts.eval.fl_gt_view_comparison import (
    REPRESENTATIVE_VIEWS,
    add_label,
    get_novel_camera,
    render_gaussian_at_view,
)
from mouse_extensions.behavior.multiview_visibility_filter import compute_visibility_counts
from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
_CKPT_BASE = Path("/node_data/joon/checkpoints/FaceLift/gslrm")
# All experiments share the same model architecture — only num_input_views differs.
# Experiment-only YAMLs contain training overrides (loss weights, ckpt dirs) which are
# not needed for inference. Use base config for all models.
_BASE_CFG = Path("configs/base/gslrm_mouse.yaml")

# All 1~6-view checkpoints trained to completion (best_psnr.pt available).
VIEW_ABLATION_EXPERIMENTS: List[Dict] = [
    {"label": "1-view",   "ckpt_dir": "base_uniform_v2_1view_v2", "n_views": 1},
    {"label": "2-view",   "ckpt_dir": "base_uniform_v2_2view_v2", "n_views": 2},
    {"label": "3-view",   "ckpt_dir": "base_uniform_v2_3view_v2", "n_views": 3},
    {"label": "4-view",   "ckpt_dir": "base_uniform_v2_4view_v2", "n_views": 4},
    {"label": "5-view",   "ckpt_dir": "base_uniform_v2_5view_v2", "n_views": 5},
    {"label": "6-view",   "ckpt_dir": "base_uniform_v2_6view_v2", "n_views": 6},
]

# α=0.1 checkpoint was never trained; confirmed available: 0.0, 0.3, 0.5, 1.0
ALPHA_ABLATION_EXPERIMENTS: List[Dict] = [
    {"label": "α=0.0 (baseline)", "ckpt_dir": "base_uniform_v2_4view_v2",         "n_views": 4},
    {"label": "α=0.3 ⭐",          "ckpt_dir": "base_uniform_v2_4view_alpha03_v3", "n_views": 4},
    {"label": "α=0.5",            "ckpt_dir": "base_uniform_v2_4view_alpha05_v3", "n_views": 4},
    {"label": "α=1.0",            "ckpt_dir": "base_uniform_v2_4view_alpha10_v3", "n_views": 4},
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_models(experiments: List[Dict], device: str) -> List[Optional[GSLRMInference]]:
    """Load inference models. Returns None for missing checkpoints (with warning)."""
    models = []
    for exp in experiments:
        ckpt = _CKPT_BASE / exp["ckpt_dir"] / "best_psnr.pt"
        if not ckpt.exists():
            print(f"[WARN] Checkpoint not found (skipping): {ckpt}")
            models.append(None)
            continue
        print(f"  Loading {exp['label']}  ←  {ckpt}")
        m = GSLRMInference(
            config_path=str(_BASE_CFG),
            checkpoint_path=str(ckpt),
            device=device,
        )
        m.config.model.num_input_views = exp["n_views"]
        models.append(m)
    return models


# ---------------------------------------------------------------------------
# Core grid generation
# ---------------------------------------------------------------------------
def run_ablation_comparison(
    exp_type: str,
    m5_dir: str,
    output_dir: str,
    frame_range: str = "3240:3360",
    cell_size: int = 512,
    fps: int = 30,
    n_filter: int = 2,
    device: str = "cuda",
) -> None:
    """Generate a grid video comparing all ablation models.

    Grid layout:
        Rows  = models (e.g., 1-view … 6-view)
        Cols  = representative novel views (bottom, top, side-left, side-right)
    """
    experiments = VIEW_ABLATION_EXPERIMENTS if exp_type == "view" else ALPHA_ABLATION_EXPERIMENTS
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Ablation Comparison: {exp_type} ===")
    print(f"Loading models...")
    models = _load_models(experiments, device)
    active = [(exp, m) for exp, m in zip(experiments, models) if m is not None]
    if not active:
        print("[ERROR] No valid checkpoints. Aborting.")
        return

    n_models = len(active)
    n_rep = len(REPRESENTATIVE_VIEWS)

    # Pre-compute representative novel cameras (render at cell_size)
    rep_cams = {
        vname: get_novel_camera(vdef["elevation"], vdef["azimuth"], resolution=cell_size)
        for vname, vdef in REPRESENTATIVE_VIEWS.items()
    }

    W_total = n_rep * cell_size
    H_total = n_models * cell_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MPEG-4; cv2 H264 unavailable on this Linux host
    out_path = str(out / f"{exp_type}_ablation_grid.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W_total, H_total))

    start, end = map(int, frame_range.split(":"))
    frame_indices = list(range(start, end))
    print(f"Rendering {len(frame_indices)} frames × {n_models} models × {n_rep} views  →  {out_path}")
    print(f"Grid: {W_total}×{H_total} px  (cell={cell_size})")

    for fi in frame_indices:
        fd = Path(m5_dir) / f"{fi:06d}"
        if not fd.exists():
            continue

        # Load full 6-view data at 512 (model native size); slice per model below
        imgs, c2ws, fxfys, idx = load_sample_data(str(fd), image_size=512, device=device)

        rows = []
        for exp, model in active:
            n_in = exp["n_views"]
            imgs_in  = imgs[:, :n_in]
            c2ws_in  = c2ws[:, :n_in]
            fxfys_in = fxfys[:, :n_in]

            result = model.predict(imgs_in, c2ws_in, fxfys_in, idx)
            gaussians = result.gaussians[0]

            # Visibility filter (always use all 6 cameras for spatial consistency)
            xyz = gaussians.get_xyz.detach().cpu().numpy()
            vc = compute_visibility_counts(xyz, str(fd), n_views=6)
            vis_mask = vc >= n_filter

            cells = []
            for vname, vdef in REPRESENTATIVE_VIEWS.items():
                cam = rep_cams[vname]
                try:
                    cell = render_gaussian_at_view(
                        gaussians, cam, resolution=cell_size, device=device, vis_mask=vis_mask,
                    )
                except Exception as e:
                    print(f"  [warn] render failed {exp['label']} {vname} frame {fi}: {e}")
                    cell = np.full((cell_size, cell_size, 3), 80, dtype=np.uint8)

                cell = add_label(cell, vdef["label"], exp["label"],
                                color=(255, 255, 255), outline=True)
                cells.append(cell)

            rows.append(np.hstack(cells))

        frame_grid = np.vstack(rows)
        cv2.putText(
            frame_grid, f"Frame {fi:04d}",
            (8, H_total - 8), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
        writer.write(cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR))

        if fi % 20 == 0:
            print(f"  frame {fi}/{end - 1}")

    writer.release()
    print(f"\nDone. Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation comparison grid video")
    parser.add_argument("--type",        choices=["view", "alpha"], required=True,
                        help="Ablation type: view (1-6 view) or alpha (weight ablation)")
    parser.add_argument("--m5-dir",      default="/home/joon/data/preprocessed/FaceLift_mouse/M5")
    parser.add_argument("--output-dir",  default=None,
                        help="Default: outputs/viz/comparison/mouse/{type}_ablation_v2")
    parser.add_argument("--frame-range", default="3240:3360",
                        help="Sample index range (default: test set 3240:3360 = 120 frames)")
    parser.add_argument("--cell-size",   type=int, default=512,
                        help="Pixel size of each grid cell (default: 512)")
    parser.add_argument("--fps",         type=int, default=30)
    parser.add_argument("--n-filter",    type=int, default=2,
                        help="Visibility filter: min cameras seeing each Gaussian")
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"outputs/viz/comparison/mouse/{args.type}_ablation_v2"

    run_ablation_comparison(
        exp_type=args.type,
        m5_dir=args.m5_dir,
        output_dir=args.output_dir,
        frame_range=args.frame_range,
        cell_size=args.cell_size,
        fps=args.fps,
        n_filter=args.n_filter,
        device=args.device,
    )


if __name__ == "__main__":
    main()
