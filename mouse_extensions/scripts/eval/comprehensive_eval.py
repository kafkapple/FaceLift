"""Comprehensive checkpoint evaluation with masked + unmasked metrics + artifact analysis.

Computes:
  - PSNR_gt (masked foreground), PSNR_int (white-bg full image)
  - IoU, Silhouette Precision/Recall
  - SSIM, LPIPS
  - Per-camera PSNR (bottom view isolation)
  - Gaussian count + opacity stats
  - Artifact: floater fraction (DBSCAN)

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.comprehensive_eval \
        --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --checkpoints \
            "6v_a0.0=/node_data/.../base_uniform_v2_6view_v2/best_psnr.pt" \
            "6v_a0.3=/node_data/.../M5t2_6view_alpha03_v3/best_psnr.pt"
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data
from mouse_extensions.visualization import render_opencv_cam

TEST_RANGE = (3240, 3600)  # M5t2 test split (80:10:10 temporal)
FILTER_PARAMS = dict(
    opacity_thres=0.04, scaling_thres=0.1, floater_thres=0.6,
    crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
)


def load_gt_rgba(frame_dir: Path, view_idx: int, device: str) -> torch.Tensor:
    """Load GT RGBA image as (4, H, W) float tensor [0,1]."""
    path = frame_dir / "images" / f"cam_{view_idx:03d}.png"
    img = np.array(Image.open(path)).astype(np.float32) / 255.0  # (H,W,4)
    return torch.from_numpy(img).permute(2, 0, 1).to(device)  # (4,H,W)


def compute_per_view_metrics(
    pred: torch.Tensor, gt_rgba: torch.Tensor,
    ssim_fn, lpips_fn,
) -> Dict:
    """Compute all metrics for one view.

    Args:
        pred: (3, H, W) rendered [0,1] white-bg
        gt_rgba: (4, H, W) GT RGBA [0,1]
    """
    gt_rgb = gt_rgba[:3]
    gt_mask = (gt_rgba[3:4] > 0.5).float()  # binary FG mask

    # PSNR_gt: masked foreground only
    fg_pred = pred * gt_mask
    fg_gt = gt_rgb * gt_mask
    n_fg = gt_mask.sum().clamp(min=1)
    mse_fg = ((fg_pred - fg_gt) ** 2).sum() / (3 * n_fg)
    psnr_gt = -10 * torch.log10(mse_fg.clamp(min=1e-10))

    # PSNR_int: white-bg full image
    gt_wb = gt_rgb * gt_mask + (1 - gt_mask)  # white bg composite
    mse_int = ((pred - gt_wb) ** 2).mean()
    psnr_int = -10 * torch.log10(mse_int.clamp(min=1e-10))

    # IoU (GT mask vs predicted non-white region)
    pred_mask = (pred.mean(0) < 0.95).float()
    gt_mask_2d = gt_mask.squeeze(0)
    intersection = (pred_mask * gt_mask_2d).sum()
    union = ((pred_mask + gt_mask_2d) > 0).float().sum().clamp(min=1)
    iou = intersection / union

    # Silhouette Precision & Recall
    pred_fg_area = pred_mask.sum().clamp(min=1)
    gt_fg_area = gt_mask_2d.sum().clamp(min=1)
    sil_precision = intersection / pred_fg_area  # low = floaters outside object
    sil_recall = intersection / gt_fg_area       # low = missing geometry

    # SSIM / LPIPS on white-bg
    ssim_val = ssim_fn(pred.unsqueeze(0), gt_wb.unsqueeze(0))
    lpips_val = lpips_fn(pred.unsqueeze(0) * 2 - 1, gt_wb.unsqueeze(0) * 2 - 1)

    return {
        "psnr_gt": psnr_gt.item(),
        "psnr_int": psnr_int.item(),
        "iou": iou.item(),
        "sil_precision": sil_precision.item(),
        "sil_recall": sil_recall.item(),
        "ssim": ssim_val.item(),
        "lpips": lpips_val.item(),
    }


def compute_gaussian_stats(gaussians) -> Dict:
    """Compute Gaussian-level artifact statistics."""
    opacity = gaussians.get_opacity.squeeze(-1).detach().cpu().numpy()
    scaling = gaussians.get_scaling.detach().cpu().numpy()
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    N = len(opacity)

    # Anisotropy
    s_max = scaling.max(axis=1)
    s_min = np.clip(scaling.min(axis=1), 1e-8, None)
    ratio = s_max / s_min

    # Floater detection via DBSCAN
    floater_frac = 0.0
    try:
        from sklearn.cluster import DBSCAN
        if N > 100:
            clustering = DBSCAN(eps=0.03, min_samples=10).fit(xyz)
            labels = clustering.labels_
            valid = labels[labels >= 0]
            if len(valid) > 0:
                _, counts = np.unique(valid, return_counts=True)
                main_cluster_size = counts.max()
                floater_frac = 1.0 - main_cluster_size / N
    except ImportError:
        pass

    return {
        "n_gaussians": N,
        "opacity_mean": float(opacity.mean()),
        "opacity_median": float(np.median(opacity)),
        "opacity_std": float(opacity.std()),
        "aniso_median": float(np.median(ratio)),
        "aniso_pct_flat": float(100 * (ratio > 30).mean()),
        "floater_frac": float(floater_frac),
    }


def eval_checkpoint(
    model, m5_dir: Path, num_input_views: int,
    ssim_fn, lpips_fn, device: str,
) -> Dict:
    """Full evaluation on test set."""
    per_cam_psnr = {i: [] for i in range(6)}
    all_metrics = {k: [] for k in [
        "psnr_gt", "psnr_int", "iou", "sil_precision", "sil_recall", "ssim", "lpips"
    ]}
    all_gs_stats = []
    n_frames = 0

    for fi in range(TEST_RANGE[0], TEST_RANGE[1]):
        fd = m5_dir / f"{fi:06d}"
        if not fd.exists():
            continue

        imgs, c2ws, fxfys, idx = load_sample_data(str(fd), image_size=512, device=device)
        imgs_in = imgs[:, :num_input_views]
        c2ws_in = c2ws[:, :num_input_views]
        fxfys_in = fxfys[:, :num_input_views]

        with torch.no_grad():
            result = model.predict(imgs_in, c2ws_in, fxfys_in, idx)

        gs = result.gaussians[0]
        gs.apply_all_filters(**FILTER_PARAMS)

        # Gaussian stats (sample every 10th frame to save time)
        if n_frames % 10 == 0:
            all_gs_stats.append(compute_gaussian_stats(gs))

        # Evaluate on all 6 views
        n_views = min(6, imgs.shape[1])
        for vi in range(n_views):
            gt_rgba = load_gt_rgba(fd, vi, device)
            c2w_v = c2ws[0, vi]
            fxfy_v = fxfys[0, vi]

            try:
                rendered = render_opencv_cam(
                    gs, 512, 512, c2w_v, fxfy_v, bg_color=(1.0, 1.0, 1.0),
                )
                pred = rendered["render"].clamp(0, 1)
            except Exception:
                continue

            m = compute_per_view_metrics(pred, gt_rgba, ssim_fn, lpips_fn)
            for k, v in m.items():
                all_metrics[k].append(v)
            per_cam_psnr[vi].append(m["psnr_gt"])

        n_frames += 1
        if n_frames % 30 == 0:
            avg = np.mean(all_metrics["psnr_gt"])
            print(f"  {n_frames} frames, PSNR_gt={avg:.2f}")

    # Aggregate
    results = {}
    for k, v in all_metrics.items():
        results[k] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}

    # Per-camera PSNR_gt
    results["per_cam_psnr_gt"] = {
        f"cam_{i}": float(np.mean(v)) for i, v in per_cam_psnr.items() if v
    }

    # Gaussian stats (averaged across sampled frames)
    if all_gs_stats:
        gs_keys = all_gs_stats[0].keys()
        results["gaussian"] = {
            k: float(np.mean([s[k] for s in all_gs_stats])) for k in gs_keys
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive checkpoint evaluation")
    parser.add_argument("--checkpoints", nargs="+", metavar="NAME=PATH", required=True)
    parser.add_argument("--config", default="configs/base/gslrm_mouse.yaml")
    parser.add_argument("--m5-dir", required=True)
    parser.add_argument("--num-input-views", type=int, default=6)
    parser.add_argument("--output-dir", default="outputs/eval/mouse/comprehensive")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)

    ckpt_dict = {}
    for item in args.checkpoints:
        name, path = item.split("=", 1)
        ckpt_dict[name] = path

    all_results = {}
    for name, ckpt_path in ckpt_dict.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name} ({ckpt_path})")
        print(f"{'='*60}")

        model = GSLRMInference(config_path=args.config, checkpoint_path=ckpt_path)
        model.config.model.num_input_views = args.num_input_views

        results = eval_checkpoint(
            model, Path(args.m5_dir), args.num_input_views,
            ssim_fn, lpips_fn, device,
        )
        all_results[name] = results

        # Print summary
        print(f"\n  PSNR_gt:  {results['psnr_gt']['mean']:.2f} ± {results['psnr_gt']['std']:.2f}")
        print(f"  PSNR_int: {results['psnr_int']['mean']:.2f} ± {results['psnr_int']['std']:.2f}")
        print(f"  IoU:      {results['iou']['mean']:.3f}")
        print(f"  Sil.Prec: {results['sil_precision']['mean']:.3f}")
        print(f"  Sil.Rec:  {results['sil_recall']['mean']:.3f}")
        print(f"  SSIM:     {results['ssim']['mean']:.3f}")
        print(f"  LPIPS:    {results['lpips']['mean']:.3f}")
        if "per_cam_psnr_gt" in results:
            print(f"  Per-cam PSNR_gt: {results['per_cam_psnr_gt']}")
        if "gaussian" in results:
            gs = results["gaussian"]
            print(f"  Gaussians: N={gs['n_gaussians']:.0f}, "
                  f"opacity={gs['opacity_mean']:.3f}, "
                  f"floater={gs['floater_frac']:.3f}")

        with open(out / f"eval_{name}.json", "w") as f:
            json.dump(results, f, indent=2)

        del model
        torch.cuda.empty_cache()

    # Summary table
    with open(out / "eval_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("COMPREHENSIVE COMPARISON TABLE")
    print(f"{'='*70}")
    header = f"{'Name':<12} {'PSNR_gt':>8} {'PSNR_int':>9} {'IoU':>6} {'SilPrec':>8} {'SSIM':>6} {'LPIPS':>6} {'N_gs':>6} {'Floater':>8}"
    print(header)
    print("-" * 70)
    for name, r in all_results.items():
        gs = r.get("gaussian", {})
        print(f"{name:<12} {r['psnr_gt']['mean']:>8.2f} {r['psnr_int']['mean']:>9.2f} "
              f"{r['iou']['mean']:>6.3f} {r['sil_precision']['mean']:>8.3f} "
              f"{r['ssim']['mean']:>6.3f} {r['lpips']['mean']:>6.3f} "
              f"{gs.get('n_gaussians', 0):>6.0f} {gs.get('floater_frac', 0):>8.3f}")

    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
