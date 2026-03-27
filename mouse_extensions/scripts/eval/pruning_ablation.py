"""Pruning ablation E1-E5: variable-isolated filter comparison.

E1: Opacity only (α=0.0)           — baseline
E2: Opacity + Visibility N≥2       — +background removal
E3: Opacity + Vis + Orientation    — +artifact filter
E4: Opacity only (α=0.3)           — training-time effect
E5: Opacity + Vis + Orientation (α=0.3) — best combined

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.pruning_ablation \
        --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --output-dir outputs/eval/mouse/pruning_ablation
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data
from mouse_extensions.visualization import render_opencv_cam
from mouse_extensions.behavior.multiview_visibility_filter import compute_visibility_counts
from mouse_extensions.model.orientation_filter import suppress_z_aligned_flat

TEST_RANGE = (3240, 3600)  # M5t2 test split (80:10:10 temporal)
CKPT_A00 = "/node_data/joon/checkpoints/FaceLift/gslrm/base_uniform_v2_6view_v2/best_psnr.pt"
CKPT_A03 = "/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_6view_alpha03_v3/best_psnr.pt"

EXPERIMENTS = [
    {"name": "E1_opacity_a00",     "ckpt": CKPT_A00, "vis": False, "orient": False},
    {"name": "E2_vis_a00",         "ckpt": CKPT_A00, "vis": True,  "orient": False},
    {"name": "E3_vis_orient_a00",  "ckpt": CKPT_A00, "vis": True,  "orient": True},
    {"name": "E4_opacity_a03",     "ckpt": CKPT_A03, "vis": False, "orient": False},
    {"name": "E5_vis_orient_a03",  "ckpt": CKPT_A03, "vis": True,  "orient": True},
]


def load_gt_rgba(frame_dir: Path, view_idx: int, device: str) -> torch.Tensor:
    path = frame_dir / "images" / f"cam_{view_idx:03d}.png"
    img = np.array(Image.open(path)).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).to(device)


def eval_single(model, m5_dir, exp, ssim_fn, lpips_fn, device):
    """Run one experiment condition on the full test set."""
    metrics = {k: [] for k in ["psnr_gt", "psnr_int", "iou", "sil_prec", "sil_rec"]}
    n_gs_list = []

    for fi in range(TEST_RANGE[0], TEST_RANGE[1]):
        fd = m5_dir / f"{fi:06d}"
        if not fd.exists():
            continue

        imgs, c2ws, fxfys, idx = load_sample_data(str(fd), image_size=512, device=device)
        with torch.no_grad():
            result = model.predict(imgs, c2ws, fxfys, idx)
        gs = result.gaussians[0]

        # Stage 1: Standard opacity filter (always applied)
        gs.apply_all_filters(opacity_thres=0.04, scaling_thres=0.1, floater_thres=0.6,
                             crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0])

        # Stage 2: Visibility filter (E2, E3, E5)
        if exp["vis"]:
            xyz = gs.get_xyz.detach().cpu().numpy()
            vc = compute_visibility_counts(xyz, str(fd), n_views=6)
            vis_mask = torch.from_numpy(vc >= 2).to(device)
            gs.filter(vis_mask)

        # Stage 3: Orientation filter (E3, E5)
        if exp["orient"]:
            suppress_z_aligned_flat(gs, z_align_thresh=0.15, attenuation=0.0, mode="prune")

        n_gs_list.append(gs.get_xyz.shape[0])

        # Evaluate on all 6 views
        for vi in range(min(6, imgs.shape[1])):
            gt_rgba = load_gt_rgba(fd, vi, device)
            gt_rgb = gt_rgba[:3]
            gt_mask = (gt_rgba[3:4] > 0.5).float()

            try:
                r = render_opencv_cam(gs, 512, 512, c2ws[0, vi], fxfys[0, vi], bg_color=(1., 1., 1.))
                pred = r["render"].clamp(0, 1)
            except Exception:
                continue

            # PSNR_gt
            fg_p, fg_g = pred * gt_mask, gt_rgb * gt_mask
            n_fg = gt_mask.sum().clamp(min=1)
            mse_fg = ((fg_p - fg_g) ** 2).sum() / (3 * n_fg)
            psnr_gt = -10 * torch.log10(mse_fg.clamp(min=1e-10))

            # PSNR_int
            gt_wb = gt_rgb * gt_mask + (1 - gt_mask)
            mse_int = ((pred - gt_wb) ** 2).mean()
            psnr_int = -10 * torch.log10(mse_int.clamp(min=1e-10))

            # IoU + Silhouette Precision/Recall
            pred_m = (pred.mean(0) < 0.95).float()
            gt_m = gt_mask.squeeze(0)
            inter = (pred_m * gt_m).sum()
            union = ((pred_m + gt_m) > 0).float().sum().clamp(min=1)

            metrics["psnr_gt"].append(psnr_gt.item())
            metrics["psnr_int"].append(psnr_int.item())
            metrics["iou"].append((inter / union).item())
            metrics["sil_prec"].append((inter / pred_m.sum().clamp(min=1)).item())
            metrics["sil_rec"].append((inter / gt_m.sum().clamp(min=1)).item())

        if len(n_gs_list) % 30 == 0:
            print(f"    {len(n_gs_list)} frames, PSNR_gt={np.mean(metrics['psnr_gt']):.2f}, "
                  f"N_gs={np.mean(n_gs_list):.0f}")

    result = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in metrics.items()}
    result["n_gaussians"] = {"mean": float(np.mean(n_gs_list)), "std": float(np.std(n_gs_list))}
    return result


def main():
    parser = argparse.ArgumentParser(description="Pruning ablation E1-E5")
    parser.add_argument("--m5-dir", required=True)
    parser.add_argument("--config", default="configs/base/gslrm_mouse.yaml")
    parser.add_argument("--output-dir", default="outputs/eval/mouse/pruning_ablation")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    m5_dir = Path(args.m5_dir)

    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)

    all_results = {}
    current_ckpt = None
    model = None

    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"{exp['name']}: vis={exp['vis']}, orient={exp['orient']}")
        print(f"{'='*60}")

        # Only reload model if checkpoint changes
        if exp["ckpt"] != current_ckpt:
            if model is not None:
                del model
            torch.cuda.empty_cache()
            model = GSLRMInference(config_path=args.config, checkpoint_path=exp["ckpt"])
            model.config.model.num_input_views = 6
            current_ckpt = exp["ckpt"]

        result = eval_single(model, m5_dir, exp, ssim_fn, lpips_fn, device)
        all_results[exp["name"]] = result

        n = result["n_gaussians"]["mean"]
        print(f"\n  PSNR_gt={result['psnr_gt']['mean']:.2f}, IoU={result['iou']['mean']:.3f}, "
              f"SilPrec={result['sil_prec']['mean']:.3f}, N_gs={n:.0f}")

        with open(out / f"{exp['name']}.json", "w") as f:
            json.dump(result, f, indent=2)

    # Summary table
    with open(out / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("PRUNING ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"{'Exp':<22} {'PSNR_gt':>8} {'PSNR_int':>9} {'IoU':>6} {'SilPrec':>8} {'N_gs':>7}")
    print("-" * 62)
    for name, r in all_results.items():
        print(f"{name:<22} {r['psnr_gt']['mean']:>8.2f} {r['psnr_int']['mean']:>9.2f} "
              f"{r['iou']['mean']:>6.3f} {r['sil_prec']['mean']:>8.3f} "
              f"{r['n_gaussians']['mean']:>7.0f}")

    # Marginal analysis
    print(f"\n{'='*70}")
    print("MARGINAL ANALYSIS")
    print(f"{'='*70}")
    names = [e["name"] for e in EXPERIMENTS]
    pairs = [(0, 1, "E2-E1: +Visibility"), (1, 2, "E3-E2: +Orientation"),
             (0, 3, "E4-E1: +α=0.3"), (2, 4, "E5-E3: +α=0.3 on filtered")]
    for i, j, label in pairs:
        dp = all_results[names[j]]["psnr_gt"]["mean"] - all_results[names[i]]["psnr_gt"]["mean"]
        di = all_results[names[j]]["iou"]["mean"] - all_results[names[i]]["iou"]["mean"]
        dn = all_results[names[j]]["n_gaussians"]["mean"] - all_results[names[i]]["n_gaussians"]["mean"]
        print(f"  {label}: ΔPSNR_gt={dp:+.2f}, ΔIoU={di:+.3f}, ΔN_gs={dn:+.0f}")


if __name__ == "__main__":
    main()
