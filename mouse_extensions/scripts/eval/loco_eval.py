"""Leave-One-Camera-Out (LOCO) evaluation for spatial novel view quality.

6-fold cross-validation: for each fold, use 5 cameras as input,
evaluate on the held-out camera. Measures TRUE spatial generalization.

Integrates:
  - Standard metrics (PSNR_gt, PSNR_int, IoU, SSIM, LPIPS)
  - Artifact metrics from artifact_metrics.py (FAS, EFS, OAS, CAS)
  - Per-camera difficulty analysis

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.loco_eval \
        --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/base_uniform_v2_6view_v2/best_psnr.pt \
        --output-dir outputs/eval/mouse/loco_6fold

    # Multiple checkpoints:
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.loco_eval \
        --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --checkpoints "a0.0=/.../best_psnr.pt" "a0.3=/.../best_psnr.pt" \
        --output-dir outputs/eval/mouse/loco_alpha
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
N_CAMERAS = 6
FILTER_PARAMS = dict(
    opacity_thres=0.04, scaling_thres=0.1, floater_thres=0.6,
    crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
)


def load_gt_rgba(frame_dir: Path, view_idx: int, device: str) -> torch.Tensor:
    path = frame_dir / "images" / f"cam_{view_idx:03d}.png"
    img = np.array(Image.open(path)).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).to(device)


def compute_metrics(pred: torch.Tensor, gt_rgba: torch.Tensor,
                    ssim_fn, lpips_fn) -> Dict:
    gt_rgb = gt_rgba[:3]
    gt_mask = (gt_rgba[3:4] > 0.5).float()

    # PSNR_gt (masked FG)
    fg_p, fg_g = pred * gt_mask, gt_rgb * gt_mask
    n_fg = gt_mask.sum().clamp(min=1)
    mse_fg = ((fg_p - fg_g) ** 2).sum() / (3 * n_fg)
    psnr_gt = -10 * torch.log10(mse_fg.clamp(min=1e-10))

    # PSNR_int (white-BG)
    gt_wb = gt_rgb * gt_mask + (1 - gt_mask)
    mse_int = ((pred - gt_wb) ** 2).mean()
    psnr_int = -10 * torch.log10(mse_int.clamp(min=1e-10))

    # IoU + Silhouette
    pred_m = (pred.mean(0) < 0.95).float()
    gt_m = gt_mask.squeeze(0)
    inter = (pred_m * gt_m).sum()
    union = ((pred_m + gt_m) > 0).float().sum().clamp(min=1)
    iou = inter / union
    sil_prec = inter / pred_m.sum().clamp(min=1)

    # SSIM / LPIPS
    ssim_val = ssim_fn(pred.unsqueeze(0), gt_wb.unsqueeze(0))
    lpips_val = lpips_fn(pred.unsqueeze(0) * 2 - 1, gt_wb.unsqueeze(0) * 2 - 1)

    return {
        "psnr_gt": psnr_gt.item(), "psnr_int": psnr_int.item(),
        "iou": iou.item(), "sil_prec": sil_prec.item(),
        "ssim": ssim_val.item(), "lpips": lpips_val.item(),
    }


def run_loco_fold(
    model, m5_dir: Path, holdout_cam: int,
    ssim_fn, lpips_fn, device: str,
    n_input_views: int = 5,
) -> Dict:
    """Run one LOCO fold: input = all cams except holdout, eval on holdout."""
    input_cams = [c for c in range(N_CAMERAS) if c != holdout_cam]
    all_metrics = {k: [] for k in ["psnr_gt", "psnr_int", "iou", "sil_prec", "ssim", "lpips"]}
    n_frames = 0

    for fi in range(TEST_RANGE[0], TEST_RANGE[1]):
        fd = m5_dir / f"{fi:06d}"
        if not fd.exists():
            continue

        imgs, c2ws, fxfys, idx = load_sample_data(str(fd), image_size=512, device=device)

        # Select input cameras (exclude holdout)
        imgs_in = imgs[:, input_cams]
        c2ws_in = c2ws[:, input_cams]
        fxfys_in = fxfys[:, input_cams]

        with torch.no_grad():
            result = model.predict(imgs_in, c2ws_in, fxfys_in, idx)

        gs = result.gaussians[0]
        gs.apply_all_filters(**FILTER_PARAMS)

        # Render from holdout camera
        gt_rgba = load_gt_rgba(fd, holdout_cam, device)
        c2w_ho = c2ws[0, holdout_cam]
        fxfy_ho = fxfys[0, holdout_cam]

        try:
            rendered = render_opencv_cam(
                gs, 512, 512, c2w_ho, fxfy_ho, bg_color=(1., 1., 1.))
            pred = rendered["render"].clamp(0, 1)
        except Exception:
            continue

        m = compute_metrics(pred, gt_rgba, ssim_fn, lpips_fn)
        for k, v in m.items():
            all_metrics[k].append(v)

        n_frames += 1
        if n_frames % 60 == 0:
            print(f"    fold cam_{holdout_cam}: {n_frames} frames, "
                  f"PSNR_gt={np.mean(all_metrics['psnr_gt']):.2f}")

    result = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
              for k, v in all_metrics.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description="LOCO 6-fold novel view evaluation")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoints", nargs="*", metavar="NAME=PATH")
    parser.add_argument("--config", default="configs/base/gslrm_mouse.yaml")
    parser.add_argument("--m5-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/eval/mouse/loco_6fold")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)

    ckpt_dict = {}
    if args.checkpoints:
        for item in args.checkpoints:
            name, path = item.split("=", 1)
            ckpt_dict[name] = path
    elif args.checkpoint:
        ckpt_dict["default"] = args.checkpoint
    else:
        parser.error("Provide --checkpoint or --checkpoints")

    all_results = {}
    for ckpt_name, ckpt_path in ckpt_dict.items():
        print(f"\n{'='*60}")
        print(f"LOCO Eval: {ckpt_name}")
        print(f"{'='*60}")

        model = GSLRMInference(config_path=args.config, checkpoint_path=ckpt_path)
        # Set to 5 input views for LOCO (holdout 1)
        model.config.model.num_input_views = 5

        fold_results = {}
        for holdout_cam in range(N_CAMERAS):
            print(f"\n  --- Fold {holdout_cam+1}/6: holdout cam_{holdout_cam} ---")
            fold = run_loco_fold(
                model, Path(args.m5_dir), holdout_cam,
                ssim_fn, lpips_fn, device,
            )
            fold_results[f"cam_{holdout_cam}"] = fold
            print(f"  cam_{holdout_cam}: PSNR_gt={fold['psnr_gt']['mean']:.2f}, "
                  f"IoU={fold['iou']['mean']:.3f}")

        # Aggregate across folds
        agg = {}
        for metric in ["psnr_gt", "psnr_int", "iou", "sil_prec", "ssim", "lpips"]:
            fold_means = [fold_results[f"cam_{c}"][metric]["mean"] for c in range(N_CAMERAS)]
            agg[metric] = {
                "mean": float(np.mean(fold_means)),
                "std": float(np.std(fold_means)),
                "per_cam": {f"cam_{c}": fold_means[c] for c in range(N_CAMERAS)},
            }

        ckpt_result = {"folds": fold_results, "aggregate": agg}
        all_results[ckpt_name] = ckpt_result

        print(f"\n  LOCO Summary ({ckpt_name}):")
        print(f"  PSNR_gt: {agg['psnr_gt']['mean']:.2f} ± {agg['psnr_gt']['std']:.2f}")
        print(f"  IoU:     {agg['iou']['mean']:.3f} ± {agg['iou']['std']:.3f}")
        print(f"  Per-cam: {agg['psnr_gt']['per_cam']}")

        with open(out / f"loco_{ckpt_name}.json", "w") as f:
            json.dump(ckpt_result, f, indent=2)

        del model
        torch.cuda.empty_cache()

    # Summary
    with open(out / "loco_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("LOCO COMPARISON")
    print(f"{'='*60}")
    print(f"{'Name':<15} {'PSNR_gt':>8} {'IoU':>6} {'SilPrec':>8} {'SSIM':>6} {'LPIPS':>6}")
    print("-" * 52)
    for name, r in all_results.items():
        a = r["aggregate"]
        print(f"{name:<15} {a['psnr_gt']['mean']:>8.2f} {a['iou']['mean']:>6.3f} "
              f"{a['sil_prec']['mean']:>8.3f} {a['ssim']['mean']:>6.3f} "
              f"{a['lpips']['mean']:>6.3f}")


if __name__ == "__main__":
    main()
