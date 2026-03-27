"""Evaluate a checkpoint on the test set (fair eval: PSNR_gt, IoU, SSIM, LPIPS).

Runs inference on all test frames and computes per-frame metrics.
Designed to evaluate checkpoints that missed validation during training.

Usage:
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.eval.checkpoint_eval \
        --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_6view_alpha03_v3/best_psnr.pt \
        --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --num-input-views 6

    # Batch eval multiple checkpoints:
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.eval.checkpoint_eval \
        --checkpoints \
            "6v_a0.3=/node_data/.../M5t2_6view_alpha03_v3/best_psnr.pt" \
            "6v_a0.5=/node_data/.../M5t2_6view_alpha05_v3/best_psnr.pt" \
            "6v_a1.0=/node_data/.../M5t2_6view_alpha10_v3/best_psnr.pt" \
        --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --num-input-views 6
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data

# M5t2 test split
TEST_RANGE = (3240, 3600)


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor,
                    ssim_fn, lpips_fn, device: str) -> Dict:
    """Compute PSNR, SSIM, LPIPS on white-bg composited images.

    load_sample_data returns RGB only (no alpha). GT images already have
    white background, so we compare directly.

    Args:
        pred: (3, H, W) rendered image [0,1] with white bg
        gt: (3, H, W) GT image [0,1] with white bg
    """
    # PSNR on full image (white bg composite — matches training eval)
    mse = ((pred - gt) ** 2).mean()
    psnr = -10 * torch.log10(mse.clamp(min=1e-10))

    # IoU: non-white pixels as foreground proxy
    pred_mask = (pred.mean(0) < 0.95).float()
    gt_mask = (gt.mean(0) < 0.95).float()
    intersection = (pred_mask * gt_mask).sum()
    union = ((pred_mask + gt_mask) > 0).float().sum()
    iou = intersection / union.clamp(min=1)

    # SSIM / LPIPS
    ssim_val = ssim_fn(pred.unsqueeze(0), gt.unsqueeze(0))
    lpips_val = lpips_fn(pred.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1)

    return {
        "psnr": psnr.item(),
        "iou": iou.item(),
        "ssim": ssim_val.item(),
        "lpips": lpips_val.item(),
    }


def eval_checkpoint(
    model: GSLRMInference,
    m5_dir: Path,
    num_input_views: int,
    frame_range: tuple = TEST_RANGE,
    device: str = "cuda",
) -> Dict:
    """Evaluate a model on the test set."""
    from mouse_extensions.visualization import render_opencv_cam

    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)

    all_metrics = {"psnr": [], "iou": [], "ssim": [], "lpips": []}
    n_frames = 0

    for fi in range(frame_range[0], frame_range[1]):
        fd = m5_dir / f"{fi:06d}"
        if not fd.exists():
            continue

        imgs, c2ws, fxfys, idx = load_sample_data(str(fd), image_size=512, device=device)

        # Slice to num_input_views
        imgs_in = imgs[:, :num_input_views]
        c2ws_in = c2ws[:, :num_input_views]
        fxfys_in = fxfys[:, :num_input_views]

        with torch.no_grad():
            result = model.predict(imgs_in, c2ws_in, fxfys_in, idx)

        gaussians = result.gaussians[0]
        gaussians.apply_all_filters(
            opacity_thres=0.04, scaling_thres=0.1, floater_thres=0.6,
            crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
        )

        # Evaluate on holdout views (views beyond num_input_views)
        n_total_views = imgs.shape[1]
        eval_views = list(range(num_input_views, n_total_views))
        if not eval_views:
            eval_views = list(range(n_total_views))  # If 6v, eval on all

        for vi in eval_views:
            gt_rgb = imgs[0, vi]  # (3, H, W) — load_sample_data returns RGB only

            c2w_v = c2ws[0, vi]
            fxfy_v = fxfys[0, vi]
            try:
                rendered = render_opencv_cam(
                    gaussians, 512, 512, c2w_v, fxfy_v,
                    bg_color=(1.0, 1.0, 1.0),
                )
                pred_rgb = rendered["render"].clamp(0, 1)
            except Exception as e:
                if n_frames < 2:
                    print(f"  [render error view {vi}] {e}")
                continue

            m = compute_metrics(pred_rgb, gt_rgb, ssim_fn, lpips_fn, device)
            for k, v in m.items():
                all_metrics[k].append(v)

        n_frames += 1
        if n_frames % 30 == 0:
            avg_psnr = np.mean(all_metrics["psnr"])
            print(f"  {n_frames} frames, running PSNR_gt={avg_psnr:.2f}")

    results = {}
    for k, v in all_metrics.items():
        results[k] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint(s) on test set")
    parser.add_argument("--checkpoint", default=None, help="Single checkpoint path")
    parser.add_argument("--checkpoints", nargs="*", metavar="NAME=PATH",
                        help="Multiple checkpoints as name=path pairs")
    parser.add_argument("--config", default="configs/base/gslrm_mouse.yaml")
    parser.add_argument("--m5-dir", required=True)
    parser.add_argument("--num-input-views", type=int, default=6)
    parser.add_argument("--output-dir", default="outputs/eval/mouse/checkpoint_eval")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt_dict = {}
    if args.checkpoints:
        for item in args.checkpoints:
            name, path = item.split("=", 1)
            ckpt_dict[name] = path
    elif args.checkpoint:
        name = Path(args.checkpoint).parent.name
        ckpt_dict[name] = args.checkpoint
    else:
        parser.error("Provide --checkpoint or --checkpoints")

    all_results = {}
    for name, ckpt_path in ckpt_dict.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        model = GSLRMInference(config_path=args.config, checkpoint_path=ckpt_path)
        model.config.model.num_input_views = args.num_input_views

        results = eval_checkpoint(
            model, Path(args.m5_dir), args.num_input_views,
        )

        all_results[name] = results
        print(f"\n  PSNR:    {results['psnr']['mean']:.2f} ± {results['psnr']['std']:.2f}")
        print(f"  IoU:     {results['iou']['mean']:.3f} ± {results['iou']['std']:.3f}")
        print(f"  SSIM:    {results['ssim']['mean']:.3f} ± {results['ssim']['std']:.3f}")
        print(f"  LPIPS:   {results['lpips']['mean']:.3f} ± {results['lpips']['std']:.3f}")

        # Save per-checkpoint results
        with open(out / f"eval_{name}.json", "w") as f:
            json.dump(results, f, indent=2)

        del model
        torch.cuda.empty_cache()

    # Save summary
    summary_path = out / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Name':<25} {'PSNR':>8} {'IoU':>8} {'SSIM':>8} {'LPIPS':>8}")
    print("-" * 60)
    for name, r in all_results.items():
        print(f"{name:<25} {r['psnr']['mean']:>8.2f} {r['iou']['mean']:>8.3f} "
              f"{r['ssim']['mean']:>8.3f} {r['lpips']['mean']:>8.3f}")


if __name__ == "__main__":
    main()
