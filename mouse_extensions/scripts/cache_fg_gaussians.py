#!/usr/bin/env python3
"""
Cache FG-aware Gaussians from trained GS-LRM for Deformation V3 training.

Key differences from extract_gaussians.py (V2 cache):
- Filters to foreground Gaussians using alpha mask (soft classification)
- Includes sparse BG sample for zero-deformation regularization
- Saves in float16 for ~30× storage reduction (237GB → ~8GB)
- Preserves original pixel indices for correspondence tracking

Output per frame (~3MB vs V2's 85MB):
    {
        fg_xyz, fg_features, fg_scaling, fg_rotation, fg_opacity,  # FG Gaussians
        fg_weights,    # alpha-based weights (1.0=core, alpha=boundary, 0.1=BG)
        fg_indices,    # original pixel indices (for inter-frame correspondence)
        bg_xyz, bg_indices,  # sparse BG sample (5%)
        metadata: {n_fg, n_bg, n_total, fg_ratio}
    }

Usage:
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.cache_fg_gaussians \
        --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_6view_alpha03_v3/best_psnr.pt \
        --data_list /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_train.txt \
        --output_dir /node_data/joon/checkpoints/FaceLift/deformation/v3/fg_cache \
        --fg_threshold 0.2 \
        --bg_sample_ratio 0.05
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[3]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_and_dataloader(checkpoint_path, config_path, data_list_path, num_frames, device):
    """Load GS-LRM model from checkpoint + separate config yaml."""
    from gslrm.model.gslrm import GSLRM
    from gslrm.data.dataset import MultiViewDataset
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    # Load config from yaml (not from checkpoint)
    config = OmegaConf.load(config_path)
    logger.info(f"Config: {config.model.num_input_views}v, image_size={config.model.image_tokenizer.image_size}")

    # Load model
    model = GSLRM(config.model)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {checkpoint_path}")

    # Create dataloader
    with open(data_list_path) as f:
        data_paths = [line.strip() for line in f if line.strip()]
    if num_frames > 0:
        data_paths = data_paths[:num_frames]

    dataset = MultiViewDataset(
        data_paths,
        num_input_views=config.model.num_input_views,
        num_views=config.model.num_views,
        background_color="white",
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    logger.info(f"Dataset: {len(data_paths)} frames")

    return model, config, dataloader


def classify_gaussians(rendered_alpha, fg_threshold=0.2, bg_sample_ratio=0.05):
    """
    Classify Gaussians into FG core / FG boundary / BG using rendered alpha.

    Soft classification (NOT hard filtering — audit requirement):
      FG core:     alpha > 0.8     → weight = 1.0
      FG boundary: threshold < alpha ≤ 0.8 → weight = alpha
      BG:          alpha ≤ threshold → 5% random sample, weight = 0.1

    Args:
        rendered_alpha: [N] or [H, W] alpha values in [0, 1]
        fg_threshold: minimum alpha for FG classification
        bg_sample_ratio: fraction of BG Gaussians to keep

    Returns:
        fg_mask: bool[N] — FG core + boundary
        bg_sample_mask: bool[N] — sampled BG
        weights: float[N] — per-Gaussian training weights
    """
    alpha = rendered_alpha.flatten()
    n_total = alpha.shape[0]

    # FG: alpha > threshold
    fg_mask = alpha > fg_threshold

    # BG: random sparse sample
    bg_mask = ~fg_mask
    n_bg = bg_mask.sum().item()
    n_bg_sample = max(1, int(n_bg * bg_sample_ratio))

    bg_indices = torch.where(bg_mask)[0]
    bg_sample_indices = bg_indices[torch.randperm(len(bg_indices))[:n_bg_sample]]
    bg_sample_mask = torch.zeros(n_total, dtype=torch.bool)
    bg_sample_mask[bg_sample_indices] = True

    # Weights: core=1.0, boundary=alpha, bg_sample=0.1
    weights = torch.zeros(n_total)
    core_mask = alpha > 0.8
    boundary_mask = fg_mask & ~core_mask

    weights[core_mask] = 1.0
    weights[boundary_mask] = alpha[boundary_mask]
    weights[bg_sample_mask] = 0.1

    return fg_mask, bg_sample_mask, weights


def extract_and_filter(model, batch, device, fg_threshold, bg_sample_ratio):
    """
    Run GS-LRM inference and filter to FG-aware Gaussians.

    GS-LRM forward returns:
      gaussian_params_raw: edict(xyz, features, scaling, rotation, opacity)
      rendered_alpha: [B, V, 1, H, W] (only if target_data exists)

    For FG classification, we use opacity (sigmoid of logit) since
    rendered_alpha requires a rendering pass with target cameras.

    Returns dict ready for torch.save().
    """
    from easydict import EasyDict as edict

    # Build batch_data in the format GS-LRM expects
    batch_data = edict()
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_data[k] = v.to(device)
        else:
            batch_data[k] = v

    with torch.no_grad():
        output = model(batch_data)

    # Get Gaussian parameters from gaussian_params_raw (edict)
    gp = output.gaussian_params_raw
    xyz = gp.xyz[0]          # [N, 3]
    features = gp.features[0]  # [N, C]
    scaling = gp.scaling[0]    # [N, 3]
    rotation = gp.rotation[0]  # [N, 4]
    opacity = gp.opacity[0]    # [N, 1]

    # FG classification via opacity (sigmoid of logit)
    # rendered_alpha is more accurate but requires target cameras
    alpha_flat = torch.sigmoid(opacity[:, 0])

    # Classify
    fg_mask, bg_sample_mask, weights = classify_gaussians(
        alpha_flat, fg_threshold, bg_sample_ratio
    )

    # Combined mask (FG + BG sample)
    keep_mask = fg_mask | bg_sample_mask

    n_fg = fg_mask.sum().item()
    n_bg = bg_sample_mask.sum().item()
    n_total = xyz.shape[0]

    # Filter and convert to float16
    result = {
        "fg_xyz": xyz[keep_mask].half().cpu(),
        "fg_features": features[keep_mask].half().cpu(),
        "fg_scaling": scaling[keep_mask].half().cpu(),
        "fg_rotation": rotation[keep_mask].half().cpu(),
        "fg_opacity": opacity[keep_mask].half().cpu(),
        "fg_weights": weights[keep_mask].half().cpu(),
        "fg_indices": torch.where(keep_mask)[0].cpu(),
        "fg_mask": fg_mask[keep_mask].cpu(),  # distinguish FG from BG sample
        "metadata": {
            "n_fg": n_fg,
            "n_bg_sample": n_bg,
            "n_total": n_total,
            "fg_ratio": n_fg / n_total,
            "fg_threshold": fg_threshold,
            "bg_sample_ratio": bg_sample_ratio,
        },
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Cache FG-aware Gaussians for Deform V3")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True, help="GS-LRM config yaml (same dir as checkpoint)")
    parser.add_argument("--data_list", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=0, help="0=all")
    parser.add_argument("--fg_threshold", type=float, default=0.2)
    parser.add_argument("--bg_sample_ratio", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config, dataloader = load_model_and_dataloader(
        args.checkpoint, args.config, args.data_list, args.num_frames, args.device
    )

    # Stats tracking
    fg_counts = []
    total_bytes = 0

    logger.info(f"Caching FG Gaussians (threshold={args.fg_threshold}, bg_ratio={args.bg_sample_ratio})")

    for idx, batch in enumerate(tqdm(dataloader, desc="Caching FG Gaussians")):
        try:
            result = extract_and_filter(
                model, batch, args.device, args.fg_threshold, args.bg_sample_ratio
            )

            save_path = output_dir / f"frame_{idx:06d}.pt"
            torch.save(result, save_path)

            fg_counts.append(result["metadata"]["n_fg"])
            total_bytes += save_path.stat().st_size

        except Exception as e:
            logger.error(f"Frame {idx}: {e}")
            continue

    # Summary
    fg_arr = np.array(fg_counts)
    logger.info(
        f"\n{'='*50}\n"
        f"Cache complete: {len(fg_counts)} frames\n"
        f"FG Gaussians: mean={fg_arr.mean():.0f}, "
        f"min={fg_arr.min()}, max={fg_arr.max()}, std={fg_arr.std():.0f}\n"
        f"Total size: {total_bytes / 1024**3:.2f} GB\n"
        f"Per-frame: {total_bytes / len(fg_counts) / 1024**2:.1f} MB\n"
        f"Reduction vs V2: {85 * len(fg_counts) / 1024:.0f} GB → "
        f"{total_bytes / 1024**3:.1f} GB "
        f"({85 * len(fg_counts) * 1024**2 / total_bytes:.0f}× smaller)\n"
        f"{'='*50}"
    )

    # Save summary
    summary = {
        "n_frames": len(fg_counts),
        "fg_stats": {"mean": float(fg_arr.mean()), "min": int(fg_arr.min()),
                     "max": int(fg_arr.max()), "std": float(fg_arr.std())},
        "total_gb": total_bytes / 1024**3,
        "config": vars(args),
    }
    torch.save(summary, output_dir / "cache_summary.pt")


if __name__ == "__main__":
    main()
