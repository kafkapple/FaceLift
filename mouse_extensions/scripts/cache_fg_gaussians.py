#!/usr/bin/env python3
# no-split: single-purpose cache generation script — splitting breaks self-contained execution
"""
Cache FG-aware Gaussians from trained GS-LRM for Deformation V3 training.

Replaces V2's full 237GB cache (85MB/frame × 2880) with FG-aware ~8GB cache.
Uses deterministic data loading (NOT RandomViewDataset which does random sampling).

Usage:
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.cache_fg_gaussians \
        --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_6view_alpha03_v3/best_psnr.pt \
        --config /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_6view_alpha03_v3/config.yaml \
        --data_list /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_train.txt \
        --output_dir /node_data/joon/checkpoints/FaceLift/deformation/v3/fg_cache \
        --fg_threshold 0.2 --bg_sample_ratio 0.05
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[3]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic Dataset — loads all 6 views without random sampling
# ---------------------------------------------------------------------------

class DeterministicViewDataset(Dataset):
    """
    Deterministic multi-view loader for GS-LRM cache generation.

    Unlike RandomViewDataset (training), this loads ALL 6 views in fixed order
    with no random sampling, augmentation, or view exclusion.

    Replicates the exact preprocessing from RandomViewDataset:
    - RGBA → RGB white background compositing
    - w2c → c2w inversion
    - Intrinsics resize_ratio scaling
    - [0, 1] float normalization
    """

    def __init__(self, data_list_path: str, target_size: int = 512):
        with open(data_list_path) as f:
            self.uid_paths = [line.strip() for line in f if line.strip()]
        self.target_size = target_size
        logger.info(f"DeterministicViewDataset: {len(self.uid_paths)} UIDs, target_size={target_size}")

    def __len__(self):
        return len(self.uid_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        uid_path = self.uid_paths[idx]
        cam_json_path = os.path.join(uid_path, "opencv_cameras.json")

        with open(cam_json_path) as f:
            data = json.load(f)
        cameras = data["frames"]

        images = []
        c2ws = []
        fxfycxcys = []

        for i, cam in enumerate(cameras):
            # Load image: RGBA → RGB on white background
            img_path = os.path.join(uid_path, cam["file_path"])
            img = Image.open(img_path)

            # Resize if needed
            resize_ratio = self.target_size / img.size[0]
            if img.size[0] != self.target_size:
                img = img.resize((self.target_size, self.target_size), Image.LANCZOS)

            # RGBA → RGB compositing on white
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # To tensor [C, H, W], float32 [0, 1]
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            images.append(img_tensor)

            # Camera: w2c → c2w
            w2c = np.array(cam["w2c"]).reshape(4, 4)
            c2w = np.linalg.inv(w2c).astype(np.float32)
            c2ws.append(torch.from_numpy(c2w))

            # Intrinsics with resize_ratio
            fxfycxcy = np.array([cam["fx"], cam["fy"], cam["cx"], cam["cy"]],
                                dtype=np.float32) * resize_ratio
            fxfycxcys.append(torch.from_numpy(fxfycxcy))

        images = torch.stack(images)       # [V, 3, H, W]
        c2ws = torch.stack(c2ws)           # [V, 4, 4]
        fxfycxcys = torch.stack(fxfycxcys) # [V, 4]

        # Index tensor: [V, 2] = [[view_idx, scene_idx], ...]
        indices = torch.stack([
            torch.arange(len(cameras)),
            torch.full((len(cameras),), idx)
        ], dim=1).long()

        return {
            "image": images,
            "c2w": c2ws,
            "fxfycxcy": fxfycxcys,
            "index": indices,
            "bg_color": torch.tensor([1.0, 1.0, 1.0]),
        }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, config_path: str, device: str):
    """Load GS-LRM model from checkpoint + config yaml."""
    from gslrm.model.gslrm import GSLRM
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    logger.info(f"Config: {config.model.num_input_views}v, "
                f"image_size={config.model.image_tokenizer.image_size}")

    model = GSLRM(config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict",
                  ckpt.get("model", ckpt)))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {checkpoint_path}")

    return model, config


# ---------------------------------------------------------------------------
# FG classification
# ---------------------------------------------------------------------------

def classify_gaussians(alpha, fg_threshold=0.2):
    """
    FG-only classification (no BG storage — δ=0 enforced via L_zero in training).

    FG core:     alpha > 0.8     → weight = 1.0
    FG boundary: threshold < alpha ≤ 0.8 → weight = alpha
    BG:          not stored (enforced via loss regularization instead)
    """
    alpha = alpha.flatten()

    fg_mask = alpha > fg_threshold

    # Weights: core=1.0, boundary=alpha value
    weights = torch.zeros_like(alpha)
    weights[alpha > 0.8] = 1.0
    boundary = fg_mask & (alpha <= 0.8)
    weights[boundary] = alpha[boundary]

    return fg_mask, weights


# ---------------------------------------------------------------------------
# Extract + filter
# ---------------------------------------------------------------------------

def extract_and_filter(model, batch, device, fg_threshold):
    """Run GS-LRM forward, extract Gaussians, classify FG."""
    from easydict import EasyDict as edict

    # Move batch to device, wrap in edict for model.forward()
    batch_data = edict()
    for k, v in batch.items():
        batch_data[k] = v.to(device) if isinstance(v, torch.Tensor) else v

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = model(batch_data)

    # Gaussian parameters from forward output
    gp = output.gaussian_params_raw
    xyz = gp.xyz[0]           # [N, 3]
    features = gp.features[0] # [N, C]
    scaling = gp.scaling[0]   # [N, 3]
    rotation = gp.rotation[0] # [N, 4]
    opacity = gp.opacity[0]   # [N, 1]

    # FG classification via sigmoid(opacity) — move to CPU float32
    alpha_flat = torch.sigmoid(opacity[:, 0]).float().cpu()

    fg_mask, weights = classify_gaussians(alpha_flat, fg_threshold)

    return {
        "fg_xyz": xyz[fg_mask].half().cpu(),
        "fg_features": features[fg_mask].half().cpu(),
        "fg_scaling": scaling[fg_mask].half().cpu(),
        "fg_rotation": rotation[fg_mask].half().cpu(),
        "fg_opacity": opacity[fg_mask].half().cpu(),
        "fg_weights": weights[fg_mask].half().cpu(),
        "fg_indices": torch.where(fg_mask)[0].cpu(),
        "metadata": {
            "n_fg": fg_mask.sum().item(),
            "n_total": xyz.shape[0],
            "fg_ratio": fg_mask.sum().item() / xyz.shape[0],
            "fg_threshold": fg_threshold,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cache FG-aware Gaussians for Deform V3")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_list", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=0, help="0=all")
    parser.add_argument("--fg_threshold", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config = load_model(args.checkpoint, args.config, args.device)
    target_size = config.model.image_tokenizer.image_size

    dataset = DeterministicViewDataset(
        os.path.expanduser(args.data_list), target_size=target_size
    )
    if args.num_frames > 0:
        dataset.uid_paths = dataset.uid_paths[:args.num_frames]

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
                            pin_memory=True)

    fg_counts = []
    total_bytes = 0

    logger.info(f"Caching FG-only Gaussians (threshold={args.fg_threshold}, no BG)")

    for idx, batch in enumerate(tqdm(dataloader, desc="Caching FG Gaussians")):
        try:
            result = extract_and_filter(
                model, batch, args.device, args.fg_threshold
            )
            save_path = output_dir / f"frame_{idx:06d}.pt"
            torch.save(result, save_path)

            fg_counts.append(result["metadata"]["n_fg"])
            total_bytes += save_path.stat().st_size
        except Exception as e:
            logger.error(f"Frame {idx}: {e}")
            import traceback; traceback.print_exc()
            continue

    fg_arr = np.array(fg_counts)
    logger.info(
        f"\n{'='*50}\n"
        f"Cache complete: {len(fg_counts)} frames\n"
        f"FG Gaussians: mean={fg_arr.mean():.0f}, "
        f"min={fg_arr.min()}, max={fg_arr.max()}, std={fg_arr.std():.0f}\n"
        f"Total size: {total_bytes / 1024**3:.2f} GB\n"
        f"Per-frame: {total_bytes / len(fg_counts) / 1024**2:.1f} MB\n"
        f"Reduction vs V2: {85 * len(fg_counts) / 1024:.0f}GB → "
        f"{total_bytes / 1024**3:.1f}GB "
        f"({85 * len(fg_counts) * 1024**2 / max(1, total_bytes):.0f}× smaller)\n"
        f"{'='*50}"
    )

    torch.save({
        "n_frames": len(fg_counts),
        "fg_stats": {"mean": float(fg_arr.mean()), "min": int(fg_arr.min()),
                     "max": int(fg_arr.max()), "std": float(fg_arr.std())},
        "total_gb": total_bytes / 1024**3,
        "config": vars(args),
    }, output_dir / "cache_summary.pt")


if __name__ == "__main__":
    main()
