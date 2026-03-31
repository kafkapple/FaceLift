#!/usr/bin/env python3
# no-split: unified deformation training orchestrator — config-driven V2/V3/future
"""
Unified Deformation Training Script.

Supports multiple loss modes via config:
  - v3_rendering: Rendering loss (PRIMARY) + ARAP + velocity + L_zero
  - v2_param_mse: Parameter MSE + ARAP + velocity (deprecated)

Usage:
    # V3 (rendering loss, FG-only cache)
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.train_deform \
        --config configs/mouse/deform_v3.yaml

    # V2 compat (param MSE)
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.train_deform \
        --config configs/mouse/deform_v2.yaml

    # Resume
    python -m mouse_extensions.scripts.train_deform \
        --config configs/mouse/deform_v3.yaml --resume
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parents[3]))

from mouse_extensions.model.deformation import (
    DeformationNetworkV2,
    DeformationConfigV2,
    GaussianParams,
)
from gslrm.model.gaussians_renderer import GaussianModel, render_opencv_cam

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """1 - SSIM between two images [C, H, W]."""
    C = img1.shape[0]
    # Simple SSIM via means/vars (fast approximation)
    mu1 = F.avg_pool2d(img1.unsqueeze(0), window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2.unsqueeze(0), window_size, stride=1, padding=window_size // 2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.avg_pool2d((img1 ** 2).unsqueeze(0), window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d((img2 ** 2).unsqueeze(0), window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d((img1 * img2).unsqueeze(0), window_size, stride=1, padding=window_size // 2) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1.0 - ssim.mean()


# ---------------------------------------------------------------------------
# Dataset: FG-only cache pairs
# ---------------------------------------------------------------------------

class FGPairDataset(Dataset):
    """Load consecutive FG Gaussian pairs from cache + target images for rendering loss."""

    def __init__(self, cache_dir: str, data_list: str = "", num_frames: int = 0):
        self.cache_dir = Path(cache_dir)
        self.frame_files = sorted(self.cache_dir.glob("frame_*.pt"))

        if num_frames > 0:
            self.frame_files = self.frame_files[:num_frames]

        self.num_frames = len(self.frame_files)

        # Load UID paths for target image access (rendering loss needs GT images)
        self.uid_paths = []
        if data_list and os.path.exists(data_list):
            with open(data_list) as f:
                self.uid_paths = [line.strip() for line in f if line.strip()]
            if num_frames > 0:
                self.uid_paths = self.uid_paths[:num_frames]

        logger.info(f"FGPairDataset: {self.num_frames} frames, {len(self.uid_paths)} UIDs")

    def __len__(self):
        return max(0, self.num_frames - 1)

    def __getitem__(self, idx):
        g_t = torch.load(self.frame_files[idx], map_location="cpu")
        g_t1 = torch.load(self.frame_files[idx + 1], map_location="cpu")

        # Load target images + cameras for t+1 (rendering loss)
        target = None
        if idx + 1 < len(self.uid_paths):
            target = self._load_target(self.uid_paths[idx + 1])

        return g_t, g_t1, idx, target

    def _load_target(self, uid_path: str) -> Optional[Dict]:
        """Load GT images and cameras for rendering loss computation."""
        import json
        from PIL import Image

        cam_json = os.path.join(uid_path, "opencv_cameras.json")
        if not os.path.exists(cam_json):
            return None

        with open(cam_json) as f:
            data = json.load(f)

        images, c2ws, intrinsics = [], [], []
        for cam in data["frames"]:
            # Load image: RGBA → RGB on white
            img_path = os.path.join(uid_path, cam["file_path"])
            img = Image.open(img_path)
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            img_t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
            images.append(img_t)

            # Camera
            w2c = np.array(cam["w2c"]).reshape(4, 4)
            c2w = torch.from_numpy(np.linalg.inv(w2c).astype(np.float32))
            c2ws.append(c2w)
            intrinsics.append(torch.tensor([cam["fx"], cam["fy"], cam["cx"], cam["cy"]], dtype=torch.float32))

        return {
            "images": torch.stack(images),       # [V, 3, H, W]
            "c2ws": torch.stack(c2ws),           # [V, 4, 4]
            "intrinsics": torch.stack(intrinsics), # [V, 4]
        }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def param_mse_loss(g_t_deformed: Dict, g_t1: Dict, weights: torch.Tensor) -> torch.Tensor:
    """Weighted MSE on Gaussian parameters (V2 primary, V3 auxiliary)."""
    loss = 0.0
    for key in ["fg_xyz", "fg_features", "fg_scaling", "fg_rotation", "fg_opacity"]:
        if key in g_t_deformed and key in g_t1:
            n = min(g_t_deformed[key].shape[0], g_t1[key].shape[0])
            diff = (g_t_deformed[key][:n] - g_t1[key][:n]) ** 2
            loss += (diff.mean(dim=-1) * weights[:n]).mean()
    return loss


def arap_loss(xyz_before: torch.Tensor, xyz_after: torch.Tensor, k: int = 8) -> torch.Tensor:
    """As-Rigid-As-Possible: preserve k-NN edge lengths."""
    n = xyz_before.shape[0]
    if n < k + 1:
        return torch.tensor(0.0, device=xyz_before.device)

    # k-NN on source positions
    dists = torch.cdist(xyz_before, xyz_before)
    _, knn_idx = dists.topk(k + 1, largest=False)
    knn_idx = knn_idx[:, 1:]  # exclude self

    # Edge lengths before/after
    edges_before = xyz_before[knn_idx] - xyz_before.unsqueeze(1)
    edges_after = xyz_after[knn_idx] - xyz_after.unsqueeze(1)

    len_before = edges_before.norm(dim=-1)
    len_after = edges_after.norm(dim=-1)

    return ((len_after - len_before) ** 2).mean()


def velocity_loss(delta_prev: Optional[torch.Tensor], delta_curr: torch.Tensor) -> torch.Tensor:
    """Penalize acceleration (change in velocity between consecutive steps)."""
    if delta_prev is None:
        return torch.tensor(0.0, device=delta_curr.device)
    n = min(delta_prev.shape[0], delta_curr.shape[0])
    return ((delta_curr[:n] - delta_prev[:n]) ** 2).mean()


def zero_regularization(model: nn.Module, n_samples: int = 1000, device: str = "cuda") -> torch.Tensor:
    """Penalize non-zero output for random BG-like positions (replaces BG data)."""
    random_pos = torch.randn(n_samples, 6, device=device) * 2.0
    time_idx = torch.zeros(n_samples, device=device)
    output = model(random_pos, time_idx)
    return output.pow(2).mean()


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    model: DeformationNetworkV2,
    g_t: Dict,
    g_t1: Dict,
    frame_idx: int,
    config: dict,
    delta_prev: Optional[torch.Tensor],
    device: str,
    target: Optional[Dict] = None,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """Single training step. Returns (loss, metrics_dict, delta_current)."""
    losses = config.get("losses", {})
    loss_mode = config.get("loss_mode", "v3_rendering")

    # Move to device + float32
    xyz_t = g_t["fg_xyz"].float().to(device)
    xyz_t1 = g_t1["fg_xyz"].float().to(device)
    weights = g_t["fg_weights"].float().to(device)

    # Ensure same size (FG count varies per frame)
    n = min(xyz_t.shape[0], xyz_t1.shape[0])
    xyz_t, xyz_t1, weights = xyz_t[:n], xyz_t1[:n], weights[:n]

    # Forward: predict deformation
    input_pair = torch.cat([xyz_t, xyz_t1], dim=-1)  # [N, 6]
    time_tensor = torch.full((n,), frame_idx, dtype=torch.float32, device=device)
    delta = model(input_pair, time_tensor)

    # Apply deformation
    parsed = model.parse_output(delta)
    xyz_deformed = xyz_t + parsed.get("position", torch.zeros_like(xyz_t))

    # Losses
    total_loss = torch.tensor(0.0, device=device)
    metrics = {}

    # Param MSE (V2 primary / V3 disabled by default)
    pw = losses.get("param_weight", 0.0)
    if pw > 0:
        g_t_deformed = {k: v.float().to(device) for k, v in g_t.items() if k.startswith("fg_")}
        l_param = param_mse_loss(g_t_deformed, {k: v.float().to(device) for k, v in g_t1.items() if k.startswith("fg_")}, weights)
        total_loss += pw * l_param
        metrics["param_mse"] = l_param.item()

    # ARAP
    aw = losses.get("arap_weight", 0.1)
    if aw > 0:
        k_nn = losses.get("arap_k", 8)
        l_arap = arap_loss(xyz_t, xyz_deformed, k=k_nn)
        total_loss += aw * l_arap
        metrics["arap"] = l_arap.item()

    # Velocity
    vw = losses.get("velocity_weight", 0.05)
    if vw > 0:
        l_vel = velocity_loss(delta_prev, delta)
        total_loss += vw * l_vel
        metrics["velocity"] = l_vel.item()

    # L_zero regularization (replaces BG data)
    zw = losses.get("zero_weight", 0.01)
    if zw > 0:
        n_zero = losses.get("zero_samples", 1000)
        l_zero = zero_regularization(model, n_zero, device)
        total_loss += zw * l_zero
        metrics["l_zero"] = l_zero.item()

    # Rendering loss — differentiable GS rendering → image space
    rw = losses.get("render_weight", 0.0)
    if rw > 0 and target is not None:
        n_views = losses.get("render_views", 3)
        ssim_w = losses.get("render_ssim_weight", 0.2)

        # Build GaussianModel from deformed FG params
        gm = GaussianModel(sh_degree=0)
        # Reshape features for GaussianModel: [N, C] → [N, 1, 3] (sh_degree=0)
        feats = g_t["fg_features"].float().to(device)[:n]
        if feats.dim() == 2:
            feats = feats.reshape(n, -1, 3)
        gm.set_data(
            xyz=xyz_deformed,
            features=feats,
            scaling=g_t["fg_scaling"].float().to(device)[:n],
            rotation=g_t["fg_rotation"].float().to(device)[:n],
            opacity=g_t["fg_opacity"].float().to(device)[:n],
        )

        # Stochastic view sampling
        all_views = list(range(target["images"].shape[0]))
        import random
        selected = random.sample(all_views, min(n_views, len(all_views)))

        l_render = torch.tensor(0.0, device=device)
        for vi in selected:
            gt_img = target["images"][vi].to(device)  # [3, H, W]
            c2w = target["c2ws"][vi].to(device)        # [4, 4]
            fxfycxcy = target["intrinsics"][vi].to(device)  # [4]
            H, W = gt_img.shape[1], gt_img.shape[2]

            rendered = render_opencv_cam(gm, H, W, c2w, fxfycxcy)
            pred_img = rendered["render"]  # [3, H, W]

            l_l1 = F.l1_loss(pred_img, gt_img)
            l_ssim = ssim_loss(pred_img, gt_img)
            l_render += (1.0 - ssim_w) * l_l1 + ssim_w * l_ssim

        l_render /= len(selected)
        total_loss += rw * l_render
        metrics["render"] = l_render.item()

    metrics["total"] = total_loss.item()
    metrics["n_fg"] = n

    return total_loss, metrics, delta.detach()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: dict):
    device = config.get("device", "cuda")

    # Dataset — data_list needed for rendering loss (GT images)
    gslrm_cfg = config.get("gslrm", {})
    gslrm_config_path = gslrm_cfg.get("config", "")
    data_list = ""
    if gslrm_config_path and os.path.exists(gslrm_config_path):
        from omegaconf import OmegaConf
        gslrm_full_config = OmegaConf.load(gslrm_config_path)
        data_list = os.path.expanduser(gslrm_full_config.training.dataset.dataset_path)

    dataset = FGPairDataset(config["cache_dir"], data_list, config.get("num_frames", 0))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Network
    net_config = DeformationConfigV2()
    net_config.hidden_dim = config.get("hidden_dim", 256)
    net_config.num_layers = config.get("num_layers", 8)
    net_config.use_positional_encoding = config.get("use_positional_encoding", True)
    net_config.use_time_embedding = config.get("use_time_embedding", True)
    net_config.predict_rotation = config.get("predict_rotation", False)
    net_config.anisotropic_scale = config.get("anisotropic_scale", False)

    model = DeformationNetworkV2(net_config).to(device)
    logger.info(f"Network: {model}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )

    # Resume
    start_epoch = 0
    output_dir = Path(config.get("output_dir", "checkpoints/deform"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.get("resume", False):
        latest = output_dir / "latest.pt"
        if latest.exists():
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"Resumed from epoch {start_epoch}")

    # WandB
    wandb_cfg = config.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        import wandb
        wandb.init(
            project=wandb_cfg.get("project", "FaceLift-Mouse"),
            name=wandb_cfg.get("name", "deform"),
            tags=wandb_cfg.get("tags", []),
            config=config,
        )

    # Training loop
    num_epochs = config.get("num_epochs", 100)
    save_every = config.get("save_every", 10)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_metrics = []
        delta_prev = None

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            g_t, g_t1, frame_idx, target = batch[0][0], batch[0][1], batch[0][2].item(), batch[0][3]

            optimizer.zero_grad()
            loss, metrics, delta_prev = train_step(
                model, g_t, g_t1, frame_idx, config, delta_prev, device, target
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_metrics.append(metrics)
            pbar.set_postfix(loss=f"{metrics['total']:.4f}", n_fg=metrics["n_fg"])

        # Epoch summary
        avg = {k: np.mean([m[k] for m in epoch_metrics if k in m]) for k in epoch_metrics[0]}
        logger.info(f"Epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in avg.items()))

        if wandb_cfg.get("enabled", False):
            wandb.log({f"train/{k}": v for k, v in avg.items()}, step=epoch)

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            ckpt_path = output_dir / f"epoch_{epoch:04d}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": config,
                "metrics": avg,
            }, ckpt_path)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": config,
            }, output_dir / "latest.pt")
            logger.info(f"Saved checkpoint: {ckpt_path}")

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified Deformation Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["resume"] = args.resume
    logger.info(f"Config: {args.config}, loss_mode={config.get('loss_mode', 'v2_param_mse')}")

    train(config)


if __name__ == "__main__":
    main()
