"""Train Neural Texture MLP on pre-computed UV maps + GT images.

Takes pre-computed UV maps (from precompute_uv_maps.py) and GT camera images,
trains an MLP to predict RGB from UV coordinates.

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m \
        mouse_extensions.scripts.neural_texture.train \
        --uv-dir outputs/analysis/mouse/neural_texture/uv_maps \
        --epochs 200 \
        --lr 1e-4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from mouse_extensions.model.neural_texture import build_neural_texture
from mouse_extensions.paths import M5_DATA


class UVTextureDataset(Dataset):
    """Dataset of (UV_map, GT_image, mask) triplets.

    Each item provides:
        uv_coords: (N, 2) UV coordinates for valid pixels
        gt_rgb: (N, 3) GT RGB values for those pixels
        pose: (66,) flattened keypoint positions (optional)
    """

    def __init__(
        self,
        uv_dir: str,
        m5_dir: str,
        kp_path: str | None = None,
        split: str = "train",
        train_ratio: float = 0.8,
    ):
        self.uv_dir = Path(uv_dir)
        self.m5_dir = Path(m5_dir)

        # Discover UV map files
        uv_files = sorted(self.uv_dir.glob("uv_*.npz"))
        if not uv_files:
            raise ValueError(f"No UV maps found in {uv_dir}")

        # Split by frame (not by camera view — avoid data leakage)
        frames = sorted(set(f.stem.split("_")[1] for f in uv_files))
        n_train = int(len(frames) * train_ratio)
        train_frames = set(frames[:n_train])
        val_frames = set(frames[n_train:])

        target_frames = train_frames if split == "train" else val_frames
        self.uv_files = [
            f for f in uv_files
            if f.stem.split("_")[1] in target_frames
        ]
        print(f"  {split}: {len(self.uv_files)} UV maps "
              f"({len(target_frames)} frames)")

        # Load keypoints if available
        self.kp_data = None
        self.kp_frame_indices = None
        if kp_path and Path(kp_path).exists():
            kp = np.load(kp_path)
            self.kp_data = kp["keypoints"]  # (N_frames, 22, 3)
            self.kp_frame_indices = kp["frame_indices"]  # MAMMAL frame indices
            print(f"  Keypoints loaded: {self.kp_data.shape}")

    def __len__(self):
        return len(self.uv_files)

    def __getitem__(self, idx):
        # Load UV map
        data = np.load(self.uv_files[idx])
        uv_map = data["uv_map"]      # (H, W, 2)
        mask = data["mask"]           # (H, W)
        m5_frame = int(data["m5_frame"])
        cam_idx = int(data["cam_idx"])
        mammal_frame = int(data["mammal_frame"])

        # Load GT image
        gt_path = self.m5_dir / f"{m5_frame:06d}" / "images" / f"cam_{cam_idx:03d}.png"
        gt_img = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
        if gt_img.shape[-1] == 4:
            gt_img = gt_img[:, :, :3]  # drop alpha

        # Extract valid pixels (where mesh is visible)
        valid_y, valid_x = np.where(mask)
        uv_coords = uv_map[valid_y, valid_x]  # (N, 2)
        gt_rgb = gt_img[valid_y, valid_x]      # (N, 3)

        result = {
            "uv": torch.from_numpy(uv_coords).float(),
            "rgb": torch.from_numpy(gt_rgb).float(),
            "n_pixels": len(valid_y),
            "m5_frame": m5_frame,
            "cam_idx": cam_idx,
        }

        # Add pose if available
        if self.kp_data is not None:
            kp_idx = np.where(self.kp_frame_indices == mammal_frame)[0]
            if len(kp_idx) > 0:
                kp = self.kp_data[kp_idx[0]]  # (22, 3) in MAMMAL mm
                # Normalize to roughly [-1, 1] for MLP input
                kp_norm = (kp - kp.mean(0)) / (kp.std() + 1e-8)
                result["pose"] = torch.from_numpy(kp_norm.flatten()).float()

        return result


def collate_fn(batch):
    """Custom collate: stack variable-length pixel arrays."""
    # Concatenate all pixels from all items
    all_uv = torch.cat([b["uv"] for b in batch], dim=0)
    all_rgb = torch.cat([b["rgb"] for b in batch], dim=0)

    result = {"uv": all_uv, "rgb": all_rgb}

    if "pose" in batch[0]:
        # Repeat each pose for its number of pixels
        poses = []
        for b in batch:
            if "pose" in b:
                poses.append(b["pose"].unsqueeze(0).expand(b["n_pixels"], -1))
        if poses:
            result["pose"] = torch.cat(poses, dim=0)

    return result


def train_epoch(model, loader, optimizer, device, use_pose):
    model.train()
    total_loss = 0
    total_pixels = 0

    for batch in loader:
        uv = batch["uv"].to(device)
        gt = batch["rgb"].to(device)
        pose = batch.get("pose")
        if pose is not None:
            pose = pose.to(device)

        pred = model(uv, pose if use_pose else None)
        loss = F.l1_loss(pred, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(uv)
        total_pixels += len(uv)

    return total_loss / max(total_pixels, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, use_pose):
    model.eval()
    total_loss = 0
    total_pixels = 0

    for batch in loader:
        uv = batch["uv"].to(device)
        gt = batch["rgb"].to(device)
        pose = batch.get("pose")
        if pose is not None:
            pose = pose.to(device)

        pred = model(uv, pose if use_pose else None)
        loss = F.l1_loss(pred, gt)

        total_loss += loss.item() * len(uv)
        total_pixels += len(uv)

    return total_loss / max(total_pixels, 1)


@torch.no_grad()
def render_validation_image(
    model, uv_file: Path, m5_dir: Path, device, use_pose, kp_data=None, kp_fi=None
):
    """Render a full image from trained neural texture for visualization."""
    data = np.load(uv_file)
    uv_map = data["uv_map"]
    mask = data["mask"]
    m5_frame = int(data["m5_frame"])
    cam_idx = int(data["cam_idx"])
    mammal_frame = int(data["mammal_frame"])

    H, W = mask.shape
    valid_y, valid_x = np.where(mask)
    uv_coords = torch.from_numpy(uv_map[valid_y, valid_x]).float().to(device)

    pose = None
    if use_pose and kp_data is not None and kp_fi is not None:
        kp_idx = np.where(kp_fi == mammal_frame)[0]
        if len(kp_idx) > 0:
            kp = kp_data[kp_idx[0]]
            kp_norm = (kp - kp.mean(0)) / (kp.std() + 1e-8)
            pose = torch.from_numpy(kp_norm.flatten()).float().to(device)
            pose = pose.unsqueeze(0).expand(len(uv_coords), -1)

    model.eval()
    pred_rgb = model(uv_coords, pose).cpu().numpy()

    # Reconstruct image
    render = np.ones((H, W, 3), dtype=np.float32)  # white background
    render[valid_y, valid_x] = pred_rgb

    # Load GT for comparison
    gt_path = m5_dir / f"{m5_frame:06d}" / "images" / f"cam_{cam_idx:03d}.png"
    gt = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
    if gt.shape[-1] == 4:
        gt = gt[:, :, :3]

    # Side-by-side: GT | Neural Texture
    comparison = np.concatenate([gt, render], axis=1)
    return (comparison * 255).clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Train Neural Texture MLP")
    parser.add_argument("--uv-dir", type=str, default="outputs/analysis/mouse/neural_texture/uv_maps")
    parser.add_argument("--m5-dir", type=str,
                        default=str(M5_DATA))
    parser.add_argument("--kp-path", type=str,
                        default="/home/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz")
    parser.add_argument("--output-dir", type=str, default="outputs/analysis/mouse/neural_texture/checkpoints")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-freqs", type=int, default=8)
    parser.add_argument("--use-pose", action="store_true")
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Dataset
    print("Loading dataset...")
    train_ds = UVTextureDataset(
        args.uv_dir, args.m5_dir, args.kp_path,
        split="train", train_ratio=0.8,
    )
    val_ds = UVTextureDataset(
        args.uv_dir, args.m5_dir, args.kp_path,
        split="val", train_ratio=0.8,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # Model
    model = build_neural_texture(
        use_pose=args.use_pose,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_freqs=args.num_freqs,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Save config
    config = vars(args)
    config["param_count"] = param_count
    json.dump(config, open(output_dir / "config.json", "w"), indent=2)

    # Training loop
    best_val_loss = float("inf")
    print(f"\nTraining for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, args.use_pose)
        val_loss = eval_epoch(model, val_loader, device, args.use_pose)
        scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train_L1={train_loss:.4f}  val_L1={val_loss:.4f}  "
                  f"lr={lr:.2e}  ({dt:.1f}s)")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
            }, output_dir / "best.pt")

        # Periodic checkpoint + visualization
        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, output_dir / f"ckpt_{epoch:04d}.pt")

            # Render validation sample
            val_uv_files = sorted(Path(args.uv_dir).glob("uv_*.npz"))
            if val_uv_files:
                kp_data = train_ds.kp_data
                kp_fi = train_ds.kp_frame_indices
                vis = render_validation_image(
                    model, val_uv_files[len(val_uv_files) // 2],
                    Path(args.m5_dir), device, args.use_pose,
                    kp_data, kp_fi,
                )
                Image.fromarray(vis).save(vis_dir / f"epoch_{epoch:04d}.png")

    print(f"\nDone. Best val_L1={best_val_loss:.4f}")
    print(f"Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
