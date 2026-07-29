# no-split: 405 lines before this commit, which only swaps a hardcoded dataset path
# for the paths.py SSOT and adds the import. Splitting a training script to land a
# one-line import would be unrelated churn in a module nobody asked to refactor.
"""Train Neural Texture MLP v2: mask intersection + LPIPS + wandb.

Improvements over v1:
- Only compute loss on pixels where mesh mask AND GT foreground intersect
- LPIPS perceptual loss for sharper textures
- wandb logging with per-epoch comparison images
- Raw texture baseline for visual comparison

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m \
        mouse_extensions.scripts.neural_texture.train_v2 \
        --uv-dir outputs/analysis/mouse/neural_texture/uv_maps \
        --epochs 300 --lr 5e-4 \
        --use-lpips --lpips-weight 0.05 \
        --wandb-project facelift-neural-texture
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


class UVTextureDatasetV2(Dataset):
    """Dataset with mask intersection filtering.

    Only extracts pixels where:
    - Mesh is visible (from UV map mask)
    - GT image shows foreground (non-white, non-background)
    """

    def __init__(self, uv_dir, m5_dir, kp_path=None,
                 split="train", train_ratio=0.8, bg_threshold=245):
        self.uv_dir = Path(uv_dir)
        self.m5_dir = Path(m5_dir)
        self.bg_threshold = bg_threshold

        uv_files = sorted(self.uv_dir.glob("uv_*.npz"))
        if not uv_files:
            raise ValueError(f"No UV maps found in {uv_dir}")

        frames = sorted(set(f.stem.split("_")[1] for f in uv_files))
        n_train = int(len(frames) * train_ratio)
        target = set(frames[:n_train] if split == "train" else frames[n_train:])

        self.uv_files = [f for f in uv_files if f.stem.split("_")[1] in target]
        print(f"  {split}: {len(self.uv_files)} UV maps ({len(target)} frames)")

        self.kp_data = self.kp_fi = None
        if kp_path and Path(kp_path).exists():
            kp = np.load(kp_path)
            self.kp_data = kp["keypoints"]
            self.kp_fi = kp["frame_indices"]

    def __len__(self):
        return len(self.uv_files)

    def __getitem__(self, idx):
        data = np.load(self.uv_files[idx])
        uv_map = data["uv_map"]
        mesh_mask = data["mask"]
        m5_frame = int(data["m5_frame"])
        cam_idx = int(data["cam_idx"])
        mammal_frame = int(data["mammal_frame"])

        # Load GT image
        gt_path = self.m5_dir / f"{m5_frame:06d}" / "images" / f"cam_{cam_idx:03d}.png"
        gt_img = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
        if gt_img.shape[-1] == 4:
            gt_img = gt_img[:, :, :3]

        # GT foreground mask: non-white pixels (mouse vs white/light background)
        gt_fg_mask = np.any(gt_img < (self.bg_threshold / 255.0), axis=-1)

        # INTERSECTION: only pixels where BOTH mesh and GT foreground exist
        valid_mask = mesh_mask & gt_fg_mask

        valid_y, valid_x = np.where(valid_mask)
        uv_coords = uv_map[valid_y, valid_x]
        gt_rgb = gt_img[valid_y, valid_x]

        result = {
            "uv": torch.from_numpy(uv_coords).float(),
            "rgb": torch.from_numpy(gt_rgb).float(),
            "n_pixels": len(valid_y),
            "m5_frame": m5_frame,
            "cam_idx": cam_idx,
            # For image-level LPIPS: store full image data
            "uv_map": torch.from_numpy(uv_map).float(),
            "gt_img": torch.from_numpy(gt_img).float(),
            "valid_mask": torch.from_numpy(valid_mask),
        }

        if self.kp_data is not None:
            kp_idx = np.where(self.kp_fi == mammal_frame)[0]
            if len(kp_idx) > 0:
                kp = self.kp_data[kp_idx[0]]
                kp_norm = (kp - kp.mean(0)) / (kp.std() + 1e-8)
                result["pose"] = torch.from_numpy(kp_norm.flatten()).float()

        return result


def collate_pixel(batch):
    """Collate for pixel-level training."""
    all_uv = torch.cat([b["uv"] for b in batch], 0)
    all_rgb = torch.cat([b["rgb"] for b in batch], 0)
    return {"uv": all_uv, "rgb": all_rgb}


def train_epoch(model, loader, optimizer, device, lpips_fn=None, lpips_w=0.05):
    model.train()
    total_l1 = 0
    total_lpips = 0
    total_px = 0

    for batch in loader:
        uv = batch["uv"].to(device)
        gt = batch["rgb"].to(device)

        pred = model(uv)
        l1_loss = F.l1_loss(pred, gt)

        loss = l1_loss
        lpips_val = 0.0

        # LPIPS on reconstructed images (if available and enough pixels)
        if lpips_fn is not None and "uv_map" in batch:
            # Reconstruct one image from the batch for LPIPS
            uv_map = batch["uv_map"][0].to(device)  # (H, W, 2)
            gt_img = batch["gt_img"][0].to(device)   # (H, W, 3)
            mask = batch["valid_mask"][0].to(device)  # (H, W)

            if mask.sum() > 100:
                H, W = uv_map.shape[:2]
                # Predict full image
                flat_uv = uv_map[mask].float()
                with torch.no_grad():
                    pass  # don't double-backprop through pixel loss
                pred_full = model(flat_uv)

                # Reconstruct image for LPIPS
                pred_img = torch.ones(H, W, 3, device=device)
                pred_img[mask] = pred_full
                gt_masked = torch.ones(H, W, 3, device=device)
                gt_masked[mask] = gt_img[mask]

                # LPIPS expects (B, 3, H, W) in [-1, 1]
                pred_lpips = pred_img.permute(2, 0, 1).unsqueeze(0) * 2 - 1
                gt_lpips = gt_masked.permute(2, 0, 1).unsqueeze(0) * 2 - 1
                lpips_val = lpips_fn(pred_lpips, gt_lpips).mean()
                loss = loss + lpips_w * lpips_val

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_l1 += l1_loss.item() * len(uv)
        total_lpips += float(lpips_val) * len(uv)
        total_px += len(uv)

    return total_l1 / max(total_px, 1), total_lpips / max(total_px, 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_l1 = 0
    total_px = 0
    for batch in loader:
        uv = batch["uv"].to(device)
        gt = batch["rgb"].to(device)
        pred = model(uv)
        total_l1 += F.l1_loss(pred, gt).item() * len(uv)
        total_px += len(uv)
    return total_l1 / max(total_px, 1)


@torch.no_grad()
def render_comparison(model, uv_file, m5_dir, device, texture_img=None):
    """Render comparison: GT | Raw Texture | Neural Texture.

    Returns (H, W*3, 3) side-by-side image.
    """
    import os
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    data = np.load(uv_file)
    uv_map = data["uv_map"]
    mask = data["mask"]
    m5_frame = int(data["m5_frame"])
    cam_idx = int(data["cam_idx"])
    H, W = mask.shape

    # GT image
    gt_path = Path(m5_dir) / f"{m5_frame:06d}" / "images" / f"cam_{cam_idx:03d}.png"
    gt_img = np.array(Image.open(gt_path))[:, :, :3]

    # Raw texture baseline (sample texture_final.png at UV coords)
    raw_render = np.ones((H, W, 3), dtype=np.uint8) * 255
    if texture_img is not None:
        th, tw = texture_img.shape[:2]
        valid_y, valid_x = np.where(mask)
        uv_coords = uv_map[valid_y, valid_x]
        tex_x = np.clip((uv_coords[:, 0] * tw).astype(int), 0, tw - 1)
        tex_y = np.clip(((1 - uv_coords[:, 1]) * th).astype(int), 0, th - 1)
        raw_render[valid_y, valid_x] = texture_img[tex_y, tex_x, :3]

    # Neural texture
    valid_y, valid_x = np.where(mask)
    uv_coords = torch.from_numpy(uv_map[valid_y, valid_x]).float().to(device)
    model.eval()
    pred = model(uv_coords).cpu().numpy()
    neural_render = np.ones((H, W, 3), dtype=np.uint8) * 255
    neural_render[valid_y, valid_x] = (pred * 255).clip(0, 255).astype(np.uint8)

    # Side-by-side: GT | Raw | Neural
    comparison = np.concatenate([gt_img, raw_render, neural_render], axis=1)
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Train Neural Texture v2")
    parser.add_argument("--uv-dir", default="outputs/analysis/mouse/neural_texture/uv_maps")
    parser.add_argument("--m5-dir", default=str(M5_DATA))
    parser.add_argument("--kp-path", default="/home/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz")
    parser.add_argument("--texture-path", default="/home/joon/data/synthetic/mouse_mesh/FaceLift_mouse/texture_final.png")
    parser.add_argument("--output-dir", default="outputs/analysis/mouse/neural_texture/v2")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-freqs", type=int, default=10)
    parser.add_argument("--use-lpips", action="store_true")
    parser.add_argument("--lpips-weight", type=float, default=0.05)
    parser.add_argument("--use-pose", action="store_true")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--bg-threshold", type=int, default=245)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(exist_ok=True)

    # wandb init
    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"neural_tex_v2_{time.strftime('%H%M')}",
            config=vars(args),
        )

    # Dataset
    print("Loading dataset (v2 with mask intersection)...")
    train_ds = UVTextureDatasetV2(
        args.uv_dir, args.m5_dir, args.kp_path,
        split="train", bg_threshold=args.bg_threshold,
    )
    val_ds = UVTextureDatasetV2(
        args.uv_dir, args.m5_dir, args.kp_path,
        split="val", bg_threshold=args.bg_threshold,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_pixel, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pixel, num_workers=2, pin_memory=True,
    )

    # Model
    model = build_neural_texture(
        use_pose=args.use_pose,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_freqs=args.num_freqs,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} params")

    # LPIPS
    lpips_fn = None
    if args.use_lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
            print("LPIPS (VGG) loaded")
        except ImportError:
            print("WARNING: lpips not installed, using L1 only")

    # Raw texture for baseline comparison
    texture_img = None
    if Path(args.texture_path).exists():
        texture_img = np.array(Image.open(args.texture_path))
        print(f"Baseline texture loaded: {texture_img.shape}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    config = vars(args)
    config["param_count"] = param_count
    json.dump(config, open(output_dir / "config.json", "w"), indent=2)

    best_val = float("inf")
    print(f"\nTraining for {args.epochs} epochs (mask intersection + {'LPIPS' if lpips_fn else 'L1 only'})...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_l1, train_lpips = train_epoch(
            model, train_loader, optimizer, device, lpips_fn, args.lpips_weight
        )
        val_l1 = eval_epoch(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Logging
        log_dict = {
            "train/l1": train_l1,
            "train/lpips": train_lpips,
            "val/l1": val_l1,
            "lr": lr,
            "epoch": epoch,
        }

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"L1={train_l1:.4f}  val={val_l1:.4f}  "
                  f"lpips={train_lpips:.4f}  lr={lr:.2e}  ({dt:.1f}s)")

        if use_wandb:
            wandb.log(log_dict, step=epoch)

        # Save best
        if val_l1 < best_val:
            best_val = val_l1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_l1,
                "config": config,
            }, output_dir / "best.pt")

        # Periodic visualization
        if epoch % args.save_every == 0 or epoch == 1:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_l1,
            }, output_dir / f"ckpt_{epoch:04d}.pt")

            # Render comparison for multiple cameras
            val_files = sorted(Path(args.uv_dir).glob("uv_*.npz"))
            if val_files:
                # Pick one frame, render all 6 cameras
                target_frame = val_files[0].stem.split("_")[1]
                frame_files = [f for f in val_files if f.stem.split("_")[1] == target_frame]

                comparisons = []
                for vf in frame_files[:6]:
                    comp = render_comparison(model, vf, args.m5_dir, device, texture_img)
                    comparisons.append(comp)

                if comparisons:
                    # Stack vertically: 6 rows of [GT | Raw | Neural]
                    grid = np.concatenate(comparisons, axis=0)
                    grid_pil = Image.fromarray(grid)
                    grid_pil.save(vis_dir / f"grid_epoch_{epoch:04d}.png")

                    if use_wandb:
                        wandb.log({
                            "vis/comparison_grid": wandb.Image(
                                grid_pil,
                                caption=f"Epoch {epoch}: GT | Raw Texture | Neural (6 views)"
                            ),
                        }, step=epoch)

    print(f"\nDone. Best val_L1={best_val:.4f}")
    print(f"Checkpoints: {output_dir}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
