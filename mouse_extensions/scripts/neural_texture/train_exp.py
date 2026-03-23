"""Unified experiment training script for neural texture ablation.

Supports variable input dimensions (2D UV / 3D XYZ), model sizes,
and loss configurations. Fixes LPIPS bug from train_v2.

Usage:
    # Smoke test (5 epochs)
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.neural_texture.train_exp \
        --map-dir outputs/analysis/mouse/neural_texture/xyz_maps --epochs 5 --exp-name smoke_xyz

    # Full run
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.neural_texture.train_exp \
        --map-dir outputs/analysis/mouse/neural_texture/xyz_maps --epochs 300 \
        --hidden-dim 256 --num-layers 6 --use-lpips \
        --wandb-project facelift-neural-texture --exp-name xyz_full
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

from mouse_extensions.model.neural_texture import NeuralTextureMLP, FourierEncoder


class CoordTextureDataset(Dataset):
    """Generic coordinate-map → RGB dataset.

    Works with any coordinate map (UV 2D, XYZ 3D) from precompute_maps.py.
    Applies mask intersection: only pixels where mesh AND GT foreground overlap.
    """

    def __init__(self, map_dir, m5_dir, split="train", train_ratio=0.8,
                 bg_threshold=245):
        self.map_dir = Path(map_dir)
        self.m5_dir = Path(m5_dir)
        self.bg_threshold = bg_threshold

        # Load metadata
        meta = json.load(open(self.map_dir / "metadata.json"))
        self.input_dim = meta["input_dim"]
        self.mode = meta.get("mode", "uv8")

        map_files = sorted(self.map_dir.glob("map_*.npz"))
        frames = sorted(set(f.stem.split("_")[1] for f in map_files))
        n_train = int(len(frames) * train_ratio)
        target = set(frames[:n_train] if split == "train" else frames[n_train:])
        self.map_files = [f for f in map_files if f.stem.split("_")[1] in target]
        print(f"  {split}: {len(self.map_files)} maps ({len(target)} frames, "
              f"mode={self.mode}, dim={self.input_dim})")

    def __len__(self):
        return len(self.map_files)

    def __getitem__(self, idx):
        data = np.load(self.map_files[idx])
        coord_map = data["coord_map"]  # (H, W, input_dim)
        mesh_mask = data["mask"]       # (H, W)
        m5_frame = int(data["m5_frame"])
        cam_idx = int(data["cam_idx"])

        # GT image
        gt_path = self.m5_dir / f"{m5_frame:06d}" / "images" / f"cam_{cam_idx:03d}.png"
        gt_img = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
        if gt_img.shape[-1] == 4:
            gt_img = gt_img[:, :, :3]

        # Mask intersection: mesh visible AND GT foreground
        gt_fg = np.any(gt_img < (self.bg_threshold / 255.0), axis=-1)
        valid = mesh_mask & gt_fg

        valid_y, valid_x = np.where(valid)

        return {
            "coords": torch.from_numpy(coord_map[valid_y, valid_x]).float(),
            "rgb": torch.from_numpy(gt_img[valid_y, valid_x]).float(),
            "n_pixels": len(valid_y),
            # Full image data for LPIPS (not concatenated by collate)
            "coord_map": torch.from_numpy(coord_map).float(),
            "gt_img": torch.from_numpy(gt_img).float(),
            "valid_mask": torch.from_numpy(valid),
            "m5_frame": m5_frame,
            "cam_idx": cam_idx,
        }


def collate_mixed(batch):
    """Collate: concatenate pixels + keep first item's image data for LPIPS."""
    result = {
        "coords": torch.cat([b["coords"] for b in batch], 0),
        "rgb": torch.cat([b["rgb"] for b in batch], 0),
    }
    # Keep first item's image-level data for LPIPS (FIX: v2 bug)
    result["coord_map"] = batch[0]["coord_map"]
    result["gt_img"] = batch[0]["gt_img"]
    result["valid_mask"] = batch[0]["valid_mask"]
    return result


def build_model(input_dim, hidden_dim, num_layers, num_freqs, use_pose=False):
    """Build neural texture model with configurable input dim."""
    encoder = FourierEncoder(in_dim=input_dim, num_freqs=num_freqs)
    in_features = encoder.out_dim

    # MLP with skip
    skip_layer = num_layers // 2
    layers_list = nn.ModuleList()
    acts_list = nn.ModuleList()
    for i in range(num_layers):
        if i == 0:
            layers_list.append(nn.Linear(in_features, hidden_dim))
        elif i == skip_layer:
            layers_list.append(nn.Linear(hidden_dim + in_features, hidden_dim))
        else:
            layers_list.append(nn.Linear(hidden_dim, hidden_dim))
        acts_list.append(nn.ReLU())

    rgb_head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(),
    )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.layers = layers_list
            self.acts = acts_list
            self.skip_layer = skip_layer
            self.rgb_head = rgb_head

        def forward(self, x, pose=None):
            feat = self.encoder(x)
            h = feat
            for i, (layer, act) in enumerate(zip(self.layers, self.acts)):
                if i == self.skip_layer:
                    h = torch.cat([h, feat], -1)
                h = act(layer(h))
            return self.rgb_head(h)

    return Model()


def train_step(model, batch, optimizer, device, lpips_fn=None, lpips_w=0.05):
    model.train()
    coords = batch["coords"].to(device)
    gt = batch["rgb"].to(device)

    pred = model(coords)
    l1_loss = F.l1_loss(pred, gt)
    loss = l1_loss
    lpips_val = 0.0

    # LPIPS on full image (FIXED: collate now passes image data)
    if lpips_fn is not None:
        coord_map = batch["coord_map"].to(device)
        gt_img = batch["gt_img"].to(device)
        mask = batch["valid_mask"].to(device)

        if mask.sum() > 100:
            flat_coords = coord_map[mask]
            pred_px = model(flat_coords)
            H, W = mask.shape

            pred_img = torch.ones(H, W, 3, device=device)
            pred_img[mask] = pred_px
            gt_masked = torch.ones(H, W, 3, device=device)
            gt_masked[mask] = gt_img[mask]

            p = pred_img.permute(2, 0, 1).unsqueeze(0) * 2 - 1
            g = gt_masked.permute(2, 0, 1).unsqueeze(0) * 2 - 1
            lpips_val = lpips_fn(p, g).mean()
            loss = loss + lpips_w * lpips_val

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return l1_loss.item(), float(lpips_val)


@torch.no_grad()
def eval_step(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        coords = batch["coords"].to(device)
        gt = batch["rgb"].to(device)
        total += F.l1_loss(model(coords), gt).item() * len(coords)
        n += len(coords)
    return total / max(n, 1)


@torch.no_grad()
def render_image(model, coord_map, mask, device):
    """Render full image from trained model."""
    H, W = mask.shape
    result = np.ones((H, W, 3), dtype=np.uint8) * 255
    if mask.sum() == 0:
        return result
    coords = torch.from_numpy(coord_map[mask]).float().to(device)
    pred = model(coords).cpu().numpy()
    vy, vx = np.where(mask)
    result[vy, vx] = (pred * 255).clip(0, 255).astype(np.uint8)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--m5-dir", default="/home/joon/data/preprocessed/FaceLift_mouse/M5_4")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--exp-name", default="exp")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-freqs", type=int, default=10)
    parser.add_argument("--use-lpips", action="store_true")
    parser.add_argument("--lpips-weight", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--bg-threshold", type=int, default=245)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.output_dir is None:
        args.output_dir = f"outputs/analysis/mouse/neural_texture/runs/{args.exp_name}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(exist_ok=True)

    # wandb
    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.exp_name, config=vars(args))

    # Data
    meta = json.load(open(Path(args.map_dir) / "metadata.json"))
    input_dim = meta["input_dim"]
    print(f"Input: {meta['mode']} (dim={input_dim})")

    train_ds = CoordTextureDataset(args.map_dir, args.m5_dir, "train",
                                   bg_threshold=args.bg_threshold)
    val_ds = CoordTextureDataset(args.map_dir, args.m5_dir, "val",
                                 bg_threshold=args.bg_threshold)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              collate_fn=collate_mixed, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            collate_fn=collate_mixed, num_workers=2, pin_memory=True)

    # Model
    model = build_model(input_dim, args.hidden_dim, args.num_layers, args.num_freqs).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params (hidden={args.hidden_dim}, layers={args.num_layers})")

    # LPIPS
    lpips_fn = None
    if args.use_lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
            print("LPIPS loaded")
        except ImportError:
            print("WARN: lpips not installed")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    json.dump({**vars(args), "input_dim": input_dim, "params": params},
              open(output_dir / "config.json", "w"), indent=2)

    best_val = float("inf")
    print(f"\nTraining {args.exp_name} for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        epoch_l1, epoch_lpips = [], []
        for batch in train_loader:
            l1, lp = train_step(model, batch, optimizer, device, lpips_fn, args.lpips_weight)
            epoch_l1.append(l1)
            epoch_lpips.append(lp)

        train_l1 = np.mean(epoch_l1)
        train_lp = np.mean(epoch_lpips)
        val_l1 = eval_step(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0

        if epoch % max(1, args.epochs // 20) == 0 or epoch <= 5:
            print(f"  E{epoch:3d}/{args.epochs} L1={train_l1:.4f} val={val_l1:.4f} "
                  f"lpips={train_lp:.4f} ({dt:.1f}s)")

        if use_wandb:
            import wandb
            wandb.log({"train/l1": train_l1, "train/lpips": train_lp,
                        "val/l1": val_l1, "epoch": epoch}, step=epoch)

        if val_l1 < best_val:
            best_val = val_l1
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "val_loss": val_l1, "config": vars(args),
                         "input_dim": input_dim}, output_dir / "best.pt")

        if epoch % args.save_every == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "val_loss": val_l1}, output_dir / f"ckpt_{epoch:04d}.pt")

            # Render comparison for first val file
            val_files = sorted(Path(args.map_dir).glob("map_*.npz"))
            if val_files:
                d = np.load(val_files[0])
                vis = render_image(model, d["coord_map"], d["mask"], device)
                gt_p = Path(args.m5_dir) / f"{int(d['m5_frame']):06d}" / "images" / f"cam_{int(d['cam_idx']):03d}.png"
                gt = np.array(Image.open(gt_p))[:, :, :3]
                comp = np.concatenate([gt, vis], axis=1)
                Image.fromarray(comp).save(vis_dir / f"e{epoch:04d}.png")

                if use_wandb:
                    wandb.log({"vis/gt_vs_pred": wandb.Image(
                        Image.fromarray(comp), caption=f"E{epoch} GT|Pred")}, step=epoch)

    print(f"\nDone: {args.exp_name} best_val={best_val:.4f}")
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
