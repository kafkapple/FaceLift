#!/usr/bin/env python3
"""
LGM 6-View Fine-tuning for Mouse Data

Full pipeline: Single Image → MVDiffusion → 6 Views → LGM → 3D Gaussian

Features:
- Uses ALL 6 MVDiffusion views for maximum 3D quality
- WandB logging with image samples every validation
- Checkpoint saving with best PSNR tracking

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_lgm_mouse_6view.py \
        --data_root data_mouse \
        --resume LGM/pretrained/model_fp16.safetensors \
        --output_dir checkpoints/lgm/mouse_6view
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from safetensors.torch import load_file
import wandb
import numpy as np

# Add LGM to path
sys.path.insert(0, str(Path(__file__).parent.parent / "LGM"))

from core.options import Options
from core.models_6view import LGM6View
from core.provider_mouse_6view import MouseDataset6View


def parse_args():
    parser = argparse.ArgumentParser(description="LGM 6-View Fine-tuning for Mouse")

    # Data
    parser.add_argument("--data_root", type=str, default="data_mouse")
    parser.add_argument("--train_split", type=str, default="data_mouse_train.txt")
    parser.add_argument("--val_split", type=str, default="data_mouse_val.txt")

    # Model
    parser.add_argument("--resume", type=str, default="LGM/pretrained/model_fp16.safetensors")
    parser.add_argument("--num_views", type=int, default=6)

    # Training
    parser.add_argument("--output_dir", type=str, default="checkpoints/lgm/mouse_6view")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)

    # Freeze
    parser.add_argument("--freeze_encoder", action="store_true")

    # WandB
    parser.add_argument("--wandb_project", type=str, default="mouse_facelift")
    parser.add_argument("--wandb_name", type=str, default="lgm_mouse_6view")
    parser.add_argument("--wandb_offline", action="store_true")

    return parser.parse_args()


def create_model_and_opt(num_views=6):
    opt = Options(
        input_size=256,
        up_channels=(1024, 1024, 512, 256, 128),
        up_attention=(True, True, True, False, False),
        splat_size=128,
        output_size=512,
        batch_size=1,
        num_views=num_views,
        num_input_views=num_views,
        fovy=49.1,
        znear=0.5,
        zfar=2.5,
        cam_radius=1.5,
    )
    model = LGM6View(opt, num_input_views=num_views)
    return model, opt


def load_pretrained(model, checkpoint_path, accelerator):
    if checkpoint_path.endswith('safetensors'):
        ckpt = load_file(checkpoint_path, device='cpu')
    else:
        ckpt = torch.load(checkpoint_path, map_location='cpu')

    state_dict = model.state_dict()
    loaded = 0
    skipped = 0

    for k, v in ckpt.items():
        if k in state_dict:
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
                loaded += 1
            else:
                accelerator.print(f'[WARN] Shape mismatch: {k}')
                skipped += 1
        else:
            skipped += 1

    accelerator.print(f"[INFO] Loaded {loaded} params, skipped {skipped}")
    return model


def compute_loss(model, data, opt):
    images_input = data['input']  # [B, 6, 9, H, W]
    gaussians = model.forward_gaussians(images_input)

    cam_view = data['cam_view']
    cam_view_proj = data['cam_view_proj']
    cam_pos = data['cam_pos']
    images_gt = data['images_output']
    masks_gt = data['masks_output']

    B = images_input.shape[0]
    num_views = images_input.shape[1]

    total_loss = 0
    total_psnr = 0
    rendered_images = []

    for v in range(num_views):
        rendered = model.gs.render(
            gaussians,
            cam_view[:, v:v+1],
            cam_view_proj[:, v:v+1],
            cam_pos[:, v:v+1],
            scale_modifier=1.0
        )
        image_pred = rendered['image'].squeeze(1)

        image_gt = images_gt[:, v]
        mask_gt = masks_gt[:, v]

        if image_pred.shape[-1] != image_gt.shape[-1]:
            image_pred = F.interpolate(
                image_pred,
                size=image_gt.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        l1_loss = F.l1_loss(image_pred * mask_gt, image_gt * mask_gt)
        mse = F.mse_loss(image_pred * mask_gt, image_gt * mask_gt)
        psnr = -10 * torch.log10(mse + 1e-8)

        total_loss += l1_loss
        total_psnr += psnr
        rendered_images.append(image_pred.detach())

    loss = total_loss / num_views
    psnr = total_psnr / num_views

    return {
        'loss': loss,
        'psnr': psnr,
        'num_gaussians': gaussians.shape[1],
        'rendered_images': rendered_images,
        'gt_images': images_gt,
    }


def log_images_to_wandb(out, step, prefix="val"):
    """Log rendered vs GT images to WandB."""
    images = []
    
    # Take first sample in batch
    for v in range(min(6, len(out['rendered_images']))):
        # Get predictions and GT, ensure float32 for proper conversion
        pred = out['rendered_images'][v][0].float().cpu().numpy()  # [3, H, W]
        gt = out['gt_images'][0, v].float().cpu().numpy()  # [3, H, W]
        
        # Clamp to 0-1 range first, then scale to 0-255
        pred = np.clip(pred, 0, 1)
        gt = np.clip(gt, 0, 1)
        
        # Convert to [H, W, 3] and scale to 0-255
        pred = (pred.transpose(1, 2, 0) * 255).astype(np.uint8)
        gt = (gt.transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Create side-by-side comparison
        comparison = np.concatenate([gt, pred], axis=1)
        images.append(wandb.Image(comparison, caption=f"View {v}: GT | Pred"))
    
    wandb.log({f"{prefix}/images": images, f"{prefix}/step": step})


def train(args):
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=1,
    )

    accelerator.print("="*60)
    accelerator.print("LGM 6-View Fine-tuning for Mouse")
    accelerator.print("="*60)
    
    model, opt = create_model_and_opt(args.num_views)

    if args.resume:
        accelerator.print(f"Loading pretrained from {args.resume}")
        model = load_pretrained(model, args.resume, accelerator)

    if args.freeze_encoder:
        accelerator.print("Freezing encoder layers...")
        for name, param in model.named_parameters():
            if 'down' in name or 'mid' in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        accelerator.print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    accelerator.print("Loading dataset...")
    train_dataset = MouseDataset6View(
        opt,
        data_root=args.data_root,
        split_file=args.train_split,
        training=True,
    )
    val_dataset = MouseDataset6View(
        opt,
        data_root=args.data_root,
        split_file=args.val_split,
        training=False,
    )

    accelerator.print(f"  Train samples: {len(train_dataset)}")
    accelerator.print(f"  Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.05,
        betas=(0.9, 0.95)
    )

    total_steps = args.num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            mode='offline' if args.wandb_offline else 'online',
        )

    global_step = 0
    best_psnr = 0

    accelerator.print(f"Starting training for {args.num_epochs} epochs...")
    accelerator.print(f"Expected Gaussians: {args.num_views} × 128 × 128 = {args.num_views * 128 * 128:,}")

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        epoch_psnr = 0

        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()

            with accelerator.autocast():
                out = compute_loss(model, data, opt)
                loss = out['loss']

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_psnr += out['psnr'].item()
            global_step += 1

            # Logging
            if global_step % args.log_every == 0 and accelerator.is_main_process:
                lr = scheduler.get_last_lr()[0]
                wandb.log({
                    'train/loss': loss.item(),
                    'train/psnr': out['psnr'].item(),
                    'train/lr': lr,
                    'train/step': global_step,
                    'train/epoch': epoch,
                })
                accelerator.print(
                    f"[{epoch}/{args.num_epochs}] Step {global_step}: "
                    f"loss={loss.item():.4f}, psnr={out['psnr'].item():.2f}, lr={lr:.2e}"
                )

            # Validation
            if global_step % args.val_every == 0:
                model.eval()
                val_loss = 0
                val_psnr = 0
                val_out = None

                with torch.no_grad():
                    for i, val_data in enumerate(val_loader):
                        out = compute_loss(model, val_data, opt)
                        val_loss += out['loss'].item()
                        val_psnr += out['psnr'].item()
                        if i == 0:
                            val_out = out  # Save first batch for visualization

                val_loss /= len(val_loader)
                val_psnr /= len(val_loader)

                if accelerator.is_main_process:
                    wandb.log({
                        'val/loss': val_loss,
                        'val/psnr': val_psnr,
                        'val/step': global_step,
                    })
                    
                    # Log images
                    if val_out is not None:
                        log_images_to_wandb(val_out, global_step, prefix="val")
                    
                    accelerator.print(f"  [VAL] loss={val_loss:.4f}, psnr={val_psnr:.2f}")

                    if val_psnr > best_psnr:
                        best_psnr = val_psnr
                        torch.save(
                            accelerator.unwrap_model(model).state_dict(),
                            output_dir / "best.pt"
                        )
                        accelerator.print(f"  ✓ Saved best model (PSNR={best_psnr:.2f})")

                model.train()

            # Save checkpoint
            if global_step % args.save_every == 0 and accelerator.is_main_process:
                ckpt_path = output_dir / f"ckpt_{global_step:08d}.pt"
                torch.save(
                    accelerator.unwrap_model(model).state_dict(),
                    ckpt_path
                )
                accelerator.print(f"  Saved checkpoint: {ckpt_path}")

        # Epoch summary
        epoch_loss /= len(train_loader)
        epoch_psnr /= len(train_loader)
        accelerator.print(f"Epoch {epoch} complete: avg_loss={epoch_loss:.4f}, avg_psnr={epoch_psnr:.2f}")

    # Final save
    if accelerator.is_main_process:
        torch.save(
            accelerator.unwrap_model(model).state_dict(),
            output_dir / "final.pt"
        )
        accelerator.print(f"Training complete! Final model saved to {output_dir / 'final.pt'}")
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
