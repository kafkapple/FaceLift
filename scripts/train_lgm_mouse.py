#!/usr/bin/env python3
"""
LGM Fine-tuning for Mouse Data

Fine-tunes LGM on mouse 6-view data:
- Input: 4 views [0°, 60°, 180°, 300°] with ray embeddings
- Supervision: All 6 views for rendering loss

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_lgm_mouse.py \
        --data_root data_mouse \
        --resume LGM/pretrained/model_fp16.safetensors \
        --output_dir checkpoints/lgm/mouse
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from safetensors.torch import load_file
import wandb

# Add LGM to path
sys.path.insert(0, str(Path(__file__).parent.parent / "LGM"))

from core.options import Options
from core.models import LGM
from core.provider_mouse import MouseDataset


def parse_args():
    parser = argparse.ArgumentParser(description="LGM Fine-tuning for Mouse")

    # Data
    parser.add_argument("--data_root", type=str, default="data_mouse")
    parser.add_argument("--train_split", type=str, default="data_mouse_train.txt")
    parser.add_argument("--val_split", type=str, default="data_mouse_val.txt")

    # Model
    parser.add_argument("--resume", type=str, default="LGM/pretrained/model_fp16.safetensors")

    # Training
    parser.add_argument("--output_dir", type=str, default="checkpoints/lgm/mouse")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)

    # Freeze options
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze UNet encoder layers")

    # WandB
    parser.add_argument("--wandb_project", type=str, default="mouse_facelift")
    parser.add_argument("--wandb_name", type=str, default="lgm_mouse_finetune")
    parser.add_argument("--wandb_offline", action="store_true")

    return parser.parse_args()


def create_model_and_opt():
    """Create LGM model with 'big' config."""
    opt = Options(
        input_size=256,
        up_channels=(1024, 1024, 512, 256, 128),
        up_attention=(True, True, True, False, False),
        splat_size=128,
        output_size=512,
        batch_size=1,
        num_views=4,
        fovy=49.1,
        znear=0.5,
        zfar=2.5,
        cam_radius=1.5,
    )
    model = LGM(opt)
    return model, opt


def load_pretrained(model, checkpoint_path, accelerator):
    """Load pretrained weights with shape matching."""
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
                accelerator.print(f'[WARN] Shape mismatch: {k} ckpt {v.shape} != model {state_dict[k].shape}')
                skipped += 1
        else:
            skipped += 1

    accelerator.print(f"[INFO] Loaded {loaded} params, skipped {skipped}")
    return model


def compute_loss(model, data, opt, step_ratio=1.0):
    """
    Compute rendering loss for LGM.

    Loss = L1 + LPIPS between rendered views and GT views
    """
    # Forward pass
    images_input = data['input']  # [B, 4, 9, H, W]
    gaussians = model.forward_gaussians(images_input)  # [B, N, 14]

    # Get camera parameters for all 6 views
    cam_view = data['cam_view']  # [B, 6, 4, 4]
    cam_view_proj = data['cam_view_proj']  # [B, 6, 4, 4]
    cam_pos = data['cam_pos']  # [B, 6, 3]
    images_gt = data['images_output']  # [B, 6, 3, H, W]
    masks_gt = data['masks_output']  # [B, 6, 1, H, W]

    B = images_input.shape[0]

    # Render all 6 views
    total_loss = 0
    total_psnr = 0

    for v in range(6):
        # Render this view
        rendered = model.gs.render(
            gaussians,
            cam_view[:, v:v+1],
            cam_view_proj[:, v:v+1],
            cam_pos[:, v:v+1],
            scale_modifier=1.0
        )
        image_pred = rendered['image'].squeeze(1)  # [B, 3, H, W]

        # GT for this view
        image_gt = images_gt[:, v]  # [B, 3, H, W]
        mask_gt = masks_gt[:, v]    # [B, 1, H, W]

        # Resize if needed
        if image_pred.shape[-1] != image_gt.shape[-1]:
            image_pred = F.interpolate(
                image_pred,
                size=image_gt.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        # L1 loss (masked)
        l1_loss = F.l1_loss(image_pred * mask_gt, image_gt * mask_gt)

        # PSNR
        mse = F.mse_loss(image_pred * mask_gt, image_gt * mask_gt)
        psnr = -10 * torch.log10(mse + 1e-8)

        total_loss += l1_loss
        total_psnr += psnr

    # Average over views
    loss = total_loss / 6
    psnr = total_psnr / 6

    return {
        'loss': loss,
        'psnr': psnr,
    }


def train(args):
    # Accelerator
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=1,
    )

    # Model
    accelerator.print("Creating model...")
    model, opt = create_model_and_opt()

    # Load pretrained
    if args.resume:
        accelerator.print(f"Loading pretrained from {args.resume}")
        model = load_pretrained(model, args.resume, accelerator)

    # Freeze encoder if specified
    if args.freeze_encoder:
        accelerator.print("Freezing encoder layers...")
        for name, param in model.named_parameters():
            if 'down' in name or 'mid' in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        accelerator.print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Dataset
    accelerator.print("Loading dataset...")
    train_dataset = MouseDataset(
        opt,
        data_root=args.data_root,
        split_file=args.train_split,
        training=True,
    )
    val_dataset = MouseDataset(
        opt,
        data_root=args.data_root,
        split_file=args.val_split,
        training=False,
    )

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

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.05,
        betas=(0.9, 0.95)
    )

    # Scheduler
    total_steps = args.num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-7
    )

    # Prepare with accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # WandB
    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            mode='offline' if args.wandb_offline else 'online',
        )

    # Training loop
    global_step = 0
    best_psnr = 0

    accelerator.print(f"Starting training for {args.num_epochs} epochs...")

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        epoch_psnr = 0

        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()

            step_ratio = (epoch + batch_idx / len(train_loader)) / args.num_epochs

            with accelerator.autocast():
                out = compute_loss(model, data, opt, step_ratio)
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

                with torch.no_grad():
                    for val_data in val_loader:
                        out = compute_loss(model, val_data, opt)
                        val_loss += out['loss'].item()
                        val_psnr += out['psnr'].item()

                val_loss /= len(val_loader)
                val_psnr /= len(val_loader)

                if accelerator.is_main_process:
                    wandb.log({
                        'val/loss': val_loss,
                        'val/psnr': val_psnr,
                        'val/step': global_step,
                    })
                    accelerator.print(f"  [VAL] loss={val_loss:.4f}, psnr={val_psnr:.2f}")

                    # Save best
                    if val_psnr > best_psnr:
                        best_psnr = val_psnr
                        torch.save(
                            accelerator.unwrap_model(model).state_dict(),
                            output_dir / "best.pt"
                        )
                        accelerator.print(f"  Saved best model (PSNR={best_psnr:.2f})")

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
