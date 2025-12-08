#!/usr/bin/env python3
# Copyright 2025 Adobe Inc.
# Modified for Mouse-FaceLift project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mouse-FaceLift Training Script

Adapts the FaceLift GS-LRM training pipeline for mouse multi-view reconstruction.
Key differences from original train_gslrm.py:
- Uses MouseViewDataset instead of RandomViewDataset
- Supports single-view input for 6-view reconstruction
- Optional data augmentation for limited real-world data

Usage:
    # Single GPU (for debugging)
    python train_mouse.py --config configs/mouse_config.yaml

    # Multi-GPU with torchrun
    torchrun --nproc_per_node 4 --nnodes 1 \
        --rdzv_id ${JOB_UUID} --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
        train_mouse.py --config configs/mouse_config.yaml

    # Overfitting test (verify code correctness)
    python train_mouse.py --config configs/mouse_config.yaml --overfit 10

Author: Claude Code (AI-assisted)
Date: 2024-12-04
"""

import argparse
import copy
import datetime
import importlib
import json
import os
import shutil
import time
import traceback
from contextlib import nullcontext
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import wandb
import yaml
from easydict import EasyDict as edict
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from rich import print

# Local imports
from gslrm.model.utils_train import (
    checkpoint_job,
    get_job_overview,
    resume_job,
    configure_lr_scheduler,
    configure_optimizer,
    print_rank0
)


class MouseTrainer:
    """Main trainer class for Mouse-FaceLift model."""

    def __init__(self, config: edict, args: argparse.Namespace):
        self.config = config
        self.args = args

        # Handle single-GPU vs multi-GPU
        self.is_distributed = self._check_distributed()

        if self.is_distributed:
            self.setup_distributed()
        else:
            self.setup_single_gpu()

        self.setup_cuda()

        # Training state
        self.fwdbwd_pass_step = 0
        self.param_update_step = 0
        self.start_fwdbwd_pass_step = 0

        # Initialize components
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        self.dataloader = None
        self.val_dataloader = None

    def _check_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return "RANK" in os.environ and "WORLD_SIZE" in os.environ

    def setup_distributed(self):
        """Initialize distributed training."""
        init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
        self.ddp_rank = int(os.environ["RANK"])
        self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
        self.ddp_local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        self.ddp_world_size = int(os.environ["WORLD_SIZE"])
        self.ddp_node_rank = int(os.environ.get("GROUP_RANK", 0))

        print_rank0(
            f"Process {self.ddp_rank}/{self.ddp_world_size} is using device "
            f"{self.ddp_local_rank}/{self.ddp_local_world_size} on node {self.ddp_node_rank}"
        )

    def setup_single_gpu(self):
        """Setup for single GPU training."""
        self.ddp_rank = 0
        self.ddp_local_rank = 0
        self.ddp_local_world_size = 1
        self.ddp_world_size = 1
        self.ddp_node_rank = 0
        print("[MouseTrainer] Running in single-GPU mode")

    def setup_cuda(self):
        """Setup CUDA device and optimization settings."""
        self.device = f"cuda:{self.ddp_local_rank}"
        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        torch.manual_seed(777 + self.ddp_rank)

        # TF32 optimization
        torch.backends.cuda.matmul.allow_tf32 = self.config.training.runtime.use_tf32
        torch.backends.cudnn.allow_tf32 = self.config.training.runtime.use_tf32

        if self.is_distributed:
            torch.distributed.barrier()

    def load_datasets(self):
        """Load training and validation datasets using MouseViewDataset."""
        # Import mouse-specific dataset
        from gslrm.data.mouse_dataset import MouseViewDataset

        # Create training dataset
        self.dataset = MouseViewDataset(self.config, split="train")

        # Handle overfitting test
        if self.args.overfit and self.args.overfit > 0:
            print_rank0(f"[Overfit Test] Using only {self.args.overfit} samples")
            indices = list(range(min(self.args.overfit, len(self.dataset))))
            self.dataset = Subset(self.dataset, indices)

        # Create validation dataset if enabled
        if self.config.validation.enabled:
            self.val_dataset = MouseViewDataset(self.config, split="val")
            if self.args.overfit and self.args.overfit > 0:
                val_indices = list(range(min(self.args.overfit, len(self.val_dataset))))
                self.val_dataset = Subset(self.val_dataset, val_indices)
        else:
            self.val_dataset = None

        self._log_dataset_examples()
        self._setup_dataloaders()

    def _log_dataset_examples(self):
        """Log example data for debugging."""
        if self.ddp_rank != 0:
            return

        print("[MouseTrainer] Dataset loaded! Example data:")
        sample = self.dataset[0] if not isinstance(self.dataset, Subset) else self.dataset.dataset[0]
        for k, v in sample.items():
            try:
                print(f"  {k}: {v.shape}")
            except:
                print(f"  {k}: {type(v)}")

        self._save_data_examples()

    def _save_data_examples(self):
        """Save example images for visual inspection."""
        from einops import rearrange
        import numpy as np
        from PIL import Image

        examples_dir = os.path.join(
            self.config.training.checkpointing.checkpoint_dir, "data_examples"
        )
        os.makedirs(examples_dir, exist_ok=True)

        # Save example image
        sample = self.dataset[0] if not isinstance(self.dataset, Subset) else self.dataset.dataset[0]
        im = sample["image"]
        im = rearrange(im, "v c h w -> h (v w) c").detach().cpu().numpy()

        # Handle RGBA vs RGB
        if im.shape[-1] >= 3:
            im_rgb = (im[..., :3] * 255).astype(np.uint8)
            Image.fromarray(im_rgb).save(os.path.join(examples_dir, "image_rgb.png"))

        if im.shape[-1] == 4:
            im_rgba = (im * 255).astype(np.uint8)
            Image.fromarray(im_rgba).save(os.path.join(examples_dir, "image_rgba.png"))

        print(f"[MouseTrainer] Saved example images to {examples_dir}")

    def _setup_dataloaders(self):
        """Setup data loaders for training and validation."""
        # Training dataloader
        if self.is_distributed:
            datasampler = DistributedSampler(self.dataset)
        else:
            datasampler = None

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.training.dataloader.batch_size_per_gpu,
            shuffle=(datasampler is None),
            num_workers=self.config.training.dataloader.num_workers,
            persistent_workers=True if self.config.training.dataloader.num_workers > 0 else False,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=self.config.training.dataloader.prefetch_factor if self.config.training.dataloader.num_workers > 0 else None,
            sampler=datasampler,
        )
        self.dataloader_iter = iter(self.dataloader)

        # Validation dataloader
        if self.val_dataset is not None:
            if self.is_distributed:
                val_datasampler = DistributedSampler(self.val_dataset)
            else:
                val_datasampler = None

            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config.training.dataloader.batch_size_per_gpu,
                shuffle=False,
                num_workers=self.config.training.dataloader.num_workers,
                persistent_workers=True if self.config.training.dataloader.num_workers > 0 else False,
                pin_memory=True,
                drop_last=True,
                prefetch_factor=self.config.training.dataloader.prefetch_factor if self.config.training.dataloader.num_workers > 0 else None,
                sampler=val_datasampler,
            )
            self.val_dataloader_iter = iter(self.val_dataloader)

    def setup_model(self):
        """Initialize the model."""
        # Download VGG model for LPIPS if needed
        if self.ddp_rank == 0 and self.config.training.losses.lpips_loss_weight > 0.0:
            import lpips
            lpips_fn = lpips.LPIPS(net="vgg")
            del lpips_fn

        if self.is_distributed:
            torch.distributed.barrier()

        # Dynamic model import
        module, class_name = self.config.model.class_name.rsplit(".", 1)
        GSLRM = importlib.import_module(module).__dict__[class_name]
        self.model = GSLRM(self.config).to(self.device)

        # Wrap with DDP if distributed
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

    def setup_optimization(self):
        """Setup optimizer, scheduler, and gradient scaler."""
        # Get actual model for parameter access
        model_module = self.model.module if self.is_distributed else self.model

        # Get job overview for scheduling
        self.job_overview = get_job_overview(
            num_gpus=self.ddp_world_size,
            num_epochs=self.config.training.schedule.num_epochs,
            num_train_samples=len(self.dataset),
            batch_size_per_gpu=self.config.training.dataloader.batch_size_per_gpu,
            gradient_accumulation_steps=self.config.training.runtime.grad_accum_steps,
            max_fwdbwd_passes=self.config.training.schedule.get("max_fwdbwd_passes", int(1e10)),
        )
        print_rank0(self.job_overview)

        # Setup optimizer
        self.optimizer, self.optim_param_dict, self.all_param_dict = configure_optimizer(
            self.model,
            self.config.training.optimizer.weight_decay,
            self.config.training.optimizer.lr,
            (self.config.training.optimizer.beta1, self.config.training.optimizer.beta2),
        )
        self.optim_param_list = list(self.optim_param_dict.values())

        # Log optimizer overview
        if self.ddp_rank == 0:
            optimizer_overview = edict(
                num_optim_params=sum(p.numel() for n, p in self.optim_param_dict.items()),
                num_all_params=sum(p.numel() for n, p in self.all_param_dict.items()),
                optim_param_names=list(self.optim_param_dict.keys())[:10],  # First 10
                freeze_param_names=list(set(self.all_param_dict.keys()) - set(self.optim_param_dict.keys()))[:10],
            )
            print(f"[MouseTrainer] Optimizer overview: {optimizer_overview}")

        # Setup scheduler
        self.lr_scheduler = configure_lr_scheduler(
            self.optimizer,
            self.job_overview.num_param_updates,
            self.config.training.schedule.warmup,
            scheduler_type="cosine",
        )

        # Setup gradient scaler for mixed precision
        enable_grad_scaler = (
            self.config.training.runtime.use_amp and
            self.config.training.runtime.amp_dtype == "fp16"
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=enable_grad_scaler)
        self.amp_dtype_mapping = {"fp16": torch.float16, "bf16": torch.bfloat16}
        print_rank0(f"[MouseTrainer] Grad scaler enabled: {enable_grad_scaler}")

    def load_checkpoint(self):
        """Load model checkpoint if available."""
        # Try loading from different sources in order of priority
        for try_load_path in [
            self.config.training.checkpointing.checkpoint_dir,
            self.args.load,
            self.config.training.checkpointing.get("resume_ckpt", ""),
        ]:
            if not try_load_path:
                continue

            print(f"[MouseTrainer] Trying to load checkpoint from: {try_load_path}")

            reset_training_state = (
                self.config.training.optimizer.get("reset_training_state", False) and
                try_load_path == self.config.training.checkpointing.get("resume_ckpt", "")
            )

            (self.optimizer, self.lr_scheduler,
             self.fwdbwd_pass_step, self.param_update_step) = resume_job(
                try_load_path,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.job_overview,
                self.config.training.schedule.warmup,
                self.config.training.optimizer.reset_lr,
                self.config.training.optimizer.reset_weight_decay,
                reset_training_state,
            )

            if self.fwdbwd_pass_step > 0:
                break

        self.start_fwdbwd_pass_step = self.fwdbwd_pass_step

        print_rank0(
            f"[MouseTrainer] Before training: fwdbwd_pass_step={self.fwdbwd_pass_step}, "
            f"param_update_step={self.param_update_step}, "
            f"lr={self.optimizer.param_groups[0]['lr']:.6f}"
        )

    def setup_wandb(self):
        """Setup Weights & Biases logging."""
        if self.ddp_rank != 0 or self.config.inference.enabled:
            return

        # Setup wandb environment
        if self.config.training.logging.wandb.offline:
            os.environ["WANDB_MODE"] = "offline"

        try:
            wandb.login()
        except Exception as e:
            print(f"[MouseTrainer] Warning: Could not login to wandb: {e}")
            return

        # Prepare config for logging
        config_copy = copy.deepcopy(self.config)
        config_copy["job_overview"] = self.job_overview

        model_module = self.model.module if self.is_distributed else self.model
        config_copy["model_overview"] = model_module.get_overview()

        # Create wandb directory
        wandb_dir = "wandb_logs"
        os.makedirs(wandb_dir, exist_ok=True)

        # Initialize wandb
        wandb.init(
            project=self.config.training.logging.wandb.project,
            name=self.config.training.logging.wandb.exp_name,
            group=self.config.training.logging.wandb.group,
            job_type=self.config.training.logging.wandb.job_type,
            config=config_copy,
            dir=wandb_dir,
        )

        wandb.run.log_code(".")
        self._save_config_files()

    def _save_config_files(self):
        """Save configuration files to checkpoint directory."""
        checkpoint_dir = self.config.training.checkpointing.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        to_regular_dict = lambda x: json.loads(json.dumps(x))

        model_module = self.model.module if self.is_distributed else self.model

        config_files = [
            ("config.yaml", self.config),
            ("job_overview.yaml", self.job_overview),
            ("model_overview.yaml", model_module.get_overview()),
        ]

        for filename, data in config_files:
            with open(os.path.join(checkpoint_dir, filename), "w") as f:
                yaml.dump(to_regular_dict(data), f)

        print("[MouseTrainer] Config files saved")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[Any, bool, bool]:
        """Execute a single training step."""
        # Determine what to create this step
        create_visual = (
            self.fwdbwd_pass_step == self.start_fwdbwd_pass_step or
            self.fwdbwd_pass_step % self.config.training.logging.vis_every == 0
        )

        create_val = (
            self.config.validation.enabled and (
                self.fwdbwd_pass_step == self.start_fwdbwd_pass_step or
                self.fwdbwd_pass_step % self.config.validation.val_every == 0
            )
        )

        # Forward pass with gradient accumulation context
        if self.is_distributed:
            ctx = (
                nullcontext()
                if (self.fwdbwd_pass_step + 1) % self.config.training.runtime.grad_accum_steps == 0
                else self.model.no_sync()
            )
        else:
            ctx = nullcontext()

        with ctx, torch.autocast(
            enabled=self.config.training.runtime.use_amp,
            device_type="cuda",
            dtype=self.amp_dtype_mapping[self.config.training.runtime.amp_dtype],
        ):
            # Set current step for the model
            model_module = self.model.module if self.is_distributed else self.model
            try:
                model_module.set_current_step(
                    self.fwdbwd_pass_step,
                    self.start_fwdbwd_pass_step,
                    self.job_overview.num_fwdbwd_passes
                )
            except:
                pass

            result = self.model(batch, create_visual=create_visual)

        # Backward pass
        loss = result.loss_metrics.loss / self.config.training.runtime.grad_accum_steps
        self.scaler.scale(loss).backward()
        self.fwdbwd_pass_step += 1

        return result, create_visual, create_val

    def optimizer_step(self) -> bool:
        """Execute optimizer step with gradient clipping."""
        should_step = (
            self.fwdbwd_pass_step % self.config.training.runtime.grad_accum_steps == 0
        )

        if should_step:
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.optim_param_list,
                max_norm=self.config.training.runtime.grad_clip_norm
            )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            self.param_update_step += 1

        return should_step

    def log_metrics(self, result, create_visual: bool, batch_time: float, batch=None):
        """Log training metrics."""
        if self.ddp_rank != 0:
            return

        # Print to console
        if self.fwdbwd_pass_step % self.config.training.logging.print_every == 0:
            lr = self.optimizer.param_groups[0]['lr']
            print(
                f"[Step {self.fwdbwd_pass_step}] "
                f"Loss: {result.loss_metrics.loss.item():.4f}, "
                f"LR: {lr:.6f}, "
                f"Time: {batch_time:.2f}s"
            )

        # Log to wandb
        if (wandb.run is not None and
            self.fwdbwd_pass_step % self.config.training.logging.wandb.log_every == 0):

            log_dict = {
                "train/loss": result.loss_metrics.loss.item(),
                "train/lr": self.optimizer.param_groups[0]['lr'],
                "train/step": self.fwdbwd_pass_step,
            }

            # Add individual loss components
            if hasattr(result.loss_metrics, 'l2_loss'):
                log_dict["train/l2_loss"] = result.loss_metrics.l2_loss.item()
            if hasattr(result.loss_metrics, 'perceptual_loss'):
                log_dict["train/perceptual_loss"] = result.loss_metrics.perceptual_loss.item()
            if hasattr(result.loss_metrics, 'lpips_loss'):
                log_dict["train/lpips_loss"] = result.loss_metrics.lpips_loss.item()

            wandb.log(log_dict, step=self.fwdbwd_pass_step)

        # Save visualizations and log to wandb
        if create_visual and self.ddp_rank == 0:
            model_module = self.model.module if self.is_distributed else self.model
            vis_dir = os.path.join(
                self.config.training.checkpointing.checkpoint_dir,
                "visualizations"
            )
            os.makedirs(vis_dir, exist_ok=True)
            try:
                step_vis_dir = os.path.join(vis_dir, f"step_{self.fwdbwd_pass_step:08d}")
                os.makedirs(step_vis_dir, exist_ok=True)
                model_module.save_visuals(step_vis_dir, result, None)

                # Log images to wandb
                if wandb.run is not None:
                    self._log_visuals_to_wandb(result, batch, step_vis_dir)
            except Exception as e:
                print(f"[MouseTrainer] Warning: Could not save visuals: {e}")

    def _log_visuals_to_wandb(self, result, batch, vis_dir):
        """Log visualization images to wandb as a grid."""
        import numpy as np
        from PIL import Image
        import glob

        try:
            # Find saved visualization images
            vis_files = sorted(glob.glob(os.path.join(vis_dir, "*.jpg"))) + \
                        sorted(glob.glob(os.path.join(vis_dir, "*.png")))

            if not vis_files:
                return

            wandb_images = {}

            # Log individual images
            for vis_file in vis_files[:5]:  # Limit to 5 images
                img_name = os.path.basename(vis_file).replace(".jpg", "").replace(".png", "")
                wandb_images[f"vis/{img_name}"] = wandb.Image(vis_file)

            # Create grid from rendered vs GT if available
            if hasattr(result, 'render') and result.render is not None:
                render = result.render  # [B, V, C, H, W] or [B, V, H, W, C]
                if render.dim() == 5:
                    # Take first batch
                    render = render[0]  # [V, C, H, W] or [V, H, W, C]
                    if render.shape[-1] == 3 or render.shape[-1] == 4:
                        # [V, H, W, C] format
                        render = render.permute(0, 3, 1, 2)  # -> [V, C, H, W]

                    # Create grid: concat all views horizontally
                    V = render.shape[0]
                    grid_images = []
                    for v in range(min(V, 6)):  # Max 6 views
                        img = render[v].detach().cpu()
                        if img.shape[0] == 4:
                            img = img[:3]  # Remove alpha
                        img = (img.clamp(0, 1) * 255).byte()
                        grid_images.append(img)

                    if grid_images:
                        # Concat horizontally
                        grid = torch.cat(grid_images, dim=2)  # [C, H, W*V]
                        grid_np = grid.permute(1, 2, 0).numpy()
                        wandb_images["vis/rendered_views_grid"] = wandb.Image(grid_np)

            # Log GT images if available
            if batch is not None and 'image' in batch:
                gt_images = batch['image'][0]  # [V, C, H, W]
                if gt_images.dim() == 4:
                    grid_images = []
                    V = gt_images.shape[0]
                    for v in range(min(V, 6)):
                        img = gt_images[v].detach().cpu()
                        if img.shape[0] == 4:
                            img = img[:3]
                        img = (img.clamp(0, 1) * 255).byte()
                        grid_images.append(img)

                    if grid_images:
                        grid = torch.cat(grid_images, dim=2)
                        grid_np = grid.permute(1, 2, 0).numpy()
                        wandb_images["vis/gt_views_grid"] = wandb.Image(grid_np)

            if wandb_images:
                wandb.log(wandb_images, step=self.fwdbwd_pass_step)

        except Exception as e:
            print(f"[MouseTrainer] Warning: Could not log visuals to wandb: {e}")

    def validate(self):
        """Run validation."""
        if not self.config.validation.enabled or self.val_dataloader is None:
            return

        model_module = self.model.module if self.is_distributed else self.model
        model_module.eval()

        val_losses = []
        with torch.no_grad(), torch.autocast(
            enabled=self.config.training.runtime.use_amp,
            device_type="cuda",
            dtype=self.amp_dtype_mapping[self.config.training.runtime.amp_dtype],
        ):
            for batch in self.val_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                result = self.model(batch, create_visual=False)
                val_losses.append(result.loss_metrics.loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

        if self.ddp_rank == 0:
            print(f"[Validation] Step {self.fwdbwd_pass_step}, Loss: {avg_val_loss:.4f}")
            if wandb.run is not None:
                wandb.log({"val/loss": avg_val_loss}, step=self.fwdbwd_pass_step)

        model_module.train()

    def save_checkpoint(self):
        """Save model checkpoint."""
        if self.ddp_rank != 0:
            return

        if self.fwdbwd_pass_step % self.config.training.checkpointing.checkpoint_every != 0:
            return

        checkpoint_dir = self.config.training.checkpointing.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_module = self.model.module if self.is_distributed else self.model

        checkpoint_job(
            checkpoint_dir,
            model_module,
            self.optimizer,
            self.lr_scheduler,
            self.fwdbwd_pass_step,
            self.param_update_step,
        )

        print(f"[MouseTrainer] Checkpoint saved at step {self.fwdbwd_pass_step}")

    def train(self):
        """Main training loop."""
        print_rank0("[MouseTrainer] Starting training...")

        model_module = self.model.module if self.is_distributed else self.model
        model_module.train()

        max_steps = self.config.training.schedule.get(
            "max_fwdbwd_passes", int(1e10)
        )

        epoch = 0
        while self.fwdbwd_pass_step < max_steps:
            # Set epoch for distributed sampler
            if self.is_distributed and hasattr(self.dataloader.sampler, 'set_epoch'):
                self.dataloader.sampler.set_epoch(epoch)

            for batch in self.dataloader:
                if self.fwdbwd_pass_step >= max_steps:
                    break

                start_time = time.time()

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Training step
                result, create_visual, create_val = self.train_step(batch)

                # Optimizer step
                self.optimizer_step()

                # Logging
                batch_time = time.time() - start_time
                self.log_metrics(result, create_visual, batch_time, batch)

                # Validation
                if create_val:
                    self.validate()

                # Checkpointing
                self.save_checkpoint()

            epoch += 1

        print_rank0("[MouseTrainer] Training complete!")

        # Final checkpoint
        if self.ddp_rank == 0:
            self.save_checkpoint()

        # Cleanup
        if self.is_distributed:
            destroy_process_group()


def load_config(config_path: str) -> edict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)


def main():
    parser = argparse.ArgumentParser(description="Mouse-FaceLift Training")
    parser.add_argument(
        "--config", type=str, default="configs/mouse_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--load", type=str, default=None,
        help="Path to checkpoint to load"
    )
    parser.add_argument(
        "--overfit", type=int, default=0,
        help="Number of samples for overfitting test (0 = disabled)"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create trainer
    trainer = MouseTrainer(config, args)

    # Setup components
    trainer.load_datasets()
    trainer.setup_model()
    trainer.setup_optimization()
    trainer.load_checkpoint()
    trainer.setup_wandb()

    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
