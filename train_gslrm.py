# Copyright 2025 Adobe Inc.
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
GSLRM Training Script

A clean, well-organized training script for the GSLRM model with support for:
- Distributed training (DDP)
- Mixed precision training (AMP) 
- Checkpointing and resuming
- Validation during training
- Inference and evaluation modes
- Weights & Biases logging
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
from torch.utils.data import DataLoader, DistributedSampler
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


class GSLRMTrainer:
    """Main trainer class for GSLRM model."""
    
    def __init__(self, config: edict, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.setup_distributed()
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
        
    def setup_distributed(self):
        """Initialize distributed training or single-GPU mode."""
        # Check if distributed environment is set up
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # Distributed mode via torchrun
            init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.ddp_node_rank = int(os.environ.get("GROUP_RANK", 0))
            self.single_gpu_mode = False

            print_rank0(
                f"Process {self.ddp_rank}/{self.ddp_world_size} is using device "
                f"{self.ddp_local_rank}/{self.ddp_local_world_size} on node {self.ddp_node_rank}"
            )
        else:
            # Single GPU mode - no distributed
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_local_world_size = 1
            self.ddp_world_size = 1
            self.ddp_node_rank = 0
            self.single_gpu_mode = True

            print("[Single GPU Mode] Running without distributed training")
        
    def setup_cuda(self):
        """Setup CUDA device and optimization settings."""
        self.device = f"cuda:{self.ddp_local_rank}"
        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        torch.manual_seed(777 + self.ddp_rank)

        # TF32 optimization
        torch.backends.cuda.matmul.allow_tf32 = self.config.training.runtime.use_tf32
        torch.backends.cudnn.allow_tf32 = self.config.training.runtime.use_tf32

        self._barrier()

    def _barrier(self):
        """Synchronization barrier - only in distributed mode."""
        if not self.single_gpu_mode:
            torch.distributed.barrier()

    @property
    def model_module(self):
        """Get underlying model (handles both DDP and single GPU modes)."""
        if self.single_gpu_mode:
            return self.model
        else:
            return self.model.module

    def _set_epoch(self, dataloader, epoch):
        """Set epoch for sampler (only for DistributedSampler)."""
        if not self.single_gpu_mode and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)

    def _no_sync(self):
        """Context manager for gradient sync control (DDP only)."""
        if self.single_gpu_mode:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()
        else:
            return self.model.no_sync()

    def load_datasets(self):
        """Load training and validation datasets."""
        # Dataset selection:
        # - MouseViewDataset: Fixed view order [0,1,2,3,4,5], camera normalization
        #   → Required for mouse data (inconsistent camera angles)
        # - RandomViewDataset: Random view sampling each step
        #   → Original FaceLift for human data (uniform 60° spacing)
        #
        # IMPORTANT: Random view order causes Plücker coordinate inconsistency
        # → Model cannot learn consistent spatial relationships → blurry output
        use_mouse_dataset = self.config.get("mouse", {}).get("use_mouse_dataset", False)

        if use_mouse_dataset:
            from gslrm.data.mouse_dataset import MouseViewDataset
            print("Using MouseViewDataset with camera normalization")
            self.dataset = MouseViewDataset(self.config, split="train")
            if self.config.validation.enabled:
                self.val_dataset = MouseViewDataset(self.config, split="val")
            else:
                self.val_dataset = None
        else:
            from gslrm.data.dataset import RandomViewDataset
            print("Using RandomViewDataset (original FaceLift)")
            self.dataset = RandomViewDataset(self.config, split="train")
            if self.config.validation.enabled:
                self.val_dataset = RandomViewDataset(self.config, split="val")
            else:
                self.val_dataset = None
            
        self._log_dataset_examples()
        self._setup_dataloaders()
        
    def _log_dataset_examples(self):
        """Log example data for debugging."""
        if self.ddp_rank != 0:
            return
            
        print("Dataset loaded! Example data:")
        for k, v in self.dataset[0].items():
            try:
                print(f"{k}: {v.shape}")
            except:
                print(f"{k}: {type(v)}")
        
        self._save_data_examples()
        
    def _save_data_examples(self):
        """Save example images for visual inspection."""
        from einops import rearrange
        import numpy as np
        from PIL import Image
        
        examples_dir = os.path.join(self.config.training.checkpointing.checkpoint_dir, "data_examples")
        os.makedirs(examples_dir, exist_ok=True)
        
        # Save example image
        im = self.dataset[0]["image"]
        im = rearrange(im, "v c h w -> h (v w) c").detach().cpu().numpy()
        im = (im[..., :4] * 255).astype(np.uint8)
        Image.fromarray(im).save(os.path.join(examples_dir, "image.png"))
            
    def _setup_dataloaders(self):
        """Setup data loaders for training and validation."""
        # Training dataloader
        if self.single_gpu_mode:
            datasampler = None
            train_shuffle = True
        else:
            datasampler = DistributedSampler(self.dataset)
            train_shuffle = False

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.training.dataloader.batch_size_per_gpu,
            shuffle=train_shuffle,
            num_workers=self.config.training.dataloader.num_workers,
            persistent_workers=True,
            pin_memory=False,
            drop_last=True,
            prefetch_factor=self.config.training.dataloader.prefetch_factor,
            sampler=datasampler,
        )
        self.dataloader_iter = iter(self.dataloader)

        # Validation dataloader
        if self.val_dataset is not None:
            if self.single_gpu_mode:
                val_datasampler = None
                val_shuffle = False
            else:
                val_datasampler = DistributedSampler(self.val_dataset)
                val_shuffle = False

            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config.training.dataloader.batch_size_per_gpu,
                shuffle=val_shuffle,
                num_workers=self.config.training.dataloader.num_workers,
                persistent_workers=True,
                pin_memory=False,
                drop_last=True,
                prefetch_factor=self.config.training.dataloader.prefetch_factor,
                sampler=val_datasampler,
            )
            self.val_dataloader_iter = iter(self.val_dataloader)
            
    def setup_model(self):
        """Initialize the model."""
        # Download VGG model for LPIPS if needed (avoid concurrent downloads)
        if self.ddp_rank == 0 and self.config.training.losses.lpips_loss_weight > 0.0:
            import lpips
            lpips_fn = lpips.LPIPS(net="vgg")
            del lpips_fn
        self._barrier()

        # Dynamic model import
        module, class_name = self.config.model.class_name.rsplit(".", 1)
        GSLRM = importlib.import_module(module).__dict__[class_name]
        self.model = GSLRM(self.config).to(self.device)

        # Apply layer-wise freezing if configured
        self._apply_layer_freeze()

        # Wrap with DDP (only in distributed mode)
        if not self.single_gpu_mode:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

    def _apply_layer_freeze(self):
        """Apply layer-wise freezing to prevent catastrophic forgetting.

        Freezes early transformer layers and patch embedder to preserve
        pretrained knowledge while allowing task-specific adaptation in
        later layers and output heads.
        """
        freeze_config = self.config.training.get("freeze_layers", {})
        if not freeze_config.get("enabled", False):
            return

        freeze_until = freeze_config.get("freeze_transformer_until", 20)
        freeze_patch = freeze_config.get("freeze_patch_embedder", True)
        freeze_pos_embed = freeze_config.get("freeze_positional_embeddings", False)

        frozen_params = 0
        total_params = 0
        frozen_names = []

        for name, param in self.model.named_parameters():
            total_params += param.numel()
            should_freeze = False

            # Freeze patch embedder
            if freeze_patch and "patch_embedder" in name:
                should_freeze = True

            # Freeze early transformer layers
            if "transformer_layers" in name:
                # Extract layer number: transformer_layers.0.xxx -> 0
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if part == "transformer_layers" and i + 1 < len(parts):
                        try:
                            layer_num = int(parts[i + 1])
                            if layer_num < freeze_until:
                                should_freeze = True
                        except ValueError:
                            pass
                        break

            # Freeze positional embeddings (optional)
            if freeze_pos_embed and "position_embeddings" in name:
                should_freeze = True

            # Freeze input layer norm (part of early processing)
            if freeze_patch and "input_layer_norm" in name:
                should_freeze = True

            if should_freeze:
                param.requires_grad = False
                frozen_params += param.numel()
                frozen_names.append(name)

        freeze_ratio = frozen_params / total_params * 100
        print_rank0(f"\n{'='*60}")
        print_rank0(f"Layer-wise Freeze Applied:")
        print_rank0(f"  - Freeze transformer layers 0-{freeze_until-1}: Yes")
        print_rank0(f"  - Freeze patch_embedder: {freeze_patch}")
        print_rank0(f"  - Freeze positional embeddings: {freeze_pos_embed}")
        print_rank0(f"  - Frozen: {frozen_params:,} / {total_params:,} params ({freeze_ratio:.1f}%)")
        print_rank0(f"  - Trainable: {total_params - frozen_params:,} params ({100-freeze_ratio:.1f}%)")
        print_rank0(f"{'='*60}\n")

    def setup_optimization(self):
        """Setup optimizer, scheduler, and gradient scaler."""
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
                optim_param_names=list(self.optim_param_dict.keys()),
                freeze_param_names=list(set(self.all_param_dict.keys()) - set(self.optim_param_dict.keys())),
            )
            print(optimizer_overview)
        
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
        print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")
        
    def load_checkpoint(self):
        """Load model checkpoint if available."""
        # Try loading from different sources in order of priority
        for try_load_path in [
            self.config.training.checkpointing.checkpoint_dir,
            self.args.load,
            self.config.training.checkpointing.get("resume_ckpt", ""),
        ]:
            print(f"try_load_path: {try_load_path}")
            if self.config.training.checkpointing.get("force_resume_ckpt", False):
                try_load_path = self.config.training.checkpointing.resume_ckpt
                
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
            f"Before training: fwdbwd_pass_step={self.fwdbwd_pass_step}, "
            f"param_update_step={self.param_update_step}, "
            f"lr={self.optimizer.param_groups[0]['lr']:.6f}"
        )
        
    def setup_wandb(self):
        """Setup Weights & Biases logging."""
        if self.ddp_rank != 0 or self.config.inference.enabled or self.config.get("evaluation", False):
            return
            
        # Setup wandb environment
        if self.config.training.logging.wandb.offline:
            os.environ["WANDB_MODE"] = "offline"
        
        # Login to wandb (will use environment variable or prompt for login)
        wandb.login()
            
        # Prepare config for logging
        config_copy = copy.deepcopy(self.config)
        config_copy["job_overview"] = self.job_overview
        config_copy["model_overview"] = self.model_module.get_overview()
        
        # Create wandb directory
        wandb_dir = "wandb_logs"
        os.makedirs(wandb_dir, exist_ok=True)
        
        # Initialize wandb with cleaner configuration
        wandb.init(
            project=self.config.training.logging.wandb.project,
            name=self.config.training.logging.wandb.exp_name,
            group=self.config.training.logging.wandb.group,
            job_type=self.config.training.logging.wandb.job_type,
            config=config_copy,
            dir=wandb_dir,
        )
        
        # Log source code
        wandb.run.log_code(".")
        
        # Backup source code
        self._save_config_files()
        
    def _save_config_files(self):
        """Save configuration files to checkpoint directory."""
        checkpoint_dir = self.config.training.checkpointing.checkpoint_dir
        to_regular_dict = lambda x: json.loads(json.dumps(x))
        
        config_files = [
            ("config.yaml", self.config),
            ("job_overview.yaml", self.job_overview),
            ("model_overview.yaml", self.model_module.get_overview()),
        ]
        
        for filename, data in config_files:
            with open(os.path.join(checkpoint_dir, filename), "w") as f:
                yaml.dump(to_regular_dict(data), f)
                
        print("Wandb setup done")
        
    def run_inference(self):
        """Run inference mode."""
        print(f"Running inference; save results to: {self.config.inference.output_dir}")
        
        if self.ddp_rank == 0:
            print("Downloading LPIPS model (rank 0 only)")
            import lpips
            
        self._barrier()
        self._set_epoch(self.dataloader, 0)
        
        self.model.eval()
        with (self._no_sync(), torch.no_grad(), 
              torch.autocast(
                  enabled=self.config.training.runtime.use_amp,
                  device_type="cuda",
                  dtype=self.amp_dtype_mapping[self.config.training.runtime.amp_dtype],
              )):
            for batch in self.dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                result = self.model(batch, create_visual=True)
                self.model_module.save_visuals(
                    self.config.inference.output_dir, result, batch, save_all=True
                )
            torch.cuda.empty_cache()
            
        self._barrier()
        
    def run_evaluation(self):
        """Run evaluation mode."""
        print(f"Running evaluation; save results to: {self.config.evaluation_out_dir}")
        
        if self.ddp_rank == 0:
            print("Downloading LPIPS model (rank 0 only)")
            import lpips
            
        self._barrier()
        self._set_epoch(self.dataloader, 0)
        
        self.model.eval()
        with (self._no_sync(), torch.no_grad(),
              torch.autocast(
                  enabled=self.config.training.runtime.use_amp,
                  device_type="cuda", 
                  dtype=self.amp_dtype_mapping[self.config.training.runtime.amp_dtype],
              )):
            for batch in self.dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                result = self.model(batch, create_visual=False)
                self.model_module.save_evaluations(
                    self.config.evaluation_out_dir, result, batch, self.dataset
                )
            torch.cuda.empty_cache()
            
        self._barrier()
        
        if self.ddp_rank == 0:
            self._summarize_evaluation_results(self.config.evaluation_out_dir)
        
    def _summarize_evaluation_results(self, evaluation_folder: str):
        """Summarize evaluation metrics into a CSV file."""
        # Get all subdirectories
        subfolders = [
            os.path.join(evaluation_folder, o)
            for o in os.listdir(evaluation_folder)
            if os.path.isdir(os.path.join(evaluation_folder, o))
        ]
        
        # Sort by integer if possible, otherwise by string
        subfolders = sorted(
            subfolders,
            key=lambda x: (
                int(os.path.basename(x)) if os.path.basename(x).isdigit()
                else os.path.basename(x)
            ),
        )
        
        # Read metrics from each subfolder
        metrics = {}
        for subfolder in subfolders:
            metrics_file = os.path.join(subfolder, "metrics.txt")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            k, v = line.split(":")
                            v = float(v.strip())
                            if k not in metrics:
                                metrics[k] = []
                            metrics[k].append(v)
                            
        # Write summary CSV
        csv_file = os.path.join(evaluation_folder, "summary.csv")
        with open(csv_file, "w") as f:
            f.write(",".join(["basename"] + list(metrics.keys())) + "\n")
            for i, subfolder in enumerate(subfolders):
                basename = os.path.basename(subfolder)
                f.write(",".join([basename] + [str(v[i]) for v in metrics.values()]) + "\n")
            f.write("\n")
            # Write average
            averages = [str(sum(v) / len(v)) for v in metrics.values()]
            f.write(",".join(["average"] + averages) + "\n")
            
        print(f"Summary written to {csv_file}")
        print(f"Average: {','.join(averages)}")
        
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
        ctx = (
            nullcontext()
            if (self.fwdbwd_pass_step + 1) % self.config.training.runtime.grad_accum_steps == 0
            else self._no_sync()
        )
        
        with ctx, torch.autocast(
            enabled=self.config.training.runtime.use_amp,
            device_type="cuda",
            dtype=self.amp_dtype_mapping[self.config.training.runtime.amp_dtype],
        ):
            # Set current step for the model
            try:
                self.model_module.set_current_step(
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
        
    def optimizer_step(self, result: Any) -> float:
        """Execute optimizer step with gradient clipping and error handling."""
        skip_optimizer_step = False
        total_grad_norm = 0.0
        
        # Check for NaN/inf loss
        if torch.isnan(result.loss_metrics.loss) or torch.isinf(result.loss_metrics.loss):
            print("WARNING: NaN or inf loss encountered, skipping optimizer step")
            skip_optimizer_step = True
            result.loss_metrics.loss.data = torch.zeros_like(result.loss_metrics.loss)
            
        if self.fwdbwd_pass_step % self.config.training.runtime.grad_accum_steps == 0:
            if not skip_optimizer_step:
                # Unscale gradients
                self.scaler.unscale_(self.optimizer)
                
                # Clean NaN/inf gradients
                with torch.no_grad():
                    for n, p in self.optim_param_dict.items():
                        if p.grad is None:
                            print(f"WARNING: step {self.fwdbwd_pass_step} found None grad for {n}")
                        else:
                            p.grad.nan_to_num_(nan=0.0, posinf=1e-3, neginf=-1e-3)
                            
                # Gradient clipping
                if self.config.training.runtime.grad_clip_norm > 0:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.optim_param_list, 
                        max_norm=self.config.training.runtime.grad_clip_norm
                    ).item()
                    
                    # Check if gradient norm is too large
                    max_allowed_norm = (
                        self.config.training.runtime.grad_clip_norm * 
                        self.config.training.runtime.get("allowed_gradnorm_factor", 20)
                    )
                    
                    if total_grad_norm > max_allowed_norm:
                        skip_optimizer_step = True
                        print(f"WARNING: step {self.fwdbwd_pass_step} grad norm too large "
                              f"{total_grad_norm} > {max_allowed_norm}, skipping optimizer step")
                              
                        if self.ddp_rank == 0:
                            wandb.log({"grad_norm": total_grad_norm}, step=self.fwdbwd_pass_step)
                            
            # Update parameters
            if not skip_optimizer_step:
                self.scaler.step(self.optimizer)
                self.param_update_step += 1

            # Always update scaler (even if optimizer step was skipped)
            # This resets the scaler state for the next iteration
            self.scaler.update()
                
            # Update learning rate and reset gradients
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            
        return total_grad_norm
        
    def log_training_metrics(self, result: Any, total_grad_norm: float, iter_time: float):
        """Log training metrics to console and wandb."""
        if self.ddp_rank != 0:
            return
            
        # Extract loss metrics
        loss_name2value = []
        for k, v in result.loss_metrics.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                loss_name2value.append((k, v.item()))
        loss_name2value = sorted(loss_name2value, key=lambda x: x[0])
        loss_values_str = [f"{k}: {v:.6f}" for k, v in loss_name2value]
        
        # Console logging
        if (self.fwdbwd_pass_step % self.config.training.logging.print_every == 0 or
            self.fwdbwd_pass_step < 100 + self.start_fwdbwd_pass_step):
            
            cur_epoch = self.fwdbwd_pass_step // self.job_overview.num_fwdbwd_passes_per_epoch
            print(f"epoch: {cur_epoch}, fwdbwd_pass_step: {self.fwdbwd_pass_step}/"
                  f"{self.job_overview.num_fwdbwd_passes_per_epoch}, time: {iter_time:.6f}, "
                  f"param_update_step: {self.param_update_step}, "
                  f"lr: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"{', '.join(loss_values_str)}")
            
        # Wandb logging
        if (self.fwdbwd_pass_step % self.config.training.logging.wandb.log_every == 0 or
            self.fwdbwd_pass_step < 100 + self.start_fwdbwd_pass_step):

            # Metrics are logged under "metrics/" prefix for clear separation from images
            log_dict = {
                "metrics/iter": self.fwdbwd_pass_step,
                "metrics/fwdbwd_pass_step": self.fwdbwd_pass_step,
                "metrics/param_update_step": self.param_update_step,
                "metrics/lr": self.optimizer.param_groups[0]["lr"],
                "metrics/iter_time": iter_time,
                "metrics/grad_norm": total_grad_norm,
                "metrics/epoch": self.fwdbwd_pass_step // self.job_overview.num_fwdbwd_passes_per_epoch,
            }
            # Loss metrics under "train/" for backward compatibility
            log_dict.update({"train/" + k: v for k, v in loss_name2value})
            wandb.log(log_dict, step=self.fwdbwd_pass_step)
            
    def save_checkpoint_if_needed(self):
        """Save checkpoint if needed."""
        if self.ddp_rank != 0:
            return
            
        save_checkpoint = (
            self.fwdbwd_pass_step % self.config.training.checkpointing.checkpoint_every == 0 or
            self.fwdbwd_pass_step == self.job_overview.num_fwdbwd_passes
        )
        
        if save_checkpoint:
            checkpoint_job(
                self.config.training.checkpointing.checkpoint_dir,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                self.fwdbwd_pass_step,
                self.param_update_step,
            )
            
    def save_visuals_if_needed(self, result: Any, batch: Dict[str, torch.Tensor], create_visual: bool):
        """Save visual outputs if needed."""
        if not create_visual or self.ddp_rank != 0:
            return

        self.model.eval()
        vis_dir = os.path.join(
            self.config.training.checkpointing.checkpoint_dir,
            f"iter_{self.fwdbwd_pass_step:08d}"
        )
        os.makedirs(vis_dir, exist_ok=True)

        self.model_module.save_visuals(vis_dir, result, batch)

        # Log images to WandB
        self._log_visuals_to_wandb(vis_dir, prefix="train")

        torch.cuda.empty_cache()
        self.model.train()

    def _log_visuals_to_wandb(self, vis_dir: str, prefix: str = "train"):
        """
        Log saved visualization images to WandB with detailed view/camera info.

        Images are logged under "images/" prefix to separate them from metrics.
        This creates distinct sections in wandb UI:
        - "images/train/..." for training visualizations
        - "images/val/..." for validation visualizations
        - "train/..." for training loss metrics (separate section)
        - "val/..." for validation metrics (separate section)
        """
        import glob
        from PIL import Image as PILImage

        wandb_images = {}

        # Get camera normalization status
        normalize_cameras = self.config.get("mouse", {}).get("normalize_cameras", False)
        norm_status = "Z-up normalized" if normalize_cameras else "Original coords"

        # Get view configuration
        num_views = self.config.training.dataset.get("num_views", 6)
        num_input_views = self.config.training.dataset.get("num_input_views", 1)

        # Image key prefix: "images/{train or val}/..."
        img_prefix = f"images/{prefix}"

        # Find and log supervision images (GT vs Pred comparison)
        supervision_files = glob.glob(os.path.join(vis_dir, "supervision_*.jpg"))
        for i, f in enumerate(supervision_files[:2]):  # Limit to 2 images
            try:
                img = PILImage.open(f)
                # Extract UIDs from filename
                filename = os.path.basename(f)
                uid_info = filename.replace("supervision_", "").replace(".jpg", "")
                caption = (
                    f"Step {self.fwdbwd_pass_step} | UIDs: {uid_info}\n"
                    f"Top: GT ({num_views} views) | Bottom: Pred\n"
                    f"Camera: {norm_status}"
                )
                wandb_images[f"{img_prefix}/supervision_{i}"] = wandb.Image(img, caption=caption)
            except Exception as e:
                print(f"Warning: Could not load image {f}: {e}")

        # Find and log input images
        input_files = glob.glob(os.path.join(vis_dir, "input_*.jpg"))
        for i, f in enumerate(input_files[:1]):  # Limit to 1 image
            try:
                img = PILImage.open(f)
                filename = os.path.basename(f)
                uid_info = filename.replace("input_", "").replace(".jpg", "")
                caption = (
                    f"Step {self.fwdbwd_pass_step} | UIDs: {uid_info}\n"
                    f"Input: {num_input_views} view(s) | Camera: {norm_status}"
                )
                wandb_images[f"{img_prefix}/input_{i}"] = wandb.Image(img, caption=caption)
            except Exception as e:
                print(f"Warning: Could not load image {f}: {e}")

        # Find turntable images (360° rotation visualization)
        turntable_files = glob.glob(os.path.join(vis_dir, "turntable_*.jpg"))
        for i, f in enumerate(turntable_files[:1]):  # Limit to 1 image
            try:
                img = PILImage.open(f)
                filename = os.path.basename(f)
                uid_info = filename.replace("turntable_", "").replace(".jpg", "")
                caption = (
                    f"Step {self.fwdbwd_pass_step} | UID: {uid_info}\n"
                    f"360° turntable render | Camera: {norm_status}"
                )
                wandb_images[f"{img_prefix}/turntable_{i}"] = wandb.Image(img, caption=caption)
            except Exception as e:
                print(f"Warning: Could not load image {f}: {e}")

        # Find gt_vs_pred images in subdirectories
        gt_pred_files = glob.glob(os.path.join(vis_dir, "**/gt_vs_pred.png"), recursive=True)
        for i, f in enumerate(gt_pred_files[:2]):  # Limit to 2 images
            try:
                img = PILImage.open(f)
                # Extract UID from directory name
                uid_dir = os.path.basename(os.path.dirname(f))
                caption = (
                    f"Step {self.fwdbwd_pass_step} | UID: {uid_dir}\n"
                    f"Top: GT | Bottom: Pred | Views: {num_views}\n"
                    f"Camera: {norm_status}"
                )
                wandb_images[f"{img_prefix}/gt_vs_pred_{i}"] = wandb.Image(img, caption=caption)
            except Exception as e:
                print(f"Warning: Could not load image {f}: {e}")

        # Find aligned_gs (Gaussian opacity/depth) images
        gs_files = glob.glob(os.path.join(vis_dir, "aligned_gs_*.jpg"))
        for i, f in enumerate(gs_files[:1]):  # Limit to 1 image
            try:
                img = PILImage.open(f)
                filename = os.path.basename(f)
                uid_info = filename.replace("aligned_gs_opacity_depth_", "").replace(".jpg", "")
                caption = (
                    f"Step {self.fwdbwd_pass_step} | UID: {uid_info}\n"
                    f"Gaussian opacity & depth | Camera: {norm_status}"
                )
                wandb_images[f"{img_prefix}/gaussian_vis_{i}"] = wandb.Image(img, caption=caption)
            except Exception as e:
                print(f"Warning: Could not load image {f}: {e}")

        if wandb_images:
            wandb.log(wandb_images, step=self.fwdbwd_pass_step)
            print(f"Logged {len(wandb_images)} images to WandB at step {self.fwdbwd_pass_step}")
        
    def run_validation(self):
        """Run validation loop."""
        print(f"Running validation at step {self.fwdbwd_pass_step}; "
              f"save results to: {self.config.validation.output_dir}")
        self._barrier()
        
        self._set_epoch(self.val_dataloader, 0)
        self.model.eval()
        
        with (self._no_sync(), torch.no_grad(),
              torch.autocast(
                  enabled=self.config.training.runtime.use_amp,
                  device_type="cuda",
                  dtype=self.amp_dtype_mapping[self.config.training.runtime.amp_dtype],
              )):
            
            log_val_metrics = {"psnr": [], "ssim": [], "lpips": [], "mask_iou": []}

            for idx, batch in enumerate(self.val_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                result = self.model(batch, create_visual=False)

                try:
                    val_metrics = self.model_module.save_validations(
                        os.path.join(self.config.validation.output_dir, f"iter_{self.fwdbwd_pass_step:08d}"),
                        result,
                        batch,
                        self.dataset,
                        save_img=(idx == 0),
                    )
                    log_val_metrics["psnr"].append(val_metrics["psnr"])
                    log_val_metrics["ssim"].append(val_metrics["ssim"])
                    log_val_metrics["lpips"].append(val_metrics["lpips"])
                    log_val_metrics["mask_iou"].append(val_metrics.get("mask_iou", 0.0))
                except Exception as e:
                    print(f"Error in saving validation results for batch {idx}: {e}")

            # Log validation metrics to wandb
            if self.ddp_rank == 0:
                wandb_log_val_metrics = {
                    "val/psnr": sum(log_val_metrics["psnr"]) / max(len(log_val_metrics["psnr"]), 1),
                    "val/ssim": sum(log_val_metrics["ssim"]) / max(len(log_val_metrics["ssim"]), 1),
                    "val/lpips": sum(log_val_metrics["lpips"]) / max(len(log_val_metrics["lpips"]), 1),
                    "val/mask_iou": sum(log_val_metrics["mask_iou"]) / max(len(log_val_metrics["mask_iou"]), 1),
                }
                wandb.log(wandb_log_val_metrics, step=self.fwdbwd_pass_step)

                # Log validation images to WandB
                val_vis_dir = os.path.join(self.config.validation.output_dir, f"iter_{self.fwdbwd_pass_step:08d}")
                self._log_visuals_to_wandb(val_vis_dir, prefix="val")

            torch.cuda.empty_cache()
            
        self._barrier()
        
        # Summarize validation results
        if self.ddp_rank == 0:
            self._summarize_evaluation_results(self.config.validation.output_dir)
            
        self._barrier()
        self.model.train()
        
    def should_stop_training(self) -> bool:
        """Check if training should stop based on configured criteria."""
        cur_epoch = self.fwdbwd_pass_step // self.job_overview.num_fwdbwd_passes_per_epoch
        
        # Check step-based early stopping
        if self.fwdbwd_pass_step > self.config.training.schedule.get("early_stop_after", int(1e10)):
            print(f"Early stopping after {self.config.training.schedule.early_stop_after} steps")
            return True
            
        # Check epoch-based early stopping
        if cur_epoch >= self.config.training.schedule.get("early_stop_after_epochs", int(1e10)) - 1:
            print(f"Early stopping after {self.config.training.schedule.early_stop_after_epochs} epochs")
            return True
            
        return False
        
    def train(self):
        """Main training loop."""
        print(f"ddp_rank={self.ddp_rank}, Starting training loop")
        self._barrier()
        
        self.model.train()
        
        while self.fwdbwd_pass_step <= self.job_overview.num_fwdbwd_passes:
            tic = time.time()
            cur_epoch = self.fwdbwd_pass_step // self.job_overview.num_fwdbwd_passes_per_epoch
            
            # Reset dataloader for new epoch
            if self.fwdbwd_pass_step % self.job_overview.num_fwdbwd_passes_per_epoch == 0:
                print(f"ddp_rank={self.ddp_rank}, Resetting dataloader epoch to {cur_epoch}")
                self._set_epoch(self.dataloader, cur_epoch)
                self.dataloader_iter = iter(self.dataloader)
                
            # Get next batch
            batch = next(self.dataloader_iter)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Training step
            result, create_visual, create_val = self.train_step(batch)
            
            # Optimizer step
            total_grad_norm = self.optimizer_step(result)
            
            # Logging
            iter_time = time.time() - tic
            self.log_training_metrics(result, total_grad_norm, iter_time)

            # Checkpointing
            self.save_checkpoint_if_needed()
            
            # Save visuals
            self.save_visuals_if_needed(result, batch, create_visual)
            
            # Synchronize after visual creation
            if create_visual:
                self._barrier()
                
            # Validation
            if create_val:
                self.run_validation()
                
            # Check early stopping conditions
            if self.should_stop_training():
                break
                
        # Save final checkpoint if needed
        if self.ddp_rank == 0:
            self.save_checkpoint_if_needed()
            
    def cleanup(self):
        """Clean up distributed training."""
        if not self.single_gpu_mode:
            self._barrier()
            destroy_process_group()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GSLRM Training Script")
    parser.add_argument("--config", "-c", type=str, required=True, 
                       help="Path to YAML configuration file")
    parser.add_argument("--load", type=str, default="", 
                       help="Force load weights from specific path")
    parser.add_argument("--set", "-s", type=str, action="append", nargs=2,
                       metavar=("KEY", "VALUE"), help="Override config values")
    return parser.parse_args()


def load_and_process_config(config_path: str, overrides: Optional[list] = None) -> edict:
    """Load and process configuration file."""
    def set_nested_key(data: dict, keys: list, value: str):
        """Set value in nested dictionary."""
        key = keys.pop(0)
        if keys:
            if key not in data:
                data[key] = {}
            set_nested_key(data[key], keys, value)
        else:
            data[key] = value_type(value)
            
    def value_type(value: str):
        """Convert string to appropriate type."""
        try:
            if value.lower() == "true":
                return True
            elif value.lower() == "false":
                return False
            else:
                try:
                    return int(value)
                except ValueError:
                    try:
                        return float(value)
                    except ValueError:
                        return value
        except AttributeError:
            return value
            
    # Load base config
    config = yaml.safe_load(open(config_path, "r"))
    
    # Apply overrides
    if overrides:
        for key_value in overrides:
            key_parts = key_value[0].split(".")
            value = key_value[1]
            set_nested_key(config, key_parts, value)
            
    return edict(config)


def main():
    """Main training function."""
    # Parse arguments and load config
    args = parse_arguments()
    config = load_and_process_config(args.config, args.set)
    print_rank0(config)
    
    # Create trainer
    trainer = GSLRMTrainer(config, args)
    
    try:
        # Setup all components
        trainer.load_datasets()
        trainer.setup_model()
        trainer.setup_optimization()
        trainer.load_checkpoint()
        trainer.setup_wandb()
        
        # Setup validation dataloader if needed
        if config.validation.enabled:
            os.makedirs(config.validation.output_dir, exist_ok=True)
            
        # Run appropriate mode
        if config.inference.enabled:
            trainer.run_inference()
        elif config.get("evaluation", False):
            trainer.run_evaluation()
        else:
            trainer.train()
            
    except Exception as e:
        print(f"Training failed with error: {e}")
        traceback.print_exc()
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()