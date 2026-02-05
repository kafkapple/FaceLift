"""Temporal training extension for GS-LRM trainer.

This module adds temporal regularization (ARAP + velocity) to the existing trainer.
NOTE: Currently uses single-frame batches with cross-batch buffering.
For proper gradient flow, the buffer stores detached tensors and only the
current frame's gradients are used. This is a simplified approach.
"""

from collections import deque
from typing import Dict, Optional
from easydict import EasyDict as edict
import torch

from .temporal_trainer import TemporalLossComputer, TemporalTrainingConfig


class TemporalTrainingMixin:
    """Mixin to add temporal training capabilities to GSLRMTrainer."""
    
    def setup_temporal_training(self, config: edict):
        """Initialize temporal training components."""
        temporal_cfg = config.get('temporal', {})
        
        self.temporal_enabled = temporal_cfg.get('enabled', False)
        if not self.temporal_enabled:
            return
        
        self.temporal_config = TemporalTrainingConfig(
            enabled=True,
            temporal_window=temporal_cfg.get('window', 2),
            arap_weight=temporal_cfg.get('arap_weight', 0.1),
            velocity_weight=temporal_cfg.get('velocity_weight', 0.01),
            warmup_steps=temporal_cfg.get('warmup_steps', 1000),
            rampup_steps=temporal_cfg.get('rampup_steps', 500),
        )
        
        self.temporal_computer = TemporalLossComputer(self.temporal_config)
        if hasattr(self, 'device'):
            self.temporal_computer = self.temporal_computer.to(self.device)
        
        # Buffer stores DETACHED tensors for reference (no gradient)
        self.xyz_buffer: deque = deque(maxlen=self.temporal_config.temporal_window)
        self.temporal_loss_log: Dict[str, float] = {}
        
        print(f"[Temporal] Setting up temporal training...")
        print(f"[Temporal] Enabled with window={self.temporal_config.temporal_window}, "
              f"arap={self.temporal_config.arap_weight}, vel={self.temporal_config.velocity_weight}")
    
    def compute_temporal_loss(
        self, 
        result: edict,
        current_step: int,
    ) -> Optional[torch.Tensor]:
        """Compute temporal loss from current result.
        
        NOTE: This simplified version only regularizes xyz distribution without
        true cross-frame temporal consistency (would require multi-frame batches).
        The ARAP loss enforces local rigidity within current frame.
        """
        if not self.temporal_enabled:
            return None
        
        if not hasattr(result, 'gaussian_params_raw') or result.gaussian_params_raw is None:
            return None
        
        cfg = self.temporal_config
        device = result.gaussian_params_raw.xyz.device
        
        # Warmup phase
        if current_step < cfg.warmup_steps:
            self.temporal_loss_log = {}
            return None
        
        # Get current xyz with gradients
        xyz = result.gaussian_params_raw.xyz  # [batch, N, 3]
        current_xyz = xyz[0]  # [N, 3]
        
        # Compute rampup factor
        if cfg.rampup_steps > 0:
            steps_since_warmup = current_step - cfg.warmup_steps
            rampup_factor = min(1.0, (steps_since_warmup + 1) / cfg.rampup_steps)
        else:
            rampup_factor = 1.0
        
        # For now: only compute ARAP-style regularization on current frame
        # This enforces local structure preservation
        loss_dict = self.temporal_computer.compute_single_frame_loss(
            current_xyz, rampup_factor
        )
        
        total_loss = loss_dict.get('loss', torch.tensor(0.0, device=device))
        
        self.temporal_loss_log = {
            k: v.item() if torch.is_tensor(v) else v 
            for k, v in loss_dict.items()
        }
        self.temporal_loss_log['temporal/rampup_factor'] = rampup_factor
        
        return total_loss
    
    def get_temporal_log_dict(self) -> Dict[str, float]:
        return self.temporal_loss_log.copy()
    
    def clear_temporal_buffer(self):
        if hasattr(self, 'xyz_buffer'):
            self.xyz_buffer.clear()


def create_temporal_train_step(trainer):
    """Create a new train_step that includes temporal loss."""
    from contextlib import nullcontext
    
    # Store original train_step
    original_train_step = trainer.train_step
    
    def train_step_with_temporal(batch):
        # Determine if we need visualization/validation
        create_visual = (trainer.fwdbwd_pass_step + 1) % trainer.config.training.logging.vis_every == 0
        create_val = trainer.config.validation.enabled and \
                     (trainer.fwdbwd_pass_step + 1) % trainer.config.validation.val_every == 0
        
        # Setup autocast context
        use_amp = trainer.config.training.runtime.use_amp
        amp_dtype_str = trainer.config.training.runtime.amp_dtype
        amp_dtype = torch.bfloat16 if amp_dtype_str == 'bf16' else getattr(torch, amp_dtype_str.replace('float', 'float'))
        autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
        
        with autocast_ctx(enabled=use_amp, dtype=amp_dtype):
            # Forward pass
            result = trainer.model(batch, create_visual=create_visual)
            
            # Compute temporal loss (after warmup)
            temporal_loss = None
            if hasattr(trainer, 'temporal_enabled') and trainer.temporal_enabled:
                temporal_loss = trainer.compute_temporal_loss(
                    result, 
                    trainer.fwdbwd_pass_step
                )
        
        # Combine losses
        grad_accum = trainer.config.training.runtime.grad_accum_steps
        total_loss = result.loss_metrics.loss / grad_accum
        
        if temporal_loss is not None and temporal_loss.requires_grad:
            total_loss = total_loss + temporal_loss / grad_accum
            # Add temporal metrics to result for logging
            for k, v in trainer.get_temporal_log_dict().items():
                key = k.replace('/', '_').replace('temporal_', '')
                setattr(result.loss_metrics, f'temporal_{key}', v)
        
        # Single backward pass
        trainer.scaler.scale(total_loss).backward()
        trainer.fwdbwd_pass_step += 1
        
        return result, create_visual, create_val
    
    return train_step_with_temporal


def enable_temporal_training(trainer, config: edict):
    """Enable temporal training on an existing trainer instance."""
    # Add mixin methods
    for attr in ['setup_temporal_training', 'compute_temporal_loss', 
                 'get_temporal_log_dict', 'clear_temporal_buffer']:
        setattr(trainer, attr, getattr(TemporalTrainingMixin, attr).__get__(trainer))
    
    trainer.setup_temporal_training(config)
    
    if trainer.temporal_enabled:
        trainer.train_step = create_temporal_train_step(trainer)
