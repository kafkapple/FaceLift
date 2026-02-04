"""Temporal GS-LRM Trainer Extension.

Extends the base GSLRMTrainer to support temporal consistency losses.
Uses a sliding window of frames to compute ARAP and velocity losses.
"""

import torch
from typing import Dict, List, Any, Tuple, Optional
from collections import deque
from easydict import EasyDict as edict

from mouse_extensions.training.temporal_trainer import (
    TemporalTrainingConfig,
    TemporalLossComputer,
)


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
        
        self.xyz_buffer: deque = deque(maxlen=self.temporal_config.temporal_window)
        self.temporal_loss_log: Dict[str, float] = {}
        
        print(f"[Temporal] Enabled with window={self.temporal_config.temporal_window}, "
              f"arap={self.temporal_config.arap_weight}, vel={self.temporal_config.velocity_weight}")
    
    def compute_temporal_loss(
        self, 
        result: edict,
        current_step: int,
    ) -> Optional[torch.Tensor]:
        """Compute temporal loss from current result and buffer."""
        if not self.temporal_enabled:
            return None
        
        if not hasattr(result, 'gaussian_params_raw') or result.gaussian_params_raw is None:
            return None
        
        xyz = result.gaussian_params_raw.xyz  # [batch, N, 3]
        self.xyz_buffer.append({'xyz': xyz[0]})
        
        if len(self.xyz_buffer) < self.temporal_config.temporal_window:
            return None
        
        temporal_loss, log_dict = self.temporal_computer.compute_temporal_loss(
            list(self.xyz_buffer),
            current_step,
        )
        
        self.temporal_loss_log = log_dict
        return temporal_loss
    
    def get_temporal_log_dict(self) -> Dict[str, float]:
        return self.temporal_loss_log.copy()
    
    def clear_temporal_buffer(self):
        if hasattr(self, 'xyz_buffer'):
            self.xyz_buffer.clear()


def create_temporal_train_step(trainer):
    """Create a new train_step that includes temporal loss.
    
    This version computes both losses in one forward pass and combines them
    before backward to avoid the double-backward issue.
    """
    from contextlib import nullcontext
    
    def train_step_with_temporal(batch):
        # Determine what to create
        create_visual = (
            trainer.fwdbwd_pass_step == trainer.start_fwdbwd_pass_step or
            trainer.fwdbwd_pass_step % trainer.config.training.logging.vis_every == 0
        )
        
        create_val = (
            trainer.config.get("validation", {}).get("enabled", False) and (
                trainer.fwdbwd_pass_step == trainer.start_fwdbwd_pass_step or
                trainer.fwdbwd_pass_step % trainer.config.get("validation", {}).get("val_every", 200) == 0
            )
        )
        
        # Gradient accumulation context
        ctx = (
            nullcontext()
            if (trainer.fwdbwd_pass_step + 1) % trainer.config.training.runtime.grad_accum_steps == 0
            else trainer._no_sync()
        )
        
        amp_dtype_mapping = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        
        with ctx, torch.autocast(
            enabled=trainer.config.training.runtime.use_amp,
            device_type="cuda",
            dtype=amp_dtype_mapping.get(trainer.config.training.runtime.amp_dtype, torch.float16),
        ):
            try:
                trainer.model_module.set_current_step(
                    trainer.fwdbwd_pass_step, 
                    trainer.start_fwdbwd_pass_step, 
                    trainer.job_overview.num_fwdbwd_passes
                )
            except:
                pass
                
            result = trainer.model(batch, create_visual=create_visual)
            
            # Compute temporal loss
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
                setattr(result.loss_metrics, k.replace('/', '_'), v)
        
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
    
    return trainer


if __name__ == '__main__':
    print('Temporal GS-LRM Trainer Extension')
