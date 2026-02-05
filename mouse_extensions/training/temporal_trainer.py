"""Temporal Training Extension for GS-LRM.

Adds temporal consistency losses (ARAP, velocity) to GS-LRM training.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

from mouse_extensions.model.deformation.temporal_regularization_efficient import (
    TemporalRegularizationEfficient,
    TemporalRegConfig,
)


@dataclass
class TemporalTrainingConfig:
    enabled: bool = False
    temporal_window: int = 2
    arap_weight: float = 0.1
    velocity_weight: float = 0.01
    isometry_weight: float = 0.0
    warmup_steps: int = 1000
    rampup_steps: int = 500


class TemporalLossComputer(nn.Module):
    def __init__(self, config: TemporalTrainingConfig):
        super().__init__()
        self.config = config
        
        reg_config = TemporalRegConfig(
            arap_weight=config.arap_weight,
            velocity_weight=config.velocity_weight,
            isometry_weight=config.isometry_weight,
        )
        self.temporal_reg = TemporalRegularizationEfficient(reg_config)
        
    def compute_temporal_loss(
        self,
        gaussian_params_sequence: List[Dict[str, torch.Tensor]],
        current_step: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.config
        device = gaussian_params_sequence[0]['xyz'].device
        
        # Check warmup
        if current_step < cfg.warmup_steps:
            return torch.tensor(0.0, device=device, requires_grad=True), {}
        
        # Compute rampup factor (starts at small value, ramps up to 1.0)
        if cfg.rampup_steps > 0:
            steps_since_warmup = current_step - cfg.warmup_steps
            # Start from a small non-zero value and ramp up
            rampup_factor = min(1.0, (steps_since_warmup + 1) / cfg.rampup_steps)
        else:
            rampup_factor = 1.0
        
        # Compute losses
        xyz_sequence = [params['xyz'] for params in gaussian_params_sequence]
        losses = self.temporal_reg.compute_losses(xyz_sequence)
        total_loss = losses['total'] * rampup_factor
        
        loss_dict = {
            f'temporal/{k}': v.item() * rampup_factor 
            for k, v in losses.items()
        }
        loss_dict['temporal/rampup_factor'] = rampup_factor
        
        return total_loss, loss_dict

    def compute_single_frame_loss(
        self,
        xyz: torch.Tensor,  # [N, 3]
        rampup_factor: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute structure preservation loss on single frame.
        
        This uses the ARAP loss component to enforce local rigidity
        without needing temporal data.
        
        Args:
            xyz: Point positions [N, 3]
            rampup_factor: Scale factor for loss (0 to 1)
            
        Returns:
            Dict with loss and component losses
        """
        cfg = self.config
        device = xyz.device
        
        if cfg.arap_weight <= 0:
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}
        
        # Use ARAP to compute edge-length preservation loss
        # For single frame, we use it as structure regularization
        # Compare xyz to itself (no change expected)
        arap_loss = self.temporal_reg.arap_loss(xyz, xyz)
        
        # Scale by weight and rampup
        total_loss = arap_loss * cfg.arap_weight * rampup_factor
        
        return {
            "loss": total_loss,
            "temporal/structure_loss": arap_loss.detach() * rampup_factor,
            "temporal/total": total_loss.detach(),
        }


def create_temporal_dataloader(
    dataset,
    temporal_window: int = 2,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
):
    from torch.utils.data import DataLoader, Sampler
    
    class TemporalSampler(Sampler):
        def __init__(self, data_source, window, shuffle):
            self.data_source = data_source
            self.window = window
            self.shuffle = shuffle
            self.valid_starts = len(data_source) - window + 1
            
        def __iter__(self):
            indices = list(range(self.valid_starts))
            if self.shuffle:
                import random
                random.shuffle(indices)
            for start_idx in indices:
                yield [start_idx + t for t in range(self.window)]
                
        def __len__(self):
            return self.valid_starts
    
    def collate_temporal(batch_list):
        return batch_list[0]
    
    sampler = TemporalSampler(dataset, temporal_window, shuffle)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_temporal,
        pin_memory=True,
    )
