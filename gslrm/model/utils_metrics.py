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

# utils_metrics.py is under the Adobe Research License. Copyright 2025 Adobe Inc.

from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from einops import reduce
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    mask: Optional[Float[Tensor, "batch 1 height width"]] = None,
) -> Float[Tensor, " batch"]:
    """
    Compute PSNR between ground truth and predicted images.
    
    Args:
        ground_truth: GT images [B, C, H, W] in [0, 1]
        predicted: Predicted images [B, C, H, W] in [0, 1]
        mask: Optional binary mask [B, 1, H, W] where 1=foreground, 0=background
              If provided, PSNR is computed only on foreground pixels.
    
    Returns:
        Per-batch PSNR values [B]
    """
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    
    if mask is not None:
        # Masked PSNR: compute MSE only on foreground pixels
        mask_binary = (mask > 0.5).float()
        # Expand mask to match channels
        mask_expanded = mask_binary.expand_as(ground_truth)
        
        # Per-sample masked MSE
        squared_error = (ground_truth - predicted) ** 2
        masked_error = squared_error * mask_expanded
        
        # Sum over C, H, W and divide by number of valid pixels per sample
        num_valid_per_sample = mask_expanded.sum(dim=(1, 2, 3)).clamp(min=1.0)
        mse = masked_error.sum(dim=(1, 2, 3)) / num_valid_per_sample
    else:
        mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    
    return -10 * mse.log10()


@lru_cache(maxsize=None)
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    mask: Optional[Float[Tensor, "batch 1 height width"]] = None,
) -> Float[Tensor, " batch"]:
    """
    Compute LPIPS perceptual distance between ground truth and predicted images.
    
    Args:
        ground_truth: GT images [B, C, H, W] in [0, 1]
        predicted: Predicted images [B, C, H, W] in [0, 1]
        mask: Optional binary mask [B, 1, H, W] where 1=foreground, 0=background
              If provided, background pixels are set to a neutral value (0.5) before
              computing LPIPS to reduce their influence on perceptual distance.
    
    Returns:
        Per-batch LPIPS values [B]
    """
    lpips_fn = get_lpips(predicted.device)
    
    if mask is not None:
        # Apply mask: set background to neutral gray (0.5) to minimize its influence
        mask_binary = (mask > 0.5).float()
        neutral_value = 0.5
        ground_truth = ground_truth * mask_binary + neutral_value * (1 - mask_binary)
        predicted = predicted * mask_binary + neutral_value * (1 - mask_binary)
    
    # feed 10 images at a time to avoid memory issues
    batch_size = 10
    values = []
    for i in range(0, ground_truth.shape[0], batch_size):
        value = lpips_fn.forward(
            ground_truth[i : i + batch_size],
            predicted[i : i + batch_size],
            normalize=True,
        )
        values.append(value)
    value = torch.cat(values, dim=0)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    mask: Optional[Float[Tensor, "batch 1 height width"]] = None,
) -> Float[Tensor, " batch"]:
    """
    Compute SSIM between ground truth and predicted images.
    
    Args:
        ground_truth: GT images [B, C, H, W] in [0, 1]
        predicted: Predicted images [B, C, H, W] in [0, 1]
        mask: Optional binary mask [B, 1, H, W] where 1=foreground, 0=background
              If provided, background pixels are set to the same neutral value
              to reduce their influence on SSIM calculation.
    
    Returns:
        Per-batch SSIM values [B]
    """
    if mask is not None:
        # Apply mask: set background to neutral gray (0.5)
        mask_binary = (mask > 0.5).float()
        neutral_value = 0.5
        ground_truth = ground_truth * mask_binary + neutral_value * (1 - mask_binary)
        predicted = predicted * mask_binary + neutral_value * (1 - mask_binary)
    
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)


@torch.no_grad()
def compute_mask_iou(
    predicted: Float[Tensor, "batch channel height width"],
    gt_mask: Float[Tensor, "batch 1 height width"],
    bg_threshold: float = 0.1,
) -> Float[Tensor, " batch"]:
    """
    Compute IoU between GT mask and predicted mask derived from rendered image.
    
    The predicted mask is computed by thresholding distance from white background.
    
    Args:
        predicted: Predicted images [B, C, H, W] in [0, 1]
        gt_mask: Ground truth mask [B, 1, H, W]
        bg_threshold: Threshold for background detection (pixels closer to white
                      than this threshold are considered background)
    
    Returns:
        Per-batch IoU values [B]
    """
    # Compute predicted mask: distance from white (1.0, 1.0, 1.0)
    color_distance = (predicted - 1.0).abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]
    pred_mask = (color_distance > bg_threshold).float()
    
    gt_mask_binary = (gt_mask > 0.5).float()
    
    # Per-sample IoU
    intersection = (pred_mask * gt_mask_binary).sum(dim=(1, 2, 3))
    union = ((pred_mask + gt_mask_binary) > 0.5).float().sum(dim=(1, 2, 3))
    
    iou = intersection / union.clamp(min=1.0)
    return iou


@torch.no_grad()
def compute_l1(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    mask: Optional[Float[Tensor, "batch 1 height width"]] = None,
    normalize_by_mask: bool = True,
) -> Float[Tensor, " batch"]:
    """
    Compute L1 distance between ground truth and predicted images.
    
    Pose Splatter uses normalized masked L1:
        L1 = sum(|pred - gt| * mask) / sum(mask)
    
    Args:
        ground_truth: GT images [B, C, H, W] in [0, 1]
        predicted: Predicted images [B, C, H, W] in [0, 1]
        mask: Optional binary mask [B, 1, H, W] where 1=foreground, 0=background
        normalize_by_mask: If True, normalize by mask area (Pose Splatter style)
    
    Returns:
        Per-batch L1 values [B]
    """
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    
    l1_error = (ground_truth - predicted).abs()
    
    if mask is not None:
        mask_binary = (mask > 0.5).float()
        mask_expanded = mask_binary.expand_as(ground_truth)
        
        masked_l1 = l1_error * mask_expanded
        
        if normalize_by_mask:
            # Pose Splatter style: normalize by mask area
            num_valid = mask_expanded.sum(dim=(1, 2, 3)).clamp(min=1.0)
            l1 = masked_l1.sum(dim=(1, 2, 3)) / num_valid
        else:
            l1 = masked_l1.mean(dim=(1, 2, 3))
    else:
        l1 = reduce(l1_error, "b c h w -> b", "mean")
    
    return l1
