# =============================================================================
# MV-Adapter Scaffolding for Future Multi-View Diffusion Replacement
# =============================================================================
# Reference: Huang et al., "MV-Adapter: Multi-View Consistent Image
#            Generation Made Easy", ICCV 2025
# GitHub: https://github.com/huanngzh/MV-Adapter
# License: Apache-2.0
#
# Architecture: Parallel decoupled attention adapter for SD2.1
#   - Frozen: Original UNet (self-attn, text cross-attn, conv blocks)
#   - Trainable: MV attention + Image cross-attn + Camera guider (~127M)
#   - Zero-initialized output projections for stable training start
#
# Status: SCAFFOLDING — structural placeholders for future implementation.
#         Requires huanngzh/MV-Adapter repo for actual weights/implementation.
#
# Created: 2026-02-13
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MVAdapterConfig:
    """Configuration for MV-Adapter integration."""

    # Base model
    base_model: str = "stabilityai/stable-diffusion-2-1"
    cross_attention_dim: int = 1024    # SD2.1 cross-attn dim
    block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280)

    # Multi-view attention
    num_views: int = 6
    mv_attention_type: str = "row"     # "row" | "row_col" | "full"
    # row: epipolar constraint (3D objects, like Era3D)
    # row_col: + vertical consistency (texture generation)
    # full: arbitrary viewpoints

    # Camera guider
    camera_embed_channels: int = 6     # ray_origin(3) + ray_direction(3)
    guider_hidden_channels: int = 64

    # Image conditioning
    use_image_cross_attention: bool = True
    frozen_encoder_timestep: int = 0   # t=0 for reference feature extraction

    # Training
    zero_init_output: bool = True      # Zero-init new layer outputs


# =============================================================================
# Camera Guider (Raymap Encoder)
# =============================================================================

class CameraGuider(nn.Module):
    """
    Multi-scale CNN encoder for camera ray conditioning.

    Encodes per-pixel camera ray maps (origin + direction) into
    multi-scale features that are injected into UNet encoder levels.

    Input: [B*N, 6, H, W] — 6ch = ray_origin(3) + ray_direction(3)
    Output: List of multi-scale features matching UNet block resolutions.

    Note: Actual implementation should follow huanngzh/MV-Adapter.
    """

    def __init__(self, config: MVAdapterConfig):
        super().__init__()
        self.config = config
        in_ch = config.camera_embed_channels
        hidden = config.guider_hidden_channels

        # Multi-scale feature extraction
        # Each block outputs features at decreasing spatial resolution
        # matching UNet's block_out_channels
        self.encoder_blocks = nn.ModuleList()
        channels = [in_ch] + list(config.block_out_channels)

        for i in range(len(config.block_out_channels)):
            block = nn.Sequential(
                nn.Conv2d(channels[i], hidden, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden, channels[i + 1], 3, stride=2, padding=1),
                nn.SiLU(),
            )
            self.encoder_blocks.append(block)

    def forward(
        self,
        raymap: torch.Tensor,  # [B*N, 6, H, W]
    ) -> List[torch.Tensor]:
        """Extract multi-scale features from camera raymap."""
        features = []
        x = raymap
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)
        return features

    @staticmethod
    def compute_raymap(
        c2w: torch.Tensor,        # [B, 4, 4]
        intrinsics: torch.Tensor,  # [B, 4] (fx, fy, cx, cy)
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Compute per-pixel ray origin + direction map.

        Returns: [B, 6, H, W] — [ray_origin(3), ray_direction(3)]
        """
        B = c2w.shape[0]
        device = c2w.device

        fx, fy, cx, cy = intrinsics[:, 0], intrinsics[:, 1], intrinsics[:, 2], intrinsics[:, 3]

        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        x = x.unsqueeze(0).expand(B, -1, -1)
        y = y.unsqueeze(0).expand(B, -1, -1)

        # Pixel → camera space
        x_cam = (x - cx.view(-1, 1, 1)) / fx.view(-1, 1, 1)
        y_cam = (y - cy.view(-1, 1, 1)) / fy.view(-1, 1, 1)
        z_cam = torch.ones_like(x_cam)

        dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [B, H, W, 3]
        dirs_cam = F.normalize(dirs_cam, dim=-1)

        # Camera → world
        rotation = c2w[:, :3, :3]  # [B, 3, 3]
        origin = c2w[:, :3, 3]     # [B, 3]

        dirs_world = torch.einsum("bhwc,bdc->bhwd", dirs_cam, rotation)
        origin_map = origin[:, None, None, :].expand(-1, height, width, -1)

        # Concat: [B, H, W, 6] → [B, 6, H, W]
        raymap = torch.cat([origin_map, dirs_world], dim=-1)
        return raymap.permute(0, 3, 1, 2)


# =============================================================================
# Parallel Decoupled Attention
# =============================================================================

class MultiViewAttentionAdapter(nn.Module):
    """
    Parallel multi-view attention layer (adapter pattern).

    Runs alongside the frozen original self-attention. Output is
    zero-initialized so initial behavior = original UNet unchanged.

    Architecture:
        Original: out = SelfAttn(x) + TextCrossAttn(x, text)
        Adapted:  out = SelfAttn(x) + TextCrossAttn(x, text)
                      + MVAttn(x)          ← NEW (zero-init)
                      + ImgCrossAttn(x, ref) ← NEW (zero-init, optional)

    Note: This is a structural placeholder. Actual attention processor
    should be ported from huanngzh/MV-Adapter's MVAdapterAttnProcessor.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_views: int = 6,
        attention_type: str = "row",
        zero_init: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_views = num_views
        self.attention_type = attention_type

        # Multi-view attention (duplicated from original self-attention)
        self.mv_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.mv_out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Zero-initialize output projection
        if zero_init:
            nn.init.zeros_(self.mv_out_proj.weight)
            nn.init.zeros_(self.mv_out_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B*N, seq_len, C]
        num_views: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute multi-view attention.

        For row-wise: attention is computed across views within same row
        For full: attention is computed across all views at all positions

        Returns: residual to add to original attention output
        """
        N = num_views or self.num_views
        BN, S, C = hidden_states.shape
        B = BN // N

        # TODO: Implement actual attention based on attention_type
        # This placeholder returns zeros (matching zero-init behavior)
        return torch.zeros_like(hidden_states)


class ImageCrossAttentionAdapter(nn.Module):
    """
    Image cross-attention using frozen UNet features as reference.

    Key insight: Uses the SAME frozen UNet at t=0 to extract reference
    features, avoiding need for a separate image encoder.

    f_ref = FrozenUNet(ref_image, t=0) → extract self-attn features
    out = CrossAttn(Q=f_in, K=f_ref, V=f_ref)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        zero_init: bool = True,
    ):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.kv_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,       # [B*N, S, C]
        reference_features: torch.Tensor,  # [B, S_ref, C]
    ) -> torch.Tensor:
        """
        Cross-attend to reference image features.

        Returns: residual to add to original attention output
        """
        # TODO: Implement actual cross-attention
        return torch.zeros_like(hidden_states)


# =============================================================================
# MV-Adapter Module (Top-Level)
# =============================================================================

class MVAdapterModule(nn.Module):
    """
    Top-level MV-Adapter module for integration with existing UNet.

    This module wraps the adapter components and provides a clean
    interface for training and inference.

    Integration approach (non-invasive):
        1. Load frozen UNet (existing)
        2. Create MVAdapterModule alongside
        3. Hook into UNet's attention blocks via attention processor replacement
        4. Only MVAdapterModule parameters are trainable

    Usage:
        config = MVAdapterConfig(num_views=6, mv_attention_type="row")
        adapter = MVAdapterModule(config)

        # For training: only adapter parameters
        optimizer = Adam(adapter.parameters(), lr=1e-4)

        # Inject into UNet (replaces attention processors)
        adapter.attach_to_unet(unet)
    """

    def __init__(self, config: MVAdapterConfig):
        super().__init__()
        self.config = config

        # Camera guider
        self.camera_guider = CameraGuider(config)

        # Multi-view attention adapters (one per UNet block)
        # TODO: Create adapters matching actual UNet block structure
        self.mv_adapters = nn.ModuleDict()
        for i, ch in enumerate(config.block_out_channels):
            self.mv_adapters[f"down_{i}"] = MultiViewAttentionAdapter(
                hidden_dim=ch,
                num_views=config.num_views,
                attention_type=config.mv_attention_type,
            )

        # Image cross-attention adapters (optional)
        if config.use_image_cross_attention:
            self.img_adapters = nn.ModuleDict()
            for i, ch in enumerate(config.block_out_channels):
                self.img_adapters[f"down_{i}"] = ImageCrossAttentionAdapter(
                    hidden_dim=ch,
                )
        else:
            self.img_adapters = None

    def attach_to_unet(self, unet: nn.Module) -> None:
        """
        Attach adapter to existing UNet by replacing attention processors.

        This is the key integration point — replaces UNet's attention
        processors with adapter-aware versions that add MV attention
        and image cross-attention as parallel residuals.

        TODO: Implement actual processor replacement following
              huanngzh/MV-Adapter's set_attn_processor pattern.
        """
        raise NotImplementedError(
            "attach_to_unet requires porting MVAdapterAttnProcessor "
            "from huanngzh/MV-Adapter. See GitHub repo for implementation."
        )

    def extract_reference_features(
        self,
        unet: nn.Module,
        ref_latent: torch.Tensor,   # [B, 4, H, W]
        text_embeds: torch.Tensor,  # [B, seq_len, C]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale reference features using frozen UNet at t=0.

        This is MV-Adapter's key trick: no separate image encoder needed.
        The frozen UNet itself extracts reference features when run with
        clean image (t=0).

        TODO: Implement feature extraction hooks.
        """
        raise NotImplementedError(
            "Reference feature extraction requires hooking into UNet "
            "self-attention layers. See MV-Adapter repo."
        )

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return only trainable adapter parameters (not frozen UNet)."""
        params = []
        params.extend(self.camera_guider.parameters())
        params.extend(self.mv_adapters.parameters())
        if self.img_adapters is not None:
            params.extend(self.img_adapters.parameters())
        return params

    def parameter_count(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {
            "camera_guider": sum(p.numel() for p in self.camera_guider.parameters()),
            "mv_attention": sum(p.numel() for p in self.mv_adapters.parameters()),
        }
        if self.img_adapters is not None:
            counts["img_cross_attention"] = sum(
                p.numel() for p in self.img_adapters.parameters()
            )
        counts["total_trainable"] = sum(counts.values())
        return counts

    def extra_repr(self) -> str:
        counts = self.parameter_count()
        return (
            f"num_views={self.config.num_views}, "
            f"mv_type={self.config.mv_attention_type}, "
            f"trainable={counts['total_trainable']:,} params"
        )


# =============================================================================
# Integration with Current FaceLift Pipeline
# =============================================================================

def create_mv_adapter_for_facelift(
    num_views: int = 6,
    attention_type: str = "row",
    use_image_conditioning: bool = True,
) -> MVAdapterModule:
    """
    Factory function to create MV-Adapter compatible with FaceLift pipeline.

    FaceLift uses SD2.1-UnCLIP as base model, so adapter must match:
    - cross_attention_dim: 1024
    - block_out_channels: (320, 640, 1280, 1280)
    - 6 views at 512x512

    Args:
        num_views: Number of output views (6 for M5 rig)
        attention_type: "row" (Era3D-like), "row_col", or "full"
        use_image_conditioning: Whether to use image cross-attention

    Returns:
        MVAdapterModule ready for training
    """
    config = MVAdapterConfig(
        num_views=num_views,
        mv_attention_type=attention_type,
        use_image_cross_attention=use_image_conditioning,
    )
    adapter = MVAdapterModule(config)

    counts = adapter.parameter_count()
    print(f"[MV-Adapter] Created with {counts['total_trainable']:,} trainable params")
    print(f"  Camera Guider: {counts['camera_guider']:,}")
    print(f"  MV Attention:  {counts['mv_attention']:,}")
    if "img_cross_attention" in counts:
        print(f"  Img CrossAttn: {counts['img_cross_attention']:,}")

    return adapter


# =============================================================================
# Comparison: Era3D (current) vs MV-Adapter (future)
# =============================================================================
#
# | Aspect              | Era3D (Current)          | MV-Adapter (Future)      |
# |---------------------|--------------------------|--------------------------|
# | Base Model          | SD2.1-UnCLIP             | SD2.1 (or SDXL)          |
# | MV Attention        | Row-wise (RMA)           | Parallel decoupled       |
# | Camera Conditioning | CLIP text ("top-front")  | Raymap (per-pixel rays)  |
# | Image Conditioning  | CLIP image embed         | Frozen UNet features     |
# | Trainable Params    | Full UNet (~1.1B)        | Adapter only (~127M)     |
# | Camera Flexibility  | Fixed canonical views    | Arbitrary viewpoints     |
# | Integration         | Modified UNet            | Processor replacement    |
# | Training Data       | Synthetic faces → mouse  | Can leverage any data    |
#
# Migration path:
#   1. Current: Era3D RMA + pose conditioning experiments
#   2. Future: Replace attention processors with MV-Adapter
#   3. Key: Camera guider + reference feature extraction are the main work
#   4. GS-LRM (Stage 2) is independent — works with any multi-view output
