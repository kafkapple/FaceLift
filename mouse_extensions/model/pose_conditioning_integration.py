# =============================================================================
# Pose Conditioning Integration for MVDiffusion UNet
# =============================================================================
# Non-invasive integration: wraps UNet forward to inject pose embeddings
# into encoder_hidden_states WITHOUT modifying original UNet code.
#
# Usage in train_diffusion.py:
#   from mouse_extensions.model.pose_conditioning_integration import (
#       PoseConditioningInjector, load_m5_cameras
#   )
#   cameras = load_m5_cameras("mouse_extensions/inference/cameras/m5_cameras.json")
#   injector = PoseConditioningInjector(method="spherical", cameras=cameras)
#   # In training loop:
#   prompt_embeddings = injector.inject(prompt_embeddings, ref_view_idx, n_views)
#
# Created: 2026-02-13
# Updated: 2026-02-24 — Added Plucker token projection support
# Updated: 2026-02-25 — Added trainable mode (encoder jointly trained with UNet)
# =============================================================================

import json
import torch
import torch.nn as nn
import numpy as np
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Dict, List, Union

from mouse_extensions.model.pose_conditioning import (
    CameraPoseConditioner,
    SphericalPoseEncoder,
)


# =============================================================================
# Camera Utilities
# =============================================================================

def load_m5_cameras(
    json_path: str = "mouse_extensions/inference/cameras/m5_cameras.json",
) -> Dict:
    """
    Load M5 camera rig definition.

    Returns:
        dict with keys:
          - c2w: [N_views, 4, 4] tensor (camera-to-world)
          - w2c: [N_views, 4, 4] tensor (world-to-camera)
          - intrinsics: [N_views, 4] tensor (fx, fy, cx, cy)
          - n_views: int
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Camera file not found: {json_path}")

    with open(path) as f:
        data = json.load(f)

    frames = data["frames"]
    n_views = len(frames)

    w2c_list = []
    intrinsics_list = []

    for frame in frames:
        w2c = torch.tensor(frame["w2c"], dtype=torch.float32)
        w2c_list.append(w2c)
        intrinsics_list.append(
            torch.tensor([frame["fx"], frame["fy"], frame["cx"], frame["cy"]],
                         dtype=torch.float32)
        )

    w2c = torch.stack(w2c_list)  # [N, 4, 4]
    c2w = torch.inverse(w2c)     # [N, 4, 4]
    intrinsics = torch.stack(intrinsics_list)  # [N, 4]

    return {
        "c2w": c2w,
        "w2c": w2c,
        "intrinsics": intrinsics,
        "n_views": n_views,
    }


def get_rotated_cameras(
    c2w: torch.Tensor,       # [N, 4, 4]
    ref_view_idx: int,
    n_views: int,
) -> torch.Tensor:
    """
    Rotate camera order to match random reference view augmentation.

    When reference_view_idx="random" and the dataset rotates target view
    indices as [ref, ref+1, ..., ref+n-1] mod n_views, the cameras must
    be rotated correspondingly.

    Args:
        c2w: All cameras [N, 4, 4]
        ref_view_idx: Current reference view index
        n_views: Number of views

    Returns:
        [N, 4, 4] rotated cameras matching the dataset's view order
    """
    rotated_indices = [(ref_view_idx + i) % n_views for i in range(n_views)]
    return c2w[rotated_indices]


# =============================================================================
# Pose Conditioning Injector
# =============================================================================

class PoseConditioningInjector(nn.Module):
    """
    Non-invasive pose conditioning for MVDiffusion UNet.

    Injects pose embeddings into encoder_hidden_states by concatenation
    along the sequence dimension. This avoids any modification to the
    UNet architecture — the extra tokens are simply attended to via
    cross-attention.

    Integration methods:
      - "concat": Concatenate pose embed as extra token to prompt sequence
      - "add": Add pose embed to existing prompt embeddings (requires match)
      - "replace_last": Replace last prompt token with pose embed

    Plucker support (added 2026-02-24):
      PluckerRayEncoder outputs spatial features [B*N, 320, H, W].
      For token-based integration (add/concat/replace_last), we apply
      global average pooling + linear projection to convert spatial→token.
      This preserves compatibility with the existing injection pipeline.

    Trainable mode (added 2026-02-25):
      When trainable=True, gradients flow through the pose encoder and
      projection layers, allowing them to be jointly trained with UNet.
      Literature consensus (MVDream, CAT3D, SPAD, SV3D) shows that jointly
      training pose projections yields better representations than frozen
      random initialization. When trainable=True:
        - @torch.no_grad() is NOT applied to inject()
        - Cached embeddings are NOT used (recomputed each call for grad graph)
        - Caller must add pose_injector.parameters() to an optimizer

    Args:
        method: Pose encoding method ("spherical", "extrinsic", "plucker")
        cameras: Camera dict from load_m5_cameras()
        embed_dim: Embedding dimension (must match UNet cross_attention_dim)
        integration: How to combine pose embed with prompt ("concat")
        plucker_resolution: H/W for Plucker ray computation (default: 64)
        trainable: If True, encoder params receive gradients (default: False)
    """

    def __init__(
        self,
        method: str = "spherical",
        cameras: Optional[Dict] = None,
        embed_dim: int = 1024,
        integration: str = "concat",
        camera_json_path: str = "mouse_extensions/inference/cameras/m5_cameras.json",
        plucker_resolution: int = 64,
        trainable: bool = False,
        spatial_token_size: int = 8,
    ):
        super().__init__()
        self.method = method
        self.integration = integration
        self.embed_dim = embed_dim
        self.plucker_resolution = plucker_resolution
        self.trainable = trainable
        self.spatial_token_size = spatial_token_size

        # Load cameras if not provided
        if cameras is None:
            cameras = load_m5_cameras(camera_json_path)

        # Store cameras as buffer (moved to device with model)
        self.register_buffer("c2w", cameras["c2w"])       # [N, 4, 4]
        self.register_buffer("intrinsics", cameras["intrinsics"])  # [N, 4]
        self.n_views = cameras["n_views"]

        # Create pose encoder
        self.conditioner = CameraPoseConditioner(
            method=method,
            embed_dim=embed_dim,
            num_views=self.n_views,
        )

        # Plucker token projection: spatial [N, spatial_dim, H, W] → token [N, embed_dim]
        if method == "plucker":
            spatial_dim = 320  # PluckerRayEncoder default output channels
            self.plucker_to_token = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),   # [N, 320, 1, 1]
                nn.Flatten(),              # [N, 320]
                nn.Linear(spatial_dim, embed_dim),  # [N, 1024]
            )

            # Spatial token projection: preserve spatial info as token sequence
            # [N, 320, H, W] → pool to SxS → flatten → project → [N, S*S, embed_dim]
            n_spatial_tokens = spatial_token_size * spatial_token_size
            self.plucker_spatial_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(spatial_token_size),  # [N, 320, S, S]
            )
            # Zero-init linear: initially contributes nothing (ControlNet strategy)
            self.plucker_spatial_linear = nn.Linear(spatial_dim, embed_dim)
            nn.init.zeros_(self.plucker_spatial_linear.weight)
            nn.init.zeros_(self.plucker_spatial_linear.bias)

        # Precompute fixed camera pose embeddings (for non-random ref)
        # Will be computed lazily and cached
        self._cached_embeddings = None

    def _compute_pose_token(
        self,
        c2w: torch.Tensor,  # [N, 4, 4] or [1, N, 4, 4]
        view_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Compute pose token embeddings for given cameras.
        Handles all methods uniformly, including Plucker spatial→token conversion.

        Args:
            c2w: Camera matrices (unbatched [N, 4, 4] or batched [1, N, 4, 4])
            view_indices: View indices for intrinsics lookup (Plucker only)

        Returns:
            [N, embed_dim] pose token embeddings
        """
        # Ensure batched format
        if c2w.dim() == 3:
            c2w_batched = c2w.unsqueeze(0)  # [1, N, 4, 4]
        else:
            c2w_batched = c2w  # already [1, N, 4, 4]

        if self.method == 'plucker':
            # Plucker needs intrinsics and resolution
            if view_indices is not None:
                intr = self.intrinsics[view_indices]  # [N, 4]
            else:
                intr = self.intrinsics  # [N, 4]
            intr_batched = intr.unsqueeze(0)  # [1, N, 4]

            # Plucker computation: dtype follows c2w (may be fp16 under
            # mixed precision). pose_conditioning.py's meshgrid also
            # matches c2w.dtype, so all operations stay consistent.
            spatial = self.conditioner(
                c2w_batched, intr_batched,
                self.plucker_resolution, self.plucker_resolution,
            )  # [N, spatial_dim, H, W]
            # Project to token: [N, embed_dim]
            token = self.plucker_to_token(spatial)
            return token
        else:
            # Spherical / Extrinsic: already returns [1, N, embed_dim]
            embeddings = self.conditioner(c2w_batched)
            return embeddings.squeeze(0)  # [N, embed_dim]

    def _compute_spatial_tokens(
        self,
        c2w: torch.Tensor,  # [N, 4, 4] or [1, N, 4, 4]
        view_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Compute spatial token sequence from Plucker ray features.
        Preserves spatial structure by downsampling instead of global pooling.

        Args:
            c2w: Camera matrices
            view_indices: View indices for intrinsics lookup

        Returns:
            [N, S*S, embed_dim] spatial token sequence
        """
        assert self.method == 'plucker', "spatial_token only supported for plucker method"

        if c2w.dim() == 3:
            c2w_batched = c2w.unsqueeze(0)
        else:
            c2w_batched = c2w

        if view_indices is not None:
            intr = self.intrinsics[view_indices]
        else:
            intr = self.intrinsics
        intr_batched = intr.unsqueeze(0)

        # Get spatial features from PluckerRayEncoder
        spatial = self.conditioner(
            c2w_batched, intr_batched,
            self.plucker_resolution, self.plucker_resolution,
        )  # [N, 320, H, W]

        # Downsample to S×S preserving spatial structure
        pooled = self.plucker_spatial_proj(spatial)  # [N, 320, S, S]
        N, C, S, _ = pooled.shape
        tokens = pooled.flatten(2).transpose(1, 2)   # [N, S*S, 320]
        tokens = self.plucker_spatial_linear(tokens)  # [N, S*S, embed_dim]
        return tokens

    def _compute_pose_embeddings(
        self,
        view_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Compute pose embeddings for given view indices.

        Args:
            view_indices: Which views to encode. None = all views in order.

        Returns:
            [N_views, embed_dim] pose embeddings
        """
        if view_indices is not None:
            cameras = self.c2w[view_indices]  # [N, 4, 4]
        else:
            cameras = self.c2w  # [N, 4, 4]

        return self._compute_pose_token(cameras, view_indices)

    def get_fixed_embeddings(self) -> torch.Tensor:
        """Get cached embeddings for fixed camera order (no rotation)."""
        if self.trainable and self.training:
            # When trainable AND in training mode, recompute for gradient graph
            return self._compute_pose_embeddings()
        if self._cached_embeddings is None:
            with torch.no_grad():
                self._cached_embeddings = self._compute_pose_embeddings()
        return self._cached_embeddings

    def inject(
        self,
        encoder_hidden_states: torch.Tensor,  # [B*N, seq_len, embed_dim]
        ref_view_idx: Union[int, torch.Tensor] = 0,
        n_views: int = 6,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Inject pose conditioning into encoder_hidden_states.

        Gradient flow is gated by BOTH trainable flag AND nn.Module training mode:
          - trainable=True + self.training=True → gradients enabled (training)
          - trainable=True + self.training=False → no gradients (validation/inference)
          - trainable=False → no gradients (legacy behavior)

        Call pose_injector.eval() before validation, pose_injector.train() after.

        Args:
            encoder_hidden_states: Original prompt embeddings [B*N, seq_len, C]
            ref_view_idx: Reference view index (int or tensor for per-batch)
            n_views: Number of views
            batch_size: Batch size B (inferred from encoder_hidden_states if None)

        Returns:
            Modified encoder_hidden_states with pose conditioning
        """
        # Only enable gradients during training AND when trainable
        use_grad = self.trainable and self.training
        grad_ctx = nullcontext() if use_grad else torch.no_grad()
        with grad_ctx:
            return self._inject_core(encoder_hidden_states, ref_view_idx, n_views, batch_size)

    def _inject_core(
        self,
        encoder_hidden_states: torch.Tensor,
        ref_view_idx: Union[int, torch.Tensor],
        n_views: int,
        batch_size: Optional[int],
    ) -> torch.Tensor:
        """Core injection logic, called within appropriate gradient context."""
        BN, seq_len, C = encoder_hidden_states.shape
        if batch_size is None:
            batch_size = BN // n_views
        device = encoder_hidden_states.device

        # Compute or retrieve pose embeddings
        if isinstance(ref_view_idx, int) and ref_view_idx == 0:
            # Fixed reference — use cached embeddings
            pose_embeds = self.get_fixed_embeddings().to(device)  # [N, C]
            # Expand for batch: [B*N, C]
            pose_embeds = pose_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            pose_embeds = pose_embeds.reshape(BN, C)
        else:
            # Random reference — compute rotated embeddings per batch
            if isinstance(ref_view_idx, int):
                # Same rotation for all batches
                rotated_cameras = get_rotated_cameras(
                    self.c2w, ref_view_idx, self.n_views
                )
                rotated_indices = [
                    (ref_view_idx + i) % self.n_views
                    for i in range(self.n_views)
                ]
                # Use unified _compute_pose_token (handles plucker)
                pose_embeds = self._compute_pose_token(
                    rotated_cameras, rotated_indices
                )  # [N, C]
                pose_embeds = pose_embeds.unsqueeze(0).expand(
                    batch_size, -1, -1
                )  # [B, N, C]
                pose_embeds = pose_embeds.reshape(BN, C)
            else:
                # Per-batch rotation (tensor of indices)
                all_embeds = []
                for b in range(batch_size):
                    idx = ref_view_idx[b].item() if torch.is_tensor(ref_view_idx) else ref_view_idx
                    rotated_cameras = get_rotated_cameras(
                        self.c2w, idx, self.n_views
                    )
                    rotated_indices = [
                        (idx + i) % self.n_views
                        for i in range(self.n_views)
                    ]
                    embeds = self._compute_pose_token(
                        rotated_cameras, rotated_indices
                    )  # [N, C]
                    all_embeds.append(embeds)
                pose_embeds = torch.stack(all_embeds).reshape(BN, C).to(device)

        # Cast pose_embeds to match encoder_hidden_states dtype.
        # When trainable, pose params are fp32 but UNet runs in fp16.
        pose_embeds = pose_embeds.to(dtype=encoder_hidden_states.dtype)

        # Inject based on integration method
        if self.integration == "concat":
            # Add pose embedding as extra token: [BN, seq_len+1, C]
            pose_token = pose_embeds.unsqueeze(1)  # [BN, 1, C]
            return torch.cat([encoder_hidden_states, pose_token], dim=1)

        elif self.integration == "add":
            # Add to first token (usually CLS/start token)
            out = encoder_hidden_states.clone()
            out[:, 0, :] = out[:, 0, :] + pose_embeds
            return out

        elif self.integration == "replace_last":
            # Replace last token with pose embedding
            out = encoder_hidden_states.clone()
            out[:, -1, :] = pose_embeds
            return out

        elif self.integration == "spatial_token":
            # Spatial token sequence: preserve Plucker spatial info as extra tokens
            # Combines global pose token (add) with spatial detail tokens (concat)
            # This requires _inject_spatial() which computes spatial tokens separately
            raise ValueError(
                "spatial_token integration must use inject_spatial() method, "
                "not inject(). See inject_spatial() for usage."
            )

        else:
            raise ValueError(f"Unknown integration method: {self.integration}")

    def inject_spatial(
        self,
        encoder_hidden_states: torch.Tensor,  # [B*N, seq_len, embed_dim]
        ref_view_idx: Union[int, torch.Tensor] = 0,
        n_views: int = 6,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Inject pose conditioning with spatial token preservation.

        For Plucker method with spatial_token integration:
        1. Adds global pose token to first prompt token (same as "add")
        2. Concatenates spatial tokens (S×S) to prompt sequence

        Result: [B*N, seq_len + S*S, embed_dim] where S = spatial_token_size

        Gradient flow: same gating as inject() (trainable + training mode).
        """
        use_grad = self.trainable and self.training
        grad_ctx = nullcontext() if use_grad else torch.no_grad()
        with grad_ctx:
            return self._inject_spatial_core(
                encoder_hidden_states, ref_view_idx, n_views, batch_size
            )

    def _inject_spatial_core(
        self,
        encoder_hidden_states: torch.Tensor,
        ref_view_idx: Union[int, torch.Tensor],
        n_views: int,
        batch_size: Optional[int],
    ) -> torch.Tensor:
        """Core spatial injection: global add + spatial token concat."""
        BN, seq_len, C = encoder_hidden_states.shape
        if batch_size is None:
            batch_size = BN // n_views
        device = encoder_hidden_states.device

        # Step 1: Compute global pose token (same as _inject_core "add")
        # Step 2: Compute spatial tokens
        if isinstance(ref_view_idx, int) and ref_view_idx == 0:
            pose_embeds = self.get_fixed_embeddings().to(device)
            spatial_tokens = self._compute_spatial_tokens(self.c2w).to(device)
            # Expand for batch
            pose_embeds = pose_embeds.unsqueeze(0).expand(batch_size, -1, -1).reshape(BN, C)
            n_spatial = spatial_tokens.shape[1]
            spatial_tokens = spatial_tokens.unsqueeze(0).expand(
                batch_size, -1, -1, -1
            ).reshape(BN, n_spatial, C)
        else:
            all_pose = []
            all_spatial = []
            for b in range(batch_size):
                idx = ref_view_idx[b].item() if torch.is_tensor(ref_view_idx) else ref_view_idx
                rotated_cameras = get_rotated_cameras(self.c2w, idx, self.n_views)
                rotated_indices = [(idx + i) % self.n_views for i in range(self.n_views)]
                pe = self._compute_pose_token(rotated_cameras, rotated_indices)
                st = self._compute_spatial_tokens(rotated_cameras, rotated_indices)
                all_pose.append(pe)
                all_spatial.append(st)
            pose_embeds = torch.stack(all_pose).reshape(BN, C).to(device)
            spatial_tokens = torch.stack(all_spatial).reshape(
                BN, all_spatial[0].shape[1], C
            ).to(device)

        # Cast dtype
        pose_embeds = pose_embeds.to(dtype=encoder_hidden_states.dtype)
        spatial_tokens = spatial_tokens.to(dtype=encoder_hidden_states.dtype)

        # Step 3: Global add (same as "add" integration)
        out = encoder_hidden_states.clone()
        out[:, 0, :] = out[:, 0, :] + pose_embeds

        # Step 4: Concat spatial tokens
        out = torch.cat([out, spatial_tokens], dim=1)
        # Result: [BN, seq_len + S*S, embed_dim]
        return out

    def extra_repr(self) -> str:
        return (
            f"method={self.method}, integration={self.integration}, "
            f"embed_dim={self.embed_dim}, n_views={self.n_views}, "
            f"trainable={self.trainable}, spatial_token_size={self.spatial_token_size}"
        )


# =============================================================================
# Training Integration Helpers
# =============================================================================

def create_pose_injector_from_config(
    config: dict,
    device: str = "cuda",
) -> Optional[PoseConditioningInjector]:
    """
    Create PoseConditioningInjector from training config dict.

    Config keys:
        pose_conditioning:
            enabled: true
            method: "spherical"       # spherical | extrinsic | plucker
            integration: "concat"     # concat | add | replace_last
            embed_dim: 1024
            camera_json: "mouse_extensions/inference/cameras/m5_cameras.json"
            plucker_resolution: 64    # Only for plucker method

    Returns:
        PoseConditioningInjector or None if not enabled
    """
    pose_cfg = config.get("pose_conditioning", {})
    if not pose_cfg.get("enabled", False):
        return None

    injector = PoseConditioningInjector(
        method=pose_cfg.get("method", "spherical"),
        integration=pose_cfg.get("integration", "concat"),
        embed_dim=pose_cfg.get("embed_dim", 1024),
        camera_json_path=pose_cfg.get(
            "camera_json",
            "mouse_extensions/inference/cameras/m5_cameras.json"
        ),
        plucker_resolution=pose_cfg.get("plucker_resolution", 64),
        trainable=pose_cfg.get("trainable", False),
        spatial_token_size=pose_cfg.get("spatial_token_size", 8),
    )

    return injector.to(device)


# =============================================================================
# Usage Example (for reference)
# =============================================================================
#
# In train_diffusion.py, add these minimal changes:
#
# 1. After model setup:
#   pose_injector = create_pose_injector_from_config(vars(cfg))
#   if pose_injector is not None:
#       pose_injector = pose_injector.to(accelerator.device)
#
# 2. In training loop (around line 665):
#   if pose_injector is not None:
#       prompt_embeddings = pose_injector.inject(
#           prompt_embeddings,
#           ref_view_idx=batch.get('ref_view_idx', 0),
#           n_views=cfg.n_views,
#           batch_size=batch_size,
#       )
#   model_output = models['unet'](
#       noisy_latents, timesteps,
#       encoder_hidden_states=prompt_embeddings,
#       class_labels=image_embeddings,
#   )
#
# 3. Dataset must return 'ref_view_idx' in batch dict for random ref mode.
