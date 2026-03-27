"""Orientation-aware Gaussian filter for bottom-view artifact suppression.

Targets flat Gaussians whose thinnest axis aligns with world Z — these appear
as thin white lines when viewed from below. Attenuates opacity rather than
deleting, making the filter safe and reversible.

Design rationale (MoA 3-model audit, 2026-03-26):
  - 93.8% of Gaussians have anisotropy ratio ≥ 30 → ratio-only filter is non-discriminative
  - Flat Gaussians are NORMAL for surface representation
  - The discriminator is orientation: min-scale axis ≈ world Z

Usage:
    from mouse_extensions.model.orientation_filter import suppress_z_aligned_flat

    gaussians.apply_all_filters(...)
    stats = suppress_z_aligned_flat(gaussians)
    print(f"Suppressed {stats['n_suppressed']} / {stats['n_total']} Gaussians")
"""

from __future__ import annotations

from typing import Dict

import torch


def _quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions [w,x,y,z] to 3x3 rotation matrices.

    Args:
        q: (N, 4) normalized quaternions.

    Returns:
        (N, 3, 3) rotation matrices.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def suppress_z_aligned_flat(
    gaussians,
    ratio_thresh: float = 30.0,
    z_align_thresh: float = 0.15,
    opacity_ceil: float = 0.4,
    attenuation: float = 0.1,
    mode: str = "attenuate",
) -> Dict[str, int]:
    """Suppress flat Gaussians whose thinnest axis aligns with world Z.

    This targets the specific geometric configuration causing "thin white line"
    artifacts in novel bottom views, without affecting legitimate flat surfaces
    (walls, floors viewed from normal angles).

    Args:
        gaussians: GaussianModel instance (modified in-place).
        ratio_thresh: Min anisotropy ratio (max_scale/min_scale) to consider.
            Default 30.0 targets the 93.8% that are structurally flat.
        z_align_thresh: Max |dot(thin_axis, Z)| to flag. Gaussians with alignment
            BELOW this are edge-on from bottom view → line artifacts. Default 0.15
            ≈ within 8.6° of horizontal. Start conservative; raise if artifacts persist.
        opacity_ceil: Max opacity for suppression. High-opacity Gaussians are
            likely real surfaces; default 0.4 protects them.
        attenuation: Opacity multiplier for flagged Gaussians (0.1 = 90% reduction).
            Only used when mode="attenuate".
        mode: "attenuate" (safe, default) or "prune" (aggressive, removes Gaussians).

    Returns:
        Dict with keys: n_total, n_flat, n_z_aligned, n_suppressed.
    """
    scaling = gaussians.get_scaling  # (N, 3) positive values
    rotation = gaussians.get_rotation  # (N, 4) unit quaternions [w,x,y,z]
    opacity = gaussians.get_opacity  # (N, 1)

    N = scaling.shape[0]
    stats = {"n_total": N, "n_flat": 0, "n_z_aligned": 0, "n_suppressed": 0}

    if N == 0:
        return stats

    # Stage 1: Identify flat Gaussians (high anisotropy ratio)
    s_max = scaling.max(dim=1).values
    s_min = scaling.min(dim=1).values.clamp(min=1e-8)
    ratio = s_max / s_min
    is_flat = ratio > ratio_thresh
    stats["n_flat"] = int(is_flat.sum().item())

    if stats["n_flat"] == 0:
        return stats

    # Stage 2: Check Z-alignment of thinnest axis
    # For each Gaussian, find which of the 3 local axes is the thinnest
    min_axis_idx = scaling.argmin(dim=1)  # (N,) indices 0/1/2

    # Convert quaternions to rotation matrices
    R = _quaternion_to_rotation_matrix(rotation)  # (N, 3, 3)

    # Extract the world-space direction of the thinnest axis
    # R[:, :, k] = world-space direction of local axis k
    batch_idx = torch.arange(N, device=R.device)
    thin_axis = R[batch_idx, :, min_axis_idx]  # (N, 3)

    # Dot product with world Z = [0, 0, 1] → just take the Z component
    z_alignment = thin_axis[:, 2].abs()  # (N,)

    # Low z_alignment = thin axis in XY plane = edge-on from bottom view = line artifact
    # High z_alignment = thin axis in Z = face-on from bottom view = NOT an artifact
    is_z_aligned = is_flat & (z_alignment < z_align_thresh)
    stats["n_z_aligned"] = int(is_z_aligned.sum().item())

    # Stage 3: Opacity gate — protect high-opacity surfaces
    is_low_opacity = opacity.squeeze(-1) < opacity_ceil
    is_artifact = is_z_aligned & is_low_opacity
    stats["n_suppressed"] = int(is_artifact.sum().item())

    if stats["n_suppressed"] == 0:
        return stats

    # Apply suppression
    if mode == "attenuate":
        # Attenuate opacity in log-odds space (where _opacity is stored)
        # opacity_activated = sigmoid(_opacity), so to multiply by `attenuation`:
        # new_logit = log(attenuation * sigmoid(old_logit) / (1 - attenuation * sigmoid(old_logit)))
        # Simpler: directly modify _opacity via the activated value
        with torch.no_grad():
            current_op = gaussians._opacity[is_artifact]  # log-odds
            activated = torch.sigmoid(current_op)
            new_activated = activated * attenuation
            # Inverse sigmoid (logit) with clamp for numerical safety
            new_activated = new_activated.clamp(1e-6, 1 - 1e-6)
            gaussians._opacity[is_artifact] = torch.log(new_activated / (1 - new_activated))
    elif mode == "prune":
        gaussians.filter(~is_artifact)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'attenuate' or 'prune'.")

    return stats
