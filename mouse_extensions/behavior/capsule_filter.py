"""Capsule Union Filter for 3D Gaussian Foreground Segmentation.

Filters Gaussians to foreground-only using anatomical capsules
defined by MAMMAL 22-keypoint skeleton. Each bone segment becomes
a capsule (cylinder + hemisphere caps) with anatomically-derived radius.

Advantages over N>=2 multi-view visibility:
- Tighter fit to actual body shape
- Handles tail correctly (per-segment tapering)
- Simultaneous body-part assignment (bone proximity)
- No rendering required (pure 3D geometry)

Usage on gpu03:
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.behavior.capsule_filter \
        --frame-idx 0 500 1000 \
        --n-views-thresh 2 \
        --output-dir outputs/behavior/capsule_filter

Author: FaceLift Mouse Extensions
Date: 2026-03-21
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# MAMMAL Skeleton Definition
# ============================================================

MAMMAL_KP_NAMES = [
    "nose", "head", "neck", "spine_mid", "spine_back", "tail_base",
    "tail_mid", "tail_tip",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_ear", "right_ear",
]

# Body-part groups for reporting
BODY_PART_GROUPS = {
    "head": ["nose_head", "head_neck", "left_ear_cap", "right_ear_cap"],
    "torso": ["neck_spine_mid", "spine_mid_back", "spine_back_tail_base"],
    "tail": ["tail_base_mid", "tail_mid_tip"],
    "front_left": ["left_shoulder_elbow", "left_elbow_wrist"],
    "front_right": ["right_shoulder_elbow", "right_elbow_wrist"],
    "hind_left": ["left_hip_knee", "left_knee_ankle"],
    "hind_right": ["right_hip_knee", "right_knee_ankle"],
}

# Bone definitions: (name, start_kp, end_kp, radius_mm)
# Radii estimated from adult mouse anatomy (~25g, ~80mm body)
BONE_DEFINITIONS = [
    # Head
    ("nose_head",            "nose",           "head",           6.0),
    ("head_neck",            "head",           "neck",           8.0),
    ("left_ear_cap",         "head",           "left_ear",       4.0),
    ("right_ear_cap",        "head",           "right_ear",      4.0),
    # Spine (thickest part)
    ("neck_spine_mid",       "neck",           "spine_mid",     10.0),
    ("spine_mid_back",       "spine_mid",      "spine_back",    10.0),
    ("spine_back_tail_base", "spine_back",     "tail_base",      8.0),
    # Tail (tapering)
    ("tail_base_mid",        "tail_base",      "tail_mid",       5.0),
    ("tail_mid_tip",         "tail_mid",       "tail_tip",       3.0),
    # Front limbs
    ("left_shoulder_elbow",  "left_shoulder",  "left_elbow",     5.0),
    ("left_elbow_wrist",     "left_elbow",     "left_wrist",     3.5),
    ("right_shoulder_elbow", "right_shoulder",  "right_elbow",   5.0),
    ("right_elbow_wrist",    "right_elbow",    "right_wrist",    3.5),
    # Hind limbs
    ("left_hip_knee",        "left_hip",       "left_knee",      6.0),
    ("left_knee_ankle",      "left_knee",      "left_ankle",     4.0),
    ("right_hip_knee",       "right_hip",      "right_knee",     6.0),
    ("right_knee_ankle",     "right_knee",     "right_ankle",    4.0),
]

# Global padding (mm) to account for fur, Gaussian spread, and KP jitter
DEFAULT_PADDING_MM = 3.0


# ============================================================
# Coordinate Transform: MAMMAL mm → GS-LRM normalized
# ============================================================

def load_kp_to_gslrm_transform(frame_dir: str) -> Optional[Dict]:
    """Load the coordinate transform from MAMMAL keypoint space to GS-LRM space.

    The transform is stored during preprocessing as part of the camera setup.
    We need: scale factor and translation to map mm coords → normalized coords.

    Returns:
        dict with 'scale', 'offset' or None if not available
    """
    # Try loading from preprocessing metadata
    meta_path = Path(frame_dir) / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if "kp_transform" in meta:
            return meta["kp_transform"]

    # Fallback: estimate from camera parameters
    cam_path = Path(frame_dir) / "opencv_cameras.json"
    if cam_path.exists():
        with open(cam_path) as f:
            cams = json.load(f)
        # GS-LRM normalizes to radius ~2.7
        # MAMMAL keypoints are in world mm coordinates
        # The transform is typically: gslrm_xyz = (mm_xyz - center) * scale
        # We'll need to derive this from the extrinsics
        return None

    return None


def transform_keypoints_to_gslrm(
    kp_mm: np.ndarray,
    gaussians_xyz: np.ndarray,
    method: str = "bbox_match",
) -> np.ndarray:
    """Transform MAMMAL keypoints (mm) to GS-LRM coordinate space.

    Args:
        kp_mm: (22, 3) keypoints in mm
        gaussians_xyz: (N, 3) Gaussian centers in GS-LRM space
        method: 'bbox_match' or 'centroid_scale'

    Returns:
        kp_gslrm: (22, 3) keypoints in GS-LRM space
    """
    if method == "bbox_match":
        # Match bounding boxes: align KP bbox to Gaussian bbox
        # Use percentile-based bbox to be robust to outliers
        g_center = np.median(gaussians_xyz, axis=0)
        g_extent = np.percentile(gaussians_xyz, 95, axis=0) - np.percentile(gaussians_xyz, 5, axis=0)

        kp_center = np.mean(kp_mm, axis=0)
        kp_extent = kp_mm.max(axis=0) - kp_mm.min(axis=0)
        kp_extent = np.maximum(kp_extent, 1e-6)  # avoid division by zero

        # Scale to match extents
        scale = g_extent / kp_extent
        # Use uniform scale (average) to preserve aspect ratio
        uniform_scale = scale.mean()

        kp_gslrm = (kp_mm - kp_center) * uniform_scale + g_center
        return kp_gslrm

    elif method == "centroid_scale":
        # Simple centroid alignment + scale matching
        g_center = np.mean(gaussians_xyz, axis=0)
        g_std = np.std(gaussians_xyz)

        kp_center = np.mean(kp_mm, axis=0)
        kp_std = np.std(kp_mm - kp_center)

        scale = g_std / max(kp_std, 1e-6)
        kp_gslrm = (kp_mm - kp_center) * scale + g_center
        return kp_gslrm

    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================
# Core: Capsule Filtering
# ============================================================

def point_to_segment_distance_sq(
    points: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
) -> np.ndarray:
    """Squared distance from points to a line segment.

    Args:
        points: (N, 3) query points
        seg_a: (3,) segment start
        seg_b: (3,) segment end

    Returns:
        dist_sq: (N,) squared distances
    """
    ab = seg_b - seg_a
    ap = points - seg_a

    seg_len_sq = np.dot(ab, ab)
    if seg_len_sq < 1e-12:
        # Degenerate segment (A == B) → distance to point A
        return np.sum(ap * ap, axis=1)

    # Parameter t: projection of AP onto AB, clamped to [0, 1]
    t = np.dot(ap, ab) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)

    # Closest point on segment
    closest = seg_a + np.outer(t, ab)
    diff = points - closest
    return np.sum(diff * diff, axis=1)


def filter_by_capsules(
    gaussians_xyz: np.ndarray,
    keypoints: np.ndarray,
    kp_names: List[str],
    bone_defs: List[Tuple] = None,
    padding_mm: float = DEFAULT_PADDING_MM,
    scale_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter Gaussians using capsule union and assign to body parts.

    Args:
        gaussians_xyz: (N, 3) Gaussian centers in GS-LRM space
        keypoints: (22, 3) keypoints in GS-LRM space (already transformed)
        kp_names: list of keypoint names matching keypoints array
        bone_defs: list of (name, start_kp, end_kp, radius_mm) tuples
        padding_mm: additional padding in mm (applied after scale)
        scale_factor: mm-to-gslrm scale factor for radii conversion

    Returns:
        mask: (N,) boolean, True for foreground
        bone_idx: (N,) int, closest bone index (-1 if background)
        bone_dist: (N,) float, distance to closest bone surface
    """
    if bone_defs is None:
        bone_defs = BONE_DEFINITIONS

    N = len(gaussians_xyz)
    kp_dict = {name: keypoints[i] for i, name in enumerate(kp_names)}

    # Track minimum distance to any capsule surface
    min_dist_sq = np.full(N, np.inf, dtype=np.float64)
    assigned_bone = np.full(N, -1, dtype=np.int32)
    bone_radii_sq = []

    for bone_idx, (bone_name, kp_start, kp_end, radius_mm) in enumerate(bone_defs):
        if kp_start not in kp_dict or kp_end not in kp_dict:
            bone_radii_sq.append(0.0)
            continue

        a = kp_dict[kp_start]
        b = kp_dict[kp_end]
        radius = (radius_mm + padding_mm) * scale_factor
        bone_radii_sq.append(radius ** 2)

        # Distance from all Gaussians to this bone segment
        dist_sq = point_to_segment_distance_sq(gaussians_xyz, a, b)

        # Update closest bone assignment
        closer = dist_sq < min_dist_sq
        min_dist_sq[closer] = dist_sq[closer]
        assigned_bone[closer] = bone_idx

    # Check if each Gaussian is within its assigned capsule radius
    bone_radii_sq = np.array(bone_radii_sq)
    valid_assignment = assigned_bone >= 0
    inside_capsule = np.zeros(N, dtype=bool)
    inside_capsule[valid_assignment] = (
        min_dist_sq[valid_assignment] <= bone_radii_sq[assigned_bone[valid_assignment]]
    )

    # Compute actual distance (not squared) for reporting
    bone_dist = np.sqrt(min_dist_sq)

    return inside_capsule, assigned_bone, bone_dist


def get_body_part_for_bone(bone_idx: int) -> str:
    """Map bone index to body part group name."""
    if bone_idx < 0 or bone_idx >= len(BONE_DEFINITIONS):
        return "background"

    bone_name = BONE_DEFINITIONS[bone_idx][0]
    for part_name, bone_names in BODY_PART_GROUPS.items():
        if bone_name in bone_names:
            return part_name
    return "unknown"


def capsule_filter_pipeline(
    gaussians_xyz: np.ndarray,
    keypoints_mm: np.ndarray,
    kp_names: List[str] = None,
    padding_mm: float = DEFAULT_PADDING_MM,
    n_views_mask: Optional[np.ndarray] = None,
) -> Dict:
    """Full capsule filtering pipeline with coordinate transform.

    Args:
        gaussians_xyz: (N, 3) in GS-LRM space
        keypoints_mm: (22, 3) in mm space
        kp_names: keypoint names (default: MAMMAL_KP_NAMES)
        padding_mm: capsule padding in mm
        n_views_mask: (N,) optional pre-filter from N>=2 visibility

    Returns:
        dict with 'mask', 'bone_idx', 'bone_dist', 'stats'
    """
    if kp_names is None:
        kp_names = MAMMAL_KP_NAMES

    # Step 1: Transform keypoints to GS-LRM space
    # Use only foreground Gaussians for bbox matching if N>=2 mask available
    if n_views_mask is not None:
        ref_xyz = gaussians_xyz[n_views_mask]
    else:
        ref_xyz = gaussians_xyz

    kp_gslrm = transform_keypoints_to_gslrm(keypoints_mm, ref_xyz)

    # Estimate scale factor: mm → GS-LRM units
    kp_extent_mm = np.ptp(keypoints_mm, axis=0).max()
    kp_extent_gslrm = np.ptp(kp_gslrm, axis=0).max()
    scale_factor = kp_extent_gslrm / max(kp_extent_mm, 1e-6)

    # Step 2: Filter by capsules
    mask, bone_idx, bone_dist = filter_by_capsules(
        gaussians_xyz, kp_gslrm, kp_names,
        padding_mm=padding_mm,
        scale_factor=scale_factor,
    )

    # Step 3: Combine with N>=2 if available
    if n_views_mask is not None:
        combined_mask = mask & n_views_mask
    else:
        combined_mask = mask

    # Step 4: Compute statistics
    body_part_counts = {}
    for part_name in BODY_PART_GROUPS:
        part_bones = BODY_PART_GROUPS[part_name]
        part_bone_indices = [
            i for i, (name, *_) in enumerate(BONE_DEFINITIONS) if name in part_bones
        ]
        part_mask = np.isin(bone_idx, part_bone_indices) & combined_mask
        body_part_counts[part_name] = int(part_mask.sum())

    stats = {
        "total_input": int(len(gaussians_xyz)),
        "capsule_only": int(mask.sum()),
        "n_views_only": int(n_views_mask.sum()) if n_views_mask is not None else None,
        "combined": int(combined_mask.sum()),
        "body_parts": body_part_counts,
        "scale_factor_mm_to_gslrm": float(scale_factor),
        "padding_mm": float(padding_mm),
    }

    return {
        "mask": combined_mask,
        "bone_idx": bone_idx,
        "bone_dist": bone_dist,
        "kp_gslrm": kp_gslrm,
        "stats": stats,
    }


# ============================================================
# SH Color Filter (complementary)
# ============================================================

def sh_color_filter(
    features_dc: np.ndarray,
    brightness_threshold: float = 0.85,
) -> np.ndarray:
    """Filter Gaussians by SH DC brightness (dark mouse on white BG).

    Args:
        features_dc: (N, 3) SH DC coefficients
        brightness_threshold: Remove Gaussians brighter than this

    Returns:
        fg_mask: (N,) boolean, True for likely foreground
    """
    C0 = 0.28209479177387814
    rgb = features_dc * C0 + 0.5  # SH2RGB conversion
    rgb = np.clip(rgb, 0.0, 1.0)
    brightness = rgb.mean(axis=-1)
    return brightness < brightness_threshold


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Capsule Union Filter for Gaussians")
    parser.add_argument("--frame-idx", type=int, nargs="+", default=[0, 500, 1000])
    parser.add_argument("--data-dir", type=str,
                        default="/home/joon/data/preprocessed/FaceLift_mouse/M5")
    parser.add_argument("--kp-path", type=str,
                        default="/node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz")
    parser.add_argument("--gaussian-dir", type=str, default=None,
                        help="Directory with per-frame Gaussian .npz files")
    parser.add_argument("--n-views-thresh", type=int, default=2)
    parser.add_argument("--padding-mm", type=float, default=DEFAULT_PADDING_MM)
    parser.add_argument("--output-dir", type=str, default="outputs/behavior/capsule_filter")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load keypoints
    kp_data = np.load(args.kp_path)
    all_kp = kp_data["keypoints"]  # (3600, 22, 3)
    kp_names = list(kp_data["keypoint_names"])

    print(f"Loaded keypoints: {all_kp.shape}")
    print(f"Keypoint names: {kp_names}")
    print(f"Frames to process: {args.frame_idx}")

    for fidx in args.frame_idx:
        print(f"\n--- Frame {fidx} ---")
        frame_id = f"{fidx:06d}"
        frame_dir = Path(args.data_dir) / frame_id

        # Load Gaussians (from render output or raw)
        if args.gaussian_dir:
            gs_path = Path(args.gaussian_dir) / f"{frame_id}.npz"
        else:
            gs_path = frame_dir / "gaussians_raw" / f"{frame_id}.npz"

        if not gs_path.exists():
            print(f"  Gaussian file not found: {gs_path}")
            continue

        gs_data = np.load(gs_path)
        xyz = gs_data["xyz"]
        print(f"  Gaussians: {xyz.shape[0]}")

        # Get keypoints for this frame
        kp_mm = all_kp[fidx]

        # Run capsule filter
        result = capsule_filter_pipeline(
            xyz, kp_mm, kp_names,
            padding_mm=args.padding_mm,
        )

        stats = result["stats"]
        print(f"  Capsule filter: {stats['total_input']} → {stats['capsule_only']}")
        print(f"  Body parts: {stats['body_parts']}")
        print(f"  Scale factor: {stats['scale_factor_mm_to_gslrm']:.6f}")

        # Save results
        result_path = out_dir / f"capsule_filter_{frame_id}.json"
        with open(result_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved: {result_path}")


if __name__ == "__main__":
    main()
