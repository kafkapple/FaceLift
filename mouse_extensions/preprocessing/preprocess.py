#!/usr/bin/env python3
"""
Unified Mouse Preprocessor
==========================

Single entry point for ALL preprocessing versions (D6-D9).

Paradigms:
    geometry_preserving (D6): Accurate PP, various crop strategies
        D6-1: Resize only, letterbox padding
        D6-2: Virtual camera relocation (PP shift)
        D6-3: PP-correct crop with triangulation
    
    pp_centered_shift (D7): Shift image to center PP at 256
        D7  : fx_only scale mode
        D7.1: individual scale mode [RECOMMENDED]
        D7.2: average scale mode
    
    precision_homography (D8): Homography with skew correction
        D8  : Standard precision
        D8.1: With 1.3x zoom
    
    native (D9): Original resolution, no transformation
        For A6000+ GPUs (4.5x memory)

Usage:
    # Using preset
    python -m mouse_extensions.preprocessing.preprocess --preset D7.1 \\
        --input-dir /path/to/raw --output-dir /path/to/D7.1

    # List available presets
    python -m mouse_extensions.preprocessing.preprocess --list-presets

Created: 2026-01-21
Updated: 2026-01-22 - Integrated D6, D9 paradigms
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .presets import PRESETS, get_preset, get_recommended, list_presets as list_preset_names


# Constants
GSLRM_EXACT_FX = 548.9937744140625

# Frame discontinuity markers (markerless_mouse_1_nerf dataset)
# These frames mark positions where temporal discontinuity occurs (some frames missing in raw data)
# Note: These frames themselves are valid and should NOT be excluded from training.
# They only indicate where the video has gaps in the original recording.
DISCONTINUITY_FRAMES = {5900, 11800, 17700}  # For documentation only, not used for exclusion

# Up-alignment utilities
def load_up_direction(vertical_lines_path: Path) -> np.ndarray:
    """Load up direction from vertical_lines.npz.
    
    The stored up vector points from floor to ceiling in the original coordinate system.
    We negate it to match the auto_orient convention (Z-down becomes Z-up after rotation).
    """
    data = np.load(vertical_lines_path)
    up = -data['up']  # Negate for coordinate convention
    return up / np.linalg.norm(up)


def compute_up_from_cameras(extrinsics: np.ndarray) -> np.ndarray:
    """Estimate up direction from camera Y-axis mean.
    
    Fallback when vertical_lines.npz is not available.
    Assumes cameras are roughly level (Y-axis points up in camera frame).
    """
    # extrinsics shape: (num_views, 4, 4) or (num_views, 3, 4)
    # Camera Y-axis is the second column of rotation matrix (pointing up in camera frame)
    if extrinsics.shape[1] == 4:
        R = extrinsics[:, :3, :3]
    else:
        R = extrinsics[:, :3, :3]
    
    # Average Y-axis across all cameras
    y_axes = R[:, :, 1]  # shape: (num_views, 3)
    up = np.mean(y_axes, axis=0)
    return up / np.linalg.norm(up)

def get_extrinsics_from_cameras(cameras: List[Dict]) -> np.ndarray:
    """Extract extrinsics array from cameras, handling both formats.
    
    Supports:
    - extrinsic: 4x4 matrix format
    - R, T: separate rotation and translation format
    
    Returns:
        np.ndarray: shape (num_views, 4, 4)
    """
    extrinsics = []
    for cam in cameras:
        if "extrinsic" in cam:
            E = cam["extrinsic"]
            if E.shape == (3, 4):
                E_full = np.eye(4)
                E_full[:3, :] = E
                extrinsics.append(E_full)
            else:
                extrinsics.append(E)
        elif "R" in cam and "T" in cam:
            E = np.eye(4)
            E[:3, :3] = cam["R"]
            T = cam["T"].flatten() if cam["T"].ndim > 1 else cam["T"]
            E[:3, 3] = T
            extrinsics.append(E)
        else:
            raise KeyError(f"Camera missing extrinsic data. Keys: {list(cam.keys())}")
    return np.stack(extrinsics)




def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Compute rotation matrix that rotates vec1 to vec2.
    
    Uses Rodrigues' rotation formula.
    """
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    if s < 1e-10:  # vectors are parallel
        if c > 0:
            return np.eye(3)
        else:
            # 180 degree rotation - find perpendicular axis
            perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(a, perp)
            axis = axis / np.linalg.norm(axis)
            # Rodrigues for 180 deg
            return 2 * np.outer(axis, axis) - np.eye(3)
    
    # Skew-symmetric cross-product matrix
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R


def apply_up_alignment_to_cameras(cameras: List[Dict], up: np.ndarray) -> List[Dict]:
    """Apply up-direction alignment to camera extrinsics.
    
    Rotates world coordinate system so that 'up' direction aligns with Z-axis.
    Handles both 'extrinsic' format and separate 'R', 'T' format.
    """
    # Compute rotation that aligns 'up' to [0, 0, 1]
    target_up = np.array([0, 0, 1])
    R_align = rotation_matrix_from_vectors(up, target_up)
    
    aligned_cameras = []
    for cam in cameras:
        cam_aligned = cam.copy()
        
        # Handle both camera formats
        if 'extrinsic' in cam:
            # 4x4 extrinsic matrix format
            E = cam['extrinsic'].copy()
            R = E[:3, :3]
            t = E[:3, 3]
            
            R_new = R @ R_align.T
            t_new = R_align @ t
            
            E_new = np.eye(4)
            E_new[:3, :3] = R_new
            E_new[:3, 3] = t_new
            cam_aligned['extrinsic'] = E_new
        else:
            # Separate R, T format (raw camera pkl)
            R = cam['R'].copy()
            T = cam['T'].copy().flatten()
            
            # Apply alignment rotation to the world
            R_new = R @ R_align.T
            T_new = R_align @ T
            
            cam_aligned['R'] = R_new
            cam_aligned['T'] = T_new.reshape(-1, 1) if cam['T'].ndim > 1 else T_new
        
        aligned_cameras.append(cam_aligned)
    
    return aligned_cameras


def compute_adaptive_zoom(masks: List[np.ndarray], target_fill: float = 0.8, 
                          zoom_range: Tuple[float, float] = (1.2, 1.5)) -> float:
    """Compute zoom factor to make object fill target_fill of frame.
    
    Args:
        masks: List of binary masks for all views
        target_fill: Target ratio of bounding box to frame (0.8 = 80%)
        zoom_range: (min_zoom, max_zoom) clipping range
    
    Returns:
        Zoom factor
    """
    max_bbox_ratio = 0
    
    for mask in masks:
        if mask is None or mask.sum() == 0:
            continue
        
        # Find bounding box
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            continue
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        bbox_h = y_max - y_min
        bbox_w = x_max - x_min
        
        # Ratio of bbox to image
        h, w = mask.shape[:2]
        ratio = max(bbox_h / h, bbox_w / w)
        max_bbox_ratio = max(max_bbox_ratio, ratio)
    
    if max_bbox_ratio == 0:
        return 1.0
    
    # Compute zoom to achieve target fill
    zoom = target_fill / max_bbox_ratio
    return np.clip(zoom, zoom_range[0], zoom_range[1])


def compute_adaptive_zoom_coverage(masks: List[np.ndarray], target_coverage: float = 0.05,
                                    zoom_range: Tuple[float, float] = (1.0, 2.5)) -> float:
    """Compute zoom factor based on foreground coverage target.
    
    M3/D10.3 method: Instead of bbox-based fill ratio, directly target
    a specific foreground coverage percentage.
    
    Args:
        masks: List of binary masks for all views
        target_coverage: Target foreground ratio (0.05 = 5%)
        zoom_range: (min_zoom, max_zoom) clipping range
    
    Returns:
        Zoom factor to achieve target coverage
    
    Rationale:
        - Current coverage = fg_pixels / total_pixels
        - Zoom squares the image area, so coverage scales linearly with zoom^2
        - zoom = sqrt(target_coverage / current_coverage)
    """
    # Compute current coverage across all views
    coverages = []
    for mask in masks:
        if mask is None:
            continue
        fg_pixels = (mask > 0).sum()
        total_pixels = mask.size
        coverage = fg_pixels / total_pixels if total_pixels > 0 else 0
        if coverage > 0:
            coverages.append(coverage)
    
    if not coverages:
        return 1.0
    
    # Use mean coverage across views
    current_coverage = np.mean(coverages)
    
    if current_coverage < 1e-6:
        return zoom_range[1]  # Max zoom if no foreground
    
    # Compute zoom to achieve target coverage
    # After zoom, image area scales by zoom^2, but we crop centered,
    # so foreground coverage increases proportionally to zoom^2
    zoom = np.sqrt(target_coverage / current_coverage)
    
    return float(np.clip(zoom, zoom_range[0], zoom_range[1]))

def compute_persample_zoom_coverage(mask: np.ndarray, target_coverage: float = 0.05,
                                     zoom_range: Tuple[float, float] = (1.0, 2.5)) -> float:
    """Compute zoom factor for a SINGLE sample based on its coverage.
    
    Unlike compute_adaptive_zoom_coverage which uses global average,
    this computes optimal zoom for each sample individually.
    
    Args:
        mask: Single binary mask (any view from the sample)
        target_coverage: Target foreground ratio (0.05 = 5%)
        zoom_range: (min_zoom, max_zoom) clipping range
    
    Returns:
        Zoom factor for this specific sample
    """
    if mask is None or mask.size == 0:
        return 1.0
    
    fg_pixels = (mask > 0).sum()
    total_pixels = mask.size
    current_coverage = fg_pixels / total_pixels if total_pixels > 0 else 0
    
    if current_coverage <= 0:
        return zoom_range[0]
    
    # zoom = sqrt(target / current) because coverage scales with zoom^2
    zoom = np.sqrt(target_coverage / current_coverage)
    return float(np.clip(zoom, zoom_range[0], zoom_range[1]))




def compute_fg_coverage_after_zoom(mask: np.ndarray, zoom: float) -> float:
    """Estimate foreground coverage after applying zoom.
    
    Args:
        mask: Binary mask (before zoom)
        zoom: Zoom factor to apply
    
    Returns:
        Estimated foreground coverage after zoom
    """
    if mask is None or mask.size == 0:
        return 0.0
    
    current_coverage = (mask > 0).sum() / mask.size
    # Coverage scales with zoom^2 (area relationship)
    return float(current_coverage * zoom ** 2)




# Config generation paths
FACELIFT_ROOT = Path("/home/joon/dev/FaceLift")
DATASET_CONFIG_DIR = FACELIFT_ROOT / "configs" / "datasets"


class Paradigm(Enum):
    OBJECT_CENTERED = "object_centered"
    GEOMETRY_PRESERVING = "geometry_preserving"
    PP_CENTERED_SHIFT = "pp_centered_shift"
    PRECISION_HOMOGRAPHY = "precision_homography"
    OBJECT_CENTERED_MVG = "object_centered_mvg"
    NATIVE = "native"
    UP_ALIGNED_ZOOM = "up_aligned_zoom"


class TransformType(Enum):
    AFFINE = "affine"
    HOMOGRAPHY = "homography"
    NONE = "none"


class ScaleMode(Enum):
    FX_ONLY = "fx_only"
    INDIVIDUAL = "individual"
    AVERAGE = "average"


@dataclass
class PreprocessConfig:
    """Unified preprocessing configuration."""
    input_dir: Path = None
    output_dir: Path = None
    camera_pkl: Path = None
    frame_interval: int = 5
    max_samples: Optional[int] = None
    val_ratio: float = 0.1
    
    # Transform settings
    paradigm: Paradigm = Paradigm.PP_CENTERED_SHIFT
    transform: TransformType = TransformType.HOMOGRAPHY
    scale_mode: ScaleMode = ScaleMode.INDIVIDUAL
    skew_correction: bool = True
    
    # Target values
    target_fx: float = GSLRM_EXACT_FX
    target_pp: Tuple[float, float] = (256.0, 256.0)
    target_distance: float = 2.7
    output_size: Optional[int] = 512
    zoom: float = 1.0
    
    # D10: Up-alignment settings
    up_alignment: bool = False
    up_source: str = "vertical_lines"  # "vertical_lines" or "camera_y_mean"
    adaptive_zoom: bool = False
    zoom_scope: str = "global"  # "global" or "per_sample"
    zoom_range: Tuple[float, float] = (1.2, 1.5)
    zoom_fill_ratio: float = 0.8
    
    # D6-specific
    d6_method: str = None  # resize_only, virtual_shift, pp_correct_crop
    object_ratio: float = 0.7
    
    # Normalization flags
    normalize_fx: bool = True
    normalize_translation: bool = True
    # P0 Fix: Post-zoom normalization
    normalize_after_zoom: bool = False
    force_pp_to_target: bool = False
    
    # M3/D10.3: Coverage-based zoom settings
    zoom_method: str = "bbox"  # "bbox" or "coverage_based"
    zoom_center_mode: str = "object"  # "object" or "image" (MVG-correct)
    target_fg_coverage: float = 0.05  # 5% foreground coverage target
    min_fg_coverage: float = 0.0  # Minimum coverage warning threshold
    zoom_after_transform: bool = False  # If True, compute coverage after applying transform
    
    # Output structure
    single_folder: bool = False  # If True, save all to samples/ instead of train/val
    
    version: str = "D7.1"

    @classmethod
    def from_preset(cls, preset_name: str, **overrides):
        """Create config from preset."""
        preset = get_preset(preset_name)
        config = cls()
        config.version = preset_name
        
        # Set paradigm
        paradigm_str = preset.get('paradigm', 'pp_centered_shift')
        config.paradigm = Paradigm(paradigm_str)
        
        # ====== GEOMETRY_PRESERVING (D6) ======
        if config.paradigm == Paradigm.GEOMETRY_PRESERVING:
            config.d6_method = preset.get('method')
            config.transform = TransformType.NONE
            config.normalize_fx = preset.get('normalize_fx', True)
            config.normalize_translation = preset.get('normalize_translation', True)
            config.output_size = preset.get('output_size', 512)
            
        # ====== PP_CENTERED_SHIFT (D7) ======
        elif config.paradigm == Paradigm.PP_CENTERED_SHIFT:
            config.transform = TransformType.AFFINE
            config.skew_correction = False
            config.scale_mode = ScaleMode(preset.get('scale_mode', 'individual'))
            config.target_fx = preset.get('target_fx', 549.0)
            
        # ====== PRECISION_HOMOGRAPHY (D8) ======
        elif config.paradigm in [Paradigm.PRECISION_HOMOGRAPHY, Paradigm.OBJECT_CENTERED_MVG]:
            config.transform = TransformType.HOMOGRAPHY
            config.skew_correction = preset.get('skew_correction', True)
            config.scale_mode = ScaleMode(preset.get('scale_mode', 'individual'))
            config.target_fx = preset.get('target_fx', GSLRM_EXACT_FX)
            config.zoom = preset.get('zoom', 1.0)
            # D8.2: Adaptive zoom support
            config.adaptive_zoom = preset.get("adaptive_zoom", False)
            config.zoom_scope = preset.get("zoom_scope", "global")
            config.zoom_range = tuple(preset.get("zoom_range", [1.0, 1.5]))
            config.zoom_fill_ratio = preset.get("zoom_fill_ratio", 0.85)
            # M3/D10.3: Coverage-based zoom
            config.zoom_method = preset.get("zoom_method", "bbox")
            config.zoom_center_mode = preset.get("zoom_center_mode", "object")
            config.target_fg_coverage = preset.get("target_fg_coverage", 0.05)
            config.min_fg_coverage = preset.get("min_fg_coverage", 0.0)
            config.zoom_after_transform = preset.get("zoom_after_transform", False)
            # ★ P0 Fix: Post-zoom normalization options
            config.normalize_after_zoom = preset.get("normalize_after_zoom", False)
            config.force_pp_to_target = preset.get("force_pp_to_target", False)
            # M3_3/M4: Safe zoom and PP correction
            config.safe_zoom = preset.get("safe_zoom", False)
            config.pp_correction = preset.get("pp_correction", False)
            
        # ====== NATIVE (D9, D9_norm) ======
        elif config.paradigm == Paradigm.NATIVE:
            config.transform = TransformType.NONE
            config.output_size = None  # Keep original
            # Read normalization settings from preset (D9_norm uses normalize_translation=True)
            config.normalize_fx = preset.get('normalize_fx', False)
            config.normalize_translation = preset.get('normalize_translation', False)
            config.target_distance = preset.get('target_distance', 2.7)
        # ====== UP_ALIGNED_ZOOM (D10) ======
        elif config.paradigm == Paradigm.UP_ALIGNED_ZOOM:
            config.transform = TransformType.HOMOGRAPHY
            config.skew_correction = preset.get('skew_correction', True)
            config.scale_mode = ScaleMode(preset.get('scale_mode', 'individual'))
            config.target_fx = preset.get('target_fx', GSLRM_EXACT_FX)
            config.zoom = preset.get('zoom', 1.0)
            # D10-specific
            config.up_alignment = preset.get('up_alignment', True)
            config.up_source = preset.get('up_source', 'vertical_lines')
            config.adaptive_zoom = preset.get('adaptive_zoom', False)
            config.zoom_range = tuple(preset.get('zoom_range', [1.2, 1.5]))
            config.zoom_fill_ratio = preset.get('zoom_fill_ratio', 0.8)
            # M3/D10.3: Coverage-based zoom
            config.zoom_method = preset.get('zoom_method', 'bbox')
            config.target_fg_coverage = preset.get('target_fg_coverage', 0.05)
            config.min_fg_coverage = preset.get('min_fg_coverage', 0.0)
            config.zoom_after_transform = preset.get('zoom_after_transform', False)

        # Output structure
        config.single_folder = preset.get('single_folder', False)
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)

        return config


class UnifiedPreprocessor:
    """Unified preprocessor for all dataset versions."""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.cameras = None
        self.num_views = 0
        self.data_loader = None
        self.center_estimator = None

    def load_cameras(self, pkl_path: Path):
        """Load camera parameters from pickle file."""
        with open(pkl_path, 'rb') as f:
            self.cameras = pickle.load(f)
        self.num_views = len(self.cameras)

    def _init_data_loader(self):
        """Initialize data loader for D6/D9."""
        try:
            from .data_loader import DataLoader
            self.data_loader = DataLoader(self.config.input_dir, source_type="raw")
            self.cameras = self.data_loader.get_camera_matrices()
            self.num_views = len(self.cameras)
        except ImportError:
            raise ImportError("data_loader module required for D6/D9 preprocessing")

    def _init_center_estimator(self):
        """Initialize center estimator for D6-3."""
        try:
            from .center_estimation import CenterEstimator
            self.center_estimator = CenterEstimator(self.cameras, method="triangulation")
        except ImportError:
            raise ImportError("center_estimation module required for D6-3")

    # =========================================================================
    # D7/D8 Methods (existing)
    # =========================================================================
    def compute_affine_transform(self, K: np.ndarray) -> np.ndarray:
        cfg = self.config
        orig_fx, orig_fy = K[0, 0], K[1, 1]
        orig_cx, orig_cy = K[0, 2], K[1, 2]

        scale_x = cfg.target_fx / orig_fx
        scale_y = cfg.target_fx / orig_fy

        if cfg.scale_mode == ScaleMode.FX_ONLY:
            scale_y = scale_x
        elif cfg.scale_mode == ScaleMode.AVERAGE:
            avg = (scale_x + scale_y) / 2
            scale_x = scale_y = avg

        shift_x = cfg.target_pp[0] - orig_cx * scale_x
        shift_y = cfg.target_pp[1] - orig_cy * scale_y

        return np.array([[scale_x, 0, shift_x], [0, scale_y, shift_y]], dtype=np.float32)

    def compute_homography_transform(self, K: np.ndarray) -> np.ndarray:
        cfg = self.config
        K_target = np.array([
            [cfg.target_fx, 0, cfg.target_pp[0]],
            [0, cfg.target_fx, cfg.target_pp[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        return (K_target @ np.linalg.inv(K)).astype(np.float32)

    def compute_transform(self, K: np.ndarray) -> np.ndarray:
        if self.config.transform == TransformType.HOMOGRAPHY:
            return self.compute_homography_transform(K)
        return self.compute_affine_transform(K)

    def apply_transform(self, image: np.ndarray, mask: np.ndarray, transform: np.ndarray):
        cfg = self.config
        size = cfg.output_size

        if cfg.transform == TransformType.HOMOGRAPHY:
            img_out = cv2.warpPerspective(image, transform, (size, size),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
            mask_out = cv2.warpPerspective(mask, transform, (size, size),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            img_out = cv2.warpAffine(image, transform, (size, size),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
            mask_out = cv2.warpAffine(mask, transform, (size, size),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return img_out, mask_out
    def apply_zoom(self, image: np.ndarray, mask: np.ndarray, zoom: float):
        """Apply zoom with configurable center mode.
        
        zoom_center_mode:
            - 'object': Crop centered on object (current behavior, PP varies)
            - 'image': Crop centered on image center (PP=256 preserved)
        """
        if zoom <= 1.0:
            return image, mask, (0, 0)

        size = self.config.output_size
        crop_size = int(size / zoom)
        
        # Get zoom center mode from config (default: object for backward compat)
        center_mode = getattr(self.config, 'zoom_center_mode', 'object')
        
        if center_mode == 'image':
            # ★ MVG-correct: Center-aligned crop (PP automatically 256)
            crop_x = (size - crop_size) // 2
            crop_y = (size - crop_size) // 2
        else:
            # Original: Object-centered crop (PP varies)
            ys, xs = np.where(mask > 127)
            cx = (xs.min() + xs.max()) / 2 if len(xs) > 0 else size / 2
            cy = (ys.min() + ys.max()) / 2 if len(ys) > 0 else size / 2
            crop_x = max(0, min(int(cx - crop_size/2), size - crop_size))
            crop_y = max(0, min(int(cy - crop_size/2), size - crop_size))

        cropped_img = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
        cropped_mask = mask[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]

        zoomed_img = cv2.resize(cropped_img, (size, size), interpolation=cv2.INTER_LANCZOS4)
        zoomed_mask = cv2.resize(cropped_mask, (size, size), interpolation=cv2.INTER_NEAREST)

        return zoomed_img, zoomed_mask, (crop_x, crop_y)

    def compute_camera_params(self, cam: Dict, zoom: float = 1.0, crop_offset = (0, 0)) -> Dict:
        cfg = self.config
        K, R, T = cam['K'], cam['R'], cam['T']

        if cfg.transform == TransformType.HOMOGRAPHY:
            fx = fy = cfg.target_fx
        else:
            orig_fx, orig_fy = K[0, 0], K[1, 1]
            scale_x = cfg.target_fx / orig_fx
            scale_y = cfg.target_fx / orig_fy
            if cfg.scale_mode == ScaleMode.FX_ONLY:
                scale_y = scale_x
            elif cfg.scale_mode == ScaleMode.AVERAGE:
                scale_x = scale_y = (scale_x + scale_y) / 2
            fx, fy = orig_fx * scale_x, orig_fy * scale_y

        cx, cy = cfg.target_pp

        if zoom > 1.0:
            crop_x, crop_y = crop_offset
            fx, fy = fx * zoom, fy * zoom
            cx, cy = (cx - crop_x) * zoom, (cy - crop_y) * zoom
        
        # ★ P0 FIX: Post-zoom normalization for adaptive zoom presets
        if getattr(cfg, 'normalize_after_zoom', False) and zoom > 1.0:
            renorm_scale = cfg.target_fx / fx
            fx = cfg.target_fx
            fy = fy * renorm_scale
            # ★ BUG FIX (2026-01-25): Center-aligned zoom preserves PP at 256
            # Only scale PP for object-centered zoom
            if getattr(cfg, 'zoom_center_mode', 'object') != 'image':
                cx = cx * renorm_scale
                cy = cy * renorm_scale
            else:
                # For center-aligned zoom, PP should remain at target (256)
                cx, cy = cfg.target_pp
        
        # Alternative: Force PP to target (for GS-LRM compatibility)
        if getattr(cfg, 'force_pp_to_target', False):
            cx, cy = cfg.target_pp

        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = R, T.flatten()
        c2w = np.linalg.inv(w2c)
        cam_pos = c2w[:3, 3]
        dist_scale = cfg.target_distance / np.linalg.norm(cam_pos)
        new_c2w = c2w.copy()
        new_c2w[:3, 3] = cam_pos * dist_scale
        new_w2c = np.linalg.inv(new_c2w)

        return {
            "w": cfg.output_size, "h": cfg.output_size,
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "w2c": new_w2c.tolist(),
            "_transform": {"method": cfg.version, "zoom": zoom, "skew_corrected": cfg.skew_correction}
        }

    # =========================================================================
    # D6 Methods (geometry_preserving)
    # =========================================================================
    def process_d6_1(self, frame_idx: int) -> Optional[Dict]:
        """D6-1: Resize only, letterbox padding."""
        images, masks = self.data_loader.load_frame(frame_idx)
        if all(m.sum() < 100 for m in masks):
            return None

        orig_h, orig_w = images[0].shape[:2]
        scale = self.config.output_size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        
        pad_top = (self.config.output_size - new_h) // 2
        pad_left = (self.config.output_size - new_w) // 2

        proc_images, proc_masks, cam_params = [], [], []

        for i in range(self.num_views):
            # Resize + pad
            resized = cv2.resize(images[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            padded = np.full((self.config.output_size, self.config.output_size, 3), 255, dtype=np.uint8)
            padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
            proc_images.append(padded)

            mask_resized = cv2.resize(masks[i], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            mask_padded = np.zeros((self.config.output_size, self.config.output_size), dtype=np.uint8)
            mask_padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = mask_resized
            proc_masks.append(mask_padded)

            # Accurate intrinsics
            cam = self.cameras[i]
            params = self._build_d6_camera_params(cam, scale, pad_left, pad_top, i)
            cam_params.append(params)

        return {"images": proc_images, "masks": proc_masks, "cameras": cam_params, "frame_idx": frame_idx}

    def process_d6_2(self, frame_idx: int) -> Optional[Dict]:
        """D6-2: Virtual camera relocation (PP shift)."""
        images, masks = self.data_loader.load_frame(frame_idx)
        if all(m.sum() < 100 for m in masks):
            return None

        orig_h, orig_w = images[0].shape[:2]
        scale = self.config.output_size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        pad_top = (self.config.output_size - new_h) // 2
        pad_left = (self.config.output_size - new_w) // 2
        target_center = self.config.output_size / 2

        # Compute mouse centroids
        mouse_centroids = []
        for mask in masks:
            coords = np.where(mask > 127)
            if len(coords[0]) > 0:
                cy = coords[0].mean() * scale + pad_top
                cx = coords[1].mean() * scale + pad_left
            else:
                cy, cx = target_center, target_center
            mouse_centroids.append((cx, cy))

        proc_images, proc_masks, cam_params = [], [], []

        for i in range(self.num_views):
            # Same as D6-1
            resized = cv2.resize(images[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            padded = np.full((self.config.output_size, self.config.output_size, 3), 255, dtype=np.uint8)
            padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
            proc_images.append(padded)

            mask_resized = cv2.resize(masks[i], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            mask_padded = np.zeros((self.config.output_size, self.config.output_size), dtype=np.uint8)
            mask_padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = mask_resized
            proc_masks.append(mask_padded)

            # Virtual PP shift
            cam = self.cameras[i]
            actual_cx = cam["cx"] * scale + pad_left
            actual_cy = cam["cy"] * scale + pad_top
            
            mouse_cx, mouse_cy = mouse_centroids[i]
            pp_offset_x = mouse_cx - target_center
            pp_offset_y = mouse_cy - target_center
            
            params = self._build_d6_camera_params(cam, scale, pad_left, pad_top, i)
            params["cx"] = actual_cx + pp_offset_x
            params["cy"] = actual_cy + pp_offset_y
            params["_transform"]["pp_offset"] = [pp_offset_x, pp_offset_y]
            cam_params.append(params)

        return {"images": proc_images, "masks": proc_masks, "cameras": cam_params, "frame_idx": frame_idx}

    def process_d6_3(self, frame_idx: int) -> Optional[Dict]:
        """D6-3: PP-correct crop with triangulation."""
        images, masks = self.data_loader.load_frame(frame_idx)
        if all(m.sum() < 100 for m in masks):
            return None

        masks_array = np.array(masks)
        result = self.center_estimator.estimate(masks_array)
        centers_2d = [tuple(c) for c in result.centers_2d]

        # Compute global crop size
        max_size = 0
        for mask in masks:
            coords = np.where(mask > 127)
            if len(coords[0]) > 0:
                h = coords[0].max() - coords[0].min()
                w = coords[1].max() - coords[1].min()
                max_size = max(max_size, h, w)
        
        crop_size = max(max_size / self.config.object_ratio, 256)
        scale = self.config.output_size / crop_size

        proc_images, proc_masks, cam_params = [], [], []

        for i in range(self.num_views):
            crop_center = centers_2d[i]
            
            proc_img = self._crop_and_resize(images[i], crop_center, crop_size)
            proc_images.append(proc_img)

            mask_rgb = np.stack([masks[i]] * 3, axis=-1)
            proc_mask = self._crop_and_resize(mask_rgb, crop_center, crop_size, bg=(0, 0, 0))
            proc_masks.append(proc_mask[:, :, 0])

            # ACCURATE PP (not 256!)
            cam = self.cameras[i]
            crop_x = crop_center[0] - crop_size / 2
            crop_y = crop_center[1] - crop_size / 2
            
            params = self._build_d6_camera_params(cam, scale, 0, 0, i)
            params["cx"] = (cam["cx"] - crop_x) * scale
            params["cy"] = (cam["cy"] - crop_y) * scale
            params["_transform"]["crop_center"] = list(crop_center)
            params["_transform"]["crop_size"] = crop_size
            cam_params.append(params)

        return {"images": proc_images, "masks": proc_masks, "cameras": cam_params, 
                "frame_idx": frame_idx, "center_3d": result.center_3d.tolist() if result.center_3d is not None else None}

    def _build_d6_camera_params(self, cam: Dict, scale: float, pad_left: int, pad_top: int, view_idx: int) -> Dict:
        """Build camera params for D6 methods."""
        cfg = self.config
        
        # Get original values
        fx_orig = cam.get("fx", cam.get("K", np.eye(3))[0, 0])
        fy_orig = cam.get("fy", cam.get("K", np.eye(3))[1, 1])
        cx_orig = cam.get("cx", cam.get("K", np.eye(3))[0, 2])
        cy_orig = cam.get("cy", cam.get("K", np.eye(3))[1, 2])
        
        fx = fx_orig * scale
        fy = fy_orig * scale
        cx = cx_orig * scale + pad_left
        cy = cy_orig * scale + pad_top
        
        # Translation normalization
        w2c_orig = cam.get("w2c", np.eye(4))
        if isinstance(w2c_orig, list):
            w2c_orig = np.array(w2c_orig)
        
        if cfg.normalize_translation:
            c2w = np.linalg.inv(w2c_orig)
            cam_pos = c2w[:3, 3]
            dist_scale = cfg.target_distance / np.linalg.norm(cam_pos)
            c2w[:3, 3] = cam_pos * dist_scale
            w2c = np.linalg.inv(c2w)
        else:
            w2c = w2c_orig.copy()
        
        # fx normalization
        if cfg.normalize_fx:
            fx_norm_scale = cfg.target_fx / fx
            fx = cfg.target_fx
            fy = fy * fx_norm_scale  # Keep aspect ratio
        
        return {
            "file_path": f"images/cam_{view_idx:03d}.png",
            "view_id": view_idx,
            "w": cfg.output_size, "h": cfg.output_size,
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "w2c": w2c.tolist(),
            "_transform": {"method": cfg.version, "scale": scale, "padding": [pad_left, pad_top]}
        }

    def _crop_and_resize(self, image: np.ndarray, crop_center: Tuple[float, float],
                         crop_size: float, bg: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Crop and resize helper for D6-3."""
        h, w = image.shape[:2]
        crop_x = int(crop_center[0] - crop_size / 2)
        crop_y = int(crop_center[1] - crop_size / 2)
        cs = int(crop_size)

        if len(image.shape) == 3:
            output = np.full((cs, cs, 3), bg, dtype=np.uint8)
        else:
            output = np.full((cs, cs), bg[0], dtype=np.uint8)

        src_x1, src_y1 = max(0, crop_x), max(0, crop_y)
        src_x2, src_y2 = min(w, crop_x + cs), min(h, crop_y + cs)
        dst_x1, dst_y1 = max(0, -crop_x), max(0, -crop_y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        output[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

        return cv2.resize(output, (self.config.output_size, self.config.output_size), 
                         interpolation=cv2.INTER_LINEAR)

    # =========================================================================
    # D9 Method (native)
    # =========================================================================
    def process_d9(self, frame_idx: int) -> Optional[Dict]:
        """D9: Original resolution, no transformation."""
        images, masks = self.data_loader.load_frame(frame_idx)
        if all(m.sum() < 100 for m in masks):
            return None

        cam_params = []
        for i in range(self.num_views):
            cam = self.cameras[i]
            
            # Keep original intrinsics
            fx = cam.get("fx", cam.get("K", np.eye(3))[0, 0])
            fy = cam.get("fy", cam.get("K", np.eye(3))[1, 1])
            cx = cam.get("cx", cam.get("K", np.eye(3))[0, 2])
            cy = cam.get("cy", cam.get("K", np.eye(3))[1, 2])
            
            w2c = cam.get("w2c", np.eye(4))
            if isinstance(w2c, list):
                w2c = np.array(w2c)
            
            h, w = images[i].shape[:2]
            
            cam_params.append({
                "file_path": f"images/cam_{i:03d}.png",
                "view_id": i,
                "w": w, "h": h,
                "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
                "w2c": w2c.tolist(),
                "_transform": {"method": "D9", "native": True}
            })

        return {"images": images, "masks": masks, "cameras": cam_params, "frame_idx": frame_idx}

    # =========================================================================
    # Main Processing
    # =========================================================================
    def process_frame_d7d8(self, frame_idx: int, video_caps, mask_caps, transforms):
        """Process frame for D7/D8 paradigms."""
        cfg = self.config
        images, masks = [], []

        for cam_idx in range(self.num_views):
            video_caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            mask_caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video_caps[cam_idx].read()
            ret_mask, mask = mask_caps[cam_idx].read()
            if not ret or not ret_mask:
                return None
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            masks.append(np.where(mask > 127, 255, 0).astype(np.uint8))

        if all(m.sum() < 100 for m in masks):
            return None

        proc_images, proc_masks, cam_params = [], [], []
        
        # Per-sample zoom calculation
        zoom_scope = getattr(cfg, 'zoom_scope', 'global')
        if zoom_scope == 'per_sample' and cfg.adaptive_zoom:
            # ★ BUG FIX (2026-01-25): Apply transform first if zoom_after_transform is True
            from mouse_extensions.preprocessing.preprocess import compute_persample_zoom_coverage
            
            if getattr(cfg, 'zoom_after_transform', False):
                # Compute coverage on TRANSFORMED mask (not original)
                temp_mask = masks[0].copy()
                M = transforms[0]
                if cfg.transform == TransformType.HOMOGRAPHY:
                    temp_mask = cv2.warpPerspective(
                        (temp_mask > 0).astype(np.uint8) * 255, M,
                        (cfg.output_size, cfg.output_size), flags=cv2.INTER_NEAREST)
                else:
                    temp_mask = cv2.warpAffine(
                        (temp_mask > 0).astype(np.uint8) * 255, M[:2],
                        (cfg.output_size, cfg.output_size), flags=cv2.INTER_NEAREST)
                # Compute zoom with optional safe_zoom constraint
                zoom_center_mode = getattr(cfg, 'zoom_center_mode', 'object')
                use_safe_zoom = getattr(cfg, "safe_zoom", False) or getattr(cfg, 'pp_correction', False)
                
                if use_safe_zoom:
                    from mouse_extensions.preprocessing.preprocess import (
                        compute_clipping_safe_zoom, compute_object_centered_safe_zoom
                    )
                    if zoom_center_mode == 'object':
                        # M4: Object-centered safe zoom
                        coverage_zoom = compute_persample_zoom_coverage(
                            temp_mask, cfg.target_fg_coverage, cfg.zoom_range)
                        safe_zoom = compute_object_centered_safe_zoom(temp_mask, cfg.output_size)
                        sample_zoom = min(coverage_zoom, safe_zoom)
                        sample_zoom = float(np.clip(sample_zoom, cfg.zoom_range[0], cfg.zoom_range[1]))
                    else:
                        # M3_3: Center-aligned safe zoom
                        sample_zoom = compute_clipping_safe_zoom(
                            temp_mask, cfg.target_fg_coverage, cfg.zoom_range, cfg.output_size)
                else:
                    sample_zoom = compute_persample_zoom_coverage(
                        temp_mask, cfg.target_fg_coverage, cfg.zoom_range)
            else:
                # Use original mask (no transform)
                target_mask = masks[0]
                # Compute zoom with optional safe_zoom constraint
                zoom_center_mode = getattr(cfg, 'zoom_center_mode', 'object')
                use_safe_zoom = getattr(cfg, "safe_zoom", False) or getattr(cfg, 'pp_correction', False)
                
                if use_safe_zoom:
                    from mouse_extensions.preprocessing.preprocess import (
                        compute_clipping_safe_zoom, compute_object_centered_safe_zoom
                    )
                    if zoom_center_mode == 'object':
                        # M4: Object-centered safe zoom
                        coverage_zoom = compute_persample_zoom_coverage(
                            target_mask, cfg.target_fg_coverage, cfg.zoom_range)
                        safe_zoom = compute_object_centered_safe_zoom(target_mask, cfg.output_size)
                        sample_zoom = min(coverage_zoom, safe_zoom)
                        sample_zoom = float(np.clip(sample_zoom, cfg.zoom_range[0], cfg.zoom_range[1]))
                    else:
                        # M3_3: Center-aligned safe zoom
                        sample_zoom = compute_clipping_safe_zoom(
                            target_mask, cfg.target_fg_coverage, cfg.zoom_range, cfg.output_size)
                else:
                    sample_zoom = compute_persample_zoom_coverage(
                        target_mask, cfg.target_fg_coverage, cfg.zoom_range)
            frame_zoom = sample_zoom
        else:
            frame_zoom = cfg.zoom
        
        for cam_idx in range(self.num_views):
            img, mask = self.apply_transform(images[cam_idx], masks[cam_idx], transforms[cam_idx])
            img, mask, crop_offset = self.apply_zoom(img, mask, frame_zoom)
            proc_images.append(img)
            proc_masks.append(mask)
            params = self.compute_camera_params(self.cameras[cam_idx], frame_zoom, crop_offset)
            params['file_path'] = f"images/cam_{cam_idx:03d}.png"
            params['view_id'] = cam_idx
            cam_params.append(params)

        return {'images': proc_images, 'masks': proc_masks, 'cameras': cam_params, 'frame_idx': frame_idx}

    def save_sample(self, sample: Dict, sample_dir: Path):
        """Save processed sample."""
        sample_dir.mkdir(parents=True, exist_ok=True)
        images_dir = sample_dir / "images"
        images_dir.mkdir(exist_ok=True)

        for i, (img, mask) in enumerate(zip(sample['images'], sample['masks'])):
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3], rgba[:, :, 3] = img, mask
            Image.fromarray(rgba).save(images_dir / f"cam_{i:03d}.png")

        with open(sample_dir / "opencv_cameras.json", 'w') as f:
            json.dump({"frames": sample['cameras'], "_preprocessing": {
                "version": self.config.version, "frame_idx": sample['frame_idx']}}, f, indent=2)

    def generate_dataset_config(self, num_train: int, num_val: int):
        """Generate dataset config YAML file."""
        cfg = self.config
        config_name = cfg.version.replace('.', '_').replace('-', '_')
        config_path = DATASET_CONFIG_DIR / f"{config_name}.yaml"
        
        preset_info = PRESETS.get(cfg.version, {})
        description = preset_info.get('description', f'{cfg.version} preprocessing')
        category = preset_info.get('paradigm', 'unknown')
        
        yaml_content = f"""# Dataset: {config_name}
# {description}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

_name: {config_name}
_description: "{description}"
_category: {category}
_preprocessing:
  paradigm: {cfg.paradigm.value}
  transform: {cfg.transform.value}
  target_fx: {cfg.target_fx if cfg.normalize_fx else 'original'}
  output_size: {cfg.output_size or 'original'}

training:
  dataset:
    dataset_path: {cfg.output_dir}/data_mouse_train.txt

validation:
  dataset_path: {cfg.output_dir}/data_mouse_val.txt

_stats:
  num_train: {num_train}
  num_val: {num_val}
"""
        
        DATASET_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n[Config Generated] {config_path}")
        return config_path

    def run(self):
        """Run preprocessing pipeline."""
        cfg = self.config
        
        print(f"\n{'='*60}")
        print(f"Unified Preprocessor: {cfg.version}")
        print(f"{'='*60}")
        print(f"Paradigm: {cfg.paradigm.value}")
        print(f"Output: {cfg.output_dir}")
        
        # ====== D6/D9: Use DataLoader ======
        if cfg.paradigm in [Paradigm.GEOMETRY_PRESERVING, Paradigm.NATIVE]:
            self._init_data_loader()
            if cfg.paradigm == Paradigm.GEOMETRY_PRESERVING and cfg.d6_method == "pp_correct_crop":
                self._init_center_estimator()
            
            total_frames = len(self.data_loader)
            frame_indices = list(range(0, total_frames, cfg.frame_interval))
            if cfg.max_samples:
                frame_indices = frame_indices[:cfg.max_samples]
            
            print(f"Processing {len(frame_indices)} frames...")
            
            # Select method
            if cfg.paradigm == Paradigm.NATIVE:
                process_fn = self.process_d9
            elif cfg.d6_method == "resize_only":
                process_fn = self.process_d6_1
            elif cfg.d6_method == "virtual_shift":
                process_fn = self.process_d6_2
            else:
                process_fn = self.process_d6_3
            
            # Process and save in single loop (streaming - memory efficient)
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            saved_count = 0
            all_paths = []
            
            # Determine save directory based on single_folder mode
            if cfg.single_folder:
                samples_dir = cfg.output_dir / "samples"
                samples_dir.mkdir(parents=True, exist_ok=True)
            else:
                samples_dir = cfg.output_dir

            for frame_idx in tqdm(frame_indices, desc="Processing"):
                sample = process_fn(frame_idx)
                if sample:
                    sample_dir = samples_dir / f"{saved_count:06d}"
                    self.save_sample(sample, sample_dir)
                    all_paths.append(str(sample_dir) + '/')
                    saved_count += 1

        # ====== D7/D8/D10: Use video captures ======
        elif cfg.paradigm in [Paradigm.PP_CENTERED_SHIFT, Paradigm.PRECISION_HOMOGRAPHY, Paradigm.UP_ALIGNED_ZOOM, Paradigm.OBJECT_CENTERED_MVG]:
            print(f"Loading cameras: {cfg.camera_pkl}")
            self.load_cameras(cfg.camera_pkl)
            
            # D10: Apply up-alignment if enabled
            if cfg.up_alignment:
                print(f"Applying up-alignment (source: {cfg.up_source})")
                if cfg.up_source == "vertical_lines":
                    vertical_lines_path = cfg.input_dir / "vertical_lines.npz"
                    if vertical_lines_path.exists():
                        up = load_up_direction(vertical_lines_path)
                        print(f"  Loaded up direction: {up}")
                    else:
                        print(f"  Warning: {vertical_lines_path} not found, using camera Y-axis")
                        extrinsics = get_extrinsics_from_cameras(self.cameras)
                        up = compute_up_from_cameras(extrinsics)
                else:  # camera_y_mean
                    extrinsics = get_extrinsics_from_cameras(self.cameras)
                    up = compute_up_from_cameras(extrinsics)
                    print(f"  Computed up from cameras: {up}")
                
                self.cameras = apply_up_alignment_to_cameras(self.cameras, up)
                print(f"  Cameras aligned to up direction")
            
            # D10.1/M3: Adaptive zoom if enabled
            if cfg.adaptive_zoom:
                # Load first frame masks to estimate zoom
                mask_dir = cfg.input_dir / "simpleclick_undist"
                mask_caps_temp = [cv2.VideoCapture(str(mask_dir / f"{i}.mp4")) for i in range(self.num_views)]
                first_masks = []
                for cap in mask_caps_temp:
                    ret, frame = cap.read()
                    if ret:
                        first_masks.append(frame[:, :, 0] > 127)
                    cap.release()
                
                # Choose zoom method
                if cfg.zoom_method == "coverage_based":
                    zoom_scope = getattr(cfg, "zoom_scope", "global")
                    if zoom_scope == "per_sample":
                        print(f"  Per-sample adaptive zoom enabled (target: {cfg.target_fg_coverage*100:.1f}%)")
                        cfg.zoom = 1.0  # Will be computed per-frame
                    else:
                        print(f"Computing global coverage-based adaptive zoom (target: {cfg.target_fg_coverage*100:.1f}%)")
                    
                    # M3 fix: Apply transform first, then compute coverage
                    if getattr(cfg, 'zoom_after_transform', False):
                        # Compute transforms (without zoom initially)
                        temp_transforms = [self.compute_transform(cam['K']) for cam in self.cameras]
                        
                        # Apply transform to masks to get post-transform coverage
                        transformed_masks = []
                        for i, (mask, M) in enumerate(zip(first_masks, temp_transforms)):
                            if mask is None:
                                continue
                            # Apply same warp that will be used on images
                            if cfg.transform.value == "homography":
                                warped = cv2.warpPerspective(mask.astype(np.uint8) * 255, M, 
                                                            (cfg.output_size, cfg.output_size))
                            else:
                                warped = cv2.warpAffine(mask.astype(np.uint8) * 255, M[:2], 
                                                       (cfg.output_size, cfg.output_size))
                            transformed_masks.append(warped > 127)
                        
                        # Compute coverage on transformed masks
                        cfg.zoom = compute_adaptive_zoom_coverage(transformed_masks, cfg.target_fg_coverage, cfg.zoom_range)
                        print(f"  (zoom_after_transform: using post-transform coverage)")
                    elif zoom_scope == "global":
                        # Original behavior: use raw mask coverage (global)
                        cfg.zoom = compute_adaptive_zoom_coverage(first_masks, cfg.target_fg_coverage, cfg.zoom_range)
                    
                    if zoom_scope == "global":
                        # Estimate coverage after zoom for all views
                        est_coverages = [compute_fg_coverage_after_zoom(m, cfg.zoom) for m in first_masks if m is not None]
                        mean_est_coverage = np.mean(est_coverages) if est_coverages else 0
                        print(f"  Global adaptive zoom: {cfg.zoom:.2f}x (est. coverage: {mean_est_coverage*100:.1f}%)")
                        
                        # Warn if below minimum
                        if cfg.min_fg_coverage > 0 and mean_est_coverage < cfg.min_fg_coverage:
                            print(f"  WARNING: Estimated coverage {mean_est_coverage*100:.1f}% < min {cfg.min_fg_coverage*100:.1f}%")
                else:
                    print(f"Computing bbox-based adaptive zoom (target fill: {cfg.zoom_fill_ratio})")
                    cfg.zoom = compute_adaptive_zoom(first_masks, cfg.zoom_fill_ratio, cfg.zoom_range)
                    print(f"  Adaptive zoom: {cfg.zoom:.2f}x")

            video_dir = cfg.input_dir / "videos_undist"
            mask_dir = cfg.input_dir / "simpleclick_undist"
            video_caps = [cv2.VideoCapture(str(video_dir / f"{i}.mp4")) for i in range(self.num_views)]
            mask_caps = [cv2.VideoCapture(str(mask_dir / f"{i}.mp4")) for i in range(self.num_views)]

            total_frames = int(video_caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = list(range(0, total_frames, cfg.frame_interval))
            if cfg.max_samples:
                frame_indices = frame_indices[:cfg.max_samples]

            print(f"Transform: {cfg.transform.value}, Scale: {cfg.scale_mode.value}, Zoom: {cfg.zoom}x")

            transforms = [self.compute_transform(cam['K']) for cam in self.cameras]

            # Process and save in single loop (streaming - memory efficient)
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            saved_count = 0
            all_paths = []
            
            # Determine save directory based on single_folder mode
            if cfg.single_folder:
                samples_dir = cfg.output_dir / "samples"
                samples_dir.mkdir(parents=True, exist_ok=True)
            else:
                samples_dir = cfg.output_dir

            for frame_idx in tqdm(frame_indices, desc="Processing"):
                sample = self.process_frame_d7d8(frame_idx, video_caps, mask_caps, transforms)
                if sample:
                    sample_dir = samples_dir / f"{saved_count:06d}"
                    self.save_sample(sample, sample_dir)
                    all_paths.append(str(sample_dir) + '/')
                    saved_count += 1

            for cap in video_caps + mask_caps:
                cap.release()

        # Generate split files
        # Always save all paths first
        with open(cfg.output_dir / "data_mouse_all.txt", 'w') as f:
            f.write('\n'.join(sorted(all_paths)))
        
        if cfg.single_folder:
            # Single folder mode: all samples in samples/, generate default split
            train_paths = []
            val_paths = []
            
            if cfg.val_ratio > 0:
                # Generate default 90/10 split using create_split logic
                np.random.seed(42)
                indices = list(range(len(all_paths)))
                np.random.shuffle(indices)
                num_val = int(len(all_paths) * cfg.val_ratio)
                val_indices = set(indices[:num_val])

                train_paths = [all_paths[i] for i in range(len(all_paths)) if i not in val_indices]
                val_paths = [all_paths[i] for i in range(len(all_paths)) if i in val_indices]
                
                for split, paths in [('train', train_paths), ('val', val_paths)]:
                    if paths:
                        with open(cfg.output_dir / f"data_mouse_{split}.txt", 'w') as f:
                            f.write('\n'.join(sorted(paths)))
        else:
            # Traditional mode: generate train/val splits
            if cfg.val_ratio > 0:
                np.random.seed(42)
                indices = list(range(len(all_paths)))
                np.random.shuffle(indices)
                num_val = int(len(all_paths) * cfg.val_ratio)
                val_indices = set(indices[:num_val])

                train_paths = [all_paths[i] for i in range(len(all_paths)) if i not in val_indices]
                val_paths = [all_paths[i] for i in range(len(all_paths)) if i in val_indices]
            else:
                train_paths = all_paths
                val_paths = []

            for split, paths in [('train', train_paths), ('val', val_paths)]:
                if paths:
                    with open(cfg.output_dir / f"data_mouse_{split}.txt", 'w') as f:
                        f.write('\n'.join(sorted(paths)))

        with open(cfg.output_dir / "metadata.json", 'w') as f:
            json.dump({
                "version": cfg.version,
                "paradigm": cfg.paradigm.value,
                "total_samples": len(all_paths),
                "num_train": len(train_paths),
                "num_val": len(val_paths),
                "val_ratio": cfg.val_ratio,
                "single_folder": cfg.single_folder,
            }, f, indent=2)

        self.generate_dataset_config(len(train_paths), len(val_paths))

        print(f"\n{'='*60}")
        print(f"Complete: {cfg.output_dir}")
        if cfg.single_folder:
            print(f"Mode: single_folder (all samples in samples/)")
        print(f"Total: {len(all_paths)}, Train: {len(train_paths)}, Val: {len(val_paths)}")
        print(f"Split files: data_mouse_all.txt" + (f", data_mouse_train.txt, data_mouse_val.txt" if train_paths else ""))
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Unified Mouse Preprocessor (D6-D9)")
    parser.add_argument('--preset', '-p', choices=list(PRESETS.keys()), help="Preset name")
    parser.add_argument('--list-presets', action='store_true', help="List available presets")
    parser.add_argument('--input-dir', type=str, help="Input directory")
    parser.add_argument('--output-dir', type=str, help="Output directory")
    parser.add_argument('--camera-pkl', default=None, help="Camera pickle path")
    parser.add_argument('--frame-interval', type=int, default=5, help="Frame interval")
    parser.add_argument('--max-samples', type=int, default=None, help="Max samples")
    parser.add_argument('--val-ratio', type=float, default=0.1, help="Validation ratio")
    parser.add_argument('--transform', choices=['affine', 'homography', 'none'])
    parser.add_argument('--scale-mode', choices=['fx_only', 'individual', 'average'])
    parser.add_argument('--zoom', type=float)
    args = parser.parse_args()

    if args.list_presets:
        print("\n=== Available Presets ===")
        recommended = get_recommended()
        for name in list_preset_names():
            info = PRESETS[name]
            rec = " [RECOMMENDED]" if name == recommended else ""
            print(f"  {name:8}{rec}")
            print(f"           {info.get('description', '')}")
        return

    if not args.input_dir or not args.output_dir:
        parser.error("--input-dir and --output-dir are required")

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    camera_pkl = Path(args.camera_pkl) if args.camera_pkl else input_dir / "new_cam.pkl"

    if args.preset:
        config = PreprocessConfig.from_preset(args.preset, input_dir=input_dir, output_dir=output_dir,
            camera_pkl=camera_pkl, frame_interval=args.frame_interval, 
            max_samples=args.max_samples, val_ratio=args.val_ratio)
    else:
        config = PreprocessConfig(input_dir=input_dir, output_dir=output_dir, camera_pkl=camera_pkl,
            frame_interval=args.frame_interval, max_samples=args.max_samples, val_ratio=args.val_ratio)

    if args.transform:
        config.transform = TransformType(args.transform)
    if args.scale_mode:
        config.scale_mode = ScaleMode(args.scale_mode)
    if args.zoom:
        config.zoom = args.zoom

    UnifiedPreprocessor(config).run()


if __name__ == "__main__":
    main()


def compute_safe_zoom(mask: np.ndarray, output_size: int = 512, margin: int = 5) -> float:
    """Compute maximum zoom that won't clip the foreground.
    
    For center-aligned zoom, calculates the maximum zoom factor such that
    the entire foreground remains within the cropped region.
    
    Args:
        mask: Binary mask (H, W)
        output_size: Image size (assumes square)
        margin: Safety margin in pixels
    
    Returns:
        Maximum safe zoom factor
    
    Math:
        - Crop size at zoom z: crop_size = output_size / z
        - Crop region (center-aligned): [(S-C)/2, (S+C)/2]
        - For object at [x_min, x_max], need:
          x_min >= (S-C)/2  and  x_max <= (S+C)/2
        - Solving: z <= S / (2 * max_dist_from_center)
    """
    if mask is None or mask.size == 0:
        return 1.0
    
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return 1.0
    
    # Object bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    
    # Distance from image center to furthest object edge
    center = output_size / 2
    max_dist_x = max(center - x_min, x_max - center) + margin
    max_dist_y = max(center - y_min, y_max - center) + margin
    max_dist = max(max_dist_x, max_dist_y)
    
    if max_dist <= 0:
        return float('inf')
    
    # Safe zoom: crop_half >= max_dist
    # crop_half = output_size / (2 * zoom)
    # => zoom <= output_size / (2 * max_dist)
    safe_zoom = output_size / (2 * max_dist)
    
    return float(safe_zoom)


def compute_clipping_safe_zoom(
    mask: np.ndarray,
    target_coverage: float = 0.05,
    zoom_range: tuple = (1.0, 2.5),
    output_size: int = 512,
) -> float:
    """Compute zoom that achieves target coverage without clipping.
    
    Combines coverage-based zoom with safe zoom constraint.
    
    Args:
        mask: Binary mask
        target_coverage: Target foreground ratio (0.05 = 5%)
        zoom_range: (min_zoom, max_zoom) range
        output_size: Image size
    
    Returns:
        Zoom factor: min(coverage_zoom, safe_zoom), clipped to range
    """
    # Target coverage zoom
    coverage_zoom = compute_persample_zoom_coverage(mask, target_coverage, zoom_range)
    
    # Safe zoom (no clipping)
    safe_zoom = compute_safe_zoom(mask, output_size)
    
    # Take minimum to prevent clipping
    final_zoom = min(coverage_zoom, safe_zoom)
    
    return float(np.clip(final_zoom, zoom_range[0], zoom_range[1]))


def compute_object_centered_safe_zoom(
    mask: np.ndarray,
    output_size: int = 512,
    margin: int = 10,
) -> float:
    """Compute maximum zoom for object-centered crop without clipping.
    
    For object-centered mode, the object is always at the center of the crop.
    Safe zoom = output_size / (object_bbox_max_dim + 2*margin)
    
    Args:
        mask: Binary mask
        output_size: Output image size
        margin: Safety margin in pixels
    
    Returns:
        Maximum safe zoom factor
    """
    if mask is None or mask.size == 0:
        return 1.0
    
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return 1.0
    
    # Object bounding box dimensions
    bbox_w = xs.max() - xs.min()
    bbox_h = ys.max() - ys.min()
    bbox_max = max(bbox_w, bbox_h) + 2 * margin
    
    if bbox_max <= 0:
        return float('inf')
    
    # For object-centered: crop_size must be >= object_size
    # crop_size = output_size / zoom
    # => zoom <= output_size / bbox_max
    safe_zoom = output_size / bbox_max
    
    return float(safe_zoom)
