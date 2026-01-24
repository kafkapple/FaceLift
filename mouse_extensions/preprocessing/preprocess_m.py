"""
M-series Preprocessing for Mouse 3D Reconstruction

Unified preprocessing pipeline supporting:
- M1: Geometric Baseline (D7_1 equivalent)
- M2: Precision Homography (D8 equivalent)  
- M3: Object-Centered Zoom (NEW - optimal for mouse)

Usage:
    python -m mouse_extensions.preprocessing.preprocess_m \
        --preset M3 \
        --input-dir /path/to/raw \
        --output-dir /path/to/M3

Author: Claude Code
Created: 2026-01-24
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

import cv2
import numpy as np
import yaml

# Local imports
# from .center_estimation import CenterEstimator  # Not needed - using internal implementation
# from .camera_normalizer import CameraNormalizer  # Not needed - using internal implementation

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class CenteringConfig:
    method: str = "triangulation"
    fallback: str = "bbox_average"
    tolerance_px: float = 15.0


@dataclass
class ZoomConfig:
    enabled: bool = True
    target_fg_coverage: float = 0.05
    min_zoom: float = 1.0
    max_zoom: float = 2.5
    coverage_estimation: str = "mask_area"


@dataclass
class CropConfig:
    method: str = "center_crop"
    output_size: int = 512
    padding_mode: str = "edge"


@dataclass
class NormalizationConfig:
    enabled: bool = True
    target_fx: float = 549.0
    target_fy: float = 549.0
    target_cx: float = 256.0
    target_cy: float = 256.0
    target_distance: float = 2.7


@dataclass
class TransformConfig:
    type: str = "affine"
    scale_mode: str = "individual"
    pp_method: str = "shift_to_256"
    interpolation: str = "bilinear"


@dataclass
class ValidationConfig:
    check_ray_error: bool = True
    max_ray_error_deg: float = 0.5
    check_fg_coverage: bool = True
    min_fg_coverage: float = 0.03
    check_center_offset: bool = True
    max_center_offset_px: float = 20.0


@dataclass
class PresetConfig:
    preset: str = "M3"
    description: str = ""
    paradigm: str = "object_centered_zoom"
    base_preset: Optional[str] = None
    centering: CenteringConfig = field(default_factory=CenteringConfig)
    zoom: ZoomConfig = field(default_factory=ZoomConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PresetConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PresetConfig":
        """Create config from dictionary."""
        return cls(
            preset=data.get("preset", "M3"),
            description=data.get("description", ""),
            paradigm=data.get("paradigm", "object_centered_zoom"),
            base_preset=data.get("base_preset"),
            centering=CenteringConfig(**data.get("centering", {})),
            zoom=ZoomConfig(**data.get("zoom", {})),
            crop=CropConfig(**data.get("crop", {})),
            normalization=NormalizationConfig(**data.get("normalization", {})),
            transform=TransformConfig(**data.get("transform", {})),
            validation=ValidationConfig(**data.get("validation", {})),
        )


# =============================================================================
# Built-in Presets
# =============================================================================

PRESETS = {
    "M1": PresetConfig(
        preset="M1",
        description="Geometric Baseline (D7_1 equivalent)",
        paradigm="geometric_baseline",
        zoom=ZoomConfig(enabled=False),
        centering=CenteringConfig(method="none"),
    ),
    "M2": PresetConfig(
        preset="M2",
        description="Precision Homography (D8 equivalent)",
        paradigm="precision_homography",
        zoom=ZoomConfig(enabled=False),
        centering=CenteringConfig(method="none"),
        transform=TransformConfig(type="homography"),
    ),
    "M3": PresetConfig(
        preset="M3",
        description="Object-Centered Zoom - optimal for mouse",
        paradigm="object_centered_zoom",
        base_preset="M1",
        centering=CenteringConfig(method="triangulation", tolerance_px=15.0),
        zoom=ZoomConfig(enabled=True, target_fg_coverage=0.05, min_zoom=1.0, max_zoom=2.5),
    ),
}


# =============================================================================
# Core Processing Classes
# =============================================================================

@dataclass
class CameraParams:
    """Camera intrinsics and extrinsics."""
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray  # 3x3 rotation matrix
    t: np.ndarray  # 3x1 translation vector
    width: int = 512
    height: int = 512

    @property
    def K(self) -> np.ndarray:
        """Intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

    @property
    def translation_norm(self) -> float:
        """Camera distance from origin."""
        return float(np.linalg.norm(self.t))

    def project(self, point_3d: np.ndarray) -> np.ndarray:
        """Project 3D point to 2D image coordinates."""
        p_cam = self.R @ point_3d.reshape(3, 1) + self.t.reshape(3, 1)
        p_2d = self.K @ p_cam
        return (p_2d[:2] / p_2d[2]).flatten()

    def copy(self) -> "CameraParams":
        """Create a deep copy."""
        return CameraParams(
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            R=self.R.copy(), t=self.t.copy(),
            width=self.width, height=self.height
        )


@dataclass
class FrameData:
    """Data for a single frame across all views."""
    frame_idx: int
    images: List[np.ndarray]  # [n_views, H, W, 3]
    masks: List[np.ndarray]   # [n_views, H, W]
    cameras: List[CameraParams]
    center_3d: Optional[np.ndarray] = None
    zoom_factor: float = 1.0
    fg_coverage: float = 0.0
    center_offset_px: float = 0.0


class ObjectCenterEstimator:
    """Estimate 3D object center from multi-view masks."""

    def __init__(self, config: CenteringConfig):
        self.config = config

    def estimate(
        self,
        masks: List[np.ndarray],
        cameras: List[CameraParams]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Estimate 3D object center using DLT triangulation.
        
        Returns:
            center_3d: (3,) array of 3D center
            info: Dictionary with diagnostic information
        """
        if self.config.method == "none":
            return np.zeros(3), {"method": "none"}

        # Get 2D centroids from masks
        centroids_2d = []
        for mask in masks:
            if mask.sum() == 0:
                centroids_2d.append(None)
                continue
            ys, xs = np.where(mask > 0)
            cx, cy = xs.mean(), ys.mean()
            centroids_2d.append(np.array([cx, cy]))

        # Filter valid views
        valid_indices = [i for i, c in enumerate(centroids_2d) if c is not None]
        if len(valid_indices) < 2:
            logger.warning("Not enough valid views for triangulation, using fallback")
            return self._fallback(masks, cameras)

        # DLT triangulation
        if self.config.method == "triangulation":
            center_3d = self._triangulate_dlt(
                [centroids_2d[i] for i in valid_indices],
                [cameras[i] for i in valid_indices]
            )
        else:
            center_3d = self._fallback(masks, cameras)[0]

        # Compute reprojection error
        reproj_errors = []
        for i in valid_indices:
            proj_2d = cameras[i].project(center_3d)
            error = np.linalg.norm(proj_2d - centroids_2d[i])
            reproj_errors.append(error)

        info = {
            "method": self.config.method,
            "valid_views": len(valid_indices),
            "mean_reproj_error": np.mean(reproj_errors),
            "max_reproj_error": np.max(reproj_errors),
        }

        return center_3d, info

    def _triangulate_dlt(
        self,
        points_2d: List[np.ndarray],
        cameras: List[CameraParams]
    ) -> np.ndarray:
        """Direct Linear Transform triangulation."""
        n_views = len(points_2d)
        A = np.zeros((2 * n_views, 4))

        for i, (pt, cam) in enumerate(zip(points_2d, cameras)):
            P = cam.K @ np.hstack([cam.R, cam.t.reshape(3, 1)])
            x, y = pt
            A[2*i] = x * P[2] - P[0]
            A[2*i + 1] = y * P[2] - P[1]

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        return (X[:3] / X[3]).astype(np.float64)

    def _fallback(
        self,
        masks: List[np.ndarray],
        cameras: List[CameraParams]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fallback to bbox average when triangulation fails."""
        # Simple: assume object is at world origin
        return np.zeros(3), {"method": "fallback", "valid_views": 0}


class AdaptiveZoomCalculator:
    """Calculate zoom factor based on foreground coverage."""

    def __init__(self, config: ZoomConfig):
        self.config = config

    def calculate(
        self,
        masks: List[np.ndarray],
        current_coverage: Optional[float] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate optimal zoom factor to achieve target FG coverage.
        
        Returns:
            zoom_factor: Zoom multiplier
            info: Dictionary with diagnostic information
        """
        if not self.config.enabled:
            return 1.0, {"enabled": False}

        # Estimate current coverage
        if current_coverage is None:
            total_pixels = sum(m.size for m in masks)
            fg_pixels = sum((m > 0).sum() for m in masks)
            current_coverage = fg_pixels / total_pixels if total_pixels > 0 else 0

        if current_coverage <= 0:
            return 1.0, {"current_coverage": 0, "zoom": 1.0}

        # Calculate zoom to achieve target coverage
        # coverage_new = coverage_old * zoom^2
        # zoom = sqrt(target / current)
        target = self.config.target_fg_coverage
        zoom = np.sqrt(target / current_coverage)
        zoom = np.clip(zoom, self.config.min_zoom, self.config.max_zoom)

        expected_coverage = current_coverage * (zoom ** 2)

        info = {
            "current_coverage": current_coverage,
            "target_coverage": target,
            "zoom": zoom,
            "expected_coverage": expected_coverage,
        }

        return float(zoom), info


class ImageTransformer:
    """Apply geometric transforms to images and cameras."""

    def __init__(self, config: TransformConfig, crop_config: CropConfig):
        self.config = config
        self.crop_config = crop_config

    def transform_view(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        camera: CameraParams,
        center_2d: np.ndarray,
        zoom: float
    ) -> Tuple[np.ndarray, np.ndarray, CameraParams]:
        """
        Apply object-centered crop with zoom.
        
        Args:
            image: Input image [H, W, 3]
            mask: Input mask [H, W]
            camera: Camera parameters
            center_2d: 2D center in image coordinates
            zoom: Zoom factor
            
        Returns:
            image_out: Transformed image
            mask_out: Transformed mask
            camera_out: Updated camera parameters
        """
        H, W = image.shape[:2]
        out_size = self.crop_config.output_size

        # Calculate crop region centered on object
        # With zoom, we crop a smaller region and scale up
        crop_size = int(out_size / zoom)
        half_crop = crop_size // 2

        cx, cy = center_2d
        x1 = int(cx - half_crop)
        y1 = int(cy - half_crop)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # Handle boundary conditions
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - W)
        pad_bottom = max(0, y2 - H)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        # Crop
        img_crop = image[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        # Pad if needed
        if any([pad_left, pad_top, pad_right, pad_bottom]):
            img_crop = cv2.copyMakeBorder(
                img_crop, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REPLICATE
            )
            mask_crop = cv2.copyMakeBorder(
                mask_crop, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )

        # Resize to output size
        interp = cv2.INTER_LINEAR if self.config.interpolation == "bilinear" else cv2.INTER_NEAREST
        img_out = cv2.resize(img_crop, (out_size, out_size), interpolation=interp)
        mask_out = cv2.resize(mask_crop, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

        # Update camera parameters
        camera_out = camera.copy()

        # Adjust for crop offset (accounting for padding)
        actual_x1 = x1 - pad_left
        actual_y1 = y1 - pad_top
        camera_out.cx = camera.cx - actual_x1
        camera_out.cy = camera.cy - actual_y1

        # Adjust for zoom (resize)
        scale = out_size / crop_size
        camera_out.fx = camera_out.fx * scale
        camera_out.fy = camera_out.fy * scale
        camera_out.cx = camera_out.cx * scale
        camera_out.cy = camera_out.cy * scale

        camera_out.width = out_size
        camera_out.height = out_size

        return img_out, mask_out, camera_out


class CameraPostNormalizer:
    """Normalize camera parameters to match pretrained model expectations."""

    def __init__(self, config: NormalizationConfig):
        self.config = config

    def normalize(self, camera: CameraParams) -> Tuple[CameraParams, Dict[str, Any]]:
        """
        Normalize camera to target focal length and distance.
        
        The key insight: we shift principal point to center (256, 256)
        This is geometrically valid because we're just defining a new
        image coordinate system.
        """
        if not self.config.enabled:
            return camera, {"enabled": False}

        camera_out = camera.copy()
        
        # Scale based on focal length
        fx_scale = self.config.target_fx / camera.fx
        
        # Apply focal length scaling
        camera_out.fx = self.config.target_fx
        camera_out.fy = camera.fy * fx_scale
        
        # Shift principal point to center
        # This is the PP-shift method from M1
        camera_out.cx = self.config.target_cx
        camera_out.cy = self.config.target_cy
        
        # Scale translation to match expected distance
        current_dist = camera.translation_norm
        if current_dist > 0:
            dist_scale = self.config.target_distance / current_dist
            camera_out.t = camera.t * dist_scale
        
        info = {
            "fx_scale": fx_scale,
            "pp_shift": (self.config.target_cx - camera.cx, self.config.target_cy - camera.cy),
            "dist_scale": self.config.target_distance / current_dist if current_dist > 0 else 1.0,
        }
        
        return camera_out, info


class QualityValidator:
    """Validate preprocessing quality."""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def validate(
        self,
        cameras: List[CameraParams],
        masks: List[np.ndarray],
        center_offsets: List[float]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Run all validation checks."""
        results = {}
        passed = True

        # Check FG coverage
        if self.config.check_fg_coverage:
            total_pixels = sum(m.size for m in masks)
            fg_pixels = sum((m > 0).sum() for m in masks)
            coverage = fg_pixels / total_pixels if total_pixels > 0 else 0
            results["fg_coverage"] = coverage
            if coverage < self.config.min_fg_coverage:
                passed = False
                results["fg_coverage_pass"] = False
            else:
                results["fg_coverage_pass"] = True

        # Check center offset
        if self.config.check_center_offset and center_offsets:
            mean_offset = np.mean(center_offsets)
            max_offset = np.max(center_offsets)
            results["mean_center_offset"] = mean_offset
            results["max_center_offset"] = max_offset
            if max_offset > self.config.max_center_offset_px:
                passed = False
                results["center_offset_pass"] = False
            else:
                results["center_offset_pass"] = True

        # Check ray error (simplified - just verify PP is at center)
        if self.config.check_ray_error:
            pp_errors = []
            for cam in cameras:
                pp_error = np.sqrt((cam.cx - 256)**2 + (cam.cy - 256)**2)
                pp_errors.append(pp_error)
            results["mean_pp_error"] = np.mean(pp_errors)
            # Ray error ≈ arctan(pp_error / fx)
            ray_errors = [np.degrees(np.arctan(e / cam.fx)) for e, cam in zip(pp_errors, cameras)]
            results["mean_ray_error_deg"] = np.mean(ray_errors)
            if np.mean(ray_errors) > self.config.max_ray_error_deg:
                passed = False
                results["ray_error_pass"] = False
            else:
                results["ray_error_pass"] = True

        results["overall_pass"] = passed
        return passed, results


# =============================================================================
# Main Preprocessor
# =============================================================================

class MSeriesPreprocessor:
    """
    M-series preprocessing pipeline for mouse 3D reconstruction.
    
    Pipeline steps:
    1. Load raw data
    2. Estimate 3D object center (M3 only)
    3. Calculate adaptive zoom (M3 only)
    4. Apply object-centered crop with zoom
    5. Normalize camera parameters
    6. Validate quality
    7. Save results
    """

    def __init__(self, config: PresetConfig):
        self.config = config
        
        # Initialize components
        self.center_estimator = ObjectCenterEstimator(config.centering)
        self.zoom_calculator = AdaptiveZoomCalculator(config.zoom)
        self.transformer = ImageTransformer(config.transform, config.crop)
        self.normalizer = CameraPostNormalizer(config.normalization)
        self.validator = QualityValidator(config.validation)

        logger.info(f"Initialized MSeriesPreprocessor with preset: {config.preset}")
        logger.info(f"  Paradigm: {config.paradigm}")
        logger.info(f"  Centering: {config.centering.method}")
        logger.info(f"  Zoom: {'enabled' if config.zoom.enabled else 'disabled'}")

    def process_frame(self, frame: FrameData) -> FrameData:
        """Process a single frame across all views."""
        n_views = len(frame.images)

        # Step 1: Estimate 3D center
        center_3d, center_info = self.center_estimator.estimate(
            frame.masks, frame.cameras
        )
        frame.center_3d = center_3d
        logger.debug(f"Frame {frame.frame_idx}: center estimation - {center_info}")

        # Step 2: Calculate zoom
        zoom, zoom_info = self.zoom_calculator.calculate(frame.masks)
        frame.zoom_factor = zoom
        frame.fg_coverage = zoom_info.get("current_coverage", 0)
        logger.debug(f"Frame {frame.frame_idx}: zoom - {zoom_info}")

        # Step 3-4: Transform each view
        new_images = []
        new_masks = []
        new_cameras = []
        center_offsets = []

        for i in range(n_views):
            # Project 3D center to 2D
            if center_3d is not None and not np.allclose(center_3d, 0):
                center_2d = frame.cameras[i].project(center_3d)
            else:
                # Fallback to image center
                H, W = frame.images[i].shape[:2]
                center_2d = np.array([W/2, H/2])

            # Calculate center offset before transform
            img_center = np.array([frame.images[i].shape[1]/2, frame.images[i].shape[0]/2])
            offset = np.linalg.norm(center_2d - img_center)
            center_offsets.append(offset)

            # Transform
            img_out, mask_out, cam_out = self.transformer.transform_view(
                frame.images[i], frame.masks[i], frame.cameras[i],
                center_2d, zoom
            )

            # Normalize camera
            cam_final, norm_info = self.normalizer.normalize(cam_out)

            new_images.append(img_out)
            new_masks.append(mask_out)
            new_cameras.append(cam_final)

        frame.images = new_images
        frame.masks = new_masks
        frame.cameras = new_cameras
        frame.center_offset_px = np.mean(center_offsets)

        return frame

    def validate_frame(self, frame: FrameData) -> Tuple[bool, Dict[str, Any]]:
        """Validate a processed frame."""
        center_offsets = [frame.center_offset_px] * len(frame.cameras)
        return self.validator.validate(frame.cameras, frame.masks, center_offsets)


# =============================================================================
# Data I/O Functions
# =============================================================================

def load_raw_data(input_dir: Path, frame_indices: Optional[List[int]] = None) -> List[FrameData]:
    """Load raw data from input directory."""
    # Implementation depends on data format
    # This is a placeholder - actual implementation would read from:
    # - Videos or image sequences
    # - Camera calibration files
    # - Mask videos or images
    raise NotImplementedError("Implement based on your data format")


def save_processed_data(frames: List[FrameData], output_dir: Path, config: PresetConfig):
    """Save processed data to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        frame_dir = output_dir / f"{frame.frame_idx:06d}"
        frame_dir.mkdir(exist_ok=True)

        # Save images
        for i, img in enumerate(frame.images):
            img_path = frame_dir / f"view_{i:02d}.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Save masks
        for i, mask in enumerate(frame.masks):
            mask_path = frame_dir / f"mask_{i:02d}.png"
            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

        # Save camera parameters
        cameras_data = []
        for cam in frame.cameras:
            cameras_data.append({
                "fx": float(cam.fx),
                "fy": float(cam.fy),
                "cx": float(cam.cx),
                "cy": float(cam.cy),
                "R": cam.R.tolist(),
                "t": cam.t.tolist(),
                "width": cam.width,
                "height": cam.height,
            })

        with open(frame_dir / "cameras.json", "w") as f:
            json.dump(cameras_data, f, indent=2)

        # Save metadata
        metadata = {
            "preset": config.preset,
            "frame_idx": frame.frame_idx,
            "zoom_factor": frame.zoom_factor,
            "fg_coverage": frame.fg_coverage,
            "center_offset_px": frame.center_offset_px,
            "center_3d": frame.center_3d.tolist() if frame.center_3d is not None else None,
        }
        with open(frame_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="M-series Preprocessing for Mouse 3D Reconstruction"
    )
    parser.add_argument(
        "--preset", type=str, default="M3",
        choices=["M1", "M2", "M3"],
        help="Preprocessing preset (default: M3)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to custom config YAML (overrides preset)"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Input directory with raw data"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Starting frame index"
    )
    parser.add_argument(
        "--end-frame", type=int, default=None,
        help="Ending frame index (exclusive)"
    )
    parser.add_argument(
        "--frame-step", type=int, default=1,
        help="Frame step"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Load configuration
    if args.config:
        config = PresetConfig.from_yaml(args.config)
        logger.info(f"Loaded custom config from {args.config}")
    else:
        config = PRESETS[args.preset]
        logger.info(f"Using built-in preset: {args.preset}")

    # Initialize preprocessor
    preprocessor = MSeriesPreprocessor(config)

    # Load data
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")

    # Note: load_raw_data needs to be implemented based on your data format
    # This is a template showing the intended usage
    print("\n" + "="*60)
    print("M-series Preprocessor Template")
    print("="*60)
    print(f"Preset: {config.preset}")
    print(f"Paradigm: {config.paradigm}")
    print(f"\nTo use this preprocessor, implement load_raw_data() for your data format.")
    print("\nKey features:")
    print("  - Object-centered cropping via 3D triangulation")
    print("  - Adaptive zoom for target FG coverage")
    print("  - Camera normalization for pretrained compatibility")
    print("  - Quality validation checks")
    print("="*60)


if __name__ == "__main__":
    main()
