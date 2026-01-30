"""Main preprocessing class for mouse inference.

Analogous to crop_face() in human face preprocessing.
Transforms arbitrary input images to M5 training format.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from mouse_extensions.inference.preprocessing.config import (
    MousePreprocessConfig,
    DEFAULT_CONFIG,
)
from mouse_extensions.inference.preprocessing.mouse_detector import (
    MouseDetector,
    DetectionResult,
)
from mouse_extensions.inference.preprocessing.coverage_normalizer import (
    CoverageNormalizer,
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """Result of preprocessing pipeline."""
    
    image: np.ndarray  # Preprocessed image (H, W, 3), RGB, uint8
    mask: Optional[np.ndarray]  # Binary mask (H, W), or None if fallback
    scale_applied: float  # Scale factor applied
    centroid: Tuple[float, float]  # Final centroid position
    detection_used: bool  # True if SAM detection was used
    original_size: Tuple[int, int]  # Original image (H, W)
    

class MouseInferencePreprocessor:
    """Preprocess arbitrary images for mouse 3D reconstruction.
    
    Analogous to crop_face() in human face preprocessing:
    - Human: MTCNN (face detection) -> rembg (background) -> crop_face (align)
    - Mouse: SAM (segmentation) -> white background -> center + coverage align
    
    M5 Training Statistics:
        - Resolution: 512×512
        - fx, fy: 549
        - cx, cy: 256 (principal point = image center)
        - Background: White (255, 255, 255)
    
    Example:
        preprocessor = MouseInferencePreprocessor(
            sam_checkpoint="checkpoints/sam/sam_vit_h.pth"
        )
        result = preprocessor.preprocess("photo.jpg")
        # result.image: 512×512 RGB, white background, mouse centered
    """
    
    def __init__(
        self,
        sam_checkpoint: Optional[str] = None,
        config: Optional[MousePreprocessConfig] = None,
        device: str = "cuda",
    ):
        """Initialize preprocessor.
        
        Args:
            sam_checkpoint: Path to SAM checkpoint. If None, uses fallback mode.
            config: Preprocessing configuration. Uses defaults if None.
            device: Torch device for SAM.
        """
        self.config = config or DEFAULT_CONFIG
        self.device = device
        
        # Initialize detector
        self.detector = MouseDetector(
            checkpoint=sam_checkpoint,
            model_type=self.config.sam_model_type,
            device=device,
            points_per_side=self.config.sam_points_per_side,
            pred_iou_thresh=self.config.sam_pred_iou_thresh,
            stability_score_thresh=self.config.sam_stability_score_thresh,
            min_area_ratio=self.config.min_area_ratio,
            max_area_ratio=self.config.max_area_ratio,
        )
        
        # Initialize normalizer
        self.normalizer = CoverageNormalizer(
            target_coverage=self.config.target_coverage,
            target_resolution=self.config.target_resolution,
            min_scale=self.config.min_scale,
            max_scale=self.config.max_scale,
            bg_color=self.config.bg_color,
        )
        
    def is_already_preprocessed(self, image: np.ndarray) -> bool:
        """Check if image appears to already be in M5 format.
        
        Heuristics:
        1. Resolution is 512×512 (±tolerance)
        2. Corners are white (background)
        3. Has centered content (not empty, not full)
        
        Args:
            image: Input image (H, W, 3), RGB, uint8.
            
        Returns:
            True if image appears preprocessed.
        """
        h, w = image.shape[:2]
        res = self.config.target_resolution
        tol = self.config.auto_detect_resolution_tolerance
        
        # Check resolution
        if not (abs(h - res) <= tol and abs(w - res) <= tol):
            return False
            
        # Check if corners are white (M5 has white background)
        corner_size = 10
        corners = [
            image[:corner_size, :corner_size],
            image[:corner_size, -corner_size:],
            image[-corner_size:, :corner_size],
            image[-corner_size:, -corner_size:],
        ]
        
        # At least 3 corners should be mostly white
        white_corners = 0
        for corner in corners:
            mean_val = corner.mean()
            if mean_val > 250:  # Near white
                white_corners += 1
                
        if white_corners < 3:
            return False
            
        # Check that center has some content (not empty white image)
        center_region = image[h//4:3*h//4, w//4:3*w//4]
        center_mean = center_region.mean()
        
        # Center should not be pure white (has some content)
        if center_mean > 252:  # Almost all white -> empty
            return False
            
        logger.info(
            f"Image appears already preprocessed: {h}×{w}, "
            f"white_corners={white_corners}, center_mean={center_mean:.1f}"
        )
        return True
        
    def preprocess(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        force: bool = False,
    ) -> PreprocessResult:
        """Preprocess image for mouse 3D reconstruction.
        
        Pipeline:
        1. Load image
        2. (Optional) Check if already preprocessed
        3. SAM detection -> mask + centroid
        4. Background removal (white composite)
        5. Center alignment + coverage normalization
        
        Args:
            image: Input image (path, array, or PIL Image).
            force: If True, always preprocess even if image appears ready.
            
        Returns:
            PreprocessResult with preprocessed image and metadata.
        """
        # Step 1: Load image
        img_array = self._load_image(image)
        original_size = (img_array.shape[0], img_array.shape[1])
        
        # Step 2: Check if already preprocessed
        if not force and self.is_already_preprocessed(img_array):
            # Already in correct format, just ensure exact resolution
            if img_array.shape[:2] != (self.config.target_resolution,) * 2:
                img_array = cv2.resize(
                    img_array,
                    (self.config.target_resolution, self.config.target_resolution),
                    interpolation=cv2.INTER_LINEAR,
                )
            return PreprocessResult(
                image=img_array,
                mask=None,
                scale_applied=1.0,
                centroid=(self.config.target_cx, self.config.target_cy),
                detection_used=False,
                original_size=original_size,
            )
            
        # Step 3: Detect mouse with SAM
        detection = self.detector.detect(img_array)
        
        if detection is None:
            # Fallback mode
            return self._fallback_preprocess(img_array, original_size)
            
        # Step 4: Background removal
        img_bg_removed = self._remove_background(img_array, detection.mask)
        
        # Step 5: Normalize coverage and center
        norm_image, norm_mask, new_centroid, scale = self.normalizer.normalize(
            img_bg_removed,
            detection.mask,
            detection.centroid,
        )
        
        return PreprocessResult(
            image=norm_image,
            mask=norm_mask,
            scale_applied=scale,
            centroid=new_centroid,
            detection_used=True,
            original_size=original_size,
        )
        
    def _load_image(
        self, image: Union[str, Path, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """Load image to numpy array.
        
        Args:
            image: Input image (path, array, or PIL Image).
            
        Returns:
            RGB uint8 numpy array (H, W, 3).
        """
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
            return np.array(img)
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                # Grayscale
                return np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                # RGBA -> RGB (composite with white)
                rgba = image.astype(np.float32) / 255.0
                alpha = rgba[:, :, 3:4]
                rgb = rgba[:, :, :3]
                bg = np.ones_like(rgb)  # White
                composite = rgb * alpha + bg * (1 - alpha)
                return (composite * 255).clip(0, 255).astype(np.uint8)
            else:
                return image.astype(np.uint8)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
            
    def _remove_background(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Remove background and composite with white.
        
        Args:
            image: RGB image (H, W, 3).
            mask: Binary mask (H, W), values 0 or 255.
            
        Returns:
            Image with white background (H, W, 3).
        """
        mask_float = (mask > 127).astype(np.float32)[:, :, np.newaxis]
        bg = np.full_like(image, self.config.bg_color, dtype=np.uint8)
        
        result = (
            image.astype(np.float32) * mask_float
            + bg.astype(np.float32) * (1 - mask_float)
        )
        return result.clip(0, 255).astype(np.uint8)
        
    def _fallback_preprocess(
        self, image: np.ndarray, original_size: Tuple[int, int]
    ) -> PreprocessResult:
        """Fallback preprocessing when SAM is unavailable.
        
        Simply resizes to target resolution.
        
        Args:
            image: Input image.
            original_size: Original image size (H, W).
            
        Returns:
            PreprocessResult with resized image.
        """
        if self.config.fallback_mode == "error":
            raise RuntimeError(
                "SAM detection failed and fallback_mode='error'. "
                "Provide SAM checkpoint or set fallback_mode='resize'."
            )
            
        logger.warning(
            "SAM detection unavailable/failed. Using simple resize fallback."
        )
        
        res = self.config.target_resolution
        resized = cv2.resize(image, (res, res), interpolation=cv2.INTER_LINEAR)
        
        return PreprocessResult(
            image=resized,
            mask=None,
            scale_applied=1.0,
            centroid=(res / 2, res / 2),
            detection_used=False,
            original_size=original_size,
        )
        
    def visualize_steps(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, np.ndarray]:
        """Run preprocessing with step-by-step visualization.
        
        Useful for debugging and verifying preprocessing quality.
        
        Args:
            image: Input image.
            output_dir: If provided, save visualizations to this directory.
            
        Returns:
            Dict of step name -> image array.
        """
        steps = {}
        
        # Step 1: Original
        img_array = self._load_image(image)
        steps["01_original"] = img_array.copy()
        
        # Step 2: Detection
        detection = self.detector.detect(img_array)
        if detection is not None:
            vis_detection = img_array.copy()
            # Draw mask overlay
            mask_rgb = np.zeros_like(vis_detection)
            mask_rgb[:, :, 1] = detection.mask  # Green overlay
            vis_detection = cv2.addWeighted(vis_detection, 0.7, mask_rgb, 0.3, 0)
            # Draw centroid
            cx, cy = int(detection.centroid[0]), int(detection.centroid[1])
            cv2.circle(vis_detection, (cx, cy), 10, (255, 0, 0), -1)
            # Draw bbox
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(vis_detection, (x1, y1), (x2, y2), (0, 255, 0), 2)
            steps["02_detection"] = vis_detection
            
            # Step 3: Background removed
            bg_removed = self._remove_background(img_array, detection.mask)
            steps["03_bg_removed"] = bg_removed
            
            # Step 4: Normalized
            norm_image, norm_mask, new_centroid, scale = self.normalizer.normalize(
                bg_removed, detection.mask, detection.centroid
            )
            steps["04_normalized"] = norm_image
            
            # Draw center crosshair
            vis_final = norm_image.copy()
            center = self.config.target_resolution // 2
            cv2.line(vis_final, (center, 0), (center, center * 2), (0, 0, 255), 1)
            cv2.line(vis_final, (0, center), (center * 2, center), (0, 0, 255), 1)
            cx, cy = int(new_centroid[0]), int(new_centroid[1])
            cv2.circle(vis_final, (cx, cy), 5, (255, 0, 0), -1)
            steps["05_final_with_guides"] = vis_final
        else:
            steps["02_detection_failed"] = img_array.copy()
            
        # Save if output_dir provided
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            for name, img in steps.items():
                cv2.imwrite(
                    str(out_path / f"{name}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )
            logger.info(f"Saved {len(steps)} visualization steps to {out_path}")
            
        return steps
