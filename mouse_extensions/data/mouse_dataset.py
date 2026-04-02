# no-split: single Dataset class — __getitem__ + camera normalization + augmentation tightly coupled
# Copyright 2025 Adobe Inc.
# Modified for Mouse-FaceLift project
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

"""
Mouse Dataset for FaceLift GS-LRM Training

This dataset handles multi-view mouse images with:
- 6 synchronized camera views
- Single input view for reconstruction
- Optional data augmentation for limited real data
"""

import json
import random
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
import os
from torch.utils.data import Dataset

# Import preprocessing utilities from mouse_extensions (required)
from mouse_extensions.data.preprocessing import (
    pil_to_np,
    normalize_camera_distance,
    normalize_camera_distance_with_intrinsics,
    normalize_cameras_to_y_up,
    normalize_cameras_to_z_up,
    get_bg_color,
    preprocess_cameras,
    PreprocessingConfig,
)



# PP correction for v12/v13 dataset bug (2026-01-17)
try:
    from mouse_extensions.scripts.solutions.pp_correction_integration import (
        apply_pp_correction, get_actual_principal_point
    )
    PP_CORRECTION_AVAILABLE = True
except ImportError:
    PP_CORRECTION_AVAILABLE = False
    print("[Warning] PP correction module not found. Using original behavior.")


class MouseViewDataset(Dataset):
    """
    Dataset for loading multi-view mouse images.

    Key differences from RandomViewDataset:
    - Fixed 6 views (no random sampling beyond num_views)
    - Single input view for inference (configurable)
    - Optional augmentation for limited real data
    - No face-specific preprocessing

    Args:
        config: Configuration object containing dataset parameters
        split: Dataset split ('train' or 'val')
    """

    def __init__(self, config, split: str):
        super().__init__()
        self.config = config
        self.split = split

        # Load dataset paths based on split
        if self.split == "train":
            dataset_path = self.config.training.dataset.dataset_path
        elif self.split == "val":
            dataset_path = self.config.get("validation", {}).get("dataset_path", "")
        elif self.split == "test":
            dataset_path = self.config.get("test", {}).get("dataset_path", "")
        else:
            raise NotImplementedError(f"Split '{split}' is not supported")

        # Load dataset paths from local file
        dataset_path = os.path.expanduser(dataset_path)
        with open(dataset_path, 'r') as f:
            self.all_data_paths = f.read().strip().split("\n")

        # Filter empty paths
        self.all_data_paths = pd.array(
            [s for s in self.all_data_paths if len(s) > 0], dtype="string"
        )

        # Extract dataset configuration
        dataset_config = self.config.training.dataset
        self.bg_color = dataset_config.get("background_color", "white")
        self.remove_alpha = dataset_config.get("remove_alpha", False)
        self.num_views = dataset_config.get("num_views", 6)
        self.num_input_views = dataset_config.get("num_input_views", 1)
        self.target_has_input = dataset_config.get("target_has_input", True)
        self.random_view_selection = dataset_config.get("random_view_selection", False)

        # Camera exclusion/inclusion for ablation experiments
        # Use either exclude_camera_indices OR include_camera_indices (not both)
        self.exclude_camera_indices = dataset_config.get("exclude_camera_indices", [])
        self.include_camera_indices = dataset_config.get("include_camera_indices", None)
        if self.exclude_camera_indices and self.include_camera_indices:
            raise ValueError("Cannot specify both exclude_camera_indices and include_camera_indices")

        # Mouse-specific settings
        mouse_config = self.config.get("mouse", {})
        aug_config = mouse_config.get("augmentation", {})
        self.use_augmentation = aug_config.get("enabled", False) and split == "train"
        self.horizontal_flip = aug_config.get("horizontal_flip", False)
        self.rotation_range = aug_config.get("rotation_range", 0)
        self.brightness_range = aug_config.get("brightness_range", [1.0, 1.0])
        self.contrast_range = aug_config.get("contrast_range", [1.0, 1.0])

        # Camera normalization settings
        self.normalize_cameras = mouse_config.get("normalize_cameras", True)
        self.target_camera_distance = mouse_config.get("target_camera_distance", 2.7)
        # Z-up vs Y-up: Human data uses Z-up, so default to Z-up for compatibility
        self.normalize_to_z_up = mouse_config.get("normalize_to_z_up", True)

        # Camera recentering: shift camera centroid to origin
        # Required for species where raw cameras are NOT origin-centered (e.g., s-DANNCE rat)
        # Mouse data is already origin-centered from preprocessing, so default=False
        self.recenter_cameras = mouse_config.get("recenter_cameras", False)

        # Auto mask generation: Create alpha channel from white background
        # This is critical for mouse images that don't have alpha channel
        # Without mask, L2 loss is dominated by background pixels (95%)
        self.auto_generate_mask = mouse_config.get("auto_generate_mask", True)
        self.mask_threshold = mouse_config.get("mask_threshold", 250)

        # Principal Point correction (2026-01-17)
        # Fixes cx,cy bug in v12/v13 datasets that causes ghosting artifacts
        # Options: "none" (original), "actual_pp" (varying cx,cy), "crop" (recommended)
        self.pp_correction_method = mouse_config.get("pp_correction", "none")
        if self.pp_correction_method != "none" and not PP_CORRECTION_AVAILABLE:
            print(f"[Warning] PP correction '{self.pp_correction_method}' requested but module not available")
            self.pp_correction_method = "none"  # Pixels > threshold are background

        print(f"[MouseViewDataset] Split: {split}, Samples: {len(self.all_data_paths)}")
        print(f"[MouseViewDataset] Views: {self.num_views}, Input views: {self.num_input_views}")
        if self.exclude_camera_indices:
            print(f"[MouseViewDataset] Excluding cameras: {self.exclude_camera_indices}")
        if self.include_camera_indices:
            print(f"[MouseViewDataset] Including only cameras: {self.include_camera_indices}")
        print(f"[MouseViewDataset] Augmentation: {self.use_augmentation}")
        up_mode = "Z-up" if self.normalize_to_z_up else "Y-up"
        print(f"[MouseViewDataset] Camera normalization: {up_mode}={self.normalize_cameras}, distance={self.target_camera_distance}")
        print(f"[MouseViewDataset] Auto mask generation: {self.auto_generate_mask}, threshold={self.mask_threshold}")
        if self.pp_correction_method != "none":
            print(f"[MouseViewDataset] PP correction: {self.pp_correction_method} (fixing cx,cy bug)")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.all_data_paths)

    def _apply_augmentation(self, image: Image.Image) -> Image.Image:
        """
        Apply data augmentation to an image.

        Args:
            image: PIL Image to augment

        Returns:
            Augmented PIL Image
        """
        if not self.use_augmentation:
            return image

        # Random horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Random rotation (small)
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = image.rotate(angle, resample=Image.BILINEAR, expand=False)

        # Random brightness/contrast (applied to RGB only)
        if image.mode == "RGBA":
            r, g, b, a = image.split()
            rgb = Image.merge("RGB", (r, g, b))
        else:
            rgb = image
            a = None

        # Brightness
        brightness = random.uniform(*self.brightness_range)
        if brightness != 1.0:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(rgb)
            rgb = enhancer.enhance(brightness)

        # Contrast
        contrast = random.uniform(*self.contrast_range)
        if contrast != 1.0:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(rgb)
            rgb = enhancer.enhance(contrast)

        if a is not None:
            # Restore background pixels to white after augmentation
            # Contrast/brightness can turn white background to gray
            import numpy as np
            rgb_np = np.array(rgb, dtype=np.float32)
            a_np = np.array(a, dtype=np.float32) / 255.0

            # Where alpha is 0 (background), set RGB to white (255)
            bg_mask = a_np < 0.5
            rgb_np[bg_mask] = 255.0

            rgb = Image.fromarray(rgb_np.astype(np.uint8))
            r, g, b = rgb.split()
            image = Image.merge("RGBA", (r, g, b, a))
        else:
            image = rgb

        return image

    def _process_image_channels(self, image: Image.Image, bg_color_255: Tuple[int, int, int]) -> Image.Image:
        """
        Process image channels, handling RGBA and other formats.

        Args:
            image: PIL Image to process
            bg_color_255: Background color as RGB tuple (0-255 range)

        Returns:
            PIL Image: Processed image in RGB or RGBA format
        """
        if image.mode == "RGBA":
            # Composite RGBA image onto background color
            background = Image.new("RGB", image.size, bg_color_255)
            alpha_mask = image.split()[3]
            background.paste(image, mask=alpha_mask)

            if self.remove_alpha:
                return background
            else:
                background.putalpha(alpha_mask)
                return background
        elif image.mode != "RGB":
            return image.convert("RGB")
        else:
            return image

    def _select_views(self, total_views: int) -> Tuple[List[int], List[int]]:
        """
        Select input and target views for training/inference.

        For mouse data:
        - Training: Random input view, all views as targets
        - Validation: Fixed first view as input

        Args:
            total_views: Total available views

        Returns:
            Tuple of (input_indices, target_indices)
        """
        all_indices = list(range(min(total_views, self.num_views)))

        # Apply camera exclusion/inclusion filtering
        if self.include_camera_indices is not None:
            all_indices = [i for i in all_indices if i in self.include_camera_indices]
        elif self.exclude_camera_indices:
            all_indices = [i for i in all_indices if i not in self.exclude_camera_indices]

        # Fixed view ordering for both training and validation
        # This ensures consistent camera-to-index mapping
        # Randomness comes from different samples, not view shuffling
        if getattr(self, 'random_view_selection', False) and getattr(self, 'split', 'train') == "train":
            input_indices = sorted(random.sample(all_indices, self.num_input_views))
        else:
            input_indices = list(range(self.num_input_views))

        if self.target_has_input:
            target_indices = all_indices
        else:
            target_indices = [i for i in all_indices if i not in input_indices]

        return input_indices, target_indices

    def __getitem__(self, idx):
        """
        Load and preprocess a multi-view mouse sample.

        Args:
            idx: Index of the sample to load

        Returns:
            dict: Contains 'image', 'c2w', 'fxfycxcy', 'index', 'bg_color'
        """
        try:
            data_json_path = os.path.join(
                self.all_data_paths[idx].strip(), "opencv_cameras.json"
            )
            data_path = os.path.dirname(data_json_path)

            # Load camera data
            with open(data_json_path, 'r') as f:
                data_json = json.load(f)

            cameras = data_json["frames"]
            total_views = len(cameras)

            bg_color = get_bg_color(self.bg_color)
            bg_color_255 = (
                int(bg_color[0] * 255),
                int(bg_color[1] * 255),
                int(bg_color[2] * 255)
            )

            # Select views
            input_indices, target_indices = self._select_views(total_views)

            # Combine: input views first, then remaining targets
            if self.target_has_input:
                image_choices = input_indices + [
                    i for i in target_indices if i not in input_indices
                ]
            else:
                image_choices = input_indices + target_indices

            # Ensure we don't exceed available views
            image_choices = image_choices[:self.num_views]

            # Extract selected camera data
            selected_cameras = [cameras[i] for i in image_choices]
            selected_image_paths = [
                os.path.join(data_path, cameras[i]["file_path"])
                for i in image_choices
            ]

            # Initialize data collection
            input_images = []
            input_fxfycxcy = []
            input_c2ws = []

            # Extract ALL camera poses for turntable trajectory (not just selected)
            all_c2ws_raw = []
            all_fxfycxcy_raw = []
            target_size = self.config.model.image_tokenizer.image_size
            resize_ratio_all = target_size / int(cameras[0].get("w", target_size))
            for cam in cameras:
                intr = np.array([cam["fx"], cam["fy"], cam["cx"], cam["cy"]])
                intr *= resize_ratio_all
                all_fxfycxcy_raw.append(intr)
                w2c = np.array(cam["w2c"])
                R, t = w2c[:3, :3], w2c[:3, 3]
                c2w = np.eye(4)
                c2w[:3, :3] = R.T
                c2w[:3, 3] = -R.T @ t
                all_c2ws_raw.append(c2w)
            all_c2ws_raw = np.array(all_c2ws_raw)
            all_fxfycxcy_raw = np.array(all_fxfycxcy_raw)

            for idx_chosen, (camera, image_path) in enumerate(
                zip(selected_cameras, selected_image_paths)
            ):
                # Load image
                image = Image.open(image_path)

                if image.size[0] != image.size[1]:
                    print(f"Warning: Image {image_path} is not square: {image.size}")

                # Resize image if needed
                target_size = self.config.model.image_tokenizer.image_size
                resize_ratio = target_size / image.size[0]
                if image.size[0] != target_size:
                    image = image.resize(
                        (target_size, target_size), resample=Image.LANCZOS
                    )

                # Apply augmentation (training only, same aug for all views in sample)
                if idx_chosen == 0 and self.use_augmentation:
                    # Store augmentation params for consistent application
                    self._current_flip = self.horizontal_flip and random.random() > 0.5
                    self._current_rotation = random.uniform(
                        -self.rotation_range, self.rotation_range
                    ) if self.rotation_range > 0 else 0
                    self._current_brightness = random.uniform(*self.brightness_range)
                    self._current_contrast = random.uniform(*self.contrast_range)

                if self.use_augmentation:
                    image = self._apply_consistent_augmentation(image)

                # Process image channels
                image = self._process_image_channels(image, bg_color_255)

                # Extract and adjust camera intrinsics
                # PP correction handles the cx,cy bug in v12/v13 datasets
                if PP_CORRECTION_AVAILABLE and self.pp_correction_method != "none":
                    image, intrinsics = apply_pp_correction(
                        image,
                        camera,
                        method=self.pp_correction_method,
                        resize_ratio=resize_ratio,
                        target_size=target_size,
                        bg_color=bg_color_255
                    )
                else:
                    intrinsics = np.array([
                        camera["fx"], camera["fy"], camera["cx"], camera["cy"]
                    ])
                    intrinsics *= resize_ratio

                # Extract camera pose (w2c -> c2w)
                # Use explicit R.T and -R.T @ t instead of np.linalg.inv() to avoid
                # numerical precision issues in the last row [0,0,0,1]
                w2c = np.array(camera["w2c"])
                R, t = w2c[:3, :3], w2c[:3, 3]
                c2w = np.eye(4)
                c2w[:3, :3] = R.T
                c2w[:3, 3] = -R.T @ t

                # Convert image to tensor
                image_np = pil_to_np(image).astype(np.float32) / 255.0

                # Auto-generate mask from white background if needed
                # This is critical for mouse images without alpha channel
                if self.auto_generate_mask and image_np.shape[2] == 3:
                    # Threshold-based mask: pixels with all RGB > threshold are background
                    threshold = self.mask_threshold / 255.0
                    is_background = np.all(image_np > threshold, axis=2)
                    alpha = (~is_background).astype(np.float32)
                    # Add alpha channel to image
                    image_np = np.concatenate([image_np, alpha[:, :, None]], axis=2)

                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

                # Collect processed data
                input_images.append(image_tensor)
                input_fxfycxcy.append(intrinsics)
                input_c2ws.append(c2w)

            # Stack all data
            input_images = torch.stack(input_images, dim=0)
            input_fxfycxcy = np.array(input_fxfycxcy)
            input_c2ws = np.array(input_c2ws)

            # Recenter cameras: shift centroid to origin
            # Required for datasets where cameras are NOT origin-centered (e.g., s-DANNCE rat)
            # Mouse data is already centered from preprocessing, so this is a no-op for mouse
            if self.recenter_cameras:
                cam_positions = input_c2ws[:, :3, 3]  # [N, 3]
                centroid = cam_positions.mean(axis=0)  # [3]
                input_c2ws[:, :3, 3] -= centroid
                all_c2ws_raw[:, :3, 3] -= centroid
                all_fxfycxcy_raw = all_fxfycxcy_raw  # intrinsics unchanged by translation

            # Normalize cameras to Z-up or Y-up coordinate system
            # IMPORTANT: Analysis shows GS-LRM pretrained model uses Z-up (not Y-up!)
            # Human data: Up Vector = [0, 0, 1], Orbit plane = XY plane
            if self.normalize_cameras:
                # Try to load up_direction from data directory
                up_direction = None
                vertical_lines_path = os.path.join(data_path, "..", "vertical_lines.npz")
                if os.path.exists(vertical_lines_path):
                    try:
                        vl_data = np.load(vertical_lines_path)
                        up_direction = vl_data.get("up_direction", None)
                        if up_direction is not None:
                            up_direction = np.array(up_direction)
                    except Exception:
                        pass

                # Use Z-up (default) or Y-up based on config
                if self.normalize_to_z_up:
                    input_c2ws = normalize_cameras_to_z_up(input_c2ws, up_direction)
                else:
                    input_c2ws = normalize_cameras_to_y_up(input_c2ws, up_direction)

            # Normalize camera distances to fixed radius
            # FaceLift pretrained model expects cameras at distance ~2.7
            if self.target_camera_distance > 0:
                # Use the new function that also adjusts fx/fy proportionally
                # This fixes the "ghost mouse" issue caused by distance/intrinsics mismatch
                input_c2ws, input_fxfycxcy = normalize_camera_distance_with_intrinsics(
                    input_c2ws, input_fxfycxcy, self.target_camera_distance
                )

        except Exception as e:
            traceback.print_exc()
            print(f"Error loading data from {self.all_data_paths[idx]}: {str(e)}")
            # Fallback to random sample
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Apply same normalization to ALL cameras (for turntable trajectory)
        if self.normalize_cameras:
            if self.normalize_to_z_up:
                all_c2ws_raw = normalize_cameras_to_z_up(all_c2ws_raw, up_direction)
            else:
                all_c2ws_raw = normalize_cameras_to_y_up(all_c2ws_raw, up_direction)
        if self.target_camera_distance > 0:
            all_c2ws_raw, all_fxfycxcy_raw = normalize_camera_distance_with_intrinsics(
                all_c2ws_raw, all_fxfycxcy_raw, self.target_camera_distance
            )

        input_c2ws = torch.from_numpy(input_c2ws).float()
        input_fxfycxcy = torch.from_numpy(input_fxfycxcy).float()
        all_c2ws_tensor = torch.from_numpy(all_c2ws_raw).float()
        all_fxfycxcy_tensor = torch.from_numpy(all_fxfycxcy_raw).float()

        image_indices = torch.from_numpy(
            np.array(image_choices)
        ).long().unsqueeze(-1)
        scene_indices = torch.tensor(idx).long().unsqueeze(0).expand_as(image_indices)
        indices = torch.cat([image_indices, scene_indices], dim=-1)

        return {
            "image": input_images,
            "c2w": input_c2ws,
            "fxfycxcy": input_fxfycxcy,
            "all_c2w": all_c2ws_tensor,
            "all_fxfycxcy": all_fxfycxcy_tensor,
            "index": indices,
            "bg_color": bg_color,
        }

    def _apply_consistent_augmentation(self, image: Image.Image) -> Image.Image:
        """
        Apply consistent augmentation across all views in a sample.
        Uses cached augmentation parameters from first view.

        Args:
            image: PIL Image to augment

        Returns:
            Augmented PIL Image
        """
        # Horizontal flip (mirror for all views)
        if getattr(self, '_current_flip', False):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Rotation
        rotation = getattr(self, '_current_rotation', 0)
        if rotation != 0:
            image = image.rotate(rotation, resample=Image.BILINEAR, expand=False)

        # Brightness/Contrast on RGB channels
        if image.mode == "RGBA":
            r, g, b, a = image.split()
            rgb = Image.merge("RGB", (r, g, b))
        else:
            rgb = image
            a = None

        brightness = getattr(self, '_current_brightness', 1.0)
        if brightness != 1.0:
            from PIL import ImageEnhance
            rgb = ImageEnhance.Brightness(rgb).enhance(brightness)

        contrast = getattr(self, '_current_contrast', 1.0)
        if contrast != 1.0:
            from PIL import ImageEnhance
            rgb = ImageEnhance.Contrast(rgb).enhance(contrast)

        if a is not None:
            r, g, b = rgb.split()
            image = Image.merge("RGBA", (r, g, b, a))
        else:
            image = rgb

        return image


class MouseSingleViewDataset(Dataset):
    """
    Dataset for single-view inference on mouse images.

    Takes a single image and returns it with the appropriate camera parameters
    for input to the GS-LRM model.
    """

    def __init__(self, image_paths: List[str], camera_json_path: str, config):
        """
        Initialize single-view dataset for inference.

        Args:
            image_paths: List of image paths to process
            camera_json_path: Path to reference opencv_cameras.json for intrinsics
            config: Model configuration
        """
        super().__init__()
        self.image_paths = image_paths
        self.config = config

        # Load reference camera parameters
        with open(camera_json_path, 'r') as f:
            data = json.load(f)
        self.reference_cameras = data["frames"]

        self.bg_color = "white"
        self.target_size = config.model.image_tokenizer.image_size
        
        # Compute average intrinsics for robustness (handles D8.1 where cx/cy varies)
        self._compute_average_intrinsics()
    
    def _compute_average_intrinsics(self):
        """Compute average intrinsics across all views for robust inference.
        
        This is important for D8.1 where cx/cy varies per view due to zoom crop offset.
        Using average ensures more consistent behavior regardless of which view the
        input image most closely matches.
        """
        cameras = self.reference_cameras
        
        # Extract parameters from all cameras
        fxs = [c["fx"] for c in cameras]
        fys = [c["fy"] for c in cameras]
        cxs = [c["cx"] for c in cameras]
        cys = [c["cy"] for c in cameras]
        
        # Check if intrinsics vary significantly (D8.1 case)
        cx_std = np.std(cxs)
        cy_std = np.std(cys)
        
        if cx_std > 5 or cy_std > 5:  # Threshold for significant variation
            # Use average intrinsics
            self._avg_fx = np.mean(fxs)
            self._avg_fy = np.mean(fys)
            self._avg_cx = np.mean(cxs)
            self._avg_cy = np.mean(cys)
            self._use_average = True
            print(f"[MouseSingleViewDataset] Using average intrinsics (cx_std={cx_std:.1f}, cy_std={cy_std:.1f})")
            print(f"  Average: fx={self._avg_fx:.2f}, fy={self._avg_fy:.2f}, cx={self._avg_cx:.2f}, cy={self._avg_cy:.2f}")
        else:
            # Use first camera (all cameras have similar intrinsics)
            self._use_average = False

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load a single image for inference.

        Returns dict with image and camera params matching first view.
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        # Process image
        if image.mode != "RGBA":
            # Add alpha channel (assume no background removal needed)
            image = image.convert("RGBA")

        # Resize
        if image.size[0] != self.target_size:
            image = image.resize(
                (self.target_size, self.target_size),
                resample=Image.LANCZOS
            )

        # Use average intrinsics if they vary (D8.1), otherwise first camera
        camera = self.reference_cameras[0]
        resize_ratio = self.target_size / camera["w"]

        if self._use_average:
            # Use pre-computed average intrinsics (for D8.1 with varying cx/cy)
            intrinsics = np.array([
                self._avg_fx * resize_ratio,
                self._avg_fy * resize_ratio,
                self._avg_cx * resize_ratio,
                self._avg_cy * resize_ratio
            ])
        else:
            # Use first camera (standard case where all views have same intrinsics)
            intrinsics = np.array([
                camera["fx"] * resize_ratio,
                camera["fy"] * resize_ratio,
                camera["cx"] * resize_ratio,
                camera["cy"] * resize_ratio
            ])

        # Use explicit R.T and -R.T @ t instead of np.linalg.inv()
        w2c = np.array(camera["w2c"])
        R, t = w2c[:3, :3], w2c[:3, 3]
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t

        # Normalize camera to Z-up (matches human data / GS-LRM pretrained)
        c2w_array = np.array([c2w])
        c2w_normalized = normalize_cameras_to_z_up(c2w_array, up_direction=None)
        c2w = c2w_normalized[0]

        # Convert to tensors
        image_np = pil_to_np(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        return {
            "image": image_tensor.unsqueeze(0),  # [1, C, H, W]
            "c2w": torch.from_numpy(c2w).float().unsqueeze(0),  # [1, 4, 4]
            "fxfycxcy": torch.from_numpy(intrinsics).float().unsqueeze(0),  # [1, 4]
            "bg_color": get_bg_color(self.bg_color),
            "image_path": image_path
        }


