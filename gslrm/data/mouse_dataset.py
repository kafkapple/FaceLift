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


def pil_to_np(pil_image):
    """Convert PIL image to numpy array, preserving RGBA alpha channel."""
    if pil_image.mode == "RGBA":
        r, g, b, a = pil_image.split()
        r, g, b, a = np.asarray(r), np.asarray(g), np.asarray(b), np.asarray(a)
        image = np.stack([r, g, b, a], axis=2)
    else:
        image = np.asarray(pil_image)
    return image


def get_bg_color(bg_color_config):
    """Generate background color based on configuration."""
    COLORS = {
        'white': np.array([1.0, 1.0, 1.0], dtype=np.float32),
        'black': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        'gray': np.array([0.5, 0.5, 0.5], dtype=np.float32)
    }

    if isinstance(bg_color_config, str):
        if bg_color_config in COLORS:
            bg_color = COLORS[bg_color_config]
        elif bg_color_config == 'random':
            bg_color = np.random.rand(3).astype(np.float32)
        elif bg_color_config == 'three_choices':
            bg_color = random.choice(list(COLORS.values()))
        else:
            raise ValueError(f"Unsupported background color: '{bg_color_config}'")
    elif isinstance(bg_color_config, (int, float)):
        if not 0 <= bg_color_config <= 1:
            raise ValueError(f"Background color must be in [0, 1], got {bg_color_config}")
        bg_color = np.array([bg_color_config] * 3, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported background color type: {type(bg_color_config)}")

    return torch.from_numpy(bg_color)


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
            dataset_path = self.config.validation.dataset_path
        else:
            raise NotImplementedError(f"Split '{split}' is not supported")

        # Load dataset paths from local file
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

        # Mouse-specific settings
        mouse_config = self.config.get("mouse", {})
        aug_config = mouse_config.get("augmentation", {})
        self.use_augmentation = aug_config.get("enabled", False) and split == "train"
        self.horizontal_flip = aug_config.get("horizontal_flip", False)
        self.rotation_range = aug_config.get("rotation_range", 0)
        self.brightness_range = aug_config.get("brightness_range", [1.0, 1.0])
        self.contrast_range = aug_config.get("contrast_range", [1.0, 1.0])

        print(f"[MouseViewDataset] Split: {split}, Samples: {len(self.all_data_paths)}")
        print(f"[MouseViewDataset] Views: {self.num_views}, Input views: {self.num_input_views}")
        print(f"[MouseViewDataset] Augmentation: {self.use_augmentation}")

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

        if self.split == "train":
            # Random input view for training diversity
            input_indices = random.sample(all_indices, self.num_input_views)
        else:
            # Fixed input view for reproducible validation
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
                intrinsics = np.array([
                    camera["fx"], camera["fy"], camera["cx"], camera["cy"]
                ])
                intrinsics *= resize_ratio

                # Extract camera pose (w2c -> c2w)
                c2w = np.linalg.inv(np.array(camera["w2c"]))

                # Convert image to tensor
                image_tensor = pil_to_np(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1)

                # Collect processed data
                input_images.append(image_tensor)
                input_fxfycxcy.append(intrinsics)
                input_c2ws.append(c2w)

            # Stack all data
            input_images = torch.stack(input_images, dim=0)
            input_fxfycxcy = np.array(input_fxfycxcy)
            input_c2ws = np.array(input_c2ws)

        except Exception as e:
            traceback.print_exc()
            print(f"Error loading data from {self.all_data_paths[idx]}: {str(e)}")
            # Fallback to random sample
            return self.__getitem__(random.randint(0, len(self) - 1))

        input_c2ws = torch.from_numpy(input_c2ws).float()
        input_fxfycxcy = torch.from_numpy(input_fxfycxcy).float()

        image_indices = torch.from_numpy(
            np.array(image_choices)
        ).long().unsqueeze(-1)
        scene_indices = torch.tensor(idx).long().unsqueeze(0).expand_as(image_indices)
        indices = torch.cat([image_indices, scene_indices], dim=-1)

        return {
            "image": input_images,
            "c2w": input_c2ws,
            "fxfycxcy": input_fxfycxcy,
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

        # Use first camera's parameters as reference
        camera = self.reference_cameras[0]
        resize_ratio = self.target_size / camera["w"]

        intrinsics = np.array([
            camera["fx"] * resize_ratio,
            camera["fy"] * resize_ratio,
            camera["cx"] * resize_ratio,
            camera["cy"] * resize_ratio
        ])

        c2w = np.linalg.inv(np.array(camera["w2c"]))

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
