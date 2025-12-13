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
Mouse Multi-View Diffusion Dataset

Adapted from FixViewDataset for mouse 6-view data.
Key differences from original:
- Images in 'images/' subdirectory
- No predefined "front" view - uses cam_000 as reference (configurable)
- Support for augmentation on limited data
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import os
from PIL import Image
from typing import Dict, List, Optional, Tuple


class MouseMVDiffusionDataset(Dataset):
    """Dataset for mouse multi-view diffusion training."""

    def __init__(self, config, split: str):
        """
        Initialize the mouse MVDiffusion dataset.

        Args:
            config: Configuration object with dataset parameters
            split: 'train' or 'val'
        """
        super().__init__()
        self.config = config
        self.split = split

        # Dataset parameters
        self.img_wh = config.get("img_wh", 512)
        self.n_views = config.get("n_views", 6)

        # Reference view configuration (which view to use as input)
        # For mouse, we might want to experiment with different reference views
        # Supports: int (fixed), "random" (random each sample), or list [0,1,2,...] (random from list)
        ref_view_config = config.get("reference_view_idx", 0)
        if ref_view_config == "random":
            self.reference_view_idx = "random"
            self.reference_view_choices = list(range(self.n_views))
        elif isinstance(ref_view_config, list):
            self.reference_view_idx = "random"
            self.reference_view_choices = ref_view_config
        else:
            self.reference_view_idx = int(ref_view_config)
            self.reference_view_choices = None

        # Load data paths
        if self.split == "train":
            with open(config.train_dataset.path, 'r') as f:
                self.all_data_paths = f.read().strip().split("\n")
            self.bg_color = config.train_dataset.bg_color
            self.use_augmentation = config.train_dataset.get("augmentation", False)
        elif self.split == "val":
            with open(config.validation_dataset.path, 'r') as f:
                self.all_data_paths = f.read().strip().split("\n")
            self.bg_color = config.validation_dataset.bg_color
            self.use_augmentation = False
        else:
            raise NotImplementedError(f"Split '{split}' is not supported")

        # Filter empty paths
        self.all_data_paths = [path for path in self.all_data_paths if len(path.strip()) > 0]
        self.all_data_paths = pd.array(self.all_data_paths, dtype="string")

        # Shuffle validation set
        if self.split == "val":
            random.shuffle(self.all_data_paths)

        # View indices for target (all views including reference)
        self.target_view_indices = list(range(self.n_views))

        # Load pre-computed color prompt embeddings
        # Using the same embeddings as original (direction-based)
        prompt_embed_path = config.get(
            "prompt_embed_path",
            "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt"
        )
        if os.path.exists(prompt_embed_path):
            self.color_prompt_embedding = torch.load(prompt_embed_path)
        else:
            # Generate default embeddings if not found
            print(f"Warning: Prompt embeddings not found at {prompt_embed_path}")
            print("Using zero embeddings - consider generating proper embeddings")
            # Shape: [n_views, seq_len, embed_dim] - typical values
            self.color_prompt_embedding = torch.zeros(self.n_views, 77, 1024)

        # Background color choices
        self._bg_color_choices = {
            'white': np.array([1., 1., 1.], dtype=np.float32),
            'black': np.array([0., 0., 0.], dtype=np.float32),
            'gray': np.array([0.5, 0.5, 0.5], dtype=np.float32),
        }

        # Augmentation parameters (for limited data)
        if self.use_augmentation:
            self.aug_brightness = config.train_dataset.get("aug_brightness", [0.9, 1.1])
            self.aug_contrast = config.train_dataset.get("aug_contrast", [0.9, 1.1])
            self.aug_hflip = config.train_dataset.get("aug_hflip", False)

        print(f"[MouseMVDiffusionDataset] Loaded {len(self.all_data_paths)} samples for {split}")
        if self.reference_view_idx == "random":
            print(f"  Reference view: random from {self.reference_view_choices} (6x data augmentation)")
        else:
            print(f"  Reference view: {self.reference_view_idx}")
        print(f"  Augmentation: {self.use_augmentation}")

    def __len__(self) -> int:
        return len(self.all_data_paths)

    def get_bg_color(self) -> np.ndarray:
        """Generate background color based on configuration."""
        if self.bg_color in self._bg_color_choices:
            return self._bg_color_choices[self.bg_color].copy()
        elif self.bg_color == 'random':
            return np.random.rand(3).astype(np.float32)
        elif self.bg_color == 'three_choices':
            return random.choice(list(self._bg_color_choices.values())).copy()
        elif isinstance(self.bg_color, (int, float)):
            return np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError(f"Background color '{self.bg_color}' is not supported")

    def load_image(self, image_path: str, bg_color: np.ndarray) -> torch.Tensor:
        """
        Load and process an image with background compositing.

        Args:
            image_path: Path to the image (RGBA or RGB)
            bg_color: Background color as RGB array [0-1]

        Returns:
            Processed image as torch tensor [H, W, 3]
        """
        image = Image.open(image_path)

        # Handle different image modes
        if image.mode == 'RGBA':
            rgba = image
        elif image.mode == 'RGB':
            # No alpha channel - treat as fully opaque
            rgba = image.convert('RGBA')
        else:
            rgba = image.convert('RGBA')

        # Resize if needed
        if rgba.size != (self.img_wh, self.img_wh):
            rgba = rgba.resize((self.img_wh, self.img_wh), Image.LANCZOS)

        # Convert to numpy and normalize
        rgba = np.array(rgba, dtype=np.float32) / 255.0
        image_rgb = rgba[..., :3]
        alpha = rgba[..., 3:4]

        # Composite with background
        image_rgb = image_rgb * alpha + bg_color[None, None, :] * (1 - alpha)

        return torch.from_numpy(image_rgb)

    def apply_augmentation(
        self,
        images: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Apply consistent augmentation to all views.

        Args:
            images: List of image tensors [H, W, 3]

        Returns:
            Augmented images
        """
        if not self.use_augmentation:
            return images

        # Random brightness
        if hasattr(self, 'aug_brightness'):
            factor = random.uniform(*self.aug_brightness)
            images = [torch.clamp(img * factor, 0, 1) for img in images]

        # Random contrast
        if hasattr(self, 'aug_contrast'):
            factor = random.uniform(*self.aug_contrast)
            mean = torch.stack(images).mean()
            images = [torch.clamp((img - mean) * factor + mean, 0, 1) for img in images]

        # Note: Horizontal flip would require flipping camera poses too
        # So we skip it for MVDiffusion training

        return images

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Args:
            index: Sample index

        Returns:
            Dictionary containing:
                - imgs_in: Input images (reference view replicated) [n_views, 3, H, W]
                - imgs_out: Target images (all views) [n_views, 3, H, W]
                - color_prompt_embeddings: Prompt embeddings for each view
        """
        data_path = self.all_data_paths[index].strip()
        bg_color = self.get_bg_color()

        # Mouse data structure: {data_path}/images/cam_XXX.png
        images_dir = os.path.join(data_path, "images")

        # Fallback to direct path if images/ doesn't exist
        if not os.path.exists(images_dir):
            images_dir = data_path

        # Determine reference view index (supports random selection for data augmentation)
        if self.reference_view_idx == "random":
            current_ref_idx = random.choice(self.reference_view_choices)
        else:
            current_ref_idx = self.reference_view_idx

        # Load reference (input) image
        ref_image_path = os.path.join(images_dir, f"cam_{current_ref_idx:03d}.png")
        ref_image = self.load_image(ref_image_path, bg_color)

        # Load all target images
        target_images = []
        for view_idx in self.target_view_indices:
            target_image_path = os.path.join(images_dir, f"cam_{view_idx:03d}.png")
            target_image = self.load_image(target_image_path, bg_color)
            target_images.append(target_image)

        # Apply augmentation (consistent across views)
        if self.use_augmentation:
            all_images = [ref_image] + target_images
            all_images = self.apply_augmentation(all_images)
            ref_image = all_images[0]
            target_images = all_images[1:]

        # Convert to [C, H, W] format
        ref_image = ref_image.permute(2, 0, 1)  # [3, H, W]
        target_images = torch.stack([img.permute(2, 0, 1) for img in target_images], dim=0)  # [n_views, 3, H, W]

        # Replicate reference image for all views (as conditioning)
        input_images = ref_image.unsqueeze(0).repeat(self.n_views, 1, 1, 1)  # [n_views, 3, H, W]

        return {
            'imgs_in': input_images.float(),
            'imgs_out': target_images.float(),
            'color_prompt_embeddings': self.color_prompt_embedding,
        }


class MouseMVDiffusionInferenceDataset(Dataset):
    """Dataset for mouse MVDiffusion inference (single images)."""

    def __init__(
        self,
        image_paths: List[str],
        img_wh: int = 512,
        bg_color: str = 'white'
    ):
        """
        Initialize inference dataset.

        Args:
            image_paths: List of input image paths
            img_wh: Image size
            bg_color: Background color
        """
        self.image_paths = image_paths
        self.img_wh = img_wh
        self.bg_color = bg_color

        self._bg_color_choices = {
            'white': np.array([1., 1., 1.], dtype=np.float32),
            'black': np.array([0., 0., 0.], dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Load and preprocess a single image for inference."""
        image_path = self.image_paths[index]

        # Get background color
        bg = self._bg_color_choices.get(self.bg_color,
                                         np.array([1., 1., 1.], dtype=np.float32))

        # Load image
        image = Image.open(image_path)
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        if image.size != (self.img_wh, self.img_wh):
            image = image.resize((self.img_wh, self.img_wh), Image.LANCZOS)

        # Process
        rgba = np.array(image, dtype=np.float32) / 255.0
        rgb = rgba[..., :3]
        alpha = rgba[..., 3:4]
        rgb = rgb * alpha + bg[None, None, :] * (1 - alpha)

        # To tensor [C, H, W]
        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()

        return {
            'image': image_tensor,
            'path': image_path,
        }
