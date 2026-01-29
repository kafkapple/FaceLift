"""GS-LRM module: 6-view images -> 3D Gaussians."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from easydict import EasyDict as edict

from mouse_extensions.inference.checkpoint_utils import find_checkpoint


class GSLRMModule:
    """GS-LRM inference module."""

    def __init__(
        self,
        checkpoint: str,
        config: str,
        device: str = "cuda",
    ):
        self.device = device
        self.checkpoint_path = find_checkpoint(checkpoint)
        self.config_path = config
        self.model = None
        self.config = None

    def load(self):
        """Lazy load the model."""
        if self.model is not None:
            return

        from omegaconf import OmegaConf
        from gslrm.model.gslrm import GSLRM

        self.config = OmegaConf.load(self.config_path)
        checkpoint = torch.load(
            self.checkpoint_path, 
            map_location=self.device, 
            weights_only=False
        )

        self.model = GSLRM(self.config)
        state_dict = checkpoint.get(
            "model", 
            checkpoint.get("model_state_dict", checkpoint)
        )
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device).eval()

        step = checkpoint.get("fwdbwd_pass_step", checkpoint.get("step", "?"))
        print(f"GS-LRM loaded (step {step})")

    def forward(self, sample: edict) -> edict:
        """Run GS-LRM inference.
        
        Args:
            sample: Dictionary with keys:
                - image: [1, V, C, H, W] tensor
                - c2w: [1, V, 4, 4] tensor
                - fxfycxcy: [1, V, 4] tensor
                - index: [1, V, 2] tensor (view_idx, scene_idx)
                
        Returns:
            Output dictionary with gaussians
        """
        self.load()

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = self.model(sample, create_visual=False, split_data=False)

        return output

    def process_images(
        self,
        images: list[Union[Image.Image, np.ndarray]],
        c2ws: np.ndarray,
        fxfycxcy: np.ndarray,
    ) -> edict:
        """Process list of images with camera params.
        
        Args:
            images: List of 6 images (PIL or numpy)
            c2ws: [6, 4, 4] camera-to-world matrices
            fxfycxcy: [6, 4] intrinsics (fx, fy, cx, cy)
            
        Returns:
            Output with gaussians
        """
        self.load()

        # Convert images to tensor
        img_list = []
        for img in images:
            if isinstance(img, Image.Image):
                img = np.array(img)
            img = img.astype(np.float32) / 255.0
            img_list.append(img)

        images_np = np.stack(img_list)  # [V, H, W, C]
        images_t = torch.from_numpy(images_np).float().to(self.device)
        images_t = images_t.permute(0, 3, 1, 2)  # [V, C, H, W]

        sample = edict({
            "image": images_t.unsqueeze(0),
            "c2w": torch.from_numpy(c2ws).float().to(self.device).unsqueeze(0),
            "fxfycxcy": torch.from_numpy(fxfycxcy).float().to(self.device).unsqueeze(0),
            "index": torch.stack([torch.arange(len(images), dtype=torch.long, device=self.device), torch.zeros(len(images), dtype=torch.long, device=self.device)], dim=-1).unsqueeze(0),
        })

        return self.forward(sample)

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
