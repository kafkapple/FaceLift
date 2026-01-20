#!/usr/bin/env python3
"""
D7.1 Preprocessing: PP-Centered Shift with Individual Scale
============================================================

Fix for D7's fy normalization issue:
- D7: scale = target_fx / orig_fx (same for x and y)
- D7.1: scale_x = target_fx / orig_fx, scale_y = target_fy / orig_fy

This properly handles non-square pixels (fx != fy).

Created: 2026-01-20
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


JUMP_FRAMES = {5900, 11800, 17700}


@dataclass
class D71Config:
    """D7.1 preprocessing configuration"""
    target_fx: float = 549.0
    target_fy: float = 549.0
    target_pp: Tuple[float, float] = (256.0, 256.0)
    target_distance: float = 2.7
    output_size: int = 512
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 0)


class D71Preprocessor:
    """
    D7.1 Preprocessing: PP-Centered Shift with Individual Scale
    
    Key difference from D7:
    - Uses separate scale_x and scale_y for proper aspect ratio handling
    """

    def __init__(self, config: D71Config = None):
        self.config = config or D71Config()

    def load_original_cameras(self, pkl_path: str) -> List[Dict]:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    def compute_transform(
        self,
        K: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        orig_size: Tuple[int, int] = (1152, 1024)
    ) -> Tuple[Dict, np.ndarray]:
        """
        Compute camera transform with INDIVIDUAL scale for x and y.
        
        Key difference from D7:
        - scale_x = target_fx / orig_fx
        - scale_y = target_fy / orig_fy
        - This properly handles non-square pixels
        """
        cfg = self.config

        # Original intrinsics
        orig_fx = K[0, 0]
        orig_fy = K[1, 1]
        orig_cx = K[0, 2]
        orig_cy = K[1, 2]

        # Step 1: Compute INDIVIDUAL scale factors (KEY FIX!)
        scale_x = cfg.target_fx / orig_fx
        scale_y = cfg.target_fy / orig_fy

        # Step 2: Compute scaled PP (using individual scales)
        scaled_cx = orig_cx * scale_x
        scaled_cy = orig_cy * scale_y

        # Step 3: Compute shift to move PP to target (256, 256)
        shift_x = cfg.target_pp[0] - scaled_cx
        shift_y = cfg.target_pp[1] - scaled_cy

        # Step 4: Build affine transform matrix (different scale for x and y!)
        affine_matrix = np.array([
            [scale_x, 0, shift_x],
            [0, scale_y, shift_y]
        ], dtype=np.float32)

        # Step 5: Distance normalization
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T.flatten()
        c2w = np.linalg.inv(w2c)
        cam_pos = c2w[:3, 3]
        current_distance = np.linalg.norm(cam_pos)

        distance_scale = cfg.target_distance / current_distance
        new_cam_pos = cam_pos * distance_scale
        new_c2w = c2w.copy()
        new_c2w[:3, 3] = new_cam_pos
        new_w2c = np.linalg.inv(new_c2w)

        # Step 6: Build camera dict
        camera_dict = {
            "w": cfg.output_size,
            "h": cfg.output_size,
            "fx": cfg.target_fx,
            "fy": cfg.target_fy,
            "cx": cfg.target_pp[0],
            "cy": cfg.target_pp[1],
            "w2c": new_w2c.tolist(),
            "_original": {
                "fx": float(orig_fx),
                "fy": float(orig_fy),
                "cx": float(orig_cx),
                "cy": float(orig_cy),
                "distance": float(current_distance),
                "aspect_ratio": float(orig_fy / orig_fx),
            },
            "_transform": {
                "method": "D7.1_PP_centered_individual_scale",
                "scale_x": float(scale_x),
                "scale_y": float(scale_y),
                "shift_x": float(shift_x),
                "shift_y": float(shift_y),
                "scaled_pp_before_shift": [float(scaled_cx), float(scaled_cy)],
                "target_distance": cfg.target_distance,
            },
        }

        return camera_dict, affine_matrix

    def transform_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        affine_matrix: np.ndarray
    ) -> np.ndarray:
        cfg = self.config
        output_size = cfg.output_size

        warped_img = cv2.warpAffine(
            image,
            affine_matrix,
            (output_size, output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        warped_mask = cv2.warpAffine(
            mask,
            affine_matrix,
            (output_size, output_size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        if len(warped_img.shape) == 2:
            warped_img = cv2.cvtColor(warped_img, cv2.COLOR_GRAY2RGB)
        elif warped_img.shape[2] == 4:
            warped_img = warped_img[:, :, :3]

        output = np.zeros((output_size, output_size, 4), dtype=np.uint8)
        output[:, :, :3] = warped_img
        output[:, :, 3] = warped_mask

        return output


def main():
    parser = argparse.ArgumentParser(description="D7.1: PP-Centered Shift with Individual Scale")
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--camera-pkl', required=True)
    parser.add_argument('--frame-interval', type=int, default=5)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--val-ratio', type=float, default=0.1)

    args = parser.parse_args()
    
    print("D7.1: Individual Scale Method")
    print("="*50)
    print("Fix: scale_x = target_fx/orig_fx, scale_y = target_fy/orig_fy")
    print("This properly handles non-square pixels (fx != fy)")
    print()
    
    # Implementation same as D7 but using D71Preprocessor
    # ... (full implementation would follow same pattern as D7)
    print("Run with full implementation: preprocess_D7_pp_centered.py with --method individual")


if __name__ == "__main__":
    main()
