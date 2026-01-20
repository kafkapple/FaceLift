#!/usr/bin/env python3
"""
D7.2 Preprocessing: PP-Centered Shift with Average Scale
=========================================================

Alternative fix for D7's fy normalization issue:
- D7: scale = target_fx / orig_fx
- D7.2: scale = (target_fx/orig_fx + target_fy/orig_fy) / 2

This maintains isotropic scaling while being more balanced.
Trade-off: Neither fx nor fy will be exactly 549, but both will be close.

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
class D72Config:
    """D7.2 preprocessing configuration"""
    target_fx: float = 549.0
    target_fy: float = 549.0
    target_pp: Tuple[float, float] = (256.0, 256.0)
    target_distance: float = 2.7
    output_size: int = 512
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 0)


class D72Preprocessor:
    """
    D7.2 Preprocessing: PP-Centered Shift with Average Scale
    
    Key difference from D7:
    - Uses average of scale_x and scale_y for isotropic scaling
    - Records actual resulting fx, fy (not forced to 549)
    """

    def __init__(self, config: D72Config = None):
        self.config = config or D72Config()

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
        Compute camera transform with AVERAGE scale.
        
        Key difference from D7:
        - scale = (scale_x + scale_y) / 2
        - Actual fx, fy are recorded (not forced to 549)
        - Maintains isotropic scaling while being more balanced
        """
        cfg = self.config

        # Original intrinsics
        orig_fx = K[0, 0]
        orig_fy = K[1, 1]
        orig_cx = K[0, 2]
        orig_cy = K[1, 2]

        # Step 1: Compute AVERAGE scale (KEY FIX!)
        scale_x = cfg.target_fx / orig_fx
        scale_y = cfg.target_fy / orig_fy
        scale = (scale_x + scale_y) / 2.0

        # Step 2: Compute actual resulting fx, fy
        actual_fx = orig_fx * scale
        actual_fy = orig_fy * scale

        # Step 3: Compute scaled PP
        scaled_cx = orig_cx * scale
        scaled_cy = orig_cy * scale

        # Step 4: Compute shift to move PP to target (256, 256)
        shift_x = cfg.target_pp[0] - scaled_cx
        shift_y = cfg.target_pp[1] - scaled_cy

        # Step 5: Build affine transform matrix (isotropic scale)
        affine_matrix = np.array([
            [scale, 0, shift_x],
            [0, scale, shift_y]
        ], dtype=np.float32)

        # Step 6: Distance normalization
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

        # Step 7: Build camera dict with ACTUAL fx, fy (not forced!)
        camera_dict = {
            "w": cfg.output_size,
            "h": cfg.output_size,
            "fx": float(actual_fx),  # Actual, not forced to 549
            "fy": float(actual_fy),  # Actual, not forced to 549
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
                "method": "D7.2_PP_centered_average_scale",
                "scale_avg": float(scale),
                "scale_x_would_be": float(scale_x),
                "scale_y_would_be": float(scale_y),
                "shift_x": float(shift_x),
                "shift_y": float(shift_y),
                "scaled_pp_before_shift": [float(scaled_cx), float(scaled_cy)],
                "target_distance": cfg.target_distance,
                "actual_fx": float(actual_fx),
                "actual_fy": float(actual_fy),
                "fx_error_from_549": float(abs(actual_fx - 549)),
                "fy_error_from_549": float(abs(actual_fy - 549)),
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
    parser = argparse.ArgumentParser(description="D7.2: PP-Centered Shift with Average Scale")
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--camera-pkl', required=True)
    parser.add_argument('--frame-interval', type=int, default=5)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--val-ratio', type=float, default=0.1)

    args = parser.parse_args()
    
    print("D7.2: Average Scale Method")
    print("="*50)
    print("Fix: scale = (target_fx/orig_fx + target_fy/orig_fy) / 2")
    print("Trade-off: fx, fy won't be exactly 549, but aspect ratio preserved")
    print()
    print("Run with full implementation: preprocess_D7_pp_centered.py with --method average")


if __name__ == "__main__":
    main()
