"""
Unified Video Generator Module

Consolidates train/val visualization into common functions.
Eliminates duplication and ensures consistency.

Author: Claude Code
Date: 2026-01-25
"""

import os
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, List
from einops import rearrange

from .turntable_config import MOUSE_CAMERA_ORDER, get_grid_row_labels


def create_dataset_views_video(
    dataset_views: np.ndarray,  # [num_views, H, W, 3] uint8
    output_path: str,
    config: Dict[str, Any],
    imageseq2video_fn,  # Function to convert image sequence to video
) -> str:
    """
    Create video cycling through dataset camera views.
    
    Args:
        dataset_views: Rendered views from each camera [num_cams, H, W, 3]
        output_path: Full path for output video
        config: Turntable config dict with keys:
            - camera_order: List of camera indices (default: MOUSE_CAMERA_ORDER)
            - dataset_views_fps: FPS for the video (default: 2)
        imageseq2video_fn: Function(frames, path, fps) to save video
    
    Returns:
        Output path
    """
    fps = config.get("dataset_views_fps", 4)
    camera_order = config.get("camera_order", MOUSE_CAMERA_ORDER)
    
    # 각 카메라에서 1초 유지 (더 천천히)
    frames_per_cam = max(2, int(fps * 1.0))
    
    # camera_order에 따라 재정렬
    ordered_views = dataset_views[camera_order]  # [num_cams_ordered, H, W, 3]
    
    # 각 뷰를 frames_per_cam 만큼 반복
    video_frames = np.repeat(ordered_views, frames_per_cam, axis=0)
    
    # Loop: 마지막에 첫 프레임 추가
    if config.get("loop", True):
        video_frames = np.concatenate([
            video_frames, 
            np.repeat(ordered_views[:1], frames_per_cam, axis=0)
        ], axis=0)
    
    imageseq2video_fn(video_frames, output_path, fps=fps)
    return output_path


def create_turntable_video(
    turntable_frames: np.ndarray,  # [num_frames, H, W, 3] uint8
    output_path: str,
    config: Dict[str, Any],
    imageseq2video_fn,
    input_visualization: Optional[np.ndarray] = None,  # [H_in, W_in, 3]
) -> Tuple[str, Optional[str]]:
    """
    Create turntable video with optional input image panel.
    
    Args:
        turntable_frames: Rendered frames [num_frames, H, W, 3]
        output_path: Full path for output video  
        config: Turntable config dict with keys:
            - fps: Frames per second (default: 10)
        imageseq2video_fn: Function(frames, path, fps) to save video
        input_visualization: Optional input image to add as panel
    
    Returns:
        Tuple of (turntable_path, turntable_with_input_path or None)
    """
    fps = config.get("fps", 10)
    
    # Save main turntable video
    imageseq2video_fn(turntable_frames, output_path, fps=fps)
    
    combined_path = None
    if input_visualization is not None:
        # Create combined video with input panel
        combined_frames = []
        input_h, input_w = input_visualization.shape[:2]
        
        for frame in turntable_frames:
            # Resize input to match frame height
            frame_h = frame.shape[0]
            if input_h != frame_h:
                import cv2
                scale = frame_h / input_h
                new_w = int(input_w * scale)
                input_resized = cv2.resize(input_visualization, (new_w, frame_h))
            else:
                input_resized = input_visualization
            
            combined = np.concatenate([input_resized, frame], axis=1)
            combined_frames.append(combined)
        
        combined_frames = np.stack(combined_frames)
        combined_path = output_path.replace(".mp4", "_with_input.mp4")
        imageseq2video_fn(combined_frames, combined_path, fps=fps)
    
    return output_path, combined_path


def create_grid_image(
    frames: np.ndarray,  # [num_frames, H, W, 3] uint8
    config: Dict[str, Any],
    add_row_labels_fn=None,  # Function to add row labels
) -> np.ndarray:
    """
    Create grid image from frames.
    
    Args:
        frames: Frames to arrange [num_frames, H, W, 3]
        config: Dict with grid_rows, grid_cols, camera_order, loop
        add_row_labels_fn: Optional function to add row labels
    
    Returns:
        Grid image [H_grid, W_grid, 3]
    """
    grid_rows = config.get("grid_rows", 6)
    grid_cols = config.get("grid_cols", 10)
    num_frames = frames.shape[0]
    
    # Limit frames to grid size
    max_frames = grid_rows * grid_cols
    if num_frames > max_frames:
        # Sample evenly
        indices = np.linspace(0, num_frames - 1, max_frames).astype(int)
        frames = frames[indices]
    elif num_frames < max_frames:
        # Pad with last frame
        padding = max_frames - num_frames
        frames = np.concatenate([frames, np.repeat(frames[-1:], padding, axis=0)], axis=0)
    
    # Reshape to grid
    h, w = frames.shape[1:3]
    grid = frames.reshape(grid_rows, grid_cols, h, w, 3)
    grid = grid.transpose(0, 2, 1, 3, 4)  # [rows, h, cols, w, 3]
    grid = grid.reshape(grid_rows * h, grid_cols * w, 3)
    
    # Add row labels if function provided
    if add_row_labels_fn is not None:
        camera_order = config.get("camera_order", MOUSE_CAMERA_ORDER)
        loop = config.get("loop", True)
        grid = add_row_labels_fn(
            grid, camera_order, grid_rows, grid_cols, h,
            label_height=25, loop=loop
        )
    
    return grid


class VideoGeneratorContext:
    """
    Context holder for visualization generation.
    Provides consistent interface for both train and val contexts.
    """
    
    def __init__(
        self,
        context: str,  # "train" or "val"
        config: Dict[str, Any],
        output_dir: str,
        item_uid: str,
        imageseq2video_fn,
        add_row_labels_fn=None,
    ):
        self.context = context
        self.config = config
        self.output_dir = output_dir
        self.item_uid = item_uid
        self.imageseq2video = imageseq2video_fn
        self.add_row_labels = add_row_labels_fn
        
        # Get turntable config
        self.turntable_cfg = config.get("visualization", {}).get("turntable", {})
    
    def generate_turntable_video(
        self,
        turntable_frames: np.ndarray,
        input_visualization: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """Generate turntable video(s) and return paths."""
        paths = {}
        
        turntable_path = os.path.join(
            self.output_dir, 
            f"turntable_{self.item_uid}.mp4"
        )
        
        main_path, combined_path = create_turntable_video(
            turntable_frames,
            turntable_path,
            self.turntable_cfg,
            self.imageseq2video,
            input_visualization,
        )
        
        paths["turntable"] = main_path
        if combined_path:
            paths["turntable_with_input"] = combined_path
        
        return paths
    
    def generate_dataset_views_video(
        self,
        dataset_views: np.ndarray,
    ) -> Optional[str]:
        """Generate dataset views video if enabled."""
        if not self.turntable_cfg.get("save_dataset_views_video", False):
            return None
        
        output_path = os.path.join(
            self.output_dir,
            f"dataset_views_{self.item_uid}.mp4"
        )
        
        return create_dataset_views_video(
            dataset_views,
            output_path,
            self.turntable_cfg,
            self.imageseq2video,
        )
    
    def generate_grid_image(
        self,
        frames: np.ndarray,
    ) -> np.ndarray:
        """Generate grid image from frames."""
        return create_grid_image(
            frames,
            self.turntable_cfg,
            self.add_row_labels,
        )
    
    def save_grid_image(
        self,
        grid: np.ndarray,
        suffix: str = "",
    ) -> str:
        """Save grid image and return path."""
        from PIL import Image
        
        filename = f"turntable_{self.item_uid}{suffix}.jpg"
        path = os.path.join(self.output_dir, filename)
        Image.fromarray(grid).save(path)
        return path


def get_video_generator(
    context: str,
    config: Dict[str, Any],
    output_dir: str,
    item_uid: str,
) -> VideoGeneratorContext:
    """
    Factory function to create VideoGeneratorContext.
    
    Imports rendering utilities lazily to avoid circular imports.
    """
    from gslrm.model.gaussians_renderer import imageseq2video, add_row_labels_to_grid
    
    return VideoGeneratorContext(
        context=context,
        config=config,
        output_dir=output_dir,
        item_uid=item_uid,
        imageseq2video_fn=imageseq2video,
        add_row_labels_fn=add_row_labels_to_grid,
    )
