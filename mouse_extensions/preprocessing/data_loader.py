#!/usr/bin/env python3
"""
Data Loader Module for Mouse Preprocessing

Handles loading of images and masks from various sources:
- Raw video files (.mp4)
- Pre-extracted images (.png)
- Pre-processed datasets (v12, v13, D1, D2)

Author: AI Research Assistant
Date: 2026-01-17
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2


class DataLoader:
    """
    Unified data loader for markerless mouse data.
    
    Supports:
    - Raw video files (videos_undist/*.mp4)
    - Mask videos (simpleclick_undist/*.mp4)
    - Pre-extracted images
    
    Args:
        data_dir: Path to data directory
        source_type: 'raw' (video files) or 'extracted' (image files)
        num_views: Number of camera views (default: 6)
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        source_type: str = 'raw',
        num_views: int = 6,
    ):
        self.data_dir = Path(data_dir)
        self.source_type = source_type
        self.num_views = num_views
        
        # Load camera params
        self.cameras = self._load_cameras()
        
        # Setup paths based on source type
        if source_type == 'raw':
            self.video_dir = self.data_dir / 'videos_undist'
            self.mask_dir = self.data_dir / 'simpleclick_undist'
            self.total_frames = self._get_video_frame_count()
        else:
            self.image_dir = self.data_dir / 'images'
            self.mask_dir = self.data_dir / 'masks'
            self.total_frames = self._count_extracted_frames()
    
    def _load_cameras(self) -> List[Dict]:
        """Load camera parameters from pickle file."""
        cam_file = self.data_dir / 'new_cam.pkl'
        if not cam_file.exists():
            # Try alternative locations
            for alt in ['cameras.pkl', 'cam_params.pkl']:
                alt_file = self.data_dir / alt
                if alt_file.exists():
                    cam_file = alt_file
                    break
        
        if not cam_file.exists():
            raise FileNotFoundError(f"Camera params not found in {self.data_dir}")
        
        with open(cam_file, 'rb') as f:
            cam_params = pickle.load(f)
        
        # Standardize camera format
        cameras = []
        for i in range(self.num_views):
            cam = cam_params[i]
            K = np.array(cam['K'])
            R = np.array(cam['R'])
            T = np.array(cam['T']).flatten()
            
            # Build w2c matrix
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = T
            
            cameras.append({
                'K': K,
                'R': R,
                'T': T,
                'w2c': w2c,
                'fx': float(K[0, 0]),
                'fy': float(K[1, 1]),
                'cx': float(K[0, 2]),
                'cy': float(K[1, 2]),
            })
        
        return cameras
    
    def _get_video_frame_count(self) -> int:
        """Get total frame count from video."""
        video_path = self.video_dir / '0.mp4'
        if not video_path.exists():
            return 0
        cap = cv2.VideoCapture(str(video_path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count
    
    def _count_extracted_frames(self) -> int:
        """Count extracted image frames."""
        if not self.image_dir.exists():
            return 0
        # Assume view 0 images
        view_dir = self.image_dir / 'view0'
        if view_dir.exists():
            return len(list(view_dir.glob('*.png')))
        return len(list(self.image_dir.glob('*_view0.png')))
    
    def load_frame(
        self,
        frame_idx: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load images and masks for a specific frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            images: List of (H, W, 3) RGB images
            masks: List of (H, W) binary masks
        """
        if self.source_type == 'raw':
            return self._load_from_video(frame_idx)
        else:
            return self._load_from_extracted(frame_idx)
    
    def _load_from_video(
        self,
        frame_idx: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load frame from video files."""
        images = []
        masks = []
        
        for view_idx in range(self.num_views):
            # Load image
            video_path = self.video_dir / f'{view_idx}.mp4'
            img = self._read_video_frame(video_path, frame_idx)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.zeros((576, 512, 3), dtype=np.uint8)
            images.append(img)
            
            # Load mask
            mask = self._load_mask(view_idx, frame_idx)
            masks.append(mask)
        
        return images, masks
    
    def _read_video_frame(
        self,
        video_path: Path,
        frame_idx: int,
    ) -> Optional[np.ndarray]:
        """Read a specific frame from video."""
        if not video_path.exists():
            return None
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    
    def _load_mask(
        self,
        view_idx: int,
        frame_idx: int,
    ) -> np.ndarray:
        """Load mask from video or image file."""
        # Try video first
        video_patterns = [
            self.mask_dir / f'{view_idx}.mp4',
            self.mask_dir / f'view{view_idx}.mp4',
        ]
        
        for video_path in video_patterns:
            if video_path.exists():
                frame = self._read_video_frame(video_path, frame_idx)
                if frame is not None:
                    if len(frame.shape) == 3:
                        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    return frame
        
        # Try PNG files
        png_patterns = [
            self.mask_dir / f'view{view_idx}' / f'{frame_idx:06d}.png',
            self.mask_dir / f'{view_idx}' / f'{frame_idx:06d}.png',
        ]
        
        for png_path in png_patterns:
            if png_path.exists():
                return cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        
        # Return empty mask
        return np.zeros((576, 512), dtype=np.uint8)
    
    def _load_from_extracted(
        self,
        frame_idx: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load from pre-extracted image files."""
        images = []
        masks = []
        
        for view_idx in range(self.num_views):
            # Try different naming patterns
            img_patterns = [
                self.image_dir / f'view{view_idx}' / f'{frame_idx:06d}.png',
                self.image_dir / f'{frame_idx:06d}_view{view_idx}.png',
            ]
            
            img = None
            for pattern in img_patterns:
                if pattern.exists():
                    img = cv2.imread(str(pattern))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    break
            
            if img is None:
                img = np.zeros((512, 512, 3), dtype=np.uint8)
            images.append(img)
            
            # Load mask
            mask = self._load_mask(view_idx, frame_idx)
            masks.append(mask)
        
        return images, masks
    
    def get_camera_matrices(self) -> List[Dict]:
        """Get camera parameters in standardized format."""
        return self.cameras
    
    def __len__(self) -> int:
        return self.total_frames
