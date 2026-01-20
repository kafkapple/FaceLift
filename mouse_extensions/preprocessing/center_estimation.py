#!/usr/bin/env python3
"""
Multi-View Object Center Estimation Module

Provides three approaches for estimating the 3D object center from multi-view masks:

1. **Triangulation**: DLT triangulation of 2D mask centroids
2. **Visual Hull (Shape Carving)**: 3D voxel grid intersection from all views  
3. **Global Average**: Simple average of 2D offsets (quick fix)

Key Insight (2026-01-17):
    Per-view 2D center estimation causes cross-view inconsistency because
    each view's centroid/bbox differs due to projection. The correct approach
    is to estimate a single 3D center and back-project to all views.

Usage:
    from mouse_extensions.preprocessing.center_estimation import CenterEstimator
    
    estimator = CenterEstimator(cameras, method='triangulation')
    center_3d, centers_2d = estimator.estimate(masks)

Author: AI Research Assistant  
Date: 2026-01-17
Based on: pose-splatter/src/preprocessing/center_estimator.py
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path


class CenterMethod(Enum):
    """Available center estimation methods."""
    TRIANGULATION = "triangulation"
    VISUAL_HULL = "visual_hull"
    GLOBAL_AVERAGE = "global_average"


@dataclass
class CenterEstimationResult:
    """Result of center estimation."""
    method: CenterMethod
    center_3d: np.ndarray           # (3,) 3D center in world coordinates
    centers_2d: np.ndarray          # (N_views, 2) projected centers per view
    confidence: float               # Estimation confidence [0, 1]
    ray_convergence_error: float    # Mean distance from rays to center (mm)
    metadata: Dict                  # Additional method-specific info


class CenterEstimator:
    """
    Multi-view object center estimator.
    
    Estimates a consistent 3D object center from multi-view segmentation masks,
    ensuring cross-view geometric consistency for 3D reconstruction.
    
    Args:
        cameras: List of camera dicts with 'K' (intrinsics) and 'w2c' (world-to-camera)
        method: Center estimation method
        grid_size: Voxel grid resolution for visual hull (default: 64)
        grid_extent: Physical extent of grid in world units (default: 0.3)
    """
    
    def __init__(
        self,
        cameras: List[Dict],
        method: Union[str, CenterMethod] = CenterMethod.TRIANGULATION,
        grid_size: int = 64,
        grid_extent: float = 0.3,
    ):
        self.cameras = cameras
        self.n_views = len(cameras)
        
        if isinstance(method, str):
            method = CenterMethod(method)
        self.method = method
        
        self.grid_size = grid_size
        self.grid_extent = grid_extent
        
        # Pre-compute camera matrices
        self._setup_camera_matrices()
        
    def _setup_camera_matrices(self):
        """Extract and compute camera matrices."""
        self.intrinsics = []
        self.extrinsics = []  # w2c
        self.c2w_matrices = []
        self.projection_matrices = []  # P = K @ [R|t]
        
        for cam in self.cameras:
            # Intrinsics
            if 'K' in cam:
                K = np.array(cam['K'])
            else:
                K = np.array([
                    [cam['fx'], 0, cam['cx']],
                    [0, cam['fy'], cam['cy']],
                    [0, 0, 1]
                ])
            self.intrinsics.append(K)
            
            # Extrinsics (world-to-camera)
            if 'w2c' in cam:
                w2c = np.array(cam['w2c'])
            else:
                raise ValueError("Camera must have 'w2c' matrix")
            
            self.extrinsics.append(w2c)
            c2w = np.linalg.inv(w2c)
            self.c2w_matrices.append(c2w)
            
            # Projection matrix P = K @ [R|t]
            P = K @ w2c[:3, :]
            self.projection_matrices.append(P)
    
    def estimate(
        self,
        masks: Union[np.ndarray, List[np.ndarray]],
        rough_center: Optional[np.ndarray] = None,
    ) -> CenterEstimationResult:
        """
        Estimate 3D object center from multi-view masks.
        
        Args:
            masks: (N_views, H, W) binary masks or list of masks
            rough_center: Optional initial 3D center estimate
            
        Returns:
            CenterEstimationResult with 3D center and per-view 2D projections
        """
        if isinstance(masks, list):
            masks = np.array(masks)
        
        # Compute 2D centroids for all methods (needed for validation)
        centroids_2d = self._compute_2d_centroids(masks)
        
        if self.method == CenterMethod.TRIANGULATION:
            return self._estimate_triangulation(masks, centroids_2d)
        elif self.method == CenterMethod.VISUAL_HULL:
            return self._estimate_visual_hull(masks, centroids_2d, rough_center)
        elif self.method == CenterMethod.GLOBAL_AVERAGE:
            return self._estimate_global_average(masks, centroids_2d)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _compute_2d_centroids(self, masks: np.ndarray) -> np.ndarray:
        """Compute 2D mask centroids for all views."""
        centroids = []
        for i, mask in enumerate(masks):
            coords = np.where(mask > 0.5)
            if len(coords[0]) > 0:
                cy = coords[0].mean()
                cx = coords[1].mean()
            else:
                # Fallback to image center
                cy, cx = mask.shape[0] / 2, mask.shape[1] / 2
            centroids.append([cx, cy])
        return np.array(centroids)
    
    # ========== Method 1: Triangulation ==========
    
    def _estimate_triangulation(
        self,
        masks: np.ndarray,
        centroids_2d: np.ndarray,
    ) -> CenterEstimationResult:
        """
        Estimate 3D center using DLT triangulation of 2D centroids.
        
        This is the most geometrically accurate method when cameras are calibrated.
        """
        # DLT triangulation
        center_3d = self._triangulate_dlt(centroids_2d)
        
        # Project back to all views
        centers_2d = self._project_to_all_views(center_3d)
        
        # Compute ray convergence error
        ray_error = self._compute_ray_convergence_error(center_3d, centroids_2d)
        
        # Confidence based on ray convergence
        confidence = np.exp(-ray_error / 10.0)  # Decay with error
        
        return CenterEstimationResult(
            method=CenterMethod.TRIANGULATION,
            center_3d=center_3d,
            centers_2d=centers_2d,
            confidence=float(confidence),
            ray_convergence_error=float(ray_error),
            metadata={
                'original_centroids_2d': centroids_2d,
                'reprojection_errors': np.linalg.norm(centers_2d - centroids_2d, axis=1),
            }
        )
    
    def _triangulate_dlt(self, points_2d: np.ndarray) -> np.ndarray:
        """
        DLT (Direct Linear Transform) triangulation.
        
        Solves the linear system: A @ X = 0
        where A is constructed from projection matrices and 2D points.
        """
        A = []
        for i, (x, y) in enumerate(points_2d):
            P = self.projection_matrices[i]
            # Each point gives 2 equations
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])
        
        A = np.array(A)
        
        # SVD solution
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        
        # Convert from homogeneous coordinates
        X = X[:3] / (X[3] + 1e-10)
        
        return X
    
    # ========== Method 2: Visual Hull ==========
    
    def _estimate_visual_hull(
        self,
        masks: np.ndarray,
        centroids_2d: np.ndarray,
        rough_center: Optional[np.ndarray] = None,
    ) -> CenterEstimationResult:
        """
        Estimate 3D center using shape carving (visual hull).
        
        Creates a 3D voxel grid and projects to each view to accumulate
        votes from mask silhouettes.
        """
        # Get rough center first (for grid placement)
        if rough_center is None:
            rough_center = self._triangulate_dlt(centroids_2d)
        
        # Create 3D grid centered on rough center
        grid = self._create_3d_grid(rough_center)
        
        # Shape carving
        volume = self._carve_volume(masks, grid)
        
        # Threshold and compute weighted centroid
        threshold = (self.n_views - 1) / self.n_views
        binary_volume = (volume >= threshold).astype(float)
        
        # Weighted centroid
        center_3d = self._compute_volume_centroid(binary_volume, grid)
        
        # Project back to all views
        centers_2d = self._project_to_all_views(center_3d)
        
        # Ray convergence error
        ray_error = self._compute_ray_convergence_error(center_3d, centroids_2d)
        
        # Confidence based on volume occupancy
        occupancy_ratio = binary_volume.sum() / binary_volume.size
        confidence = min(1.0, occupancy_ratio * 10)  # Scale occupancy
        
        # Compute reprojection error (distance from back-projected to original centroids)
        reproj_errors = np.linalg.norm(centers_2d - centroids_2d, axis=1)
        
        return CenterEstimationResult(
            method=CenterMethod.VISUAL_HULL,
            center_3d=center_3d,
            centers_2d=centers_2d,
            confidence=float(confidence),
            ray_convergence_error=float(ray_error),
            metadata={
                'rough_center': rough_center,
                'volume_shape': binary_volume.shape,
                'occupancy_ratio': float(occupancy_ratio),
                'threshold': float(threshold),
                'original_centroids_2d': centroids_2d,
                'reprojection_errors': reproj_errors,
            }
        )
    
    def _create_3d_grid(self, center: np.ndarray) -> np.ndarray:
        """Create 3D coordinate grid centered on given point."""
        half = self.grid_extent / 2
        coords = np.linspace(-half, half, self.grid_size)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        grid = np.stack([X, Y, Z], axis=-1) + center
        return grid
    
    def _carve_volume(
        self,
        masks: np.ndarray,
        grid: np.ndarray,
    ) -> np.ndarray:
        """
        Shape carving: project grid to each view and accumulate votes.
        """
        D = self.grid_size
        points = grid.reshape(-1, 3)
        votes = np.zeros(len(points), dtype=np.float32)
        
        for view_idx in range(self.n_views):
            K = self.intrinsics[view_idx]
            w2c = self.extrinsics[view_idx]
            mask = masks[view_idx]
            H, W = mask.shape
            
            # Transform to camera coordinates
            points_hom = np.hstack([points, np.ones((len(points), 1))])
            points_cam = (w2c @ points_hom.T).T[:, :3]
            
            # Check if in front of camera
            valid = points_cam[:, 2] > 0
            
            # Project to image
            points_img = (K @ points_cam.T).T
            u = points_img[:, 0] / (points_img[:, 2] + 1e-10)
            v = points_img[:, 1] / (points_img[:, 2] + 1e-10)
            
            # Check bounds
            in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H) & valid
            
            # Sample mask
            view_votes = np.zeros(len(points), dtype=np.float32)
            in_image_idx = np.where(in_image)[0]
            
            if len(in_image_idx) > 0:
                u_int = u[in_image_idx].astype(int)
                v_int = v[in_image_idx].astype(int)
                view_votes[in_image_idx] = mask[v_int, u_int] / 255.0
            
            votes += view_votes
        
        volume = votes.reshape(D, D, D) / self.n_views
        return volume
    
    def _compute_volume_centroid(
        self,
        volume: np.ndarray,
        grid: np.ndarray,
    ) -> np.ndarray:
        """Compute weighted centroid of volume."""
        total = volume.sum()
        if total < 1e-10:
            return grid[grid.shape[0]//2, grid.shape[1]//2, grid.shape[2]//2]
        
        weights = volume / total
        centroid = np.sum(grid * weights[..., None], axis=(0, 1, 2))
        return centroid
    
    # ========== Method 3: Global Average ==========
    
    def _estimate_global_average(
        self,
        masks: np.ndarray,
        centroids_2d: np.ndarray,
    ) -> CenterEstimationResult:
        """
        Simple global average approach (quick fix).
        
        Computes the mean offset from image center across all views
        and applies uniformly. Less accurate but fast.
        """
        H, W = masks[0].shape
        image_center = np.array([W / 2, H / 2])
        
        # Compute offsets from image center
        offsets = centroids_2d - image_center
        
        # Global average offset
        mean_offset = offsets.mean(axis=0)
        
        # Apply same offset to all views
        uniform_centers_2d = np.tile(image_center + mean_offset, (self.n_views, 1))
        
        # Estimate 3D center using triangulation of uniform centers
        # (This is approximate but maintains consistency)
        center_3d = self._triangulate_dlt(uniform_centers_2d)
        
        # Reproject to get actual consistent 2D centers
        centers_2d = self._project_to_all_views(center_3d)
        
        # Ray convergence error (will be higher than triangulation)
        ray_error = self._compute_ray_convergence_error(center_3d, centroids_2d)
        
        # Lower confidence due to approximation
        offset_variance = np.var(offsets, axis=0).mean()
        confidence = np.exp(-offset_variance / 100.0)
        
        return CenterEstimationResult(
            method=CenterMethod.GLOBAL_AVERAGE,
            center_3d=center_3d,
            centers_2d=centers_2d,
            confidence=float(confidence),
            ray_convergence_error=float(ray_error),
            metadata={
                'mean_offset': mean_offset,
                'offset_variance': float(offset_variance),
                'original_centroids_2d': centroids_2d,
            }
        )
    
    # ========== Common Utilities ==========
    
    def _project_to_all_views(self, point_3d: np.ndarray) -> np.ndarray:
        """Project 3D point to all camera views."""
        centers_2d = []
        point_hom = np.append(point_3d, 1.0)
        
        for i in range(self.n_views):
            P = self.projection_matrices[i]
            projected = P @ point_hom
            u = projected[0] / (projected[2] + 1e-10)
            v = projected[1] / (projected[2] + 1e-10)
            centers_2d.append([u, v])
        
        return np.array(centers_2d)
    
    def _compute_ray_convergence_error(
        self,
        center_3d: np.ndarray,
        points_2d: np.ndarray,
    ) -> float:
        """
        Compute mean distance from camera rays to the estimated 3D center.
        
        This measures how well the rays from all views converge at the center.
        """
        total_error = 0.0
        
        for i in range(self.n_views):
            # Camera position
            c2w = self.c2w_matrices[i]
            cam_pos = c2w[:3, 3]
            
            # Ray direction from camera through 2D point
            K_inv = np.linalg.inv(self.intrinsics[i])
            point_hom = np.array([points_2d[i, 0], points_2d[i, 1], 1.0])
            ray_cam = K_inv @ point_hom
            ray_cam = ray_cam / np.linalg.norm(ray_cam)
            
            # Transform to world coordinates
            ray_world = c2w[:3, :3] @ ray_cam
            ray_world = ray_world / np.linalg.norm(ray_world)
            
            # Distance from ray to center
            # d = ||(center - cam_pos) - ((center - cam_pos) · ray) * ray||
            v = center_3d - cam_pos
            proj_length = np.dot(v, ray_world)
            closest_point = cam_pos + proj_length * ray_world
            distance = np.linalg.norm(center_3d - closest_point)
            
            total_error += distance
        
        return total_error / self.n_views


def estimate_center_for_frame(
    masks: np.ndarray,
    cameras: List[Dict],
    method: str = 'triangulation',
    **kwargs
) -> CenterEstimationResult:
    """
    Convenience function for single-frame center estimation.
    
    Args:
        masks: (N_views, H, W) binary masks
        cameras: List of camera dicts
        method: 'triangulation', 'visual_hull', or 'global_average'
        
    Returns:
        CenterEstimationResult
    """
    estimator = CenterEstimator(cameras, method=method, **kwargs)
    return estimator.estimate(masks)


def compare_all_methods(
    masks: np.ndarray,
    cameras: List[Dict],
    **kwargs
) -> Dict[str, CenterEstimationResult]:
    """
    Compare all three center estimation methods.
    
    Args:
        masks: (N_views, H, W) binary masks
        cameras: List of camera dicts
        
    Returns:
        Dict mapping method name to result
    """
    results = {}
    for method in CenterMethod:
        estimator = CenterEstimator(cameras, method=method, **kwargs)
        results[method.value] = estimator.estimate(masks)
    return results


# ========== Method 4: Per-View 2D Centroid (Broken - for comparison) ==========

class PerView2DCentroidResult:
    """
    Result for per-view 2D centroid method (the broken approach).
    
    This method computes 2D centroid independently for each view,
    which causes cross-view inconsistency in multi-view reconstruction.
    Included for comparison purposes only.
    """
    def __init__(
        self,
        centroids_2d: np.ndarray,
        cross_view_error: float,
        triangulated_center_3d: np.ndarray,
    ):
        self.method = "per_view_2d"
        self.centroids_2d = centroids_2d  # (N_views, 2) - each view's independent centroid
        self.cross_view_error = cross_view_error  # Error measuring inconsistency
        self.triangulated_center_3d = triangulated_center_3d  # What 3D point these would give
        self.confidence = 0.0  # Always low - this method is broken


def compute_per_view_2d_error(
    masks: np.ndarray,
    cameras: List[Dict],
) -> Dict:
    """
    Compute error metrics for the per-view 2D centroid approach.
    
    This demonstrates why per-view 2D centroid is problematic:
    - Each view's centroid is computed independently
    - These centroids don't correspond to the same 3D point
    - Using them for cropping causes cross-view inconsistency
    
    Returns:
        Dict with:
        - centroids_2d: Per-view 2D centroids
        - cross_view_std: Standard deviation of reprojection errors
        - triangulation_residual: How far off the rays are from intersecting
        - equivalent_pixel_error: Pixel-space error from using per-view centers
    """
    # Compute 2D centroid for each view independently
    centroids_2d = []
    for mask in masks:
        coords = np.where(mask > 0.5)
        if len(coords[0]) > 0:
            cy = coords[0].mean()
            cx = coords[1].mean()
        else:
            cy, cx = mask.shape[0] / 2, mask.shape[1] / 2
        centroids_2d.append([cx, cy])
    centroids_2d = np.array(centroids_2d)
    
    # Now triangulate these 2D points to see what 3D point they would give
    # This 3D point will have high residual because the 2D points are inconsistent
    estimator = CenterEstimator(cameras, method='triangulation')
    
    # Triangulate the per-view centroids
    tri_center = estimator._triangulate_dlt(centroids_2d)
    
    # Compute ray convergence error (how far rays miss each other)
    ray_error = estimator._compute_ray_convergence_error(tri_center, centroids_2d)
    
    # Project the triangulated center back to each view
    projected_centers = estimator._project_to_all_views(tri_center)
    
    # Compute the difference between original 2D centroids and projected centers
    # This shows how inconsistent the per-view centroids are
    reprojection_diff = centroids_2d - projected_centers
    pixel_errors = np.linalg.norm(reprojection_diff, axis=1)
    
    return {
        'method': 'per_view_2d',
        'centroids_2d': centroids_2d.tolist(),
        'triangulated_center_3d': tri_center.tolist(),
        'ray_convergence_error': float(ray_error),
        'reprojection_errors': pixel_errors.tolist(),
        'mean_reprojection_error': float(pixel_errors.mean()),
        'max_reprojection_error': float(pixel_errors.max()),
        'std_reprojection_error': float(pixel_errors.std()),
        'confidence': 0.0,  # This method is broken
        '_description': 'Per-view 2D centroid (BROKEN - for comparison only)',
    }


def compare_all_methods_with_perview(
    masks: np.ndarray,
    cameras: List[Dict],
    **kwargs
) -> Dict:
    """
    Compare all methods including the broken per-view 2D approach.
    
    This function is specifically for validation/comparison purposes
    to demonstrate why per-view 2D centroid doesn't work.
    """
    # Get results from proper 3D methods
    results = compare_all_methods(masks, cameras, **kwargs)
    
    # Add per-view 2D centroid results
    perview_result = compute_per_view_2d_error(masks, cameras)
    results['per_view_2d'] = perview_result
    
    return results
