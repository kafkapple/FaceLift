# Copyright 2026 FaceLift Mouse Extensions
# GS-LRM integration for Deformation Network training

"""
GS-LRM integration: Generate Gaussian pseudo ground truth for deformation learning.

This module bridges GS-LRM inference with the deformation training pipeline:
1. Load pre-trained GS-LRM model
2. Process frame pairs to generate G_t and G_{t+1}
3. Convert GS-LRM output to GaussianParams format
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from easydict import EasyDict as edict

from .gaussian_params import GaussianParams


class GSLRMGaussianGenerator:
    """
    Wrapper for GS-LRM to generate Gaussian pseudo GT.
    
    Used in deformation training to create target Gaussians
    that the deformation network learns to predict.
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        sh_degree: int = 2,
    ):
        """
        Initialize GS-LRM model for Gaussian generation.
        
        Args:
            config_path: Path to GS-LRM config YAML
            checkpoint_path: Path to GS-LRM checkpoint
            device: Torch device
            sh_degree: Spherical harmonics degree (for GaussianParams parsing)
        """
        self.device = device
        self.sh_degree = sh_degree
        
        # Lazy import to avoid circular dependencies
        from mouse_extensions.inference.gslrm_pipeline import GSLRMInference
        
        self.gslrm = GSLRMInference(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        
        print(f"GSLRMGaussianGenerator initialized (sh_degree={sh_degree})")
    
    @torch.no_grad()
    def generate_gaussians(
        self,
        images: torch.Tensor,
        c2ws: torch.Tensor,
        fxfycxcys: torch.Tensor,
        index: Optional[torch.Tensor] = None,
    ) -> GaussianParams:
        """
        Generate Gaussians from multi-view images.
        
        Args:
            images: [B, V, C, H, W] multi-view images
            c2ws: [B, V, 4, 4] camera-to-world matrices
            fxfycxcys: [B, V, 4] intrinsics (fx, fy, cx, cy)
            index: [B, V, 2] optional index tensor
            
        Returns:
            GaussianParams with generated Gaussians
        """
        B, V = images.shape[:2]
        
        # Create index if not provided
        if index is None:
            index = torch.zeros(B, V, 2, dtype=torch.long, device=self.device)
            for b in range(B):
                for v in range(V):
                    index[b, v] = torch.tensor([v, b])
        
        # Move to device
        images = images.to(self.device)
        c2ws = c2ws.to(self.device)
        fxfycxcys = fxfycxcys.to(self.device)
        index = index.to(self.device)
        
        # Run GS-LRM
        result = self.gslrm.predict(images, c2ws, fxfycxcys, index)
        
        # Extract Gaussian parameters
        gaussian_params = self._extract_gaussian_params(result)
        
        return gaussian_params
    
    def _extract_gaussian_params(self, result: edict) -> GaussianParams:
        """
        Extract GaussianParams from GS-LRM result.
        
        Args:
            result: GS-LRM forward result containing gaussian_model
            
        Returns:
            GaussianParams instance
        """
        # Get gaussian model from result
        gm = result.gaussian_model
        
        # Extract parameters (these are already on GPU)
        xyz = gm.get_xyz  # [N, 3]
        features = gm.get_features  # [N, F] SH features
        scaling = gm._scaling  # [N, 3] log-scale (pre-activation)
        rotation = gm._rotation  # [N, 4] quaternion
        opacity = gm._opacity  # [N, 1] logit (pre-activation)
        
        # Create GaussianParams
        return GaussianParams(
            xyz=xyz,
            features=features,
            scaling=scaling,
            rotation=rotation,
            opacity=opacity,
        )
    
    def generate_pair(
        self,
        frame_t_data: Dict,
        frame_t1_data: Dict,
    ) -> Tuple[GaussianParams, GaussianParams]:
        """
        Generate Gaussian pair for consecutive frames.
        
        Args:
            frame_t_data: Dict with images, c2ws, fxfycxcys for frame t
            frame_t1_data: Dict with images, c2ws, fxfycxcys for frame t+1
            
        Returns:
            (G_t, G_{t+1}) tuple of GaussianParams
        """
        G_t = self.generate_gaussians(
            images=frame_t_data["images"],
            c2ws=frame_t_data["c2ws"],
            fxfycxcys=frame_t_data["fxfycxcys"],
        )
        
        G_t1 = self.generate_gaussians(
            images=frame_t1_data["images"],
            c2ws=frame_t1_data["c2ws"],
            fxfycxcys=frame_t1_data["fxfycxcys"],
        )
        
        return G_t, G_t1


class GaussianCache:
    """
    Cache for pre-computed Gaussians to speed up training.
    
    Since GS-LRM is expensive, we pre-compute and cache all Gaussians
    for the training set before starting deformation training.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize Gaussian cache.
        
        Args:
            cache_dir: Directory to save/load cached Gaussians (optional)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._memory_cache: Dict[int, GaussianParams] = {}
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def has(self, frame_idx: int) -> bool:
        """Check if frame is cached."""
        if frame_idx in self._memory_cache:
            return True
        if self.cache_dir:
            return (self.cache_dir / f"frame_{frame_idx:06d}.pt").exists()
        return False
    
    def get(self, frame_idx: int) -> Optional[GaussianParams]:
        """Get cached Gaussian for frame."""
        # Try memory cache first
        if frame_idx in self._memory_cache:
            return self._memory_cache[frame_idx]
        
        # Try disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"frame_{frame_idx:06d}.pt"
            if cache_file.exists():
                data = torch.load(cache_file, weights_only=False)
                params = GaussianParams(
                    xyz=data["xyz"],
                    features=data["features"],
                    scaling=data["scaling"],
                    rotation=data["rotation"],
                    opacity=data["opacity"],
                )
                self._memory_cache[frame_idx] = params
                return params
        
        return None
    
    def put(self, frame_idx: int, params: GaussianParams, save_to_disk: bool = True):
        """Cache Gaussian for frame."""
        # Store in memory
        self._memory_cache[frame_idx] = params.detach()
        
        # Save to disk
        if save_to_disk and self.cache_dir:
            cache_file = self.cache_dir / f"frame_{frame_idx:06d}.pt"
            data = {
                "xyz": params.xyz.cpu(),
                "features": params.features.cpu(),
                "scaling": params.scaling.cpu(),
                "rotation": params.rotation.cpu(),
                "opacity": params.opacity.cpu(),
            }
            torch.save(data, cache_file)
    
    def clear_memory(self):
        """Clear memory cache (disk cache remains)."""
        self._memory_cache.clear()
    
    def __len__(self) -> int:
        """Number of cached frames."""
        if self.cache_dir:
            return len(list(self.cache_dir.glob("frame_*.pt")))
        return len(self._memory_cache)


# ============================================================
# Unit Tests
# ============================================================

def _test_gslrm_integration():
    """Unit test for GS-LRM integration (mock test without actual model)."""
    import tempfile
    
    print("Testing GS-LRM Integration...")
    
    # Test 1: GaussianCache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = GaussianCache(cache_dir=tmpdir)
        
        # Create mock GaussianParams
        N = 100
        params = GaussianParams(
            xyz=torch.randn(N, 3),
            features=torch.randn(N, 27),
            scaling=torch.randn(N, 3),
            rotation=torch.randn(N, 4),
            opacity=torch.randn(N, 1),
        )
        
        # Test put/get
        assert not cache.has(0), "Should not have frame 0 yet"
        cache.put(0, params)
        assert cache.has(0), "Should have frame 0 now"
        
        retrieved = cache.get(0)
        assert retrieved is not None, "Should retrieve frame 0"
        assert torch.allclose(params.xyz, retrieved.xyz), "xyz should match"
        print("  GaussianCache: ✓")
        
        # Test disk persistence
        cache2 = GaussianCache(cache_dir=tmpdir)
        assert cache2.has(0), "Should load from disk"
        retrieved2 = cache2.get(0)
        assert torch.allclose(params.xyz, retrieved2.xyz), "Should match original"
        print("  Disk persistence: ✓")
        
        print(f"  Cache size: {len(cache)} frames")
    
    print("All tests passed! ✓")
    print("\nNote: Full GSLRMGaussianGenerator test requires actual model.")
    return True


if __name__ == "__main__":
    _test_gslrm_integration()
