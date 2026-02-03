# Copyright 2026 FaceLift Mouse Extensions
# TemporalDataset: Dataset for loading sequential frames for deformation training

"""
TemporalDataset: Load consecutive frame pairs/sequences for deformation learning.

Training paradigm:
- Load frame t and frame t+1
- GS-LRM generates G_t and G_{t+1} independently
- Deformation network learns: G_t -> G'_{t+1} ≈ G_{t+1}
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

import numpy as np


@dataclass
class TemporalDatasetConfig:
    """Configuration for temporal dataset."""
    
    # Data paths
    data_dir: str = ""
    split_file: Optional[str] = None  # data_mouse_train.txt
    
    # Sequence settings
    sequence_length: int = 2  # Number of consecutive frames
    frame_stride: int = 1     # Stride between frames (1 = consecutive)
    
    # Sampling
    num_views: int = 6
    
    # Augmentation
    random_start: bool = True  # Random start frame in sequence
    
    # Cache
    cache_gaussians: bool = False  # Cache pre-computed Gaussians


class TemporalFrameDataset(Dataset):
    """
    Dataset that returns consecutive frame pairs for deformation training.
    
    Each sample contains:
    - frame_t: Data for frame at time t
    - frame_t1: Data for frame at time t+1
    - frame_indices: (t, t+1) tuple
    """
    
    def __init__(self, config: TemporalDatasetConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        
        # Load frame list
        self.frames = self._load_frame_list()
        
        # Build temporal pairs/sequences
        self.sequences = self._build_sequences()
        
        # Gaussian cache (optional)
        self._gaussian_cache: Dict[int, torch.Tensor] = {}
    
    def _load_frame_list(self) -> List[str]:
        """Load list of available frames."""
        if self.config.split_file:
            split_path = Path(self.config.split_file)
            if not split_path.is_absolute():
                split_path = self.data_dir / split_path
            
            with open(split_path, "r") as f:
                frames = [line.strip() for line in f if line.strip()]
            return frames
        
        # Fallback: scan directory
        frame_dirs = sorted(self.data_dir.glob("frame_*"))
        return [d.name for d in frame_dirs]
    
    def _build_sequences(self) -> List[Tuple[int, ...]]:
        """Build list of valid frame sequences."""
        T = len(self.frames)
        seq_len = self.config.sequence_length
        stride = self.config.frame_stride
        
        sequences = []
        
        # Slide window over frames
        for start in range(T):
            # Build sequence indices
            seq = []
            for i in range(seq_len):
                idx = start + i * stride
                if idx >= T:
                    break
                seq.append(idx)
            
            # Only keep complete sequences
            if len(seq) == seq_len:
                sequences.append(tuple(seq))
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a temporal sequence.
        
        Returns:
            dict with:
                - frame_indices: tuple of frame indices
                - frame_paths: tuple of frame paths
                - (images, cameras loaded by collate_fn or external loader)
        """
        seq_indices = self.sequences[idx]
        
        # Get frame paths
        frame_paths = tuple(
            str(self.data_dir / self.frames[i])
            for i in seq_indices
        )
        
        return {
            "frame_indices": seq_indices,
            "frame_paths": frame_paths,
            "sequence_idx": idx,
        }
    
    def get_frame_info(self, frame_idx: int) -> Dict:
        """Get info for a specific frame."""
        frame_name = self.frames[frame_idx]
        frame_path = self.data_dir / frame_name
        
        return {
            "frame_idx": frame_idx,
            "frame_name": frame_name,
            "frame_path": str(frame_path),
        }
    
    def cache_gaussian(self, frame_idx: int, gaussian_tensor: torch.Tensor):
        """Cache pre-computed Gaussian for a frame."""
        if self.config.cache_gaussians:
            self._gaussian_cache[frame_idx] = gaussian_tensor.cpu()
    
    def get_cached_gaussian(self, frame_idx: int) -> Optional[torch.Tensor]:
        """Get cached Gaussian if available."""
        return self._gaussian_cache.get(frame_idx)
    
    def clear_cache(self):
        """Clear Gaussian cache."""
        self._gaussian_cache.clear()


class TemporalPairDataset(TemporalFrameDataset):
    """
    Simplified dataset that always returns pairs (t, t+1).
    Convenience wrapper for the common 2-frame case.
    """
    
    def __init__(self, config: TemporalDatasetConfig):
        # Force sequence_length to 2
        config.sequence_length = 2
        super().__init__(config)
    
    def __getitem__(self, idx: int) -> Dict:
        result = super().__getitem__(idx)
        
        # Unpack for convenience
        t, t1 = result["frame_indices"]
        path_t, path_t1 = result["frame_paths"]
        
        result.update({
            "t": t,
            "t1": t1,
            "path_t": path_t,
            "path_t1": path_t1,
        })
        
        return result


# ============================================================
# Unit Tests
# ============================================================

def _test_temporal_dataset():
    """Unit test for temporal dataset."""
    import tempfile
    import os
    
    print("Testing TemporalDataset...")
    
    # Create temp directory with mock frames
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock frame directories
        num_frames = 10
        for i in range(num_frames):
            frame_dir = Path(tmpdir) / f"frame_{i:04d}"
            frame_dir.mkdir()
            # Create dummy file
            (frame_dir / "data.txt").write_text(f"frame {i}")
        
        print(f"  Created {num_frames} mock frames")
        
        # Test 1: Basic loading
        config = TemporalDatasetConfig(
            data_dir=tmpdir,
            sequence_length=2,
            frame_stride=1,
        )
        dataset = TemporalFrameDataset(config)
        
        assert len(dataset.frames) == num_frames
        assert len(dataset.sequences) == num_frames - 1  # n-1 pairs
        print(f"  Loaded: {len(dataset.frames)} frames, {len(dataset.sequences)} sequences")
        
        # Test 2: Get item
        item = dataset[0]
        assert "frame_indices" in item
        assert item["frame_indices"] == (0, 1)
        print(f"  Item 0: indices={item['frame_indices']}")
        
        # Test 3: Sequence length > 2
        config3 = TemporalDatasetConfig(
            data_dir=tmpdir,
            sequence_length=3,
            frame_stride=1,
        )
        dataset3 = TemporalFrameDataset(config3)
        assert len(dataset3.sequences) == num_frames - 2  # n-2 triplets
        assert dataset3[0]["frame_indices"] == (0, 1, 2)
        print(f"  Sequence length 3: {len(dataset3.sequences)} sequences")
        
        # Test 4: Frame stride > 1
        config4 = TemporalDatasetConfig(
            data_dir=tmpdir,
            sequence_length=2,
            frame_stride=2,
        )
        dataset4 = TemporalFrameDataset(config4)
        assert dataset4[0]["frame_indices"] == (0, 2)
        print(f"  Stride 2: first pair = {dataset4[0]['frame_indices']}")
        
        # Test 5: Pair dataset
        pair_config = TemporalDatasetConfig(data_dir=tmpdir)
        pair_dataset = TemporalPairDataset(pair_config)
        item_pair = pair_dataset[0]
        assert "t" in item_pair and "t1" in item_pair
        assert item_pair["t"] == 0 and item_pair["t1"] == 1
        print(f"  Pair dataset: t={item_pair['t']}, t1={item_pair['t1']}")
        
        # Test 6: Gaussian caching
        config.cache_gaussians = True
        dataset_cache = TemporalFrameDataset(config)
        fake_gaussian = torch.randn(100, 38)
        dataset_cache.cache_gaussian(0, fake_gaussian)
        cached = dataset_cache.get_cached_gaussian(0)
        assert cached is not None
        assert torch.equal(cached, fake_gaussian)
        print("  Gaussian caching: ✓")
    
    print("All tests passed! ✓")
    return True


if __name__ == "__main__":
    _test_temporal_dataset()
