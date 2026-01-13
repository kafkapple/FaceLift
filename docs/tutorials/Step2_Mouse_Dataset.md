# Step 2: MouseViewDataset 구현

> FaceLift의 RandomViewDataset을 기반으로 Mouse 데이터용 Dataset 클래스를 구현합니다.

## 2.1 원본 RandomViewDataset 분석

### 원본 코드 위치
`gslrm/data/dataset.py`

### 핵심 특징
- 32개 뷰 중 랜덤하게 N개 선택
- Human face 데이터 (60° 간격 균일 배치)
- `opencv_cameras.json`에서 카메라 파라미터 로드

```python
# 원본 dataset.py 핵심 구조
class RandomViewDataset(Dataset):
    def __init__(self, config, split: str):
        self.num_views = config.training.dataset.num_views  # 8
        self.num_input_views = config.training.dataset.num_input_views  # 4
        
    def __getitem__(self, idx):
        # 32개 뷰 중 랜덤 샘플링
        view_indices = random.sample(range(32), self.num_views)
        # 이미지 로드, 카메라 파라미터 로드
        return {"image": images, "c2w": c2ws, "fxfycxcy": intrinsics, ...}
```

---

## 2.2 Mouse Dataset 요구사항

| 항목 | RandomViewDataset | MouseViewDataset |
|------|-------------------|------------------|
| 뷰 개수 | 32 | **6** (고정) |
| 뷰 선택 | 랜덤 샘플링 | **고정 순서** [0,1,2,3,4,5] |
| 카메라 거리 | 균일 | **뷰마다 다름** → 정규화 필요 |
| 좌표계 | Z-up | 변환 필요할 수 있음 |

### Why 고정 뷰 순서?

**문제**: 랜덤 뷰 순서 → Plücker 좌표 불일치 → 학습 불안정

```python
# Plücker 좌표: Ray direction + moment
# 뷰 순서가 바뀌면 ray 방향이 달라져서 모델이 혼란
plucker = (ray_direction, ray_origin × ray_direction)
```

**해결**: 항상 동일한 순서 [0,1,2,3,4,5]로 고정

---

## 2.3 MouseViewDataset 구현

### 파일 생성
`gslrm/data/mouse_dataset.py`

```python
# Copyright 2025 Adobe Inc.
# Modified for Mouse-FaceLift project

"""
MouseViewDataset: 6-view mouse multi-view dataset.

Key differences from RandomViewDataset:
- Fixed 6 views (not random sampling)
- Camera distance normalization
- PP-centered preprocessing assumed
"""

import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


def normalize_camera_distance(c2w_matrices: np.ndarray, target_distance: float = 2.7):
    """
    Normalize camera distances to a fixed radius from origin.
    
    FaceLift pretrained model expects cameras at distance=2.7.
    
    Args:
        c2w_matrices: Camera-to-world matrices [N, 4, 4]
        target_distance: Target camera distance (default: 2.7)
        
    Returns:
        Normalized c2w matrices [N, 4, 4]
    """
    normalized = c2w_matrices.copy()
    
    for i in range(len(normalized)):
        # Extract camera position
        cam_pos = normalized[i, :3, 3]
        current_dist = np.linalg.norm(cam_pos)
        
        if current_dist > 0:
            # Scale position to target distance
            scale = target_distance / current_dist
            normalized[i, :3, 3] = cam_pos * scale
            
    return normalized


class MouseViewDataset(Dataset):
    """
    Dataset for loading 6-view mouse images with fixed view order.
    
    Expected data structure:
        sample_dir/
        ├── images/
        │   ├── cam_000.png
        │   ├── cam_001.png
        │   └── ... (6 views)
        └── opencv_cameras.json
    """
    
    def __init__(self, config, split: str):
        super().__init__()
        self.config = config
        self.split = split
        
        # Load dataset paths
        if split == "train":
            dataset_path = config.training.dataset.dataset_path
        elif split == "val":
            dataset_path = config.validation.dataset_path
        else:
            raise ValueError(f"Unknown split: {split}")
            
        with open(dataset_path, 'r') as f:
            self.all_data_paths = f.read().strip().split("\n")
        self.all_data_paths = pd.array(
            [s for s in self.all_data_paths if len(s) > 0], dtype="string"
        )
        
        # Dataset config
        dataset_config = config.training.dataset
        self.num_views = dataset_config.get("num_views", 6)
        self.num_input_views = dataset_config.get("num_input_views", 1)
        self.target_has_input = dataset_config.get("target_has_input", True)
        self.bg_color = dataset_config.get("background_color", "white")
        self.image_size = config.model.image_tokenizer.image_size  # 512
        
        # Mouse-specific config
        mouse_config = config.get("mouse", {})
        self.normalize_cameras = mouse_config.get("normalize_cameras", False)
        self.target_camera_distance = mouse_config.get("target_camera_distance", 2.7)
        
    def __len__(self):
        return len(self.all_data_paths)
    
    def __getitem__(self, idx):
        sample_path = str(self.all_data_paths[idx])
        
        # Load camera parameters
        camera_file = f"{sample_path}/opencv_cameras.json"
        with open(camera_file, 'r') as f:
            cameras = json.load(f)
        
        # Load images and cameras (FIXED order: 0,1,2,3,4,5)
        images = []
        c2ws = []
        fxfycxcy = []
        
        for view_idx in range(self.num_views):
            # Load image
            img_path = f"{sample_path}/images/cam_{view_idx:03d}.png"
            img = Image.open(img_path)
            
            # Resize if needed
            if img.size != (self.image_size, self.image_size):
                img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            
            # Convert to tensor [C, H, W]
            img_np = np.array(img).astype(np.float32) / 255.0
            if img_np.ndim == 2:
                img_np = np.stack([img_np] * 3, axis=-1)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            images.append(img_tensor)
            
            # Load camera params
            cam_key = f"cam_{view_idx:03d}"
            cam = cameras[cam_key]
            
            # Extrinsics: w2c -> c2w
            w2c = np.array(cam["w2c"]).reshape(4, 4)
            c2w = np.linalg.inv(w2c)
            c2ws.append(c2w)
            
            # Intrinsics: [fx, fy, cx, cy]
            K = np.array(cam["K"]).reshape(3, 3)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            fxfycxcy.append([fx, fy, cx, cy])
        
        # Stack tensors
        images = torch.stack(images, dim=0)  # [V, C, H, W]
        c2ws = np.stack(c2ws, axis=0)  # [V, 4, 4]
        fxfycxcy = np.array(fxfycxcy)  # [V, 4]
        
        # Optional: Normalize camera distances
        if self.normalize_cameras and self.target_camera_distance > 0:
            c2ws = normalize_camera_distance(c2ws, self.target_camera_distance)
        
        # Convert to tensors
        c2ws = torch.from_numpy(c2ws).float()
        fxfycxcy = torch.from_numpy(fxfycxcy).float()
        
        # Background color
        if self.bg_color == "white":
            bg_color = torch.ones(3)
        elif self.bg_color == "black":
            bg_color = torch.zeros(3)
        else:
            bg_color = torch.ones(3) * 0.5  # gray
            
        return {
            "image": images,      # [V, C, H, W]
            "c2w": c2ws,          # [V, 4, 4]
            "fxfycxcy": fxfycxcy, # [V, 4]
            "bg_color": bg_color, # [3]
            "index": idx,
        }
```

---

## 2.4 train_gslrm.py 수정

### 변경 위치
`train_gslrm.py` 의 `load_datasets()` 메서드

### 추가할 코드

```python
def load_datasets(self):
    """Load training and validation datasets."""
    
    # NEW: Check if mouse dataset should be used
    use_mouse_dataset = self.config.get("mouse", {}).get("use_mouse_dataset", False)
    
    if use_mouse_dataset:
        from gslrm.data.mouse_dataset import MouseViewDataset
        print("Using MouseViewDataset with camera normalization")
        self.dataset = MouseViewDataset(self.config, split="train")
        if self.config.validation.enabled:
            self.val_dataset = MouseViewDataset(self.config, split="val")
        else:
            self.val_dataset = None
    else:
        # Original: use RandomViewDataset
        from gslrm.data.dataset import RandomViewDataset
        self.dataset = RandomViewDataset(self.config, split="train")
        if self.config.validation.enabled:
            self.val_dataset = RandomViewDataset(self.config, split="val")
        else:
            self.val_dataset = None
    
    self._log_dataset_examples()
    self._setup_dataloaders()
```

---

## 2.5 테스트

```python
# 테스트 스크립트
from gslrm.data.mouse_dataset import MouseViewDataset
from easydict import EasyDict as edict
import yaml

# Config 로드
with open("configs/mouse/gslrm.yaml") as f:
    config = edict(yaml.safe_load(f))

# Dataset 생성
dataset = MouseViewDataset(config, split="train")

# 샘플 확인
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")  # [6, 4, 512, 512]
print(f"C2W shape: {sample['c2w'].shape}")      # [6, 4, 4]
print(f"Intrinsics shape: {sample['fxfycxcy'].shape}")  # [6, 4]
```

---

## 다음 단계

✅ MouseViewDataset 구현 완료

→ [Step3: 전처리 스크립트](./Step3_Preprocessing.md)

---

*Created: 2026-01-13*
