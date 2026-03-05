# Phase 2: Training Data Loading

> **Navigation**: [← Hub](../PIPELINE_DEEP_DIVE.md) | [Prev: PH1](PH1_PREPROCESSING.md) | [Next: PH3 →](PH3_MODEL_FORWARD_LOSS.md)
>
> **핵심 파일**: `gslrm/data/mouse_dataset.py`

---

### 2.1 Entry Point

**File**: `train_gslrm.py:157-187`

```python
# train_gslrm.py:167-175
def load_datasets(self):
    use_mouse_dataset = self.config.get("mouse", {}).get("use_mouse_dataset", False)

    if use_mouse_dataset:
        from gslrm.data.mouse_dataset import MouseViewDataset
        self.dataset = MouseViewDataset(self.config, split="train")
        self.val_dataset = MouseViewDataset(self.config, split="val")
```

### 2.2 Dataset Loading Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Dataset Initialization                                  │
├─────────────────────────────────────────────────────────────────┤
│ File: mouse_dataset.py:49-130                                   │
│                                                                 │
│ class MouseViewDataset(Dataset):                                │
│     def __init__(self, config, split="train"):                  │
│         # :60-70 - Load sample list from txt                    │
│         self.samples = self._load_sample_list(                  │
│             config.training.dataset.dataset_path)               │
│         # :72-85 - View configuration                           │
│         self.num_views = config.training.dataset.num_views      │
│         self.num_input_views = ...num_input_views               │
│         self.random_view_selection = ...random_view_selection    │
│         # :87-95 - Camera normalization                         │
│         self.target_camera_distance = config.mouse              │
│                                        .target_camera_distance  │
│         self.normalize_to_z_up = config.mouse.normalize_to_z_up │
│         # :97-105 - Mask settings                               │
│         self.auto_generate_mask = config.mouse.auto_generate_mask│
│         self.mask_threshold = config.mouse.get("mask_threshold")│
│         # :107-115 - Camera exclusion                           │
│         self.exclude_camera_indices = ...exclude_camera_indices  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: View Selection                                          │
├─────────────────────────────────────────────────────────────────┤
│ File: mouse_dataset.py:218-260                                  │
│                                                                 │
│ def _select_views(self, total_views: int):                      │
│     all_indices = list(range(total_views))                      │
│     # Apply camera exclusion                                    │
│     if self.exclude_camera_indices:                             │
│         all_indices = [i for i in all_indices                   │
│                        if i not in self.exclude_camera_indices] │
│     # Random or fixed selection                                 │
│     if self.random_view_selection and self.split == "train":    │
│         input_indices = sorted(                                 │
│             random.sample(all_indices, self.num_input_views))   │
│     else:                                                       │
│         input_indices = all_indices[:self.num_input_views]      │
│     remaining = [i for i in all_indices                         │
│                  if i not in input_indices]                     │
│     target_indices = input_indices + remaining                  │
│     return input_indices, target_indices                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Load Images and Cameras                                 │
├─────────────────────────────────────────────────────────────────┤
│ File: mouse_dataset.py:270-350                                  │
│                                                                 │
│ def __getitem__(self, idx):                                     │
│     sample_path = self.samples[idx]                             │
│     # :280-290 - Load camera JSON                               │
│     with open(sample_path / "opencv_cameras.json") as f:        │
│         camera_data = json.load(f)                              │
│     # :295-310 - Load images per view                           │
│     images = []                                                 │
│     for i in target_indices:                                    │
│         img = Image.open(sample_path / f"{i}.png").convert("RGB")│
│         images.append(pil_to_np(img.resize((512, 512))))        │
│     # :315-330 - Extract camera parameters                      │
│     c2ws, intrinsics = [], []                                   │
│     for i in target_indices:                                    │
│         frame = camera_data["frames"][i]                        │
│         c2w = np.array(frame["c2w"])           # [4, 4]         │
│         K = np.array(frame["K"])               # [fx,fy,cx,cy]  │
│         c2ws.append(c2w); intrinsics.append(K)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Camera Normalization (Runtime)                          │
├─────────────────────────────────────────────────────────────────┤
│ File: mouse_dataset.py:335-380                                  │
│ Uses: mouse_extensions/data/preprocessing.py:52-140             │
│                                                                 │
│     if self.normalize_to_z_up:                                  │
│         c2ws = normalize_cameras_to_z_up(c2ws)                  │
│     if self.target_camera_distance > 0:                         │
│         c2ws, intrinsics =                                      │
│             normalize_camera_distance_with_intrinsics(           │
│                 c2ws, intrinsics, self.target_camera_distance)  │
│         # ★ fx, fy also scaled together                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Auto Mask Generation (Optional)                         │
├─────────────────────────────────────────────────────────────────┤
│ File: mouse_dataset.py:385-420                                  │
│                                                                 │
│     if self.auto_generate_mask:                                 │
│         for img in images:                                      │
│             bg_mask = (img > self.mask_threshold).all(axis=-1)  │
│             fg_mask = ~bg_mask                                  │
│             masks.append(fg_mask.astype(np.float32))            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Return Batch Dictionary                                 │
├─────────────────────────────────────────────────────────────────┤
│ File: mouse_dataset.py:425-480                                  │
│                                                                 │
│     return {                                                    │
│         "image": images_tensor,         # [V, 3, H, W]          │
│         "c2w": c2ws_tensor,             # [V, 4, 4]             │
│         "fxfycxcy": fxfycxcy_tensor,    # [V, 4]                │
│         "index": indices,               # [V, 2]                │
│         "bg_color": bg_color,           # [3]                   │
│         "mask": masks_tensor,           # [V, 1, H, W] (opt.)   │
│     }                                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Config → Code Mapping

```yaml
# configs/base/gslrm_mouse.yaml + configs/datasets/M5.yaml

training:
  dataset:
    dataset_path: .../M5/data_mouse_train.txt  # → mouse_dataset.py:60
    num_views: 6                                # → mouse_dataset.py:72
    num_input_views: 5                          # → mouse_dataset.py:74
    random_view_selection: true                  # → mouse_dataset.py:235
    exclude_camera_indices: [5]                  # → mouse_dataset.py:228

mouse:
  use_mouse_dataset: true          # → train_gslrm.py:167
  target_camera_distance: 2.7      # → mouse_dataset.py:360
  normalize_to_z_up: true          # → mouse_dataset.py:340
  auto_generate_mask: true         # → mouse_dataset.py:390
  mask_threshold: 0.5              # → mouse_dataset.py:395
```

---

*← [PH1](PH1_PREPROCESSING.md) | [Hub](../PIPELINE_DEEP_DIVE.md) | [Next: PH3 →](PH3_MODEL_FORWARD_LOSS.md)*
