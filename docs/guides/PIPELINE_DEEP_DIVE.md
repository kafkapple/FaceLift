# Pipeline Deep Dive: mouse_extensions Code Walkthrough

> **Navigation**: [← INDEX](../INDEX.md) | [EXPERIMENT_MASTER_GUIDE](EXPERIMENT_MASTER_GUIDE.md) | [CH1](chapters/CH1_ENVIRONMENT_AND_DATA.md) | [CH2](chapters/CH2_GSLRM_CODE_FLOW.md)
>
> **Related**: [KEYPOINT_3D_PIPELINE](../KEYPOINT_3D_PIPELINE.md) | [MVDIFFUSION_FINETUNE_GUIDE](MVDIFFUSION_FINETUNE_GUIDE.md)

**Created**: 2026-01-22
**Updated**: 2026-03-05
**Project**: FaceLift Mouse
**Server**: ssh gpu03 (`/home/joon/dev/FaceLift`)
**Status**: Reference Document (SSOT)

> **v2.0 Changes (2026-03-05)**: Phase 4-7 추가 (Pose Conditioning, E2E Inference, Fair Evaluation, 3D Keypoint Pipeline). Obsidian → 서버 SSOT 이전.

---

## Overview

이 문서는 FaceLift Mouse 프로젝트의 전체 파이프라인을 **실제 코드 파일과 라인 번호**와 함께 상세히 설명합니다.
모든 경로는 `/home/joon/dev/FaceLift/` 기준입니다.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         FACELIFT MOUSE PIPELINE                              │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬───────────┤
│ PHASE 1  │ PHASE 2  │ PHASE 3  │ PHASE 4  │ PHASE 5  │ PHASE 6  │ PHASE 7  │
│ Preproc  │ Training │ Forward  │ Pose     │ E2E      │ Fair     │ 3D KP    │
│          │ Data     │ & Loss   │ Cond.    │ Inference│ Eval     │ Pipeline │
│          │          │          │          │          │          │          │
│ Raw →    │ Dataset  │ GSLRM    │ Plucker  │ 1 img →  │ FL vs PS │ Render → │
│ M5 fmt   │ → Batch  │ Forward  │ Spatial  │ 6 views  │ metrics  │ Detect → │
│          │          │ → Loss   │ Token    │ → 3DGS   │          │ Triang.  │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴───────────┘
```

---

## Phase 1: Data Preprocessing

### 1.1 Entry Point

**File**: `mouse_extensions/preprocessing/preprocess.py`
**Preset**: M5 (Affine, fx=549, cx=cy=256, dist=2.7)

```bash
cd /home/joon/dev/FaceLift
python -m mouse_extensions.preprocessing.preprocess \
    --preset M5 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M5
```

### 1.2 Preprocessing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Load Raw Data                                           │
├─────────────────────────────────────────────────────────────────┤
│ File: preprocess.py:158-180                                     │
│                                                                 │
│ def _load_cameras(self) -> Dict[int, List[CameraParams]]:       │
│     cameras_path = self.input_dir / "cameras.pkl"               │
│     with open(cameras_path, "rb") as f:                         │
│         raw_cameras = pickle.load(f)                            │
│                                                                 │
│ Input:                                                          │
│   - cameras.pkl: 6 views × N frames                             │
│   - Each camera: K (3×3), R (3×3), t (3×1)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Load Preset Configuration                               │
├─────────────────────────────────────────────────────────────────┤
│ File: presets.py:52-89                                          │
│                                                                 │
│ PRESETS = {                                                     │
│     "M5": {                                                     │
│         "paradigm": "affine",                                   │
│         "transform": "affine",                                  │
│         "skew_correction": False,                               │
│         "target_fx": 548.9937744140625,   # GS-LRM pretrained   │
│         "target_pp": (256, 256),          # Principal Point      │
│         "output_size": (512, 512),                              │
│         "camera_normalization": {                               │
│             "enabled": True,                                    │
│             "target_distance": 2.7,       # Z-distance norm     │
│             "z_up_alignment": True,                             │
│         },                                                      │
│     },                                                          │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Compute Homography Transform                            │
├─────────────────────────────────────────────────────────────────┤
│ File: preprocess.py:220-260                                     │
│                                                                 │
│ def compute_homography_transform(self, K: np.ndarray) -> H:     │
│     """                                                         │
│     Homography: H = K_target @ K_orig^-1                        │
│                                                                 │
│     K_orig (fx=844, skew=0.4°) → K_target (fx=549, skew=0°)    │
│     """                                                         │
│     K_target = np.array([                                       │
│         [cfg.target_fx, 0, cfg.target_pp[0]],                   │
│         [0, cfg.target_fx, cfg.target_pp[1]],                   │
│         [0, 0, 1]                                               │
│     ])                                                          │
│     H = K_target @ np.linalg.inv(K)                             │
│     return H                                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Apply Transform to Images                               │
├─────────────────────────────────────────────────────────────────┤
│ File: preprocess.py:262-300                                     │
│                                                                 │
│ def apply_transform(self, image: np.ndarray, H: np.ndarray):    │
│     warped = cv2.warpPerspective(                               │
│         image, H,                                               │
│         (self.cfg.output_size[1], self.cfg.output_size[0]),     │
│         flags=cv2.INTER_LINEAR,                                 │
│         borderMode=cv2.BORDER_CONSTANT,                         │
│         borderValue=(255, 255, 255)  # White background         │
│     )                                                           │
│     return warped                                               │
│                                                                 │
│ Input : 1152×1024 raw image                                     │
│ Output: 512×512 normalized image                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Camera Normalization                                    │
├─────────────────────────────────────────────────────────────────┤
│ File: camera_normalizer.py:45-120                               │
│                                                                 │
│ class CameraNormalizer:                                         │
│     def normalize(self, c2w, intrinsics):                       │
│         # 1. Z-up alignment                                     │
│         if self.z_up_alignment:                                 │
│             c2w = self._align_to_z_up(c2w)                      │
│         # 2. Distance normalization                             │
│         current_distance = np.linalg.norm(c2w[:3, 3])           │
│         scale = self.target_distance / current_distance         │
│         c2w[:3, 3] *= scale                                     │
│         # 3. Intrinsics scaling                                 │
│         intrinsics[0] *= scale  # fx                            │
│         intrinsics[1] *= scale  # fy                            │
│         return c2w, intrinsics                                  │
│                                                                 │
│ Result: translation_norm ~237 → 2.7, fx 844 → 548.99           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Save Output                                             │
├─────────────────────────────────────────────────────────────────┤
│ File: preprocess.py:320-380                                     │
│                                                                 │
│ Output Structure:                                               │
│ M5/                                                             │
│ ├── sample_000000/                                              │
│ │   ├── 0.png, 1.png, ..., 5.png  (512×512 RGBA)               │
│ │   └── opencv_cameras.json                                     │
│ │       {                                                       │
│ │         "frames": [                                           │
│ │           {                                                   │
│ │             "file_path": "0.png",                             │
│ │             "K": [548.99, 548.99, 256.0, 256.0],              │
│ │             "c2w": [[...], [...], [...], [...]]  // 4×4       │
│ │           }, ...                                              │
│ │         ]                                                     │
│ │       }                                                       │
│ ├── sample_000001/ ...                                          │
│ └── data_mouse_t2_train.txt  (absolute paths)                   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Parameters by Preset (M-Series)

| Preset | Paradigm | fx | cx, cy | Skew | Distance |
|--------|----------|-----|--------|------|----------|
| **M5** | **affine** | **549** | **256, 256** | ❌ | **2.7** |
| M5h | homography | 549 | 256, 256 | ✅ | 2.7 |
| M5h_1 | homo+global zoom | 549 | 256, 256 | ✅ | 2.7 |
| M5h_2 | homo+per-sample zoom | 549 | 256, 256 | ✅ | 2.7 |

---

## Phase 2: Training Data Loading

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

## Phase 3: Model Forward & Loss Computation

### 3.1 Forward Pass Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ File: gslrm/model/gslrm.py:1526-1800                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ batch = {image: [B,6,3,512,512], c2w: [B,6,4,4], ...}          │
│                              │                                  │
│   1. Data Split (:1545)      input 5 / target 6                 │
│   2. Plucker Coords (:1560)  [B, 5, 9, H, W] = RGB(3)+Plk(6)  │
│   3. Patch Tokenize (:1575)  [B, V×P, D], P=(512/16)²=1024     │
│   4. Transformer (:1595)     24 layers, 16 heads                │
│   5. Gaussian Gen (:1615)    xyz, features, scale, rot, opacity │
│   6. Render (:820-913)       [B, V, 3, H, W] + alphas           │
│   7. Loss (:343-530)         L2 + Perceptual + SSIM + BG + Alpha│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Loss Computation Details

**File**: `gslrm/model/gslrm.py:343-530`, `mouse_extensions/model/loss_extensions.py`

#### 3.2.1 Mask Computation (`loss_extensions.py:113-260`)

```python
# compute_mask_from_config() — Priority:
# mask_mode (explicit) > use_rendered_alpha_mask > use_predicted_mask > gt > none

# MaskType enum (L17-23):
#   NONE / GT / RGB_PRED / ALPHA

# mask_mode="alpha" (L159-215):
mask = (rendered_alpha > alpha_threshold).float()  # default threshold=0.5
# min_mask_ratio safety: adaptive threshold if coverage too low

# mask_mode="rgb_pred" (L220-232): DEPRECATED
# White-BG assumption, mouse data에서 IoU ~0.06 → 비효과적
```

#### 3.2.2 Alpha Loss (`loss_extensions.py:430-493`)

```python
# compute_alpha_loss(rendered_alpha, gt_alpha)
# Input: rendered_alpha [B*V, 1, H, W], gt_alpha [B*V, 1, H, W]
#
# Loss types:
#   "mse"   (L463):  F.mse_loss
#   "bce"   (L466):  ★ AMP 비호환 → autocast(enabled=False) 강제
#   "dice"  (L473):  1 - 2*intersection / (sum_pred + sum_gt)
#   "focal" (L480):  (1-p_t)^gamma * bce, fg/bg 불균형 대응
```

#### 3.2.3 Opacity Regularization (`loss_extensions.py:619-663`)

```python
# compute_opacity_regularization(opacity) — Floater artifact 감소
# Input: opacity [N] or [B, N], range [0, 1]
# ★ bf16 호환: opacity.float().clamp(1e-4, 1-1e-4)
#
# Types:
#   "entropy":   -p*log(p) - (1-p)*log(1-p)  (p=0,1 최소)
#   "l1_sparse": |mean - target_sparsity|
#   "l2_binary": min(p, 1-p)^2
```

#### 3.2.4 Mask IoU (`loss_extensions.py:264-315`)

```python
# compute_mask_iou(rendering, gt_mask) → scalar IoU
# intersection = (pred_mask * gt_mask).sum()
# union = ((pred_mask + gt_mask) > 0.5).float().sum()
# iou = intersection / union.clamp(min=1.0)
```

---

## Phase 4: Pose Conditioning (MVDiffusion)

> **Why**: MVDiffusion은 6개 뷰의 상대적 카메라 위치를 implicit하게 학습하지만,
> 명시적 카메라 포즈를 주입하면 multi-view consistency가 향상될 수 있다.
> 3가지 인코딩 방식(Spherical, Extrinsic, Plucker)을 구현하여 비교 실험.

### 4.1 Pose Encoding Architectures

**File**: `mouse_extensions/model/pose_conditioning.py`

```
┌─────────────────────────────────────────────────────────────────┐
│ 3 Encoder Architectures                                         │
├──────────────────┬──────────────────┬───────────────────────────┤
│ SphericalPose    │ ExtrinsicPose    │ PluckerRay                │
│ (L44-114)        │ (L117-175)       │ (L178-291)               │
├──────────────────┼──────────────────┼───────────────────────────┤
│ c2w → spherical  │ c2w → R,t direct │ c2w + K → pixel rays     │
│ (az, el, dist)   │ 6D rotation repr │ direction + moment        │
│ ↓                │ ↓                │ ↓                         │
│ Fourier encode   │ flatten + MLP    │ 2-layer Conv2d 1×1        │
│ (3×128=384D)     │ (9D or 12D)      │ (6 → 128 → 320)          │
│ ↓                │ ↓                │ ↓                         │
│ MLP → 1024D      │ MLP → 1024D      │ [B, 320, H, W] spatial   │
│                  │                  │                           │
│ ★ Global token   │ ★ Global token   │ ★ Spatial features        │
│ [B, N, 1024]     │ [B, N, 1024]     │ Per-pixel geometry        │
└──────────────────┴──────────────────┴───────────────────────────┘
```

#### PluckerRay 상세 (`pose_conditioning.py:218-291`)

```python
# compute_plucker_coordinates(c2w, intrinsics, height, width)
#
# 1. Pixel grid 생성 (L250-256)
#    meshgrid(0..H-1, 0..W-1) → [H, W, 2]
#
# 2. Camera space normalization (L264-266)
#    x = (u - cx) / fx,  y = (v - cy) / fy,  z = 1.0
#
# 3. World space ray direction (L273-277)
#    direction = einsum('bij,hwj->bhwi', R, ray_cam)  # [B, H, W, 3]
#    direction = normalize(direction)
#
# 4. Camera origin in world (L280)
#    origin = c2w[:, :3, 3]  # [B, 3]
#
# 5. Plucker moment (L283)
#    moment = cross(origin, direction)  # [B, H, W, 3]
#
# 6. Concat + rearrange (L286-289)
#    plucker = cat([direction, moment])  # [B, H, W, 6] → [B, 6, H, W]
```

### 4.2 Injection into MVDiffusion UNet

**File**: `mouse_extensions/model/pose_conditioning_integration.py`

#### PoseConditioningInjector (L113-542) — 핵심 클래스

```
┌─────────────────────────────────────────────────────────────────┐
│ Injection Modes (L114-136)                                      │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│ "concat"     │ "add"        │ "replace"    │ "spatial_token"    │
│ prompt seq에  │ 첫 token에   │ 마지막 token │ Plucker 공간 정보   │
│ extra token  │ 더하기       │ 교체         │ 64개 토큰 추가      │
│ 추가          │              │              │                    │
│ H3 실험       │ H6a 실험     │ 미사용       │ H7/H7v2 실험       │
├──────────────┴──────────────┴──────────────┴────────────────────┤
│                                                                 │
│ UNet Input:  encoder_hidden_states [B*N, seq_len, 1024]         │
│                              │                                  │
│       ┌──────────────────────┼────────────────────────┐         │
│       ▼                      ▼                        ▼         │
│   inject()              inject()               inject_spatial() │
│   (concat/replace)      (add mode)             (spatial_token)  │
│       │                      │                        │         │
│       ▼                      ▼                        ▼         │
│ [B*N, seq+1, 1024]    [B*N, seq, 1024]    [B*N, seq+64, 1024]  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Spatial Token 상세 (H7/H7v2, L264-304)

```python
# _compute_spatial_tokens() — Plucker rays → spatial token sequence
#
# Plucker spatial features: [N, 320, H, W]
#                              │
# AdaptiveAvgPool2d(S=8) → [N, 320, 8, 8]
#                              │
# flatten + transpose    → [N, 64, 320]      (S*S = 64 tokens)
#                              │
# plucker_spatial_linear  → [N, 64, 1024]     ★ Zero-init (ControlNet)
#
# Why zero-init (L204-205):
#   초기 학습 시 spatial tokens가 0을 출력 → 기존 모델 동작 보존
#   학습 진행에 따라 점진적으로 공간 정보 기여 → 안정적 fine-tuning
```

#### Trainable vs Frozen Mode (L212-217, L327-334)

```
trainable=True  + training=True   → gradient 흐름, cache 무효화
trainable=True  + training=False  → no_grad (검증/추론)
trainable=False + any             → 항상 no_grad (레거시 frozen)

★ pose_injector weights are saved with MVDiffusion checkpoint
  (Bug fix 2026-02-27: 이전에는 누락되어 추론 시 random weights 사용됨)
```

#### M5 Camera Utilities (L39-106)

```python
# load_m5_cameras(json_path) — M5 rig 정의 로드
# Output: {c2w: [6, 4, 4], w2c: [6, 4, 4], intrinsics: [6, 4], n_views: 6}

# get_rotated_cameras(c2w, ref_view_idx, n_views)
# Random reference view augmentation: 카메라 순서를 ref 기준으로 회전
# indices = [(ref + i) % n for i in range(n)]
```

### 4.3 Config & Commands

```yaml
# configs/experiments/pose_conditioning config:
pose_conditioning:
  method: "plucker"           # spherical / extrinsic / plucker
  injection_mode: "add"       # concat / add / replace_last / spatial_token
  embed_dim: 1024
  trainable: true
  # Plucker-specific:
  plucker_hidden_dim: 128
  plucker_spatial_dim: 320
  spatial_token_size: 8       # H7: 8×8 = 64 tokens
```

```bash
# MVDiffusion fine-tuning with pose conditioning
CUDA_VISIBLE_DEVICES=5 accelerate launch train_diffusion.py \
    --config configs/mvdiffusion/mouse_M5t2.yaml \
    --pose_config configs/experiments/H6a_v2_plucker_add.yaml
```

---

## Phase 5: E2E Inference Pipeline

> **Why**: GS-LRM은 GT 6-view로 PSNR 23.84를 달성하지만,
> 실제 deployment에서는 단일 이미지만 가용. MVDiffusion이 6-view를 생성하고
> GS-LRM이 3D로 재구성하는 End-to-End 파이프라인이 필요.

### 5.1 Pipeline Architecture

**File**: `mouse_extensions/inference/end_to_end.py`

```
┌─────────────────────────────────────────────────────────────────┐
│ EndToEndPipeline (L22-307)                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Input: Single image (or M5 sample)                              │
│                              │                                  │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ Stage 0 (Optional): SAM Preprocessing (L147-171)         │   │
│ │   SAM detection → white BG → center align → normalize    │   │
│ └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ Stage 1: MVDiffusion (L174-189)                          │   │
│ │   Input : [1, 3, H, W] reference image                   │   │
│ │   Model : SD2.1-UnCLIP + Era3D RMA                       │   │
│ │   Output: [6, 3, H, W] multi-view images                 │   │
│ │   ★ Pose conditioning injected here (if configured)      │   │
│ └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ Stage 2: GS-LRM (L191-216)                               │   │
│ │   Input : images [1, V, 3, H, W]                         │   │
│ │           c2ws   [1, V, 4, 4]     (from m5_cameras.json) │   │
│ │           fxfycxcys [1, V, 4]                             │   │
│ │           index  [1, V, 2]        (view_idx, scene_idx)   │   │
│ │   Output: 3D Gaussian parameters → PLY, video, RRD       │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│ run_from_views() (L228-307) — GS-LRM only mode                 │
│   View ablation (L265-290):                                     │
│     1v → [0], 2v → [0,3], 3v → [0,2,4] (evenly distributed)   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 CLI Entry Point

**File**: `mouse_extensions/scripts/inference/run_e2e_inference.py`

#### Path 분기 로직 (L312-317)

```python
use_mvdiffusion = (
    args.input_image is not None or
    (args.sample_dir is not None and args.input_view_idx is not None) or
    (args.data_dir is not None and args.input_view_idx is not None)
)
```

| Path | Trigger | 용도 |
|------|---------|------|
| **Path 1** | `--sample_dir` (no input_view_idx) | GS-LRM only (GT views) |
| **Path 2a** | `--input_image` | E2E from single image |
| **Path 2b** | `--data_dir + --input_view_idx` | E2E batch (test set) |
| **Batch Path 1** | `--data_dir` (no input_view_idx) | GS-LRM only batch |

#### Batch Path 2b 상세 (L482-518) — 가장 일반적인 사용

```
for each sample in data_dir/split.txt:
  1. Load reference image: sample/images/cam_{input_view_idx:03d}.png
  2. MVDiffusion inference → 6 views
  3. GS-LRM inference → Gaussians
  4. Render turntable + eval views
  5. Save to output_dir/samples/{sample_id}/
```

### 5.3 Commands

```bash
# ★ E2E Batch Inference (test set, MOST COMMON)
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt --input_view_idx 0 --skip_preprocess \
    --model M5t --gslrm_config configs/base/gslrm_mouse.yaml \
    --gslrm_checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt \
    --mvdiffusion_checkpoint <CHECKPOINT_PATH> \
    --mvdiffusion_base checkpoints/mvdiffusion/pipeckpts \
    --prompt_embed_path mvdiffusion/data/mouse_prompt_embeds_6view_1024 \
    --prefer_ema --no_turntable --no_mesh \
    --num_steps 50 --guidance_scale 3.0 --seed 42 \
    --output_dir outputs/phase3_e2e/<experiment_name>

# GS-LRM only (GT input, for upper bound)
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt --skip_preprocess \
    --model M5t --gslrm_config configs/base/gslrm_mouse.yaml \
    --gslrm_checkpoint <CHECKPOINT_PATH> \
    --no_turntable --no_mesh \
    --output_dir outputs/tier_comparison/<experiment_name>
```

### 5.4 E2E Verification Protocol

> ⚠️ **CRITICAL**: `--input_view_idx` 누락 시 GS-LRM only (Path 1)로 실행됨.
> PSNR_gt ~20 dB (GT input) vs ~8 dB (E2E) — 혼동 주의.

```bash
# 1. Pipeline Path 확인
grep 'Batch Path' LOG   # → "Path 2b: MVDiffusion" 확인

# 2. MVDiffusion 로딩 확인
grep 'Loading MVDiffusion' LOG   # → 존재해야 함

# 3. run_config.json 확인
python -c "import json; d=json.load(open('run_config.json')); print(d['args']['input_view_idx'])"
# → 0 이어야 함 (null이면 GS-LRM only)

# 4. Sanity check: E2E PSNR_fg 범위 7-10 dB. 12+ dB → GS-LRM only 오인

# 5. generated_views/ 디렉토리 존재 확인 (E2E에서만 생성)
ls outputs/.../samples/sample_003240/generated_views/
```

---

## Phase 6: Fair Evaluation

> **Why**: FaceLift (feed-forward) vs PoseSplatter (per-scene optimization)은
> 근본적으로 다른 모델 유형. 공정한 비교를 위해 5가지 공정성 보장이 필요.

### 6.1 Fairness Guarantees

```
1. Test-only:    학습 데이터 제외 (M5t2 test: frame 3240-3599)
2. GT alpha mask: 양측 모두 GT RGBA alpha로 FG/BG 분리
3. Unified metric: 동일 함수로 PSNR/SSIM/L1/IoU 계산
4. FG-only:      배경 제외, 전경 픽셀만 평가
5. Coverage-aware: pred_fg ∩ gt_fg 영역과 gt_fg 전체 영역 별도 추적
```

### 6.2 Metric Computation

**File**: `mouse_extensions/scripts/eval/fair_comparison.py`

#### compute_all_metrics (L156-225) — 핵심 함수

```python
# Input: pred [H,W,3], gt [H,W,3], gt_mask [H,W], pred_mask [H,W] (opt.)
#
# Output metrics dict:
#   psnr_gt_masked     — GT FG 영역의 PSNR (GT mask 기준)
#   psnr_intersection  — pred_fg ∩ gt_fg 영역의 PSNR (순수 색상 정확도)
#   ssim_gt_masked     — GT mask bbox crop 기반 SSIM
#   l1_gt_masked       — FG 픽셀 L1: sum|diff| / (3 * sum(mask))
#   iou                — 실루엣 binary IoU (pred_fg vs gt_fg)
#   coverage           — pred가 gt_fg를 얼마나 커버하는지 (recall)
#   pred_precision     — pred_fg 중 gt_fg와 겹치는 비율 (precision)
#   color_bias_r/g/b   — intersection에서 pred-gt 평균 색 편차
```

#### evaluate_facelift (L288-467) — 평가 루프

```
1. Render path 자동 감지 (L331-337):
   E2E: cam_000/render_view_NN.png
   GS-LRM standalone: render_view_NN.png

2. GT 로드 (L360-365): M5 RGBA → alpha > 127 → binary mask

3. 해상도 불일치 처리 (L368-376): 중앙 crop

4. Per-frame metrics 수집 → Overall 통계 (mean, std, median)

5. Output JSON: model, evaluation, config, fairness, overall, per_view
```

### 6.3 Visualization Grid (L232-281)

```
┌──────────┬──────────┬──────────┐
│ GT       │ Render   │ GT Mask  │
│ (white)  │ (white)  │ (gray)   │
├──────────┼──────────┼──────────┤
│ Mask     │ Error    │ Raw      │
│ Compare  │ Map (5×) │ Render   │
│ G=inter  │          │          │
│ R=GT only│          │          │
│ B=pred   │          │          │
└──────────┴──────────┴──────────┘
Filename: {frame}_view{NN}_psnr{X.X}_int{X.X}_iou{X.XX}_cov{XX%}.png
```

### 6.4 Commands

```bash
# FL 평가
python mouse_extensions/scripts/eval/fair_comparison.py evaluate_fl \
    --render_dir outputs/phase3_e2e/<experiment>/samples \
    --gt_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --views 1 2 3 4 5 \
    --output outputs/phase3_e2e/<experiment>/fair_eval.json \
    --save_vis outputs/phase3_e2e/<experiment>/vis --vis_every 10

# FL vs PS 비교 리포트
python mouse_extensions/scripts/eval/fair_comparison.py compare \
    --facelift outputs/.../fair_eval.json \
    --baseline outputs/.../ps_eval.json \
    --output_dir outputs/.../comparison/
```

---

## Phase 7: 3D Keypoint Pipeline

> **Why**: 3D Gaussian에서 행동 분석용 3D keypoint를 추출하려면
> novel view 렌더링 → 2D 검출 → 다시점 삼각측량 파이프라인이 필요.
> Oracle (perfect 2D) vs Neural (HRNet) 비교로 domain gap 정량화.

### 7.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase A-C: Oracle Saturation (상한선)                            │
│   Perfect 2D keypoints (GT 3D → project) + noise σ              │
│   → multi-view triangulation → MPJPE                            │
│   → "views × noise" saturation curve                            │
│                                                                 │
│ Result: σ=5px, 24views → MPJPE 1.52mm (upper bound)            │
├─────────────────────────────────────────────────────────────────┤
│ Phase D: Neural Detection                                       │
│                                                                 │
│   [render_novel_views_for_detection.py]  (facelift env)         │
│     GS-LRM + M5 test frames                                    │
│     → Gaussian predict → turntable N views                      │
│     → outputs/{N}views/{frame_id}/cam_*.png + cameras.json      │
│                              │                                  │
│   [detect_and_triangulate.py]  (mmpose env)                     │
│     HRNet-w48 2D keypoint detection (22 joints)                 │
│     → multi-view triangulation (DLT)                            │
│     → FaceLift coords → MAMMAL mm coords                        │
│     → MPJPE / PA-MPJPE vs GT 3D                                 │
│                                                                 │
│ Result: 24views → MPJPE 55.5mm, Det.Rate 29.1%                 │
├─────────────────────────────────────────────────────────────────┤
│ Phase E: Oracle vs Neural Comparison                            │
│   Gap = Neural / Oracle = 12~37× (domain gap 지배적)            │
│   Detection rate ~29% = primary bottleneck                      │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Novel View Rendering

**File**: `mouse_extensions/scripts/keypoint_detection/render_novel_views_for_detection.py`

#### render_frame (L134-228) — 단일 프레임 처리

```python
# 1. GS-LRM predict (L150-156)
#    load_sample_data() → 4 input views → gslrm_model.predict()
#    → Gaussian splat parameters

# 2. Gaussian filtering (L160-166)
#    opacity_thres=0.04, scaling_thres=0.1, floater_thres=0.6
#    crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0]

# 3. Turntable cameras (L168-172)
#    get_turntable_cameras_safe(num_views, render_size)
#    → c2ws [N, 4, 4], fxfycxcys [N, 4]

# 4. Render each view (L191-217)
#    render_single_view(gaussians, c2w, fxfycxcy, ...)
#    → RGBA PNG (cam_000.png, cam_001.png, ...)

# 5. Save cameras.json (L220-226)
#    cameras_to_serializable() → {c2w, w2c, K, fxfycxcy, P}
```

#### cameras_to_serializable (L95-131) — Projection matrix 생성

```python
# c2w (4×4) → w2c = inv(c2w)
# K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
# Rt = w2c[:3, :]  (3×4)
# P = K @ Rt        (3×4 projection matrix for triangulation)
```

### 7.3 Detection & Triangulation

**File**: `mouse_extensions/scripts/keypoint_detection/detect_and_triangulate.py`

> ⚠️ 이 스크립트는 **mmpose conda env**에서 실행해야 합니다 (facelift env 아님).

#### 좌표계 변환 상수 (L34-40)

```python
M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])  # mm
M5_DISTANCE_SCALE = 2.7 / 307.785  # = 0.008772

def facelift_to_mammal(points_3d):
    """FaceLift normalized coords → MAMMAL world mm."""
    return points_3d / M5_DISTANCE_SCALE + M5_SCENE_CENTER
```

#### detect_keypoints_single (L91-129)

```python
# Input: MMPose model, image path, bbox [x1,y1,x2,y2]
# Process: inference_topdown() → pred_instances.keypoints[0]
# Output: (22, 3) ndarray — last channel = confidence score
```

#### triangulate_and_eval (L173-226) — 핵심 알고리즘

```
Input: keypoints_2d (num_views, 22, 3), proj_matrices (num_views, 3, 4),
       gt_3d_mm (22, 3)

1. triangulate_batch()       → pred_3d_fl (FaceLift normalized space)
2. facelift_to_mammal()      → pred_3d_mm (millimeters)
3. compute_mpjpe()           → MPJPE (mm)
4. compute_pa_mpjpe()        → Procrustes-aligned MPJPE
5. per-joint detection rate  → conf > threshold인 뷰 비율

Output: {pred_3d_mm, mpjpe, mpjpe_std, per_joint_error,
         pa_mpjpe, detection_rates, mean_detection_rate}
```

#### 22-Joint Definition (L315-322)

```
L_ear, R_ear, nose, neck, body_middle, tail_root, tail_middle, tail_end,
L_paw, L_paw_end, L_elbow, L_shoulder,
R_paw, R_paw_end, R_elbow, R_shoulder,
L_foot, L_knee, L_hip, R_foot, R_knee, R_hip
```

### 7.4 Commands

```bash
# Step 1: Render novel views (facelift env)
conda activate facelift
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.keypoint_detection.render_novel_views_for_detection \
    --config configs/base/gslrm_mouse.yaml \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt \
    --data_root ~/data/preprocessed/FaceLift_mouse/M5 \
    --output_dir outputs/triangulation/neural_detection/renders \
    --num_views 6 12 24 --render_size 384

# Step 2: Detect + Triangulate (mmpose env)
conda activate mmpose
python -m mouse_extensions.scripts.keypoint_detection.detect_and_triangulate \
    --render_dir outputs/triangulation/neural_detection/renders/24views \
    --mmpose_config mouse_extensions/configs/mmpose/hrnet_w48_mouse_22kp.py \
    --mmpose_checkpoint /node_data/joon/checkpoints/mmpose/hrnet_w48_mouse_best.pth \
    --output_dir outputs/triangulation/neural_detection/results/24views \
    --conf_threshold 0.3

# Step 3: Oracle vs Neural comparison
python -m mouse_extensions.scripts.keypoint_detection.compare_oracle_vs_real \
    --oracle_path outputs/triangulation/oracle_saturation/saturation_analysis.json \
    --neural_dir outputs/triangulation/neural_detection/results \
    --output_dir outputs/triangulation/neural_detection/comparison

# Step 4: View-count comparison visualization
python -m mouse_extensions.scripts.keypoint_detection.viewcount_comparison_viz \
    --results_dir outputs/triangulation/neural_detection/results \
    --oracle_path outputs/triangulation/oracle_saturation/saturation_analysis.json \
    --output_dir outputs/triangulation/neural_detection/comparison/viewcount
```

---

## Potential Issues & Error Points

### ⚠️ Issue 1: Dataset Path Format

**Location**: `mouse_dataset.py:60-70`

```
# Bad:  relative paths in data_mouse_train.txt
# Good: absolute paths (/home/joon/data/.../sample_000000)
```

### ⚠️ Issue 2: Camera Normalization Mismatch

**Location**: `preprocess.py` vs `mouse_dataset.py`

```
Preprocessing (M5):     fx=548.99, distance→2.7
Runtime normalization:  distance→2.7 (fx도 스케일링)

★ Double normalization risk:
  - 전처리에서 이미 2.7로 정규화
  - 런타임에서 다시 정규화 시도
  - 해결: target_camera_distance=0 (비활성화) 또는 2.7 (idempotent)
```

### ⚠️ Issue 3: E2E Camera-Image Mismatch

**Location**: `mouse_extensions/inference/mvdiffusion_pipeline.py`

```
문제: MVDiffusion의 compute_cameras()가 합성 orbit 카메라 생성 (elevation=0°)
실제: M5 6대 카메라 elevation -75°~+81°, 비정규 azimuth
결과: GS-LRM render 100% 흰색 (카메라-이미지 불일치)

해결: cameras/m5_cameras.json에서 실제 카메라 로드 + --camera_json CLI
★ MVDiffusion은 카메라를 명시적으로 입력받지 않음 (prompt embedding 뷰 구분)
★ GS-LRM에 넘기는 카메라는 학습 시 실제 카메라와 일치 필수
```

### ⚠️ Issue 4: View 5 Elevation Outlier

```
View 0: +14.9°, View 1: +20.6°, View 2: +11.3°
View 3: +10.7°, View 4: +26.5°, View 5: +30.8° ⚠️ OUTLIER (z-score=1.54)

해결: E4_6 config로 View 5 제외 실험
  exclude_camera_indices: [5]
```

### ⚠️ Issue 5: bf16 + Log Operation NaN

**Location**: `loss_extensions.py:388`

```python
# Problem: entropy = -opacity * torch.log(opacity)  # bf16 NaN
# Solution:
opacity_f = opacity.float()
entropy = -opacity_f * torch.log(opacity_f.clamp(1e-4, 1-1e-4))
```

### ⚠️ Issue 6: E2E Path Confusion

**Location**: `run_e2e_inference.py:312-317`

```
★ --input_view_idx 누락 시 Path 1 (GS-LRM only)로 분기.
  → PSNR_gt ~20+ dB (GT input) → E2E로 오인하기 쉬움.
  → 반드시 run_config.json의 args.input_view_idx 확인.
```

### ⚠️ Issue 7: Pose Injector Weights 미저장 버그 (수정 완료)

```
2026-02-27 이전: pose_injector weights가 checkpoint에 저장되지 않음
  → 추론 시 random weights 사용 → E2E 성능 저하
수정: train_diffusion.py, mvdiffusion_pipeline.py에 save/load 추가
영향 파일: end_to_end.py, run_e2e_inference.py
```

### ⚠️ Issue 8: mmpose vs facelift Conda Env

```
render_novel_views_for_detection.py → facelift env (PyTorch + diff_gauss)
detect_and_triangulate.py           → mmpose env (mmpose + mmdet)

★ 두 env 혼용 시 import 에러. Step별로 env 전환 필수.
```

---

## Quick Reference Commands

### Preprocessing
```bash
python -m mouse_extensions.preprocessing.preprocess --preset M5 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M5
```

### GS-LRM Training
```bash
CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5 -e E0_1
```

### MVDiffusion Training
```bash
CUDA_VISIBLE_DEVICES=5 accelerate launch train_diffusion.py \
    --config configs/mvdiffusion/mouse_M5t2.yaml \
    --pose_config configs/experiments/H7v2_plucker_spatial.yaml
```

### E2E Inference (batch)
```bash
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt --input_view_idx 0 --skip_preprocess \
    --model M5t --gslrm_config configs/base/gslrm_mouse.yaml \
    --gslrm_checkpoint <GS-LRM_CKPT> --mvdiffusion_checkpoint <MVDIFF_CKPT> \
    --mvdiffusion_base checkpoints/mvdiffusion/pipeckpts \
    --prompt_embed_path mvdiffusion/data/mouse_prompt_embeds_6view_1024 \
    --prefer_ema --no_turntable --no_mesh \
    --num_steps 50 --guidance_scale 3.0 --seed 42 \
    --output_dir outputs/phase3_e2e/<name>
```

### Fair Evaluation
```bash
python mouse_extensions/scripts/eval/fair_comparison.py evaluate_fl \
    --render_dir outputs/phase3_e2e/<exp>/samples \
    --gt_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --output outputs/phase3_e2e/<exp>/fair_eval.json
```

### 3D Keypoint Pipeline
```bash
# Step 1 (facelift env): Render novel views
python -m mouse_extensions.scripts.keypoint_detection.render_novel_views_for_detection \
    --config configs/base/gslrm_mouse.yaml --checkpoint <CKPT> \
    --num_views 6 12 24 --output_dir outputs/triangulation/.../renders

# Step 2 (mmpose env): Detect + Triangulate
python -m mouse_extensions.scripts.keypoint_detection.detect_and_triangulate \
    --render_dir outputs/triangulation/.../renders/24views \
    --mmpose_config mouse_extensions/configs/mmpose/hrnet_w48_mouse_22kp.py \
    --mmpose_checkpoint <MMPOSE_CKPT> \
    --output_dir outputs/triangulation/.../results/24views
```

---

## File Summary

| File | Location | Lines | Purpose |
|------|----------|:-----:|---------|
| **Phase 1: Preprocessing** | | | |
| `preprocess.py` | `mouse_extensions/preprocessing/` | ~400 | Main preprocessing |
| `presets.py` | `mouse_extensions/preprocessing/` | ~150 | Preset definitions (M5, M5h) |
| `camera_normalizer.py` | `mouse_extensions/preprocessing/` | ~200 | Camera normalization |
| **Phase 2: Data Loading** | | | |
| `mouse_dataset.py` | `gslrm/data/` | ~560 | Training dataset class |
| `preprocessing.py` | `mouse_extensions/data/` | ~250 | Runtime camera utilities |
| **Phase 3: Model & Loss** | | | |
| `gslrm.py` | `gslrm/model/` | ~2300 | Main model (forward + loss) |
| `loss_extensions.py` | `mouse_extensions/model/` | ~830 | Mask, alpha, opacity, depth loss |
| `train_gslrm.py` | `/` | ~1100 | Training entry point |
| **Phase 4: Pose Conditioning** | | | |
| `pose_conditioning.py` | `mouse_extensions/model/` | ~360 | 3 encoder architectures |
| `pose_conditioning_integration.py` | `mouse_extensions/model/` | ~540 | UNet injector + M5 cameras |
| **Phase 5: E2E Inference** | | | |
| `end_to_end.py` | `mouse_extensions/inference/` | ~310 | E2E pipeline coordinator |
| `mvdiffusion_pipeline.py` | `mouse_extensions/inference/` | ~210 | MVDiffusion stage |
| `gslrm_pipeline.py` | `mouse_extensions/inference/` | ~430 | GS-LRM stage + outputs |
| `run_e2e_inference.py` | `mouse_extensions/scripts/inference/` | ~580 | CLI entry point |
| `m5_cameras.json` | `mouse_extensions/inference/cameras/` | - | M5 6-camera config |
| **Phase 6: Fair Evaluation** | | | |
| `fair_comparison.py` | `mouse_extensions/scripts/eval/` | ~760 | Fair comparison metrics |
| **Phase 7: 3D Keypoint** | | | |
| `render_novel_views_for_detection.py` | `mouse_extensions/scripts/keypoint_detection/` | ~300 | GS-LRM → turntable render |
| `detect_and_triangulate.py` | `mouse_extensions/scripts/keypoint_detection/` | ~400 | HRNet detect → triangulate |
| `compare_oracle_vs_real.py` | `mouse_extensions/scripts/keypoint_detection/` | ~350 | Oracle vs neural comparison |
| `viewcount_comparison_viz.py` | `mouse_extensions/scripts/keypoint_detection/` | ~380 | View-count comparison viz |

---

*FaceLift Mouse Project | Pipeline Deep Dive v2.0 | 2026-03-05*
