# Phase 1: Data Preprocessing

> **Navigation**: [← Hub](../PIPELINE_DEEP_DIVE.md) | [Next: PH2 →](PH2_DATA_LOADING.md)
>
> **핵심 파일**: `mouse_extensions/preprocessing/preprocess.py`

---

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

*← [Hub](../PIPELINE_DEEP_DIVE.md) | [Next: PH2 →](PH2_DATA_LOADING.md)*
