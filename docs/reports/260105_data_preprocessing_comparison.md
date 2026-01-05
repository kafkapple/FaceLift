# Data Preprocessing Comparison Report

**Date:** 2025-01-05  
**Servers:** joon, gpu03

## 1. Data Folders Overview

### joon Server
| Folder | Samples | Date |
|--------|---------|------|
| data_mouse_centered | 3,597 | Dec 31 |
| data_mouse_uniform | 3,597 | Jan 4 |

### gpu03 Server
| Folder | Samples | Date |
|--------|---------|------|
| data_mouse | 2,000 | Dec 14 |
| data_mouse_pixel_based | 2,000 | Dec 20 |
| data_mouse_real_camera | 3,597 | Dec 21 |
| data_mouse_real_pixel | 3,597 | Dec 21 |
| data_mouse_synthetic_v4 | 14 | Dec 21 |

---

## 2. Preprocessing Script Mapping

| Script | Output Folder | Key Method |
|--------|---------------|------------|
| `preprocess_center_align_all_views.py` | data_mouse_centered | Bbox center alignment |
| `preprocess_uniform_scale.py` | data_mouse_uniform | Uniform scale across views |
| `preprocess_pixel_based.py` | data_mouse_pixel_based | Center of mass + pixel scale |
| `generate_synthetic_data.py` | data_mouse_synthetic_v4 | MVDiffusion generation |

---

## 3. Camera Parameters Comparison

| Feature | centered | uniform | pixel_based | real_camera |
|---------|----------|---------|-------------|-------------|
| **fx/fy** | Original (varies) | Original (varies) | **Normalized (548.99)** | Original (varies) |
| **cx/cy** | Original (varies) | Original (varies) | **Normalized (256.0)** | Original (varies) |
| **Translation** | Original (large) | Original (large) | **Normalized (small)** | Original (large) |
| **Scale Info** | None | uniform_scale | pixel_based | None |

### Example Values (cam_000)
| Parameter | centered | uniform | pixel_based |
|-----------|----------|---------|-------------|
| fx | 725.47 | 721.97 | 548.99 |
| fy | 819.65 | 815.70 | 548.99 |
| cx | 267.27 | 222.73 | 256.00 |
| cy | 245.61 | 252.33 | 256.00 |
| t_x | 10.34 | 10.34 | 0.08 |
| t_y | 66.41 | 66.41 | 0.54 |
| t_z | 236.70 | 236.70 | 1.93 |

---

## 4. Image Properties Comparison

| Feature | centered | uniform | pixel_based | real_camera |
|---------|----------|---------|-------------|-------------|
| **Resolution** | 512x512 | 512x512 | 512x512 | 512x512 |
| **Channels** | RGBA | RGBA | RGBA | **RGB** |
| **Avg File Size** | ~47KB | ~45KB | **~250KB** | ~27KB |
| **Background** | Removed | Removed | Removed | **Not removed** |

---

## 5. Preprocessing Method Details

### 5.1 Center Align (data_mouse_centered)
- **Script:** `preprocess_center_align_all_views.py`
- **Method:** Bounding box center alignment
- **Features:**
  - Finds object bbox in each view
  - Translates image to center the object
  - Keeps original camera intrinsics
  - Background removed (alpha channel)

### 5.2 Uniform Scale (data_mouse_uniform)
- **Script:** `preprocess_uniform_scale.py`
- **Method:** Sample-wise uniform scaling
- **Features:**
  - target_ratio: 0.6 (object occupies 60% of image)
  - safe_margin: 0.05
  - Applies same scale to all views in a sample
  - Updates camera intrinsics accordingly
  - Stores `uniform_scale_info` in JSON

### 5.3 Pixel Based (data_mouse_pixel_based)
- **Script:** `preprocess_pixel_based.py`
- **Method:** Center of mass + pixel scale normalization
- **Features:**
  - target_size_ratio: 0.3
  - output_size: 512
  - **Normalizes camera intrinsics (fx=fy, cx=cy=center)**
  - **Normalizes camera translation to small values**
  - Per-view scale factors based on object size
  - Stores detailed `pixel_based_preprocessing` stats

### 5.4 Real Camera (data_mouse_real_camera)
- **Source:** Original markerless mouse data
- **Features:**
  - Original camera parameters preserved
  - **No background removal (RGB only)**
  - Used as baseline for comparison

### 5.5 Synthetic V4 (data_mouse_synthetic_v4)
- **Script:** `generate_synthetic_data.py`
- **Method:** MVDiffusion model inference
- **Features:**
  - Uses normalized cameras (same as pixel_based)
  - Generated from trained MVDiffusion checkpoint
  - Only 14 samples (test set)

---

## 6. Failed Preprocessing

### data_mouse_real_pixel
- **Attempted:** pixel_based preprocessing on real_camera data
- **Result:** Failed - "no_alpha_channel"
- **Reason:** Real camera images are RGB only, pixel_based requires alpha for segmentation

---

## 7. Recommendations

1. **For GS-LRM Training:**
   - Use `data_mouse_pixel_based` or `data_mouse_uniform` (normalized cameras)
   - Avoid `data_mouse_real_camera` (no background removal)

2. **For MVDiffusion Training:**
   - Use `data_mouse_centered` or `data_mouse_uniform`
   - These preserve original camera intrinsics

3. **Missing Data:**
   - gpu03 needs uniform/centered versions (only has pixel_based)
   - joon needs pixel_based version

---

## 8. Summary Table

| Dataset | Samples | BG Removal | Camera Norm | Scale Norm | Server |
|---------|---------|------------|-------------|------------|--------|
| data_mouse_centered | 3,597 | ✅ | ❌ | ❌ | joon |
| data_mouse_uniform | 3,597 | ✅ | ❌ | ✅ | joon |
| data_mouse_pixel_based | 2,000 | ✅ | ✅ | ✅ | gpu03 |
| data_mouse_real_camera | 3,597 | ❌ | ❌ | ❌ | gpu03 |
| data_mouse_synthetic_v4 | 14 | ✅ | ✅ | ✅ | gpu03 |

