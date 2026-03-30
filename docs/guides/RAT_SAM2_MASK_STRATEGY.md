# Rat SAM2 Mask Annotation & Propagation Strategy

## Overview

Two rat datasets require SAM2 mask generation for GS-LRM training:

| Dataset | Type | Cameras | Total Frames | Existing Masks | Strategy |
|---------|------|:-------:|:------------:|:--------------:|----------|
| **SCN2A_WK1** (single) | Lone rat | 6 | 90,000 | 6×1024 (kp-guided) | Automated (kp_sam2_lone.py) |
| **M3_M4** (social) | Rat pair | 6 | 90,000 | Cam1: 11 ann + 408 prop | Manual → Video Propagation |

## Dataset 1: SCN2A_WK1 — RAT2 Expansion (Automated)

**No manual annotation needed.** Uses keypoint-guided per-frame segmentation.

### Current State (RAT1)
- 1024 frames × 6cam (step=50, range 0~51150)
- RAT1 FT: val PSNR 17.49 dB, overfitting 17.5 dB gap

### RAT2 Plan
```bash
# step=30 → ~3000 frames × 6cam = ~18,000 masks
ssh gpu03
cd /home/joon/dev/sdannce-poc
conda activate sdannce

CUDA_VISIBLE_DEVICES=6 python segmentation/kp_sam2_lone.py \
    --session_dir /home/joon/data/sdannce/rat/dataverse/SCN2A_WK1_2022_09_16_M1 \
    --output /home/joon/data/sdannce/rat/dataverse/SCN2A_WK1_2022_09_16_M1/sam2_masks_rat2 \
    --cameras 1,2,3,4,5,6 \
    --start 0 --end 89100 --step 30
```

**Time estimate:** ~1-2 hours (GPU 6, ~4GB VRAM)

---

## Dataset 2: M3_M4 Social Pair — Manual + Propagation

### Current State
- **Location:** `/home/joon/dev/sdannce-poc/data/2022_09_22_M3_M4/`
- **Video:** 6cam × 1920×1200 × 50fps × 30min = 90,000 frames
- **Keypoints:** `SDANNCE/bsl0.5_FM_rat{1,2}/save_data_AVG0.mat` (per-rat 3D keypoints)
- **Existing annotations:** Camera1 only, 11 keyframes (frames 0, 500, 1000, 5000, 20000, 20430, 20435, 20500, 45000, 60000, 80000)
- **Existing propagation:** Camera1 sparse_200 → 408 masks

### Annotation Strategy

#### Option A: Keypoint-Guided (Recommended First)

M3_M4 has per-rat SDANNCE keypoints → `kp_sam2_segment.py` can generate masks automatically using positive spine points + cross-negative prompts.

```bash
# Test on small range first
CUDA_VISIBLE_DEVICES=6 python segmentation/kp_sam2_segment.py \
    --start 0 --end 1000 --step 50 \
    --cameras 1,2,3,4,5,6 \
    --output /home/joon/dev/sdannce-poc/data/2022_09_22_M3_M4/kp_masks_test
```

**Pros:** 6cam simultaneous, no manual work, leverages existing keypoints
**Cons:** Quality depends on keypoint confidence, social occlusion may degrade

#### Option B: Manual Annotation + Video Propagation

For cameras/frames where kp-guided fails, use manual annotation → SAM2 propagation.

**Step 1: Manual Annotation (per camera)**

Target: **10-15 keyframes per camera**, spaced evenly + at behavior transitions.

```bash
# Launch annotator (SSH tunnel required for web UI)
ssh gpu03
cd /home/joon/dev/sdannce-poc
conda activate sdannce

# Option 1: Gradio annotator (single camera, interactive SAM2)
CUDA_VISIBLE_DEVICES=6 python segmentation/sam2_annotator.py --share

# Option 2: Multi-camera FastAPI viewer (6cam synchronized)
python viewers/mask_annotator.py \
    --session /home/joon/dev/sdannce-poc/data/2022_09_22_M3_M4 \
    --port 8770

# SSH tunnel from local Mac:
ssh -L 8770:localhost:8770 gpu03
# Open: http://localhost:8770
```

**Recommended keyframe selection (per camera):**
- Frames: 0, 5000, 10000, 15000, 20000, 30000, 45000, 60000, 75000, 85000
- Add 2-3 frames at behavior transitions (grooming, rearing, locomotion)
- Each frame takes ~2 min → **~25 min per camera, ~2.5 hours total**

**Step 2: Video Propagation**

```bash
# Per-camera propagation using keyframe-segments mode
CUDA_VISIBLE_DEVICES=6 python segmentation/sam2_propagate.py \
    --session /home/joon/dev/sdannce-poc/data/2022_09_22_M3_M4 \
    --cameras 1 \
    --mode keyframe-segments \
    --context 2000 \
    --model base_plus

# For all 6 cameras (after all annotated):
for cam in 1 2 3 4 5 6; do
    CUDA_VISIBLE_DEVICES=6 python segmentation/sam2_propagate.py \
        --session /home/joon/dev/sdannce-poc/data/2022_09_22_M3_M4 \
        --cameras $cam \
        --mode sparse --step 30 \
        --model base_plus
done
```

#### Recommended Approach: A → B Hybrid

1. **First:** Run kp_sam2_segment.py on full range (step=50, test quality)
2. **Evaluate:** Check mask quality on 5-10 sample frames per camera
3. **If quality OK** (>90% IoU): Use kp-guided for all, skip manual
4. **If quality poor** (<80% IoU): Manual annotate problem frames → propagate to fill gaps
5. **Final:** Merge kp-guided + propagated masks, keeping higher-quality per frame

---

## Data Paths Summary

```
/home/joon/data/sdannce/rat/dataverse/
└── SCN2A_WK1_2022_09_16_M1/
    ├── videos/Camera{1-6}/0.mp4          # Raw video
    ├── SDANNCE/bsl0.5_FM/                # Keypoints (single rat)
    ├── sam2_masks/Camera{1-6}/           # RAT1 masks (1024 per cam)
    ├── sam2_masks_rat2/Camera{1-6}/      # RAT2 masks (TODO)
    └── calibration/                       # Camera params

/home/joon/dev/sdannce-poc/data/
└── 2022_09_22_M3_M4/
    ├── videos/Camera{1-6}/0.mp4          # Raw video
    ├── SDANNCE/bsl0.5_FM_rat{1,2}/       # Per-rat keypoints
    ├── masks/Camera1/annotations/        # 11 manual keyframes
    ├── masks/Camera1/propagated/         # 408 propagated masks
    └── calibration/                       # Camera params
```

## Environment

```bash
# Server: gpu03
conda activate sdannce
# Python 3.x, torch 2.10.0+cu128, sam2 installed
# SAM2 checkpoint: /home/joon/dev/sdannce-poc/checkpoints/sam2/sam2.1_hiera_base_plus.pt
```

## Mask Format

```python
# NPZ with per-animal boolean masks
mask = np.load('mask_000123.npz')
mask['animal_0']  # shape=(1200, 1920), dtype=bool — Rat 1
mask['animal_1']  # shape=(1200, 1920), dtype=bool — Rat 2 (social only)
```

## Priority & Timeline

| Priority | Task | Time | Blocking? |
|:--------:|------|:----:|:---------:|
| **1** | SCN2A_WK1 RAT2 kp-guided (automated) | 1-2h | No |
| **2** | M3_M4 kp_sam2_segment.py test (automated) | 1h | No |
| **3** | Evaluate kp-guided quality → decide manual needs | 30min | 2 |
| **4a** | If OK: full kp-guided run | 2-3h | 3 |
| **4b** | If not: manual annotation + propagation | 3-4h | 3 |
| **5** | GS-LRM format conversion | 30min | 4 |
| **6** | Fine-tune training | 4-6h | 5 |

---

*Created: 2026-03-29 | FaceLift Rat SAM2 Strategy*
