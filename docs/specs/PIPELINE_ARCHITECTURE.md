# FaceLift Pipeline Architecture

> **Navigation**: [Index](../INDEX.md) | [Experiment Registry](../experiments/EXPERIMENT_REGISTRY.md)
> **SSOT**: 2-stage pipeline (MV-Diffusion + GS-LRM) architecture, I/O, loss, config
> **Last Updated**: 2026-02-13
> **명칭 규약**: Stage 1 = "MV-Diffusion" (SD2.1-UnCLIP + Era3D RMA). "MVDiffusion"은 Tang et al. (NeurIPS 2023) 별개 논문.
> `mvdiffusion/` 폴더명은 Era3D 코드 구조명이며, Tang et al. 무관. 상세: [MULTIVIEW_DIFFUSION_THEORY](MULTIVIEW_DIFFUSION_THEORY.md)

---

## 1. Pipeline Overview

```
                        ┌─────────────────────────────────────────────────┐
                        │               FaceLift Pipeline                  │
                        │                                                  │
  Input Image(s)        │   Stage 1: MV-Diffusion    Stage 2: GS-LRM       │    Output
  (1-6 views)    ──────►│   (Multi-view Generation) (3D Reconstruction)  ──────► 3D Gaussians
  512x512 RGB           │   SD 2.1-unclip based     Transformer-based     │    + Rendered Views
                        │                                                  │
                        └─────────────────────────────────────────────────┘
```

### 1.1 Stage Summary

| | Stage 1: MV-Diffusion | Stage 2: GS-LRM |
|---|---|---|
| **Task** | Multi-view image generation | 3D Gaussian reconstruction |
| **Input** | 1 reference image (512x512) | 4-6 multi-view images + cameras |
| **Output** | 6 consistent multi-view images | 3D Gaussian splats (xyz, SH, scale, rot, opacity) |
| **Base Model** | Stable Diffusion 2.1 Unclip | Custom Transformer (24L, 1024d) |
| **Training** | Diffusion denoising loss (MSE) | L2 + Perceptual + optional SSIM/LPIPS |
| **Inference** | DDIM/DDPM sampling | Single forward pass |

---

## 2. Stage 1: MV-Diffusion (Multi-view Generation)

### 2.1 Architecture

```
Reference Image ──► CLIP Image Encoder ──► Image Embedding (1024-dim)
                                                │
                                                ▼
                         ┌──────────────────────────────────────┐
                         │    Stable Diffusion 2.1 Unclip UNet  │
                         │    + Multi-view Cross-Attention       │
                         │    + Sparse MV Attention              │
                         │    + Camera Embeddings                │
                         │    + Custom Self-Attention Blocks     │
                         └──────────────────────────────────────┘
                                                │
                                                ▼
                         6 Denoised Latents ──► VAE Decoder ──► 6 Views (512x512)
```

### 2.2 UNet Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `unclip` | true | SD 2.1 unclip variant (image-conditioned) |
| `num_views` | 6 | Multi-view output |
| `sample_size` | 64 | Latent size (512/8) |
| `zero_init_conv_in` | true | Zero-initialized input conv |
| `multiview_attention` | true | Cross-view attention |
| `sparse_mv_attention` | true | Sparse multi-view attention |
| `selfattn_block` | custom | Custom self-attention implementation |
| `cd_attention_last` | false | Cross-domain attention placement |
| `cd_attention_mid` | false | |
| `addition_downsample` | false | No extra downsampling |

### 2.3 Training Configuration

| Parameter | Original (FaceLift) | Mouse (M5t2) | Notes |
|-----------|-------------------|--------------|-------|
| **Base Model** | SD 2.1 Unclip | SD 2.1 Unclip | Same |
| **Prompt Embedding** | CLIP text (1024-dim) | Pre-computed mouse prompt (1024-dim) | `clr_embeds.pt` |
| **Batch Size** | 8 | 4 | GPU memory limited |
| **Gradient Accum** | 2 | 4 | Effective BS: 16 both |
| **Learning Rate** | 1e-4 | 5e-5 | Mouse: more conservative |
| **LR Scheduler** | piecewise_constant | piecewise_constant | Step rule: 1:N,0.5 |
| **LR Warmup** | 10 steps | 100 steps | Mouse: longer warmup |
| **Max Steps** | 20,000 | 10,000 | Mouse: less data |
| **Optimizer** | AdamW | AdamW | Same |
| **Adam betas** | (0.9, 0.999) | (0.9, 0.999) | Same |
| **Weight Decay** | 1e-2 | 1e-2 | Same |
| **Max Grad Norm** | 1.0 | 1.0 | Same |
| **Mixed Precision** | fp16 | fp16 | Same |
| **EMA** | true | true | Same |
| **CFG Drop Rate** | 0.05 | 0.05 | Same |
| **CFG Drop Type** | drop_as_a_whole | drop_as_a_whole | Same |
| **SNR Gamma** | 5.0 | 5.0 | Min-SNR weighting |
| **xformers** | true | true | Memory efficient attention |
| **Checkpoint Every** | 5,000 | 1,000 | Mouse: more frequent |
| **Checkpoints Limit** | 20 | 2 | Mouse: disk space |
| **Val Steps** | 1,000 | 200 | Mouse: more frequent val |
| **Val Guidance** | [1.0, 3.0] | [1.0, 3.0] | Same |
| **Num Workers** | 32 | 8 | Server dependent |

### 2.4 Loss Function

MV-Diffusion uses the standard diffusion denoising loss with Min-SNR reweighting:

```
L_MVDiff = E_t [ w(t) · ||ε_θ(x_t, t, c) - ε||² ]

where:
  ε_θ(x_t, t, c)  = UNet noise prediction, conditioned on c = {CLIP embedding, prompt embeds}
  ε                = ground-truth noise added at timestep t
  x_t              = noised latent: x_t = α_t · x_0 + σ_t · ε

  SNR(t) = α_t² / σ_t²
  w(t)   = min(SNR(t), γ) / SNR(t),   γ = 5.0   (Min-SNR weighting)

  When SNR(t) ≤ γ:  w(t) = 1.0            (low-noise, keep full weight)
  When SNR(t) > γ:  w(t) = γ / SNR(t) < 1  (high-noise, reduce weight)
```

**Effect**: Min-SNR weighting downweights loss at high-noise (low SNR) timesteps where
the signal is dominated by noise. This stabilizes training and improves convergence.

| Component | Weight | Description |
|-----------|--------|-------------|
| **MSE (denoising)** | 1.0 | Noise prediction error |
| **Min-SNR weighting** | gamma=5.0 | Timestep reweighting |
| **Perceptual / LPIPS** | — | Not used (same as original) |

### 2.5 Data Augmentation

| Augmentation | Original FaceLift | Mouse (All variants) | Notes |
|-------------|-------------------|---------------------|-------|
| `bg_color` | "three_choices" | "three_choices" | Same (random white/black/gray) |
| `augmentation` | *(key absent, default=false)* | **true** | Mouse-added |
| `aug_brightness` | *(none)* | **[0.9, 1.1]** | Mouse-added |
| `aug_contrast` | *(none)* | **[0.9, 1.1]** | Mouse-added |
| `aug_hflip` | *(none)* | **false** | Mouse is asymmetric |

**Note**: Augmentation settings are **identical across all mouse configs** (M5t, M5t2, all experimental variants).
Augmentation does not increase effective sample count; the same 2,880 (M5t2) or 1,198 (M5t) samples are jittered differently each epoch.

---

## 3. Stage 2: GS-LRM (3D Gaussian Reconstruction)

### 3.1 Architecture

```
Multi-view Images (4-6) ──► Patch Tokenizer ──► Image Tokens
       [B,V,3,512,512]        (patch=8)       [B, V*4096, 1024]
                                                    │
Camera Poses ──► Plucker Coordinates ──►            │
  [B,V,4,4]     [B,V,6,H,W]             ┌──────────┘
       │                                 │
       ▼                                 ▼
  ┌────────────────────────────────────────────────────┐
  │              Transformer (24 layers)                │
  │              d=1024, d_head=64                      │
  │              Image Tokens + Gaussian Tokens         │
  │              (with gradient checkpointing)          │
  └────────────────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
    Gaussian Upsampler          Pixel-Aligned Decoder
    (n_gaussians=2 tokens)      (per-patch Gaussians)
            │                         │
            └──────────┬──────────────┘
                       ▼
              All Gaussian Params
              (xyz, SH, scale, rot, opacity)
                       │
                       ▼
              Gaussian Splatting Renderer
              (deferred rendering for training)
                       │
                       ▼
              Rendered Images [B, V_target, 3, H, W]
              Rendered Alpha  [B, V_target, 1, H, W]
```

### 3.2 Model Configuration

| Parameter | Original (FaceLift) | Mouse | Notes |
|-----------|-------------------|-------|-------|
| **Image Size** | 512 | 512 | Same |
| **Patch Size** | 8 | 8 | Same → 64x64 patches |
| **Input Channels** | 9 | 9 | 3 RGB + 3 direction + 3 reference |
| **Transformer d** | 1024 | 1024 | Same |
| **Transformer d_head** | 64 | 64 | Same → 16 heads |
| **Transformer layers** | 24 | 24 | Same |
| **n_gaussians** | 2 | 2 | Learnable tokens |
| **SH Degree** | 0 | 0 | DC only (no view-dependent color) |
| **Upsample Factor** | 1 | 1 | No upsampling |
| **Num Views (total)** | 8 | 6 | Target rendering views |
| **Num Input Views** | 6 (train: 4) | 4 | Encoder input views |
| **hard_pixelalign** | true | true | Pixel-aligned Gaussians |
| **clip_xyz** | true | true | Clip positions to [-1,1]^3 |
| **use_custom_plucker** | true | true | Custom Plucker coordinates |

### 3.3 Gaussian Output Parameters

Each Gaussian has **11 parameters** (with SH degree 0):

| Parameter | Dim | Activation | Description |
|-----------|-----|-----------|-------------|
| **xyz** | 3 | raw (clipped to [-1,1]^3) | 3D position |
| **features** | 3 | raw → SH DC component | Color (RGB via SH) |
| **scaling** | 3 | `exp(x - 2.3)`, clamped max=-1.20 | Anisotropic scale |
| **rotation** | 4 | raw quaternion | Orientation |
| **opacity** | 1 | `sigmoid(x - 2.0)` | Transparency |

**Total Gaussians per sample**: `n_gaussians(2) + V * (H/patch)^2 = 2 + 4*64^2 = 16,386`

### 3.4 Training Configuration

| Parameter | Original (FaceLift) | Mouse (E0_1) | Notes |
|-----------|-------------------|--------------|-------|
| **Optimizer** | AdamW | AdamW | Same |
| **Learning Rate** | 1e-4 | 1e-6 | Mouse: finetuning (100x lower) |
| **Adam betas** | (0.9, 0.95) | (0.9, 0.95) | Same |
| **Weight Decay** | 0.05 | 0.05 | Same |
| **Grad Clip Norm** | 1.0 | 50.0 | Mouse: more lenient |
| **Grad Accum Steps** | 1 | 1 | Same |
| **Batch Size/GPU** | 2 | 2 | Same |
| **Mixed Precision** | bf16 | bf16 | Same |
| **TF32** | true | true | Same |
| **Warmup Steps** | 500 | 500 | Same |
| **L2 Warmup** | 500 | 500 | Same (perceptual off during warmup) |
| **Max Steps** | 20,000 | 15,000 | Mouse: slightly less |
| **Checkpoint Every** | 5,000 | 100 | Mouse: more frequent |
| **Val Every** | 5,000 | 200 | Mouse: more frequent |
| **Early Stop Patience** | - | 10 | 10 val cycles (~2000 steps) |
| **Grad Checkpoint** | every 1 layer | every 1 layer | Same |
| **Pretrained Checkpoint** | stage_2 | `ckpt_0000000000021125.pt` | Finetuning from pretrained |
| **Reset LR** | false | true | Mouse: restart LR schedule |
| **target_has_input** | true | true | Target views include input views |
| **random_view_selection** | - | true (train only) | Random input view sampling |

### 3.5 Loss Functions

```
L_total = w_l2 · L_l2  +  w_perc · L_perceptual  +  w_lpips · L_lpips
        + w_ssim · L_ssim  +  w_pa · L_pixelalign  +  w_pd · L_pointsdist
        + w_alpha · L_alpha  +  w_bg · L_bg

where:
  L_l2         = (1/N) Σ ||I_render - I_gt||²           (pixel-wise MSE)
  L_perceptual = Σ_l ||φ_l(I_render) - φ_l(I_gt)||²     (VGG feature matching, layers l)
  L_lpips      = Σ_l w_l · ||φ_l(I_render) - φ_l(I_gt)||²  (learned perceptual distance)
  L_ssim       = 1 - SSIM(I_render, I_gt)                (structural similarity)
  L_alpha      = (1/N) Σ ||α_render - α_gt||²            (alpha mask supervision)
  L_bg         = (1/N) Σ ||I_render · (1 - α_gt)||²      (background penalty)

Default weights (Original FaceLift = Mouse E0_1):
  w_l2 = 1.0,  w_perc = 0.5,  w_lpips = 0.0,  w_ssim = 0.0
  w_pa = 0.0,  w_pd = 0.0,    w_alpha = 0.0,   w_bg = 0.0

  → L_total = 1.0 · L_l2 + 0.5 · L_perceptual

L2 Warmup (step < 500):
  w_perc → 0.0,  w_lpips → 0.0    (only L_l2 active for stable initial convergence)
```

| Loss | Weight (Original) | Weight (Mouse E0_1) | Description |
|------|-------------------|---------------------|-------------|
| **L2 (MSE)** | 1.0 | 1.0 | Pixel-wise reconstruction |
| **Perceptual (VGG)** | 0.5 | 0.5 | Feature-level texture quality |
| **LPIPS** | 0.0 | 0.0 | Learned perceptual distance |
| **SSIM** | 0.0 | 0.0 | Structural similarity |
| **Pixel Align** | 0.0 | 0.0 | Ray-Gaussian alignment |
| **Points Dist** | 0.0 | 0.0 | Depth regularization |
| **Alpha (LGM)** | 0.0 | 0.0 | Alpha mask supervision |
| **Background** | 0.0 | 0.0 | Background penalty |

### 3.6 Available but Unused Losses (for ablation)

| Loss | Config Key | When Used |
|------|-----------|-----------|
| `alpha_loss_weight` | `training.losses.alpha_loss_weight` | H6 experiments (alpha mask supervision) |
| `bg_loss_weight` | `training.losses.bg_loss_weight` | H6 experiments (background penalty) |
| `ssim_loss_weight` | `training.losses.ssim_loss_weight` | H7 experiments |
| `mask_mode` | `training.losses.mask_mode` | `none`/`gt`/`alpha` — controls masking for L2/perceptual |
| `masked_l2_loss` | `training.losses.masked_l2_loss` | Foreground-only L2 |
| `masked_perceptual_loss` | `training.losses.masked_perceptual_loss` | Gray-out background for VGG |

---

## 4. Data Pipeline

### 4.1 Raw Data

| Property | Value |
|----------|-------|
| **Source** | DANNCE `markerless_mouse_1` (Dunn et al., Nature Methods, 2021) |
| **Preprocessed by** | MAMMAL_mouse (An Liang, Nature Comms, 2023) → `markerless_mouse_1_nerf` |
| **Cameras** | 6 synchronized views |
| **Raw Resolution** | 1152 x 1024 (undistorted) → 512 x 512 (after M5 preprocessing) |
| **Raw FPS** | **100 fps** |
| **Recording Duration** | **180 sec (3 min)** |
| **Total Raw Frames** | 18,000 (per camera) |
| **Sampling Interval** | 5 (→ effective **20 fps**) |
| **Total Samples** | 3,600 (18,000 / 5) |
| **Server Path** | `/home/joon/data/raw/markerless_mouse_1_nerf/` |

**Data provenance**:
- DANNCE demo originally provides 1,000 frames for inference testing; full sequence is 18,000 frames
- MAMMAL_mouse preprocesses `markerless_mouse_1`: undistort, extract 2D keypoints, silhouettes
- `_nerf` suffix is a local naming convention (not from any external project)
- FaceLift and MAMMAL_mouse **share the same raw data** on server (MAMMAL uses symlink)

**vs PoseSplatter dataset**: PoseSplatter uses a **separate** 30-min DANNCE-style recording
(54,000 frames/cam, 30fps, 1536×2048, SAM2 masks). Different recording from `markerless_mouse_1`.

### 4.2 Preprocessing (M5 Series)

```
Raw Video (6 cams) ──► Frame Extraction (interval=5)
                       ──► Affine Transform (target_fx=549)
                       ──► PP Centering (cx=cy=256)
                       ──► Batch Uniform Camera Normalization (avg_dist=2.7)
                       ──► Split into Train/Val/Test
                       ──► Save as {frame_id}/images/cam_{000-005}.png + opencv_cameras.json
```

| Step | Description |
|------|-------------|
| **Affine Transform** | Scale intrinsics so fx ≈ 549 (pretrained expectation) |
| **PP Centering** | Shift principal point to cx=cy=256 (image center) |
| **Batch Uniform Norm** | Recenter 6 cameras to origin, scale avg distance to 2.7 |
| **Z-up Convention** | Normalize to Z-up coordinate system |

### 4.3 Dataset Splits (M5 Series)

| Dataset | Total | Train | Val | Test | Split Strategy | Shuffle |
|---------|-------|-------|-----|------|---------------|---------|
| **M5** | 3,600 | 2,880 | 360 | 360 | Temporal 80:10:10 | No (consecutive) |
| **M5t** | 3,600 | 1,198 | 1,198 | 1,204 | Temporal 1:1:1 | No (consecutive) |
| **M5t2** | 3,600 | 2,880 | 360 | 360 | Temporal 80:10:10 | No (consecutive) |

**Key differences:**
- **M5 vs M5t2**: Same split ratio (80:10:10), same preprocessing (M5 series), different text file locations. M5t2 is the canonical version with `_t2_` suffix in filenames.
- **M5t**: Pose Splatter compatible 1:1:1 split. Less training data (1,198 vs 2,880).
- **All temporal**: No random shuffle. Consecutive frames stay in same split → no data leakage.
- **Shuffling**: None at split level. `random_view_selection: true` in training randomly selects which 4 of 6 views are input.

| Dataset | Frame Range (Train) | Frame Range (Val) | Frame Range (Test) |
|---------|-------------------|------------------|-------------------|
| **M5** | 000000-002879 | 002880-003239 | 003240-003599 |
| **M5t** | 000000-001197 | 001198-002395 | 002396-003599 |
| **M5t2** | 000000-002879 | 002880-003239 | 003240-003599 |

### 4.4 Per-sample Structure

```
{frame_id}/
├── images/
│   ├── cam_000.png   # 512x512 RGB
│   ├── cam_001.png
│   ├── cam_002.png
│   ├── cam_003.png
│   ├── cam_004.png
│   └── cam_005.png
└── opencv_cameras.json
    {
      "frames": [
        {
          "w2c": [[4x4]],     # World-to-Camera matrix
          "fx": 549.0,         # Focal length X
          "fy": 549.0,         # Focal length Y
          "cx": 256.0,         # Principal point X (centered)
          "cy": 256.0,         # Principal point Y (centered)
          "w": 512, "h": 512
        },
        ... (6 cameras)
      ]
    }
```

### 4.5 Camera Configuration

| Camera | Azimuth | Elevation | Usage |
|--------|---------|-----------|-------|
| cam_000 | -147 deg | +14.9 deg | Reference view (always input) |
| cam_001 | +34 deg | +20.6 deg | Input (standard) |
| cam_002 | +86 deg | +11.3 deg | Input (standard) |
| cam_003 | -11 deg | +10.7 deg | Input (standard) |
| cam_004 | +144 deg | +26.5 deg | Input (standard) |
| cam_005 | -64 deg | **+30.8 deg** | Outlier (highest elevation) |

**CCW Camera Order**: `[0, 4, 2, 1, 3, 5]` (for turntable visualization)

---

## 5. Inference Pipeline

### 5.1 End-to-End (MV-Diffusion + GS-LRM)

```
Single Image ──► [Optional: SAM Preprocess] ──► MV-Diffusion (50 steps, cfg=3.0)
                                                      │
                                                      ▼
                                                 6 Multi-view Images
                                                      │
                                                      ▼
                                                 GS-LRM (single forward)
                                                      │
                                                      ▼
                                                 3D Gaussians
                                                      │
                                    ┌─────────────────┼─────────────────┐
                                    ▼                 ▼                 ▼
                              Turntable         PLY Mesh          Comparison
                              Video (.mp4)      (.ply)            Grid (.png)
```

### 5.2 GS-LRM Only (from preprocessed multi-view)

```
6 Preprocessed Views ──► GS-LRM (single forward) ──► 3D Gaussians ──► Outputs
```

### 5.3 Output Files (via TurntableRenderer)

| File | Description | Config Source |
|------|-------------|---------------|
| `turntable_orbit_{uid}.mp4` | 360 deg orbit video | orbit_views=120, fps=30 |
| `turntable_orbit_with_input_{uid}.mp4` | Orbit + input strip | Same + input_strip_height_ratio=0.25 |
| `turntable_view_with_input_{uid}.mp4` | Camera trajectory | view_num_frames=144, fps=10 |
| `turntable_{uid}.jpg` | 6x6 grid image | grid_rows=6, grid_cols=6 |

---

## 6. Original vs Mouse: Key Differences

| Aspect | Original FaceLift | Mouse FaceLift |
|--------|------------------|----------------|
| **Subject** | Human faces | Laboratory mouse |
| **Camera Count** | 6 | 6 |
| **Camera Arrangement** | Frontal hemisphere | Full surround (top-angled) |
| **Training Data** | Large-scale (Objaverse, etc.) | 3,600 samples (1 mouse) |
| **MV-Diffusion Steps** | 20,000 | 10,000 |
| **MV-Diffusion LR** | 1e-4 | 5e-5 |
| **GS-LRM Training** | From scratch (staged) | Finetuning from pretrained |
| **GS-LRM LR** | 1e-4 | 1e-6 |
| **GS-LRM Max Steps** | 20,000 | 15,000 |
| **GS-LRM Input Views** | 6 (train), 4 (val) | 4 (configurable) |
| **GS-LRM Target Views** | 8 | 6 |
| **Mask Strategy** | None | None (E0_1), gt+alpha (experiments) |
| **Prompt** | CLIP text embedding | Pre-computed mouse-specific 1024-dim |
| **Val Frequency** | 5,000 steps | 200 steps |
| **Background** | White | White (train: random white/black/gray for MVDiff) |

---

## 7. Config System

### 7.1 Modular Config Structure

```
configs/
├── base/gslrm_mouse.yaml          # Base config (all defaults)
├── datasets/M5t2.yaml             # Dataset paths + split info
├── experiments/E0_1_facelift.yaml  # Experiment-specific overrides
├── mvdiffusion/mouse_mvdiffusion_M5t2.yaml  # MV-Diffusion config
└── visualization/turntable_*.yaml  # Visualization settings
```

### 7.2 Config Merging Order

```python
# train_gslrm.py: load_modular_config(dataset='M5t2', experiment='E0_1_facelift')
config = merge(
    base/gslrm_mouse.yaml,     # Defaults
    datasets/M5t2.yaml,        # Dataset overrides (paths, stats)
    experiments/E0_1_facelift.yaml  # Experiment overrides (losses, views)
)
```

### 7.3 E0_1_facelift Experiment (Baseline)

Only 3 overrides on top of base config:
```yaml
model:
  num_views: 6           # 6 target views
  num_input_views: 4     # 4 input views

training:
  dataset:
    random_view_selection: true  # Randomly select which 4 of 6 are input
  losses:
    mask_mode: none       # No masking (original FaceLift behavior)
    alpha_loss_weight: 0.0
    bg_loss_weight: 0.0
```

---

## 8. Checkpoints

### 8.1 Pretrained (Stage 2)

| Component | Path | Description |
|-----------|------|-------------|
| **GS-LRM** | `checkpoints/gslrm/ckpt_0000000000021125.pt` | Pretrained on Objaverse (stage 2) |
| **MV-Diffusion** | `checkpoints/mvdiffusion/pipeckpts/` | SD 2.1 unclip base |

### 8.2 Mouse Finetuned

| Component | Path | Description |
|-----------|------|-------------|
| **GS-LRM** | `checkpoints/gslrm/M5t2_E0_1_facelift/best_psnr.pt` | Best validation PSNR |
| **MV-Diffusion** | `checkpoints/mvdiffusion/mouse_M5t2/checkpoint-5000/` | EMA checkpoint |
| **Prompt** | `mvdiffusion/data/mouse_prompt_embeds_6view_1024/clr_embeds.pt` | Pre-computed |

---

## 9. Related Documents

- [[../datasets/M5_SERIES_SPEC]] - M5 camera normalization details
- [[../datasets/RAW_DATA]] - Raw data source and sampling
- [[../experiments/EXPERIMENT_CONFIG_GUIDE]] - Config system details
- [[../experiments/COMMANDS]] - Training/inference commands
- [[../guides/MVDIFFUSION_FINETUNE_GUIDE]] - MV-Diffusion finetuning guide

---

*FaceLift Pipeline Architecture v1.0 | 2026-02-10*

> 이론 상세: [MULTIVIEW_DIFFUSION_THEORY](MULTIVIEW_DIFFUSION_THEORY.md)
