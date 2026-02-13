---
marp: true
theme: default
paginate: true
---

# FaceLift: Single-Image to 3D Mouse Reconstruction

**Two-Stage Pipeline: MV-Diffusion + GS-LRM**

---

# Problem

- Reconstruct 3D mouse body from a **single image**
- No multi-view capture at inference time
- Target: laboratory mouse (DANNCE dataset, 6 synchronized cameras)

---

# Pipeline Overview

- **Stage 1 — MV-Diffusion**: single image → 6 consistent multi-view images
- **Stage 2 — GS-LRM**: 6 views + cameras → 3D Gaussian Splatting
- Output: 3D Gaussians (position, color, scale, rotation, opacity)

---

# Stage 1: MV-Diffusion

- Based on **Stable Diffusion 2.1 Unclip** (image-conditioned)
- Generates 6 views simultaneously via **multi-view cross-attention**
- Key modifications over vanilla SD:
  - Sparse multi-view attention across views
  - View-specific **prompt embeddings** encode view direction implicitly (not explicit camera poses)
  - CLIP image embedding from reference view as class conditioning
- **No camera pose input** — view geometry is learned through prompt embeddings + attention
- Input: 1 reference image (512x512) + 6 view-specific prompt embeddings [6, 77, 1024]
- Output: 6 denoised views (512x512 RGB)

---

# Stage 2: GS-LRM

- **Transformer-based** 3D reconstruction (not NeRF)
- Architecture:
  - Each image **concatenated with Plucker ray coordinates** [B, V, 9, H, W] (3 RGB + 6 Plucker)
  - Patch tokenizer (patch=8) → image tokens
  - 24-layer Transformer (d=1024, 16 heads)
  - Gaussian Upsampler + Pixel-Aligned Decoder
  - Differentiable Gaussian Splatting renderer
- **Camera geometry enters here** via Plucker coordinates (not in MV-Diffusion)
- Single forward pass at inference (no optimization loop)

---

# Stage 2: GS-LRM — Gaussian Output

- Each Gaussian: 11 parameters
  - xyz (3): position, clipped to [-1,1]^3
  - color (3): SH degree 0 (DC only, no view-dependent effects)
  - scale (3): anisotropic, exp activation
  - rotation (4): quaternion
  - opacity (1): sigmoid activation
- ~16K Gaussians per sample (2 learnable + 4×64² pixel-aligned)

---

# Train vs Inference: MV-Diffusion I/O

**Training (finetune)**
- Input from dataloader:
  - `imgs_in`: reference view replicated [6, 3, H, W] (CLIP image embedding as class label)
  - `imgs_out`: 6 GT target views [6, 3, H, W]
  - `color_prompt_embeddings`: view-specific text embeddings [6, 77, 1024]
- Forward: add noise to 6 target latents → UNet denoises conditioned on reference CLIP embedding + prompt embeds
- **No camera poses** — view direction encoded implicitly in prompt embeddings
- Output: predicted noise (epsilon)
- Loss: MSE(predicted noise, actual noise) with Min-SNR weighting

**Inference**
- Input: 1 reference image (512x512) + pre-computed prompt embeddings [6, 77, 1024]
- Forward: DDIM sampling, 50 steps, guidance scale 3.0
- **No camera poses** — same prompt embeddings as training
- Output: 6 denoised views [6, 3, 512, 512] in [0, 1]

---

# Train vs Inference: GS-LRM I/O

**Training (finetune)**
- Input from dataloader: 6 views + 6 cameras + view indices
  - `image` [B, 6, 3, 512, 512], `c2w` [B, 6, 4, 4], `fxfycxcy` [B, 6, 4]
- `split_data=True`: splits into **4 input + 6 target** views
  - Input views: randomly selected 4 of 6 (`random_view_selection`)
  - Target views: all 6 (includes input views, `target_has_input=True`)
- Forward: input images **concatenated with Plucker ray coordinates** (from c2w + intrinsics) → Transformer → Gaussians → render at target cameras
- **Camera poses used here** (not in MV-Diffusion) — Plucker coordinates are the geometric conditioning
- Output: rendered images [B, 6, 3, 512, 512] + rendered alpha + loss
- Loss: L2(render, GT) + 0.5 × Perceptual(render, GT)

**Inference**
- `split_data=False`: no input/target split, all views as encoder input
- Same Plucker coordinate computation from camera parameters
- No loss computation, no target rendering
- Output: raw Gaussian parameters (xyz, SH, scale, rotation, opacity)

---

# MV-Diffusion Training Details

- Finetuned from SD 2.1 Unclip base
- **Loss**: `L = E_t[ w(t) · ||ε_θ(x_t, t, c) - ε||² ]`
  - `w(t) = min(SNR(t), γ) / SNR(t)`,  `γ = 5.0`  (Min-SNR weighting)
  - `SNR(t) = α_t² / σ_t²` — downweights high-noise timesteps
  - No perceptual / LPIPS loss (same as original)
- LR: 5e-5 (original: 1e-4), effective batch size: 16
- 10K steps (original: 20K) — smaller dataset
- **Data augmentation** (mouse-added, original has none):
  - `bg_color: three_choices` — random white/black/gray (original also uses this)
  - `augmentation: true` — brightness [0.9, 1.1], contrast [0.9, 1.1] jitter (**mouse-only**)
  - `aug_hflip: false` — no horizontal flip (mouse is asymmetric)
  - Applied to **all mouse configs** (M5t, M5t2, all variants) — no per-dataset difference
  - Does not increase sample count; same 2,880 samples, just perturbed each epoch

---

# GS-LRM Training Details

- **Finetuned** from Objaverse-pretrained checkpoint (not from scratch)
- LR: 1e-6 (original: 1e-4) — 100x lower for finetuning
- **Loss**: `L_total = w_l2 · L_l2 + w_perc · L_perceptual`
  - `L_l2 = (1/N) Σ ||I_render - I_gt||²` — pixel-wise MSE (w=1.0)
  - `L_perc = Σ_l ||φ_l(I_render) - φ_l(I_gt)||²` — VGG feature matching (w=0.5)
  - **L2 warmup** (step < 500): `w_perc → 0`, only L_l2 active
  - Available but unused: LPIPS, SSIM, Alpha, BG (all w=0.0, for ablation)
- Mixed precision: bf16
- Gradient clip norm: 50.0 (relaxed vs original 1.0)
- Validation every 200 steps, early stopping patience: 10 cycles

---

# Inference Path 1: GS-LRM Only (6-view input)

- **Use case**: evaluate GS-LRM reconstruction quality in isolation
- **Input**: 6 preprocessed GT views + `opencv_cameras.json` from M5 dataset
- **Pipeline**:
  - Load 6 images [1, 6, 3, 512, 512] + cameras from sample directory
  - GS-LRM forward (`split_data=False`) → all 6 as encoder input
  - Output: 3D Gaussians → filter (opacity/scale/floater/crop)
- **Outputs**: PLY mesh, turntable video, render grid, comparison grid, RRD
- **Script**: `run_e2e_inference.py --sample_dir <path>` (no MV-Diffusion loaded)
- **Metric**: measures pure GS-LRM reconstruction capacity (upper bound)

---

# Inference Path 2: End-to-End (1-view → 3D)

- **Use case**: full pipeline — single image to 3D reconstruction
- **Input**: 1 image (raw photo or M5 sample view)
- **Pipeline**:
  - [Optional] SAM-based preprocessing: detect mouse, remove BG, center-align
  - MV-Diffusion: 1 image → 6 views (50 DDIM steps, cfg=3.0, seed=42)
  - Save generated views to `generated_views/`
  - GS-LRM: 6 generated views + M5 fixed cameras → 3D Gaussians
- **Outputs**: same as Path 1 + `generated_views/` + `preprocessed_input.png`
- **Script**: `run_e2e_inference.py --input_image <path>` or `--input_view_idx 0`
- **Metric**: measures full system quality (MV-Diffusion + GS-LRM combined)

---

# Path 1 vs Path 2: Key Differences

- **Path 1 (GS-LRM only)**
  - 6 real GT images → perfect input quality
  - Tests reconstruction capacity only
  - PSNR ~23.5 (6-view), ~21.5 (4-view), ~19.9 (3-view)
  - No MV-Diffusion dependency → faster, simpler

- **Path 2 (End-to-End)**
  - 1 image → 6 generated images → quality limited by MV-Diffusion
  - Tests full pipeline: view synthesis + reconstruction
  - PSNR ~21.3 (full white-bg) — degraded by generation artifacts
  - MV-Diffusion is the bottleneck (confirmed)
  - ~2 dB gap between Path 1 and Path 2 → MV-Diffusion error propagation

---

# Batch Inference & Temporal Videos

- Both paths support **batch mode**: `--data_dir` or `--split` for multiple samples
- Frame range control: `--start_frame`, `--end_frame`, `--num_frames`, `--frame_step`
- **Temporal video generation** (post-processing):
  - Collects per-sample turntable videos
  - `TemporalVideoRenderer`: composites temporal progression across samples
  - Fixed-angle videos: same orbit angle across time → shows pose changes
  - Rotating grid: multi-sample orbit in sync

---

# Data: M5 Series

- Source: DANNCE `markerless_mouse_1` (Dunn et al., 2021), preprocessed by MAMMAL_mouse
  - 6 cameras, **100fps**, **3 min** recording → **18,000 frames** per camera
  - Raw resolution: 1152 x 1024 → preprocessed to 512 x 512
- Sampling: every 5th frame → **3,600 samples** (18,000 / 5, effective 20fps)
- Preprocessing:
  - Affine transform to target fx=549
  - Principal point centering (cx=cy=256)
  - Batch uniform camera normalization (avg distance=2.7)
- **Split strategy differs across variants**:
  - **M5**: 80:10:10, **random shuffle** — frames scattered across splits
  - **M5t2**: 80:10:10, **temporal (consecutive)** — no shuffle, no data leakage
  - **M5t**: 1:1:1, **temporal** — ~1,200 each (PoseSplatter-compatible)
- M5 random split risks data leakage (adjacent frames in train+val)
- M5t2 fixes this: same ratio as M5, but strictly chronological
- **vs PoseSplatter**: separate 30min recording (54K frames/cam, 30fps, 1536x2048) + SAM2 masks — **different data, 3x longer**

---

# Original FaceLift vs Mouse Adaptation

- **Data scale**: large-scale (Objaverse) → 3,600 samples (single mouse)
- **Camera arrangement**: frontal hemisphere → full surround (top-angled)
- **GS-LRM training**: from scratch → finetuning (LR 100x lower)
- **MV-Diffusion**: 20K steps → 10K steps
- **Validation frequency**: 5,000 → 200 steps (fast feedback on small data)
- **Prompt**: CLIP text → pre-computed mouse-specific embedding

---

# Evaluation Protocol

- **Key issue**: mouse occupies only **2.3% of frame** → massive BG boost in PSNR
  - PSNR_full_white ≈ PSNR_fg + 16.4 dB
- Adopted **literature standard** (LGM, GS-LRM, PoseSplatter):
  - White background compositing
  - Full-image PSNR / SSIM / LPIPS
- Additional FG-independent metrics:
  - Masked L1 (foreground pixel accuracy)
  - Silhouette IoU (shape quality)

---

# Comparison with PoseSplatter

- PoseSplatter: **6 real views** → 3DGS (feed-forward, ~30ms)
- FaceLift: **1 image** → 6 generated views → 3DGS (much harder task)
- **Data difference** (separate recordings, same DANNCE-style setup):
  - PoseSplatter: 30min, 30fps, 54K frames/cam, 1536x2048, SAM2 masks, 1:1:1 temporal
  - FaceLift: 3min, 100fps, 18K frames/cam, 1152x1024→512x512, MAMMAL silhouettes, 80:10:10
- FG-independent comparison:
  - **Masked L1**: FaceLift 0.297 vs PoseSplatter 0.317 — FaceLift has **better color accuracy**
  - **Silhouette IoU**: FaceLift 0.518 vs PoseSplatter 0.868 — FaceLift has **worse shape**
- Interpretation: color reconstruction is strong; geometric consistency is the bottleneck

---

# Current Results Summary

- **Path 1** GS-LRM only (GT views → 3D): PSNR ~23.5 (6-view), monotonically increases with view count
- **Path 2** E2E (1 img → 6 gen → 3D): PSNR ~21.3 (full white-bg)
- ~2 dB gap → **MV-Diffusion is the bottleneck** (confirmed by H3 diagnosis)
- View count matters: 6-view > 5 > 4 > 3 (uniform selection, v2)
- Data diversity > epoch count (2,880×6 > 1,198×20)

---

# Key Findings

- **Single-image 3D mouse reconstruction is feasible** (FaceLift PoC confirmed)
- MV-Diffusion **view generation quality** limits final 3D quality
- **Shape (IoU)** is harder than **color (L1)** — geometry bottleneck
- Small dataset (3,600 samples) requires careful finetuning:
  - Low LR, frequent validation, early stopping
  - Temporal splits prevent data leakage
- BG fraction dominates PSNR — always report FG-normalized metrics alongside

---

# Next Steps

- Reduced-view MV-Diffusion (3-view, 4-view generation)
- Alpha mask supervision (H6) for sharper silhouettes
- SSIM loss weighting (H7) for structural improvement
- Literature-based architecture improvements (H8)
- Cross-evaluation with PoseSplatter on identical test frames

---
