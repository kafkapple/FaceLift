# H1-bis-v2: Same-Test-Set Revalidation with Metrics v2

> ← [RESEARCH_HYPOTHESES.md](../RESEARCH_HYPOTHESES.md) | **상태**: ✅ 완료 | **Updated**: 2026-02-09

Created: 2026-02-09

## Motivation

H1 diagnosis (2026-02-05) confirmed that MVDiffusion is a bottleneck when undertrained.
However, M5t and M5t2 used **different test sets**, making cross-experiment absolute
PSNR comparison invalid. This experiment resolves that by evaluating both MVDiffusion
models on the **same test set** with **full N** and **literature-standard metrics**.

### Problems Addressed

| Issue | H1 (original) | H1-bis-v2 (this) |
|-------|---------------|-------------------|
| Test set | Different (M5t: 33:33:33, M5t2: 80:10:10) | **Same** (M5t2 test, frames 3240-3599) |
| Sample size | N=50 (random subset) | **N=360** (full test set) |
| Metrics | FG-only PSNR (~19-20 dB) | **White-bg full-image PSNR** + masked L1 + IoU |
| Statistical test | Aggregate mean/std only | **Per-sample paired t-test** |

## Experimental Design

### Fixed Variables

| Variable | Value |
|----------|-------|
| Test set | M5t2 test: frames 3240-3599 (N=360) |
| Split file | `data_mouse_t2_test.txt` |
| GS-LRM checkpoint | `M5t2_E0_1_facelift/best_psnr.pt` (same for all) |
| Input view (E2E) | cam_000 (view 0) |
| Guidance scale | 3.0 |
| Diffusion steps | 50 |
| Prompt embeddings | `mouse_prompt_embeds_6view_1024` (6-view models) |
| Metrics | compute_e2e_metrics v2.0 |

### Conditions

| Condition | Label | MVDiffusion | Purpose |
|-----------|-------|-------------|---------|
| **A** | `gslrm_only` | None (6 GT views) | Upper bound |
| **B** | `e2e_M5t2` | `mouse_M5t2/checkpoint-5000` (2880 train, ~6ep) | Sufficient data |
| **C** | `e2e_M5t` | `mouse_M5t/checkpoint-8000` (1198 train, ~20ep) | Insufficient data |

### Hypotheses

1. **H1 revalidation**: Gap(C) >> Gap(B) ≈ 0, confirming MVDiffusion bottleneck
2. **Data quantity effect**: E2E(B) > E2E(C) on same test set, confirming data matters
3. **Statistical rigor**: Paired t-test on N=360 provides definitive significance

### Expected Results

| Condition | Expected PSNR_wh | Expected Gap | Basis |
|-----------|-----------------|--------------|-------|
| A (GS-LRM only) | ~35 dB | 0 (reference) | H8 4view run |
| B (E2E M5t2) | ~21 dB | small (~0) | H5 baseline |
| C (E2E M5t) | ~19-20 dB | large (~1-2 dB) | H1 M5t gap |

## Commands

### Condition A: GS-LRM Only (upper bound)

```bash
export CUDA_VISIBLE_DEVICES=5
python -m mouse_extensions.scripts.inference.run_e2e_inference \
  --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
  --split data_mouse_t2_test.txt \
  --gslrm_checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt \
  --output_dir outputs/h1bis_v2/gslrm_only \
  --no_turntable --no_mesh
```

### Condition B: E2E with M5t2 MVDiffusion (sufficient data)

```bash
export CUDA_VISIBLE_DEVICES=5
python -m mouse_extensions.scripts.inference.run_e2e_inference \
  --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
  --split data_mouse_t2_test.txt \
  --input_view_idx 0 \
  --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse_M5t2/checkpoint-5000 \
  --gslrm_checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt \
  --prompt_embed_path mvdiffusion/data/mouse_prompt_embeds_6view_1024 \
  --output_dir outputs/h1bis_v2/e2e_M5t2 \
  --guidance_scale 3.0 \
  --no_turntable --no_mesh
```

### Condition C: E2E with M5t MVDiffusion (insufficient data)

```bash
export CUDA_VISIBLE_DEVICES=5
python -m mouse_extensions.scripts.inference.run_e2e_inference \
  --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
  --split data_mouse_t2_test.txt \
  --input_view_idx 0 \
  --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse_M5t/checkpoint-8000 \
  --gslrm_checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt \
  --prompt_embed_path mvdiffusion/data/mouse_prompt_embeds_6view_1024 \
  --output_dir outputs/h1bis_v2/e2e_M5t \
  --guidance_scale 3.0 \
  --no_turntable --no_mesh
```

### Metrics v2 Evaluation (all conditions)

```bash
python -m mouse_extensions.scripts.eval.compute_e2e_metrics \
  outputs/h1bis_v2/gslrm_only \
  outputs/h1bis_v2/e2e_M5t2 \
  outputs/h1bis_v2/e2e_M5t \
  --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
  --skip_input_view 0
```

## Output Structure

```
outputs/h1bis_v2/
├── gslrm_only/          # Condition A
│   ├── samples/
│   ├── metrics_v2.json
│   └── run_config.json
├── e2e_M5t2/            # Condition B
│   ├── samples/
│   ├── metrics_v2.json
│   └── run_config.json
├── e2e_M5t/             # Condition C
│   ├── samples/
│   ├── metrics_v2.json
│   └── run_config.json
└── comparison_report.md  # Final comparison
```

## Analysis Plan

### 1. Aggregate Comparison

| Condition | PSNR_wh | PSNR_fg | SSIM | Masked L1 | IoU | N |
|-----------|---------|---------|------|-----------|-----|---|
| A: GS-LRM only | | | | | | 360 |
| B: E2E M5t2 | | | | | | 360 |
| C: E2E M5t | | | | | | 360 |

### 2. Gap Analysis

| | Gap (A-B) | Gap (A-C) | Gap (B-C) |
|---|---|---|---|
| PSNR_wh | | | |
| Paired t-stat | | | |
| p-value | | | |
| Significant? | | | |

### 3. Per-View Breakdown

Evaluate if MVDiffusion quality varies by view angle.

## Execution Notes

- Conditions B and C are sequential (same GPU, ~1.5h each)
- Condition A is fast (GS-LRM only, ~15min)
- Total estimated time: ~3.5h on 1 GPU
- Can run A in parallel with B/C on different GPUs if available
- Per-sample PSNR values should be saved for paired t-test
