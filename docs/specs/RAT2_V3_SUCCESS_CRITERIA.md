# RAT2 v3 Recentered Training — Success Criteria

> **Version**: 1.0 | **Date**: 2026-04-02
> **Method**: MoA+Audit 3-iteration deliberation (9 models × 3 rounds = 27 API calls)
> **Scope**: NeurIPS 2026 D&B Track submission

## 0. Design Principles

```
v2 failure lesson:
  "PSNR ≥ threshold" is NECESSARY but NOT SUFFICIENT.
  Success = Image Quality AND Geometry Validity AND Generalization
```

**Primary metric**: `PSNR_gt` (masked foreground only, via `fair_comparison.py`)
- Full-image PSNR/SSIM are inflated by white background → do NOT use as primary

**2-Tier system**:
- **Minimum**: below this = FAIL (abort or investigate)
- **Target**: paper-quality result for NeurIPS submission

---

## 1. Step-by-Step Quantitative Milestones

### 1.1 PSNR_gt Trajectory (calibrated from v3 actual data)

**Calibration anchors**:
- v3 step 1 = 3.5 dB, step 101 = 4.67, step 201 = 14.3 (measured)
- v1 step 5200 = 17.49 dB (best, 1001 frames, 17.5 dB train/val gap)
- Mouse 6v = 20.12 dB PSNR_gt (same architecture, 6 cameras)

| Step | Minimum | Target | Rationale |
|------|---------|--------|-----------|
| 500 | 16.0 dB | 17.5 dB | Step 201=14.3, cascade phase → +1.7 dB in 300 steps is conservative |
| 1000 | 17.0 dB | 18.5 dB | Densification complete; should approach v1 peak |
| 2000 | 17.5 dB | 19.0 dB | v1 peak surpassed; 3× data advantage materializing |
| 5000 | 18.5 dB | 20.0 dB | v1-equivalent training point; clear improvement required |
| 10000 | 19.0 dB | 20.5 dB | Plateau region; fine detail refinement |
| 15000 | 19.5 dB | 21.0 dB | Final; mouse ceiling=20.12, rat anatomy may limit |

**Buffer rule**: Minimum(N+1) < Target(N) to avoid brittle progression.

### 1.2 Full Metric Milestones

| Step | PSNR_gt | IoU | LPIPS_fg | SSIM_fg | Train/Val Gap |
|------|---------|-----|----------|---------|---------------|
| 500 | ≥16.0 | ≥0.75 | ≤0.06 | ≥0.70 | monitor |
| 1000 | ≥17.0 | ≥0.80 | ≤0.05 | ≥0.75 | ≤5.0 |
| 5000 | ≥18.5 | ≥0.85 | ≤0.04 | ≥0.82 | ≤2.5 |
| 10000 | ≥19.0 | ≥0.88 | ≤0.035 | ≥0.85 | ≤2.0 |
| 15000 | ≥19.5 | ≥0.90 | ≤0.03 | ≥0.87 | ≤1.5 (target) |

**Note**: SSIM_fg and LPIPS_fg must use masked foreground region only.
Use `fair_comparison.py` with `--eval_mode masked_fg`.

---

## 2. 3D Geometry Verification (Mesh-Free)

### 2.1 Gaussian Quality Metrics

Tool: `mouse_extensions/scripts/eval/gaussian_quality_metrics.py`

| Metric | Minimum Pass | Target | Failure Mode |
|--------|-------------|--------|--------------|
| Floater ratio (DBSCAN) | ≤1.5% | ≤0.5% | Disconnected artifact clusters |
| Anisotropy ratio (median) | ≤10.0 | ≤5.0 | Needle-like Gaussians → cardboarding |
| Opacity ambiguity | ≤0.15 | ≤0.05 | Semi-transparent fog |
| Outlier count (outside bbox) | ≤2.0% | ≤0.5% | Scene boundary violations |

**DBSCAN parameters** (reproducibility): `eps=0.05 × scene_diameter, min_samples=50`

### 2.2 Recentering Verification (v3-specific)

| Check | Method | Pass Criterion |
|-------|--------|----------------|
| Centroid stability | `mean(gaussian_xyz)` at each checkpoint | \|mean_xyz\| < 0.5 |
| Camera convergence | Camera center distribution after recentering | Within [-2, 2] normalized |
| Clip_xyz safety | Count Gaussians at bbox boundary | < 0.1% at boundaries |

### 2.3 Orbit Turntable Visual Inspection

Checklist (every 1000 steps):
- [ ] No floaters/sinkers visible in 360° rotation
- [ ] No cardboarding (thin-plate) effect from side views
- [ ] No temporal flickering between consecutive orbit frames
- [ ] Body parts (limbs, tail) connected to torso
- [ ] Back-view quality comparable to front (gap < 3 dB PSNR)

### 2.4 Multi-View Consistency

| Check | Method | Pass |
|-------|--------|------|
| Per-camera IoU variance | IoU std across 6 cameras | < 0.05 |
| Per-camera PSNR variance | PSNR std across 6 cameras | < 1.5 dB |
| Bottom view penalty | Bottom PSNR vs mean | > -2.0 dB |

---

## 3. Early Abort Conditions

### 3.1 Hard Abort (immediate stop)

| Condition | Rationale | Action |
|-----------|-----------|--------|
| PSNR_gt < 16.0 at step 500 | Below current step 201 trajectory | Debug data/optimizer |
| \|mean_xyz\| > 5.0 at any step | Recentering drift → v2 failure | Check position gradients |
| Floater ratio > 5% at step 1000 | Catastrophic geometry collapse | Review densification params |
| Loss = NaN/Inf | Training instability | Check float precision |
| IoU < 0.60 at step 1000 | Silhouette completely wrong | Check masks/cameras |

### 3.2 Soft Warning (investigate, 24h to decide)

| Condition | Rationale | Action |
|-----------|-----------|--------|
| PSNR improvement < 0.3 dB / 1000 steps | Premature plateau | LR schedule review |
| Train/val gap > 3.0 dB at step 5000 | Overfitting signal | Increase regularization |
| Anisotropy P95 > 50.0 at step 5000 | Needle Gaussians forming | Opacity pruning threshold |
| LPIPS_fg > 0.06 at step 5000 | Perceptual quality stalled | D-SSIM loss weight |

---

## 4. Final Acceptance Criteria (NeurIPS Submission)

### 4.1 Minimum Viable (all must pass)

```
□ PSNR_gt ≥ 19.5 dB (final checkpoint)
□ IoU ≥ 0.90
□ LPIPS_fg ≤ 0.03
□ Floater ratio ≤ 1.5%
□ |mean_xyz| ≤ 0.5 (recentering confirmed)
□ Train/val gap ≤ 2.5 dB
□ Per-camera IoU std ≤ 0.05
□ Orbit visual inspection passed (no major artifacts)
□ v1 surpassed by ≥ 1.0 dB PSNR_gt
```

### 4.2 Paper Quality (target for strong submission)

```
□ PSNR_gt ≥ 21.0 dB
□ IoU ≥ 0.92
□ LPIPS_fg ≤ 0.02
□ Floater ratio ≤ 0.5%
□ Train/val gap ≤ 1.5 dB
□ Mouse-rat PSNR gap < 2.0 dB (cross-species generalization)
□ Orbit video: publication quality, no visible artifacts
```

### 4.3 NeurIPS D&B Track Specific

| Requirement | Verification |
|-------------|-------------|
| Reproducibility | ±0.5 dB across 2 independent runs (same seed → exact match) |
| Data quality | SAM2 mask IoU vs manual annotation ≥ 0.92 |
| Camera calibration | Reprojection error < 2.0 pixels |
| Multi-species | Same pipeline for mouse + rat, results in same paper |
| Code release | Training + eval scripts runnable from README |

---

## 5. Comparison Protocol

### 5.1 Fair Comparison Table

```bash
# Run for each checkpoint
python -m mouse_extensions.scripts.eval.fair_comparison \
    --model_path $CHECKPOINT \
    --eval_mode masked_fg \
    --output $OUTPUT_DIR
```

| Model | Steps | Frames | PSNR_gt↑ | IoU↑ | LPIPS↓ | Gap | Geo |
|-------|-------|--------|----------|------|--------|-----|-----|
| RAT2 v1 | 5000 | 1001 | 17.49 | — | — | 17.5 | ✓ |
| RAT2 v2* | 5000 | 2967 | 18.67 | — | — | — | ✗ |
| RAT2 v3 @5k | 5000 | 2967 | TBD | TBD | TBD | TBD | TBD |
| RAT2 v3 @15k | 15000 | 2967 | TBD | TBD | TBD | TBD | TBD |
| Mouse 6v | — | — | 20.12 | 0.886 | — | — | ✓ |

*v2: INVALID — 3D geometry corrupted by clip_xyz + non-normalized cameras

### 5.2 Cross-Species Protocol

Compare improvement over zero-shot baseline:
- Mouse: Δ PSNR = final - zero_shot_baseline
- Rat: Δ PSNR = final - zero_shot_baseline (2.77 dB)

This normalizes for species difficulty.

---

## 6. Monitoring Automation

### 6.1 Checkpoint Evaluation Script

```bash
# Run at each milestone (500, 1000, 2000, 5000, 10000, 15000)
python -m mouse_extensions.scripts.eval.comprehensive_eval \
    --checkpoint $CKPT_PATH \
    --dataset RAT2_despilled \
    --output_dir outputs/eval/rat/RAT2_v3_step${STEP}/

python -m mouse_extensions.scripts.eval.gaussian_quality_metrics \
    --ply_path ${CKPT_DIR}/iter_${STEP}/gaussians_*.ply \
    --output_dir outputs/eval/rat/RAT2_v3_step${STEP}/
```

### 6.2 Decision Points

```
Step 500:   Go/No-Go #1 → geometry check required
Step 1000:  Go/No-Go #2 → LR plateau check
Step 5000:  Mid-term → fair_comparison with v1
Step 10000: NeurIPS viability assessment
Step 15000: Final evaluation → all acceptance criteria
```

---

*MoA+Audit deliberation: 3 iterations × 3 models, convergence achieved on iteration 2*
*Audit corrections: step 5k/15k plateau fixed, DBSCAN params specified, SSIM/LPIPS added*
