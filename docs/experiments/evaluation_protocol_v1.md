# FL vs PS Evaluation Protocol v1.0

> SSOT for all FaceLift vs Pose-Splatter comparison experiments.
> Created: 2026-02-22 | Based on NVS literature survey + project-specific analysis.

---

## 1. Background & Motivation

### 1.1 Models Under Comparison

| Model | Type | Input | Pipeline |
|-------|------|:-----:|----------|
| **FL GS-LRM** | Feed-forward | N GT views | Multi-view images → GS-LRM → 3D Gaussian |
| **FL E2E** | Feed-forward | 1 image | Image → MVDiffusion → GS-LRM → 3D Gaussian |
| **PS (Pose-Splatter)** | Per-scene optimization | N views | N views → per-scene 3DGS optimization (50 epochs) |

### 1.2 Dataset

- **M5t2**: Mouse surgical data, 6 fixed cameras, 3600 frames, 512×512
- **Split**: Train 0-2879 (80%), Val 2880-3239 (10%), Test 3240-3599 (10%)
- **Camera space**: M5 (HFOV=50°, fx=549) — all experiments must use M5 data

### 1.3 Literature Survey Summary

| Source | Cameras | Holdout Strategy |
|--------|:-------:|-----------------|
| 3DGS (Kerbl 2023) | 50-300 | Every 8th image (~12.5%) |
| N3DV (Li 2022) | 21 | 1 camera holdout |
| **Pose-Splatter (2025)** | **6** | **Primary: 1 holdout; Alt: 2 holdout** |
| 4DGS (Wu 2024) | multi | 1 holdout + test frames |
| GS-LRM (Zhang 2024) | sparse | All non-input views are novel |

**Conclusion**: For 6-camera sparse setup, **leave-one-out (1 holdout)** is the primary standard per Pose-Splatter's own precedent and N3DV convention. 2-holdout is an ablation alternative.

---

## 2. Evaluation Protocols

### 2.1 Three Orthogonal Dimensions

```
┌───────────────────────────────────────────────────────┐
│  Protocol A: Temporal Generalization (Primary)         │
│  - Train views: ALL 6 cameras                          │
│  - Train frames: 0-2879                                │
│  - Eval: ALL 6 views × test frames 3240-3599           │
│  - Measures: Can the model generalize to unseen TIME?  │
├───────────────────────────────────────────────────────┤
│  Protocol B: Spatial Generalization (NVS)              │
│  - Train views: 5 cameras (1 held out)                 │
│  - Train frames: 0-2879                                │
│  - Eval: holdout view × ALL frames                     │
│  - Measures: Can the model render UNSEEN viewpoints?   │
│  - Repeat: 6-fold (each camera held out once)          │
├───────────────────────────────────────────────────────┤
│  Protocol C: Combined (Strictest)                      │
│  - Train views: 5 cameras (1 held out)                 │
│  - Train frames: 0-2879                                │
│  - Eval: holdout view × test frames 3240-3599          │
│  - Measures: Both spatial AND temporal generalization   │
└───────────────────────────────────────────────────────┘
```

### 2.2 Protocol Priority

| Priority | Protocol | Role | Reported As |
|:--------:|----------|------|-------------|
| **P1** | A (Temporal) | Main comparison table | Primary results |
| **P2** | B (Spatial NVS) | Novel view synthesis capability | Secondary results |
| **P3** | C (Combined) | Most rigorous evaluation | Supplementary |

### 2.3 Metrics

| Metric | Key | Description | Higher=Better |
|--------|-----|-------------|:-------------:|
| **PSNR (FG)** | `psnr_gt_masked` | Foreground-masked PSNR | Yes |
| **PSNR (Intersect)** | `psnr_intersection` | Overlap-only PSNR (pure color) | Yes |
| **IoU** | `iou` | Silhouette intersection-over-union | Yes |
| **Coverage** | `coverage` | Predicted mask coverage of GT | Yes |
| **SSIM (FG)** | `ssim_gt_masked` | Foreground-masked SSIM | Yes |

---

## 3. Experiment Matrix

### 3.1 View Conditions

**Holdout strategy: Leave-one-out (1 holdout)**

| View Condition | Total Cameras | Train/Input Views | Holdout View | FL GS-LRM Model |
|:-:|:-:|:-:|:-:|:--|
| **6v** | 6 | 5 (views 0-4) | view 5 | 5-view GS-LRM |
| **5v** | 5 | 4 (views 0-3) | view 4 | 4-view GS-LRM |
| **4v** | 4 | 3 (views 0-2) | view 3 | 3-view GS-LRM |

### 3.2 Full 9-Experiment Matrix

| # | Method | View Cond. | Input/Train | Holdout | Protocol A Data | Protocol B/C Data |
|:-:|--------|:-:|:-:|:-:|:-:|:-:|
| 1 | FL GS-LRM | 6v | 5 GT views | v5 | ✅ `gslrm_5view_fair.json` | ✅ per_view `view_5` |
| 2 | FL GS-LRM | 5v | 4 GT views | v4 | ✅ `gslrm_4view_fair.json` | ✅ per_view `view_4` |
| 3 | FL GS-LRM | 4v | 3 GT views | v3 | ✅ `gslrm_3view_fair.json` | ✅ per_view `view_3` |
| 4 | FL E2E | 6v | 1 image | v5 | ✅ `e2_resume_20k_fair.json` | ✅ per_view `view_5` |
| 5 | FL E2E | 5v | 1 image | v4 | ✅ same | ✅ per_view `view_4` |
| 6 | FL E2E | 4v | 1 image | v3 | ✅ same | ✅ per_view `view_3` |
| 7 | PS M5 | 6v | Per-scene 5v | v5 | ⚠️ Existing* | ❌ Need training |
| 8 | PS M5 | 5v | Per-scene 4v | v4 | ❌ Need training | ❌ Need training |
| 9 | PS M5 | 4v | Per-scene 3v | v3 | ❌ Need training | ❌ Need training |

*PS M5 6v: Existing `m5_baseline_gs` has holdout=[5] (matches!). Protocol A already done via `fair_test_only_evaluation.json`. Protocol B/C need extraction from existing results.

### 3.3 Data Availability Summary

**FL (gpu03)**: All 6 experiments immediately available from existing fair eval JSONs.
- GS-LRM per_view metrics cover views 1-5 for all model variants
- E2E (e2_resume_20k) per_view metrics cover views 1-5
- Renders exist for all frames × all views

**PS (joon)**: 1 of 3 experiments exists, 2 need training.
- **PS 6v (holdout=[5])**: ✅ m5_baseline_gs (5-train/1-holdout, already matches!)
- **PS 5v (holdout=[4])**: ❌ Need new training (4 views train, holdout view 4)
- **PS 4v (holdout=[3])**: ❌ Need new training (3 views train, holdout view 3)

### 3.4 E2E Note

FL E2E uses a single model (`E2_resume_20k`) with 4-view GS-LRM backbone. It always takes 1 input image regardless of view condition. The "view condition" only changes which holdout view is used for evaluation. E2E does NOT benefit from more available GT views — it represents the **practical single-image inference** scenario.

---

## 4. FL GS-LRM View-Model Mapping

The FL GS-LRM "N-view" model takes N GT images as input. For fair comparison:

| View Cond. | FL Input Views | Which Are Novel? | Holdout View |
|:-:|:--|:--|:-:|
| 6v | views [0,1,2,3,4] → 5v model | view 5 is novel | 5 |
| 5v | views [0,1,2,3] → 4v model | views 4,5 are novel | 4 |
| 4v | views [0,1,2] → 3v model | views 3,4,5 are novel | 3 |

**Key insight**: For Protocol B (NVS), the holdout view is always a "novel view" for FL GS-LRM since it was NOT provided as input. This is directly comparable to PS where the holdout view was NOT used during per-scene optimization.

---

## 5. PS Training Plan

### 5.1 Existing Experiment

**PS M5 6v** (`m5_baseline_gs`):
- Config: holdout_views=[5], 6 cameras, M5 data, 50 epochs
- Status: ✅ Complete (trained 2026-02-20~21, ~25h)
- Fair eval: ✅ `fair_test_only_evaluation.json` exists

### 5.2 New Experiments Needed

| Experiment | Config Name | holdout_views | Train Views | Est. Time |
|-----------|-------------|:-------------:|:-----------:|:---------:|
| PS M5 5v | `m5_5view_holdout4` | [4] | [0,1,2,3,5] | ~25h |
| PS M5 4v | `m5_4view_holdout3` | [3] | [0,1,2,5] | ~25h |

**Note**: PS M5 5v trains on views [0,1,2,3,5] (skipping view 4). PS M5 4v trains on views [0,1,2,5] (skipping view 3). View selection preserves the non-holdout views from the 6v experiment.

Wait — this creates an asymmetry. In the 4v condition, PS would train on [0,1,2,5] but FL uses [0,1,2]. We need to decide:

**Option A (Consistent holdout)**: PS always drops the holdout view. Remaining views = N-1.
- 6v: train [0,1,2,3,4], holdout [5] → 5 train views
- 5v: train [0,1,2,3], holdout [4] → 4 train views (drop view 5 too)
- 4v: train [0,1,2], holdout [3] → 3 train views (drop views 4,5 too)

**Option B (Keep all non-holdout)**: PS trains on all available views except holdout.
- 6v: train [0,1,2,3,4], holdout [5]
- 5v: train [0,1,2,3,5], holdout [4]
- 4v: train [0,1,2,4,5], holdout [3]

**Selected: Option A** — This matches FL exactly:
- Both FL and PS use the SAME input views [0..N-2] for each condition
- Holdout view = view N-1 in each condition
- Fair comparison: same information given to each method

### 5.3 Final PS Training Matrix

| Experiment | holdout_views | Train Views | N Train |
|-----------|:-------------:|:-----------:|:-------:|
| PS M5 6v | [5] | [0,1,2,3,4] | 5 | ✅ exists |
| PS M5 5v | [4] | [0,1,2,3] | 4 | ❌ train |
| PS M5 4v | [3] | [0,1,2] | 3 | ❌ train |

Wait — for PS M5 5v with `holdout_views=[4]`, PS would still SEE views 0-3 AND view 5 during optimization (since the dataset has 6 cameras and only view 4 is held out). We need to also EXCLUDE views 5 from training data for the 5v condition.

**Corrected**: Use `holdout_views=[4,5]` for 5v condition (exclude views 4 AND 5), but evaluate on view 4 only. For 4v condition, use `holdout_views=[3,4,5]` (exclude views 3,4,5), evaluate on view 3 only.

| Experiment | holdout_views | Available for Train | Eval View | N Train |
|-----------|:-------------:|:-------------------:|:---------:|:-------:|
| PS M5 6v | [5] | [0,1,2,3,4] | view 5 | 5 |
| PS M5 5v | [4,5] | [0,1,2,3] | view 4 | 4 |
| PS M5 4v | [3,4,5] | [0,1,2] | view 3 | 3 |

This exactly matches FL GS-LRM input views for each condition.

---

## 6. Execution Plan

### Priority Order

| P | Task | Server | GPU | Time | Blocks |
|:-:|------|:------:|:---:|:----:|:------:|
| 1 | ✅ Stop SSIM05/10 (done) | gpu03 | 5,6 freed | — | — |
| 2 | Extract FL holdout metrics | gpu03 | — | ~10min | Report |
| 3 | PS M5 5v config + launch | joon | 0 | ~25h | PS eval |
| 4 | Write report system modules | local | — | ~1h | Report |
| 5 | PS M5 5v eval (after training) | joon | 0 | ~15min | Report |
| 6 | PS M5 4v config + launch | joon | 0 | ~25h | PS eval |
| 7 | PS M5 4v eval (after training) | joon | 0 | ~15min | Report |
| 8 | Generate 9-experiment HTML report | gpu03 | — | ~10min | All above |

### GPU Allocation (gpu03)

| GPU | Current | Status |
|:-:|---------|--------|
| 4 | H7 SSIM03 | Running (declining but not collapsed — let finish) |
| 5 | **FREE** | Available for new experiments |
| 6 | **FREE** | Available for new experiments |
| 7 | HP M5_4 | Running (healthy, PSNR 21.71) |

### Key Decision: SSIM03 (GPU 4)

SSIM03 best=21.10, current=19.69, declining but not collapsed. Baseline=21.71.
**Recommendation**: Let it run to completion for completeness of H7 ablation results. If GPU 4 is needed urgently, it can be stopped (result already conclusive: SSIM 0.3 < baseline 0.1).

---

## 7. Reference Data (SSOT, extracted 2026-02-22)

### Protocol A: Temporal Generalization (all views × test frames)

| Condition | FL GS-LRM | FL E2E | PS M5 |
|:---------:|:---------:|:------:|:-----:|
| **6v** (5-input) | **22.16** | 8.20 | 13.78 |
| **5v** (4-input) | **20.66** | 8.20 | *training* |
| **4v** (3-input) | **18.56** | 8.20 | *training* |

### Protocol B: Spatial NVS (holdout view × test frames)

| Condition | Holdout | FL GS-LRM | FL E2E | PS M5 |
|:---------:|:-------:|:---------:|:------:|:-----:|
| **6v** | view 5 | **16.81** | 9.10 | 13.16 |
| **5v** | view 4 | **16.70** | 8.15 | *training* |
| **4v** | view 3 | **15.50** | 7.14 | *training* |

### Key Observations (FL only, PS pending)

1. **FL GS-LRM Protocol A vs B gap**: 22.16→16.81 (6v), 20.66→16.70 (5v), 18.56→15.50 (4v)
   - Holdout view is consistently ~5-6 dB lower than overall (novel view harder)
2. **FL E2E Protocol B**: view 5 (9.10) > view 4 (8.15) > view 3 (7.14)
   - E2E performance varies significantly by holdout view position
3. **PS M5 6v**: Protocol B (13.16) < Protocol A (13.78) — holdout view harder, as expected

### PS M5 Fair Eval Status

| Model | PSNR_fg (A) | PSNR_fg (B) | IoU | Coverage | Status |
|-------|:-----------:|:-----------:|:---:|:--------:|:------:|
| 6v (holdout=[5]) | 13.78 | 13.16 | 0.846 | 0.893 | ✅ Complete |
| 5v (holdout=[4,5]) | — | — | — | — | ⏳ Training (~25h) |
| 4v (holdout=[3,4,5]) | — | — | — | — | ⏳ Queue (after 5v) |

**Data source**: `experiments/comparison/9exp_unified_metrics.json`

---

## 8. Ablation Experiments Status

### H6: Alpha Mask Supervision — REJECTED
- 0.3→21.34, 0.5→21.20, 1.0→20.84 (baseline 21.71)
- Conclusion: Monotonic decline. Alpha supervision harmful.

### H7: SSIM Weight — REJECTED
- 0.1(baseline)=21.71, 0.3=21.10↓, 0.5=collapsed(10.17), 1.0=collapsed(4.72)
- SSIM05/10 stopped (2026-02-22). SSIM03 running for completeness.
- Conclusion: Baseline weight 0.1 is optimal.

### HP: Preprocessing — IN PROGRESS
- M0 (raw): Diverged → stopped. Raw data incompatible with pretrained model.
- M5_4 (center only): PSNR=21.71 @ step 4600. Healthy, approaching baseline.
- M5_5: Pending (after M5_4 completes).

---

## Appendix: Experiment Checklist

### Before
- [ ] Split files verified (`data_mouse_t2_{train,val,test}.txt`)
- [ ] `num_input_views` correct (avoid 6-view for GS-LRM)
- [ ] Pretrained checkpoint path valid
- [ ] WandB project: `FaceLift-Mouse`

### During
- [ ] val/loss monitored (1.0 = bug)
- [ ] best_psnr.json created
- [ ] Turntable rendering checked

### After
- [ ] Test set evaluated (once only!)
- [ ] GT vs Pred visualization generated
- [ ] Results logged to experiment docs

---


## 9. Navigation

| Link | Document |
|------|----------|
| Hub | [[INDEX]] |
| FL vs PS | [[fl_vs_ps_comparison]] |
| Bottleneck analysis | [[mvdiffusion_bottleneck_analysis]] |
| DA1 experiment | [[domain_adaptation_DA1]] |
| Metrics protocol | [[theory/METRICS_PROTOCOL]] |
| Dataset specs | [[datasets/PREPROCESSING_REGISTRY]] |

---

*Evaluation Protocol v1.0 | 2026-02-22*
