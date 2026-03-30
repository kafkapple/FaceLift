# Gaussian Count Fair Comparison: FaceLift vs PoseSplatter

> **Purpose**: Effective Gaussian 수를 매칭하여 "Gaussian 수 차이가 아닌 Gaussian 품질 차이"를 검증
> **Status**: Experimental Design (2026-03-30)

---

## 1. Background

### Gaussian Count Reality (코드 직접 측정, 2026-03-30)

| Stage | FaceLift (GS-LRM, α=0.3) | PoseSplatter |
|-------|:-------------------------:|:------------:|
| **Raw model output** | **1,572,866** /frame | ~8,500 /frame |
| opacity > 0.01 | 22,674 | — |
| opacity > 0.05 | 15,757 | — |
| **opacity > 0.1** | **11,920** | **~8,500** |
| opacity > 0.3 | 5,181 | — |
| opacity > 0.5 | 2,472 | — |
| Comprehensive eval (apply_all_filters, 0.04) | 17,274 | — |

**Key insight**: FL raw output의 98.5%가 opacity < 0.01 (배경 노이즈). opacity > 0.1에서 ~12K로, PS의 ~8.5K와 comparable.

### Why This Experiment Matters

- FL의 PSNR 우위(+10.06 dB)가 "더 많은 Gaussian" 때문인지 검증
- Gaussian 수를 매칭하면 **per-Gaussian representation quality** 직접 비교 가능
- NeurIPS reviewer의 예상 질문: "Is the advantage from more Gaussians or better Gaussians?"

---

## 2. Experiment Design

### 2.1 Matched-Count Evaluation Protocol

**목표**: FL과 PS의 effective Gaussian 수를 ~8,500 ± 500으로 매칭

**FL Gaussian 수 조절 방법**:

```python
# Method A: Opacity threshold 조절 (simplest)
# opacity > 0.1 → ~12K (still 1.4x PS)
# opacity > 0.2 → ~7.5K (close match)
# opacity > 0.15 → ~9K (bracketing)

# Method B: Top-K selection by opacity
# Sort by opacity descending, take top 8500
# Ensures exact count match

# Method C: apply_all_filters with adjusted params
# opacity_thresh=0.15, scaling_thresh=0.08, floater_thresh=0.5
```

**추천**: Method B (Top-K) — 정확한 수 매칭, 가장 높은 opacity의 Gaussian만 유지.

### 2.2 Experimental Conditions

| Condition | FL Gaussians | PS Gaussians | 비고 |
|-----------|:----------:|:----------:|------|
| **PS-matched** | ~8,500 (Top-K) | ~8,500 (as-is) | **Primary comparison** |
| FL-half | ~4,250 (Top-K) | ~8,500 | FL이 절반만으로도 경쟁력? |
| FL-default | ~17,274 (filter) | ~8,500 | 기존 comparison (unfair) |
| FL-full | 1,572,866 (raw) | ~8,500 | 완전 unfair, 참고용 |

### 2.3 Metrics (동일 프로토콜)

- **PSNR_gt_masked**: FG-only PSNR (직접 비교 가능)
- **IoU**: Silhouette IoU
- **L1_masked**: FG L1
- **N_gaussians**: 렌더링에 사용된 실제 Gaussian 수 기록

### 2.4 Data

- **Test set**: M5t2 test (frames 3240-3599, 360 frames × 6 views)
- **Resolution**: 512×512 (양쪽 동일)
- **Camera**: 동일 M5 GT cameras

---

## 3. Implementation Plan

### Step 1: Top-K Gaussian Selection (FaceLift)

```python
def select_topk_gaussians(gaussians, k=8500, vis_mask=None):
    """Select top-K Gaussians by opacity, optionally filtered by vis_mask."""
    opacity = gaussians.get_opacity.squeeze()
    if vis_mask is not None:
        # Set invisible Gaussians to -inf
        opacity = opacity.clone()
        opacity[~torch.from_numpy(vis_mask).to(opacity.device)] = -float('inf')
    topk_idx = torch.topk(opacity, min(k, len(opacity))).indices
    mask = torch.zeros(len(opacity), dtype=torch.bool, device=opacity.device)
    mask[topk_idx] = True
    return mask.cpu().numpy()
```

### Step 2: Evaluation Script Extension

```bash
# Add --max-gaussians flag to comprehensive_eval.py
python -m mouse_extensions.scripts.eval.comprehensive_eval \
    --checkpoint M5t2_6view_alpha03_v3/best_psnr.pt \
    --max-gaussians 8500 \
    --output-dir experiments/comparison/gaussian_matched/
```

### Step 3: Comparison Report

```bash
python -m mouse_extensions.scripts.eval.compare_with_baseline \
    --facelift_metrics experiments/comparison/gaussian_matched/metrics.json \
    --baseline_metrics baselines/pose_splatter/posesplatter_fair.json \
    --output_dir experiments/comparison/gaussian_matched_report/
```

---

## 4. Expected Outcomes

### Hypothesis A: FL still wins (per-Gaussian quality is better)
- FL의 feed-forward prediction이 per-scene optimization보다 더 informative한 Gaussian 배치
- GS-LRM의 transformer attention이 global context를 활용하여 각 Gaussian의 위치/색상이 더 정확

### Hypothesis B: FL advantage decreases significantly
- FL의 우위가 주로 Gaussian 수에서 비롯
- 수 매칭 시 성능 차이 축소 → 아키텍처 차이보다 representation density가 핵심

### Hypothesis C: FL still wins on IoU but PS catches up on PSNR
- FL의 silhouette prediction이 강점 (α loss training)
- PS의 per-scene optimization이 texture detail에서 유리

---

## 5. Input Views Clarification (SSOT)

| 비교 유형 | FaceLift 입력 | PS 입력 | 공정성 |
|-----------|:---:|:---:|:---:|
| **Tier A: GS-LRM 6v vs PS 6v** | 6 GT views → GS-LRM (feed-forward) | 6 GT views → per-scene optimization | ✅ **Fair** (동일 입력) |
| Tier B: E2E vs PS | 1 image → MVDiff → GS-LRM | 6 GT views | ❌ Unfair (다른 task) |
| View Ablation | 1-6 GT views → GS-LRM | N/A | FL 내부 비교 |

**현재 프로젝트 주 사용 모드**: GS-LRM standalone (1-6 GT views), E2E는 MVDiffusion 병목으로 중단.

---

## 6. Resolution SSOT

| 맥락 | FaceLift | PoseSplatter | 비고 |
|------|:--------:|:------------:|------|
| Model native | 512×512 | 384×512 (Duke 4x ds) | 다른 pretraining data |
| **M5 Fair Eval** | **512×512** | **512×512** (576→crop) | **양쪽 통일** |
| PS 논문 보고 | N/A | 384×512 or 768×1024 | Duke dataset 기준 |

> ⚠️ 이전 METRICS_PROTOCOL.md의 "384×512" 기술은 PS 논문 기준이며, M5 Fair Eval에서는 512×512로 통일됨.

---

*Created: 2026-03-30 | FaceLift Gaussian Count Fair Comparison*
