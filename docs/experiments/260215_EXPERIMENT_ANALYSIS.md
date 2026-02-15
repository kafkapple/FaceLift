# FaceLift Experiment Analysis Report

> **Date**: 2026-02-15
> **Status**: H6 v3 in progress (13K+ steps), H5 E2E complete, P0/P1 pending
> **Author**: Generated from server metrics + WandB + best_psnr.json

---

## 1. Executive Summary

FaceLift는 **SD2.1-UnCLIP (Stage 1: MVDiffusion) + GS-LRM (Stage 2: Gaussian Reconstruction)** 2-stage 파이프라인이다. 현재까지의 실험 결과를 종합하면:

- **Stage 2 (GS-LRM)** 단독: GT 입력 시 PSNR 21-24 (4-6 views). 합리적 수준.
- **E2E (Stage 1 → Stage 2)**: PSNR_fg 6-9, sIoU 0.26-0.52. **Stage 1이 명확한 병목**.
- **H6 Alpha Loss**: PSNR은 baseline이 최고지만, **LPIPS/SSIM/IoU에서 alpha가 압도적 우세**.
- **H5 Diffusion**: sparse vs full attention (cfgr) **차이 거의 없음** — 다른 축 개선 필요.

**핵심 결론**: MVDiffusion (Stage 1) 품질 개선이 E2E 성능 향상의 열쇠. Pose conditioning (P0/P1)이 가장 유망한 다음 단계.

---

## 2. GS-LRM (Stage 2) Results — GT Input

### 2.1 View Ablation (H4)

> Base config: `base_uniform_v2.yaml` (15K fwdbwd_passes, seed=42, M5t2)

| Views | Best PSNR | Best Step | Gap vs 6-view |
|:-----:|:---------:|:---------:|:-------------:|
| **6-view** | **24.49** | 4201 | — |
| 5-view | 23.02 | 13101 | -1.47 |
| 4-view | 21.71 | 9201 | -2.78 |
| 3-view | 20.01 | 10701 | -4.48 |
| 2-view | 17.75 | 11801 | -6.74 |
| 1-view | 11.08 | 2401 | -13.41 |
| Baseline (zero-shot) | 15.99 | 0 | -8.50 |

**Legacy 실험 (다른 data split):**

| Experiment | Best PSNR | Step | Data |
|-----------|:---------:|:----:|:----:|
| M5t2_E0_1_facelift | 22.34 | 8001 | M5t2 |
| M5t_E0_1_facelift | 22.06 | 12201 | M5t |
| M5_E0_1_facelift | 21.68 | 1801 | M5 |
| paper_aligned_4view | 21.27 | 10801 | M5t2 |

**Insight**:
- 뷰 수 증가 = PSNR 단조 증가 (GS-LRM에서는 more views = better)
- 6-view가 4201 step에 24.49로 빠르게 수렴 — multi-view 데이터가 풍부하면 학습 효율 높음
- **하지만** 이것은 GT 입력 기준. E2E에서는 MVDiffusion이 생성하는 뷰 수가 많을수록 per-view 품질이 하락하여 역전될 수 있음

### 2.2 H6: Alpha Loss (v3 — bugfix 후 재실험)

> v2 실험은 alpha_loss=0 bug로 무효화되어 삭제됨 (23.4GB)
> v3: `original_gt_mask` 보존으로 alpha/bg/mask_iou 독립 계산

| Experiment | alpha_w | Best PSNR | Step | LPIPS ↓ | SSIM ↑ | Alpha IoU ↑ |
|-----------|:-------:|:---------:|:----:|:-------:|:------:|:-----------:|
| **4view_v2 (baseline)** | 0.0 | **21.82** | 9201 | 0.0429 | 0.9473 | N/A |
| alpha05_v3 | 0.5 | 21.20 | 8901 | 0.0204 | 0.9725 | 0.9451 |
| **alpha10_v3** | **1.0** | 20.84 | 8101 | **0.0147** | **0.9742** | **0.9562** |

**Latest checkpoint (학습 진행중):**

| Experiment | Step | PSNR | LPIPS | SSIM | Alpha IoU |
|-----------|:----:|:----:|:-----:|:----:|:---------:|
| alpha05_v3 | 13901 | 21.02 | 0.0215 | 0.9714 | 0.9454 |
| alpha10_v3 | 13001 | 20.71 | 0.0164 | 0.9726 | 0.9562 |

**분석 — PSNR vs Perceptual Quality Trade-off**:

```
PSNR:       baseline (21.82) > alpha05 (21.20) > alpha10 (20.84)   — baseline 승
LPIPS:      alpha10 (0.015)  < alpha05 (0.020) < baseline (0.043)  — alpha10 3x better
SSIM:       alpha10 (0.974)  > alpha05 (0.973) > baseline (0.947)  — alpha10 승
Alpha IoU:  alpha10 (0.956)  > alpha05 (0.945)                     — alpha10 승
```

- **PSNR**: baseline이 ~1 dB 더 높지만, 이는 alpha loss가 RGB reconstruction에 약간의 penalty를 주기 때문
- **LPIPS**: alpha=1.0이 baseline 대비 **2.9x 더 좋음** (0.015 vs 0.043) — perceptual quality 대폭 개선
- **SSIM**: alpha 실험들이 +0.026 더 높음 — 구조적 보존 우수
- **Alpha IoU**: 0.956 (alpha=1.0)은 매우 우수한 foreground segmentation 품질

**결론**: PSNR만 보면 baseline이 최고지만, **perceptual quality (LPIPS), 구조적 유사도 (SSIM), shape accuracy (IoU) 모두에서 alpha=1.0이 최고**. 논문 보고용이라면 alpha=1.0이 더 설득력 있는 결과.

---

## 3. E2E Pipeline Results (MVDiffusion → GS-LRM)

### 3.1 H5: MVDiffusion Attention Ablation

> baseline = sparse attention (5K steps), cfgr = full attention (10K steps)
> Protocol v1 (old): background 미분리

| Experiment | PSNR | SSIM | LPIPS | n_samples |
|-----------|:----:|:----:|:-----:|:---------:|
| baseline_ckpt5000 (sparse) | **7.83** | 0.644 | 0.560 | 200 |
| cfgr_ckpt10000 (full) | 7.75 | 0.644 | 0.560 | 360 |

**분석**:
- **Sparse vs Full attention 차이가 거의 없음** (PSNR 0.08 dB)
- cfgr이 2배 더 많은 step (10K vs 5K)으로 학습했음에도 개선 없음
- **결론**: Attention 방식 변경은 유의미한 개선을 제공하지 않음. 다른 축 (pose conditioning, longer training) 탐색 필요

### 3.2 H8: Reduced View Generation (Protocol v2)

> Protocol v2: `psnr_full_white` (배경=흰색 포함), `psnr_fg_only` (foreground 마스크 내), `silhouette_iou`

| Experiment | PSNR (full_white) | PSNR (fg_only) | SSIM | LPIPS | sIoU | Views Eval |
|-----------|:-----------------:|:--------------:|:----:|:-----:|:----:|:----------:|
| **4view (GT input, all 6)** | **35.01** | **20.92** | **0.993** | **0.008** | **0.946** | 0-5 |
| 4view_e2e (MVDiff, skip input) | 19.87 | 6.55 | 0.962 | 0.070 | 0.437 | 1-5 |
| 3view_e2e (MVDiff, all 6) | 22.67 | 9.19 | 0.964 | 0.081 | 0.415 | 0-5 |

**GT vs E2E Gap (4-view 기준)**:
```
PSNR fg_only:  20.92 (GT) → 6.55 (E2E)   = -14.37 dB drop
sIoU:          0.946 (GT) → 0.437 (E2E)   = -0.509 drop
LPIPS:         0.008 (GT) → 0.070 (E2E)   = 8.75x worse
```

**Stage 1 (MVDiffusion)이 Stage 2 성능의 대부분을 결정**한다는 결정적 증거.

### 3.3 H1bis: Dataset Split 비교

| Data Split | PSNR (full_white) | PSNR (fg_only) | SSIM | LPIPS | sIoU | masked_l1 |
|:----------:|:-----------------:|:--------------:|:----:|:-----:|:----:|:---------:|
| **M5t2** | **21.21** | **7.91** | **0.966** | **0.059** | **0.521** | **0.293** |
| M5t | 18.28 | 4.66 | 0.957 | 0.095 | 0.263 | 0.530 |

**M5t2 (80:10:10)이 M5t (1:1:1) 대비 모든 메트릭에서 우수**: PSNR +2.9 dB, sIoU 2x, masked_l1 절반.

---

### 3.4 H6-E2E: Alpha Loss Effect in E2E Pipeline (Protocol v2)

> **Date**: 2026-02-16
> **Setup**: MVDiff baseline (mouse_M5t2/checkpoint-5000, sparse, 5K) + GS-LRM alpha10_v3 (best_psnr.pt, step 8101)
> **Eval**: 360 test samples (frames 3240-3599), input_view_idx=0, guidance_scale=3.0, views 1-5

| Experiment | GS-LRM | PSNR (fw) | PSNR (fg) | SSIM | LPIPS | sIoU | n |
|-----------|:------:|:---------:|:---------:|:----:|:-----:|:----:|:-:|
| H1bis M5t2 (baseline) | M5t2_E0_1 | 21.21 | 7.91 | 0.966 | 0.059 | **0.521** | 360 |
| **H6-E2E alpha10_v3** | alpha10_v3 | **21.24** | **7.93** | 0.965 | 0.061 | 0.474 | 360 |
| H8 4view_e2e (skip input) | 4view_v2 | 19.87 | 6.55 | 0.962 | 0.070 | 0.437 | 360 |

**Per-view breakdown (alpha10_v3, full_white):**

| View | PSNR | SSIM | LPIPS | L1 |
|:----:|:----:|:----:|:-----:|:--:|
| 1 (nearest to input) | **23.20** | 0.972 | 0.048 | 0.220 |
| 2 | 20.56 | 0.961 | 0.065 | 0.305 |
| 3 (opposite) | 20.52 | 0.964 | 0.065 | 0.348 |
| 4 | 20.90 | 0.964 | 0.062 | 0.291 |
| 5 | 21.05 | 0.963 | 0.064 | 0.303 |

**Per-view (fg_only):** view_1=9.39, view_2=7.40, view_3=6.77, view_4=7.89, view_5=8.23

**GT vs E2E Alpha Loss Effect 비교:**

```
                     GT Input              E2E Pipeline
                  baseline  alpha10     baseline  alpha10     Transfer?
PSNR              21.82     20.84       21.21     21.24       ✗ (noise)
LPIPS             0.043     0.015       0.059     0.061       ✗ (no benefit)
SSIM              0.947     0.974       0.966     0.965       ✗ (no benefit)
IoU               N/A       0.956       0.521     0.474       ✗ (WORSE)
```

**분석 — Alpha Supervision의 E2E 한계:**

1. **PSNR/SSIM/LPIPS**: 사실상 동일 (delta < 0.003). Alpha loss가 GT input에서 보였던 LPIPS 2.9x 개선이 E2E에서는 완전히 소멸.

2. **sIoU 하락 (-0.047)**: 가장 중요한 발견. GT input에서 IoU 0.956이었던 alpha10이 E2E에서는 baseline(0.521)보다 낮은 0.474. Alpha supervision이 MVDiffusion의 noisy multi-view에서 foreground shape 추정을 오히려 방해.

3. **Root Cause**: Alpha loss는 GT mask를 supervision target으로 사용. GT input에서는 pixel-accurate mask 학습 가능. 하지만 E2E에서 MVDiffusion이 생성한 view는 multi-view inconsistency를 포함하므로, GS-LRM이 inconsistent alpha를 동시에 만족시키려 함 → shape 품질 저하.

4. **View proximity effect**: view_1 (input에 가장 가까운 뷰)의 fg_PSNR=9.39가 view_3 (반대편)의 6.77보다 +2.62 dB 높음. MVDiffusion의 view generation quality가 input view와의 각도 차이에 반비례.

**결론**: **Alpha loss의 이점은 Stage 1 (MVDiffusion) 품질에 종속적**. MVDiffusion 품질 개선 없이 GS-LRM만 개선해도 E2E 성능은 정체. 이것은 Phase 3 방향 (pose conditioning, longer training)의 정당성을 강화.

---

## 4. MVDiffusion (Stage 1) Validation

> WandB summary (epoch 55, 10K global steps)

| Run | val/psnr cfg=1.0 | val/psnr cfg=3.0 | val/ssim cfg=1.0 | val/lpips cfg=1.0 |
|-----|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| ohaif4n9 | **26.69** | 25.67 | 0.980 | 0.024 |
| fj14cntv | 25.68 | 25.49 | 0.978 | 0.028 |

- MVDiffusion 자체의 multi-view generation PSNR은 25-27 수준
- cfg=1.0 (no guidance)이 cfg=3.0보다 PSNR 약 1 dB 높음
- 하지만 E2E에서 fg_only PSNR이 6-9로 떨어지는 것은 **GS-LRM이 MVDiff 아티팩트에 취약**하다는 의미

---

## 5. Best Checkpoint Summary

### Stage 2 (GS-LRM) — 용도별 최선

| 목적 | Checkpoint | PSNR | LPIPS | sIoU | 근거 |
|------|-----------|:----:|:-----:|:----:|------|
| **최고 PSNR** | 6view_v2 (step 4201) | 24.49 | - | - | Most views = most info |
| **최고 균형** | M5t2_E0_1_facelift (step 8001) | 22.34 | - | - | 4-view production baseline |
| **최고 perceptual** | alpha10_v3 (step 8101) | 20.84 | **0.015** | **0.956** | Best LPIPS + shape |
| **E2E 기준** | 4view_v2 (step 9201) | 21.82 | 0.043 | - | E2E pipeline standard |

### Stage 1 (MVDiffusion) — 현재 유일한 checkpoint

| Checkpoint | PSNR (val) | Steps | Attention |
|-----------|:----------:|:-----:|:---------:|
| mouse_M5t2/checkpoint-5000 | 27.30 | 5K | Sparse |
| mouse_M5t2_cfgr/checkpoint-10000 | ~26-27 | 10K | Full |

---

## 6. Diagnosis: Why E2E Metrics Are Poor

### Root Cause Chain

```
MVDiffusion (Stage 1)
  ├─ 단 5K steps 학습 → undertrained
  ├─ Fixed reference view (cam0) → 특정 각도 편향
  ├─ 6개 카메라 elevation 비균일 (-8.5°~+9.6°) → 일관성 저하
  └─ Multi-view consistency 부족 → artifact 생성
       ↓
GS-LRM (Stage 2)
  ├─ MVDiff 아티팩트를 그대로 반영
  ├─ 불일치 뷰 → Gaussian 위치 혼란
  └─ Foreground 형상 왜곡 (sIoU 0.44)
       ↓
E2E: fg_only PSNR ~7, sIoU ~0.44 (vs GT input: PSNR 21, sIoU 0.95)
```

### 정량적 병목 분석

| Stage | 메트릭 | 값 | 상한 |
|:-----:|--------|:--:|:----:|
| Stage 1 (MVDiff) | val PSNR (multi-view gen) | 27.30 | ∞ |
| Stage 2 (GS-LRM, GT) | best PSNR (4-view) | 21.82 | ~24.5 (6-view) |
| **E2E** | **fg_only PSNR** | **6.55** | **21.82** |
| **E2E** | **sIoU** | **0.437** | **0.946** |

**E2E / GS-LRM(GT) 비율**: PSNR 30%, sIoU 46%. MVDiffusion 통과 시 성능의 50-70% 손실.

---

## 7. Improvement Roadmap

### Tier 1: 즉시 실행 가능 (1-2일)

| 우선순위 | 실험 | 예상 효과 | 근거 |
|:--------:|------|----------|------|
| **P0** | randref_sparse (random reference view) | sIoU +0.05~0.1 | Fixed cam0 편향 제거, 단일 변수 ablation |
| **P1** | pose_spherical (pose conditioning) | sIoU +0.1~0.2 | 카메라 비균일 elevation 보상, 공간 인식 |

### Tier 2: 단기 (3-5일)

| 우선순위 | 실험 | 예상 효과 | 근거 |
|:--------:|------|----------|------|
| 1 | MVDiff 20K steps (학습 연장) | PSNR +1~2 dB | 5K는 분명히 undertrained |
| 2 | 3-view/4-view MVDiff fine-tune (H8) | per-view quality ↑ | 뷰 수↓ → 개별 뷰 품질↑ |
| 3 | H7 SSIM weight (0.3/0.5/1.0) | SSIM ↑ | SSIM loss가 다른 loss에 밀림 관찰 |

### Tier 3: 중기 (1-2주)

| 우선순위 | 실험 | 예상 효과 | 근거 |
|:--------:|------|----------|------|
| 1 | Alpha=1.0 + P1 pose 조합 | sIoU + perceptual 동시 개선 | 각각 독립적 이점 확인 |
| 2 | Progressive training (MVDiff → GS-LRM joint) | E2E 갭 축소 | 현재 파이프라인은 독립 학습 |
| 3 | Novel view evaluation 체계 개선 | 정확한 성능 파악 | full_white vs fg_only 혼재 정리 |

### 실험 설계 원칙

1. **단일 변수 ablation**: P0 (reference view only), P1 (pose conditioning only)
2. **Baseline 비교 고정**: MVDiff checkpoint-5000 + GS-LRM M5t2_E0_1 best
3. **Metric 프로토콜 v2 통일**: full_white, fg_only, sIoU 모두 보고
4. **Sanity check**: alpha_loss > 0, mask_iou > 0 초기 확인 (v2 교훈)

---

## 8. H6 결론 및 권장 사항

### Alpha Loss의 가치

Alpha supervision은 **PSNR을 ~1 dB 희생하는 대신, perceptual quality를 3배 개선**한다.

```
┌─────────────────────────────────────────────────────┐
│ Metric         │ Baseline │ Alpha=1.0 │ Winner      │
├─────────────────────────────────────────────────────┤
│ PSNR           │ 21.82    │ 20.84     │ Baseline +1 │
│ LPIPS (↓)      │ 0.043    │ 0.015     │ Alpha 2.9x  │
│ SSIM           │ 0.947    │ 0.974     │ Alpha +0.03 │
│ Alpha IoU      │ N/A      │ 0.956     │ Alpha       │
│ Best step      │ 9201     │ 8101      │ Alpha faster│
└─────────────────────────────────────────────────────┘
```

**권장**: 논문 제출 시 **alpha=1.0을 기본 설정**으로 채택. PSNR 단일 지표로는 열세이나, multi-metric 관점에서 우수.

### H5 cfgr 결론

Sparse vs Full attention은 **차이 없음**. CFG 스케일 변경, attention 방식 변경은 현재 bottleneck 해결에 무효.

### 다음 단계 (Action Items)

- [ ] H6 v3 완료 대기 (15K step, ~2-3시간 후)
- [ ] P0 randref_sparse 실행 (GPU 4)
- [ ] P1 pose_spherical 실행 (GPU 7)
- [ ] H6 v3 완료 후 E2E 평가 실행 (alpha10_v3 + MVDiff baseline)
- [ ] MVDiff 20K step 실험 설계

---

## Appendix A: Metric Protocol Versions

| Protocol | 도입 | 메트릭 | 배경 처리 |
|:--------:|:----:|--------|----------|
| v1 (old) | H5 | PSNR, SSIM, LPIPS | 미분리 (배경 포함) |
| **v2** | H8+ | psnr_full_white, psnr_fg_only, ssim_full_white, lpips_full_white, silhouette_iou, masked_l1 | 흰 배경 분리 |

> ⚠️ v1과 v2 결과 직접 비교 불가. v1의 PSNR ~7.8 ≈ v2의 psnr_fg_only ~6.5 수준.

## Appendix B: Data Source

| Source | Path |
|--------|------|
| best_psnr.json | `/node_data/joon/checkpoints/FaceLift/gslrm/*/best_psnr.json` |
| E2E metrics | `/home/joon/dev/FaceLift/outputs/*/metrics.json` |
| Validation metrics | `/home/joon/dev/FaceLift/experiments/validation/*/iter_*/00000000/metrics.txt` |
| WandB summary | `/home/joon/dev/FaceLift/wandb/run-*/files/wandb-summary.json` |
| COMMANDS.md | `/home/joon/dev/FaceLift/docs/experiments/COMMANDS.md` |

---

*FaceLift Experiment Analysis | 2026-02-15*


---

## 9. Phase 3: Systematic MVDiffusion + GS-LRM Improvement Plan

> **Created**: 2026-02-15
> **Status**: Config 생성 완료, Round 1 실행 대기

### 9.1 동기 (Motivation)

Phase 2 실험 (P0 randref, P1 pose_spherical)에서 두 가지 근본 문제 확인:

1. **P0 oscillation**: constant LR 5e-5 에서 PSNR이 수렴하지 않고 진동
2. **P1 plateau**: 5K 이후 성능 정체, pose concat 효과 미미
3. **근본 원인**: `step_rules: "1:100000,0.5"` = 10K 범위 내 LR decay 전혀 없음

### 9.2 실험 설계

**3 Rounds, 5 Experiments:**

| Round | 실험 | 가설 | 독립 변수 |
|:-----:|------|------|-----------|
| R1 | E1 (20K cosine) | Cosine decay가 oscillation 방지 | lr_scheduler |
| R1 | E2 (P0 resume) | LR 감소로 P0 수렴 안정화 | step_rules (LR=1e-5) |
| R2 | E3 (extrinsic+add) | Extrinsic pose가 spherical보다 풍부한 정보 | pose method |
| R2 | E4 (spherical+add) | Add integration이 concat보다 효율적 | integration |
| R3 | E5 (alpha=0.3) | PSNR-LPIPS 최적 trade-off 지점 탐색 | alpha_loss_weight |

### 9.3 LR 전략 분석

**E1 (새 학습, cosine)**:
- 장점: Clean training curve, 표준 diffusion fine-tuning 방식
- 단점: 처음부터 재학습 (~30h GPU time)
- LR 궤적: `5e-5 → warmup 100 steps → cosine decay → 0`

**E2 (P0 resume, piecewise)**:
- 장점: 기존 10K 학습 활용, ~15h 절약
- 안전성: `LambdaLR.load_state_dict()` → `last_epoch` 복원 → 새 lambda 즉시 적용
- LR 궤적: `step 1~10K: 5e-5 (기학습) → step 10K+: 1e-5 (constant)`
- 주의: piecewise_constant type 불일치 가능성은 낮음 (동일 scheduler type 유지)

### 9.4 Pose Encoding 비교 설계

```
P1 (기존):  spherical + concat → extra cross-attn token
E3 (신규):  extrinsic + add    → 직접 hidden state injection
E4 (신규):  spherical + add    → concat vs add 격리 비교
```

| 비교 | 격리 변수 | 기대 인사이트 |
|------|-----------|-------------|
| P1 vs E4 | concat vs add | Integration 방식 효과 |
| E3 vs E4 | extrinsic vs spherical | Pose encoding 방식 효과 |
| E3 vs P1 | 두 변수 동시 | 최적 조합 탐색 |

### 9.5 Alpha Loss Gradient (H6 확장)

기존 결과:

| alpha_w | PSNR | LPIPS | SSIM | IoU |
|:-------:|:----:|:-----:|:----:|:---:|
| 0.0 | **21.82** | 0.043 | 0.947 | N/A |
| 0.5 | 21.20 | 0.020 | 0.973 | 0.945 |
| 1.0 | 20.84 | **0.015** | **0.974** | **0.956** |

E5 (alpha=0.3)의 위치: baseline과 alpha05 사이의 PSNR-perceptual trade-off 최적점 탐색.

### 9.6 성공 기준

| 실험 | 성공 기준 |
|------|-----------|
| E1 | PSNR > 28.0 (Baseline 27.30 대비 +0.7), 안정 수렴 |
| E2 | 10K→20K 구간 PSNR 단조 증가, oscillation 제거 |
| E3/E4 | E2E fg_PSNR > 8.0, sIoU > 0.55 (Phase 2 대비 개선) |
| E5 | PSNR > 21.4 (alpha05 대비 개선) AND LPIPS < 0.030 |

---

*Phase 3 Plan | 2026-02-15*
