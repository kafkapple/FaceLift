# Experiment Registry

> **목적**: 실험 설정 및 결과 기록
> **명령어**: → [[COMMANDS]]
> **Updated**: 260215

---

## 1. 명명 규칙

```
{카테고리}_{번호}_{이름}
예: E0_1_facelift, uniform_v2/E0_1_4view
```

---

## 2. 실험 카테고리

| 카테고리 | mask_mode | 설명 |
|----------|-----------|------|
| **E0** | none | FaceLift Baseline ⭐ |
| **E1** | gt | GT Mask 기반 |
| **uniform_v2/** | none | View 수 ablation (uniform sampling) |

---

## 3. View Ablation 결과

### R1: Inference-time (260205, 비균일 샘플링)

| Views | PSNR | 비고 |
|-------|------|------|
| **3** | **21.12** | ⭐ Best (비균일) |
| 2 | 20.20 | |
| 4-6 | 19.58 | Baseline |

> ⚠️ R1은 비균일 뷰 샘플링. v2 (uniform)에서 결과 반전됨

### R2: Uniform v2 Training (✅ 최종, 260210)

| Views | Val PSNR (dB) | Best Step | Δ vs Baseline | 상태 |
|:-----:|:-------------:|:---------:|:-------------:|:----:|
| **6** | **24.49** | 4,201 | +8.50 | ✅ |
| 5 | 23.02 | 13,101 | +7.03 | ✅ |
| 4 | 21.71 | 9,201 | +5.72 | ✅ |
| 3 | 20.01 | 10,701 | +4.02 | ✅ |
| 2 | 17.75 | 11,801 | +1.76 | ✅ |
| 1 | 11.08 | 2,401 | -4.91 | ✅ |
| baseline | 15.99 | 0 | — | ✅ |

**결론**: 뷰 수 ↑ = PSNR ↑ (**단조 증가**, diminishing returns 없음). 6-view 최적.

---

## 4. MV-Diffusion 실험

### Phase 1-2 (완료)

| Config | sparse_mv | ref_view | LR | 상태 | 결과 |
|--------|:---------:|:--------:|:--:|:----:|:----:|
| **M5t2** (baseline) | true | 0 | piecewise 5e-5 | ✅ ckpt-5000 | PSNR_wh=21.29 |
| M5t2_cfgr | false | 0 | piecewise 5e-5 | ✅ 완료 | PSNR_wh=20.81 (-0.48 dB) |
| ~~M5t2_cyclic~~ | false | all | - | ❌ 폐기 | 2변수 동시 변경 |
| **P0: M5t2_randref_sparse** | true | **random** | piecewise 5e-5 | ✅ ckpt-10K | PSNR ~27, oscillation |
| **P1: M5t2_pose_spherical** | true | **random** | piecewise 5e-5 | ✅ 완료 | 5K 후 plateau |

### Phase 3 (260215~, 실행중)

| # | Config | 핵심 변경 | LR | 상태 | GPU |
|:-:|--------|----------|:--:|:----:|:---:|
| **E1** | M5t2_20k_cosine | cosine 5e-5→0, 20K, 새 학습 | cosine | 🔄 ~31/20K | 4 |
| **E2** | M5t2_randref_20k_resume | P0 resume, 1e-5 after 10K | piecewise | 🔄 ~10054/20K | 7 |
| **E3** | M5t2_pose_extrinsic_add | extrinsic pose + add, cosine | cosine | ⏳ R1 후 | 4 |
| **E4** | M5t2_pose_spherical_add | spherical + add (vs P1 concat), cosine | cosine | ⏳ R1 후 | 7 |

---

## 5. H3 진단 결과

### H3 (260205)

| Dataset | GS-LRM Test | E2E Test | Gap |
|---------|-------------|----------|-----|
| M5t | 20.56 | 19.15 | **+1.41** |
| M5t2 | 19.58 | 19.63 | -0.05 |

### H3-bis 재검증 (260209, metrics_v2)

| Condition | PSNR_wh | PSNR_fg | SSIM | IoU |
|-----------|---------|---------|------|-----|
| A: GS-LRM only | 35.01 | 20.92 | 0.9929 | 0.946 |
| B: E2E M5t2 | 21.21 | 7.91 | 0.9662 | 0.521 |
| C: E2E M5t | 18.28 | 4.66 | 0.9566 | 0.263 |

**결론**: MVDiffusion은 항상 병목 (13-17 dB gap)

---

## 6. H6: Alpha Loss 결과 (v3, bugfix 후)

> v2 실험은 `alpha_loss=0` bug로 무효화 (삭제됨)
> v3: `original_gt_mask` 보존으로 alpha/bg/mask_iou 독립 계산
> ⚠️ **메트릭 주의**: 아래 표는 **Val PSNR** (training log). Fair eval (PSNR_gt)은 [[UNIFIED_ABLATION_REPORT]] §3 참조.

### 4-View (Val PSNR)

| Config | alpha_w | Val PSNR | Step | LPIPS ↓ | SSIM ↑ | Alpha IoU ↑ | 상태 |
|--------|:-------:|:--------:|:----:|:-------:|:------:|:-----------:|:----:|
| **4view_v2 (baseline)** | 0.0 | **21.82** | 9201 | 0.0429 | 0.9473 | N/A | ✅ |
| alpha03_v3 | 0.3 | 21.34 | - | - | - | - | ✅ (ckpt 존재) |
| alpha05_v3 | 0.5 | 21.20 | 8901 | 0.0204 | 0.9725 | 0.9451 | ✅ |
| alpha10_v3 | 1.0 | 20.84 | 8101 | **0.0147** | **0.9742** | **0.9562** | ✅ |

> 4v baseline: Val PSNR=21.82, Test PSNR_gt=**20.66** (fair eval). 두 수치는 다른 프로토콜.

### 6-View (Fair Eval, PSNR_gt — 2026-03-22)

| Config | alpha_w | PSNR_gt | IoU | SSIM | 상태 |
|--------|:-------:|:-------:|:---:|:----:|:----:|
| **6view_v2 (baseline)** | 0.0 | **23.84 ± 1.68** | 0.954 | 0.9627 | ✅ |
| 6view_alpha03_v3 | 0.3 | 23.29 ± 1.87 | **0.956** | 0.9607 | ✅ |
| 6view_alpha05_v3 | 0.5 | 23.00 ± 1.87 | 0.953 | 0.9593 | ✅ |
| 6view_alpha10_v3 | 1.0 | 22.55 ± 1.86 | 0.949 | 0.9573 | ✅ |

**핵심**: PSNR_gt는 baseline 최고. α=0.3은 IoU 최고 (0.956) + PSNR_gt 손실 최소 (-0.55 dB) → **best trade-off**.
**Novel view artifact**: Alpha loss → alpha entropy 3.4× 개선 (4v 데이터 기준). → [[ALPHA_LOSS_NOVEL_VIEW_ANALYSIS]], [[UNIFIED_ABLATION_REPORT]] §3 참조.

---

## 7. MV-Diffusion Ablation (H5: Paper-Aligned)

> **목표**: 원 논문 (Lyu et al., 2024) 설정 재현 + 개별 변수 효과 분리
> **Baseline**: M5t2 (aug=ON, lr=5e-5, 10K steps) → PSNR_wh=21.29

### LR Scaling Note

| | Paper | Ours | Ratio |
|--|-------|------|-------|
| GPUs | 8× A100 | 1× A6000 | 1/8 |
| Effective batch | 64 | 16 | 1/4 |
| LR (paper) | 1e-4 | 1e-4 | ❌ diverge |
| LR (sqrt scaling) | 1e-4 | **5e-5** | ✅ stable |

> `lr_scaled = lr_paper × √(batch_ours / batch_paper) = 1e-4 × √(16/64) = 5e-5`

### Ablation Table

| ID | Config | 변경사항 | LR | Steps | 상태 | 결과 |
|----|--------|---------|:--:|:-----:|:----:|------|
| **E0** | M5t2_paper_aligned | aug=OFF, lr=1e-4, 20K | 1e-4 | 20K | ❌ diverge @3956 | loss→2.5, silent crash |
| **E0v2** | M5t2_E0v2 | aug=OFF, lr=5e-5(scaled), 20K | 5e-5 | 20K | 🔄 | paper-aligned (LR scaled) |
| **E2** | M5t2_noaug | aug=OFF only | 5e-5 | 10K | 🔄 step ~7K | noaug isolation test |
| E6 | (planned) | lr=7.5e-5 only | 7.5e-5 | 10K | ⏳ | intermediate LR test |

### E0 Divergence Analysis (260211)

```
Step ~3600: loss=0.01-0.02 (normal)
Step ~3800: loss=0.039→0.139→0.164 (onset)
Step ~3900: loss=0.345→1.2→2.09→2.51 (explode)
Step  3956: process killed (NaN→CUDA crash)
```

**Root cause**: lr=1e-4 with effective batch=16 (paper uses batch=64).
**Resolution**: Use sqrt-scaled lr=5e-5 for E0v2.

---

## 8. GS-LRM Experiments

### Uniform v2 (View Ablation)

| Views | Val PSNR | Best Step | 상태 |
|-------|----------|-----------|------|
| **6** | **24.49** | 4201 | ✅ completed |
| 5 | 22.63 | 2901 | ✅ completed |
| 4 | 21.50 | 3801 | ✅ completed |
| 3 | 19.92 | 3801 | ✅ completed |
| 2 | 17.70 | 8801 | ✅ completed |
| 1 | 11.08 | 2401 | ✅ completed |

### Paper-Aligned

| Config | Views | LR | Loss | 상태 |
|--------|:-----:|:--:|:----:|:----:|
| paper_aligned_4view | 4 | 1e-4 | no LPIPS/SSIM | 🔄 step ~8K/20K |
| paper_aligned_6view | 6 | 1e-4 | no LPIPS/SSIM | ✅ PSNR=24.49 |

---

## 9. 체크포인트

| 모델 | 경로 |
|------|------|
| GS-LRM Pretrained | `checkpoints/gslrm/ckpt_*.pt` |
| GS-LRM Best | `checkpoints/gslrm/{exp}/best_psnr.pt` |
| MVDiffusion Baseline | `checkpoints/mvdiffusion/mouse_M5t2/` |
| P0 randref (10K) | `/node_data/.../mouse_M5t2_randref_sparse/checkpoint-10000` |
| P1 pose_spherical | `/node_data/.../mouse_M5t2_pose_spherical/` |
| **E1 20k_cosine** | `/node_data/.../mouse_M5t2_20k_cosine/` (학습중) |
| **E2 resume** | `/node_data/.../mouse_M5t2_randref_sparse/` (재사용, 학습중) |
| H6 alpha05_v3 | `/node_data/.../uniform_v2/4view_alpha05_v3/` (학습중) |
| H6 alpha10_v3 | `/node_data/.../uniform_v2/4view_alpha10_v3/` (학습중) |

---

## Optimal Settings & Results Summary

> Consolidated from `training_optimal_settings.md` (2026-02-24)

### Stage 1: MVDiffusion (SD2.1-UnCLIP + Era3D RMA)

#### Optimal Settings

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| LR scheduler | Cosine (best color) or Piecewise+resume (best overall) | E1 vs E2 trade-off |
| Steps | 20K | 11K→20K: marginal gain (+0.03 PSNR_int) |
| Sparse attention | Yes | Full attention (cfgr) worse in all metrics |
| Reference view | Random (if resuming) or Fixed (v0) | E2 random+resume = best PSNR_gt |
| Pose conditioning | Skip | No benefit (E3 ≈ E2) |

#### E2E Results (360 test frames)

| Experiment | PSNR_gt | PSNR_int | IoU | Coverage |
|-----------|:-------:|:--------:|:---:|:--------:|
| Baseline 5K | 7.93 | 13.70 | 0.474 | 74.0% |
| cfgr 10K | 7.75 | 15.75 | 0.491 | 67.5% |
| E1 cosine 20K | 7.90 | **16.15** | **0.528** | 70.5% |
| **E2 resume 20K** | **8.20** | 15.63 | 0.521 | **71.5%** |
| E3 pose 10K | 8.10 | 15.88 | 0.523 | 71.6% |

#### MVDiff Direct Quality (A1, E1 cosine 20K)

| View | Angle | Sil IoU | PSNR_int | Coverage |
|:----:|:-----:|:-------:|:--------:|:--------:|
| 0 (ref) | 0° | 0.975 | 25.13 | 99.9% |
| 1 | 60° | 0.621 | 18.90 | 78.9% |
| 2 | 120° | 0.500 | 16.56 | 64.9% |
| 3 | 180° | 0.509 | 17.09 | 65.3% |
| 4 | 240° | 0.464 | 17.02 | 60.5% |
| 5 | 300° | 0.423 | 16.95 | 55.0% |

**결론**: 아키텍처 내 최적 도달 (PSNR_gt 7.9-8.2 수렴). 실루엣 위치 오류가 지배적 (Sil IoU=0.582). 돌파 시 아키텍처 변경 필요.

### Stage 2: GS-LRM

#### Optimal Settings

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| num_input_views | 4 (현재) or 6 (향후) | 6v=+2.15 dB val, 단 E2E에서 v4,v5 미사용 |
| Losses | L2(1.0) + Perceptual(0.5) | Alpha reg는 성능 하락 |
| opacity_reg_weight | 0.0 (disabled) | E5: -1.0 dB val (harmful) |
| Early stopping | patience=10, val_every=200 | ~8K steps에서 수렴 |

#### View Ablation (Val PSNR)

| Views | Val PSNR | Delta from 4v |
|:-----:|:--------:|:---:|
| 6 | 24.49 | +2.15 |
| 4 | 22.34 | baseline |
| 3 | ~19.5 | -2.8 |
| 1 | 11.08 | -11.26 |

#### E5 Alpha Regularization

| Metric | 4v Baseline | E5 Alpha(0.3) | Delta |
|--------|:---:|:---:|:---:|
| Val PSNR | **22.34** | 21.34 | -1.00 |
| E2E PSNR_gt | 7.9-8.2 | ~7.9-8.2 | 0 (0% transfer) |

**결론**: 4v baseline 최적. Stage 2 개선은 E2E에 0% 전이 → 추가 실험 불필요.

### E2E Transfer Rate Analysis

| Improvement Source | Val Improvement | E2E Improvement | Transfer Rate |
|-------------------|:---:|:---:|:---:|
| MVDiff val (Stage 1) | +3.59 dB | +0.27 dB | ~7.5% |
| GS-LRM val (Stage 2) | +2.3 dB | 0.0 dB | **0%** |
| Cosine LR (PSNR_int) | — | +0.52 dB (color) | ~15% |

### Sensitivity Analysis (A2: Oracle MVDiff)

GS-LRM 4v에 GT/MVDiff 혼합 입력 → 뷰별 민감도:

| Level | Config | PSNR_gt | IoU | Finding |
|:-----:|--------|:-------:|:---:|---------|
| 0 | 6 GT (upper) | 21.02 | 0.943 | Upper bound |
| 1 | Replace v5 | 21.02 | 0.943 | v5 미사용 (영향 0) |
| 2 | Replace v4,5 | 21.02 | 0.943 | v4도 미사용 (영향 0) |
| 3 | Replace v3,4,5 | 15.82 | 0.780 | v3 교체 시 **-5.2 dB 급락** |
| 4 | Replace v2-5 | 10.44 | 0.614 | v2 추가 -5.4 dB |
| 5 | Replace v1-5 (≈E2E) | 7.90 | 0.528 | v1 교체 -2.5 dB |

**핵심**: `num_input_views=4`에서 실제 사용 뷰는 0,1,2,3번. Views 4,5는 완전 무시됨.

### Bottleneck Summary

```
GS-LRM 6v GT:  PSNR_gt=21.02, IoU=0.943  ← Upper bound
       ↓
MVDiff:         Sil IoU=0.582              ← 86% of quality loss HERE
       ↓
E2E Best:       PSNR_gt=8.20, IoU=0.521   ← Final output
```

| Priority | Target | Action | Impact |
|:--------:|--------|--------|:------:|
| P1 | MVDiff silhouette | 실루엣 예측 정확도 개선 | High |
| P2 | MVDiff views 1-3 | 인접 뷰 일관성 강화 (A2 critical) | High |
| P3 | GS-LRM views | 4→6 (현재 v4,5 미사용) | Medium |
| Low | Training strategy | 더 긴 학습/다른 LR → 이미 수렴 | Negligible |
| Low | GS-LRM regularization | Alpha/opacity → 성능 하락 (E5) | Negative |

### Recommended E2E Configuration

| Component | Setting | Checkpoint |
|-----------|---------|-----------|
| **MVDiff** | E2 (resume+random ref+sparse+piecewise) | `mouse_M5t2_randref_sparse/checkpoint-20000` |
| **GS-LRM** | 4v baseline (L2+Perceptual) | `M5t2_E0_1_facelift/best_psnr.pt` |

Alternative (color accuracy 중시): MVDiff E1 (cosine LR) + GS-LRM 4v baseline.

---

*Experiment Registry v7.0 | 2026-02-24*
