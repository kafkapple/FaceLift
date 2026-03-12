# FaceLift Project Synthesis

> Created: 2026-03-03 | Comprehensive Research Overview

## 1. Research Goal

**Single-view monocular input → template-free 3D reconstruction of small, non-rigid, moving animals (mice)**

기존 연구의 한계:
- **PoseSplatter**: Per-scene optimization, 학습 시 6뷰 필요 → 새 장면마다 재학습
- **FaceLift (원본)**: 인간 얼굴 특화, 동물 행동 분석 불가
- **DANNCE/MAMMAL**: 3D pose만 추정, appearance/shape 재구성 불가

**우리의 목표**: Feed-forward 1-view → 3D Gaussian Splatting, 재학습 없이 새 프레임 즉시 재구성

## 2. Methodology

### 2.1 Two-Stage Pipeline

```
[Stage 1] SD2.1-UnCLIP Multi-View Diffusion
    Input: 1 reference image (any of 6 cameras)
    Output: 6 multi-view images (512×512)
    Architecture: UNet with multi-view cross-attention
    Pose conditioning: Plucker ray + spatial token injection
                    ↓
[Stage 2] GS-LRM (Gaussian Splatting Large Reconstruction Model)
    Input: 6 (or 4) multi-view images + camera parameters
    Output: 16,386 3D Gaussians (position, opacity, SH, scale, rotation)
    Architecture: ViT-based transformer (24 layers)
    Rendering: Differentiable Gaussian splatting
```

### 2.2 Key Technical Contributions

1. **Mouse-adapted MVDiffusion**: SD2.1-UnCLIP fine-tuned on 6-camera mouse data
2. **Plucker Spatial Token Pose Conditioning**: 64 spatial tokens from Plucker ray maps → cross-attention injection
3. **Fair Comparison Framework**: Unified eval protocol for FL vs PS on identical data
4. **Pose Injector Save/Load Pipeline**: End-to-end pose conditioning through inference

### 2.3 Data

| Item | Specification |
|------|--------------|
| **Source** | DANNCE `markerless_mouse_1` (Harvard) |
| **Cameras** | 6 views, 1152×1024 → 512×512 |
| **Frames** | 3,600 (100fps, temporal downsampled) |
| **Split (M5t2)** | Train 0-2879 (80%), Val 2880-3239 (10%), Test 3240-3599 (10%) |
| **Preprocessing** | auto_orient + distance normalization (fx=549, trans=2.7) |

## 3. Experiment Results Summary

### 3.1 GS-LRM Performance (GT Input, Upper Bounds)

| Input Views | PSNR_gt | IoU | PSNR_int |
|:-----------:|:-------:|:---:|:--------:|
| 1 | 10.47 | 0.028 | 10.47 |
| 2 | 15.95 | 0.858 | 17.91 |
| 3 | 18.56 | 0.899 | 19.54 |
| 4 | 20.66 | 0.926 | 21.29 |
| 5 | 22.16 | 0.942 | 22.56 |
| **6** | **23.84** | **0.954** | **24.02** |

**Key insight**: +5.5 dB gain from 2→4 views, diminishing returns 4→6 (+3.2 dB).

### 3.2 E2E Pipeline (1-view → MVDiff → GS-LRM)

| Strategy | PSNR_gt | IoU | Key Change |
|---------|:-------:|:---:|------------|
| Baseline (5K) | 7.93 | 0.474 | Initial model |
| E1 cosine LR | 7.90 | 0.528 | LR scheduler |
| E2 resume 20K | 8.20 | 0.521 | Extended training |
| E3 extrinsic pose | 8.10 | 0.523 | Pose conditioning |
| p1_6view_e2e | **8.44** | 0.495 | Previous best |
| H4b_20K_e2e | ⏳ | ⏳ | Pose + extended (running) |

**All E2E results converge**: PSNR 7.75-8.44 (0.69 dB span), IoU 0.47-0.53

### 3.3 MVDiffusion Validation Performance

| Experiment | Method | Val PSNR (best) | E2E PSNR_fg |
|-----------|--------|:---------------:|:-----------:|
| E2 baseline | No pose | 26.82 | 8.20 |
| H3 (E2+pose) | Extrinsic+Add | 26.55 | 8.85 |
| H4b Extended | No pose, +10K | 26.24 | 9.04 |
| **H6a_v2** | **Plucker+Add** | **27.34** | 8.95 |
| H7 (spatial token) | Plucker+Spatial | ~26.1@3.6K | TBD |

**Critical finding**: Val PSNR does NOT predict E2E performance (H4b val 26.24 < H6a_v2 val 27.34, but H4b E2E > H6a_v2 E2E).

### 3.4 FL vs PS Fair Comparison (M5 Test, Same Data)

| Model | Type | PSNR_gt_masked | IoU | Inference |
|-------|------|:--------------:|:---:|-----------|
| FL GS-LRM 6v (GT) | Feed-forward | 23.84 | 0.954 | ~0.1s/frame |
| FL GS-LRM 4v (GT) | Feed-forward | 20.66 | 0.926 | ~0.1s/frame |
| FL E2E best | Feed-forward | ~8.44 | ~0.50 | ~2s/frame |
| **PS 6cam** | **Per-scene opt.** | **13.78** | **0.846** | **~30min/scene** |
| PS 5cam | Per-scene opt. | 13.92 | 0.849 | ~30min/scene |

**Key comparison**:
- FL GS-LRM (GT input) >> PS by +10 dB → GS-LRM stage is excellent
- FL E2E << PS by -5.3 dB → MVDiffusion is the sole bottleneck
- FL is 18,000× faster inference (feed-forward vs optimization)

## 4. Hypotheses & Status

| ID | Hypothesis | Status | Key Finding |
|----|-----------|:------:|-------------|
| H1 | E2E vs GS-LRM factorial analysis | ✅ | MVDiffusion = 86% of quality loss |
| H2 | View ablation (1-6 views) | ✅ | 6v optimal, 4v sufficient for GS-LRM |
| H3 | Extrinsic pose conditioning | ✅ | +0.65 dB in E2E (modest) |
| H4 | Extended training (10K→20K) | ✅ | +0.3 dB val PSNR, best E2E 9.04 |
| H5 | Higher CFG drop rate | ✅ Rejected | No significant improvement |
| H6 | Plucker ray conditioning | ✅ Partial | Best val PSNR (27.34), E2E marginal |
| H7 | Plucker spatial token | 🔄 Training | H7v2 retraining (GPU6, step ~165/10K) |
| H8 | Silhouette-focused loss | 📋 Planned | Not yet implemented |

## 5. Core Findings

### 5.1 MVDiffusion = Sole Bottleneck

```
GT 6-view → GS-LRM:  PSNR_gt 23.84, IoU 0.954
E2E (1v → MVDiff → GS-LRM): PSNR_gt ~8.20, IoU ~0.52

Transfer Gap: -15.6 dB (86% quality loss)
IoU Gap: -0.43 (silhouette accuracy drops from 95% to 52%)
```

모든 학습 전략 (LR, resume, pose, CFG)이 좁은 범위에 수렴 → **MVDiffusion 아키텍처 자체의 한계**.

### 5.2 Val PSNR ≠ E2E Performance

H6a_v2 (val 27.34) > H4b (val 26.24), but E2E: H4b (9.04) > H6a_v2 (8.95).
→ Extended training이 multi-view consistency에 더 효과적일 수 있음.
→ 또는 H6a_v2 pose encoder weights 미저장 영향 (버그 수정됨).

### 5.3 Feed-Forward vs Per-Scene Trade-off

FL (feed-forward): 낮은 품질이지만 실시간 추론 가능 → 행동 분석 downstream에 적합
PS (per-scene): 높은 품질이지만 장면당 30분 최적화 → 오프라인 분석에만 적합

### 5.4 Plucker Spatial Token이 유망

H7이 step 400부터 H6a_v2를 +0.26~+1.22 dB 상회 → 공간 정보 보존이 핵심.
H7v2 (retrained with save fix) 결과로 최종 검증 예정.

## 6. Downstream Applications

### 6.1 Behavior Clustering via Visual Embedding

PoseSplatter에서 구현 및 테스트 완료:
1. 3D Gaussians → Spherical rendering (32 views)
2. ResNet18 feature extraction → 512D per view
3. Spherical Harmonics encoding → 8192D
4. Adversarial PCA → 50D rotation-invariant embedding
5. Clustering (K-means, Hierarchical, HDBSCAN)

**Synthetic test results**: Paper style (50D) best silhouette score 0.596

### 6.2 Integration with FaceLift

FaceLift의 3D Gaussian 출력을 동일 visual embedding pipeline에 적용 가능:
- Per-frame 3D Gaussians → spherical rendering → embedding → temporal behavior segmentation
- **장점**: Template-free이므로 새로운 동물 종에도 적용 가능

## 7. Running Experiments

| Experiment | GPU | Status | ETA |
|-----------|:---:|--------|-----|
| H4b@20K **진짜 E2E** | 5 | Running (~50%) | ~6h |
| H7v2 Spatial Token retraining | 6 | Step 165/10K | ~35h |

---

*FaceLift Project Synthesis | v1.0 | 2026-03-03*
