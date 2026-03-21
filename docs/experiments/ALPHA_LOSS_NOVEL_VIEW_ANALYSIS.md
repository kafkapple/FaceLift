# Alpha Loss & Novel View Artifact Analysis

> **Version**: v2.0 | **Created**: 2026-03-16 | **Status**: ACTIVE
> **Navigation**: [← INDEX](../INDEX.md) | [H6_ALPHA_MASK](../hypotheses/H6_ALPHA_MASK.md) | [PHASE2_NOVEL_VIEW_ROADMAP](PHASE2_NOVEL_VIEW_ROADMAP.md)
> **Related**: [EXPERIMENT_REGISTRY](EXPERIMENT_REGISTRY.md) | [mesh_gs_pair_collection](mesh_gs_pair_collection.md) | [MESH_GUIDED_REFINEMENT](../../mouse_extensions/docs/MESH_GUIDED_REFINEMENT.md)

---

## 1. Problem Statement

### 1.1 Artifact Description

Bottom view (elevation=-70°) 등 **extrapolated novel view**에서 GS-LRM 렌더링 시 **하얀색, 가늘고 긴 Gaussian artifact** (floaters) 다수 발생.

- **발생 조건**: Training camera 분포에서 크게 벗어난 시점
- **형태**: 얇고 긴 흰색 선 (elongated white streaks)
- **원인 요약**: Overhead 카메라 6개로 학습 → 바닥면 정보 부재 → 배경 Gaussian이 edge-on으로 보임

### 1.2 Root Cause Analysis (3-Model Consensus)

Multi-model 심의 (Claude, Gemini-2.5-pro, GPT-4o)에서 합의된 메커니즘:

```
Training views (모두 overhead)
    ↓
넓은 흰색 배경 → 효율적 표현을 위해 대형 flat Gaussian 배치 (pancake shape)
    ↓
Training view에서는 정상 렌더링 (위에서 보면 넓은 흰 면)
    ↓
Bottom view에서 edge-on으로 관찰 → 얇고 긴 흰색 선으로 렌더링
```

**핵심 메커니즘**: 3DGS의 Gaussian은 anisotropic 3D ellipsoid. 배경을 표현하기 위해 model이 학습한 극도로 납작한 Gaussian (scale_z ≪ scale_xy) 이 unseen angle에서 artifact로 나타남.

---

## 2. H6 Alpha Loss 재평가

### 2.1 기존 결론 (2026-02-22): ❌ 기각

H6 실험은 PSNR 하락만을 근거로 기각됨:

| Alpha Weight | PSNR | vs Baseline | 판정 |
|:------------:|:----:|:-----------:|:----:|
| 0.0 (baseline) | **21.82** | — | — |
| 0.3 | 21.34 | -0.48 | ❌ |
| 0.5 | 21.20 | -0.62 | ❌ |
| 1.0 | 20.84 | -0.98 | ❌ |

### 2.2 누락된 데이터: Perceptual Metrics

EXPERIMENT_REGISTRY에는 기록되어 있으나 H6 문서에서 **누락**되었던 핵심 데이터:

| Alpha Weight | PSNR | LPIPS ↓ | SSIM ↑ | Alpha IoU ↑ |
|:------------:|:----:|:-------:|:------:|:-----------:|
| 0.0 (baseline) | **21.82** | 0.0429 | 0.9473 | N/A |
| 0.5 | 21.20 | 0.0204 | 0.9725 | 0.9451 |
| 1.0 | 20.84 | **0.0147** | **0.9742** | **0.9562** |

**LPIPS 3배 개선**, SSIM/IoU 대폭 상승 → **geometry 품질이 크게 개선**되었다는 강한 신호.

### 2.3 재평가 근거

| 관점 | PSNR 기각 | 재평가 |
|------|-----------|--------|
| **메트릭 적합성** | PSNR = pixel-wise 밝기 오차 | Novel view artifact = geometry 문제 → LPIPS/IoU가 더 적합 |
| **PSNR 하락 원인** | Alpha loss가 유해 | Mask boundary 근처 세밀 텍스처(모피) 약간 blurring → pixel 오차 증가 |
| **Novel view 평가** | ❌ 수행하지 않음 | Alpha loss가 배경 Gaussian 억제 → novel view에서 효과 기대 |
| **문헌 지지** | — | LGM, GaussianObject, Compact-3DGS 모두 alpha/opacity supervision 사용 |

### 2.4 결론: 조건부 재활성화

**H6 상태 변경: ❌ 기각 → 🔄 재평가 중 (novel view 관점)**

기존 Training-view PSNR 기준으로는 기각이 맞으나, **novel view artifact 감소라는 새로운 평가 축**에서 재검증 필요.

---

## 3. Theoretical Background

### 3.1 Alpha Loss의 Artifact 억제 메커니즘

```
Alpha Loss: L_α = Loss(rendered_alpha, GT_mask)
    ↓
GT silhouette 외부의 rendered alpha → 0으로 강제
    ↓
배경 영역의 Gaussian opacity 억제
    ↓
대형 flat background Gaussian 제거/약화
    ↓
Bottom view에서 edge-on artifact 감소
```

**핵심**: Alpha loss는 "장면이 이 2D 경계 안에만 존재한다"고 학습시킴. 이는 unseen view에서도 **Gaussian opacity 자체가 낮아지므로** 효과가 전파됨.

### 3.2 관련 문헌

| 논문 | 기법 | 관련도 | 핵심 |
|------|------|:------:|------|
| **LGM** (ECCV 2024) | MSE alpha supervision | ⭐⭐⭐ | "faster convergence of the shape" |
| **GaussianObject** (SIG Asia 2024) | BCE alpha supervision | ⭐⭐⭐ | Object-level reconstruction |
| **Compact-3DGS** (2024) | Anisotropy regularization | ⭐⭐⭐ | Elongated Gaussian 직접 제약 |
| **StableGS** (2025) | Entropy-based opacity reg | ⭐⭐ | Opacity → 0 or 1 강제 |
| **DIFIX3D+** (CVPR 2025) | Single-step diffusion 3DGS refinement | ⭐⭐⭐ | Post-hoc artifact removal |
| **RegNeRF** (2022) | Depth smoothness + few-shot | ⭐⭐ | Geometric regularization |
| **FSGS** (2023) | Depth prior for few-shot GS | ⭐⭐ | Floater 억제 via depth |

### 3.3 Gaussian Anisotropy Regularization

Alpha loss와 별도로, Gaussian shape 자체를 제약하는 방법:

```python
# Compact-3DGS style anisotropy penalty
L_aniso = mean(max(scales) / min(scales))  # per Gaussian
```

극도로 납작한 "pancake" Gaussian을 직접 페널티 → artifact 근원 차단.

---

## 4. Existing Experiments & Gaps

### 4.1 완료된 실험

| 실험 | Config | 결과 | Novel View 평가 |
|------|--------|------|:---------------:|
| H6 alpha=0.3 | `4view_alpha03_v3` | checkpoint 존재 (미보고) | ❌ |
| H6 alpha=0.5 | `4view_alpha05_v3` | PSNR=21.20, LPIPS=0.0204 | ❌ |
| H6 alpha=1.0 | `4view_alpha10_v3` | PSNR=20.84, LPIPS=0.0147 | ❌ |
| E5 opacity reg | entropy-based | -1.0 dB (유해) | ❌ |
| Ghost pruning | post-hoc | 부분 효과 | — |

### 4.2 미완료/필요 실험

| 우선순위 | 실험 | 목적 | 비용 |
|:--------:|------|------|:----:|
| **P0** | H6 checkpoint → bottom view 렌더링 | Alpha loss의 novel view artifact 효과 검증 | **0** (기존 ckpt) |
| **P1** | Artifact Score 정량 메트릭 정의 | Silhouette 외부 pixel intensity 합산 | 낮음 |
| **P2** | Alpha + anisotropy reg 결합 | Pancake Gaussian 직접 제약 | 중간 |
| **P3** | MAMMAL mesh depth supervision | 강한 geometric prior | 중간 |
| **P4** | Scheduled alpha loss (점진적 증가) | PSNR 하락 최소화 | 중간 |
| **P5** | 6-view + alpha loss 학습 | 최종 모델 적용 | 높음 |

---

## 5. Code Architecture Review

### 5.1 구현 현황

Alpha loss 코드는 **두 개 모듈에 중복 구현**:

| 모듈 | 위치 | 역할 |
|------|------|------|
| `loss_extensions.py:418-612` | `AlphaLossType`, `compute_alpha_loss()`, `AlphaLossComputer` | Legacy, 독립 모듈 |
| `mask_losses.py:56-224` | `AlphaLossType`, `compute_alpha_supervision_loss()`, `MaskLossComputer` | Refactored, 문헌 기반 |
| `gslrm.py:428-435` | GS-LRM 통합 호출 | `mask_losses.py`의 함수 사용 |

### 5.2 발견된 이슈

1. **`AlphaLossType` 중복 정의** — `loss_extensions.py`에 NONE 포함 (5 values), `mask_losses.py`에는 미포함 (4 values)
2. **`compute_alpha_loss()` vs `compute_alpha_supervision_loss()`** — 거의 동일한 구현 중복
3. **`gslrm.py:432` dice/focal 무시** — `AlphaLossType.MSE if ... == "mse" else AlphaLossType.BCE`로 dice/focal이 묵시적으로 BCE로 fallback

### 5.3 Checkpoint 매핑

| 실험 | Config Path | Checkpoint Path (서버) |
|------|-------------|----------------------|
| Baseline (4v) | `base_uniform_v2.yaml` | `base_uniform_v2_4view_v2/best_psnr.pt` |
| Alpha 0.3 | `+ 4view_alpha03_v3.yaml` | `base_uniform_v2_4view_alpha03_v3/best_psnr.pt` |
| Alpha 0.5 | `+ 4view_alpha05_v3.yaml` | `base_uniform_v2_4view_alpha05_v3/best_psnr.pt` |
| Alpha 1.0 | `+ 4view_alpha10_v3.yaml` | `base_uniform_v2_4view_alpha10_v3/best_psnr.pt` |
| Baseline (6v) | `base_uniform_v2.yaml` (override 6) | `base_uniform_v2_6view_v2/best_psnr.pt` |

> **참고**: `base_uniform_v2.yaml`의 기본 `num_input_views: 4` → 4-view checkpoint와 호환.
> 6-view checkpoint는 별도 experiment config에서 `num_input_views: 6` override 필요.

---

## 6. P0 Experiment Plan

### 6.1 목표

기존 H6 alpha loss checkpoint에서 bottom view (-70°) 렌더링하여 **artifact 감소 여부 시각적 확인**.

### 6.2 방법

`collect_dataset.py generate --phase gslrm`을 이용:
- **Frames**: 5개 대표 프레임 (0, 500, 1000, 1500, 2000)
- **Checkpoints**: baseline (4v), alpha05, alpha10
- **출력**: 4개 novel view (bottom, top, front_low, side_low) × 5 frames × 3 checkpoints

### 6.3 Commands

```bash
cd /home/joon/dev/FaceLift

# 1. Baseline (4-view, no alpha)
CUDA_VISIBLE_DEVICES=7 python -m mouse_extensions.scripts.novel_view.collect_dataset \
    generate --phase gslrm \
    --frames 0 500 1000 1500 2000 \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/base_uniform_v2_4view_v2/best_psnr.pt \
    --config configs/mouse/uniform/base_uniform_v2.yaml \
    --output_dir outputs/datasets/novel_view_alpha_comparison/baseline_4v \
    --force

# 2. Alpha 0.5
CUDA_VISIBLE_DEVICES=7 python -m mouse_extensions.scripts.novel_view.collect_dataset \
    generate --phase gslrm \
    --frames 0 500 1000 1500 2000 \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/base_uniform_v2_4view_alpha05_v3/best_psnr.pt \
    --config configs/mouse/uniform/base_uniform_v2.yaml \
    --output_dir outputs/datasets/novel_view_alpha_comparison/alpha05 \
    --force

# 3. Alpha 1.0
CUDA_VISIBLE_DEVICES=7 python -m mouse_extensions.scripts.novel_view.collect_dataset \
    generate --phase gslrm \
    --frames 0 500 1000 1500 2000 \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/base_uniform_v2_4view_alpha10_v3/best_psnr.pt \
    --config configs/mouse/uniform/base_uniform_v2.yaml \
    --output_dir outputs/datasets/novel_view_alpha_comparison/alpha10 \
    --force
```

### 6.4 Literature-Based Gaussian Quality Metrics (v2, 2026-03-16)

초기 CAS (Composite Artifact Score) 메트릭은 connected components 기반으로 **6v의 connected haze를 foreground에 흡수**하여 잘못된 순위를 산출. 문헌 기반 메트릭으로 전환.

#### 6.4.1 3D Gaussian Space Metrics (from predicted Gaussians, 3 frames avg)

| Metric | 6v Baseline | 4v Baseline | 4v Alpha=0.5 | 4v Alpha=1.0 | 방향 |
|--------|:-----------:|:-----------:|:------------:|:------------:|:----:|
| **Aniso Ratio (mean)** | **71,833** | 67,941 | 68,258 | **61,922** ↓ | lower=better |
| **Aniso Ratio (p95)** | **328,469** | 306,851 | 295,234 | **258,628** ↓ | lower=better |
| **Aniso Ratio (p99)** | **536,715** | 490,285 | 513,890 | **435,075** ↓ | lower=better |
| % Aniso > 10 | 66.3% | 66.1% | 66.3% | 66.1% | — |
| % Aniso > 50 | 62.1% | 61.9% | 61.8% | 61.6% | lower=better |
| **Isotropy Score (mean)** | 14.92 | 14.65 | 15.07 | 15.12 | lower=better |
| **Opacity Ambiguity %** | **0.92** | 1.17 | 1.03 | 1.02 | lower=better |
| **Mean Opacity** | 0.0065 | 0.0074 | **0.0049** | **0.0047** ↓ | context |
| Scale Magnitude | 16.51 | 16.37 | 16.87 | 16.62 | lower=better |
| Spatial Outlier % | 86.5% | 86.5% | 85.8% | 85.7% | lower=better |
| **N Gaussians** | **884,738** | 589,826 | 589,826 | 589,826 | context |

#### 6.4.2 Rendered Alpha Metrics (from rendered novel views)

| Metric | 6v Baseline | 4v Baseline | 4v Alpha=0.5 | 4v Alpha=1.0 | 방향 |
|--------|:-----------:|:-----------:|:------------:|:------------:|:----:|
| **Alpha Sparsity (bottom)** | 0.0111 | 0.0146 | **0.0041** | **0.0039** ↓ | lower=better |
| **Alpha Sparsity (top)** | 0.0303 | 0.0236 | **0.0017** | **0.0015** ↓ | lower=better |
| **Alpha Entropy (bottom)** | 2.97 | **3.37** | **1.14** ↓ | **1.00** ↓ | lower=better |
| **Alpha Entropy (top)** | 4.46 | 4.01 | **0.73** ↓ | **0.55** ↓ | lower=better |

#### 6.4.3 Metric Definitions (문헌 근거)

**Anisotropy Ratio** (Compact-3DGS, 2024):
```
R_aniso = max(s_x, s_y, s_z) / min(s_x, s_y, s_z)
```
Pancake Gaussian (R >> 1)을 직접 탐지. 높을수록 elongated artifact 위험.

**Isotropy Score** (LightGaussian, Gao et al., 2024):
```
L_iso = Var(log(s_x), log(s_y), log(s_z))
```
0 = perfect sphere, 높을수록 비구형. 학습 시 regularization loss로도 사용.

**Opacity Ambiguity** (다수 논문):
```
Ambiguity% = fraction of Gaussians with 0.1 < sigmoid(opacity) < 0.9
```
Ideal: bimodal (0 or 1). 중간 opacity = floater 가능성.

**Alpha Sparsity** (regularization 일반 원리):
```
Sparsity = mean(min(alpha, 1 - alpha))
```
0 = perfect binary. 높을수록 fuzzy/hazy edges. 0.5 = 최악.

**Alpha Entropy** (InfoNeRF, Kim et al., 2021; DietNeRF, Jain et al., 2021):
```
H(A) = -Σ p(a) * log2(p(a))  (Shannon entropy of alpha histogram)
```
낮을수록 simple distribution (대부분 0 or 1). 높을수록 복잡/noisy.

### 6.5 결과 해석

**핵심 발견**:

1. **Alpha loss의 가장 뚜렷한 효과**: Alpha Entropy/Sparsity에서 **3~4배 개선**
   - Bottom Alpha Entropy: 3.37 (baseline) → 1.00 (alpha=1.0) = 3.4배
   - Bottom Alpha Sparsity: 0.0146 → 0.0039 = 3.7배
   - Alpha loss가 rendered alpha를 더 binary(0/1)로 만듦 → sharper edges

2. **Anisotropy Ratio**: 6v > 4v > 4v+alpha
   - 6v 가장 높음 (71,833) → pancake Gaussian 가장 많음
   - Alpha=1.0 가장 낮음 (61,922) → 14% 감소 → pancake 억제 확인
   - 6v가 50% 더 많은 Gaussians (884K vs 590K) → 더 많은 pancake

3. **Mean Opacity**: Alpha loss → 36% 감소 (0.0074 → 0.0047)
   - 배경 Gaussian opacity 전체적으로 억제

4. **초기 CAS 메트릭의 실패 원인**: Connected components가 6v의 hazy edge를 foreground에 흡수 → 잘못된 방향. 문헌 기반 메트릭은 올바른 방향성 확인.

### 6.6 Recommended Metrics (우선순위)

| 순위 | 메트릭 | 근거 | 구현 |
|:----:|--------|------|------|
| **1** | **Alpha Entropy** | InfoNeRF 문헌, 가장 큰 차이 (3.4x), novel view 직접 평가 | `gaussian_quality_metrics.py` |
| **2** | **Alpha Sparsity** | 직관적, 3.7x 차이, alpha loss 효과 명확 | `gaussian_quality_metrics.py` |
| **3** | **Anisotropy Ratio (p95)** | Compact-3DGS 문헌, 3D 공간 메트릭, pancake 직접 탐지 | `gaussian_quality_metrics.py` |
| **4** | **Mean Opacity** | 전체적 배경 억제 확인 | `gaussian_quality_metrics.py` |
| ~~5~~ | ~~CAS (deprecated)~~ | ~~Connected components 결함으로 폐기~~ | ~~`artifact_metrics.py`~~ |

### 6.7 Updated Priority

| 우선순위 | 실험 | 근거 |
|:--------:|------|------|
| **P1** | **6-view + alpha loss 학습** | 6v에서 anisotropy 최대 (71,833), alpha 미적용 |
| **P2** | Gaussian anisotropy reg | P95 anisotropy 328K→259K 감소 추가 가능 |
| **P3** | MAMMAL mesh depth supervision | Geometric prior |

---

## 7. Risks & Mitigations

| 리스크 | 심각도 | 대응 |
|--------|:------:|------|
| Alpha loss가 training view 질감 손상 | ⚠️ | PSNR vs artifact tradeoff 정량화 |
| GT mask 품질이 alpha loss 효과 좌우 | ⚠️ | Mask 품질 검증 (IoU 0.95+ 확인됨) |
| 4-view 결과가 6-view에 일반화되지 않을 수 있음 | ⚠️ | P5에서 6-view + alpha 학습 필요 |
| Over-constraining → 세밀 구조(수염, 발) 손실 | ⚠️ | Scheduled weight, 적정 weight 탐색 |

---

## 8. Alternative Approaches (심의에서 제안됨)

| 접근 | 출처 | 적용 시점 |
|------|------|----------|
| **Bounding box pruning** | Gemini | Alpha loss 효과 미미할 경우 대안 |
| **Gaussian anisotropy loss** | Claude/Gemini | P2: alpha + anisotropy 결합 |
| **MAMMAL mesh depth supervision** | All models | P3: 강한 geometric prior |
| **Generative prior (diffusion)** | Gemini | 장기 연구 방향 |
| **Adversarial learning** | GPT | GS-LRM feed-forward 특성과 호환성 검토 필요 |

---

## 9. Related Documents

### Backlinks

| 문서 | 관계 |
|------|------|
| ↑ [[../INDEX]] | MoC |
| ↔ [[../hypotheses/H6_ALPHA_MASK]] | H6 가설 (재평가 대상) |
| ↔ [[PHASE2_NOVEL_VIEW_ROADMAP]] | Phase 2 artifact removal |
| ↔ [[EXPERIMENT_REGISTRY]] | 실험 결과 기록 |
| ↔ [[mesh_gs_pair_collection]] | Novel view dataset pipeline |
| ↔ [[../../mouse_extensions/docs/MESH_GUIDED_REFINEMENT]] | Mesh-guided 전략 |
| ↓ `mouse_extensions/model/loss_extensions.py` | Alpha loss 구현 |
| ↓ `mouse_extensions/model/mask_losses.py` | Mask loss 통합 모듈 |
| ↓ `gslrm/model/gslrm.py:428-435` | GS-LRM 통합 |

---

*Alpha Loss & Novel View Artifact Analysis | v1.0 | 2026-03-16*

## 10. 6-View Alpha Loss Results (2026-03-22)

### 10.1 Training Status

3개 6-view + alpha loss variant 학습 완료 (15840 steps each):

| Variant | α Weight | Checkpoint |
|---------|:--------:|-----------|
| M5t2_6view_alpha03_v3 | 0.3 | `/node_data/.../ckpt_0000000000015840.pt` |
| M5t2_6view_alpha05_v3 | 0.5 | `/node_data/.../ckpt_0000000000015840.pt` |
| M5t2_6view_alpha10_v3 | 1.0 | `/node_data/.../ckpt_0000000000015840.pt` |

### 10.2 GT-View Metrics (512×512, test set)

| Model | PSNR↑ | SSIM↑ |
|-------|:-----:|:-----:|
| Baseline (α=0) | 34.00 ± 4.63 | 0.9898 |
| **α=0.3** | **34.10 ± 4.56** | **0.9912** |
| α=0.5 | 34.07 ± 4.46 | 0.9912 |
| α=1.0 | 33.82 ± 4.30 | 0.9911 |

> α=0.3이 PSNR/SSIM 모두 최고. α=1.0은 PSNR 미세 감소.

### 10.3 Novel-View Artifact Metrics (512×512)

| Model | FG Ratio↓ | Edge Density↓ |
|-------|:---------:|:------------:|
| Baseline | 0.0329 | 38.36 |
| α=0.3 | 0.0318 | 37.76 |
| α=0.5 | 0.0316 | 37.62 |
| α=1.0 | 0.0314 | 37.93 |

> Alpha loss → FG ratio 감소 (artifact 줄어듦). Edge density 감소 추세.

### 10.4 Resolution Fix (2026-03-22)

이전 렌더 해상도 384×384 → **512×512로 수정** (training resolution 일치).
- PSNR +0.4 dB 향상 (33.64→34.00 baseline)
- GT와 동일 해상도로 리사이즈 없이 직접 비교

### 10.5 Gaussian Count (GS-LRM 6v, frame 3310)

| Filter | Count |
|--------|:-----:|
| Total (512×512 × n_gaussians=2) | 1,048,578 |
| Opacity > 0.01 (visible) | 36,380 (3.5%) |
| Opacity > 0.5 (solid) | 4,571 (0.4%) |
| N≥2 multiview filter | ~39K-73K |

### 10.6 Result Files

| Type | Location |
|------|----------|
| Grids (512) | `outputs/report/6v_alpha_comparison_512/grids/` |
| Videos (512) | `outputs/report/6v_alpha_comparison_512/videos/` |
| Metrics JSON | `outputs/report/6v_alpha_comparison_512/metrics/` |

### 10.7 Conclusion

- **α=0.3이 6-view에서 최적**: PSNR 최고 + artifact 감소
- Alpha loss는 GT-view 품질을 유지하면서 novel view artifact를 줄임
- 384→512 해상도 수정으로 이전 PSNR 값이 과소평가되었음을 확인

