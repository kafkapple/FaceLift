# FaceLift Phase 3: MVDiffusion Systematic Improvement Report

> **Date**: 2026-02-16
> **Phase**: 3 (MVDiffusion + GS-LRM joint optimization)
> **Status**: Round 1 실행중 (E1-E5 + E3 pose)
> **Author**: Experiment analysis from server metrics + WandB + E2E evaluation

---

## 1. Executive Summary

Phase 3는 FaceLift E2E 파이프라인의 핵심 병목인 MVDiffusion (Stage 1) 개선에 집중한다.
Phase 2에서 확인된 두 가지 근본 문제 — P0 oscillation과 P1 plateau — 를 해결하기 위해
LR 전략, pose conditioning, alpha loss gradient를 체계적으로 탐색한다.

### 핵심 발견 (이 보고서에서)

1. **Alpha loss E2E 무효**: GT input에서 LPIPS 2.9x 개선이 E2E에서는 완전 소멸. sIoU 오히려 하락 (-0.047)
2. **MVDiffusion = 유일한 E2E 개선 경로**: GS-LRM 단독 개선으로는 E2E 성능 정체
3. **View proximity bias**: input view에 가까운 뷰일수록 fg_PSNR 높음 (9.39 vs 6.77)
4. **4 GPU 전부 활용**: E1/E2 (LR), E5 (alpha gradient), E3 (pose extrinsic) 동시 실행

---

## 2. 배경 및 동기 (Background)

### 2.1 파이프라인 구조

```
Input Image (cam_0)
    │
    ▼
┌──────────────────────┐
│  Stage 1: MVDiffusion │  SD2.1-UnCLIP + Era3D RMA
│  (multi-view gen)     │  6 views, 512×512
└──────────┬───────────┘
           │ 6 generated views
           ▼
┌──────────────────────┐
│  Stage 2: GS-LRM     │  Transformer → 3D Gaussians
│  (3D reconstruction)  │  Render from any viewpoint
└──────────────────────┘
```

### 2.2 Phase 1-2 핵심 결론 (선행 실험)

| 가설 | 실험 | 결론 |
|------|------|------|
| **H1**: MVDiff가 병목 | H1bis E2E | ✅ GT→E2E 시 PSNR -70%, sIoU -54% |
| **H4**: 뷰 수 증가 = 개선 | View ablation | ✅ 6v(24.49) > 4v(21.71) > 1v(11.08) (GT) |
| **H5**: Attention type 차이 | sparse vs full | ✗ 차이 없음 (PSNR 0.08 dB) |
| **H6**: Alpha loss 효과 | alpha=0/0.5/1.0 | ✅ GT에서 perceptual 3x 개선, 但 E2E 무효 |
| **P0**: Random ref view | randref_sparse | 진행중 (E2 resume) |
| **P1**: Pose conditioning | spherical+concat | 5K→plateau, 새 전략 필요 |

### 2.3 Phase 2에서 발견된 문제

1. **P0 LR oscillation**: constant LR 5e-5에서 PSNR이 수렴하지 않고 진동
2. **P1 pose plateau**: 5K step 이후 성능 정체, concat integration 효과 미미
3. **근본 원인**: `step_rules: "1:100000,0.5"` = 10K 범위 내 LR decay 전무

---

## 3. Phase 3 실험 설계 (Experiment Design)

### 3.1 설계 원칙

| 원칙 | 설명 |
|------|------|
| **단일 변수 ablation** | 실험당 하나의 변수만 변경 |
| **Baseline 고정** | MVDiff checkpoint-5000 + GS-LRM M5t2_E0_1 |
| **Metric protocol v2** | full_white, fg_only, sIoU 모두 보고 |
| **Sanity check** | alpha_loss>0, mask_iou>0 초기 확인 (v2 교훈) |

### 3.2 실험 매트릭스

| Round | ID | 실험명 | 독립 변수 | Base | GPU | Status |
|:-----:|:--:|--------|-----------|------|:---:|:------:|
| R1 | E1 | cosine_20K | lr_scheduler=cosine, 20K steps | scratch | 4 | 🔄 3K/20K |
| R1 | E2 | P0_resume_20K | LR 5e-5→1e-5 at 10K, resume | P0 ckpt-10K | 7 | 🔄 13K/20K |
| R2 | E3 | extrinsic+add | pose=extrinsic, integration=add | ckpt-5000 | 6 | 🔄 시작 |
| R2 | E4 | spherical+add | pose=spherical, integration=add | ckpt-5000 | — | 대기 |
| R3 | E5 | alpha=0.3 | alpha_loss_weight=0.3 (GS-LRM) | base_uniform_v2 | 5 | 🔄 시작 |

### 3.3 비교 설계

```
LR 전략 비교:
  E1 (cosine from scratch)  vs  E2 (piecewise resume from P0)
  → 격리 변수: LR schedule type

Pose encoding 비교:
  P1 (spherical+concat)  vs  E4 (spherical+add)  vs  E3 (extrinsic+add)
  │                          │                       │
  └── P1 vs E4: integration  └── E3 vs E4: encoding └── E3 vs P1: 두 변수
      (concat vs add)             (extrinsic vs sph)

Alpha gradient:
  baseline(0.0) → E5(0.3) → alpha05(0.5) → alpha10(1.0)
  → PSNR-LPIPS trade-off curve 완성
```

---

## 4. 실험 결과 (Results)

### 4.1 H6 v3: Alpha Loss — GT Input (완료 ✅)

> v3: `original_gt_mask` 보존으로 alpha/bg/mask_iou 독립 계산 (v2 bugfix)
> 모두 base_uniform_v2.yaml + M5t2 (15840 iters = 15K fwdbwd_passes)

| Experiment | alpha_w | PSNR | LPIPS ↓ | SSIM | IoU | Best Step |
|-----------|:-------:|:----:|:-------:|:----:|:---:|:---------:|
| **baseline (4view_v2)** | 0.0 | **21.82** | 0.043 | 0.947 | N/A | 9201 |
| **alpha05_v3** | 0.5 | 21.20 | 0.020 | 0.973 | 0.945 | 8901 |
| **alpha10_v3** | **1.0** | 20.84 | **0.015** | **0.974** | **0.956** | 8101 |

**시각화 (metric landscape):**
```
PSNR (higher=better):
  baseline ████████████████████████ 21.82
  alpha05  ██████████████████████   21.20
  alpha10  █████████████████████    20.84

LPIPS (lower=better):
  alpha10  ██                       0.015  ← 3x better
  alpha05  ████                     0.020
  baseline ████████████             0.043

Alpha IoU:
  alpha10  ████████████████████     0.956
  alpha05  ███████████████████      0.945
  baseline (N/A)
```

**결론**: PSNR ~1dB 희생으로 perceptual quality 2.9x 개선. **논문용 alpha=1.0 권장.**

### 4.2 H6-E2E: Alpha Loss — E2E Pipeline (완료 ✅)

> **핵심 실험**: Alpha loss의 GT 이점이 E2E pipeline으로 전이되는가?
> Setup: MVDiff baseline (ckpt-5000, sparse) + 각 GS-LRM variant
> Eval: 360 test samples, input_view_idx=0, guidance_scale=3.0

| Experiment | GS-LRM | PSNR (fw) | PSNR (fg) | SSIM | LPIPS | sIoU |
|-----------|:------:|:---------:|:---------:|:----:|:-----:|:----:|
| **baseline (H1bis M5t2)** | M5t2_E0_1 | 21.21 | 7.91 | **0.966** | **0.059** | **0.521** |
| **alpha10_v3 E2E** | alpha10_v3 | **21.24** | **7.93** | 0.965 | 0.061 | 0.474 |
| 4view_e2e (skip input) | 4view_v2 | 19.87 | 6.55 | 0.962 | 0.070 | 0.437 |

**Per-view 분석 (alpha10_v3):**

| View | PSNR (fw) | PSNR (fg) | SSIM | LPIPS | L1 | 각도 (from input) |
|:----:|:---------:|:---------:|:----:|:-----:|:--:|:-----------------:|
| 1 | **23.20** | **9.39** | **0.972** | **0.048** | 0.220 | ~60° |
| 2 | 20.56 | 7.40 | 0.961 | 0.065 | 0.305 | ~120° |
| 3 | 20.52 | 6.77 | 0.964 | 0.065 | 0.348 | ~180° (반대편) |
| 4 | 20.90 | 7.89 | 0.964 | 0.062 | 0.291 | ~240° |
| 5 | 21.05 | 8.23 | 0.963 | 0.064 | 0.303 | ~300° |

**GT → E2E 전이 분석:**

| Metric | GT: baseline | GT: alpha10 | GT 개선 | E2E: baseline | E2E: alpha10 | E2E 개선 | 전이율 |
|--------|:-----------:|:-----------:|:-------:|:-------------:|:------------:|:--------:|:------:|
| PSNR | 21.82 | 20.84 | -0.98 | 21.21 | 21.24 | +0.03 | — |
| LPIPS | 0.043 | 0.015 | **-65%** | 0.059 | 0.061 | +3% | **0%** |
| SSIM | 0.947 | 0.974 | +0.027 | 0.966 | 0.965 | -0.001 | **0%** |
| IoU | N/A | 0.956 | — | 0.521 | 0.474 | **-0.047** | **역전** |

#### 4.2.1 가설: Alpha Loss E2E 무효화 메커니즘

**H6-E2E-1: Noisy Multi-view Hypothesis (채택)**

Alpha loss는 GT mask 기반 supervision. GT input에서는 pixel-perfect mask 학습 가능.
하지만 MVDiffusion output은:
- Multi-view consistency 부족 (view 간 foreground boundary 불일치)
- 생성 아티팩트 (blurring, ghosting)
- View-dependent color shift

GS-LRM이 이런 noisy input에서 alpha를 학습하면, inconsistent alpha signal에 과적합하여
오히려 shape 품질이 하락한다.

```
GT Input:  clean views → alpha learns true shape   → IoU 0.956 ✅
E2E:       noisy views → alpha learns noise pattern → IoU 0.474 ✗
```

**H6-E2E-2: View Proximity Bias (확인됨)**

fg_PSNR의 view별 분포가 input view (cam_0)와의 각도 차이에 반비례:
```
View 1 (~60°):   9.39 dB  ← 가장 가까움
View 5 (~300°):  8.23 dB
View 4 (~240°):  7.89 dB
View 2 (~120°):  7.40 dB
View 3 (~180°):  6.77 dB  ← 가장 멀음 (반대편)
```

이것은 MVDiffusion이 input view 정보를 점진적으로 잃어가며, 반대편 뷰에서
가장 큰 hallucination을 생성한다는 것을 의미한다.
→ **Pose conditioning**이 이 angular decay를 보상할 수 있는 잠재력 확인.

**H6-E2E-3: Foreground Size Effect**

FG fraction이 전체 이미지의 2.3%로 매우 작음. 이것은:
- full_white PSNR이 높게 나오는 이유 (배경이 97.7% 지배)
- fg_only PSNR이 full_white 대비 ~13 dB 낮은 이유
- sIoU가 small object에서 더 민감한 이유 (몇 pixel 차이가 큰 IoU 변화)

### 4.3 E1: Cosine LR Schedule (실행중)

> 가설: Cosine decay가 constant LR의 oscillation 방지
> Config: `mouse_mvdiffusion_M5t2_20k_cosine.yaml`

| 설정 | 값 |
|------|-----|
| Base | scratch (pipeckpts) |
| LR | 5e-5 → cosine decay → 0 |
| Warmup | 100 steps |
| Max steps | 20,000 |
| Output | mouse_M5t2_20k_cosine |
| GPU | 4 (A6000) |

**현재 진행**: checkpoint-3000 (15%). 성공 기준: PSNR > 28.0 (baseline 27.30 대비 +0.7), 안정 수렴.

### 4.4 E2: P0 Resume with Reduced LR (실행중)

> 가설: LR 감소로 P0 oscillation 안정화
> Config: `mouse_mvdiffusion_M5t2_randref_20k_resume.yaml`

| 설정 | 값 |
|------|-----|
| Base | P0 checkpoint-10000 (resume) |
| LR | step 1-10K: 5e-5 (기학습) → step 10K+: 1e-5 (감소) |
| Max steps | 20,000 |
| Output | mouse_M5t2_randref_sparse (P0 디렉토리 재사용) |
| GPU | 7 (A6000) |

**현재 진행**: checkpoint-13000 (65%). P0의 random reference view 학습을 이어받아 LR만 감소.

### 4.5 E3: Extrinsic Pose Conditioning + Add (실행중)

> 가설: 6D rotation + 3D translation이 spherical보다 풍부한 공간 정보 제공
> Additive integration이 concat보다 효율적 (sequence length 불변)
> Config: `mouse_mvdiffusion_M5t2_pose_extrinsic_add.yaml`

| 설정 | 값 |
|------|-----|
| Base | ckpt-5000/unet (baseline MVDiff) |
| Pose method | extrinsic (6D rotation + 3D translation) |
| Integration | add (direct hidden state injection) |
| LR | 5e-5, cosine schedule |
| Max steps | 10,000 |
| Output | mouse_M5t2_pose_extrinsic_add |
| GPU | 6 (A6000) |

**Pose Conditioning 구현 상세:**

```python
# ExtrinsicPoseEncoder
rotation (B, N, 3, 3) → 6D repr (B, N, 6) → Linear → (B, N, embed_dim)
translation (B, N, 3) → Linear → (B, N, embed_dim)
combined = rotation_embed + translation_embed  # (B, N, 1024)

# Add Integration
hidden_states = hidden_states + pose_embedding  # No sequence extension
```

비교 대상:
- P1 (spherical+concat): azimuth/elevation/distance → cross-attn token 추가
- E4 (spherical+add): P1과 동일 encoding, add integration

### 4.6 E5: Alpha Loss Weight=0.3 (실행중)

> 가설: baseline(0.0)과 alpha05(0.5) 사이 최적 PSNR-LPIPS trade-off 존재
> Config: `4view_alpha03_v3.yaml` + `base_uniform_v2.yaml`

| 설정 | 값 |
|------|-----|
| Base | base_uniform_v2 (GS-LRM 표준) |
| alpha_loss_weight | 0.3 |
| Max fwdbwd_passes | 15,000 (→ 15,840 actual) |
| Output | base_uniform_v2_4view_alpha03_v3 |
| GPU | 5 (A6000) |

**예상 결과 (interpolation):**

| alpha_w | PSNR (예상) | LPIPS (예상) |
|:-------:|:-----------:|:----------:|
| 0.0 | 21.82 | 0.043 |
| **0.3** | **~21.5** | **~0.028** |
| 0.5 | 21.20 | 0.020 |
| 1.0 | 20.84 | 0.015 |

---

## 5. 종합 분석 (Cross-Experiment Analysis)

### 5.1 E2E 병목 구조

```
                          ┌─────────────────────────┐
                          │   E2E Performance Cap    │
                          │   = min(Stage1, Stage2)  │
                          └────────────┬────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
      Stage 1 (MVDiff)         Stage 2 (GS-LRM)         Interaction
      ═══════════════         ════════════════         ════════════
      val PSNR: 27.3          GT PSNR: 21.8            E2E PSNR: 7.9
      ↓ bottleneck            ✓ adequate               ↓ dominated
                                                         by Stage 1
```

**정량적 증거:**

| 변경 | Stage 2 GT 효과 | E2E 효과 | 전이율 |
|------|:--------------:|:--------:|:------:|
| Alpha loss (0→1) | LPIPS 2.9x ↑ | 0% | **0%** |
| Alpha loss (0→1) | IoU +0.956 | IoU -0.047 | **역전** |
| Attention (sparse→full) | — | PSNR +0.08 | ~0% |
| 학습 데이터 (1:1:1→8:1:1) | — | PSNR +2.9 dB | 높음 |

→ **Stage 2 개선은 E2E로 전이되지 않음. Stage 1 또는 데이터 개선만 유효.**

### 5.2 개선 축 우선순위 (Updated)

| 순위 | 축 | 실험 | 기대 근거 | Phase 3 상태 |
|:----:|:--:|------|----------|:----------:|
| **1** | LR 최적화 | E1, E2 | P0 oscillation 해결 | 🔄 실행중 |
| **2** | Pose conditioning | E3, E4 | View proximity bias 보상 | 🔄 E3 시작 |
| **3** | Longer training | E1 (20K) | 5K = undertrained | 🔄 E1에 포함 |
| 4 | Alpha gradient | E5 | Trade-off curve 완성 | 🔄 실행중 |
| ~~5~~ | ~~Attention type~~ | ~~H5~~ | ~~무효 확인~~ | ✗ 종료 |
| ~~6~~ | ~~Alpha E2E~~ | ~~H6-E2E~~ | ~~전이 불가 확인~~ | ✗ 종료 |

### 5.3 아직 미검증 가설

| ID | 가설 | 검증 방법 | 우선순위 |
|:--:|------|----------|:--------:|
| H7 | SSIM loss weight 조절로 structural quality 개선 | SSIM=0.3/0.5/1.0 ablation | P3 |
| H8-ext | 3-view MVDiff fine-tune로 per-view quality 향상 | H4 E2E 확장 | P2 |
| H9 | Alpha + Pose 조합이 E2E에서 시너지 | E3/E4 완료 후 | P2 |
| H10 | Progressive training (MVDiff→GS-LRM joint) | 파이프라인 재설계 | P4 |

---

## 6. 기술적 세부사항 (Technical Details)

### 6.1 서버 환경

| 항목 | 값 |
|------|-----|
| Server | gpu03 (ssh via ProxyJump storage) |
| GPU | NVIDIA RTX A6000 × 4 (GPU 4-7), 49GB each |
| Framework | PyTorch + Accelerate + WandB |
| Conda env | `facelift` (Python 3.11) |
| Data | M5t2: 3600 frames, 80:10:10 split, 6 cameras |
| Checkpoint root | `/node_data/joon/checkpoints/FaceLift` (symlink) |

### 6.2 Metric Protocol v2

| Metric | 정의 | 용도 |
|--------|------|------|
| `psnr_full_white` | 흰 배경 포함 전체 이미지 PSNR | 논문 보고 (literature standard) |
| `psnr_fg_only` | GT mask 내부만 PSNR | 순수 foreground quality |
| `ssim_full_white` | 전체 이미지 SSIM | 구조적 유사도 |
| `lpips_full_white` | AlexNet 기반 perceptual distance | 지각적 품질 |
| `silhouette_iou` | Binary mask IoU | Foreground shape accuracy |
| `masked_l1` | GT mask 영역 L1 loss | 색상 정확도 |

### 6.3 Alpha Loss Bug (v2→v3)

**발견일**: 2026-02-13
**증상**: alpha_loss=0 항상, mask_iou=0 항상
**원인**: `mask_mode: none` → `compute_mask_from_config()` → GT mask=None → alpha_loss 계산 불가
**수정**: `gslrm.py`에 `original_gt_mask` 변수 보존, alpha/bg/mask_iou loss가 독립적으로 GT mask 사용
**교훈**: Loss 컴포넌트는 config-driven masking과 독립된 데이터 경로를 가져야 함

---

## 7. 다음 단계 (Next Steps)

### 즉시 (GPU 가동 중)
- [ ] E1/E2 수렴 모니터링 (WandB 확인)
- [ ] E3 pose injection 검증 (초기 validation loss 확인)
- [ ] E5 alpha=0.3 중간 결과 확인

### 단기 (E1/E2 완료 후)
- [ ] E1 vs E2 LR 전략 비교 (cosine vs piecewise)
- [ ] 승자 checkpoint로 E2E 평가
- [ ] E4 (spherical+add) 실행 → E3 vs E4 비교
- [ ] E5 완료 → alpha gradient curve (0.0/0.3/0.5/1.0)

### 중기
- [ ] E3/E4 결과 기반 최적 pose 전략 확정
- [ ] Alpha + Pose 조합 실험 (H9)
- [ ] MVDiff 20K checkpoint E2E 평가 (vs 5K baseline)

---

## Appendix A: Checkpoint 전체 목록

### GS-LRM (Stage 2)

| Name | Views | Alpha | PSNR | Step | Data |
|------|:-----:|:-----:|:----:|:----:|:----:|
| base_uniform_v2_6view_v2 | 6 | 0 | **24.49** | 4201 | M5t2 |
| base_uniform_v2_5view_v2 | 5 | 0 | 23.02 | 13101 | M5t2 |
| M5t2_E0_1_facelift | 4 | 0 | 22.34 | 8001 | M5t2 |
| base_uniform_v2_4view_v2 | 4 | 0 | 21.82 | 9201 | M5t2 |
| base_uniform_v2_4view_alpha05_v3 | 4 | 0.5 | 21.20 | 8901 | M5t2 |
| base_uniform_v2_4view_alpha10_v3 | 4 | 1.0 | 20.84 | 8101 | M5t2 |
| base_uniform_v2_4view_alpha03_v3 | 4 | 0.3 | 🔄 | — | M5t2 |
| base_uniform_v2_3view_v2 | 3 | 0 | 20.01 | 10701 | M5t2 |
| base_uniform_v2_2view_v2 | 2 | 0 | 17.75 | 11801 | M5t2 |
| base_uniform_v2_1view_v2 | 1 | 0 | 11.08 | 2401 | M5t2 |

### MVDiffusion (Stage 1)

| Name | Steps | LR | Features | Status |
|------|:-----:|:--:|----------|:------:|
| mouse_M5t2 (baseline) | 5K | 5e-5 const | sparse, fixed ref | ✅ |
| mouse_M5t2_cfgr | 10K | 5e-5 const | full attn | ✅ |
| mouse_M5t2_randref_sparse (P0) | 10K | 5e-5 const | random ref | ✅ |
| mouse_M5t2_pose_spherical (P1) | 10K | 5e-5 const | spherical+concat | ✅ |
| mouse_M5t2_20k_cosine (E1) | 20K | 5e-5 cosine | sparse | 🔄 3K |
| mouse_M5t2_randref_sparse (E2) | 20K | 5e-5→1e-5 | random ref, resume | 🔄 13K |
| mouse_M5t2_pose_extrinsic_add (E3) | 10K | 5e-5 cosine | extrinsic+add | 🔄 시작 |

## Appendix B: E2E Evaluation Commands

```bash
# E2E inference
export CUDA_VISIBLE_DEVICES=N && python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --mvdiffusion_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/{MVDIFF}/checkpoint-{N} \
    --gslrm_checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/{GSLRM}/best_psnr.pt \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt \
    --output_dir outputs/{NAME} \
    --input_view_idx 0 \
    --guidance_scale 3.0

# Metrics computation
python -m mouse_extensions.scripts.eval.compute_e2e_metrics \
    --output_dir outputs/{NAME} \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --skip_input_view 0
```

---

*FaceLift Phase 3 Report | 2026-02-16 | gpu03*

## Addendum: MVDiffusion Training Analysis (260216 02:00 KST)

> See:  for full WandB validation curves

### Key Finding: Baseline was severely under-trained

| Experiment | Best Val PSNR (cfg=3.0) | LPIPS | Δ vs baseline |
|-----------|------------------------|-------|---------------|
| Baseline (ckpt-5000) | 24.11 | 0.0352 | — |
| E1 cosine (step 4000, 22%) | 26.62 | 0.0248 | **+2.51 dB** |
| E2 resume (step 14400, 73%) | 27.70 | 0.0206 | **+3.59 dB** |

- E1: Cosine LR이 같은 step 수에서 ~2.5x 효율적인 수렴
- E2: Resume + reduced LR로 절대 성능 최고 (27.70 PSNR, 0.021 LPIPS)
- **E2E 전이율 측정 예정**: E2 완료 후 auto-chain으로 E2E eval 자동 실행



## Addendum: MVDiffusion Training Analysis (260216 02:00 KST)

> See: 260216_MVDIFF_TRAINING_ANALYSIS.md for full WandB validation curves

### Key Finding: Baseline was severely under-trained

| Experiment | Best Val PSNR (cfg=3.0) | LPIPS | Delta vs baseline |
|-----------|------------------------|-------|---------------|
| Baseline (ckpt-5000) | 24.11 | 0.0352 | -- |
| E1 cosine (step 4000, 22%) | 26.62 | 0.0248 | **+2.51 dB** |
| E2 resume (step 14400, 73%) | 27.70 | 0.0206 | **+3.59 dB** |

- E1: Cosine LR이 같은 step 수에서 약 2.5x 효율적인 수렴
- E2: Resume + reduced LR로 절대 성능 최고 (27.70 PSNR, 0.021 LPIPS)
- **E2E 전이율 측정 예정**: E2 완료 후 auto-chain으로 E2E eval 자동 실행
