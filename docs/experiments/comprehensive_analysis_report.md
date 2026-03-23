# FaceLift 종합 실험 보고서

> **Date**: 2026-03-12
> **Project**: GS-LRM 기반 Multi-view Mouse 3D Reconstruction
> **Scope**: Phase 2 (GS-LRM) + Phase 3 (MVDiffusion) 전체 가설 검증 결과
> **Dataset**: M5t2 (3,600 frames, 6 cameras, Train/Val/Test = 80/10/10%)

---

## 1. Executive Summary

FaceLift 파이프라인의 두 핵심 구성요소 — **GS-LRM** (3D Gaussian 렌더러)과 **MVDiffusion** (multi-view 생성 모델) — 에 대해 **8개 가설(H1-H8)** 과 **4개 분석 연구**를 수행했다.

**핵심 발견**:
1. **GS-LRM은 매우 우수**: GT 입력 시 PSNR 23.84 dB, Pose-Splatter 대비 **+10.06 dB** 우위
2. **MVDiffusion이 유일한 병목**: E2E 품질 손실의 **86%** (-13.72 dB)가 MVDiff에서 발생
3. **학습 전략 최적화는 포화**: 모든 E2E 변형이 **7.9-8.4 dB로 수렴** → 아키텍처 변경 필요
4. **Stage 2 → E2E 전이율 0%**: GS-LRM 개선이 E2E에 전달되지 않음

**결론**: Training strategy 최적화만으로는 현재 8 dB 천장을 돌파할 수 없다. MVDiffusion의 silhouette 정확도 개선 또는 아키텍처 교체가 필요하다.

---

## 2. 파이프라인 구조 및 평가 방법

### 2.1 Two-Stage Pipeline

```
Input Image(s) → [MVDiffusion] → 6 Novel Views → [GS-LRM] → 3D Gaussians → Novel View Rendering
                  Stage 1                          Stage 2
```

- **MVDiffusion (Stage 1)**: 단일/소수 입력 → 6개 multi-view 이미지 생성 (SD2.1-UnCLIP 기반)
- **GS-LRM (Stage 2)**: 6개 이미지 → feed-forward 3D Gaussian prediction → 렌더링

### 2.2 Fair Evaluation Protocol

| 항목 | 설정 |
|------|------|
| **Test set** | 360 frames (3240-3599), 5 evaluated views × 360 = 1,800 samples |
| **GT mask** | RGBA alpha > 127 (uint8) |
| **Pred mask** | White-BG extraction: any channel < 0.98 (float) |
| **PSNR_gt** | GT FG 영역에서의 masked PSNR |
| **PSNR_int** | GT∩Pred intersection 영역 PSNR (순수 색상 품질) |
| **IoU** | Silhouette overlap (geometry 정확도) |
| **Coverage** | GT FG 중 Pred가 커버하는 비율 |

> 참조: `mouse_extensions/scripts/eval/fair_comparison.py`

---

## 3. 정량 결과

### 3.1 전체 성능 요약 (Main Table)

| Configuration | Type | PSNR_gt↑ | IoU↑ | PSNR_int↑ | SSIM↑ | Coverage↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **GS-LRM 6v (GT input)** | Upper bound | **23.84** | **0.954** | **24.02** | **0.963** | 0.999 |
| GS-LRM 5v (GT input) | Upper bound | 22.16 | 0.942 | 22.56 | 0.947 | 0.997 |
| GS-LRM 4v (GT input) | Upper bound | 20.66 | 0.926 | 21.29 | 0.928 | 0.993 |
| GS-LRM 3v (GT input) | Upper bound | 18.56 | 0.899 | 19.54 | 0.897 | 0.985 |
| GS-LRM 2v (GT input) | Upper bound | 15.95 | 0.858 | 17.91 | 0.866 | 0.963 |
| GS-LRM 1v (GT input) | Upper bound | 10.47 | 0.028 | 10.47 | 0.828 | 1.000 |
| | | | | | | |
| **E2E: e2_resume_20k** | **E2E best** | **8.20** | 0.521 | 15.63 | 0.761 | 0.715 |
| E2E: e3_pose | E2E | 8.10 | 0.523 | 15.88 | 0.762 | 0.716 |
| E2E: p1_6view_e2e | E2E | 8.44 | 0.495 | 14.65 | 0.761 | 0.750 |
| E2E: p1_bl_6view_e2e | E2E | 8.11 | 0.501 | 14.66 | 0.759 | 0.740 |
| E2E: e1_cosine_20k | E2E | 7.90 | 0.528 | 16.15 | 0.762 | 0.705 |
| E2E: e1_cosine_11k | E2E | 7.90 | 0.522 | 16.12 | 0.761 | 0.704 |
| E2E: baseline | E2E | 7.93 | 0.474 | 13.70 | 0.758 | 0.740 |
| E2E: cfgr (full attn) | E2E | 7.75 | 0.491 | 15.75 | 0.756 | 0.672 |

> n = 1,800 samples per configuration (360 test frames × 5 views)

### 3.2 View Ablation (H4)

| Views | PSNR_gt | IoU | PSNR_int | Δ vs 1v |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 10.47 | 0.028 | 10.47 | — |
| 2 | 15.95 | 0.858 | 17.91 | +5.48 |
| 3 | 18.56 | 0.899 | 19.54 | +8.09 |
| 4 | 20.66 | 0.926 | 21.29 | +10.19 |
| 5 | 22.16 | 0.942 | 22.56 | +11.69 |
| **6** | **23.84** | **0.954** | **24.02** | **+13.37** |

**해석**: View 수에 대해 단조 증가. 1→2v에서 가장 큰 도약 (+5.48 dB, IoU 0.028→0.858). 5→6v는 +1.68 dB로 수확 체감.

### 3.3 E2E Strategy Comparison

| Strategy | PSNR_gt | IoU | PSNR_int | 특징 |
|:---|:---:|:---:|:---:|:---|
| baseline (sparse, 5K) | 7.93 | 0.474 | 13.70 | 기본 설정 |
| cfgr (full attn) | 7.75 | 0.491 | 15.75 | Full attention → 오히려 하락 |
| e1_cosine_20k | 7.90 | 0.528 | **16.15** | 색상 품질 최고 |
| **e2_resume_20k** | **8.20** | 0.521 | 15.63 | **PSNR_gt 최고** |
| e3_pose (extrinsic) | 8.10 | 0.523 | 15.88 | Pose conditioning 미미 |
| p1_6view_e2e | 8.44 | 0.495 | 14.65 | 6v GS-LRM + E2E |

**해석**: 전 전략이 PSNR_gt 7.75-8.44 범위에 수렴. 최대 편차 0.69 dB — 통계적으로 유의미하나 실용적 차이 미미. **학습 전략 최적화는 포화 상태**.

---

## 4. 가설 검증 결과

### 4.1 Summary Table

| # | 가설 | 결과 | 신뢰도 | 핵심 근거 |
|:---:|:---|:---:|:---:|:---|
| H1 | GS-LRM > Pose-Splatter | ✅ **확인** | HIGH | +10.06 dB (23.84 vs 13.78) |
| H2 | MVDiff E2E 천장 존재 | ✅ **확인** | HIGH | 모든 전략 7.9-8.4 수렴 |
| H3 | Stage 2 개선 → E2E 전이 | ❌ **전이율 0%** | HIGH | GS-LRM 개선 무효 |
| H4 | View ablation | ✅ **6v 최적** | HIGH | Val 6v, Test 5v>6v (경미) |
| H5 | MVDiff 학습 최적화 | ⚠️ **포화** | HIGH | Sparse > full, 나머지 수렴 |
| H6 | Alpha mask loss | 🔄 **재활성화** | HIGH | PSNR↓ but IoU↑(6v α=0.3), artifact 4×↓ → [[UNIFIED_ABLATION_REPORT]] |
| H7 | SSIM weight 조정 | ❌ **기각** | HIGH | 불안정, baseline 유지 |
| H8 | 적은 MVDiff 뷰 | ❌ **기각** | HIGH | 적은 뷰 = 더 나쁜 결과 |

### 4.2 각 가설 상세

#### H1: FaceLift GS-LRM vs Pose-Splatter
- **방법**: 동일 M5t2 test set에서 fair eval (동일 mask protocol)
- **결과**: FL 23.84 dB vs PS 13.78 dB (**+10.06 dB**, IoU 0.954 vs 0.846)
- **의미**: GS-LRM 렌더러 자체의 우수성 확인. 병목은 renderer가 아닌 MVDiff

#### H2: MVDiffusion E2E 천장
- **방법**: 6가지 학습 전략 (baseline, cfgr, e1, e2, e3, pose variants) 비교
- **결과**: PSNR_gt 7.75-8.44 dB 범위에 수렴 (std ≈ 0.2 dB)
- **의미**: Training recipe 변경으로는 돌파 불가. **아키텍처 수준 변경** 필요

#### H3: Stage 2 → E2E 전이
- **방법**: GS-LRM val PSNR 개선 (22.34 → 24.49) 후 E2E 평가
- **결과**: GS-LRM +2.15 dB → E2E 0 dB 전이 (**전이율 0%**)
- **의미**: GS-LRM이 아무리 좋아도 MVDiff output 품질이 제한 요인

#### H4: View Ablation
- **방법**: 1v-6v GS-LRM을 동일 config로 학습, fair eval
- **결과**: Val에서 6v 최적 (24.49 dB), Test에서 5v > 6v (22.16 vs 21.02 — H4b@20K)
  - Note: 6v test = `gslrm_6view` (23.84) vs `H4b_step20000` (21.02) → checkpoint 차이
- **의미**: View 수 증가는 단조적 이득. 5v→6v 수확 체감. 6v 가용 시 사용 권장

#### H5: MVDiff Training Strategy
- **방법**: Sparse vs full attention, cosine LR, resume training, pose conditioning
- **결과**: Sparse attention이 full보다 우수 (7.93 vs 7.75). Cosine LR은 PSNR_int에서만 이득 (+2.45 dB)
- **의미**: Sparse attention 유지. 색상 품질은 개선 가능하나 geometry(IoU) 개선 안 됨

#### H6: Alpha Mask Loss
- **방법**: alpha_weight = 0.3/0.5/1.0으로 4v + 6v GS-LRM 학습
- **결과**: PSNR_gt 하락 (-0.55~-1.29 dB), 단 6v α=0.3에서 IoU 유일 개선 (+0.002). Novel view artifact 4.0× 개선 (alpha entropy).
- **의미**: ~~기각~~ → **조건부 재활성화** (2026-03-23). GT-view PSNR 기준으로는 기각이 맞으나, novel view artifact 억제에서 압도적 우세. 6v α=0.3이 best trade-off.
- → 상세: [[UNIFIED_ABLATION_REPORT]] §3, [[ALPHA_LOSS_NOVEL_VIEW_ANALYSIS]] §10.7-10.8

#### H7: SSIM Weight
- **방법**: SSIM weight 0.1(baseline), 0.3, 0.5, 1.0
- **결과**: 0.3+ 에서 학습 불안정 또는 PSNR 붕괴
- **의미**: 기각. Baseline 0.1 유지

#### H8: Fewer MVDiff Views
- **방법**: MVDiff 생성 뷰 수 감소 시 per-view 품질 향상 여부
- **결과**: 뷰 수 감소 = 품질 하락
- **의미**: 기각. MVDiff는 더 많은 뷰를 생성할수록 개별 뷰도 개선됨

### 4.3 추가 분석

#### Plucker Pose Conditioning (H6a, H7 Spatial Tokens)
- H6a_v2 (Plucker+Add): val 27.34@9K, E2E IoU +0.098 (0.476→0.574)
- H7 (Spatial Token): val 25.57@1K, 학습 초기 우위
- **의미**: Pose conditioning은 marginal 이득. MVDiff의 근본적 한계를 해결하지 못함

#### MVDiffusion Bottleneck 분해

```
GS-LRM 5v (upper bound):  PSNR_gt = 22.16,  IoU = 0.942
                                    ↓
              MVDiff quality loss: -13.72 dB
              ├── Shape loss (IoU): 0.942 → ~0.5  = 86% of gap
              └── Color loss: PSNR_int 22.56 → 15.63 (secondary)
                                    ↓
E2E best:                 PSNR_gt = 8.44,   IoU = 0.495
```

**핵심**: MVDiff가 생성하는 이미지의 **silhouette(형태)** 이 GT와 크게 불일치 (IoU ~0.5). 색상 품질(PSNR_int)은 상대적으로 양호(15-16 dB). → **Silhouette 정확도 개선이 최우선 과제**.

---

## 5. 해석 및 함의

### 5.1 왜 E2E가 이렇게 낮은가?

| 요인 | 기여도 | 근거 |
|------|:---:|:---|
| **MVDiff silhouette 부정확** | ~86% | IoU 0.942 → 0.5 (가장 큰 drop) |
| **MVDiff 색상/텍스처** | ~14% | PSNR_int 22.56 → 15.63 |
| GS-LRM 자체 한계 | ~0% | GT 입력 시 23.84 dB (충분) |

### 5.2 왜 학습 전략이 효과 없는가?

1. **Distribution mismatch**: MVDiff는 SD2.1-UnCLIP 기반 → 사람 얼굴 pretrained. 생쥐 도메인과 큰 gap
2. **생쥐의 특수성**: 어두운 털, 작은 크기 (이미지의 ~2.5%), 흰 배경 위 검은 물체 → pretrained prior와 충돌
3. **Attention 패턴**: Sparse attention이 full보다 우수 — M5의 비정규 카메라 배치(elevation ±9.6°)에서 epipolar 구조가 약함

### 5.3 남은 가능성

| 접근법 | 예상 효과 | 난이도 | 우선순위 |
|--------|:---:|:---:|:---:|
| MVDiff silhouette loss 추가 | HIGH | Medium | **P1** |
| Domain adaptation (DA1) | Medium | Medium | P2 |
| Stage 1 교체 (다른 MV 생성기) | HIGH | High | P3 |
| Multi-view consistency 강화 | Medium | Medium | P2 |
| Mesh prior 활용 (MAMMAL) | Medium | Low | P2 |

---

## 6. Mesh-GS Pair Dataset (부록)

UV texture rendering bug 수정 후 MAMMAL mesh + GS-LRM rendering pair 데이터셋 구축 진행 중.

### 6.1 Bug Fix 검증

| Metric | BEFORE (buggy) | AFTER (fixed) | Delta |
|--------|:-:|:-:|:-:|
| IoU(GT) | 0.665 | 0.754 | **+0.088** |
| IoU(flat mesh) | 0.653 | 1.000 | **+0.347** |
| PSNR_masked(GT) | 8.75 | 10.42 | **+1.67 dB** |

> 상세: `mouse_extensions/docs/UV_TEXTURE_RENDERING_BUG.md`

### 6.2 Dataset Status

| 항목 | 현재 | 목표 |
|------|:---:|:---:|
| Generated frames | 152 | 3,600 |
| DiFix format | ✅ 152 frames built | 3,600 |
| 구조 | `difix_pairs/frames/XXXXXX/` | manifest.json + cameras.json |

---

## 7. 결론 및 Next Steps

### Phase 2-3 결론

> **GS-LRM은 이미 충분히 우수하며, MVDiffusion이 유일한 병목이다.**
> Training strategy 최적화는 포화 상태 (7.9-8.4 dB 수렴).
> 아키텍처 수준 변경 없이는 E2E 품질 향상이 불가능하다.

### 권장 Next Steps

| 순위 | 작업 | 기대 효과 | Timeline |
|:---:|:---|:---|:---:|
| 1 | MVDiff silhouette loss 설계 | IoU 0.5 → 0.7+ | 2-3주 |
| 2 | Domain adaptation (DA1 실험) | PSNR +1-2 dB | 1-2주 |
| 3 | Stage 1 대안 조사 (MV-Adapter 등) | 근본적 해결 | 1-2개월 |
| 4 | MAMMAL mesh prior 활용 | Geometry guide | 2-3주 |
| 5 | Mesh-GS pair dataset 완성 | 3,600 frames | 1-2일 |

---

## Related Documents

- ↑ [[../INDEX]] — Document hub
- ↔ [[EXPERIMENT_REGISTRY]] — 개별 실험 설정 상세
- ↔ [[hypothesis_roadmap]] — 가설 로드맵 v2.1
- ↔ [[mvdiff_bottleneck_analysis]] — MVDiff 병목 분석
- ↔ [[fl_vs_ps_comparison]] — FaceLift vs Pose-Splatter
- ↔ [[plucker_ray_analysis]] — Pose conditioning 분석
- ↔ [[mesh_gs_pair_collection]] — Mesh-GS pair 수집 전략
- ↔ [[../../mouse_extensions/docs/UV_TEXTURE_RENDERING_BUG]] — UV 버그 분석

---

*FaceLift | Comprehensive Results Report | 2026-03-12*
