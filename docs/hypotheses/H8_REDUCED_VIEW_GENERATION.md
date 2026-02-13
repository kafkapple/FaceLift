# H8: Reduced View Generation (뷰 수 감소 생성)

> **가설**: MV-Diffusion 생성 뷰 수를 6→3~4로 줄이면, per-view 일관성과 품질이 개선되어
> downstream GS-LRM 재구성 품질이 향상될 것이다.
>
> ← [RESEARCH_HYPOTHESES.md](../RESEARCH_HYPOTHESES.md) | **상태**: 🔄 3/4-view E2E 완료 | **Updated**: 2026-02-11

---

## 1. 동기

### 1.1 현재 병목: MV-Diffusion 품질

H3 진단 결과, MVDiffusion이 E2E 파이프라인의 병목:
- M5t (1198 train): GS-LRM(GT) 20.56 vs E2E 19.15 → **Gap +1.41 dB**
- M5t2 (2880 train): Gap ≈ 0 → 충분한 학습으로 해소 가능

**문제**: 6개 뷰를 동시에 생성할 때 **뷰 간 inconsistency**가 발생.
예: 한 뷰에서 귀가 보이는데 인접 뷰에서 사라지거나, 텍스처 불일치.
이 inconsistency가 GS-LRM 3D 재구성에서 **ghosting artifact**로 이어짐.

### 1.2 핵심 아이디어

> "6개를 제대로 일관성 있게 생성하지 못할 바에는, 수를 줄여서 더 잘 만들자"

```
현재:  1장 입력 → [MVDiff 6뷰] → [GS-LRM 4입력/6타겟] → 3D
                    ↑ 병목: 6뷰 일관성 부족

제안:  1장 입력 → [MVDiff 3~4뷰] → [GS-LRM 3~4입력/6타겟] → 3D
                    ↑ 적은 뷰 = 높은 per-view 품질
```

---

## 2. 문헌 근거

### 2.1 모델별 생성 뷰 수 비교

| 모델 | 뷰 수 | 일관성 메커니즘 | 비고 |
|------|-------|----------------|------|
| MVDream | **4** | 3D Self-Attention | 4뷰 = Janus 해결 임계점 |
| Zero123++ | 6 | Tiled 3×2 Joint | 산업 표준 |
| Era3D | 6 | Row-wise Attention | Dense보다 품질↑ |
| Wonder3D | 6 | Cross-Domain Attention | RGB + Normal |
| Instant3D | **4** | Tiled 2×2 Joint | 최소 충분 |
| **LGM** | **4** | Multi-view Gaussian | ⭐ 4뷰로 SOTA |
| **GRM** | **4** | Pixel-aligned Transformer | ⭐ 4뷰로 SOTA |
| CAT3D | 5-7/batch | Autoregressive | NeurIPS 2024 Oral |
| SV3D | 21 | Video Temporal | 과도하게 많음 |

**관찰**: LGM, GRM 등 최신 SOTA 재구성 모델은 **4뷰 입력**으로 설계됨.

### 2.2 핵심 증거: Fewer Views = Better Quality

**증거 1: Era3D — Attention Token 감소 → 품질 향상**

| Attention | Memory | PSNR | LPIPS |
|-----------|--------|------|-------|
| Dense (전체 뷰) | 35.3 GB | 20.73 | 0.140 |
| **Row-wise (제한)** | **1.7 GB** | **20.92** | **0.137** |

논문: *"reducing the number of attention tokens, allowing the model to focus on valuable tokens"*
→ **token 수 감소가 품질을 향상**시킬 수 있음

**증거 2: InstantMesh — 적은 뷰가 inconsistency 완화**

논문: *"it is practical to use less input views for reconstruction, which can alleviate the multi-view inconsistency issue"*
→ Diffusion inconsistency가 심하면 **차라리 적은 뷰가 나음**

**증거 3: MVDream — 4뷰 = Janus 해결 임계점**

| Views | Janus Problem |
|-------|---------------|
| 1 | Severe |
| 2 | Greatly reduced |
| **4** | **Barely any** |

→ 4뷰면 multi-face 문제 해결에 충분

**증거 4: CAT3D — 동시 생성 뷰 수와 품질**

| 설정 | PSNR |
|------|------|
| 3 cond + 1 target | 18.85 |
| 3 cond + 5 target | 21.66 |

→ 동시 생성 뷰가 많을수록 개별 품질도 향상 (attention이 감당 가능한 범위 내에서)
→ 단, *"not all views may be 3D consistent"* — 과하면 일관성 저하

**증거 5: MEt3R Consistency 벤치마크**

| 방식 | Consistency Score (↓) |
|------|----------------------|
| 3D prior (동시 생성) | 0.026 |
| Cross-view attention | 0.036 |
| Sequential | 0.069 |
| Independent | 0.120 |

→ 동시 생성이 일관성에 유리하나, **뷰가 많을수록 유지 어려움**

### 2.3 Correspondence Attention의 O(N²) 문제

MVDiffusion의 Correspondence-Aware Attention:
```
복잡도 = O(F² × (hw)²)
F = 뷰 수, h×w = feature map 해상도

6뷰: 36 × (hw)² attention pairs
4뷰: 16 × (hw)² attention pairs  (-56%)
3뷰:  9 × (hw)² attention pairs  (-75%)
```

뷰를 줄이면 attention 용량이 **적은 뷰에 집중** → per-view 품질 개선 기대

### 2.4 반론: 뷰 수 감소의 위험

| 위험 | 근거 | 심각도 |
|------|------|--------|
| Coverage 부족 | Pose Splatter: 6cam(IoU 0.87) >> 4cam(0.72) | ⚠️ 높음 |
| Pretrained weight 불일치 | 6뷰로 pretrained → 3뷰 fine-tune 시 attention 패턴 변화 | ⚠️ 중간 |
| 정보량 감소 | 3뷰는 object 뒷면 정보 부족 | ⚠️ 중간 |

**핵심 반론**: Pose Splatter ablation에서 뷰 수 감소의 품질 하락이 크다.
그러나 이는 **GT 이미지** 기준이며, MV-Diffusion **생성** 이미지는 이미 불완전.
→ "완벽한 6뷰" vs "덜 완벽한 3뷰"가 아닌, **"불완전한 6뷰" vs "덜 불완전한 3뷰"** 비교.

---

## 3. 기술적 실현 가능성

### 3.1 MV-Diffusion 아키텍처 분석

`num_views` 파라미터는 **config에서 변경 가능**:
```yaml
n_views: 3  # 현재 6 → 3으로 변경
```

코드 상 `rearrange(key, "(b t) d c -> b (t d) c", t=num_views)` — t에 어떤 값이든 가능.

**변경 필요 사항:**
1. Config: `n_views: 3` (or 4)
2. Prompt embeddings: `[6, 77, 1024]` → `[3, 77, 1024]` (뷰 방향 선택)
3. Dataset: 6뷰 중 선택한 3뷰만 로드
4. Pretrained checkpoint: 6뷰 기반 → fine-tune 필요 (from scratch 아님)

### 3.2 GS-LRM 연동

GS-LRM은 `num_input_views`와 `num_views` (total targets)가 **독립**:
```yaml
model:
  num_views: 6           # target rendering (GT 6뷰 사용)
  num_input_views: 3     # MV-Diffusion 생성 3뷰 입력
```

**Loss 계산**: Input 3뷰 → 3DGS 생성 → GT 6뷰와 비교.
Holdout 3뷰(GT)로 일반화 능력 평가 가능.

**단, 주의**: `target_has_input: true` (기본) 시 input 뷰도 target에 포함.
- 3 input + 3 holdout = 6 total target → **문제 없음**
- Holdout 뷰에서의 PSNR이 진정한 일반화 지표

### 3.3 뷰 선택 전략

6개 카메라 배치:
```
        cam_0 (top-front)
cam_5 (front-left)   cam_1 (front-right)

        cam_3 (top-back)
cam_4 (top-left)     cam_2 (top-right)
```

| 전략 | 선택 뷰 | Coverage | 장점 |
|------|---------|----------|------|
| **균등 분산 3뷰** | 0, 2, 4 | 120° 간격 | 최대 coverage |
| **전면 집중 3뷰** | 0, 1, 5 | 전면 120° | 얼굴 디테일 |
| **최적 4뷰** | 0, 1, 3, 4 | 대각선 | LGM/GRM 스타일 |
| **H4 최적 3뷰** | H4 결과 기반 | 데이터 기반 | 실험적 결정 |

→ **균등 분산 3뷰 (0, 2, 4)** 가 가장 합리적인 시작점

---

## 4. 실험 설계

### 4.1 전제 조건 (H4 결과 필요)

| 전제 | 확인 방법 | 결과 |
|------|----------|------|
| GS-LRM 3뷰 GT 성능 | H4 3view_v2 PSNR | ⏳ 실행 중 |
| GS-LRM 4뷰 GT 성능 | H4 4view_v2 PSNR | ⏳ 실행 중 |
| 3뷰 vs 4뷰 차이 | PSNR gap | ⏳ |

**진행 조건**: H4에서 3view PSNR이 4view 대비 -2dB 이내면 3뷰 생성 실험 가치 있음.
만약 3view << 4view (>3dB 차이) → 4뷰 생성으로 방향 전환.

### 4.2 단계별 실험

| 단계 | 실험 | 목적 | 의존성 |
|------|------|------|--------|
| **S0** | H4 결과 분석 | 전제 검증 | H4 완료 |
| **S1** | View selection ablation | 어떤 3뷰가 최적? | S0 |
| **S2** | MVDiff n_views=4 fine-tune | 4뷰 생성 품질 | S1 |
| **S3** | MVDiff n_views=3 fine-tune | 3뷰 생성 품질 | S1 |
| **S4** | E2E: MVDiff→GS-LRM 통합 | 최종 평가 | S2/S3 |

### 4.3 S1: View Selection (GS-LRM only, GT 이미지)

H4에서 `random_view_selection: true`로 학습 중.
추가 실험: **고정 뷰 조합 비교** (GT 이미지 사용, MV-Diffusion 불필요)

| Config | Input Views | 뷰 번호 | 설명 |
|--------|-------------|---------|------|
| 3view_spread_v2 | 3 | [0, 2, 4] | 120° 균등 분산 |
| 3view_front_v2 | 3 | [0, 1, 5] | 전면 집중 |
| 3view_diag_v2 | 3 | [0, 2, 3] | 전면+후면 대각 |
| 4view_spread_v2 | 4 | [0, 1, 3, 4] | 90° 균등 분산 |

### 4.4 S2-S3: MV-Diffusion Fine-tune

**기본 변경:**
```yaml
# n_views: 4 또는 3
n_views: 4
# 새 prompt embeddings 필요
prompt_embed_path: mvdiffusion/data/mouse_prompt_embeds_4view_1024/clr_embeds.pt
```

**학습 전략:**
1. 기존 6뷰 pretrained checkpoint에서 시작
2. n_views=4 (또는 3)로 fine-tune
3. Attention layer의 weight는 대부분 재사용 가능 (num_views는 reshape에만 사용)

### 4.5 평가 메트릭

| 메트릭 | 의미 | 측정 위치 |
|--------|------|----------|
| MVDiff per-view PSNR | 생성 뷰 품질 (vs GT) | S2/S3 |
| MVDiff consistency | 뷰 간 일관성 | S2/S3 |
| GS-LRM holdout PSNR | 3D 재구성 일반화 | S4 |
| GS-LRM mask_iou | Shape 정확도 | S4 |

---

## 5. 예상 결과 시나리오

### 시나리오 A: 3뷰 생성이 유리 (가설 지지)

```
MVDiff 3뷰 per-view PSNR > MVDiff 6뷰 per-view PSNR
AND
E2E 3뷰 PSNR ≈ E2E 6뷰 PSNR (또는 더 좋음)
```
→ **결론**: 생성 품질 개선이 정보량 감소를 상쇄
→ **후속**: 3뷰 파이프라인 최적화

### 시나리오 B: 4뷰가 최적 (절충)

```
MVDiff 4뷰 per-view PSNR > MVDiff 6뷰
AND
E2E 4뷰 PSNR > E2E 3뷰 (coverage 부족)
```
→ **결론**: 4뷰가 품질과 coverage의 sweet spot
→ **후속**: LGM/GRM 스타일 4뷰 파이프라인 채택

### 시나리오 C: 6뷰가 여전히 최적 (가설 기각)

```
MVDiff 3-4뷰 per-view PSNR ≈ MVDiff 6뷰 (큰 차이 없음)
AND
E2E 6뷰 PSNR > E2E 3-4뷰 (coverage 중요)
```
→ **결론**: MV-Diffusion 품질 문제는 뷰 수가 아닌 다른 원인
→ **후속**: H5 (더 긴 학습, attention 개선) 집중

### 시나리오 D: Pretrained 불일치 문제 (실패)

```
6뷰→3뷰 fine-tune 시 MVDiff 품질 크게 저하
```
→ **결론**: Pretrained weight가 6뷰에 과적합
→ **후속**: 3뷰 from-scratch 학습 또는 progressive 뷰 감소

---

## 6. H4-H5-H8 교차 의존성

```
H4 (View Ablation) ──────────────────┐
  결과: 최적 GS-LRM 입력 뷰 수        │
                                      ├──→ H8 (Reduced View Gen)
H5 (MV-Diffusion)  ───────────────────┤     결과: 최적 생성 뷰 수
  결과: 현재 6뷰 생성 품질 기준선       │
                                      │
H6 (Alpha Mask) ─────────────────────┘
  결과: Mask 유무에 따른 shape 품질
```

**H8은 H4 + H5 결과에 의존하는 상위 가설.**
H4에서 3뷰 성능이 나쁘면 H8은 4뷰로 방향 전환.
H5에서 cyclic이 성공하면 6뷰 품질이 충분할 수 있어 H8 우선순위 하락.

---

## 7. 현황 및 일정

| 상태 | 내용 |
|------|------|
| ✅ 완료 | 문헌 조사 (17개 논문) |
| ✅ 완료 | 아키텍처 실현 가능성 확인 |
| ✅ 완료 | 실험 설계 |
| ⏳ 대기 | **S0: H4 Round 1 결과** (전제 조건) |
| ⏳ 대기 | S1: View selection ablation |
| ⏳ 대기 | S2-S3: MVDiff fine-tune |

**예상 일정:**
```
H4 R1 완료 (현재 ~50%)
    │
    ▼
S0: 결과 분석 (1일)
    │
    ├─ 3view 성능 충분 → S1 진행
    └─ 3view 부족 → 4view로 전환
    │
    ▼
S1: View selection (GPU 4개, ~9h)
    │
    ▼
S2/S3: MVDiff fine-tune (5-10K steps, ~12h)
    │
    ▼
S4: E2E 평가 (1일)
```

---

## 8. 참고 문헌

### 핵심 (직접 관련)

| 논문 | 뷰 수 | 핵심 발견 |
|------|-------|----------|
| **Era3D** (NeurIPS 2024) | 6 | Fewer attention tokens → better quality |
| **MVDream** (ICLR 2024) | 4 | 4뷰 = Janus 해결 임계점 |
| **InstantMesh** (2024) | 6 | Fewer views alleviate inconsistency |
| **CAT3D** (NeurIPS 2024 Oral) | 5-7 | Joint modeling 품질↑, but consistency 한계 |
| **LGM/GRM** (ECCV 2024) | 4 | 4뷰 입력으로 SOTA 3D 재구성 |
| **MEt3R** (CVPR 2025) | - | Consistency와 quality는 독립 축 |

### 보조

| 논문 | 관련성 |
|------|--------|
| MVDiffusion (NeurIPS 2023) | CAA mechanism, scalability 한계 |
| MVDiffusion++ (ECCV 2024) | 32뷰 + View Dropout |
| Zero123++ | 6뷰 tiled 표준 |
| SyncDreamer (ICLR 2024) | 16뷰 한계 사례 |
| Pose Splatter (NeurIPS 2025) | 동물 카메라 수 ablation |

---

*H8 Reduced View Generation | v1.0 | 2026-02-07*


---

## 3-view E2E Results (260211)

### Per-View Metrics (360 test samples)

| View | PSNR_wh (dB) | SSIM | LPIPS | Type |
|------|:---:|:---:|:---:|------|
| view_0 | **35.51** | 0.9938 | 0.0079 | Input (reconstruction) |
| view_1 | 16.50 | 0.9501 | 0.1132 | Novel (MVDiff generated) |
| view_2 | 16.02 | 0.9471 | 0.1212 | Novel (MVDiff generated) |
| **Overall** | **22.67** | 0.9637 | 0.0808 | Avg all views |
| **Novel avg** | **16.26** | 0.9486 | 0.1172 | Avg novel only |

### Cross-Configuration Comparison

| Pipeline | Views | Novel PSNR | Overall PSNR | Notes |
|----------|:-----:|:----------:|:------------:|-------|
| GS-LRM only (GT input) | 6 | - | 24.49 | Upper bound |
| 6-view E2E | 6 | ~24.0 | 21.21 | Standard pipeline |
| 4-view E2E | 4 | - | 19.87 | Reduced views |
| **3-view E2E** | **3** | **16.26** | **22.67*** | *inflated by input ratio |

> *3-view overall(22.67)이 6-view(21.21)보다 높은 이유: input view(35.51 dB)가 전체의 1/3을 차지.
> Novel view만 비교하면 3-view(16.26) << 6-view(~24.0)로 **7.7 dB 열위**.

### Preliminary Conclusion

1. **H8 가설 기각 경향**: 뷰 수 감소가 per-view 품질 개선으로 이어지지 않음
2. **GS-LRM 입력 부족이 지배적**: 3개 뷰로는 3D 재구성에 필요한 coverage 부족
3. **6-view 유지 권장**: 현재 파이프라인에서 6-view가 최적
4. **추가 검증 필요**: 4-view E2E의 novel-only PSNR 확인 후 최종 결론
