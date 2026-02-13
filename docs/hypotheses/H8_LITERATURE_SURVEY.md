# H8 문헌 조사: Multi-View Diffusion 뷰 수와 품질 관계

> **목적**: MV-Diffusion 생성 뷰 수 감소 가설(H8)의 근거가 되는 문헌 종합 조사
> **조사일**: 2026-02-07
> **논문 수**: 17편
>
> ← [H8_REDUCED_VIEW_GENERATION.md](H8_REDUCED_VIEW_GENERATION.md)

---

## 1. Multi-View Diffusion 모델 종합 비교

### 1.1 모델별 뷰 수 · 해상도 · 일관성 메커니즘

| Model | Year | Venue | Views | Resolution | Consistency Mechanism | 비고 |
|-------|------|-------|-------|------------|----------------------|------|
| MVDream | 2023 | ICLR 2024 | **4** | 256² | 3D Self-Attention | 뷰 수 ablation 유일 |
| SyncDreamer | 2023 | ICLR 2024 | 16 | 256² | 3D-Aware Attn + Spatial Vol | 과다 뷰 |
| MVDiffusion | 2023 | NeurIPS 2023 | 8 (panorama) | 512² | Correspondence-Aware Attn (CAA) | FaceLift 기반 |
| Zero123++ | 2023 | - | **6** | 320² (tiled) | Joint 3×2 Tiled | 산업 표준 |
| Wonder3D | 2023 | CVPR 2024 | **6** | 256² | Cross-Domain Attn (RGB+Normal) | |
| Era3D | 2024 | NeurIPS 2024 | **6** | 512² | Row-wise Attn (EFReg) | ⭐ 핵심 증거 |
| Instant3D | 2023 | ICLR 2024 | **4** | 256² (tiled 2×2) | Joint Tiled | |
| One-2-3-45++ | 2023 | CVPR 2024 | **6** | 256² (tiled 3×2) | Joint Tiled | |
| CRM | 2024 | ECCV 2024 | **6** (ortho) | 256² | Triplane Spatial Corr | |
| InstantMesh | 2024 | - | **6** (Zero123++) | 320² | Zero123++ tiling | ⭐ 핵심 증거 |
| SV3D | 2024 | ECCV 2024 | 21 (video) | 576² | Video Temporal | |
| MVDiffusion++ | 2024 | ECCV 2024 | 32 | 512² | Pose-free Self-Attn + View Dropout | |
| CAT3D | 2024 | NeurIPS 2024 Oral | 5-7/batch | 256² | Autoregressive Multi-batch | ⭐ 핵심 증거 |
| LGM | 2024 | ECCV 2024 Oral | **4** (recon input) | 512² | Multi-view Gaussian Fusion | ⭐ |
| GRM | 2024 | ECCV 2024 | **4** (recon input) | 512² | Pixel-aligned Gaussian Trans | ⭐ |
| MEt3R | 2025 | CVPR 2025 | - (metric) | - | Consistency Benchmark | ⭐ 핵심 증거 |
| 3DEnhancer | 2025 | CVPR 2025 | - (enhancement) | - | Multi-view Enhancement | |

**통계**: 6뷰 = 7편, 4뷰 = 5편, 기타 = 5편. **4뷰와 6뷰가 지배적**.

### 1.2 뷰 수 추세

```
2023:  4뷰 (MVDream, Instant3D)  |  6뷰 (Zero123++, Wonder3D)  |  16뷰 (SyncDreamer)
2024:  4뷰 (LGM, GRM)           |  6뷰 (Era3D, CRM)           |  32뷰 (MVDiff++)
                                 ↓
                    4-6뷰로 수렴, 과다 뷰는 특수 목적
```

---

## 2. 핵심 증거: 뷰 수 ↔ 품질 관계

### 2.1 직접 증거: MVDream 뷰 수 Ablation

**유일하게 1/2/4뷰 모델을 동일 아키텍처에서 비교한 논문.**

| Views | Janus Problem | Multi-view Consistency |
|-------|---------------|----------------------|
| 1 | Severe | Poor |
| 2 | Greatly reduced | Moderate |
| **4** | **Barely any** | **Good** |

**결론**: 4뷰 = multi-face 문제 해결의 **최소 충분 조건**.
**한계**: per-view PSNR 정량 비교는 미제공.

### 2.2 핵심 증거: Era3D Attention Token 감소 → 품질 향상

**가장 강력한 간접 증거.** 같은 6뷰에서 attention 방식만 변경.

| Attention | Memory (512²) | Time | PSNR | LPIPS | Chamfer |
|-----------|---------------|------|------|-------|---------|
| Dense (전체 뷰 cross) | 35.32 GB | 220.41 ms | 20.73 | 0.140 | 0.0239 |
| **Row-wise (제한)** | **1.66 GB** | **2.23 ms** | **20.92** | **0.137** | **0.0232** |

논문 원문: *"reducing the number of attention tokens, allowing the model to focus more on valuable tokens"*

**시사점**: Attention token 수 감소 → 모델이 **중요한 정보에 집중** → 품질 향상.
뷰 수 감소도 token 수 감소 효과 → **유사한 품질 향상 기대**.

### 2.3 핵심 증거: InstantMesh — Fewer Views = Less Inconsistency

논문 원문: *"it is practical to use less input views for reconstruction, which can alleviate the multi-view inconsistency issue in some cases"*

**시사점**: Diffusion 생성 이미지의 inconsistency가 downstream 재구성을 해치므로,
**차라리 적지만 일관된 뷰가 나음**.

### 2.4 핵심 증거: CAT3D — Joint Modeling 효과 + 한계

**동시 생성 뷰 수 ablation:**

| Setting | In-Domain PSNR | Out-of-Domain PSNR |
|---------|---------------|-------------------|
| 3 cond + 1 target | 18.85 | 14.12 |
| 3 cond + **5 target** | **21.66** | **14.63** |

*"jointly modeling multiple output views improves sample metrics -- even metrics that evaluate each output image independently"*

**그러나 한계도 명시:**
*"not all views may be 3D consistent with each other"*

80뷰→720뷰 확장 시:
- Object geometry: 개선
- Background textures: **blurrier** (inconsistencies)

**시사점**: 동시 생성 뷰↑ → 개별 품질↑ (감당 가능 범위 내) BUT 일관성↓ (과하면).
**Sweet spot 존재**.

### 2.5 핵심 증거: MEt3R Consistency Benchmark

| Generation 방식 | Consistency Score (↓ better) |
|----------------|------------------------------|
| 3D prior (동시) | 0.026 |
| Cross-view attn (동시) | 0.036 |
| Sequential | 0.069 |
| Independent (per-view) | 0.120 |

**추가 발견**: 최고 consistency(DFM, 0.026)를 가진 모델의 출력이 blurry.
→ **Consistency와 per-view quality는 독립적인 두 축**.

---

## 3. Correspondence Attention의 Scaling 분석

### 3.1 이론적 복잡도

Correspondence-Aware Attention (MVDiffusion):
```
Attention map: R^(F·h·w × F·h·w)
복잡도: O(F² × (hw)²)

F=6: 36 × (hw)²  attention pairs
F=4: 16 × (hw)²  (-56%)
F=3:  9 × (hw)²  (-75%)
```

### 3.2 실증적 메모리 · 속도 비교 (Era3D, 512×512)

| Approach | Memory | Speedup |
|----------|--------|---------|
| Dense attention (6v) | 35.32 GB | 1× |
| Row-wise attention (6v) | 1.66 GB | ~99× |

### 3.3 MV-Diffusion 저자의 확장성 한계 언급

*"The primary limitation lies in its computational time and resource requirements... the memory-intensive nature, resulting from the parallel denoising, limits its scalability... challenges for applications that require a large number of images."*

### 3.4 확장성 해결 전략 비교

| Strategy | Model | Complexity | Trade-off |
|----------|-------|-----------|-----------|
| **뷰 수 감소** | (제안) | O(F²) 직접 감소 | Coverage 감소 |
| Tiled generation | Zero123++, One-2-3-45++ | O(1) per view | 뷰 수 고정 |
| Row-wise attention | Era3D | O(F·h·w²) | Epipolar prior 필요 |
| View dropout | MVDiffusion++ | 학습시만 | 추론시 무효 |
| Autoregressive | CAT3D | O(F) sequential | 축적 오류 |
| 3D representation | SyncDreamer | Spatial volume | 3D prior 의존 |

---

## 4. Downstream 3D Reconstruction 최적 뷰 수

### 4.1 Feed-forward 재구성 모델의 설계 선택

| Model | Input Views | Quality (PSNR) |
|-------|-------------|----------------|
| LGM | **4** | 23.79 |
| GRM | **4** | 30.05 |
| InstantMesh | 6 | Best SSIM/LPIPS |
| GS-LRM | 2-4 | 0.23s inference |
| CRM | 6 | 10s total |

**관찰**: LGM, GRM 등 최신 SOTA는 **4뷰 입력** 설계. 6뷰 불필요 주장.

### 4.2 Pose Splatter 동물 카메라 수 Ablation (⚠️ 반론)

| Cameras | IoU (Mouse) | PSNR |
|---------|-------------|------|
| 6 | **0.868** | **33.5** |
| 5 | 0.760 | 29.0 |
| 4 | 0.721 | 28.2 |

**주의**: GT 이미지 기준. MV-Diffusion 생성 이미지 기준이 아님.
"완벽한 6뷰" vs "불완전한 3뷰"가 아닌, **"불완전한 6뷰" vs "덜 불완전한 3뷰"** 비교가 핵심.

### 4.3 SyncDreamer 16뷰의 한계

*"16 views are relatively very sparse for COLMAP so it sometimes fails to reconstruct."*
→ 뷰 수 증가 ≠ 품질 보장. Architecture와 일관성이 더 중요.

---

## 5. 종합 분석

### 5.1 Competing Perspectives

| 관점 | 주장 | 근거 |
|------|------|------|
| **뷰 감소 유리** | Attention 집중 → per-view 품질↑ | Era3D, InstantMesh |
| **뷰 증가 유리** | 더 많은 관측 → 3D 정확도↑ | CAT3D, Pose Splatter |
| **Sweet spot 존재 (4-6)** | 일관성 유지 가능한 최대 뷰 수 | MVDream(4), 다수 모델(6) |
| **Architecture > Count** | 뷰 수보다 attention 설계가 핵심 | Era3D row-wise |

### 5.2 FaceLift 프로젝트 시사점

1. 현재 MV-Diffusion 6뷰 생성은 합리적이나, **병목이 확인된 상황**
2. **4뷰 생성**이 가장 안전한 첫 시도 (LGM/GRM 검증, MVDream 임계점)
3. **3뷰 생성**은 H4에서 3view 성능이 충분할 때만 시도
4. **병행 전략**: Attention 효율화 (Era3D 스타일 row-wise) + 뷰 수 감소 조합

### 5.3 연구 공백 (Research Gap)

| Gap | 설명 |
|-----|------|
| 동일 아키텍처 3/4/6/8뷰 비교 | 존재하지 않음 → **H8이 기여 가능** |
| Attention token vs per-view 품질 정량화 | 간접 증거만 존재 |
| 생성 이미지 기준 downstream 뷰 수 비교 | GT vs 생성 이미지 차이 미검증 |

---

## 6. 참고 문헌 (신뢰도 평가 포함)

| # | 논문 | Venue | 핵심 기여 | 신뢰도 |
|---|------|-------|----------|--------|
| 1 | Era3D | NeurIPS 2024 | Row-wise > Dense attention | ⭐⭐⭐⭐⭐ |
| 2 | CAT3D | NeurIPS 2024 Oral | 뷰 수 ablation + scaling | ⭐⭐⭐⭐⭐ |
| 3 | MEt3R | CVPR 2025 | Consistency benchmark | ⭐⭐⭐⭐½ |
| 4 | MVDream | ICLR 2024 | 1/2/4뷰 Janus ablation | ⭐⭐⭐⭐½ |
| 5 | Pose Splatter | NeurIPS 2025 | 동물 카메라 수 ablation | ⭐⭐⭐⭐½ |
| 6 | LGM | ECCV 2024 Oral | 4뷰 SOTA 재구성 | ⭐⭐⭐⭐ |
| 7 | GRM | ECCV 2024 | 4뷰 재구성 | ⭐⭐⭐⭐ |
| 8 | MVDiffusion | NeurIPS 2023 | CAA mechanism, scalability | ⭐⭐⭐⭐ |
| 9 | Zero123++ | 2023 | 6뷰 tiled 표준 | ⭐⭐⭐⭐ |
| 10 | InstantMesh | 2024 | Fewer views = less inconsistency | ⭐⭐⭐⭐ |
| 11 | SyncDreamer | ICLR 2024 | 16뷰 한계 사례 | ⭐⭐⭐⭐ |
| 12 | MVDiffusion++ | ECCV 2024 | 32뷰 + View Dropout | ⭐⭐⭐⭐ |
| 13 | GS-LRM | ECCV 2024 | 2-4뷰 입력 재구성 | ⭐⭐⭐⭐ |
| 14 | One-2-3-45++ | CVPR 2024 | 6뷰 tiled | ⭐⭐⭐⭐ |
| 15 | Instant3D | ICLR 2024 | 4뷰 tiled | ⭐⭐⭐½ |
| 16 | Wonder3D | CVPR 2024 | 6뷰 + normal | ⭐⭐⭐½ |
| 17 | Correspondence-Attn Alignment | 2024 | F²hw 복잡도 분석 | ⭐⭐⭐½ |

---

*H8 Literature Survey | v1.0 | 2026-02-07*
