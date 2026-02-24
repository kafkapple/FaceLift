# Multi-view Diffusion & 3D Reconstruction: Theory Reference

> **목적**: FaceLift 파이프라인의 불변 이론 기반 — 모델 계보, attention 메커니즘, 포즈 조건화, 대안 모델 비교
> **성격**: 이론/배경 문서 (실험 결과는 → [Hypothesis Roadmap (H8)](../experiments/hypothesis_roadmap.md))
> ← [RESEARCH_HYPOTHESES](../RESEARCH_HYPOTHESES.md) | [INDEX](../INDEX.md)

---

## 1. 용어 정의 및 파이프라인 명칭

### 논문 기반 공식 명칭

| Stage | 논문 표기 | 약칭 (본 프로젝트) | 원본 모델 | ⚠️ 오해 여지 |
|:-----:|----------|-------------------|----------|-------------|
| **1** | "multi-view diffusion model" | **MV-Diffusion** 또는 **Stage 1** | SD2.1-UnCLIP + Era3D RMA | "MVDiffusion"은 Tang et al. 논문과 혼동 |
| **2** | "transformer-based Gaussian reconstructor" | **GS-LRM** 또는 **Stage 2** | GS-LRM (Zhang et al., ECCV 2024) | 없음 |

> FaceLift 논문 원문: *"a **multi-view latent diffusion model** to generate consistent side and back views"*
> → "multi-view diffusion"은 **기법 설명**이지 특정 모델명이 아님
>
> ⚠️ `mvdiffusion/` 폴더 = Era3D 코드 구조 약칭. Tang et al.의 "MVDiffusion" 논문과 **무관**.

### 각 Stage의 원본 → 변형 차별화

**Stage 1 (Multi-view Diffusion)**:
```
SD2.1 → SD2.1-UnCLIP (+ CLIP image conditioning)
  └→ Era3D (+ row-wise multi-view attention, NeurIPS 2024)
       └→ FaceLift (+ 합성 얼굴 데이터 fine-tune + input view reconstruction loss)
            └→ Mouse adaptation (+ M5 6-camera rig fine-tune)
```

| 구분 | 원본 (SD2.1-UnCLIP) | Era3D 추가 | FaceLift 추가 | Mouse 추가 |
|------|:-------------------:|:----------:|:-------------:|:----------:|
| Image conditioning | CLIP image embed | 동일 | 동일 | 동일 |
| Multi-view attention | 없음 | **Row-wise MV attention** | 동일 | 동일 |
| View direction | 없음 | CLIP text ("front view") | 동일 | 수정 ("top-front view") |
| Training data | LAION | Objaverse | **합성 얼굴** | **합성 생쥐** |
| **Input reconstruction** | 없음 | 없음 | **있음** (핵심 기여) | 동일 |
| UNet class | `UNet2DConditionModel` | **`UNetMV2DConditionModel`** | 동일 | 동일 |

> **FaceLift 핵심 기여**: Input view reconstruction loss — 입력 이미지를 정확히 재생성하도록 강제,
> 합성→실사 도메인 갭 완화. 논문: *"we enforce the input view reconstruction during training"*

**Stage 2 (GS-LRM)**:
```
GS-LRM (Zhang et al., ECCV 2024, ICLR 2025)
  └→ FaceLift (two-stage training: Objaverse pretrain → synthetic head fine-tune)
       └→ Mouse adaptation (+ M5 rig fine-tune)
```

| 구분 | 원본 GS-LRM | FaceLift 변형 |
|------|:-----------:|:------------:|
| 아키텍처 | Transformer + per-pixel Gaussian | **변경 없음** |
| 입력 | 2-4 posed images + Plücker rays | 6 posed images + Plücker rays |
| 카메라 표현 | RGB(3) + Plücker(6) = 9ch/pixel | 동일 |
| 학습 전략 | Single-stage | **Two-stage** (Objaverse → domain-specific) |
| 출력 | Per-pixel Gaussians | 동일 |

> GS-LRM 입력 형식: 각 픽셀에 RGB 3채널 + Plücker ray 6채널 = 9채널.
> Patchify → linear projection → transformer blocks → per-pixel Gaussian parameters.
> 0.23초/A100 추론.

---

## 2. Multi-view Attention 메커니즘 이론

### 2.1 핵심 개념: Epipolar Geometry와 Multi-view Consistency

Multi-view diffusion에서 뷰 간 일관성의 핵심은 **epipolar constraint**:
- 뷰 i의 한 점은 뷰 j의 epipolar line 위에 대응점을 가짐
- Fundamental matrix F로 정의: `l_j = F_{ij} · p_i`
- Attention을 이 기하학적 관계에 따라 제한하면 효율성과 정확성 동시 확보

### 2.2 Row-wise Multi-view Attention (RMA) — Era3D/FaceLift 사용

**전제 조건**: Canonical orthogonal camera 배치 (6개 카메라, 동일 elevation, 등간격 azimuth)

**수학적 근거**:
```
카메라가 동일 높이에서 원점을 바라봄
  → 모든 epipole이 수평선(horizon) 위에 위치
  → Fundamental matrix F의 구조상 epipolar line이 항상 수평
  → epipolar line = 이미지 행(row)
  → 별도 F 계산 불필요, 행 단위로 attention 수행
```

> Era3D 논문: *"given the orthogonal camera setup, the epipolar constraint between
> any pair of views can be simplified to a row-wise correspondence"*

**구현**:
```
각 행(row r)에 대해:
  tokens = [view_0의 row_r, view_1의 row_r, ..., view_5의 row_r]
         = [N × S tokens]  (N=6 views, S=row width=64 at latent)
  output = MultiHeadAttention(tokens)  # Standard self-attention
```

**복잡도**: O(S × (NS)²) = O(N²S³)

### 2.3 Correspondence-aware Attention — MVDiffusion (Tang et al.)

**전제 조건**: 없음 (임의 카메라 배치 지원)

**메커니즘**:
```
뷰 i의 pixel p:
  1. F_{ij} 계산 (또는 사전 계산)
  2. 뷰 j에서 epipolar line l = F_{ij} · p
  3. l 위 K점 등간격 sampling → bilinear interpolation으로 feature 추출
  4. 추출한 K개 feature와 cross-attention
  5. 모든 뷰 쌍 (i,j)에 대해 반복
```

**복잡도**: O(N² × S² × K), K ≈ S → 실질 O(N²S³)

### 2.4 효율 비교: 왜 RMA가 12x 빠른가

점근적 복잡도는 동일(O(N²S³))하지만 **상수 계수**에서 12x 차이:

| Factor | MVDiffusion | RMA |
|--------|:-----------:|:---:|
| F matrix 계산 | 필요 (뷰 쌍마다) | **불필요** |
| Epipolar sampling | K점 bilinear interpolation | **불필요** (이미 정렬) |
| Memory access | 불규칙 (epipolar line 방향 따라) | **연속** (행 단위) |
| GPU cache | 비효율 (scatter access) | **효율** (coalesced) |
| 정보 손실 | K점으로 근사 (sampling) | **없음** (행 전체 사용) |
| Custom kernel | 필요 | **불필요** (standard attention) |

> Era3D 논문: *"~12× reduction in computation compared to
> correspondence-aware attention at 512×512 resolution"*
>
> 실측: MVDiffusion-style ~2.4 TFLOPs vs RMA ~0.2 TFLOPs (6 views, 512×512)

**핵심**: RMA는 canonical 배치에서 sampling 근사 없이 행 전체를 보므로,
**더 빠르면서 정보 손실도 없음**. 단, canonical 배치를 **전제**로 함.

### 2.5 비교 종합표

| 모델 | 논문 | 메커니즘 | 카메라 | 복잡도 | FaceLift 참조 |
|------|------|----------|--------|--------|:------------:|
| **MVDiffusion** | Tang, NeurIPS'23 | Correspondence-aware | 임의 | O(N²S²K) | ❌ |
| **FaceLift (Era3D RMA)** | Li, NeurIPS'24 | **Row-wise (RMA)** | Canonical | O(N²S³) | ✅ |
| Zero123++ | Shi, arXiv'23 | Global self-attention | 고정 6뷰 | O(N²S⁴) | ❌ |
| SyncDreamer | Liu, ICLR'24 | Volume attention (3D) | 고정 16뷰 | 높음 | ❌ |
| Wonder3D++ | Long, TPAMI'25 | Cross-domain MV attention | 카메라+도메인 스위처 | 중간 | ❌ |
| SV3D | Voleti, ECCV'24 | Temporal (video) | Orbital | O(T²S⁴) | ❌ |
| Free3D | Zheng, CVPR'24 | RCN + pseudo-3D cross-attn | 임의 | 낮음 | ❌ |
| MV-Adapter | Huang, ICCV'25 | Decoupled parallel attention | 카메라 guider | 중간 | ❌ |
| Pippo | Kant, CVPR'25 | DiT + attention biasing | Plücker | 중간 | ❌ |
| 3DEnhancer | Luo, CVPR'25 | RMA + epipolar aggregation | Plücker | 중간 | ❌ |

---

## 3. 대안 모델 상세 비교 (교체 가능성)

### 3.1 FaceLift Stage 1 교체 후보

| 모델 | Base | Views | Resolution | 카메라 조건 | Open Weights | 교체 적합도 |
|------|------|:-----:|:----------:|:---------:|:------------:|:----------:|
| **FaceLift (현재)** | SD2.1-UnCLIP | 6 | 512 | CLIP text | ✅ HF | baseline |
| **MV-Adapter** | SD2.1/SDXL | 6 | 512/768 | Camera guider | ✅ HF | ⭐⭐⭐ |
| **Wonder3D++** | SD | 6 | 256 | Camera+domain | ✅ GitHub | ⭐⭐ |
| Zero123++ | SD1.5 | 6 | 320 | 고정 pose | ✅ HF | ⭐ |
| **Free3D** | SD | N | 256 | RCN (per-pixel) | ✅ GitHub | ⭐⭐ |
| SV3D | SVD | Orbital | 576×576 | Orbital angles | ✅ HF | ⭐ |
| Pippo | DiT | Dense | 256→512 | Plücker | ❌ (code only) | ⭐ |
| 3DEnhancer | SD | N | 512 | Plücker + RMA | ✅ GitHub | ⭐⭐⭐ |

### 3.2 교체 추천

**1순위: MV-Adapter** (ICCV 2025)
- **Why**: SD2.1 512 해상도 + 6뷰 = FaceLift와 동일 설정
- **장점**: Plug-and-play adapter → 기존 UNet weight 보존, camera guider로 임의 카메라 지원
- **주의**: Decoupled attention ≠ RMA → 코드 구조 대폭 변경 필요

**2순위: 3DEnhancer** (CVPR 2025)
- **Why**: RMA + near-view epipolar aggregation + Plücker = RMA 확장
- **장점**: RMA 기반이므로 코드 호환성 높음, Plücker로 카메라 일반화
- **단점**: Enhancement 용도 (coarse → fine) → generation 용도로 전환 필요

**현실적 권장**: 현재 Era3D RMA 유지 + pose conditioning 추가가 가장 실용적.
모듈 교체는 연구 목적으로만 권장.

---

## 4. Camera/View Conditioning 이론

### 4.1 현재 FaceLift의 2-channel 조건화

```
Channel 1: CLIP Image Embedding (전역 — "무엇을" 생성)
  input_image → CLIP Vision → image_embeds [2048]
  → class_embedding projection (2048 → 1280)
  → timestep_embedding에 additive

Channel 2: CLIP Text Embedding (뷰별 — "어느 방향에서" 생성)
  고정 프롬프트 ("top-front view" 등) → CLIP Text → [6, 77, 1024]
  → UNet cross-attention의 key/value
```

### 4.2 Pose Conditioning 3가지 방식 (구현됨, 미활성)

`mouse_extensions/model/pose_conditioning.py`에 3개 인코더 구현:

#### (A) SphericalPoseEncoder — M5 rig 최적
```
c2w [4,4] → (azimuth, elevation, distance) 추출
  → Fourier encoding (64 freq, max_freq=10)
  → MLP → [B, N, 1024] (= text embed와 동일 shape)
```
- **이론**: NeRF의 positional encoding 원리. 구면 좌표 3개 scalar → 고차원 매핑
- **적합**: 원점을 바라보는 구형 배치 (M5 rig)
- **한계**: 원점 미주시 카메라, look-at 방향 변화에 취약

#### (B) ExtrinsicPoseEncoder — 임의 카메라
```
c2w [4,4] → rotation_6d [6] + translation [3] = 9D
  → MLP → [B, N, 1024]
```
- **이론**: Zhou et al. (2019) "On the Continuity of Rotation Representations"
  - 3×3 rotation matrix의 첫 2열(6D) = 연속적 표현 (Euler/quaternion보다 학습 안정)
- **적합**: 모든 SE(3) 카메라
- **한계**: 9D → 1024D MLP는 과학습 위험, 더 많은 학습 데이터 필요

#### (C) PluckerRayEncoder — 최고 표현력 (Dense)
```
c2w + intrinsics → 각 픽셀별 ray (direction, moment) = 6D/pixel
  → [B, 6, H, W] → Conv2d layers → [B, C, H/8, W/8]
  → UNet latent에 concat 또는 별도 cross-attention
```
- **이론**: GS-LRM Stage 2에서 이미 사용. 각 픽셀이 3D 공간의 어떤 ray를 보는지 명시
- **적합**: 픽셀 수준 카메라 정보 필요 시 (3DEnhancer, Pippo 등 최신 모델)
- **한계**: UNet input channel 변경 필요 → pretrained weight 비호환

### 4.3 통합 경로 비교

| 경로 | 코드 변경 | Checkpoint 호환 | 일반화 수준 | 리스크 |
|------|:--------:|:--------------:|:----------:|:------:|
| A: Spherical → cross-attn | ~100줄 | 부분 호환 | M5 rig | 낮음 |
| B: Extrinsic → cross-attn | ~100줄 | 부분 호환 | 임의 SE(3) | 중간 |
| C: Plücker → UNet concat | ~150줄 | 비호환 | 픽셀 수준 | 높음 |
| D: Text embed 유지 + random ref | **1줄** | **완전 호환** | M5 내 | **없음** |

**통합 방법 (A/B 공통)**:
```python
# pose_embeds [B, N, 1024] 를 text embeds 대신 cross-attention에 전달:
unet_out = unet(latent, t,
    encoder_hidden_states=pose_embeds,  # was: prompt_embeds [B, 77, 1024]
    class_labels=image_embeds,
)
```

**주의**: text embed [B, 77, 1024]와 pose embed [B, N, 1024]는 seq_len이 다름(77 vs N).
이를 해결하는 방법:
1. Pose embed를 77 length로 padding/projection
2. 또는 concat: [B, 77+N, 1024] (text + pose)
3. 또는 별도 cross-attention layer 추가

### 4.4 구현 우선순위 (근거 포함)

| 순위 | 방식 | 근거 |
|:---:|------|------|
| **1** | Random ref (`reference_view_idx: "random"`) | Config 1줄, 코드 이미 존재, 즉시 실험 가능 |
| **2** | Spherical pose → cross-attn 교체 | M5 rig에 최적, 출력 dim 호환, ~100줄 |
| **3** | Extrinsic pose | 임의 카메라 일반화 필요 시, Spherical과 동일 패턴 |
| **4** | Plücker Ray concat | UNet 구조 변경 필요, 최신 연구에서 유망하나 리스크 높음 |

> **핵심 판단 근거**: 구현은 **FaceLift/Mouse checkpoint에서 시작** (NOT Era3D).
> Why: Mouse M5t2 ckpt가 이미 도메인 적응됨 → pose encoder만 re-init하고 나머지 freeze 가능.
> Era3D에서 시작하면 mouse 도메인 재학습 비용 발생.

---

## 5. GS-LRM Stage 2 이론

### 5.1 아키텍처

```
Input: N posed images → per-pixel [RGB(3) + Plücker(6)] = 9 channels
  → Patchify (p×p patches) → flatten → linear projection → d-dim tokens
  → Concatenate all views' tokens
  → Transformer blocks (self-attention across all tokens)
  → Decode: per-pixel Gaussian parameters (position, scale, rotation, opacity, SH)
  → Differentiable rendering → loss vs GT
```

### 5.2 Plücker Ray 표현

```
카메라 c2w [4,4] + intrinsics [3,3]:
  각 픽셀 (u,v)에 대해:
    direction d = normalize(R @ K_inv @ [u, v, 1])
    origin o = t (카메라 위치)
    moment m = o × d
    plücker = (d, m) ∈ R⁶
```

> GS-LRM이 **Stage 2에서 이미 Plücker ray를 사용**하므로,
> Stage 1에서 PluckerRayEncoder를 도입하면 파이프라인 전체가 Plücker 기반으로 통일됨.

### 5.3 FaceLift의 GS-LRM 변형

| 원본 | FaceLift 변형 |
|------|:------------:|
| 2-4 views | 6 views (M5 rig) |
| Objaverse single-stage | **Two-stage** (Objaverse pretrain → head/mouse fine-tune) |
| General objects/scenes | Domain-specific (얼굴/생쥐) |
| 아키텍처 변경 | **없음** |

---

## References

- [FaceLift (Lyu et al., ICCV 2025)](https://arxiv.org/abs/2412.17812)
- [Era3D (Li et al., NeurIPS 2024)](https://arxiv.org/abs/2405.11616)
- [MVDiffusion (Tang et al., NeurIPS 2023)](https://arxiv.org/abs/2307.01097)
- [GS-LRM (Zhang et al., ECCV 2024 / ICLR 2025)](https://arxiv.org/abs/2404.19702)
- [MV-Adapter (Huang et al., ICCV 2025)](https://arxiv.org/abs/2412.03632)
- [Wonder3D++ (Long et al., TPAMI 2025)](https://arxiv.org/abs/2511.01767)
- [Zero123++ (Shi et al., 2023)](https://arxiv.org/abs/2310.15110)
- [SV3D (Voleti et al., ECCV 2024)](https://arxiv.org/abs/2403.12008)
- [Free3D (Zheng & Vedaldi, CVPR 2024)](https://arxiv.org/abs/2312.04551)
- [3DEnhancer (Luo et al., CVPR 2025)](https://arxiv.org/abs/2412.18565)
- [Pippo (Kant et al., CVPR 2025)](https://arxiv.org/abs/2502.07785)
- [6D Rotation Representation (Zhou et al., CVPR 2019)](https://arxiv.org/abs/1812.07035)

---

*Multi-view Diffusion Theory Reference v1.0 | Created: 2026-02-13*
