# Stage 1 Multi-View Diffusion Replacement Candidates

## 1. Overview

### 현재 병목 (Current Bottleneck)

FaceLift는 2-stage pipeline으로 구성됨:

```
Stage 1: Single Image → Multi-View Diffusion → 6 Views (256×256, upscaled 512×512)
         (SD2.1-UnCLIP + Era3D Row-wise Multi-view Attention)

Stage 3: 6 Multi-View Images + Camera Poses → GS-LRM → 16,386 Gaussians → 3D
```

**핵심 문제**: Stage 1의 개선이 E2E 성능으로 전이되지 않음.

| Metric | Stage 1 Only | E2E | Transfer Rate |
|--------|-------------|-----|---------------|
| Val PSNR (best) | 27.70 | ~3.9 | ~14% |
| GT Input Upper Bound | - | 24.49 (6-view) | - |

Stage 1 단독 PSNR이 27.70까지 도달해도, E2E에서는 ~3.9 수준에 머문다. GT 이미지를 직접 입력하면 GS-LRM이 24.49를 달성하므로, **Stage 1이 생성하는 multi-view 이미지의 질이 GS-LRM에 전달되는 과정에서 심각한 정보 손실이 발생**한다.

### Why — 왜 대안이 필요한가

1. **Transfer Gap**: Stage 1 PSNR 개선 → E2E 개선 전이율이 극히 낮음 (~14%)
2. **Coverage 문제**: 생성된 view의 silhouette가 GT와 ~50% IoU → 형상 자체가 부정확
3. **Resolution 제약**: 256×256 생성 → 512×512 upscale은 디테일 손실 불가피
4. **Camera Rigidity**: Era3D의 고정 6-view 카메라 배치가 M5 mouse 데이터셋의 실제 카메라 배치와 불일치할 가능성

### How — 어떤 관점에서 대안을 평가하는가

대안 탐색은 세 가지 전략으로 구분됨:

- **Tier 1 (Drop-in)**: 동일 아키텍처 계열, 최소 코드 변경으로 교체 가능
- **Tier 2 (E2E)**: 2-stage 구조 자체를 우회, single-stage로 3D 직접 생성
- **Tier 3 (Next-Gen)**: 최신 SOTA 모델, 최고 잠재력이나 통합 난이도 최상

---

## 2. Selection Criteria (평가 기준)

### 필수 조건 (Hard Requirements)

| Criterion | Description | 이유 |
|-----------|-------------|------|
| **GS-LRM Compatibility** | N개 multi-view image + known camera poses 출력 | Stage 3 재활용을 위한 최소 조건 |
| **Camera Flexibility** | 임의 카메라 포즈 지원 또는 M5 카메라 배치 매칭 | 고정 orbital 카메라는 M5와 불일치 가능 |
| **Open-Source** | Weight + Code 공개 | 재현성 및 fine-tuning 가능성 |
| **Single Image Input** | 단일 이미지에서 3D 복원 | FaceLift의 핵심 use-case 유지 |

### 선호 조건 (Soft Requirements)

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Resolution** | High | 512×512+ native generation이 이상적 |
| **View Count** | Medium | 6+ views, 많을수록 GS-LRM 입력 풍부 |
| **Fine-tuning Cost** | High | A6000 48GB × 4 환경에서 학습 가능 여부 |
| **Multi-view Consistency** | Critical | View 간 3D 일관성 (E2E transfer의 핵심) |
| **Inference Speed** | Low | 연구 단계이므로 속도보다 품질 우선 |

### Mouse Reconstruction 특수 요구사항

FaceLift의 target domain은 실험용 mouse (M5 데이터셋)이며, 일반적인 object reconstruction과 다른 특성이 있음:

- **Textureless Surface**: 거의 단색 (회백색), texture cue 부족 → shape 정확도가 핵심
- **Deformable Body**: 비강체 변형 (숨쉬기, 자세 변화) → temporal consistency 필요
- **Small Silhouette**: 이미지 내 mouse가 차지하는 비율이 작음 (~2.5%) → silhouette extraction 민감
- **Known Camera Setup**: M5는 고정 카메라 6대 → 카메라 포즈가 정확히 알려져 있음

---

## 3. Tier 1: Drop-in Replacements

동일 아키텍처 계열로, 기존 코드 구조를 최대한 유지하면서 교체 가능한 후보군.

### 3.1 Era3D (Original Updated Weights)

| Item | Detail |
|------|--------|
| **Paper** | Era3D: High-Resolution Multiview Diffusion (ICCV 2025) |
| **Architecture** | SD2.1-UnCLIP + Row-wise Multi-view Attention (RMA) |
| **Output** | 6 views, 256×256 (up to 512×512) |
| **Input** | Single image |
| **Code** | 현재 FaceLift `mvdiffusion/` 폴더가 Era3D RMA 아키텍처 기반 |
| **Difficulty** | ★☆☆☆☆ |

**현황**: FaceLift의 Stage 1이 SD2.1-UnCLIP + Era3D RMA 아키텍처를 사용 중. 다만 Era3D 공식 repo의 최신 weight나 개선된 학습 기법이 반영되지 않았을 가능성이 있음.

**개선 가능성**:
- Era3D 공식 최신 checkpoint 적용
- RMA attention 레이어의 hyper-parameter 조정
- Focal diffusion 기법 (Era3D 논문의 고해상도 기법) 활성화 여부 확인

**한계**:
- 근본적으로 동일 아키텍처이므로 획기적 개선 기대 어려움
- Transfer gap 문제는 아키텍처 자체의 한계일 가능성

**Mouse Domain 적합성**: ★★★☆☆
- 이미 mouse 데이터로 fine-tuning된 상태, 추가 개선 여지 제한적

**권장 조치**: 최신 Era3D 공식 weight와 학습 설정 비교 후, 유의미한 차이가 있으면 시도. 그러나 이것만으로는 근본적 해결 불가.

---

### 3.2 Zero123++

| Item | Detail |
|------|--------|
| **Paper** | Zero123++: A Single Image to Consistent Multi-view Diffusion Base Model |
| **Architecture** | SD2.1 fine-tuned, tiled multi-view output (3×2 grid) |
| **Output** | 6 views, 320×320 per view (tiled in 960×640 image) |
| **Input** | Single image (foreground-segmented) |
| **Code** | [github.com/SUDO-AI-3D/zero123plus](https://github.com/SUDO-AI-3D/zero123plus) |
| **License** | Apache 2.0 |
| **Difficulty** | ★★☆☆☆ |

**아키텍처 분석**:
- SD2.1을 backbone으로 사용 (FaceLift와 동일 base)
- 6개 view를 single 이미지의 tile로 동시 생성 → view 간 일관성 확보
- Reference attention으로 input condition 주입
- 고정 elevation/azimuth 카메라 배치 (30°, 90° elevation × 6 azimuth)

**GS-LRM 호환성 분석**:
```
Zero123++ Output: 960×640 tiled image (3×2 grid, 각 320×320)
                  ↓ tile 분리 + resize
GS-LRM Input:    6× 512×512 images + camera extrinsics/intrinsics
```
- Tile 분리 adapter 필요 (비교적 단순)
- 카메라 포즈: Zero123++의 고정 6-pose → GS-LRM 카메라 매트릭스로 변환 필요
- M5 카메라 배치와의 불일치는 fine-tuning으로 해결 가능

**장점**:
- 320×320 native resolution (현재 256×256 대비 +56%)
- View 간 일관성이 tile 기반 생성으로 구조적으로 보장됨
- 활발한 커뮤니티, InstantMesh 등 후속 프로젝트에서 검증됨
- SD2.1 base → 기존 fine-tuning 경험/코드 일부 재활용 가능

**단점**:
- Tiled output format → 분리 로직 필요
- 고정 카메라 배치 → M5 카메라 배치로 fine-tuning 필요
- Era3D 대비 attention 구조 상이 → 기존 코드 adapter 재작성

**Mouse Domain 적합성**: ★★★☆☆
- Textureless object에서 tile boundary artifact 발생 가능성
- 해상도 향상은 mouse의 작은 silhouette 문제에 도움

**예상 소요**: 2-3주 (adapter 구현 + fine-tuning)

---

### 3.3 SV3D (Stable Video Diffusion for 3D)

| Item | Detail |
|------|--------|
| **Paper** | SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Stable Video Diffusion |
| **Architecture** | SVD (Stable Video Diffusion) fine-tuned for orbital 3D views |
| **Output** | 21 orbital views (continuous trajectory) |
| **Input** | Single image |
| **Code** | [github.com/Stability-AI/generative-models](https://github.com/Stability-AI/generative-models) |
| **Variants** | SV3D_u (unconditional orbit), SV3D_p (pose-conditioned) |
| **Difficulty** | ★★★☆☆ |

**아키텍처 분석**:
- SVD (video diffusion)를 3D orbital motion으로 fine-tuning
- Video의 temporal consistency가 3D multi-view consistency로 자연스럽게 전이
- SV3D_p 변형: 카메라 궤적을 조건으로 입력 → **카메라 포즈 유연성 확보**

**GS-LRM 호환성 분석**:
```
SV3D Output: 21 frames (576×576), orbital trajectory
             ↓ 6 frames 선택 + pose extraction
GS-LRM Input: 6× 512×512 images + camera extrinsics/intrinsics
```
- 21개 view에서 6개를 sampling → GS-LRM 입력에 유리 (더 많은 정보)
- SV3D_p의 pose conditioning → M5 카메라 배치에 맞춤 가능
- 단, orbital trajectory는 M5의 임의 카메라 배치와 다를 수 있음

**장점**:
- 576×576 native resolution (현재 대비 2.25× 면적)
- 21개 view → GS-LRM에 더 풍부한 multi-view 정보 제공
- Video diffusion의 temporal consistency → 3D consistency 양호
- SV3D_p로 카메라 포즈 지정 가능

**단점**:
- SVD backbone → SD2.1 대비 메모리/연산 비용 증가
- Orbital trajectory 가정 → 비대칭 카메라 배치에서의 성능 불확실
- Fine-tuning 시 SVD weight를 모두 로드해야 함 (VRAM 부담)
- A6000 48GB에서 fine-tuning 가능 여부 확인 필요

**Mouse Domain 적합성**: ★★☆☆☆
- Orbital trajectory는 mouse의 위/아래 비대칭 구조에 부적합할 수 있음
- Video diffusion의 temporal smoothing이 mouse의 미세한 deformation을 평활화할 위험

**예상 소요**: 3-4주 (SVD adapter + view selection + fine-tuning)

---

### 3.4 MV-Adapter (SDXL Multi-View Adapter)

| Item | Detail |
|------|--------|
| **Paper** | MV-Adapter: Multi-View Consistent Image Generation Made Easy |
| **Architecture** | SDXL + separate multi-view adapter (207M params) |
| **Output** | N views, 768×768 (native) |
| **Input** | Single image (or text prompt) |
| **Code** | 공개 (HuggingFace) |
| **기존 조사** | `docs/theory/MV_ADAPTER_TECHNICAL.md` 상세 분석 완료 |
| **기존 코드** | `mouse_extensions/model/mv_adapter.py` scaffolding 존재 |
| **Difficulty** | ★★★☆☆ |

**아키텍처 분석**:
- SDXL을 frozen backbone으로 사용, multi-view adapter만 학습
- Adapter는 decoupled multi-view attention 포함 (207M trainable params)
- 임의 개수, 임의 포즈의 view 생성 가능 → **카메라 유연성 최고**

**GS-LRM 호환성 분석**:
```
MV-Adapter Output: N× 768×768 images + arbitrary camera poses
                   ↓ resize (optional)
GS-LRM Input:     6× 512×512 images + camera extrinsics/intrinsics
```
- 출력 format이 GS-LRM과 거의 완벽 호환
- 카메라 포즈를 직접 지정 → M5 카메라 배치 그대로 사용 가능
- 해상도 downscale (768→512) 또는 GS-LRM 입력 해상도 업그레이드 가능

**장점**:
- 768×768 native resolution (현재 대비 9× 면적)
- 카메라 포즈 완전 유연 → M5 카메라 배치와 정확히 매칭 가능
- Adapter-only 학습 (207M) → full model 대비 효율적
- 이미 scaffolding 코드 존재 (`mouse_extensions/model/mv_adapter.py`)
- SDXL의 강력한 image prior 활용

**단점**:
- SDXL backbone이 SD2.1 대비 상당히 무거움 (VRAM ~20GB+ for inference)
- A6000 48GB에서 fine-tuning 시 batch size 제약
- SDXL → GS-LRM 파이프라인 연결 시 latent space 불일치 가능
- 기존 `mvdiffusion/` 코드와 호환 불가 → 상당한 코드 재작성

**Mouse Domain 적합성**: ★★★★☆
- 높은 해상도는 mouse의 미세한 형태 디테일 (귀, 발, 꼬리) 포착에 유리
- 카메라 유연성은 M5 setup과의 정확한 매칭에 결정적 장점
- Textureless surface에서 SDXL의 강력한 prior가 hallucination 유발 가능성 → fine-tuning으로 억제 필요

**예상 소요**: 3-4주 (adapter integration + M5 camera fine-tuning)

**기존 자산 활용도**: 높음
- `docs/theory/MV_ADAPTER_TECHNICAL.md` — 기술 분석 문서 존재
- `mouse_extensions/model/mv_adapter.py` — scaffolding 코드 존재
- SDXL adapter 패턴은 FaceLift의 multi-view attention 교체에 적합

---

## 4. Tier 2: E2E Alternatives

2-stage 구조 자체를 우회하여 single image → 3D를 직접 수행하는 모델. Transfer gap 문제를 근본적으로 해결할 수 있지만, GS-LRM을 포기해야 하므로 파이프라인 전면 재설계 필요.

### 4.1 LGM (Large Gaussian Model)

| Item | Detail |
|------|--------|
| **Paper** | LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation |
| **Architecture** | Multi-view images → U-Net backbone → 3D Gaussian prediction |
| **Output** | 3D Gaussians (직접, intermediate multi-view 없음) |
| **Input** | Single image (+ optional multi-view) |
| **Code** | [github.com/3DTopia/LGM](https://github.com/3DTopia/LGM) |
| **License** | MIT |
| **Difficulty** | ★★★★☆ |

**아키텍처 분석**:
- 내부적으로 multi-view 생성 (4 views) → 각 view에서 pixel-aligned Gaussian 예측
- Asymmetric U-Net으로 multi-view feature 융합
- 고속 inference (~5초 per object)

**GS-LRM 관계**:
- GS-LRM을 **대체** (bypass), 기존 Stage 3 불필요
- 출력이 직접 3D Gaussians → 별도 reconstruction 단계 없음
- 단, LGM의 Gaussian representation이 FaceLift GS-LRM과 다를 수 있음

**장점**:
- Transfer gap 문제 근본 해결 (중간 2D representation 없음)
- 매우 빠른 inference
- MIT license, 활발한 개발

**단점**:
- 복잡한 geometry에서 품질 제한 (coarse Gaussians)
- FaceLift의 기존 Stage 3 (GS-LRM) 코드 전부 폐기
- 내부 4-view 생성이 M5 카메라 배치와 불일치
- Mouse 같은 textureless object에서의 성능 미검증

**Mouse Domain 적합성**: ★★☆☆☆
- Pixel-aligned Gaussian은 textureless surface에서 ambiguity 높음
- 4 views는 mouse의 복잡한 형태 (특히 아래쪽)를 커버하기 불충분

**예상 소요**: 5-6주 (전체 파이프라인 재설계)

---

### 4.2 GRM (Gaussian Reconstruction Model)

| Item | Detail |
|------|--------|
| **Paper** | GRM: Large Gaussian Reconstruction Model for Efficient 3D Reconstruction and Generation |
| **Architecture** | Transformer-based, multi-view → feed-forward 3D Gaussians |
| **Output** | High-density 3D Gaussians |
| **Input** | Multi-view images (typically 4-6 views) |
| **Code** | [github.com/jclarkk/GRM](https://github.com/jclarkk/GRM) |
| **Difficulty** | ★★★★☆ |

**아키텍처 분석**:
- Pixel-aligned transformer로 multi-view features를 3D로 lifting
- LGM 대비 더 큰 모델, 더 높은 품질
- Feed-forward (no optimization per scene) → fast inference

**GS-LRM과의 비교**:
- GRM은 사실상 GS-LRM의 경쟁 모델
- 둘 다 multi-view → 3D Gaussians feed-forward 구조
- GRM이 transformer 기반으로 더 풍부한 feature interaction

**장점**:
- GS-LRM 대비 더 나은 3D 품질 가능성
- Stage 1은 유지하고 Stage 3만 교체하는 전략도 가능

**단점**:
- Stage 3 교체는 본 문서의 주 관심사 (Stage 1 교체)와 다른 방향
- 독립적으로 사용 시 여전히 multi-view input 필요
- 학습 코드/데이터 파이프라인 구축 필요

**Mouse Domain 적합성**: ★★★☆☆
- Transformer 기반 feature interaction이 textureless surface에서 유리할 수 있음
- 단, mouse 전용 fine-tuning 필요

**예상 소요**: 4-5주 (Stage 3 교체) / 6-8주 (E2E with multi-view gen)

---

### 4.3 InstantMesh

| Item | Detail |
|------|--------|
| **Paper** | InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models |
| **Architecture** | Zero123++ (multi-view) → FlexiCubes (mesh extraction) |
| **Output** | Textured mesh (NOT Gaussians) |
| **Input** | Single image |
| **Code** | [github.com/TencentARC/InstantMesh](https://github.com/TencentARC/InstantMesh) |
| **Difficulty** | ★★★★☆ |

**아키텍처 분석**:
- Zero123++로 multi-view 생성 → sparse-view LRM으로 triplane 생성 → FlexiCubes로 mesh 추출
- 본질적으로 2-stage이지만, 두 stage가 더 잘 통합되어 있음

**GS-LRM 호환성**:
- 출력이 mesh (Gaussians 아님) → GS-LRM 파이프라인과 직접 호환 불가
- 중간 multi-view output (Zero123++ 부분)만 추출하여 GS-LRM에 입력하는 hybrid 접근 가능

**장점**:
- E2E 파이프라인이 이미 통합됨
- Zero123++의 multi-view 품질이 검증됨
- Mesh 출력은 downstream application에서 범용적

**단점**:
- Gaussian 출력이 아님 → FaceLift 렌더링 파이프라인과 불일치
- FlexiCubes mesh는 Gaussian splatting 대비 렌더링 품질 열세
- 실질적으로 Zero123++을 쓰는 것과 유사 (Tier 1의 3.2와 중복)

**Mouse Domain 적합성**: ★★☆☆☆
- Mesh 기반 → textureless surface에서 vertex color 제한
- FlexiCubes의 topology 유연성은 mouse의 deformation에 유리

**예상 소요**: 4-5주 (hybrid 접근) / 7-8주 (full 교체)

---

## 5. Tier 3: Next-Gen Models

최신 SOTA 모델들로 최고 잠재력을 보유하지만, 통합 난이도가 가장 높고 일부는 아직 완전히 공개되지 않음.

### 5.1 TRELLIS.2 (Microsoft)

| Item | Detail |
|------|--------|
| **Paper** | TRELLIS: Structured 3D Latents for Scalable and Versatile 3D Generation |
| **Architecture** | SLAT (Structured Latent) + DiT (Diffusion Transformer), ~4B params |
| **Output** | 3DGS, mesh, radiance field (유연한 출력 형식) |
| **Input** | Single/multi image, text |
| **Code** | [github.com/microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) |
| **Difficulty** | ★★★★★ |

**아키텍처 분석**:
- 3D를 structured latent (SLAT) 공간에서 직접 모델링
- DiT (Diffusion Transformer) 기반 → scalability 우수
- 출력 형식을 3DGS, mesh, radiance field 중 선택 가능

**핵심 혁신**:
- 2D diffusion이 아닌 **3D latent space에서 직접 diffusion** → transfer gap 근본 해결
- 4B 파라미터 규모 → 복잡한 geometry와 appearance 동시 모델링

**장점**:
- 3DGS 직접 출력 가능 → GS-LRM 파이프라인에 직접 연결 가능성
- SOTA 품질 (2025년 기준 최고 수준)
- 유연한 출력 형식
- Microsoft 지원으로 유지보수 기대

**단점**:
- 4B params → A6000 48GB × 4로도 fine-tuning 불가능할 수 있음 (추정 80GB+ 필요)
- 공개 weight의 mouse domain fine-tuning 가능성 미확인
- 파이프라인 통합 복잡도 최상
- 모델 크기로 인한 inference 시간 증가

**Mouse Domain 적합성**: ★★★☆☆ (잠재력 높으나 실현 불확실)
- 3D latent space diffusion은 textureless surface에서도 shape prior 활용 가능
- 단, mouse 같은 niche domain에 대한 사전학습 데이터 부족

**예상 소요**: 8-12주 (환경 구축 + fine-tuning + 파이프라인 통합)

---

### 5.2 Hunyuan3D-2.1 (Tencent)

| Item | Detail |
|------|--------|
| **Paper** | Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation |
| **Architecture** | Multi-view generation + 3D reconstruction (2-stage, tightly coupled) |
| **Output** | Textured mesh + (optional) Gaussians |
| **Input** | Single image or text |
| **Code** | [github.com/Tencent/Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) |
| **License** | MIT (v2.1) |
| **Difficulty** | ★★★★☆ |

**아키텍처 분석**:
- Stage 1: Hunyuan-DiT 기반 multi-view 생성 (고해상도)
- Stage 2: 학습된 reconstruction network으로 3D 복원
- 두 stage가 joint training → transfer gap 최소화

**장점**:
- E2E 학습으로 stage 간 transfer gap 최소화 (FaceLift의 핵심 문제 해결)
- MIT license → 자유로운 수정 및 배포
- 고해상도 multi-view 생성
- 활발한 개발 (v2.0 → v2.1 빠른 업데이트)

**단점**:
- Hunyuan-DiT backbone은 SDXL보다도 무거움
- Fine-tuning 자원 요구량 상당
- Mouse domain에 대한 사전학습 데이터 없음
- 중국어 중심 문서화 (영어 문서 부족할 수 있음)

**Mouse Domain 적합성**: ★★★☆☆
- Joint training 접근은 mouse domain에서도 transfer gap 해결에 유망
- 단, 대규모 fine-tuning 필요

**예상 소요**: 6-8주

---

### 5.3 DiffSplat

| Item | Detail |
|------|--------|
| **Paper** | DiffSplat: Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation |
| **Architecture** | 3D Gaussian space에서 직접 diffusion denoising |
| **Output** | 3D Gaussians (native) |
| **Input** | Single/multi image |
| **Code** | 제한적 공개 (2025 초 기준) |
| **Difficulty** | ★★★★★ |

**아키텍처 분석**:
- 2D 이미지 diffusion을 3D Gaussian 공간으로 repurpose
- Gaussian 파라미터 (position, color, opacity, covariance)에 직접 noise를 가하고 denoise
- 중간 2D representation 완전 제거

**핵심 혁신**:
- **Gaussian-native diffusion** → multi-view 생성이라는 중간 단계 자체를 제거
- 이론적으로 transfer gap이 0 (중간 stage 없음)

**장점**:
- Transfer gap 근본 해결 (중간 표현 없음)
- 출력이 직접 3D Gaussians → FaceLift 렌더링 파이프라인 재활용 가능
- 학술적으로 가장 promising한 방향

**단점**:
- 매우 초기 단계 연구
- 공개 코드/weight 제한적
- 학습 안정성 미검증
- A6000 환경에서의 실행 가능성 미확인

**Mouse Domain 적합성**: ★★☆☆☆ (잠재력 최고, 현실성 최저)
- Gaussian space에서의 직접 diffusion은 shape 정확도에 이론적 최적
- 현 시점에서 실용적 적용 불가

**예상 소요**: 12주+ (코드 구현 포함 시)

---

## 6. Priority Ranking

### Difficulty vs Expected Improvement Matrix

```
Expected E2E Improvement
     ↑
High │              [DiffSplat]
     │         [TRELLIS.2]  [Hunyuan3D]
     │     [MV-Adapter]
     │  [Zero123++] [GRM]
Med  │     [SV3D]       [LGM]
     │                [InstantMesh]
     │  [Era3D]
Low  │
     └──────────────────────────────→
        Low    Medium    High    Very High
                  Difficulty
```

### Quantified Priority Table

| Rank | Candidate | Difficulty (1-5) | Expected Improvement | Effort (weeks) | Risk | Score* |
|------|-----------|-------------------|---------------------|-----------------|------|--------|
| **1** | **MV-Adapter** | 3 | High | 3-4 | Medium | **A** |
| **2** | **Zero123++** | 2 | Medium | 2-3 | Low | **A-** |
| **3** | **Hunyuan3D-2.1** | 4 | High | 6-8 | Medium-High | **B+** |
| **4** | **SV3D** | 3 | Medium | 3-4 | Medium | **B** |
| **5** | **GRM** | 4 | Medium-High | 4-5 | Medium | **B** |
| **6** | **Era3D (updated)** | 1 | Low | 1 | Very Low | **B-** |
| **7** | **LGM** | 4 | Medium | 5-6 | High | **C+** |
| **8** | **TRELLIS.2** | 5 | Very High | 8-12 | Very High | **C** |
| **9** | **InstantMesh** | 4 | Low-Medium | 4-5 | Medium | **C-** |
| **10** | **DiffSplat** | 5 | Very High | 12+ | Very High | **C-** |

*Score = (Expected Improvement × 2 - Difficulty - Risk) / Effort. 정성적 종합 평가.*

### Mouse Domain 특화 Ranking (재평가)

일반 3D reconstruction이 아닌 **mouse reconstruction** 관점에서 재평가하면:

| Rank | Candidate | 이유 |
|------|-----------|------|
| **1** | **MV-Adapter** | 카메라 유연성(M5 매칭), 고해상도, scaffolding 존재 |
| **2** | **Zero123++** | 빠른 통합, view 일관성, 해상도 개선 |
| **3** | **SV3D** | 다수 view 생성이 mouse의 작은 silhouette 보완 |
| **4** | **Hunyuan3D-2.1** | Joint training으로 transfer gap 해결 |
| **5** | **GRM** | Stage 3 교체로 reconstruction 품질 개선 |

---

## 7. Recommended Roadmap

### Phase 1: Quick Win (1-2주)

**목표**: 기존 파이프라인에서 최대한 빠르게 개선 확인

```
Step 1.1: Era3D 최신 공식 weight 비교 (1-2일)
  → 유의미한 차이 없으면 skip

Step 1.2: Zero123++ adapter 구현 (1-2주)
  → Tile 분리 + camera pose 매핑
  → M5 데이터로 quick fine-tuning
  → E2E PSNR 측정하여 baseline 대비 개선 확인
```

**판단 기준**: E2E PSNR 5%+ 개선 시 Phase 2A, 미미할 시 Phase 2B로 진행.

### Phase 2A: Primary Replacement (3-4주)

**목표**: MV-Adapter 통합으로 본격적 Stage 1 개선

```
Step 2A.1: MV-Adapter 환경 구축 (3-5일)
  → SDXL + adapter weight 로드
  → A6000 VRAM 프로파일링

Step 2A.2: GS-LRM 연결 파이프라인 구현 (1주)
  → 기존 mouse_extensions/model/mv_adapter.py 활용
  → Camera pose conditioning with M5 extrinsics
  → Output format adapter (768→512 resize)

Step 2A.3: M5 데이터 fine-tuning (1-2주)
  → Adapter-only training (207M params)
  → Multi-view consistency loss 추가
  → E2E evaluation loop 구축

Step 2A.4: 비교 평가 (2-3일)
  → Fair evaluation protocol 적용
  → Stage 1 PSNR + E2E PSNR + Coverage(IoU) 측정
  → Transfer rate 계산
```

### Phase 2B: E2E Exploration (6-8주, Phase 2A 결과에 따라)

**목표**: Transfer gap이 아키텍처적 한계라면, E2E 접근 탐색

```
Step 2B.1: Hunyuan3D-2.1 환경 구축 + 기본 평가 (2주)
  → Pretrained model로 mouse 입력 테스트
  → Zero-shot 성능 확인

Step 2B.2: Mouse domain fine-tuning (3-4주)
  → M5 데이터셋 adaptation
  → Output format을 Gaussian으로 설정

Step 2B.3: FaceLift 대비 비교 평가 (1주)
```

### Phase 3: Long-term (선택적, 12주+)

```
TRELLIS.2 또는 DiffSplat 기반 연구
→ 학술적 가치가 높으나 실용적 적용까지 시간 소요
→ 논문 출판 목적이라면 고려
```

### Decision Tree

```
                    Era3D 최신 weight 비교
                    ↓
              유의미한 개선?
            /              \
          Yes               No
          ↓                  ↓
     적용 후 평가      Zero123++ adapter
                         ↓
                   E2E 개선 확인?
                  /              \
                Yes               No
                ↓                  ↓
          MV-Adapter           Hunyuan3D-2.1
          (Phase 2A)           (Phase 2B)
              ↓                    ↓
        Transfer rate          E2E 접근으로
        개선 확인?             전환
        /        \
      Yes         No
      ↓            ↓
    완료        Phase 2B
```

---

## 8. Cross-References

### 기존 문서

| Document | Location | 관련 내용 |
|----------|----------|-----------|
| MV-Adapter 기술 분석 | `docs/theory/MV_ADAPTER_TECHNICAL.md` | Adapter 아키텍처 상세 |
| MV-Adapter scaffolding | `mouse_extensions/model/mv_adapter.py` | 초기 구현 코드 |
| Fair comparison protocol | `fair_comparison.py` (gpu03) | 공정 평가 스크립트 |
| Fair test-only eval | `fair_test_only_eval.py` (joon) | PS test-only 평가 |

### 핵심 논문

| Paper | Year | Relevance |
|-------|------|-----------|
| Era3D (Lyu et al.) | ICCV 2025 | 현재 Stage 1 아키텍처 |
| Zero123++ (Shi et al.) | 2024 | Tier 1 후보 |
| SV3D (Voleti et al.) | 2024 | Tier 1 후보 |
| MV-Adapter (Huang et al.) | 2024 | Tier 1 후보 (최우선) |
| LGM (Tang et al.) | ECCV 2024 | Tier 2 후보 |
| TRELLIS (Xiang et al.) | 2025 | Tier 3 후보 |
| Hunyuan3D (Yang et al.) | 2025 | Tier 3 후보 |
| DiffSplat | 2025 | Tier 3 후보 |

### 관련 실험 기록

| Experiment | Key Finding |
|-----------|-------------|
| M5t2 MVDiff checkpoint-5000 | Val PSNR 27.70, E2E PSNR ~3.9 |
| M5t2 GS-LRM 6view_v2 best | GT input → PSNR 24.49 (upper bound) |
| Transfer rate analysis | Stage 1 → E2E ~14% transfer |

---

## Appendix: VRAM & Compute Estimates

### 학습 환경: gpu03 GPU 4-7 (A6000 48GB × 4)

| Model | Inference VRAM | Fine-tuning VRAM (est.) | Feasibility |
|-------|---------------|------------------------|-------------|
| Era3D (SD2.1) | ~12GB | ~24GB (bs=4) | OK |
| Zero123++ (SD2.1) | ~14GB | ~28GB (bs=4) | OK |
| SV3D (SVD) | ~20GB | ~40GB (bs=2) | Tight |
| MV-Adapter (SDXL) | ~22GB | ~38GB (bs=2) | Tight |
| LGM | ~16GB | ~32GB (bs=4) | OK |
| GRM | ~20GB | ~40GB (bs=2) | Tight |
| Hunyuan3D-2.1 | ~30GB | ~48GB+ (bs=1) | Marginal |
| TRELLIS.2 (4B) | ~40GB+ | ~80GB+ | NOT feasible (single GPU) |
| DiffSplat | TBD | TBD | Unknown |

**참고**: 모든 VRAM 추정치는 fp16/bf16 기준. Gradient checkpointing, LoRA, DeepSpeed 등으로 절감 가능.

---

*Stage 1 Replacement Candidates v1.0 | 2026-02-23*
