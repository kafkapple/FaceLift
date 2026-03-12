# Mesh-Guided Diffusion Refinement for Unseen-View Artifact Correction

> **One-liner**: MAMMAL parametric mesh를 structural bridge로 활용하여, bottom-view 입력 없이 3DGS 재구성의 unseen-region artifact를 diffusion model로 보정하는 2-track 전략.

## Executive Summary

- **문제**: 6개 상부 고정 카메라 → bottom-view 데이터 부재 → 가늘고 긴 하얀 아티팩트 다수
- **전략 A**: Mesh-Guided Paired Training — (1) GS-LRM artifact → MAMMAL mesh style, (2) Mesh style → photorealistic
- **전략 B**: View Ablation Augmentation — 뷰 체계적 제거로 degraded/GT 쌍 생성 → diffusion refinement 학습
- **핵심 발견**: "Parametric animal mesh + 3DGS + diffusion refinement" 교차 영역은 사실상 비어 있음 → **높은 novelty**
- **우선순위**: Phase 0 (데이터 수집 PoC) → Phase 1 (zero-shot 적용) → Phase 2 (fine-tuning) → Phase 3 (통합 논문)
- **타겟**: CVPR/ECCV/NeurIPS — 3D Vision / Neural Rendering

---

## 1. Problem Definition

### 1.1 핵심 이슈

FaceLift 파이프라인 (MVDiffusion → GS-LRM)에서 bottom-view 재구성 시:
- GS-LRM이 관측되지 않은 영역에 Gaussian을 무질서하게 배치
- 가늘고 긴 하얀 아티팩트 (elongated white floaters) 다수 발생
- 기존 fair eval: PSNR_fg ~7-8 dB, IoU ~0.5 (coverage 문제 포함)

### 1.2 제약 조건

- Bottom-view 카메라 추가 불가 (하드웨어 제약)
- 생쥐 자유 이동 → 프레임별 pose 변화
- Real-time 요구 없음 (오프라인 분석)

---

## 2. Proposed Strategies

### 2.1 Strategy A — Mesh-Guided Paired Training

MAMMAL parametric mesh를 "구조적 브릿지"로 활용하는 2단계 파이프라인.

#### A-1: Artifact → Mesh Style (Novel View)

| 항목 | 내용 |
|------|------|
| **입력** | GS-LRM rendered image (artifact 포함) at novel view |
| **출력** | MAMMAL mesh UV textured rendering at same camera |
| **학습 목표** | 아티팩트 → 깨끗한 mesh 기반 연속 픽셀값 |
| **모델 후보** | pix2pix, ControlNet (depth-conditioned), DIFIX-3D |

#### A-2: Mesh Style → Photorealistic (GT View)

| 항목 | 내용 |
|------|------|
| **입력** | MAMMAL mesh UV textured rendering at GT camera |
| **출력** | GT camera RGB image |
| **학습 목표** | mesh 렌더링 → 자연스러운 실사 이미지 |
| **모델 후보** | CycleGAN, domain adaptation diffusion |

#### 통합 추론

```
GS-LRM artifact render → [A-1] → mesh-style clean → [A-2] → photorealistic
```

### 2.2 Strategy B — View Ablation Data Augmentation

#### 핵심 관찰

생쥐가 자유롭게 움직이므로, 6개 고정 카메라에서도 다양한 각도/거리의 이미지 정보 수집 가능.
뷰를 체계적으로 줄이면(6→5→4→...→1), 제거된 뷰 방향에서 artifact가 발생 → bottom-view 문제 시뮬레이션.

#### 데이터 생성 파이프라인

```
Full 6-view input → GS-LRM → clean renders (pseudo-GT)
N-view input (N<6) → GS-LRM → degraded renders (artifact)
→ (degraded, clean/GT) 쌍으로 수집
→ DIFIX-3D 등으로 artifact removal 학습
```

#### 장점

- 대규모 학습 데이터 자동 생성 가능
- 다양한 degradation 패턴 포착
- MAMMAL mesh 품질에 비의존적

---

## 3. Literature Survey

### 3.1 핵심 관련 논문

| # | 논문 | Venue | 핵심 방법론 | 관련성 |
|---|------|-------|-------------|--------|
| 1 | **Difix3D+** | CVPR 2025 Best Paper Finalist | Single-step diffusion으로 3DGS artifact 정제 | 즉시 적용 가능 (코드/모델 공개) |
| 2 | **GSFix3D** | arXiv 2025.08 | Mesh + Gaussian dual conditioning diffusion | 우리 접근과 가장 일치 |
| 3 | **GoMAvatar** | CVPR 2024 | SMPL mesh 위에 Gaussian 배치 (Gaussians-on-Mesh) | MAMMAL에 적용 가능한 선례 |
| 4 | **Deceptive-NeRF/3DGS** | ECCV 2024 | Diffusion으로 pseudo-observation 생성 → 재학습 | Strategy B 이론적 기반 |
| 5 | **FreeFix** | arXiv 2026.01 | Fine-tuning 없이 per-pixel confidence 기반 selective refinement | Zero-shot 적용 가능 |
| 6 | **MILo** | 2025 | Mesh-in-the-Loop: differentiable mesh ↔ Gaussian 양방향 gradient | Mesh prior 직접 통합 |
| 7 | **SuGaR** | CVPR 2024 | Surface-aligned Gaussian + Poisson mesh → 공동 최적화 | Mesh-Gaussian binding 기초 |
| 8 | **Mani-GS** | CVPR 2025 | Gaussian-Mesh binding + self-adaption (mesh 부정확해도 OK) | MAMMAL 부정확성 tolerance |
| 9 | **3DGS-Enhancer** | NeurIPS 2024 | Video diffusion prior로 view consistency as temporal consistency | Multi-view consistency 참고 |
| 10 | **GSFixer** | arXiv 2025.08 | DiT video diffusion + reference view semantic/geometric feature | Reference-conditioned refinement |
| 11 | **FixingGS** | arXiv 2025.09 | Training-free score distillation, adaptive progressive enhancement | Zero-shot, unreliable viewpoint 감지 |
| 12 | **ArtiFixer** | arXiv 2025.03 | Auto-regressive diffusion, 3D representation as conditioning | 3D conditioning 철학 |
| 13 | **StruGauAvatar** | 2025 | Anchored + Free Gaussian dual structure on DMTet mesh | Mesh-aligned + free Gaussian 분리 |
| 14 | **LooseControl** | SIGGRAPH 2024 | ControlNet depth conditioning을 coarse depth로 확장 | MAMMAL depth map conditioning |
| 15 | **RealisticDreamer** | arXiv 2025.11 | Video diffusion SDS + real depth warping guidance | Few-view depth guidance |

### 3.2 Research Gap Analysis

| 영역 | 기존 연구 현황 | Gap |
|------|---------------|-----|
| 3DGS artifact correction | Difix3D+, FreeFix, FixingGS 등 활발 | General scene 중심, animal 도메인 미검증 |
| Mesh-guided 3DGS | GSFix3D (유일하게 dual conditioning) | Parametric animal mesh 활용 사례 전무 |
| Hybrid Mesh+GS | GoMAvatar, SuGaR, Mani-GS | Human(SMPL) 전용, animal 확장 미검증 |
| Camera topology artifact | 없음 | 특정 카메라 배치의 구조적 artifact 분석 부재 |
| View ablation augmentation | Deceptive-NeRF 유사 개념 | 체계적 뷰 제거 기반 augmentation 미문서화 |

### 3.3 Novelty 근거

> **"Parametric animal mesh (MAMMAL) + 3DGS (GS-LRM) + diffusion refinement"의 교차 영역은 사실상 비어 있다.**

- GSFix3D만이 mesh+Gaussian dual conditioning 시도 (2025.08, peer review 미완)
- Human 도메인(SMPL)은 풍부하나 Animal parametric model 활용 연구 전무
- Camera topology에 따른 구조적 artifact 분석 연구 부재

---

## 4. Feasibility Assessment

### 4.1 Strategy A

| 항목 | 평가 | 근거 |
|------|------|------|
| 기술적 실현 가능성 | ★★★★☆ | Image-to-image translation 성숙 기술, 좌표계 통일이 관건 |
| Novelty | ★★★★★ | Parametric animal mesh as semantic bridge — 선례 없음 |
| 리스크 | ★★★☆☆ (중간) | MAMMAL fitting 품질 의존, cascading error, domain gap |
| 논문 가치 | ★★★★★ | 새로운 연구 영역 개척 |

**전제 조건**: MAMMAL mesh의 GT view fitting 품질이 충분해야 함 → Phase 0에서 검증

### 4.2 Strategy B

| 항목 | 평가 | 근거 |
|------|------|------|
| 기술적 실현 가능성 | ★★★★★ | 데이터 생성 자동화 + supervised learning |
| Novelty | ★★★☆☆ | View ablation augmentation은 기존 개념의 확장 |
| 리스크 | ★★☆☆☆ (낮음) | 데이터 기반이라 견고, MAMMAL 무관 |
| 논문 가치 | ★★★★☆ | 체계적 데이터 증강 전략으로서 기여 |

**핵심 가정**: View ablation으로 생긴 artifact가 실제 bottom-view artifact와 유사한 패턴인지 검증 필요

### 4.3 Comparative Summary

| 기준 | Strategy A | Strategy B | A+B Hybrid |
|------|-----------|-----------|------------|
| Quick-win | ☆ | ★★★ | ★★ |
| 최종 품질 잠재력 | ★★★★★ | ★★★★ | ★★★★★ |
| 구현 복잡도 | 높음 (2-stage) | 중간 | 높음 |
| MAMMAL 의존도 | 높음 | 없음 | 중간 |
| 3D consistency | 간접 (mesh guide) | 간접 (data-driven) | 양쪽 |

---

## 5. Risk Assessment

### 5.1 공통 리스크

- **카메라 좌표계 불일치**: MAMMAL, GS-LRM, Blender 간 intrinsic/extrinsic, Y-up vs Z-up 통일 필수
- **DIFIX-3D 도메인**: Mouse 도메인 fine-tuning 없이 성능 보장 불가
- **3D consistency**: 2D image-to-image 보정은 multi-view consistency를 깨뜨릴 위험

### 5.2 Strategy A 리스크

- MAMMAL mesh ≠ GT (approximate parametric model)
- Mesh 렌더링 ↔ 실제 카메라 이미지 간 domain gap
- A-1 → A-2 cascading error 증폭

### 5.3 Strategy B 리스크

- View ablation artifact ≠ actual bottom-view artifact (각도/패턴 차이)
- Over-smoothing: 아티팩트 제거 시 fine detail(털 등) 손실
- 대규모 데이터 생성/학습의 계산 비용

---

## 6. Implementation Plan

### Phase 0 — 데이터 수집 PoC (즉시, 1-3일)

| # | 작업 | 목적 | 산출물 |
|---|------|------|--------|
| 0-1 | GT 6뷰: GT RGB vs MAMMAL mesh 렌더링 | MAMMAL fitting 품질 검증 | Image pairs + PSNR/SSIM |
| 0-2 | Novel view: MAMMAL vs GS-LRM 렌더링 | Artifact 시각화 + mesh bridge 가능성 | Image pairs + comparison grid |
| 0-3 | View ablation (6→5→4→3) | Degradation 패턴 확인 | Ablation renders + metrics |

### Phase 1 — Zero-Shot Quick Win (1-2주)

| 우선순위 | 작업 | 방법 |
|----------|------|------|
| P1 | Difix3D+ zero-shot | 코드/모델 공개, bottom-view에 바로 적용 |
| P2 | FreeFix zero-shot | Per-pixel confidence selective refinement |
| P3 | Strategy B 데이터 파이프라인 | View ablation → (degraded, GT) 쌍 자동 수집 |

### Phase 2 — Fine-tuning (2-4주)

| 우선순위 | 작업 | 방법 |
|----------|------|------|
| P4 | Strategy A-1 | GS-LRM → MAMMAL: pix2pix baseline → ControlNet |
| P5 | Strategy B | DIFIX-3D mouse 도메인 fine-tuning |
| P6 | Strategy A-2 | MAMMAL → Photorealistic: domain transfer |

### Phase 3 — Integration & Paper (4-8주)

| 작업 | 방법 |
|------|------|
| A+B 하이브리드 | Mesh-conditioned diffusion + ablation augmentation |
| Comprehensive ablation study | Baseline vs P1 vs P2 vs P3 |
| Paper writing | CVPR/ECCV/NeurIPS submission |

---

## 7. Paper Positioning

### 제목 방향

*"Mesh-Guided Diffusion Refinement for Unseen-View Artifact Correction in Multi-View 3D Gaussian Splatting"*

### 핵심 기여 (Contributions)

1. Parametric animal mesh를 structural prior로 활용한 3DGS artifact 보정 (첫 시도)
2. View ablation 기반 체계적 data augmentation for diffusion refinement
3. Mesh-bridge 2-stage pipeline: artifact → mesh style → photorealistic
4. Multi-view mouse reconstruction에서의 comprehensive ablation study

### 타겟 학회

CVPR, ECCV, NeurIPS — 3D Vision / Neural Rendering track

---

## Related Documents

- `↑ MOC`: `mouse_extensions/docs/` (domain docs)
- `↔ Architecture`: `GSLRM_ARCHITECTURE_ANALYSIS.md`
- `↔ Keypoints`: `KEYPOINT_VIZ.md`
- `↔ Fair Eval`: `mouse_extensions/scripts/eval/fair_comparison.py`
- `↓ Experiments`: (Phase 0 PoC results — TBD)

---

*FaceLift Research Note | Created: 2026-03-11 | Author: joon*
