# Phase 2: Novel View & Multi-Species Roadmap

> **Version**: v2.0 | **Created**: 2026-03-12 | **Status**: ACTIVE
> **Navigation**: [← INDEX](../INDEX.md) | [hypothesis_roadmap](hypothesis_roadmap.md) | [mesh_gs_pair_collection](mesh_gs_pair_collection.md)
> **Target**: NeurIPS 2026 (Evaluations & Datasets Track primary, Main Track secondary)
> **v2.0 변경**: Novel View Enhancement 파이프라인 추가 (§3.3-3.5), 우선순위 재정립, UV texture 이슈 반영

---

## 1. Phase Structure Overview

```
Phase 1: MVDiff Bottleneck Resolution (기존 작업, 유지보수 모드)
  ├── Silhouette loss, Domain Adaptation, Spatial Token
  ├── Goal: Break E2E ceiling (8.44 → 13+ dB)
  └── Status: H7v2 완료(미평가), DA1 미시작, Sil loss 미구현

Phase 2: Novel View + Multi-Species + Dataset (★ 현재 포커스)
  ├── Novel view rendering (bottom, body-part cam follow, extreme angles)
  ├── Multi-species generalization (s-DANNCE: rat, mouse, marmoset)
  ├── Benchmark dataset construction
  └── Goal: NeurIPS 2026 Evaluations & Datasets Track submission
```

### Phase 간 관계

| 항목 | Phase 1 | Phase 2 |
|------|---------|---------|
| **핵심 질문** | MVDiff 병목 해결 가능한가? | Novel view + multi-species benchmark 기여 |
| **GS-LRM** | 공통 활용 (6v best: 23.84 dB) | 공통 활용 |
| **평가 인프라** | fair_comparison.py, metrics_v2 | 공통 + novel view metrics 확장 |
| **데이터셋** | M5t2 (mouse, 3600 frames) | M5t2 + s-DANNCE (multi-species) |
| **시각화** | turntable_renderer, alpha_viz | 공통 + novel camera trajectories |
| **좌표계** | COORDINATE_SYSTEMS.md | 공통 |
| **독립성** | Phase 2는 Phase 1 성공에 의존하지 않음 (GT 6v 입력 활용 가능) |

---

## 2. Research Narrative

### 2.1 핵심 논문 스토리

**"BehaviorSplatter: Multi-Species Novel-View 3D Reconstruction Benchmark for Animal Behavior Analysis"**

기존 동물 행동 분석은 고정된 카메라 뷰에 제한됨. 3D Gaussian Splatting으로 재구성하면 **임의 시점**(bottom view, body-part camera follow)에서 관찰 가능 → 기존에 불가능했던 행동 패턴 분석 가능.

### 2.2 차별화 전략

| vs | 차별점 |
|----|--------|
| **Pose Splatter** (NeurIPS 2025) | Full-surface appearance (not just skeleton) + scientifically motivated novel views + multi-species benchmark |
| **Animal4D** (NeurIPS 2025) | Lab-scale calibrated multi-view (not web video) + controllable novel views + behavior-driven evaluation |
| **s-DANNCE** (Nat MI 2023) | 3D appearance reconstruction (not just keypoints) + novel view synthesis |
| **PAIR-R24M** | Appearance + novel view evaluation (not just pose metrics) |

### 2.3 Target Track

**Primary: NeurIPS 2026 Evaluations & Datasets Track**
- Multi-view animal NVS benchmark 부재 → 독자적 기여
- Dataset + evaluation protocol + baseline comparisons
- Double-blind 면제 (dataset 특성)

**Secondary: Main Track**
- Body-part camera follow의 방법론적 혁신이 충분하면 전환 가능

---

## 3. Novel View Types & Scientific Value

### 3.1 Priority Novel Views

| View Type | Elevation | Azimuth | Scientific Value | 난이도 |
|-----------|:---------:|:-------:|-----------------|:------:|
| **Bottom** | -70° | 0° | Gait analysis, paw-ground interaction, balance | Low (infra ready) |
| **Top** | +70° | 0° | Social interaction topology, arena coverage | Low |
| **Front Low** | -30° | 0° | Facial expression, whisker movement | Low |
| **Side Low** | -30° | 90° | Lateral gait, body elongation | Low |
| **Body-Part Cam Follow** | Dynamic | Dynamic | Micro-behavior (grooming, feeding, social) | **High** |
| **Orbital Sweep** | Variable | 0→360° | Full coverage consistency check | Low (turntable ready) |

### 3.2 Body-Part Camera Follow 설계

```
Input: 22-joint keypoint trajectory (DANNCE/MAMMAL)
Target joints: head, spine_mid, tail_base, left_paw, right_paw

Camera Generation Pipeline:
1. Select target joint j for frame t
2. Get 3D position: p_j(t) from triangulation
3. Generate look-at camera: eye = p_j(t) + offset, target = p_j(t)
4. Smooth trajectory: CubicSpline over [t-w, t+w]
5. Render from GS-LRM reconstructed Gaussians

Reusable modules:
- mouse_extensions/analysis/triangulation_analysis.py (joint positions)
- mouse_extensions/visualization/turntable_renderer.py (make_look_at_c2w)
- mouse_extensions/inference/modules/renderer.py (render_from_cameras)
```

### 3.3 Novel View Enhancement Pipeline (3-Model Consensus)

> GS-LRM novel view renders에서 Gaussian artifact(white spikes/splats) 제거 + 품질 향상 전략.
> **핵심 교훈**: 텍스트 프롬프트만으로는 view direction을 교정할 수 없음 → 전처리 분리 필수.

```
[GS-LRM Novel View Render]
    │
    ├─ Tier 0: Raw GS Render (원본, 벤치마크 primary)
    │
    ├─ Stage 1: Classical Inpainting (OpenCV)
    │   ├─ Morphological filtering (thin white vertical structures)
    │   ├─ cv2.inpaint(TELEA) — view-agnostic, 원본 보존
    │   └─ Output: Tier 1 (Artifact-removed)
    │
    ├─ Stage 2: AI-Enhanced Texture (선택적)
    │   ├─ Nano Banana API (fal.ai) 또는 Gemini image editing
    │   ├─ 단계별: Level 1(보간 정리) → Level 2(모피 향상) → Level 3(포토리얼)
    │   └─ Output: Tier 2 (AI-enhanced, 벤치마크 GT 불가, qualitative only)
    │
    └─ Reference: MAMMAL Mesh Render (pseudo-GT for novel views)
```

#### 벤치마크 데이터 제공 구조 (3-Model Consensus)

| Tier | 데이터 | 벤치마크 역할 | 제공 |
|:----:|--------|-------------|:----:|
| **Tier 0** | Raw GS Render | **Primary evaluation target** | ✅ 필수 |
| **Tier 1** | Stage 1 cleaned | Post-processed comparison | ✅ 필수 |
| **Tier 2** | Stage 2 AI-enhanced | Qualitative demo only (논문 Figure) | ⚠️ 선택 |
| **Ref** | MAMMAL mesh render | **Pseudo-GT** for novel views w/o real GT | ✅ 필수 |
| **Mask** | Artifact binary masks | Annotation (후속 연구용) | ✅ 권장 |

> ⚠️ **Stage 2 주의**: AI 모델의 hallucination, view inconsistency, reproducibility 문제로 벤치마크 GT로 사용 불가. 논문에 명확한 disclaimer 필수.

#### 모델별 실험 설정 (Experiment Configuration)

**GS-LRM (Tier 0 생성)**

| 항목 | 설정 |
|------|------|
| Checkpoint | `base_uniform_v2_6view_v2/best_psnr.pt` |
| Input | 6-view GT (M5t2, 384×384) |
| Precision | bf16 (학습 시), fp32 (추론 시 render) |
| Render resolution | 384×384 |
| Background | White (1.0, 1.0, 1.0) |

**MAMMAL Mesh (Pseudo-GT)**

| 항목 | 설정 |
|------|------|
| Fitting config | `v012345_kp22` (6 views × 22 keypoints, 최대) |
| Renderer | pyrender (OffscreenRenderer, EGL backend) |
| UV texture | 512×512 PNG, face-level vertex expansion mapping |
| Lighting (textured) | Ambient=1.0, Key=5.0, 4× Fill=3.0 |
| Lighting (flat) | Ambient=0.3, Directional=3.0 |

**Stage 1: OpenCV Artifact Removal**

| 항목 | 설정 |
|------|------|
| Detection | Morphological filtering (thin vertical white structures) |
| Inpainting | `cv2.inpaint(TELEA)`, radius=3 |
| Thresholds | TBD (P1에서 튜닝) |

**Stage 2: AI Enhancement Models**

| 모델 | API | 양자화/설정 | 비용 | 용도 |
|------|-----|-----------|------|------|
| **Nano Banana** | fal.ai (`fal-ai/nano-banana`) | Default (API-managed) | ~$0.01/img | Artifact smoothing, texture refinement |
| **Gemini** | Google AI (`gemini-2.0-flash`) | Default (API-managed) | ~$0.005/img | Image editing with text prompt |
| **Stable Diffusion Inpaint** | Local / HF | fp16, guidance_scale=7.5, strength=0.3-0.6 | Free (GPU) | Mask-guided inpainting (Tier 1 mask 활용) |

> **재현성**: Stage 2 실험 시 반드시 기록 — API model version, seed, prompt, timestamp, input/output pair 보존.
> **양자화 주의**: Local 모델 사용 시 fp16 vs fp32 결과 차이 문서화. API 모델은 양자화 제어 불가 (서버 측 관리).

### 3.4 UV Texture Bug & Mesh Pipeline (선결 과제)

| 항목 | 상태 | 상세 |
|------|:----:|------|
| UV seam vertex expansion bug | ✅ 해결됨 | 14,522→~40K vertex expansion → normals 왜곡 ("skull" artifact). Fix: trimesh native OBJ + face-level correspondence |
| Per-frame OBJ loading | ✅ 해결됨 | `expanded_to_orig` mapping으로 원본 topology 유지 |
| Frame indexing (step=5) | ✅ 해결됨 | M5 idx N → MAMMAL frame N*5 |
| MAMMAL fitting quality | ✅ 최적 | `v012345_kp22` config (6 views × 22 keypoints, 최대 설정) |
| Per-sample centering | ✅ 이해됨 | 카메라 centroid→origin, 평균 거리→2.7 정규화 |

> MAMMAL mesh = **pseudo-GT** (template-based, 절대적 GT 아님). Keypoint detection error + occlusion + template constraint 존재.

### 3.5 Novel View Evaluation Protocol (GT 부재 시)

Novel view에는 실제 GT 이미지가 없으므로 다중 메트릭 조합:

| Category | Metric | 설명 |
|----------|--------|------|
| **vs Pseudo-GT** | PSNR, SSIM, LPIPS | GS render vs MAMMAL mesh render |
| **Perceptual (no-ref)** | NIQE, BRISQUE | GT 없이 이미지 품질 평가 |
| **Artifact-specific** | Spike count, total area, density | Stage 1 detection 활용 |
| **Temporal** | Optical flow consistency, feature tracking smoothness | 시퀀스 일관성 |
| **Multi-view** | Epipolar consistency, normal map consistency | 뷰 간 일관성 |
| **Semantic** | CLIP-score | "ventral view of mouse" vs rendered image |
| **Human eval** | Likert scale (realism, artifact, consistency) | 벤치마크 최종 검증 |

### 3.6 Artifact Removal at Scale (3600 frames × 10 views)

```
Strategy A: Synthetic Artifact Injection (추천)
  1. Clean source: MAMMAL mesh render (UV bug fixed)
  2. Artifact modeling: 실제 GS spike 패턴 분석 (길이, 폭, 각도, 밀도, 위치)
  3. Programmatic injection: clean image + synthetic artifacts → training pairs
  4. Train: lightweight inpainting model (or tune Stage 1 params)

Strategy B: Semi-supervised + Active Learning
  1. Stage 1 (OpenCV) → initial artifact masks
  2. High-uncertainty samples → expert review
  3. Iterative refinement

Scale: 3600 × 10 = 36,000 images
  - Stage 1 (OpenCV): ~0.1s/image → ~1hr total (embarrassingly parallel)
  - Stage 2 (API): ~2s/image → ~20hr, cost concern → selective application
```

---

## 4. Multi-Species Strategy

### 4.1 Species Roadmap

| Priority | Species | Dataset Source | Cameras | Frames | Status |
|:--------:|---------|---------------|:-------:|:------:|:------:|
| **P0** | Mouse | M5t2 (DANNCE) | 6 | 3,600 | ✅ Ready |
| **P1** | Rat | Rat7M / PAIR-R24M | 6-24 | 7M+ | 📋 Data prep needed |
| **P2** | Marmoset | s-DANNCE | 6 | TBD | 📋 Data availability check |
| P3 | Chickadee | s-DANNCE | 6 | TBD | ⏳ Optional stretch |

### 4.2 Cross-Species Evaluation Protocol

```
Per species:
  1. Camera calibration → M5 format conversion
  2. GS-LRM zero-shot evaluation (mouse-trained → new species)
  3. GS-LRM fine-tune (species-specific, ~1K frames)
  4. Novel view rendering (all 6 types above)
  5. Metrics: PSNR, IoU, LPIPS, temporal consistency (TC)
  6. Body-part cam follow (species-specific joints)
```

### 4.3 Minimum Viable Dataset (NeurIPS 수준)

| 요구사항 | 최소 | 권장 |
|----------|:----:|:----:|
| Species | 2 (mouse + rat) | 3 (+ marmoset) |
| Frames per species | 500 test | 1000+ test |
| Novel view types | 4 (bottom, top, front_low, side_low) | 6 (+ body-part follow) |
| Behavior categories | 3 (locomotion, grooming, resting) | 5+ |
| Evaluation metrics | PSNR, IoU, TC | + LPIPS, FID, behavior classification acc |

---

## 5. Priority Ordering (v2.0 — Novel View Enhancement 중심)

> Claude / Gemini / GPT-4o 심의 종합 (2026-03-12 2차 심의 반영)
> **현재 포커스**: 6v GT 기반 novel view 생성 + enhancement 실험

| 순위 | 작업 | 근거 | ETA | 의존성 |
|:----:|------|------|:---:|:------:|
| **P0** | Novel view rendering (bottom/top/front_low/side_low) | 인프라 ready, 6v GT 활용 | 1-2일 | — |
| **P1** | Stage 1 artifact removal (OpenCV inpainting) | View-agnostic, 즉시 적용 가능 | 1일 | P0 |
| **P2** | MAMMAL mesh novel view render (pseudo-GT) | UV bug 해결됨, 비교 기준 확보 | 1-2일 | P0 |
| **P3** | Artifact mask generation + dataset annotation | 벤치마크 annotation, 후속 연구 자산 | 1일 | P1 |
| **P4** | Stage 2 AI enhancement 실험 (Nano Banana/Gemini) | Qualitative demo, 논문 Figure | 2-3일 | P1 |
| **P5** | Body-part camera follow 구현 | 최대 차별화, 22-joint pipeline 재활용 | 3-5일 | P0 |
| **P6** | Temporal consistency 평가 | Novel view 시퀀스 품질 검증 | 2-3일 | P0 |
| **P7** | Multi-species data prep (Rat7M) | NeurIPS 일반성 입증 | 3-5일 | — |
| **P8** | Evaluation protocol 구축 (no-ref + pseudo-GT) | §3.5 메트릭 구현 | 2-3일 | P2 |
| **P9** | s-DANNCE marmoset data | 3종 달성 | 5-7일 | P7 |
| P10 | Behavior label integration | downstream task 입증 | 5-7일 | P5 |
| P11 | H7v2 E2E eval (Phase 1) | Phase 1 잔여, 비용 0 | 0.5일 | — |
| P12 | Silhouette loss (Phase 1) | E2E 개선 | 3-5일 | — |

### Critical Path (최단 경로)

```
Week 1: P0 → P1 → P2 → P3 (Novel view dataset 기본 구축)
Week 2: P4 + P5 (AI enhancement 실험 + body-part follow)
Week 3: P6 + P7 + P8 (Temporal eval + multi-species + eval protocol)
Week 4: P9 + P10 (Marmoset + behavior labels)
```

---

## 6. Shared Infrastructure (Phase 1 ↔ Phase 2)

| Module | Location | Phase 1 | Phase 2 |
|--------|----------|:-------:|:-------:|
| GS-LRM checkpoints | `6view_v2/best_psnr.pt` | ✅ | ✅ |
| fair_comparison.py | `scripts/eval/` | ✅ | ✅ (extended) |
| turntable_renderer.py | `visualization/` | ✅ | ✅ |
| poc_mesh_gs_pairs.py | `scripts/eval/` | — | ✅ |
| triangulation_analysis.py | `analysis/` | — | ✅ |
| COORDINATE_SYSTEMS.md | `mouse_extensions/docs/` | ✅ | ✅ |
| M5 preprocessing | `preprocessing/` | ✅ | ✅ (multi-species extension) |
| Config system | `configs/mouse/` | ✅ | ✅ |

---

## 7. Risks & Mitigations

| Risk | Severity | Mitigation |
|------|:--------:|------------|
| Stage 1 false positive (실제 털을 artifact로 오인) | 🟡 | Threshold 보수적 설정 + manual validation on 100-sample subset |
| Stage 2 AI view inconsistency / hallucination | 🔴 | 벤치마크 GT로 사용 불가, qualitative only + disclaimer |
| Stage 2 API reproducibility | 🟡 | 결과 이미지 저장 + seed 고정 + API 버전 기록 |
| MAMMAL mesh pseudo-GT 정확도 한계 | 🟡 | Template fitting 한계 명시, "reference" not "GT"로 표기 |
| Nano Banana/Gemini API 비용 | 🟡 | 선택적 적용 (논문 Figure용 소수 프레임만) |
| s-DANNCE raw video 접근 불가 | 🔴 | PAIR-R24M (public) 대체, Rat7M 우선 |
| Body-part follow 렌더링 품질 저하 | 🟡 | Close-up 대신 medium shot, smooth trajectory |
| NeurIPS deadline miss (~May 2026) | 🟡 | MVP: mouse only, 4 novel views + Stage 1 + pseudo-GT |
| GPU 자원 부족 (GPU 0-3 타인 사용) | 🟡 | GPU 4-7 활용, 5-7 점유 확인 필요 |

---

## 8. Related Documents

| Document | Content |
|----------|---------|
| [[hypothesis_roadmap]] | Phase 1 가설 테스트 SSOT |
| [[mesh_gs_pair_collection]] | Mesh-GS pair pipeline (Phase 2 핵심) |
| [[../../mouse_extensions/docs/COORDINATE_SYSTEMS]] | 좌표계 변환 참조 |
| [[../../mouse_extensions/docs/UV_TEXTURE_RENDERING_BUG]] | UV texture bug 분석 & 수정 (✅ 해결) |
| [[../../mouse_extensions/docs/MESH_GUIDED_REFINEMENT]] | Mesh-guided artifact refinement 전략 (📋 계획) |
| [[KEYPOINT_3D_PIPELINE]] | 22-joint triangulation |
| [[STAGE1_REPLACEMENT_CANDIDATES]] | Stage 1 대체 모델 (Phase 1) |
| [[FL_vs_PS_comparison]] | FL vs PS 비교 (Phase 1) |
| [[NOVEL_VIEW_RENDERING]] | Novel view 렌더링 문서 |
| [[DIFIX_TRAINING_STRATEGY]] | DiFix 3D+ 2.5-stage training strategy |

---

## 9. Competitive Landscape & NeurIPS Positioning

| 경쟁자 | 데이터 | 방법 | 우리 차별점 |
|--------|--------|------|------------|
| **Pose Splatter** (NeurIPS 2025) | Lab multi-view | 3DGS + shape carving → skeleton | Full-surface appearance + novel view benchmark + artifact annotation |
| **Animal4D** (NeurIPS 2025) | Web monocular 30K video | 4D quadruped reconstruction | Lab-calibrated multi-view + controllable views + multi-tier quality |
| **s-DANNCE** (Nat MI 2023) | Lab 6-camera | 3D keypoint tracking | 3D appearance (not just keypoints) + NVS evaluation |
| **PAIR-R24M** | Lab 24-camera | Rat pair pose | Multi-species appearance reconstruction |

### NeurIPS 2026 Eval & Datasets Track 기여 포인트

1. **첫 multi-view animal NVS benchmark** (lab-scale, calibrated)
2. **Multi-tier quality dataset** (Raw GS / Cleaned / AI-enhanced / Mesh pseudo-GT)
3. **Artifact annotation** (binary masks → 후속 연구 자산)
4. **Novel view evaluation protocol** (no-ref + pseudo-GT + temporal consistency)
5. **Cross-species generalization** (mouse → rat → marmoset)

---

*Phase 2 Roadmap v2.0 | Updated: 2026-03-12 | BehaviorSplatter NeurIPS 2026*
