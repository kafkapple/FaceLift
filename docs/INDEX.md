# FaceLift Mouse Documentation

> **MoC (Map of Content)** — Central hub for all project documents.
> **Updated**: 2026-03-16 | **Version**: v14.0
> **Active**: 33개 | **Archive**: 16개 (docs/_archive/, git history에 보존)
>
> **⭐ Phase Structure**: Phase 1 (MVDiff bottleneck) = 유지보수 | **Phase 2 (Novel View + Multi-Species) = 현재 포커스**

---

## Quick Links

| 목적 | 문서 |
|------|------|
| **Phase 2 로드맵 (START HERE)** | [[experiments/PHASE2_NOVEL_VIEW_ROADMAP]] **v1.0** |
| **FL vs PS 비교 (SSOT)** | [[experiments/FL_vs_PS_comparison]] **v11** |
| **가설 SSOT (로드맵)** | [[experiments/hypothesis_roadmap]] **v3.0** |
| **평가 프로토콜** | [[experiments/evaluation_protocol_v1]] **v1.0** |
| **명령어 SSOT** | [[experiments/COMMANDS]] |
| **DiFix 학습 전략** | [[experiments/DIFIX_TRAINING_STRATEGY]] **v1.0** |
| **Alpha Loss 분석** | [[experiments/ALPHA_LOSS_NOVEL_VIEW_ANALYSIS]] **v1.0** |
| **파이프라인 아키텍처** | [[theory/PIPELINE_ARCHITECTURE]] |
| **Mesh Rendering** | [[theory/MAMMAL_MESH_RENDERING_PIPELINE]] **v1.0** |

---

## Phase Structure (2026-03-12~)

| Phase | Focus | Status | Key Doc |
|:-----:|-------|:------:|---------|
| **Phase 1** | MVDiff bottleneck (Sil loss, DA, Spatial Token) | 🔧 유지보수 | [[hypothesis_roadmap]] |
| **Phase 2** ⭐ | Novel View + Multi-Species + NeurIPS Dataset Track | **🔬 ACTIVE** | [[experiments/PHASE2_NOVEL_VIEW_ROADMAP]] |

> 기본 포커스는 Phase 2. Phase 1 작업은 명시적 요청 시에만.

---

## 문서 구조

### experiments/ (실험 & 비교)

#### Core Documents (최신, 빈번 참조)

| 문서 | 내용 | 상태 | 관련 |
|------|------|:----:|------|
| **[[PHASE2_NOVEL_VIEW_ROADMAP]]** | **⭐ Phase 2 로드맵**: Novel View + Multi-Species + NeurIPS Dataset Track | **✅ v1.0** | → mesh_gs_pair, KEYPOINT_3D |
| **[[DIFIX_TRAINING_STRATEGY]]** | DiFix 3D+ 학습 전략: 2.5-stage curriculum, 3 pair types, data pipeline | **✅ v1.0** | → mesh_gs_pair, PHASE2 |
| **[[ALPHA_LOSS_NOVEL_VIEW_ANALYSIS]]** | Alpha loss novel view artifact 억제 효과 분석 | **✅ v1.0** | → H6, PHASE2 |
| **[[FL_vs_PS_comparison]]** | FL vs PS 통합 비교 (9-exp) | **✅ v11** | → eval_protocol |
| **[[hypothesis_roadmap]]** | 가설 검정 결과 + 우선순위 | ✅ v3.0 | → H4-H7 |
| **[[comprehensive_analysis_report]]** | 종합 실험 보고서 (H1-H8) | **✅ v1.0** | → 모든 실험 |
| **[[evaluation_protocol_v1]]** | NVS 평가 프로토콜 (Fair Eval) | **✅ v1.0** | → FL_vs_PS |
| **[[mesh_gs_pair_collection]]** | Novel view dataset pipeline v2.0 | **✅ v2.0** | → PHASE2 |
| **[[DATASET_QA_VIEWER]]** | Novel view dataset QA viewer | **✅ v1.0** | → mesh_gs_pair |

#### Reference Documents (안정, 참조용)

| 문서 | 내용 | 상태 |
|------|------|:----:|
| ~~training_optimal_settings~~ | _(삭제됨 — bottleneck에 통합)_ | 📦 |
| ~~mvdiff_improvement_roadmap~~ | _(삭제됨 — bottleneck에 통합)_ | 📦 |
| ~~FL_PS_metric_consistency~~ | _(삭제됨 — evaluation_protocol에 통합)_ | 📦 |
| ~~260216_PHASE3_REPORT~~ | _(삭제됨 — comprehensive_analysis_report로 대체)_ | 📦 |
| ~~260216_MVDIFF_TRAINING_ANALYSIS~~ | _(삭제됨 — comprehensive_analysis_report로 대체)_ | 📦 |
| **[[REPORT_SYSTEM_GUIDE]]** | HTML 레포트 시스템 통합 가이드 (코드+메트릭+프로토콜) | **✅ v1.0** |
| ~~STAGE1_REPLACEMENT_CANDIDATES~~ | _(archived → Phase 1)_ | 📦 |
| ~~mvdiffusion_bottleneck_analysis~~ | _(archived → Phase 1)_ | 📦 |
| ~~domain_adaptation_DA1~~ | _(archived → Phase 1)_ | 📦 |

#### Operational Guides

| 문서 | 내용 | 상태 |
|------|------|:----:|
| [[COMMANDS]] | 명령어 SSOT (GS-LRM, MVDiff, Turntable) | ✅ |
| [[EXPERIMENT_CONFIG_GUIDE]] | Config 시스템 (Modular 3-Layer) | ✅ |
| [[EXPERIMENT_REGISTRY]] | 실험 설정/결과 기록 | ✅ |
| [[INFERENCE_E2E_GUIDE]] | E2E 추론 파이프라인 + CLI | ✅ |
| ~~TRAINING_LOGGING_GUIDE~~ | _(archived → Phase 1)_ | 📦 |
| ~~VISUALIZATION_SETTINGS~~ | _(archived → Phase 1)_ | 📦 |
| **[[KEYPOINT_3D_PIPELINE]]** | 3D Keypoint 삼각측량 + FL vs PS 비교 | **🆕 v1.0** |
| **[[DATASET_QA_VIEWER]]** | Novel view dataset QA viewer (stdlib HTTP, exclude 관리) | **🆕 v1.0** |

### datasets/

| 문서 | 내용 |
|------|------|
| **[[PREPROCESSING_REGISTRY]]** | 전처리 SSOT (M/D-series 전체) |
| [[M5_SERIES_SPEC]] | M5 카메라 정규화 상세 (**← camera mismatch 핵심 참조**) |
| [[RAW_DATA]] | Raw 데이터 출처 (DANNCE) |

### guides/ (입문 & 실습)

| 문서 | 내용 | 상태 |
|------|------|:----:|
| **[[EXPERIMENT_MASTER_GUIDE]]** | **전체 실험 마스터 가이드 (초심자 시작점)** | **🆕 v1.0** |
| **[[RESEARCH_EXPERIMENT_NOTES]]** | **연구 노트 (Lab Meeting용, 가설→실험→결과)** | **🆕 v1.0** |
| → [[chapters/CH1_ENVIRONMENT_AND_DATA]] | 환경, 데이터 구조, 전처리, Dataset 코드 | 🆕 |
| → [[chapters/CH2_GSLRM_CODE_FLOW]] | Config 시스템, 학습 루프, 모델 Forward Pass, Loss | 🆕 |
| → [[chapters/CH3_EXPERIMENTS_AND_RESULTS]] | 전체 실험 흐름 (Phase 1→6), 코드, 명령어, 결과 | 🆕 |
| **[[PIPELINE_DEEP_DIVE]]** | **mouse_extensions 코드 워크스루 Hub (MoC + Issues + QuickRef)** | **🆕 v3.0** |
| → [[pipeline/PH1_PREPROCESSING]] | Phase 1: Data Preprocessing (M5) | 🆕 |
| → [[pipeline/PH2_DATA_LOADING]] | Phase 2: Training Data Loading | 🆕 |
| → [[pipeline/PH3_MODEL_FORWARD_LOSS]] | Phase 3: Model Forward & Loss | 🆕 |
| → [[pipeline/PH4_POSE_CONDITIONING]] | Phase 4: Pose Conditioning (Plucker) | 🆕 |
| → [[pipeline/PH5_E2E_INFERENCE]] | Phase 5: E2E Inference Pipeline | 🆕 |
| → [[pipeline/PH6_FAIR_EVALUATION]] | Phase 6: Fair Evaluation | 🆕 |
| → [[pipeline/PH7_3D_KEYPOINT]] | Phase 7: 3D Keypoint Pipeline | 🆕 |
| [[MOUSE_DATASET_GUIDE]] | MouseViewDataset 구현 참조 | ✅ |
| ~~MVDIFFUSION_FINETUNE_GUIDE~~ | _(archived → Phase 1)_ | 📦 |
| ~~POSE_SPLATTER_GUIDE~~ | _(archived → Phase 1)_ | 📦 |

### hypotheses/ (연구 가설)

| 문서 | 핵심 질문 | 상태 |
|------|----------|:----:|
| [[H4_VIEW_ABLATION]] | 최적 입력 뷰 수 | ✅ 6-view 단조 증가 |
| [[H5_MVDIFFUSION]] | MVDiff 개선 방법 | ✅ Phase 3 완료 |
| [[H6_ALPHA_MASK]] | Alpha mask loss 효과 | **🔄 재평가 중** (novel view 관점, [[ALPHA_LOSS_NOVEL_VIEW_ANALYSIS]]) |
| [[H7_SSIM_WEIGHT]] | SSIM weight 최적값 | **❌ 기각** (0.5/1.0 collapse, 0.3 하락) |
| [[GENERALIZATION_ROADMAP]] | 카메라/피사체 일반화 | 📋 계획 |
| [[RMA_CAMERA_ANALYSIS]] | M5 카메라 비균일 배치 | 📊 분석 완료 |

> **Active hypotheses**: H_T1 (distribution mismatch) → DA1 실험 진행중
> See: [[experiments/mvdiffusion_bottleneck_analysis]] §5

### theory/

| 문서 | 내용 |
|------|------|
| **[[PIPELINE_ARCHITECTURE]]** | 2-stage 파이프라인 (MVDiff + GS-LRM) |
| **[[MAMMAL_MESH_RENDERING_PIPELINE]]** | MAMMAL mesh rendering 3단계: fitting → UV texture → per-frame render | **🆕 v1.0** |
| **[[MULTIVIEW_DIFFUSION_THEORY]]** | MVDiff/GS-LRM 이론 기반 |
| [[METRICS_PROTOCOL]] | 메트릭 프로토콜 (White-BG, FG-only) |
| [[MV_ADAPTER_TECHNICAL]] | MV-Adapter 아키텍처 분석 |
| [[SLIDES_FACELIFT_PIPELINE]] | 파이프라인 발표 슬라이드 (Marp) |

### mouse_extensions/docs/ (구현 상세)

| 문서 | 내용 | 상태 |
|------|------|:----:|
| [[../../mouse_extensions/docs/COORDINATE_SYSTEMS]] | 좌표계 변환 (MAMMAL↔FaceLift↔OpenGL) | ✅ |
| [[../../mouse_extensions/docs/DATASET_FRAME_INDEXING]] | 프레임 인덱싱 & 데이터 명세 | ✅ |
| [[../../mouse_extensions/docs/CAMERA_CALIBRATION]] | 카메라 캘리브레이션 상세 | ✅ |
| [[../../mouse_extensions/docs/UV_TEXTURE_RENDERING_BUG]] | UV 텍스처 렌더링 버그 분석 & 수정 | ✅ 해결 |
| [[../../mouse_extensions/docs/MESH_GUIDED_REFINEMENT]] | Mesh-guided artifact refinement 제안 | 📋 계획 |

### experiments/ — Mesh-GS Pair Collection

| 문서 | 내용 | 상태 |
|------|------|:----:|
| [[mesh_gs_pair_collection]] | Novel view dataset pipeline v2.0 (tier-based, metadata) | **✅ v2.0** |

### tools/ — Novel View Pipeline

| 도구 | 위치 | 용도 |
|------|------|------|
| **collect_dataset.py** | `mouse_extensions/scripts/novel_view/` | Tier-based dataset collection (migrate/generate/manifest/visualize) |

### tools/ (스크립트 & 레포트)

| 도구 | 위치 | 용도 |
|------|------|------|
| **Report system** | `mouse_extensions/scripts/report/` | YAML→HTML 비교 레포트 자동 생성 |
| **Fair eval** | `mouse_extensions/scripts/eval/fair_comparison.py` | Fair comparison metric 계산 |
| **Camera follow** | `mouse_extensions/scripts/render_camera_follow.py` | 13 camera targets + body stabilization |
| **Generated reports** | `reports/` | HTML 비교 레포트 |

### _archive/ (Phase 1 보존)

> 16개 Phase 1 문서가 `docs/_archive/`로 이동됨 (git history에도 보존).
> 분류: `phase1_experiments/` (10), `phase1_guides/` (4), `phase1_root/` (2)

---

## 핵심 결과 요약 (2026-03-03 기준)

### FL vs PS Fair Comparison (M5 Same-Camera, Fair Eval)

| Metric | FL GS-LRM 6v (GT) | FL E2E Best | PS 6-cam |
|--------|:-----------------:|:-----------:|:--------:|
| PSNR_gt | **23.84** | 9.04 | 13.78 |
| IoU | **0.954** | 0.577 | 0.846 |
| Coverage | ~1.0 | ~0.75 | 0.919 |

> FL GS-LRM >> PS by +10.06 dB. E2E bottleneck = Stage 1 MVDiffusion (86% of gap).
> **All E2E strategies converge**: PSNR_gt 7.75–9.04 dB → architecture change needed.
> See: [[experiments/FL_vs_PS_comparison]] + [[experiments/mvdiffusion_bottleneck_analysis]] + [[E2E_EXPERIMENT_ANALYSIS]]

### Active Experiments (2026-03-03)

| GPU | Experiment | Purpose | ETA |
|:---:|-----------|---------|:---:|
| gpu03:5 | H4b@20K E2E (재실행) | 진짜 E2E 평가 (--input_view_idx 0) | ~6h |
| gpu03:6 | H7v2 Spatial Token (재학습) | Pose injector save fix 반영 | ~35h |

> **Phase 2 Priority** (현재 포커스): P0: H7v2 E2E eval, P1: Novel view rendering, P2: Body-part cam follow, P3: Multi-species (Rat7M), P4: Temporal consistency
> **Phase 1 Backlog**: Silhouette loss, Domain Adaptation (명시적 요청 시)
> See: [[experiments/PHASE2_NOVEL_VIEW_ROADMAP]] for Phase 2 plan, [[NEURIPS_GAP_ANALYSIS]] for Phase 1 gap analysis

### Completed Hypotheses

| Hypothesis | Result | Reference |
|-----------|--------|-----------|
| H4: View Ablation | ✅ 6-view monotonic | [[hypotheses/H4_VIEW_ABLATION]] |
| H5: MVDiffusion | ✅ Phase 3 done | [[hypotheses/H5_MVDIFFUSION]] |
| H6: Alpha Mask | ❌ Rejected | [[hypotheses/H6_ALPHA_MASK]] |
| H7: SSIM Weight | ❌ Rejected | [[hypotheses/H7_SSIM_WEIGHT]] |
| HP: Preprocessing | ✅ M0 diverges, M5_4 baseline match | [[experiments/hypothesis_roadmap]] |
| **H_T1: Dist. Mismatch** | **🔬 Testing (DA1)** | [[experiments/mvdiffusion_bottleneck_analysis]] §5 |

---

## 실험 결과 JSON 위치

```
experiments/comparison/
├── tier/                          # All fair eval JSONs (17)
│   ├── gslrm_{1-6}view_fair.json  # View ablation
│   ├── {e1,e2,e3}_*_fair.json     # MVDiff variants
│   └── p1_*_fair.json             # E2E pipeline
├── fair/                          # Legacy fair comparison
├── FL_vs_PS/                      # Cross-method comparison
└── 9exp_unified_metrics.json      # Unified 9-experiment metrics
```

---

## Obsidian 교차 참조

> 로컬 Obsidian vault에서 이론/분석/논문 초안을 관리합니다.
> **SSOT 규칙**: 실험 수치/설정은 본 서버 docs/ 우선. Obsidian은 학습/참조/논문 작성용.

| Obsidian 문서 | 내용 |
|--------------|------|
| `docs/INDEX.md` (v5.0) | Obsidian 전체 문서 인덱스 |
| `docs/paper/` | 논문 초안 (BehaviorSplatter), NeurIPS Gap 분석 |
| `docs/theory/` | 파이프라인 이론 (Stage1/2, 좌표계, Loss) |
| `_Notes/` | 연구 노트 (세션별, `docs/notes/`에서 통합 이전) |

---

## 용어 참고 (Terminology)

| 용어 | 의미 |
|------|------|
| **Multi-view Diffusion / Stage 1** | FaceLift의 1장→6장 생성 모델. SD2.1-UnCLIP + Era3D RMA 기반 |
| **MVDiff** (약칭) | "multi-view diffusion"의 줄임말. Tang et al. "MVDiffusion" 논문과 무관 |
| `mvdiffusion/` (코드 폴더) | Era3D 코드베이스에서 상속된 명칭. 변경 비용이 높아 유지 |
| **GS-LRM / Stage 3** | Transformer 기반 3D Gaussian 예측 모델 |

---

*MoC v14.0 | Updated: 2026-03-16 | PIPELINE_DEEP_DIVE entries added from server*
