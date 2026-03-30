# FaceLift Mouse Documentation

> **MoC (Map of Content)** — Central hub for all project documents.
> **Updated**: 2026-03-24 | **Version**: v16.3
> **Active**: 52개 | **Archive**: 19개 (docs/_archive/, git history에 보존)
>
> **Phase Structure**: Phase 1 (MVDiff bottleneck) = 유지보수 | **Phase 2 (Novel View + Multi-Species) = 현재 포커스**
>
> **Obsidian (이론/논문/NeurIPS 전략)**: `~/Documents/Obsidian/30_Projects/_CODES/FaceLift/docs/`
> - `neurips/`: 논문 초안, Gap 분석, Executive Summary, 프로젝트 종합
> - `theory/`: 좌표계, Stage1/2, Loss 수식
> - `_Notes/`: 24개 연구 노트 (260114~260317)

---

## Quick Links

| 목적 | 문서 |
|------|------|
| **Phase 2 로드맵 (START HERE)** | [[experiments/PHASE2_NOVEL_VIEW_ROADMAP]] **v2.0** |
| **FL vs PS 비교 (SSOT)** | [[experiments/fl_vs_ps_comparison]] **v11** |
| **가설 SSOT (로드맵)** | [[experiments/hypothesis_roadmap]] **v3.0** |
| **평가 프로토콜** | [[experiments/evaluation_protocol_v1]] **v1.0** |
| **명령어 SSOT** | [[experiments/COMMANDS]] |
| ~~DiFix 학습 전략~~ | ~~[[experiments/DIFIX_TRAINING_STRATEGY]]~~ **[DEPRECATED — PoC 실패 260322, mode collapse]** |
| **Master Results Table** | [[experiments/MASTER_RESULTS_TABLE]] **v1.0** |
| **통합 Ablation 보고서** | [[experiments/UNIFIED_ABLATION_REPORT]] **v1.0** |
| **Alpha Loss 분석** | [[experiments/ALPHA_LOSS_NOVEL_VIEW_ANALYSIS]] **v2.0** |
| **파이프라인 아키텍처** | [[specs/PIPELINE_ARCHITECTURE]] |
| **Mesh Rendering** | [[specs/MAMMAL_MESH_RENDERING_PIPELINE]] **v1.0** |
| **카메라 컨벤션 (SSOT)** | Obsidian `theory/COORDINATE_SYSTEMS.md` → "Camera Conventions" 섹션 |

---

## Phase Structure (2026-03-12~)

| Phase | Focus | Status | Key Doc |
|:-----:|-------|:------:|---------|
| **Phase 1** | MVDiff bottleneck (Sil loss, DA, Spatial Token) | 🔧 유지보수 | [[experiments/hypothesis_roadmap]] |
| **Phase 2** ⭐ | Novel View + Multi-Species + NeurIPS Dataset Track | **🔬 ACTIVE** | [[experiments/PHASE2_NOVEL_VIEW_ROADMAP]] |

> 기본 포커스는 Phase 2. Phase 1 작업은 명시적 요청 시에만.

---

## 문서 구조

### experiments/ (실험 & 비교)

#### Core Documents (최신, 빈번 참조)

| 문서 | 내용 | 상태 | 관련 |
|------|------|:----:|------|
| **[[experiments/PHASE2_NOVEL_VIEW_ROADMAP]]** | **Phase 2 로드맵**: Novel View + Multi-Species + NeurIPS Dataset Track | **✅ v2.0** | → mesh_gs_pair, KEYPOINT_3D |
| ~~[[experiments/DIFIX_TRAINING_STRATEGY]]~~ | ~~DiFix 3D+ 학습 전략~~: **[DEPRECATED — PoC 실패 260322 S25, mode collapse]** | ~~v1.0~~ | — |
| **[[experiments/DIFIX_TYPE2_MAMMAL_PLAN]]** | DiFix Type 2 재시도: MAMMAL mesh bottom view pair + 학습 (MoA audit 승인) | **v1.0** | → DIFIX_TRAINING_STRATEGY |
| **[[experiments/DEFORM_V3_FG_AWARE_PLAN]]** | ⭐ **Deform V3**: FG-aware rendering loss, V2 중단 후 전환 (4× MoA Audit) | **v1.0** | → DEFORMATION_4DGS, TEMPORAL |
| **[[experiments/DEFORMATION_4DGS_EXPERIMENT_PLAN]]** | 4D-GS Deformation 실험 계획: Phase 0-3, 논문 기준 학습 | **v1.0** | → TEMPORAL_CONSISTENCY_STUDY |
| **[[experiments/TEMPORAL_CONSISTENCY_STUDY]]** | Temporal flickering 분석 + smoothing 방법 비교 (EMA/OptFlow/DeformV2) | **✅ v1.0** | → TEMPORAL_EVAL_STANDARD |
| **[[experiments/TEMPORAL_EVAL_STANDARD]]** | Temporal 평가 기준 SSOT (프레임/뷰/메트릭) | **✅ v1.0** | → TEMPORAL_CONSISTENCY_STUDY |
| **[[experiments/ALPHA_LOSS_NOVEL_VIEW_ANALYSIS]]** | Alpha loss novel view artifact 억제 효과 분석 | **✅ v1.0** | → [[hypotheses/H6_ALPHA_MASK]], PHASE2 |
| **[[experiments/ALPHA_COMPREHENSIVE_EVAL_260326]]** | ⭐ **6v Alpha 종합 평가**: PSNR_gt/int, IoU, Sil.Prec, per-cam, Gaussian stats | **✅ 완료** | → METRICS_PROTOCOL §7 |
| **[[experiments/CHECKPOINT_INVENTORY_260326]]** | 체크포인트 현황 + P0-P3 실행 계획 (17 ckpts) | **✅ P0 완료** | → H8, Pruning Design |
| **[[experiments/fl_vs_ps_comparison]]** | FL vs PS 통합 비교 (9-exp) | **✅ v12** | → eval_protocol, GAUSSIAN_COUNT |
| **[[experiments/GAUSSIAN_COUNT_FAIR_COMPARISON]]** | Gaussian 수 매칭 공정 비교 실험 설계 | 📋 Design | → fl_vs_ps, METRICS_PROTOCOL |
| **[[experiments/hypothesis_roadmap]]** | 가설 검정 결과 + 우선순위 | ✅ v3.0 | → H4-H7 |
| **[[experiments/UNIFIED_ABLATION_REPORT]]** | **통합 Ablation 보고서**: View/Alpha/E2E/Resolution 일관 비교 | **✅ v1.0** | → 모든 ablation |
| **[[experiments/comprehensive_analysis_report]]** | 종합 실험 보고서 (H1-H8) | **✅ v1.0** | → 모든 실험 |
| **[[experiments/evaluation_protocol_v1]]** | NVS 평가 프로토콜 (Fair Eval) | **✅ v1.0** | → FL_vs_PS |
| **[[experiments/mesh_gs_pair_collection]]** | Novel view dataset pipeline v2.0 | **✅ v2.0** | → PHASE2, [[experiments/DATASET_QA_VIEWER]] |
| **[[experiments/DATASET_QA_VIEWER]]** | Novel view dataset QA viewer (stdlib HTTP, exclude 관리) | **✅ v1.0** | → mesh_gs_pair |
| **[[experiments/REPORT_SYSTEM_GUIDE]]** | HTML 레포트 시스템 통합 가이드 (코드+메트릭+프로토콜) | **✅ v1.0** | |
| **[[experiments/PAST_SESSION_AUDIT_260322]]** | 과거 7개 항목 감사 (HLAC/DiFix/canonical set/VAME 등) | **✅ v2.0** (bias-corrected) | → PHASE2, hypothesis_roadmap |
| **[[experiments/HLAC_COMPREHENSIVE_260322]]** | HLAC 종합 분석: 2992 frames, K=8, Gaussian vs KP feature comparison | **✅ v1.0** | → BehaviorSplatter, PAST_SESSION_AUDIT |

#### Reference Documents (안정, 삭제/통합된 문서)

| 문서 | 내용 | 상태 |
|------|------|:----:|
| ~~training_optimal_settings~~ | _(삭제됨 — bottleneck에 통합)_ | 📦 |
| ~~mvdiff_improvement_roadmap~~ | _(삭제됨 — bottleneck에 통합)_ | 📦 |
| ~~FL_PS_metric_consistency~~ | _(삭제됨 — evaluation_protocol에 통합)_ | 📦 |
| ~~260216_PHASE3_REPORT~~ | _(삭제됨 — comprehensive_analysis_report로 대체)_ | 📦 |
| ~~260216_MVDIFF_TRAINING_ANALYSIS~~ | _(삭제됨 — comprehensive_analysis_report로 대체)_ | 📦 |

| **[[experiments/NOVEL_VIEW_QUALITY_STRATEGY]]** | Novel view 품질 개선 전략 (3-model deliberation 기반) | **✅ v1.0** | → PHASE2, ALPHA_LOSS |
| **[[experiments/RELATED_WORK_SURVEY]]** | NeurIPS 2026 Dataset Track 관련 논문 조사 (2026-03-19) | **✅ v1.0** | → PHASE2, gap analysis |

#### Operational Guides

| 문서 | 내용 | 상태 |
|------|------|:----:|
| [[experiments/COMMANDS]] | 명령어 SSOT (GS-LRM, MVDiff, Turntable) | ✅ |
| [[experiments/EXPERIMENT_CONFIG_GUIDE]] | Config 시스템 (Modular 3-Layer) | ✅ |
| [[experiments/EXPERIMENT_REGISTRY]] | 실험 설정/결과 기록 | ✅ |
| [[experiments/INFERENCE_E2E_GUIDE]] | E2E 추론 파이프라인 + CLI | ✅ |
| **[[KEYPOINT_3D_PIPELINE]]** | 3D Keypoint 삼각측량 + FL vs PS 비교 | **✅ v1.0** |

### datasets/ (데이터셋 & 전처리)

| 문서 | 내용 | 상태 |
|------|------|:----:|
| **[[datasets/PREPROCESSING_REGISTRY]]** | 전처리 SSOT (M/D-series 전체) | ✅ SSOT |
| [[datasets/M5_SERIES_SPEC]] | M5 카메라 정규화 상세 (**← camera mismatch 핵심 참조**) | ✅ |
| [[datasets/RAW_DATA]] | Raw 데이터 출처 (DANNCE→MAMMAL→M5t2 버전별 상세 명세) | ✅ v2.0 |
| **[[datasets/MULTI_ANIMAL_PREPROCESSING]]** | 다중 동물 (s-DANNCE) 전처리 스펙: 마스크 전략, Plucker ray, 파이프라인 | **🆕 v1.0** |
| **[[datasets/SDANNCE_VIDEO_AVAILABILITY]]** | **s-DANNCE 데이터셋 종합 가이드 (SSOT)**: Harvard Dataverse 17개 전수조사, SCN2A_WK1 lone rat, 카메라 특성, 다운로드 명령어 | **✅ v2.0** |
| **[[datasets/PREPROCESSING_COMPARISON]]** | M5t2 vs RAT2 전처리 정량 비교 (FG coverage, intrinsics, split) | **✅ v1.0** |
| _(미생성)_ `datasets/SDANNCE_PREPROCESSING.md` | s-DANNCE→GS-LRM 전처리 통합 스펙 — **계획됨**, 미작성 | ⏳ 작성 필요 |
| **RAT2 Dataset Config** | `configs/datasets/RAT2.yaml` — 2-phase HLAC stratified split (2371 train / 297 val / 299 test), commit `8f36c88` | **✅ 260324** |

### guides/ (입문 & 실습)

| 문서 | 내용 | 상태 |
|------|------|:----:|
| **[[guides/EXPERIMENT_MASTER_GUIDE]]** | **전체 실험 마스터 가이드 (초심자 시작점)** | **✅ v1.0** |
| **[[guides/RESEARCH_EXPERIMENT_NOTES]]** | 연구 노트 (Lab Meeting용, 가설→실험→결과) | **✅ v1.0** |
| [[guides/MOUSE_DATASET_GUIDE]] | MouseViewDataset 구현 참조 | ✅ |
| **[[guides/PIPELINE_DEEP_DIVE]]** | mouse_extensions 코드 워크스루 Hub (MoC + Issues + QuickRef) | **✅ v3.0** |
| **[[guides/VISUALIZATION_GUIDE]]** | 시각화 모듈 가이드 (cinematic, 6-view grid, N-filter, 배치) | **✅ v1.0** |
| [[guides/RAT_FT_DATA_GUIDE]] | Rat FT 데이터 준비 가이드 | ✅ |
| **[[guides/RAT_SAM2_MASK_STRATEGY]]** | Rat SAM2 마스크 전략 (SCN2A_WK1 + M3_M4, 자동/수동 하이브리드) | **✅ v1.0** |
| **[[guides/RAT_SAM2_QUICKSTART]]** | Rat SAM2 뷰어 실행 Quick Start (annotator + propagation 명령어) | **✅ v1.0** |
| [[guides/MAMMAL_REFIT_HANDOFF]] | MAMMAL 23 bad frames 재피팅 핸드오프 (환경, 프레임 매핑, 검증) | **✅ v1.0** |

#### guides/chapters/ (EXPERIMENT_MASTER_GUIDE 하위)

| 문서 | 내용 |
|------|------|
| → [[guides/chapters/CH1_ENVIRONMENT_AND_DATA]] | CH1: 환경, 데이터 구조, 전처리, Dataset 코드 |
| → [[guides/chapters/CH2_GSLRM_CODE_FLOW]] | CH2: Config 시스템, 학습 루프, 모델 Forward Pass, Loss |
| → [[guides/chapters/CH3_EXPERIMENTS_AND_RESULTS]] | CH3: 전체 실험 흐름 (Phase 1→6), 코드, 명령어, 결과 |

#### guides/pipeline/ (PIPELINE_DEEP_DIVE 하위)

| 문서 | 내용 |
|------|------|
| → [[guides/pipeline/PH1_PREPROCESSING]] | Phase 1: Data Preprocessing (M5) |
| → [[guides/pipeline/PH2_DATA_LOADING]] | Phase 2: Training Data Loading |
| → [[guides/pipeline/PH3_MODEL_FORWARD_LOSS]] | Phase 3: Model Forward & Loss |
| → [[guides/pipeline/PH4_POSE_CONDITIONING]] | Phase 4: Pose Conditioning (Plucker) |
| → [[guides/pipeline/PH5_E2E_INFERENCE]] | Phase 5: E2E Inference Pipeline |
| → [[guides/pipeline/PH6_FAIR_EVALUATION]] | Phase 6: Fair Evaluation |
| → [[guides/pipeline/PH7_3D_KEYPOINT]] | Phase 7: 3D Keypoint Pipeline |

### hypotheses/ (연구 가설)

| 문서 | 핵심 질문 | 상태 |
|------|----------|:----:|
| [[hypotheses/H4_VIEW_ABLATION]] | 최적 입력 뷰 수 | ✅ 6-view 단조 증가 |
| [[hypotheses/H5_MVDIFFUSION]] | MVDiff 개선 방법 | ✅ Phase 3 완료 |
| [[hypotheses/H6_ALPHA_MASK]] | Alpha mask loss 효과 | **✅ 완료 (α=0.3@6v best trade-off, IoU↑ +0.002, PSNR -0.55 dB)** → [[experiments/ALPHA_LOSS_NOVEL_VIEW_ANALYSIS]] |
| [[hypotheses/H7_SSIM_WEIGHT]] | SSIM weight 최적값 | **❌ 기각** (0.5/1.0 collapse, 0.3 하락) |
| [[hypotheses/GENERALIZATION_ROADMAP]] | 카메라/피사체 일반화 | 📋 계획 |
| [[hypotheses/RMA_CAMERA_ANALYSIS]] | M5 카메라 비균일 배치 | 📊 분석 완료 |

> **Active hypotheses**: H_T1 (distribution mismatch) → DA1 실험 진행중
> See: [[_archive/phase1_experiments/mvdiff_bottleneck_analysis]] §5

### specs/ (기술 사양)

| 문서 | 내용 |
|------|------|
| **[[specs/PIPELINE_ARCHITECTURE]]** | 2-stage 파이프라인 사양 (MVDiff + GS-LRM I/O, config) |
| **[[specs/MAMMAL_MESH_RENDERING_PIPELINE]]** | MAMMAL mesh rendering 3단계 사양 |
| [[specs/METRICS_PROTOCOL]] | 평가 메트릭 프로토콜 (White-BG, FG-only) |
| **[[specs/NEURAL_TEXTURE_ANALYSIS]]** | MoReMouse 기반 neural texture 분석 + 3-model deliberation + 구현 전략 | **🆕 v1.0** |
| **[[specs/NEURAL_TEXTURE_RENDERING_DIAGNOSIS]]** | Neural texture 렌더 품질 진단: mesh fitting vs UV encoding (MAMMAL 팀 핸드오프) | **🆕 v1.0** |
| **[[specs/FRAME_SELECTION_LITERATURE_REVIEW]]** | Pose-diversity frame selection 문헌 조사: FPS, k-means, FisherRF, coreset, 3-stage hybrid 제안 | **🆕 v1.0** |
| **[[specs/KEYPOINT_ABLATION_FRAMEWORK]]** | 5-tier keypoint ablation 프레임워크: tier별 문헌 근거 + task matrix + YAML SSOT | **🆕 v1.0** |
| **[[specs/PATHS_SSOT_DESIGN]]** | outputs/ v2 경로 설계 + paths.py factory | ✅ |
| **[[specs/RAT_PREPROCESSING_STRATEGY]]** | RAT zero-pad vs crop vs hybrid 전처리 비교 + 3-model audit + 실험 계획 | **🆕 v1.0** |

> **이론 문서 → Obsidian으로 이동** (2026-03-17):
> MULTIVIEW_DIFFUSION_THEORY → `Obsidian/docs/theory/`, MV_ADAPTER_TECHNICAL → `Obsidian/docs/research/`, SLIDES → `Obsidian/Presentation/`
> **4D-GS 전환 전략 분석** → `Obsidian/docs/research/4DGS_PIVOT_STRATEGIC_ANALYSIS.md` (260323, 3-model audit)

### outputs/reports/ (실험 보고서)

| 문서 | 내용 | 상태 |
|------|------|:----:|
| **[[outputs/reports/260321_behaviorsplatter_comprehensive]]** | **BehaviorSplatter 종합 보고서**: N>=2 filter, opacity 분석, feature 필터링 부재 발견, 실험 계획, s-DANNCE 현황 | **✅ SSOT** |
| — | **Behavior Metrics**: `mouse_extensions/behavior/metrics.py` — Silhouette, TPI, Entropy Rate, Bout stats, Short Bout Ratio, Autocorrelation, Cohen's d, McNemar, KS test 등 20+ metrics. Obsidian [[260317_Core_Hypothesis_Dense_Temporal_Stability]] §4 참조 | **✅ v2.0** |
| [[outputs/reports/260320_gaussian_bodypart_analysis]] | Gaussian body-part 분포 분석 + bone-distance 한계 | ✅ |

> **🔴 Critical**: 이전 covariance feature 실험에 GT mask 필터링 미적용 → 재검증 필수. 상세: [[outputs/reports/260321_behaviorsplatter_comprehensive]] §4

### mouse_extensions/docs/ (구현 상세)

| 문서 | 내용 | 상태 |
|------|------|:----:|
| [[mouse_extensions/docs/COORDINATE_SYSTEMS]] | 좌표계 변환 (MAMMAL↔FaceLift↔OpenGL) | ✅ |
| [[mouse_extensions/docs/DATASET_FRAME_INDEXING]] | 프레임 인덱싱 & 데이터 명세 (step=5 rule) | ✅ |
| [[mouse_extensions/docs/CAMERA_CALIBRATION]] | 카메라 캘리브레이션 & 렌더링 파이프라인 | ✅ |
| [[mouse_extensions/docs/UV_TEXTURE_RENDERING_BUG]] | UV 텍스처 렌더링 버그 분석 & 수정 | ✅ 해결 |
| [[mouse_extensions/docs/MESH_GUIDED_REFINEMENT]] | Mesh-guided artifact refinement 제안 | 📋 계획 |

### docs/ root (특수 문서)

| 문서 | 내용 | 상태 |
|------|------|:----:|
| **[[KEYPOINT_3D_PIPELINE]]** | 3D Keypoint 삼각측량 + FL vs PS 비교 | **✅ v1.0** |
| [[outputs_inventory_README]] | outputs/ 디렉토리 전체 인벤토리 | ✅ 참조 |
| [[outputs/STRUCTURE]] | outputs/ v2 구조 문서 (디렉토리 맵, 레거시, 심링크, 사이즈) | **🆕 v1.0** |
| [[EXECUTIVE_SUMMARY]] | → Obsidian `neurips/executive_summary` (프로젝트 총괄 요약) | 🔗 Stub |
| [[PROJECT_SYNTHESIS]] | → Obsidian `neurips/project_synthesis` (프로젝트 종합) | 🔗 Stub |
| [[NEURIPS_GAP_ANALYSIS]] | → Obsidian `neurips/gap_analysis` (NeurIPS Gap 분석) | 🔗 Stub |

### tools/ (스크립트 & 레포트)

| 도구 | 위치 | 용도 |
|------|------|------|
| **collect_dataset.py** | `mouse_extensions/scripts/novel_view/` | Tier-based dataset collection |
| **Report system** | `mouse_extensions/scripts/report/` | YAML→HTML 비교 레포트 자동 생성 |
| **Fair eval** | `mouse_extensions/scripts/eval/fair_comparison.py` | Fair comparison metric 계산 |
| **Ablation comparison video** | `mouse_extensions/scripts/eval/ablation_comparison.py` | View(1-6v) / Alpha(α=0-1) 분리 그리드 영상 생성 (gpu03, `--type view\|alpha`) |
| **Ablation charts** | `mouse_extensions/scripts/eval/ablation_chart.py` | View + Alpha ablation 정량 차트 PNG (로컬 실행 가능) |
| **Camera follow** | `mouse_extensions/scripts/render_camera_follow.py` | 13 camera targets + body stabilization |
| **Gaussian distributions** | `mouse_extensions/behavior/analyze_gaussian_distributions.py` | Per-body-part parameter analysis + modality tests |
| **Radial filtering** | `mouse_extensions/behavior/view_projected_filtering.py --mode radial` | Radius sweep grid for body-part Gaussian filtering |
| **HLAC frame selection** | `mouse_extensions/scripts/select_hlac_frames.py` | HLAC-stratified frame selection for RAT2 annotation (commit `9f4036f`) |

### _archive/ (Phase 1 보존)

> 19개 Phase 1 문서가 `docs/_archive/`로 이동됨 (git history에도 보존).
> 분류: `phase1_experiments/` (10), `phase1_guides/` (4), `phase1_root/` (2), `meta/` (3)

---

## 핵심 결과 요약 (2026-03-24 updated)

### FL vs PS Fair Comparison (M5 Same-Camera, Fair Eval)

| Metric | FL GS-LRM 6v (GT) | FL E2E Best | PS 6-cam |
|--------|:-----------------:|:-----------:|:--------:|
| PSNR_gt | **23.84** | 9.04 | 13.78 |
| IoU | **0.954** | 0.577 | 0.846 |
| Coverage | ~1.0 | ~0.75 | 0.919 |

> FL GS-LRM >> PS by +10.06 dB. E2E bottleneck = Stage 1 MVDiffusion (86% of gap).
> See: [[experiments/fl_vs_ps_comparison]] + [[_archive/phase1_experiments/mvdiff_bottleneck_analysis]] + [[_archive/phase1_experiments/E2E_EXPERIMENT_ANALYSIS]]

### Completed Hypotheses

| Hypothesis | Result | Reference |
|-----------|--------|-----------|
| H4: View Ablation | ✅ 6-view monotonic | [[hypotheses/H4_VIEW_ABLATION]] |
| H5: MVDiffusion | ✅ Phase 3 done | [[hypotheses/H5_MVDIFFUSION]] |
| H6: Alpha Mask | ✅ 완료 (α=0.3@6v best trade-off) | [[hypotheses/H6_ALPHA_MASK]] |
| H7: SSIM Weight | ❌ Rejected | [[hypotheses/H7_SSIM_WEIGHT]] |
| **H8: Opacity & Anisotropy** | **✅ 완료 (unimodal, orientation filter 구현)** | [[hypotheses/H8_opacity_anisotropy_analysis]] |
| HP: Preprocessing | ✅ M0 diverges, M5_4 baseline match | [[experiments/hypothesis_roadmap]] |
| **H_T1: Dist. Mismatch** | **🔬 Testing (DA1)** | [[_archive/phase1_experiments/mvdiff_bottleneck_analysis]] §5 |

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

## Documentation Split Principle (문서 배치 원칙)

**서버 docs/ = "이걸 어떻게 실행하나?"** — 구현/실행에 직접 필요한 문서만.
**Obsidian = "이게 왜 이런 설계인가?"** — 이론/분석/전략/논문/연구 노트.

| 서버 (이 위치) | 역할 | Obsidian | 역할 |
|---------------|------|----------|------|
| `experiments/` | 실험 설정, 결과, 비교 | `neurips/` | 논문 초안, Gap 분석, 전략 |
| `specs/` | 기술 사양 (I/O, 프로토콜) | `theory/` | 수학, 이론, 원리, 모델 계보 |
| `guides/` | 실행 가이드, 코드 워크스루 | `research/` | 분석, 비교, 대안 기술 |
| `datasets/` | 전처리 레지스트리, 데이터 명세 | `_Notes/` | 날짜 기반 연구 일지 |
| `hypotheses/` | 가설 검증 결과 (수치 테이블) | `Presentation/` | 발표 슬라이드 |

**판단 기준**: "이 문서 없이 실험을 실행할 수 있는가?" Yes → Obsidian. No → 서버.

---

## 용어 참고 (Terminology)

| 용어 | 의미 |
|------|------|
| **Multi-view Diffusion / Stage 1** | FaceLift의 1장→6장 생성 모델. SD2.1-UnCLIP + Era3D RMA 기반 |
| **MVDiff** (약칭) | "multi-view diffusion"의 줄임말. Tang et al. "MVDiffusion" 논문과 무관 |
| `mvdiffusion/` (코드 폴더) | Era3D 코드베이스에서 상속된 명칭. 변경 비용이 높아 유지 |
| **GS-LRM / Stage 3** | Transformer 기반 3D Gaussian 예측 모델 |

---

*MoC v16.3 | Updated: 2026-03-24 | S43+: RAT2 config, HLAC frame selection, H6 완료, DiFix deprecated*
