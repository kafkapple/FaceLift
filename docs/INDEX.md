# FaceLift Mouse Documentation

> **MoC (Map of Content)** — Central hub for all project documents.
> **Updated**: 2026-03-12 | **Version**: v12.0
> **Active**: 48개 | **Archive**: 삭제됨 (현행 문서에 통합)

---

## Quick Links

| 목적 | 문서 |
|------|------|
| **실험 마스터 가이드 (START HERE)** | [[guides/EXPERIMENT_MASTER_GUIDE]] **v1.0** |
| **FL vs PS 비교 (SSOT)** | [[experiments/FL_vs_PS_comparison]] **v11** |
| **Stage 1 병목 분석** | [[experiments/mvdiffusion_bottleneck_analysis]] **v1.1** |
| **Domain Adaptation** | [[experiments/domain_adaptation_DA1]] **v1.1** |
| **평가 프로토콜** | [[experiments/evaluation_protocol_v1]] **v1.0** |
| **가설 SSOT (로드맵)** | [[experiments/hypothesis_roadmap]] **v3.0** |
| **명령어 SSOT** | [[experiments/COMMANDS]] |
| **레포트 시스템 가이드** | [[experiments/REPORT_SYSTEM_GUIDE]] **v1.0** |
| **Stage 1 대체 후보** | [[experiments/STAGE1_REPLACEMENT_CANDIDATES]] **v1.0** |
| **파이프라인 아키텍처** | [[theory/PIPELINE_ARCHITECTURE]] |
| **📄 논문 초안** | [[PAPER_DRAFT_BehaviorSplatter]] **🆕 v1.0** |
| **📊 1페이지 요약** | [[EXECUTIVE_SUMMARY]] **🆕 v1.0** |
| **🎯 NeurIPS Gap 분석** | [[NEURIPS_GAP_ANALYSIS]] **🆕 v1.0** |
| **🔍 프로젝트 종합** | [[PROJECT_SYNTHESIS]] **🆕 v1.0** |
| **📋 문서 감사** | [[DOCUMENT_AUDIT_260303]] **🆕 v1.0** |

---

## 문서 구조

### experiments/ (실험 & 비교)

#### Core Documents (최신, 빈번 참조)

| 문서 | 내용 | 상태 | 관련 |
|------|------|:----:|------|
| **[[FL_vs_PS_comparison]]** | FL vs PS 통합 비교 (9-exp) | **✅ v11** | → eval_protocol, bottleneck |
| **[[mvdiffusion_bottleneck_analysis]]** | Stage 1 전송률(14%) + 3가설(H_T1/T2/T3) + 전체 실험 현황 | **🆕 v1.1** | → DA1, hypothesis_roadmap |
| **[[domain_adaptation_DA1]]** | GS-LRM domain adaptation (H_T1 검증) | **🆕 v1.1** | → bottleneck, training_settings |
| **[[evaluation_protocol_v1]]** | 9-exp NVS 평가 프로토콜 (Temporal/Spatial/Combined) | **🆕 v1.0** | → FL_vs_PS, bottleneck |
| **[[hypothesis_roadmap]]** | 가설 검정 결과 + 우선순위 | ✅ v2.1 | → H4-H7, bottleneck |
| **[[comprehensive_analysis_report]]** | **종합 실험 보고서** (H1-H8 + Phase 2-3 전체 정량 비교) | **🆕 v1.0** | → 모든 실험 문서 |

#### Reference Documents (안정, 참조용)

| 문서 | 내용 | 상태 |
|------|------|:----:|
| ~~training_optimal_settings~~ | _(삭제됨 — bottleneck에 통합)_ | 📦 |
| ~~mvdiff_improvement_roadmap~~ | _(삭제됨 — bottleneck에 통합)_ | 📦 |
| ~~FL_PS_metric_consistency~~ | _(삭제됨 — evaluation_protocol에 통합)_ | 📦 |
| ~~260216_PHASE3_REPORT~~ | _(삭제됨 — comprehensive_analysis_report로 대체)_ | 📦 |
| ~~260216_MVDIFF_TRAINING_ANALYSIS~~ | _(삭제됨 — comprehensive_analysis_report로 대체)_ | 📦 |
| **[[REPORT_SYSTEM_GUIDE]]** | HTML 레포트 시스템 통합 가이드 (코드+메트릭+프로토콜) | **🆕 v1.0** |
| **[[STAGE1_REPLACEMENT_CANDIDATES]]** | Stage 1 대체 모델 후보 연구 (10개 모델 분석) | **🆕 v1.0** |

#### Operational Guides

| 문서 | 내용 | 상태 |
|------|------|:----:|
| [[COMMANDS]] | 명령어 SSOT (GS-LRM, MVDiff, Turntable) | ✅ |
| [[EXPERIMENT_CONFIG_GUIDE]] | Config 시스템 (Modular 3-Layer) | ✅ |
| [[evaluation_protocol_v1]] | 평가 유형 / Split 전략 | ✅ |
| [[EXPERIMENT_REGISTRY]] | 실험 설정/결과 기록 | ✅ |
| [[INFERENCE_E2E_GUIDE]] | E2E 추론 파이프라인 + CLI | ✅ |
| [[TRAINING_LOGGING_GUIDE]] | 학습 단위/WandB 로깅 | ✅ |
| [[VISUALIZATION_SETTINGS]] | Turntable/시각화 설정 | ✅ |
| **[[KEYPOINT_3D_PIPELINE]]** | 3D Keypoint 삼각측량 + FL vs PS 비교 | **🆕 v1.0** |

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
| [[MVDIFFUSION_FINETUNE_GUIDE]] | Multi-view diffusion (Stage 1) fine-tuning | ✅ |
| [[MOUSE_DATASET_GUIDE]] | MouseViewDataset 구현 참조 | ✅ |
| [[POSE_SPLATTER_GUIDE]] | Pose Splatter 비교 프로토콜 | ✅ |

### hypotheses/ (연구 가설)

| 문서 | 핵심 질문 | 상태 |
|------|----------|:----:|
| [[H4_VIEW_ABLATION]] | 최적 입력 뷰 수 | ✅ 6-view 단조 증가 |
| [[H5_MVDIFFUSION]] | MVDiff 개선 방법 | ✅ Phase 3 완료 |
| [[H6_ALPHA_MASK]] | Alpha mask loss 효과 | **❌ 기각** (0.3/0.5/1.0 모두 baseline 이하) |
| [[H7_SSIM_WEIGHT]] | SSIM weight 최적값 | **❌ 기각** (0.5/1.0 collapse, 0.3 하락) |
| [[GENERALIZATION_ROADMAP]] | 카메라/피사체 일반화 | 📋 계획 |
| [[RMA_CAMERA_ANALYSIS]] | M5 카메라 비균일 배치 | 📊 분석 완료 |

> **Active hypotheses**: H_T1 (distribution mismatch) → DA1 실험 진행중
> See: [[experiments/mvdiffusion_bottleneck_analysis]] §5

### theory/

| 문서 | 내용 |
|------|------|
| **[[PIPELINE_ARCHITECTURE]]** | 2-stage 파이프라인 (MVDiff + GS-LRM) |
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
| [[mesh_gs_pair_collection]] | MAMMAL mesh + GS-LRM pair 수집 전략 | ✅ |

### tools/ (스크립트 & 레포트)

| 도구 | 위치 | 용도 |
|------|------|------|
| **Report system** | `mouse_extensions/scripts/report/` | YAML→HTML 비교 레포트 자동 생성 |
| **Fair eval** | `mouse_extensions/scripts/eval/fair_comparison.py` | Fair comparison metric 계산 |
| **DA datagen** | `mouse_extensions/scripts/domain_adapt/` | MVDiff→GS-LRM dataset 생성 |
| **Generated reports** | `reports/` | HTML 비교 레포트 |

### _archive/ (삭제됨)

> ⚠️ _archive/ 디렉토리는 삭제됨. 이전 문서들은 현행 문서에 통합되었거나 폐기됨.
> 주요 통합: comprehensive_analysis_report (←PHASE3_REPORT), evaluation_protocol_v1 (←metric_consistency)

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

> **NeurIPS Top 5**: P1: Silhouette loss, P2: Domain Adaptation, P3: H7v2 완성, P4: Multi-species, P5: Behavior validation
> See: [[NEURIPS_GAP_ANALYSIS]] for detailed plan

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

## 용어 참고 (Terminology)

| 용어 | 의미 |
|------|------|
| **Multi-view Diffusion / Stage 1** | FaceLift의 1장→6장 생성 모델. SD2.1-UnCLIP + Era3D RMA 기반 |
| **MVDiff** (약칭) | "multi-view diffusion"의 줄임말. Tang et al. "MVDiffusion" 논문과 무관 |
| `mvdiffusion/` (코드 폴더) | Era3D 코드베이스에서 상속된 명칭. 변경 비용이 높아 유지 |
| **GS-LRM / Stage 3** | Transformer 기반 3D Gaussian 예측 모델 |

---

*MoC v12.0 | Updated: 2026-03-12 | comprehensive_analysis_report 추가, _archive 정리, mouse_extensions/docs 연결, 깨진 링크 수정*
