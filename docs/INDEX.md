# FaceLift Mouse Documentation

> **MoC (Map of Content)** — Central hub for all project documents.
> **Updated**: 2026-02-23 | **Version**: v10.1
> **Active**: 38개 | **Archive**: 22개

---

## Quick Links

| 목적 | 문서 |
|------|------|
| **실험 마스터 가이드 (START HERE)** | [[guides/EXPERIMENT_MASTER_GUIDE]] **v1.0** |
| **FL vs PS 비교 (SSOT)** | [[experiments/FL_vs_PS_comparison]] **v11** |
| **Stage 1 병목 분석** | [[experiments/mvdiffusion_bottleneck_analysis]] **v1.1** |
| **Domain Adaptation** | [[experiments/domain_adaptation_DA1]] **v1.1** |
| **평가 프로토콜** | [[experiments/evaluation_protocol_v1]] **v1.0** |
| **가설 로드맵** | [[experiments/hypothesis_roadmap]] |
| **명령어 SSOT** | [[experiments/COMMANDS]] |
| **레포트 시스템 가이드** | [[experiments/REPORT_SYSTEM_GUIDE]] **v1.0** |
| **Stage 1 대체 후보** | [[experiments/STAGE1_REPLACEMENT_CANDIDATES]] **v1.0** |
| **파이프라인 아키텍처** | [[theory/PIPELINE_ARCHITECTURE]] |

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

#### Reference Documents (안정, 참조용)

| 문서 | 내용 | 상태 |
|------|------|:----:|
| [[training_optimal_settings]] | Stage 1/2 최적 학습 설정 | ✅ |
| [[mvdiff_improvement_roadmap]] | MVDiff 아키텍처 개선 로드맵 | ✅ (→ bottleneck에서 요약) |
| [[FL_PS_metric_consistency]] | 메트릭/데이터 일관성 코드 레벨 검증 | ✅ v2.1 |
| [[_archive/260216_PHASE3_REPORT]] (archived) | Phase 3 종합 보고서 | ✅ |
| [[_archive/260216_MVDIFF_TRAINING_ANALYSIS]] (archived) | MVDiff LR 전략 분석 | ✅ |
| **[[REPORT_SYSTEM_GUIDE]]** | HTML 레포트 시스템 통합 가이드 (코드+메트릭+프로토콜) | **🆕 v1.0** |
| **[[STAGE1_REPLACEMENT_CANDIDATES]]** | Stage 1 대체 모델 후보 연구 (10개 모델 분석) | **🆕 v1.0** |

#### Operational Guides

| 문서 | 내용 | 상태 |
|------|------|:----:|
| [[COMMANDS]] | 명령어 SSOT (GS-LRM, MVDiff, Turntable) | ✅ |
| [[EXPERIMENT_CONFIG_GUIDE]] | Config 시스템 (Modular 3-Layer) | ✅ |
| [[EXPERIMENT_REGISTRY]] | 실험 설정/결과 기록 | ✅ |
| [[INFERENCE_E2E_GUIDE]] | E2E 추론 파이프라인 + CLI | ✅ |
| [[TRAINING_LOGGING_GUIDE]] | 학습 단위/WandB 로깅 | ✅ |
| [[VISUALIZATION_SETTINGS]] | Turntable/시각화 설정 | ✅ |

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
| **[[PIPELINE_DEEP_DIVE]]** | **mouse_extensions 코드 워크스루 (Phase 1-7, 라인번호 포함)** | **🆕 v2.0** |
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

### tools/ (스크립트 & 레포트)

| 도구 | 위치 | 용도 |
|------|------|------|
| **Report system** | `mouse_extensions/scripts/report/` | YAML→HTML 비교 레포트 자동 생성 |
| **Fair eval** | `mouse_extensions/scripts/eval/fair_comparison.py` | Fair comparison metric 계산 |
| **DA datagen** | `mouse_extensions/scripts/domain_adapt/` | MVDiff→GS-LRM dataset 생성 |
| **Generated reports** | `reports/` | HTML 비교 레포트 |

### _archive/ (22 files)

완료·대체·폐기된 문서. 역사적 참조용.

<details>
<summary>아카이브 목록 (펼치기)</summary>

| 문서 | 사유 |
|------|------|
| FL_vs_PS_comparison_v5/v6/v7 | v11으로 대체 |
| comparison_report | v11에 통합 |
| H1bis_v2_REVALIDATION | ✅ 완료 |
| H8_* (3 files) | ✅ 완료 (기각) |
| hp_preprocessing_ablation | experiments/ (실험 진행 중) |
| RTX3060_GUIDE | A6000 환경으로 이전 |
| Others (10 files) | 통합/폐기 |

</details>

---

## 핵심 결과 요약 (2026-02-22 기준)

### FL vs PS Fair Comparison (M5 Same-Camera)

| Metric | FL GS-LRM 6v | FL E2E | PS M5 6v |
|--------|:-----------:|:------:|:--------:|
| PSNR_fg | **23.84** | 8.44 | 13.78 |
| IoU | **0.954** | 0.495 | 0.846 |
| Coverage | ~1.0 | 0.750 | 0.893 |
| PSNR_inter | **~23.84** | — | 20.47 |

> FL GS-LRM >> PS by +10.06 dB. E2E bottleneck = Stage 1 multi-view diffusion (14% transfer rate).
> See: [[experiments/FL_vs_PS_comparison]] + [[experiments/mvdiffusion_bottleneck_analysis]]

### Active Experiments (2026-02-22)

| GPU | Experiment | Purpose | ETA |
|:---:|-----------|---------|:---:|
| gpu03:4 | HP M5_5 (center+norm) | Preprocessing ablation | ~12h |
| gpu03:5 | H3 MVDiff (E2+Pose) | H_T3 test / H_T1 negative control | ~28h |
| gpu03:6 | DA1 datagen → fine-tune | **H_T1 direct test** (primary) | ~4h + ~15h |
| gpu03:7 | HP M5_4 (center only) | Preprocessing ablation | ~8h |
| joon:0 | PS M5 5v | 9-exp comparison | ~18h |

> See: [[experiments/domain_adaptation_DA1]] for DA1 pipeline details

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

*MoC v10.3 | Updated: 2026-03-05 | eval_protocol duplicate removed*
