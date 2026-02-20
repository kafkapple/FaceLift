# FaceLift Mouse Documentation

> **MoC (Map of Content)** — 중앙 허브
> **Updated**: 2026-02-20 | **Active**: 27개 | **Archive**: 22개

---

## Quick Links

| 목적 | 문서 |
|------|------|
| **FL vs PS 비교 (SSOT)** | [[experiments/FL_vs_PS_comparison]] **v10** |
| **메트릭 코드 검증** | [[experiments/FL_PS_metric_consistency]] v2.1 (→ v9에 통합) |
| **가설 로드맵** | [[experiments/FaceLift_hypothesis_roadmap]] |
| **학습 최적 설정** | [[experiments/FaceLift_training_optimal_settings]] |
| **가설 대시보드** | [[RESEARCH_HYPOTHESES]] |
| **명령어 SSOT** | [[experiments/COMMANDS]] |
| **파이프라인 아키텍처** | [[theory/PIPELINE_ARCHITECTURE]] |

---

## 문서 구조

### experiments/ (실험 & 비교)

| 문서 | 내용 | 상태 |
|------|------|:----:|
| **[[FL_vs_PS_comparison]]** | FL vs PS 통합 비교 (Tier A/B/C + Option A) | **✅ v10** |
| **[[comprehensive_analysis_report]]** | Tier A/B/C + View Ablation 종합 분석 보고서 | **🆕 v1.0** |
| [[FL_PS_metric_consistency]] | 메트릭/데이터 일관성 코드 레벨 검증 | ✅ v2.1 (→v9) |
| **[[FaceLift_hypothesis_roadmap]]** | 가설 검정 결과 + 다음 우선순위 | ✅ |
| **[[FaceLift_training_optimal_settings]]** | Stage 1/2 최적 학습 설정 | ✅ |
| **[[MVDiff_improvement_roadmap]]** | MVDiff 개선 로드맵 (Pose Conditioning, Virtual Camera) | **🆕 v1.0** |
| [[260216_MVDIFF_TRAINING_ANALYSIS]] | MVDiff LR 전략 분석 (Phase 3) | ✅ |
| [[260216_PHASE3_REPORT]] | Phase 3 종합 보고서 | ✅ |
| [[COMMANDS]] | 명령어 SSOT (GS-LRM, MVDiff, Turntable) | ✅ |
| [[EVALUATION_GUIDE]] | 평가 유형 / Split 전략 | ✅ |
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

### guides/

| 문서 | 내용 |
|------|------|
| [[MVDIFFUSION_FINETUNE_GUIDE]] | MVDiffusion fine-tuning |
| [[MOUSE_DATASET_GUIDE]] | MouseViewDataset 구현 참조 |
| [[POSE_SPLATTER_GUIDE]] | Pose Splatter 비교 프로토콜 |

### hypotheses/ (연구 가설)

| 문서 | 핵심 질문 | 상태 |
|------|----------|:----:|
| [[H4_VIEW_ABLATION]] | 최적 입력 뷰 수 | ✅ 6-view 단조 증가 |
| [[H5_MVDIFFUSION]] | MVDiff 개선 방법 | ✅ Phase 3 완료 |
| [[H6_ALPHA_MASK]] | Alpha mask loss 효과 | ⏳ 대기 |
| [[H7_SSIM_WEIGHT]] | SSIM weight 최적값 | ⏳ 대기 |
| [[GENERALIZATION_ROADMAP]] | 카메라/피사체 일반화 | 📋 계획 |
| [[RMA_CAMERA_ANALYSIS]] | M5 카메라 비균일 배치 | 📊 분석 완료 |

### theory/

| 문서 | 내용 |
|------|------|
| **[[PIPELINE_ARCHITECTURE]]** | 2-stage 파이프라인 (MVDiff + GS-LRM) |
| **[[MULTIVIEW_DIFFUSION_THEORY]]** | MVDiff/GS-LRM 이론 기반 |
| [[METRICS_PROTOCOL]] | 메트릭 프로토콜 (White-BG, FG-only) |
| [[MV_ADAPTER_TECHNICAL]] | MV-Adapter 아키텍처 분석 |
| [[SLIDES_FACELIFT_PIPELINE]] | 파이프라인 발표 슬라이드 (Marp) |

### _archive/ (22 files)

완료·대체·폐기된 문서. 역사적 참조용.

<details>
<summary>아카이브 목록 (펼치기)</summary>

| 문서 | 사유 |
|------|------|
| FL_vs_PS_comparison_v5/v6/v7 | v9으로 대체 |
| comparison_report | v9에 통합 |
| 260215_EXPERIMENT_ANALYSIS | 일회성 보고서 |
| H1bis_v2_REVALIDATION | ✅ 완료 |
| H8_LITERATURE_SURVEY | ✅ 완료 |
| H8_REDUCED_VIEW_GENERATION | ✅ 완료 (기각) |
| H8_VIEW_GENERALIZATION_ANALYSIS | ✅ 완료 |
| HP_PREPROCESSING_ABLATION | 이론만 확인, 실험 무기한 대기 |
| RTX3060_GUIDE | A6000 환경으로 이전 |
| M5_MIGRATION_GUIDE | 마이그레이션 완료 |
| DEFORMATION_INTEGRATION_GUIDE | 현재 연구 범위 외 |
| MOUSE_QUICK_REFERENCE | COMMANDS.md로 통합 |
| MOUSE_REFERENCE_DETAILS | PREPROCESSING_REGISTRY로 통합 |
| UNIFORM_EXPERIMENT_PROTOCOL | 현재 실험 체계로 대체 |
| DOCUMENTATION_REQUEST_SPEC | 일회성 |
| TURNTABLE_VIS_GUIDE | VIS_SETTINGS + COMMANDS로 머지 |
| TEMPORAL_EXPERIMENTS_PLAN | DEFORMATION_GUIDE로 머지 |
| TWO_PHASE_TRAINING_STRATEGY | 초기 전략 문서 (역사적) |
| TRAINING_STEPS_CONVENTION | 현재 체계와 다름 |

</details>

---

## 핵심 결과 요약 (260220 기준)

### Tier A: GS-LRM GT 6-view vs PS (유일한 유효 정량 비교)

| Metric | GS-LRM 6v GT | PS 6v | Delta |
|--------|:-----------:|:-----:|:-----:|
| PSNR_gt_masked | **23.84** | 16.71 | **+7.13** |
| IoU | **0.954** | 0.827 | **+0.127** |
| PSNR_intersection | **24.02** | 20.54 | **+3.48** |

> v10.0 correction: 6v was 21.02 in v9 (A2 oracle, 4v model). Proper 6v model gives 23.84.

> GS-LRM 6v가 PS보다 +7.13 dB 우수 (동일 입력 뷰 수). View count monotonically improves quality.

### Tier B: E2E vs PS — INVALID (camera mismatch)

> ⚠️ M5 (fx=549, HFOV=50°) vs fj5_ds2 (fx≈810, HFOV≈35°) 카메라 미스매치로
> pixel-wise 비교 불가. **Option A (M5→PS 재학습)** 진행중. → [[FL_vs_PS_comparison]] §11

### PS M5 Retraining (Option A) — In Progress

- Coordinate system fixed (auto_orient), training started
- Fair Tier B comparison expected after completion

### Tier C: 파이프라인 병목

| Stage | PSNR_gt | Drop |
|-------|:-------:|:----:|
| GS-LRM GT 6v | 23.84 | baseline |
| GS-LRM GT 4v | 20.66 | -3.18 |
| GS-LRM GT 1v | 10.47 | -13.37 |
| **E2E (MVDiff→GS-LRM)** | **8.20** | **-15.64** |

> MVDiffusion이 유일한 병목 (-15.64 dB)

### Phase 3 + Camera Mismatch 결론

- MVDiff 학습 전략 (cosine/resume/pose) 무관: E2E PSNR 7.9-8.2 수렴
- **카메라 미스매치 발견**: M5 affine warp vs fj5_ds2 simple downsample → B-3 무효화
- **Option A 준비 완료**: convert_m5_for_ps.py (scale=0.008772, ell=0.00193)
- **아키텍처 변경 필요** (view consistency, silhouette supervision)

---

## 실험 결과 JSON 위치

```
experiments/comparison/
├── fair/                          # FL E2E fair eval
│   ├── facelift_fair.json         # FL E2E best (E2 resume)
│   └── fair_comparison_merged.json
├── tier/                          # Tier A/C + 모든 variant fair eval
│   ├── gslrm_{1,4,6}view_fair.json
│   ├── {baseline,cfgr,e1,e2,e3}_*_fair.json
│   ├── p1_*_fair.json             # P1 6-view E2E results
│   └── tier_comparison*.json
└── FL_vs_PS/                      # 비교 문서 원본 + JSON
    ├── metrics_comparison.json
    └── comprehensive_analysis_report.md  # 종합 분석 보고서
```

---

*MoC v8.0 | 2026-02-20*
