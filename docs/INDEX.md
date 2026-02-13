# FaceLift Mouse Documentation

> **MoC (Map of Content)** - 중앙 허브
> **Updated: 260213
> **Active**: 30개 | **Archive**: 7개

---

## Quick Links

| 목적 | 문서 |
|------|------|
| **가설 허브** | [[RESEARCH_HYPOTHESES]] |
| **명령어 SSOT** | [[experiments/COMMANDS]] |
| **실험 설정** | [[experiments/EXPERIMENT_REGISTRY]] |
| **데이터셋 SSOT** | [[datasets/PREPROCESSING_REGISTRY]] |
| **파이프라인 아키텍처** | [[theory/PIPELINE_ARCHITECTURE]] |

---

## 문서 구조

### hypotheses/ (연구 가설)

| 문서 | 내용 | 상태 |
|------|------|------|
| **[[hypotheses/H4_VIEW_ABLATION]]** | View 수 최적화 | ✅ v2 uniform 완료 |
| [[hypotheses/H5_MVDIFFUSION]] | MVDiffusion 개선 | 🔄 진행중 |
| [[hypotheses/H6_ALPHA_MASK]] | Alpha mask | ⏳ 대기 |
| [[hypotheses/H7_SSIM_WEIGHT]] | SSIM weight | ⏳ 대기 |
| [[hypotheses/H8_REDUCED_VIEW_GENERATION]] | 축소 뷰 생성 | 🔄 진행중 |
| [[hypotheses/H8_VIEW_GENERALIZATION_ANALYSIS]] | 뷰 최소화 + 일반화 분석 | 📊 분석 |
| [[hypotheses/GENERALIZATION_ROADMAP]] | 카메라/피사체 일반화 로드맵 (P0→P4) | 📋 계획 |
| [[hypotheses/RMA_CAMERA_ANALYSIS]] | RMA + M5 카메라 비균일 배치 분석 | 📊 분석 |
| [[hypotheses/H8_LITERATURE_SURVEY]] | H8 문헌 조사 | ✅ 완료 |
| [[hypotheses/H1bis_v2_REVALIDATION]] | H3-bis 재검증 | ✅ 완료 |
| [[hypotheses/HP_PREPROCESSING_ABLATION]] | 전처리 Ablation | ⏳ 대기 |

### experiments/ (실험 가이드)

| 문서 | 내용 |
|------|------|
| **[[experiments/COMMANDS]]** | 명령어 SSOT (GS-LRM, MV-Diffusion, Turntable) |
| [[experiments/EXPERIMENT_REGISTRY]] | 실험 설정/결과 기록 |
| [[experiments/EXPERIMENT_CONFIG_GUIDE]] | Config 시스템 (Modular 3-Layer + Schema) |
| [[experiments/EVALUATION_GUIDE]] | 평가 유형/Split 전략 |
| [[experiments/INFERENCE_E2E_GUIDE]] | E2E 추론 (통합 파이프라인 + CLI) |
| [[experiments/TRAINING_LOGGING_GUIDE]] | 학습 단위/WandB 로깅 |
| [[experiments/VISUALIZATION_SETTINGS]] | Turntable/시각화 설정 + Output File System |
| [[experiments/RTX3060_GUIDE]] | RTX 3060 (12GB) 실험 설정 |

### datasets/

| 문서 | 내용 |
|------|------|
| **[[datasets/PREPROCESSING_REGISTRY]]** | 전처리 SSOT (M/D-series 전체) |
| [[datasets/M5_SERIES_SPEC]] | M5 계열 카메라 정규화 상세 |
| [[datasets/RAW_DATA]] | Raw 데이터 출처 (DANNCE) |

### guides/ (실행 가이드)

| 문서 | 내용 |
|------|------|
| [[guides/MVDIFFUSION_FINETUNE_GUIDE]] | MVDiffusion fine-tuning |
| [[guides/DEFORMATION_INTEGRATION_GUIDE]] | Temporal deformation + 실험 우선순위 |
| [[guides/MOUSE_DATASET_GUIDE]] | MouseViewDataset 구현 참조 |
| [[guides/M5_MIGRATION_GUIDE]] | M3 → M5 마이그레이션 |
| [[guides/POSE_SPLATTER_GUIDE]] | Pose Splatter 비교 프로토콜 |

### theory/

| 문서 | 내용 |
|------|------|
| **[[theory/PIPELINE_ARCHITECTURE]]** | 2-stage 파이프라인 아키텍처 (MV-Diffusion + GS-LRM) |
| **[[theory/MULTIVIEW_DIFFUSION_THEORY]]** | MV-Diffusion/GS-LRM 이론 기반 (모델 계보, attention, pose conditioning) |
| [[theory/METRICS_PROTOCOL]] | 메트릭 프로토콜 (White-BG, PoseSplatter 비교) |
| [[theory/MV_ADAPTER_TECHNICAL]] | MV-Adapter 아키텍처 분석 + FaceLift 통합 평가 |
| [[theory/SLIDES_FACELIFT_PIPELINE]] | 파이프라인 발표 슬라이드 (Marp) |

### _archive/ (7 files)

| 문서 | 사유 |
|------|------|
| MOUSE_QUICK_REFERENCE | COMMANDS.md로 통합 |
| MOUSE_REFERENCE_DETAILS | PREPROCESSING_REGISTRY로 통합 |
| UNIFORM_EXPERIMENT_PROTOCOL | 현재 실험 체계로 대체 |
| DOCUMENTATION_REQUEST_SPEC | 일회성 문서 |
| TURNTABLE_VIS_GUIDE | VIS_SETTINGS + COMMANDS로 머지 |
| TEMPORAL_EXPERIMENTS_PLAN | DEFORMATION_GUIDE로 머지 |
| TWO_PHASE_TRAINING_STRATEGY | 초기 전략 문서 (역사적) |

---

## 핵심 인사이트

### View Ablation (260209, uniform v2)
- **뷰 수 ↑ = PSNR ↑ (단조 증가)**: 6-view(23.46) > 5(22.63) > 4(21.50) > 3(19.92)
- 이전 R1(260205, 비균일): 3-view best → v2(uniform)에서 반전됨

### H3 진단 (260205)
- **MVDiffusion이 병목** (M5t: +1.41 PSNR gap)
- **데이터 다양성 > Epoch** (2880x6 > 1198x20)

### Novel View Quality (260211, E1 quick eval)
- **View 0 (input): 39.5 dB vs Views 1-5 (novel): 24.0 dB = 15.5 dB gap**
- View 2 (90deg) consistently worst (~21 dB)
- CFG guidance scale 3.0 improves novel views by +1.0 dB

---

*MoC v5.0 | 260211*
