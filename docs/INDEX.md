# FaceLift Mouse Documentation

> **MoC (Map of Content)** - 중앙 허브
> **Updated**: 260207
> **Server**: 26개 (실험/데이터/실행 가이드) | **Obsidian**: 26+22개 (이론/연구/노트)

---

## 🔬 연구 대시보드

| 문서 | 설명 |
|------|------|
| **[[RESEARCH_HYPOTHESES]]** | ⭐ **가설/실험 메인 허브** |

---

## Quick Links

| 목적 | 문서 |
|------|------|
| ⚡ **명령어** | [[experiments/COMMANDS]] |
| 🔬 실험 설정 | [[experiments/EXPERIMENT_REGISTRY]] |
| 📊 데이터셋 | [[datasets/PREPROCESSING_REGISTRY]] |

---

## 문서 구조

### 연구 가설 (hypotheses/)
| 문서 | 내용 | 상태 |
|------|------|------|
| **[[hypotheses/H4_VIEW_ABLATION]]** | View 수 최적화 | 🔄 실행 대기 |
| [[hypotheses/H5_MVDIFFUSION]] | MVDiffusion 개선 | 🔄 진행중 |
| [[hypotheses/H6_ALPHA_MASK]] | Alpha mask | ⏳ 대기 |
| [[hypotheses/H7_SSIM_WEIGHT]] | SSIM weight | ⏳ 대기 |
| [[hypotheses/H8_LITERATURE_SURVEY]] | 문헌 기반 개선 | ⏳ 대기 |
| [[hypotheses/H8_REDUCED_VIEW_GENERATION]] | 축소 뷰 생성 | ⏳ 대기 |

### 실험 가이드 (experiments/)
| 문서 | 내용 |
|------|------|
| **[[experiments/COMMANDS]]** | ⭐ 명령어 SSOT |
| [[experiments/EXPERIMENT_REGISTRY]] | 실험 설정/결과 |
| [[experiments/EXPERIMENT_CONFIG_GUIDE]] | Config 시스템 상세 |
| [[experiments/EVALUATION_GUIDE]] | 평가 기준/Split |
| [[experiments/INFERENCE_E2E_GUIDE]] | E2E 추론 |
| [[experiments/RTX3060_GUIDE]] | RTX 3060 설정 |
| [[experiments/TRAINING_LOGGING_GUIDE]] | 학습 로깅 |
| [[experiments/TRAINING_STEPS_CONVENTION]] | Step 규약 |
| [[experiments/UNIFORM_EXPERIMENT_PROTOCOL]] | 통합 실험 프로토콜 |
| [[experiments/VISUALIZATION_SETTINGS]] | 시각화 설정 |
| [[experiments/TEMPORAL_EXPERIMENTS_PLAN]] | Temporal 실험 계획 |

### datasets/
| 문서 | 내용 |
|------|------|
| [[datasets/PREPROCESSING_REGISTRY]] | 전처리 SSOT |
| [[datasets/M5_SERIES_SPEC]] | M5 시리즈 스펙 |
| [[datasets/RAW_DATA]] | Raw 데이터 정보 |

### guides/ (실행 가이드)
| 문서 | 내용 |
|------|------|
| [[guides/MVDIFFUSION_FINETUNE_GUIDE]] | MVDiffusion finetune |
| [[guides/MOUSE_DATASET_GUIDE]] | 데이터셋 가이드 |
| [[guides/M5_MIGRATION_GUIDE]] | M5 마이그레이션 |
| [[guides/POSE_SPLATTER_GUIDE]] | Pose Splatter 비교 |
| [[guides/DEFORMATION_INTEGRATION_GUIDE]] | Temporal consistency |

---

## 핵심 인사이트

### View Ablation (260205)
- **3-view (21.12) > 4-view (19.58)** (Inference)

### H3 진단 (260205)
- **MVDiffusion이 병목** (M5t: +1.41 PSNR gap)
- **데이터 다양성 > Epoch** (2880×6 > 1198×20)

---

*MoC v4.0 | 260207*
