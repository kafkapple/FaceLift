# FaceLift Mouse Documentation

> **MoC (Map of Content)** - 중앙 허브
> **Updated**: 260207

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

### 연구 가설 (experiments/)
| 문서 | 내용 | 상태 |
|------|------|------|
| **[[experiments/H4_VIEW_ABLATION]]** | View 수 최적화 | 🔄 실행 대기 |
| [[experiments/H5_MVDIFFUSION]] | MVDiffusion 개선 | 🔄 진행중 |
| [[experiments/H6_ALPHA_MASK]] | Alpha mask | ⏳ 대기 |

### 실험 가이드 (experiments/)
| 문서 | 내용 |
|------|------|
| **[[experiments/COMMANDS]]** | ⭐ 명령어 SSOT |
| [[experiments/EXPERIMENT_REGISTRY]] | 실험 설정/결과 |
| [[experiments/EXPERIMENT_CONFIG_GUIDE]] | Config 시스템 상세 |
| [[experiments/INFERENCE_E2E_GUIDE]] | E2E 추론 |

### datasets/
| 문서 | 내용 |
|------|------|
| [[datasets/PREPROCESSING_REGISTRY]] | 전처리 SSOT |
| [[datasets/M5_SERIES_SPEC]] | M5 시리즈 스펙 |

### guides/
| 문서 | 내용 |
|------|------|
| [[guides/POSE_SPLATTER_GUIDE]] | Pose Splatter 비교 |
| [[guides/DEFORMATION_INTEGRATION_GUIDE]] | Temporal consistency |

### research/
| 날짜 | 주제 |
|------|------|
| [[research/260205_Research_Notes]] | H1 진단, View Ablation |
| [[research/260204_Research_Notes]] | 4D Gaussian Methods |
| [[research/260203_Research_Notes]] | Deformation, CFG |

### 보고서 (outputs/reports/)
| 문서 | 내용 |
|------|------|
| [[../outputs/reports/h1_diagnosis_report]] | H3 병목 분석 |
| [[../outputs/reports/view_ablation_report]] | View Ablation |

---

## 핵심 인사이트

### View Ablation (260205)
- **3-view (21.12) > 4-view (19.58)** (Inference)

### H3 진단 (260205)
- **MVDiffusion이 병목** (M5t: +1.41 PSNR gap)
- **데이터 다양성 > Epoch** (2880×6 > 1198×20)

---

*MoC v3.0 | 260207*
