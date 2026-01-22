# FaceLift Mouse - Documentation Hub (MoC)

> **Last Updated**: 2026-01-23
> **Single Source of Truth** for all FaceLift Mouse documentation

---

## 📁 Document Structure

```
docs/
├── 00_MoC_INDEX.md              ← 현재 문서 (중앙 허브)
│
├── practical/                    ⚡ 실용적 실험 문서
│   ├── QUICK_REFERENCE.md       # 빠른 시작, 명령어
│   ├── datasets/                # 데이터셋 설정
│   ├── config/                  # 설정 시스템
│   └── experiments/             # 실험 파이프라인
│
├── theory/                       📚 이론 문서
│   ├── model/                   # GS-LRM 아키텍처
│   ├── camera/                  # MVG, 카메라 기하학
│   ├── loss/                    # Loss 수식, 가중치
│   ├── mask/                    # Mask, Alpha 이론
│   └── metrics/                 # 평가 지표
│
├── reference/                    📖 참조 문서
├── tutorials/                    📘 튜토리얼 (Step 0-5)
├── troubleshooting/              🔧 문제 해결
├── reports/                      📊 날짜별 보고서
└── _archive/                     🗄️ 아카이브 (정리된 구문서)
```

---

## ⚡ Practical (실용적 실험 문서)

### Quick Start
| 문서 | 용도 |
|------|------|
| **[MOUSE_QUICK_REFERENCE](./practical/MOUSE_QUICK_REFERENCE.md)** | 빠른 시작, 명령어, 실험 권장 |

### Datasets (데이터셋)
| 문서 | 용도 |
|------|------|
| **[PREPROCESSING_REGISTRY](./practical/datasets/PREPROCESSING_REGISTRY.md)** | 전처리 프리셋 비교 (D1~D9) |
| [MOUSE_DATASET](./practical/datasets/MOUSE_DATASET.md) | 데이터셋 출처 (DANNCE) |

### Config (설정)
| 문서 | 용도 |
|------|------|
| [CONFIG_MODULAR](./practical/config/CONFIG_MODULAR.md) | 모듈화 config 시스템 |
| [EXPERIMENT_MATRIX](./practical/config/EXPERIMENT_MATRIX.md) | 실험 파라미터 조합 |

### Experiments (실험)
| 문서 | 용도 |
|------|------|
| [EXPERIMENT_REGISTRY](./practical/experiments/EXPERIMENT_REGISTRY.md) | 실험 ID 레지스트리 |
| [D9_MASK_EXPERIMENT_PIPELINE](./practical/experiments/D9_MASK_EXPERIMENT_PIPELINE.md) | D9 마스크 실험 |

---

## 📚 Theory (이론 문서)

### Model (아키텍처)
| 문서 | 내용 |
|------|------|
| **[GS-LRM_ARCHITECTURE_GUIDE](./theory/model/GS-LRM_ARCHITECTURE_GUIDE.md)** | 3DGS vs GS-LRM, 픽셀당 Gaussian |
| [FLOATER_ARTIFACT_ANALYSIS](./theory/model/FLOATER_ARTIFACT_ANALYSIS.md) | Ghosting 원인, 해결책 |

### Camera (MVG, 카메라 기하학)
| 문서 | 내용 |
|------|------|
| **[coordinate_transformation_guide](./theory/camera/coordinate_transformation_guide.md)** | 좌표계 변환, 정규화 이론, 시각화 |

### Loss (손실 함수)
| 문서 | 내용 |
|------|------|
| **[GS-LRM_Loss_Formula](./theory/loss/GS-LRM_Loss_Formula.md)** | Loss 함수, 가중치, Value Range |

### Metrics (평가 지표)
| 문서 | 내용 |
|------|------|
| **[METRICS_GUIDE](./theory/metrics/METRICS_GUIDE.md)** | PSNR, SSIM, mask_iou, 진단 가이드 |

### Mask (마스크)
| 문서 | 내용 |
|------|------|
| **[ALPHA_MASK_COMPLETE_GUIDE](./theory/mask/ALPHA_MASK_COMPLETE_GUIDE.md)** | mask_mode, alpha_loss, 시각화 |
| [Research_Note_Mask_Binarization_Issue](./theory/mask/Research_Note_Mask_Binarization_Issue.md) | 마스크 이진화 이슈 분석 |

---

## 📖 Reference & Tutorials

### Reference (참조)
| 문서 | 내용 |
|------|------|
| [Code_Location_Registry](./reference/Code_Location_Registry.md) | 핵심 코드 위치 |
| [Config_Options](./reference/Config_Options.md) | 설정 옵션 상세 |
| [Project_Structure](./reference/Project_Structure.md) | 프로젝트 구조 |

### Tutorials (튜토리얼)
| 단계 | 문서 | 내용 |
|------|------|------|
| 0 | [Branch_Setup](./tutorials/Step0_Branch_Setup.md) | 브랜치 설정 |
| 1 | [Fork_and_Setup](./tutorials/Step1_Fork_and_Setup.md) | 포크 및 환경 |
| 2 | [Mouse_Dataset](./tutorials/Step2_Mouse_Dataset.md) | 데이터셋 준비 |
| 3 | [Preprocessing](./tutorials/Step3_Preprocessing.md) | 전처리 |
| 4 | [Config_Setup](./tutorials/Step4_Config_Setup.md) | 설정 |
| 5 | [Training](./tutorials/Step5_Training.md) | 학습 |

### Troubleshooting (문제 해결)
| 문서 | 내용 |
|------|------|
| [clip_tokenizer_error](./troubleshooting/clip_tokenizer_merges_error.md) | CLIP 토크나이저 오류 |

---

## 🚀 Quick Start

```bash
# 1. 환경 설정
ssh gpu03
cd /home/joon/dev/FaceLift

# 2. 학습 (D7.1 + mask_mode=gt)
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E_quick_alpha

# 3. Alpha Threshold 시각화
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_alpha_thresholds \
    --checkpoint checkpoints/gslrm/D7_1_E_quick_alpha/ckpt_step_500.pt \
    --thresholds 0.3 0.5 0.7 0.9
```

→ 상세: [MOUSE_QUICK_REFERENCE](./practical/MOUSE_QUICK_REFERENCE.md)

---

## 🔬 Experiment Tracking

### 현재 권장 실험

| 우선순위 | 데이터셋 | 실험 | 목적 |
|----------|---------|------|------|
| P1 | D7.1 | E_quick_alpha | 기본 검증 (mask_mode=gt) |
| P2 | D8 | E_quick_alpha | Skew 보정 비교 |
| P3 | D7_5 | E_quick_alpha | 확대 마우스 |

### 핵심 지표

| 지표 | 목표 | 모니터링 |
|------|------|----------|
| PSNR | >25 dB | WandB `train/psnr` |
| mask_iou | >0.8 | WandB `train/mask_iou` |
| fg_coverage | ~0.05 | WandB (정상 범위) |

→ 상세: [GS-LRM_Loss_Formula](./theory/loss/GS-LRM_Loss_Formula.md)

---

## 🗄️ Archive

### Reports
| 위치 | 내용 |
|------|------|
| [reports/](./reports/) | 날짜별 분석 보고서 |

### Archive (정리된 구문서)
| 폴더 | 내용 |
|------|------|
| [_archive/camera/](./_archive/camera/) | 이전 카메라 문서 |
| [_archive/guides/](./_archive/guides/) | 이전 가이드 문서 |
| [_archive/legacy/](./_archive/legacy/) | 레거시 문서 |

---

## 🔗 Backlink Protocol

### Navigation Header (모든 핵심 문서)

```markdown
> **Navigation**: [← MoC](../00_MoC_INDEX.md) | [Practical](../practical/) | [Theory](../theory/)
```

### 문서 규칙

1. **Practical** 문서: 실행 가능한 명령어, 설정 예시 포함
2. **Theory** 문서: 수식, 원리, 검증 근거 포함
3. **새 문서 생성 시**: MoC에 링크 추가

---

## 📊 Document Status

| 카테고리 | 문서 | 상태 |
|----------|------|------|
| Practical | MOUSE_QUICK_REFERENCE | ✅ Active |
| Practical/datasets | PREPROCESSING_REGISTRY | ✅ Active |
| Theory/model | GS-LRM_ARCHITECTURE_GUIDE | ✅ Active |
| Theory/model | FLOATER_ARTIFACT_ANALYSIS | ✅ Active |
| Theory/camera | coordinate_transformation_guide | ✅ Active |
| Theory/loss | GS-LRM_Loss_Formula | ✅ Active |
| Theory/mask | ALPHA_MASK_COMPLETE_GUIDE | ✅ Active |
| Theory/metrics | METRICS_GUIDE | ✅ Active |
| Reference | Code_Location_Registry | ✅ Active |
| Tutorials | Step0-5 | ✅ Active |
| MoC | 00_MoC_INDEX (이 문서) | ✅ Active |

---

*FaceLift Mouse Project | MoC Dashboard v3.2 | 2026-01-23*
