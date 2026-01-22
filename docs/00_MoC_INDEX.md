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
│   ├── camera/                  # MVG, 카메라 기하학
│   ├── loss/                    # Loss 수식, 가중치
│   └── mask/                    # Mask, Alpha 이론
│
├── reports/                      📊 날짜별 보고서
└── legacy/                       ⚠️ Deprecated
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

### Camera (MVG, 카메라 기하학)
| 문서 | 내용 |
|------|------|
| [coordinate_transformation_guide](./theory/camera/coordinate_transformation_guide.md) | 좌표계 변환 이론 |
| [Camera_Preprocessing_Analysis_Report](./theory/camera/Camera_Preprocessing_Analysis_Report.md) | 카메라 행렬 분석 |

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

## 📁 Archive & Legacy

### Reports
| 위치 | 내용 |
|------|------|
| [reports/archive/](./reports/archive/) | 날짜별 보고서 |

### Legacy (⚠️ Deprecated)
| 문서 | 상태 |
|------|------|
| [D7_SCALE_MODES](./legacy/D7_SCALE_MODES.md) | ⚠️ D7 시리즈 레거시 |
| [v3_vs_v5_comparison](./legacy/v3_vs_v5_comparison.md) | ⚠️ 초기 버전 비교 |

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
| Practical/datasets | MOUSE_DATASET | ✅ Active |
| Theory/loss | GS-LRM_Loss_Formula | ✅ Active |
| Theory/mask | ALPHA_MASK_COMPLETE_GUIDE | ✅ Active |
| Theory/metrics | METRICS_GUIDE | ✅ Active |
| MoC | 00_MoC_INDEX (이 문서) | ✅ Active |

---

*FaceLift Mouse Project | MoC Dashboard v3.0 | 2026-01-23*
