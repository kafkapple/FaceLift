# FaceLift Mouse - Documentation Hub (MoC)

> **Last Updated**: 2026-01-22
> **Single Source of Truth** for all FaceLift Mouse documentation

---

## 📚 Core Reference Documents (권위 문서)

| 문서 | 용도 | 상태 |
|------|------|------|
| **[MOUSE_QUICK_REFERENCE.md](./MOUSE_QUICK_REFERENCE.md)** | 빠른 시작, 명령어, 실험 권장 | ⭐ Primary |
| **[PREPROCESSING_REGISTRY.md](./PREPROCESSING_REGISTRY.md)** | 전처리 프리셋 비교 (D1~D9) | ⭐ Primary |
| **[GS-LRM_Loss_Formula.md](./GS-LRM_Loss_Formula.md)** | Loss 함수, 가중치, Value Range | ⭐ Primary |
| **[ALPHA_MASK_COMPLETE_GUIDE.md](./ALPHA_MASK_COMPLETE_GUIDE.md)** | mask_mode, alpha_loss, 시각화 | ⭐ Primary |

### 문서 계층 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                    MoC Dashboard (이 문서)                       │
│                         ▲ 중심 허브                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ QUICK_REFERENCE │  │  PREPROCESSING  │  │   LOSS_FORMULA  │ │
│  │   (How-To)      │  │   (Datasets)    │  │   (Training)    │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
│           │                    │                    │          │
│           └────────────────────┼────────────────────┘          │
│                                │                               │
│                    ┌───────────┴───────────┐                   │
│                    │  ALPHA_MASK_GUIDE     │                   │
│                    │  (Mask & Visualization)│                   │
│                    └───────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 즉시 실행 가능한 명령어

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

→ 상세: [MOUSE_QUICK_REFERENCE.md](./MOUSE_QUICK_REFERENCE.md)

---

## 📖 Technical Deep-Dives

### 전처리 & 카메라

| 문서 | 내용 |
|------|------|
| [Camera_Preprocessing_Analysis_Report.md](./Camera_Preprocessing_Analysis_Report.md) | 카메라 행렬 분석, 좌표 변환 |
| [coordinate_transformation_guide.md](./coordinate_transformation_guide.md) | 좌표계 변환 이론 |
| [260121_Camera_Data_Pipeline.md](./260121_Camera_Data_Pipeline.md) | 데이터 파이프라인 상세 |

### 설정 & 실험

| 문서 | 내용 |
|------|------|
| [CONFIG_MODULAR.md](./CONFIG_MODULAR.md) | 모듈화 config 시스템 |
| [EXPERIMENT_REGISTRY.md](./EXPERIMENT_REGISTRY.md) | 실험 ID 레지스트리 |
| [EXPERIMENT_MATRIX.md](./EXPERIMENT_MATRIX.md) | 실험 파라미터 조합 |

### 마스크 & Loss

| 문서 | 내용 |
|------|------|
| [D9_MASK_EXPERIMENT_PIPELINE.md](./D9_MASK_EXPERIMENT_PIPELINE.md) | D9 마스크 실험 파이프라인 |
| [Research_Note_Mask_Binarization_Issue.md](./Research_Note_Mask_Binarization_Issue.md) | 마스크 이진화 이슈 분석 |

---

## 🔬 Experiment Tracking

### 현재 권장 실험

| 우선순위 | 데이터셋 | 실험 | 목적 |
|----------|---------|------|------|
| P1 | D7.1 | E_quick_alpha | 기본 검증 (mask_mode=gt) |
| P2 | D8 | E_quick_alpha | Skew 보정 비교 |
| P3 | D7_5 | E_quick_alpha | 확대 마우스 |

→ 상세: [PREPROCESSING_REGISTRY.md](./PREPROCESSING_REGISTRY.md) Section "권장 프리셋"

### 핵심 지표

| 지표 | 목표 | 모니터링 |
|------|------|----------|
| PSNR | >25 dB | WandB `train/psnr` |
| mask_iou | >0.8 | WandB `train/mask_iou` |
| fg_coverage | ~0.05 | WandB (정상 범위) |

→ 상세: [GS-LRM_Loss_Formula.md](./GS-LRM_Loss_Formula.md)

---

## 📁 Archive

### 날짜별 보고서

| 위치 | 내용 |
|------|------|
| [reports/archive/](./reports/archive/) | 2024년 보고서 |
| [reports/](./reports/) | 현재 보고서 |

### Legacy 문서

| 문서 | 상태 |
|------|------|
| D7_SCALE_MODES.md | ⚠️ D7 시리즈 레거시 |
| v3_vs_v5_comparison.md | ⚠️ 초기 버전 비교 |
| preprocessing_theory_v5.md | ⚠️ 이론 레거시 |

---

## 🔗 Backlink Protocol

### 모든 문서에 추가할 헤더

```markdown
> **Navigation**: [← MoC Dashboard](./reports/00_MoC_INDEX.md) | [Quick Reference](./MOUSE_QUICK_REFERENCE.md)
```

### 문서 간 참조 규칙

1. **권위 문서 수정 시**: 이 MoC에 변경 기록
2. **새 문서 생성 시**: MoC에 링크 추가
3. **문서 이동/삭제 시**: MoC 링크 업데이트

---

## 📊 Document Status

| 문서 | Version | Last Updated | Status |
|------|---------|--------------|--------|
| MOUSE_QUICK_REFERENCE | - | 2026-01-22 | ✅ Active |
| PREPROCESSING_REGISTRY | v4.1 | 2026-01-22 | ✅ Active |
| GS-LRM_Loss_Formula | - | 2026-01-22 | ✅ Active |
| ALPHA_MASK_COMPLETE_GUIDE | v1.1 | 2026-01-22 | ✅ Active |
| MoC Dashboard (이 문서) | v2.0 | 2026-01-22 | ✅ Active |

---

## 🔧 Maintenance

### 주간 점검 항목

- [ ] 권위 문서 4개 최신 상태 확인
- [ ] 새 실험 결과 반영 여부
- [ ] Dead link 점검

### 문서 추가 절차

1. 문서 작성
2. MoC에 링크 추가
3. 관련 권위 문서에 상호 참조 추가

---

*FaceLift Mouse Project | MoC Dashboard v2.0*
