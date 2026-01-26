# FaceLift Mouse: Document Index

> **목적**: 핵심 문서 간 연결 및 변경 추적
> **최종 업데이트**: 2026-01-25

---

## 📋 문서 계층 구조

```
docs/
├── 00_MoC_INDEX.md                    # 중앙 허브 (진입점)
│
├── practical/                          # 실용 문서
│   ├── QUICK_START.md
│   ├── EXPERIMENT_REGISTRY.md          ← 실험 ID 목록
│   ├── EXPERIMENT_SCHEMA.md            ← 명명 규칙
│   ├── HYPOTHESIS_EXPERIMENT_PLAN.md   ← 가설 및 계획
│   └── TRAINING_LOGGING_GUIDE.md       ← 학습 지표
│
├── theory/                             # 이론 문서
│   └── camera/
│       ├── coordinate_transformation_guide.md
│       └── PP_FIX_MVG_THEORY.md        ← ★ NEW (PP 고정 이론)
│
├── analysis/                           # 분석 보고서
│   ├── COMPREHENSIVE_ANALYSIS_260125.md ← ★ NEW (종합 분석)
│   ├── TRAIN_VAL_GAP_ANALYSIS.md
│   └── GHOSTING_SOLUTION_STRATEGY.md
│
└── preprocessing/
    └── PREPROCESSING_REGISTRY.md
```

---

## 🔗 문서 간 연결 (Backlinks)

### COMPREHENSIVE_ANALYSIS_260125.md
**참조**: 
- [[HYPOTHESIS_EXPERIMENT_PLAN]] - 가설 검증 계획
- [[TRAIN_VAL_GAP_ANALYSIS]] - Gap 상세 분석
- [[PP_FIX_MVG_THEORY]] - PP 고정 이론적 근거

**참조됨**:
- [[00_MoC_INDEX]]
- [[EXPERIMENT_REGISTRY]]

### PP_FIX_MVG_THEORY.md
**참조**:
- [[COMPREHENSIVE_ANALYSIS_260125]] - 종합 분석
- [[coordinate_transformation_guide]] - 좌표 변환 이론
- [[PREPROCESSING_REGISTRY]] - 전처리 버전

**참조됨**:
- [[COMPREHENSIVE_ANALYSIS_260125]]
- [[HYPOTHESIS_EXPERIMENT_PLAN]]

### HYPOTHESIS_EXPERIMENT_PLAN.md
**참조**:
- [[EXPERIMENT_REGISTRY]] - 실험 설정
- [[TRAIN_VAL_GAP_ANALYSIS]] - Gap 분석
- [[PP_FIX_MVG_THEORY]] - PP 이론

**참조됨**:
- [[00_MoC_INDEX]]
- [[COMPREHENSIVE_ANALYSIS_260125]]

---

## 📝 변경 이력

| 날짜 | 문서 | 변경 내용 |
|------|------|-----------|
| 2026-01-25 | **COMPREHENSIVE_ANALYSIS_260125** | ★ 신규 - PP 버그 발견, 종합 분석 |
| 2026-01-25 | **PP_FIX_MVG_THEORY** | ★ 신규 - MVG 이론 기반 PP 고정 분석 |
| 2026-01-25 | EXPERIMENT_SCHEMA | 명명 규칙 체계화 |
| 2026-01-25 | EXPERIMENT_REGISTRY | v2.4 - 새 명명 체계 |
| 2026-01-25 | preprocess.py | zoom_center_mode 옵션 추가 |
| 2026-01-25 | presets.py | M3_persample_centered 프리셋 추가 |

---

## ⚠️ 동기화 필수 항목

문서 수정 시 연관 문서도 함께 확인:

| 수정 대상 | 함께 확인 필요 |
|-----------|----------------|
| 전처리 코드 | PREPROCESSING_REGISTRY, PP_FIX_MVG_THEORY |
| 실험 설정 | EXPERIMENT_REGISTRY, EXPERIMENT_SCHEMA |
| 가설 결과 | HYPOTHESIS_EXPERIMENT_PLAN, COMPREHENSIVE_ANALYSIS |
| 카메라 이론 | coordinate_transformation_guide, PP_FIX_MVG_THEORY |

---

*Document Index v1.0 | 2026-01-25*
