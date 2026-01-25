# FaceLift Mouse Documentation Map of Content (MoC)

> **최종 업데이트**: 2026-01-26
> **목적**: 모든 핵심 문서의 중앙 네비게이션 허브

---

## 1. 문서 계층 구조

```
docs/
+-- 00_MoC_INDEX.md (이 문서)
|
+-- analysis/                     # 분석 보고서
|   +-- COMPREHENSIVE_ANALYSIS_260125.md
|   +-- PP_MVG_COMPREHENSIVE_ANALYSIS.md
|   +-- TRAIN_VAL_GAP_ANALYSIS.md
|
+-- datasets/                     # 데이터셋 명세
|   +-- VERSION_SCHEMA.md          # 버전 체계 설명 ★NEW
|   +-- M3_SERIES_SPEC.md         # M3 시리즈 상세
|
+-- theory/                       # 이론 문서
|   +-- PP_FX_MVG_ANALYSIS.md     # PP/fx 종합 분석 (v2.1) ⭐
|   +-- PREPROCESSING_METHODS_COMPARISON.md  # 전처리 방식 비교 ★NEW
|   +-- camera/
|       +-- (legacy)
|
+-- PREPROCESSING_REGISTRY.md     # 전처리 레지스트리
+-- EXPERIMENT_REGISTRY.md        # 실험 레지스트리
+-- HYPOTHESIS_EXPERIMENT_PLAN.md # 가설 검증 계획
+-- MOUSE_QUICK_REFERENCE.md      # 빠른 참조
```

---

## 2. 데이터셋 Quick Reference

### 2.1 MVG-Correct 데이터셋 (권장)

| ID | 설명 | PP | fx | 상세 |
|----|------|-----|-----|------|
| D3_normalized | PP shift, no zoom | 256 | 549 | PSNR 27.09 |
| D7_1 | Affine transform | 256 | 549 | PSNR 20.93 |
| D8 | Homography + skew | 256 | 549 | PSNR 20.21 |
| **M3_1** | Global zoom + MVG | 256 | 549 | [[M3_SERIES_SPEC]] |
| **M3_2** | Per-sample zoom + MVG | 256 | 549 | [[M3_SERIES_SPEC]] (권장) |

### 2.2 Deprecated 데이터셋 (MVG 부정합)

| ID | 문제점 | Ray Error | 대체 |
|----|--------|-----------|------|
| M3 | fx=739 미정규화 | 6.96도 | M3_1 |
| M3_norm | PP 가변 | 13.62도 | M3_1 |
| M3_persample | PP 가변 | 16.15도 | M3_2 |

---

## 3. 문서 관계도 (Backlinks)

### 3.1 PP/MVG 정합성 문서 체인

```
PP_FIX_MVG_THEORY.md (이론)
    |-- PP_MVG_COMPREHENSIVE_ANALYSIS.md (분석)
    |-- M3_SERIES_SPEC.md (구현)
    +-- PREPROCESSING_REGISTRY.md (프리셋)
```

### 3.2 실험 분석 문서 체인

```
HYPOTHESIS_EXPERIMENT_PLAN.md (가설)
    |-- EXPERIMENT_REGISTRY.md (실험)
    |-- TRAIN_VAL_GAP_ANALYSIS.md (Gap)
    +-- PP_MVG_COMPREHENSIVE_ANALYSIS.md (PP)
```

### 3.3 전체 Backlink Matrix

| From \ To | MoC | PP_Theory | PP_Analysis | M3_Spec | PreReg | ExpReg | Gap |
|------------|-----|-----------|-------------|---------|--------|--------|-----|
| **00_MoC_INDEX** | - | O | O | O | O | O | O |
| **PP_FIX_MVG_THEORY** | O | - | O | - | O | - | - |
| **PP_MVG_COMPREHENSIVE** | O | O | - | - | O | O | O |
| **M3_SERIES_SPEC** | O | - | O | - | O | O | O |
| **PREPROCESSING_REG** | O | O | O | O | - | O | - |
| **EXPERIMENT_REG** | O | - | - | - | O | - | O |
| **TRAIN_VAL_GAP** | O | - | O | - | - | O | - |

---

## 4. 주제별 빠른 참조

### 4.1 PP (Principal Point) 문제

| 질문 | 참조 문서 |
|------|-----------|
| PP 정합성이란? | [[PP_FIX_MVG_THEORY]] |
| 데이터셋별 PP 현황은? | [[PP_MVG_COMPREHENSIVE_ANALYSIS]] |
| M3 시리즈 어떤 걸 써야? | [[M3_SERIES_SPEC]] |
| 새 프리셋 추가는? | [[PREPROCESSING_REGISTRY]] |

### 4.2 실험 설정

| 질문 | 참조 문서 |
|------|-----------|
| 어떤 실험 설정 써야? | [[EXPERIMENT_REGISTRY]] |
| Train-Val Gap 의미는? | [[TRAIN_VAL_GAP_ANALYSIS]] |
| 가설 검증 계획은? | [[HYPOTHESIS_EXPERIMENT_PLAN]] |

---

## 5. 검증 스크립트

| 스크립트 | 용도 |
|----------|------|
| verify_pp_mvg_consistency.py | PP 분포, Ray Error 검증 |

```bash
cd /home/joon/dev/FaceLift
/home/joon/anaconda3/envs/facelift/bin/python \
    mouse_extensions/scripts/diagnostics/verify_pp_mvg_consistency.py --verbose
```

---

## 6. 권장 워크플로우

### 새 실험 시작 시

1. [[M3_SERIES_SPEC]] 확인 -> M3_1 또는 M3_2 선택
2. 전처리 실행
3. verify_pp_mvg_consistency.py 로 PP 검증
4. [[EXPERIMENT_REGISTRY]] 참조 -> 실험 설정 선택
5. 학습 실행
6. [[TRAIN_VAL_GAP_ANALYSIS]] 와 결과 비교

---

## 7. 변경 이력

| 날짜 | 변경 |
|------|------|
| 2026-01-25 | M3_1, M3_2 도입 (M3_norm_centered, M3_persample_centered에서 이름 변경) |
| 2026-01-25 | M3_SERIES_SPEC.md 신규 생성 |
| 2026-01-25 | PP_MVG_COMPREHENSIVE_ANALYSIS.md 신규 생성 |
| 2026-01-25 | verify_pp_mvg_consistency.py 신규 생성 |

---

*MoC Index v2.1 | 2026-01-25*
