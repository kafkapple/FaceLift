# FaceLift Mouse Documentation Map of Content (MoC)

> **최종 업데이트**: 2026-01-26
> **목적**: 모든 핵심 문서의 중앙 네비게이션 허브


**See [SYNC_GUIDE.md](SYNC_GUIDE.md) for documentation management workflow.**

---

## 1. 문서 계층 구조

```
docs/
├── 00_MoC_INDEX.md (이 문서)
│
├── datasets/                     # 데이터셋 문서 (SSOT) ⭐
│   ├── 00_INDEX.md               # Dataset Hub ★NEW
│   ├── VERSION_SCHEMA.md         # 버전 체계
│   ├── M3_SERIES_SPEC.md         # M3 시리즈 상세
│   ├── PREPROCESSING_REGISTRY.md # 전처리 레지스트리 (통합) ★UPDATED
│   ├── EXPERIMENT_RESULTS.md     # 실험 결과표 ★NEW
│   ├── HYPOTHESIS_VERIFICATION.md# 가설 검증 매트릭스 ★NEW
│   ├── RAW_DATA.md               # 원본 데이터 출처 ★NEW
│   └── CAMERA_CONFIG.md          # 카메라 배치 ★NEW
│
├── theory/                       # 이론 문서
│   ├── PP_FX_MVG_ANALYSIS.md     # PP/fx 종합 분석 ⭐
│   ├── PREPROCESSING_METHODS_COMPARISON.md
│   └── camera/ (legacy)
│
├── analysis/                     # 분석 보고서
│   ├── COMPREHENSIVE_ANALYSIS_260125.md
│   ├── PP_MVG_COMPREHENSIVE_ANALYSIS.md
│   └── TRAIN_VAL_GAP_ANALYSIS.md
│
├── practical/                    # 실무 가이드
│   ├── MOUSE_QUICK_REFERENCE.md  # 빠른 참조
│   └── datasets/ (→ ../datasets/ 리다이렉트)
│
├── EXPERIMENT_REGISTRY.md        # 실험 레지스트리
├── HYPOTHESIS_EXPERIMENT_PLAN.md # 가설 검증 계획
└── PREPROCESSING_REGISTRY.md     # (→ datasets/ 리다이렉트)
```

---

## 2. 데이터셋 Quick Reference

### 2.1 M-Series (권장)

| Alias | Preset | PP | fx | Coverage | 상태 |
|-------|--------|-----|-----|----------|------|
| M1 | D7.1 | 256 | 549 | ~50% | ✅ 기준선 |
| M2 | D8 | 256 | 549 | ~50% | ✅ 정밀 |
| M3 | D10.3 | 가변 | 739 | ~78% | ⚠️ H2 검증용 |
| **M3_1** | M3_1 | 256 | 549 | ~78% | ✅ MVG-correct |
| **M3_2** | M3_2 | 256 | 549 | ~78% | ⭐ **권장** |

**상세**: [[datasets/00_INDEX]]

### 2.2 실험 결과 요약

| Dataset | Val PSNR | Coverage | 비고 |
|---------|----------|----------|------|
| D3_normalized | **27.09** | 84.3% | 최고 성능 |
| D7_1 | 20.93 | 50.5% | 기준선 |
| M3_norm | 17.09 | 78.5% | PP 문제 |
| **M3_1/M3_2** | TBD | 78%+ | 검증 완료 |

**상세**: [[datasets/EXPERIMENT_RESULTS]]

---

## 3. 문서 관계도 (Backlinks)

### 3.1 데이터셋 문서 체인 (SSOT)

```
datasets/00_INDEX.md (Hub)
    ├── VERSION_SCHEMA.md (버전 체계)
    ├── M3_SERIES_SPEC.md (M3 상세)
    ├── PREPROCESSING_REGISTRY.md (프리셋)
    ├── EXPERIMENT_RESULTS.md (결과)
    ├── HYPOTHESIS_VERIFICATION.md (가설)
    ├── RAW_DATA.md (원본)
    └── CAMERA_CONFIG.md (카메라)
```

### 3.2 PP/MVG 이론 체인

```
theory/PP_FX_MVG_ANALYSIS.md (이론)
    ├── analysis/PP_MVG_COMPREHENSIVE_ANALYSIS.md (분석)
    ├── datasets/M3_SERIES_SPEC.md (구현)
    └── datasets/PREPROCESSING_REGISTRY.md (프리셋)
```

### 3.3 실험 분석 체인

```
HYPOTHESIS_EXPERIMENT_PLAN.md (가설)
    ├── EXPERIMENT_REGISTRY.md (실험)
    ├── datasets/EXPERIMENT_RESULTS.md (결과)
    └── analysis/TRAIN_VAL_GAP_ANALYSIS.md (Gap)
```

---

## 4. 주제별 빠른 참조

### 4.1 데이터셋 선택

| 질문 | 참조 문서 |
|------|-----------|
| 어떤 데이터셋 써야? | [[datasets/00_INDEX]] |
| M3 시리즈 차이는? | [[datasets/M3_SERIES_SPEC]] |
| 실험 결과 비교? | [[datasets/EXPERIMENT_RESULTS]] |
| 가설 검증 현황? | [[datasets/HYPOTHESIS_VERIFICATION]] |

### 4.2 전처리

| 질문 | 참조 문서 |
|------|-----------|
| 프리셋 정의는? | [[datasets/PREPROCESSING_REGISTRY]] |
| 원본 데이터는? | [[datasets/RAW_DATA]] |
| 카메라 배치는? | [[datasets/CAMERA_CONFIG]] |

### 4.3 PP/MVG 이론

| 질문 | 참조 문서 |
|------|-----------|
| PP 정합성이란? | [[theory/PP_FX_MVG_ANALYSIS]] |
| Ray Error 공식? | [[datasets/HYPOTHESIS_VERIFICATION#ray-error]] |

---

## 5. 검증 스크립트

| 스크립트 | 용도 |
|----------|------|
| verify_pp_mvg_consistency.py | PP 분포, Ray Error 검증 |

```bash
cd /home/joon/dev/FaceLift
python mouse_extensions/scripts/diagnostics/verify_pp_mvg_consistency.py \
    --datasets M3_1,M3_2 --verbose
```

---

## 6. 권장 워크플로우

### 새 실험 시작

1. [[datasets/00_INDEX]] 확인 → M3_2 권장
2. 전처리 실행 (--preset M3_2)
3. verify_pp_mvg_consistency.py 검증
4. [[EXPERIMENT_REGISTRY]] → E1_2_gt_alpha 선택
5. 학습 실행 (`-d M3_2 -e E1_2_gt_alpha`)
6. [[datasets/EXPERIMENT_RESULTS]] 와 비교

### Quick Command

```bash
# M3_2 전처리
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_2 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_2

# 학습 (권장)
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E1_2_gt_alpha
```

---

## 7. 변경 이력

| 날짜 | 변경 |
|------|------|
| **2026-01-26** | **datasets/ 문서 통합 (SSOT 단일화)** ★ |
| 2026-01-26 | 00_INDEX, EXPERIMENT_RESULTS, HYPOTHESIS_VERIFICATION, RAW_DATA, CAMERA_CONFIG 신규 |
| 2026-01-26 | practical/datasets, preprocessing → datasets 리다이렉트 |
| 2026-01-25 | M3_1, M3_2 도입 |
| 2026-01-25 | M3_SERIES_SPEC.md 신규 |

---

*MoC Index v3.0 | 2026-01-26 | Dataset Documentation Unified*
