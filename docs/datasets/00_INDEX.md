# Dataset Documentation Hub

> **Navigation**: [← MoC](../00_MoC_INDEX.md) | [CLAUDE.md](../../CLAUDE.md)
> **SSOT**: 모든 데이터셋 관련 문서의 중앙 허브

---

## Quick Reference

| 분류 | 권장 | 설명 |
|------|------|------|
| **기준선** | M1 (D7.1) | Affine, PP=256, fx=549 |
| **정밀** | M2 (D8) | Homography, skew 보정 |
| **최적** | M3_2 | Per-sample zoom, MVG-correct ⭐ |

---

## 문서 구조

### Core Documents

| 문서 | 내용 | 상태 |
|------|------|------|
| [[VERSION_SCHEMA]] | 버전 계층, M-Series, Split 방식 | ✅ |
| [[PREPROCESSING_REGISTRY]] | 프리셋 정의, 전처리 명령어 | ✅ |
| [[M3_SERIES_SPEC]] | M3 시리즈 상세 명세 | ✅ |
| [[EXPERIMENT_RESULTS]] | 실험 결과 비교표 | ✅ |
| [[HYPOTHESIS_VERIFICATION]] | H1-H5 가설 검증 매트릭스 | ✅ |

### Reference Documents

| 문서 | 내용 |
|------|------|
| [[RAW_DATA]] | 원본 데이터 출처, 샘플링 전략 |
| [[CAMERA_CONFIG]] | 6카메라 배치, View 선택 |

### Theory (→ ../theory/)

| 문서 | 내용 |
|------|------|
| [[../theory/PP_FX_MVG_ANALYSIS]] | PP/fx 이론, MVG 정합성 |
| [[../theory/RAY_ERROR_THEORY]] | Ray Error 계산 공식 |

---

## 데이터셋 분류 체계

### Category 1: Legacy (⛔ 사용 금지)
Object-centered crop + PP 미보정 → **geometry_broken**
- D1, D4, D6-1, D6-2, D6-3

### Category 2: Object-Centered Zoom (⚠️ Deprecated)
Adaptive zoom + Object-centered → **PP 가변 → ray error**
- M3, M3_norm, M3_persample

### Category 3: PP-Centered Shift (✅ Stable)
PP를 256으로 shift → **pretrained 호환**
- M1 (D7.1), M2 (D8), D7_1_t

### Category 4: Precision Homography + MVG (⭐ Recommended)
Center-aligned zoom + PP=256 자동 → **MVG 정합**
- M3_1 (Global zoom)
- M3_2 (Per-sample zoom) ⭐

---

## M-Series 요약

| Alias | Preset | 변환 | PP | fx | Coverage | 상태 |
|-------|--------|------|-----|-----|----------|------|
| M1 | D7.1 | Affine | 256 | 549 | ~50% | ✅ 기준선 |
| M2 | D8 | Homography | 256 | 549 | ~50% | ✅ 정밀 |
| M3 | D10.3 | Homo+Zoom | 가변 | 739 | ~78% | ⚠️ H2 검증용 |
| M3_1 | M3_1 | Global Zoom | 256 | 549 | ~78% | ✅ MVG-correct |
| M3_2 | M3_2 | Per-sample | 256 | 549 | ~78% | ✅ **권장** ⭐ |

---

## D7 계열 관계도

```
D7 (기본: random split, fx_only scale)
├── D7_1: individual scale (별도 scale_x, scale_y)
│   └── D7_1_t: D7_1 + temporal split (★ 공정 평가)
├── D7_2: average scale (동일 scale_x = scale_y)
├── D7_5: optimal scale
│   └── D7_5b: object-aware optimal
└── D7_t: D7 + temporal split
```

---

## Quick Start

### 전처리 실행
```bash
# 권장: M3_2
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_2 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_2
```

### 학습 실행
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E1_2_gt_alpha
```

### 검증
```bash
python mouse_extensions/scripts/diagnostics/verify_pp_mvg_consistency.py \
    --datasets M3_2 --verbose
```

---

## 관련 문서

- [[../00_MoC_INDEX]] - 프로젝트 문서 허브
- [[../practical/MOUSE_QUICK_REFERENCE]] - 빠른 참조
- [[../EXPERIMENT_REGISTRY]] - 실험 레지스트리

---

*Dataset Documentation v1.0 | 2026-01-26*
