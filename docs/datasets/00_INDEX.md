# Dataset Documentation Hub

> **Navigation**: [← MoC](../00_MoC_INDEX.md) | [CLAUDE.md](../../CLAUDE.md)
> **SSOT**: 모든 데이터셋 관련 문서의 중앙 허브

---

## Quick Reference

| 분류 | 권장 | 상세 |
|------|------|------|
| **기준선** | [[presets/D7_1\|M1 (D7.1)]] | Affine, PP=256, fx=549 |
| **정밀** | [[presets/D8\|M2 (D8)]] | Homography, skew 보정 |
| **최적** | [[presets/M3_2\|M3_2]] ⭐ | Per-sample zoom, MVG-correct |

---

## 문서 구조

### Individual Dataset Specs (presets/)

| 데이터셋 | Alias | 상태 | 상세 |
|----------|-------|------|------|
| [[presets/D7_1]] | M1 | ✅ 기준선 | Affine, PSNR 20.93 |
| [[presets/D8]] | M2 | ✅ 정밀 | Homography, PSNR 20.21 |
| [[presets/M3]] | D10.3 | ⚠️ H2 검증용 | fx=739 미정규화 |
| [[presets/M3_1]] | - | ✅ MVG-correct | Global zoom |
| [[presets/M3_2]] | - | ⭐ **권장** | Per-sample zoom |
| [[presets/D3_normalized]] | - | 📊 Reference | PSNR 27.09 최고 |

### Core Documents

| 문서 | 내용 |
|------|------|
| [[VERSION_SCHEMA]] | 버전 계층, M-Series, Split 방식 |
| [[PREPROCESSING_REGISTRY]] | 프리셋 정의, 전처리 명령어 |
| [[M3_SERIES_SPEC]] | M3 시리즈 상세 명세 |
| [[EXPERIMENT_RESULTS]] | 실험 결과 비교표 |
| [[HYPOTHESIS_VERIFICATION]] | H1-H5 가설 검증 매트릭스 |

### Reference Documents

| 문서 | 내용 |
|------|------|
| [[RAW_DATA]] | 원본 데이터 출처, 샘플링 전략 |
| [[CAMERA_CONFIG]] | 6카메라 배치, View 선택 |

---

## 데이터셋 분류 체계

### Category 1: Legacy (⛔ 사용 금지)
Object-centered crop + PP 미보정 → **geometry_broken**
- D1, D4, D6-1, D6-2, D6-3

### Category 2: Object-Centered Zoom (⚠️ Deprecated)
Adaptive zoom + Object-centered → **PP 가변 → ray error**
- [[presets/M3]], M3_norm, M3_persample

### Category 3: PP-Centered Shift (✅ Stable)
PP를 256으로 shift → **pretrained 호환**
- [[presets/D7_1\|M1 (D7.1)]], [[presets/D8\|M2 (D8)]], D7_1_t

### Category 4: Precision Homography + MVG (⭐ Recommended)
Center-aligned zoom + PP=256 자동 → **MVG 정합**
- [[presets/M3_1]] (Global zoom)
- [[presets/M3_2]] (Per-sample zoom) ⭐

---

## M-Series 요약

| Alias | Preset | 변환 | PP | fx | Coverage | 상세 |
|-------|--------|------|-----|-----|----------|------|
| M1 | D7.1 | Affine | 256 | 549 | ~50% | [[presets/D7_1]] |
| M2 | D8 | Homography | 256 | 549 | ~50% | [[presets/D8]] |
| M3 | D10.3 | Homo+Zoom | 가변 | 739 | ~78% | [[presets/M3]] |
| M3_1 | M3_1 | Global Zoom | 256 | 549 | ~78% | [[presets/M3_1]] |
| **M3_2** | M3_2 | Per-sample | 256 | 549 | ~78% | [[presets/M3_2]] ⭐ |

---

## D7 계열 관계도

```
D7 (기본: random split, fx_only scale)
├── D7_1 (M1): individual scale ← [[presets/D7_1]]
│   └── D7_1_t: D7_1 + temporal split
├── D7_2: average scale
├── D7_5: optimal scale
│   └── D7_5b: object-aware optimal
└── D7_t: D7 + temporal split
```

---

## 실험 결과 요약

| Dataset | Val PSNR | Coverage | 상세 |
|---------|----------|----------|------|
| [[presets/D3_normalized]] | **27.09** | 84.3% | ⭐ 최고 |
| [[presets/D7_1]] | 20.93 | 50.5% | 기준선 |
| [[presets/D8]] | 20.21 | 50.5% | 정밀 |
| [[presets/M3_1]] | TBD | 78%+ | 검증 완료 |
| [[presets/M3_2]] | TBD | 78%+ | **권장** |

상세: [[EXPERIMENT_RESULTS]]

---

## Quick Start

### 전처리 실행 (권장: M3_2)
```bash
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

*Dataset Documentation v2.0 | 2026-01-26 | Individual Dataset Specs Added*
