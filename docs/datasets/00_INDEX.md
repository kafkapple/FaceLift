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
| [[EXPERIMENT_NAMING]] | 실험 명명규칙, E1-E5 시리즈, 21개 설정 |
| [[EXPERIMENT_RESULTS]] | 실험 결과 비교표, PSNR 분석 |
| [[HYPOTHESIS_VERIFICATION]] | H1-H6 가설 검증 매트릭스 |
| [[VIEW_SELECTION_ANALYSIS]] | Ghosting 분석, Novel view synthesis |
| [[PREPROCESSING_REGISTRY]] | 프리셋 정의, 전처리 명령어 |
| [[VERSION_SCHEMA]] | 버전 계층, M-Series, Split 방식 |
| [[M3_SERIES_SPEC]] | M3 시리즈 상세 명세 |

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
| M3_2b | M3_2b | Conservative | 256 | 549 | ~60% | H2 baseline |
| M3_3 | M3_3 | Safe Zoom | 256 | 549 | ~78% | H2 test (0% clip) |
| M4 | M4 | Object-centered | 가변 | 549 | ~78% | H1 test (PP correction) |

---

## 가설 검증 요약

| ID | 가설 | 상태 | 상세 |
|----|------|------|------|
| H1 | Coverage↑ → PSNR↑ | ✅ | +6 PSNR (50%→74%) |
| H2 | fx=549 필수 | ⚠️ | PP가 더 중요 |
| H3 | PP=256 필수 | ✅ | +7 PSNR 효과 |
| H4 | Aspect Ratio | 🔲 | 미검증 |
| H5 | Per-sample zoom | 🔄 | 진행 중 |
| H6 | View Quality | ✅ | Novel view 한계 |

상세: [[HYPOTHESIS_VERIFICATION]]

---

## 실험 결과 요약

| Dataset | Val PSNR | Coverage | 상세 |
|---------|----------|----------|------|
| [[presets/D3_normalized]] | **27.09** | 74% | ⭐ 최고 |
| [[presets/D7_1]] | 20.93 | 50% | 기준선 |
| [[presets/D8]] | 20.21 | 50% | 정밀 |
| [[presets/M3_1]] | TBD | 78%+ | 검증 완료 |
| [[presets/M3_2]] | TBD | 78%+ | **권장** |

상세: [[EXPERIMENT_RESULTS]]

---

## Quick Start

### 전처리 실행 (권장: M3_2)
```bash
cd /home/joon/dev/FaceLift

python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_2 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_2
```

### 학습 실행 (권장: E1_2_alpha)
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E1_2_alpha
```

### 대안 실험
```bash
# M3_1 (Global zoom) 비교
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_1 -e E1_2_alpha

# Random view selection (H6 검증)
CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E1_2_random_baseline
```

### 검증
```bash
python mouse_extensions/scripts/diagnostics/verify_pp_mvg_consistency.py \
    --datasets M3_2 --verbose
```

### 가설 검증 실험 (M3 Variants)

상세 계획: [[../experiments/HYPOTHESIS_VERIFICATION_PLAN]]

**전처리:**
```bash
# M3_2b (H2 baseline: conservative zoom)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_2b \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_2b

# M3_3 (H2 test: safe zoom, 0% clipping)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_3 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_3

# M4 (H1 test: object-centered + PP correction)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M4 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M4
```

**학습:**
```bash
# H2: M3_2b vs M3_3 (Coverage 영향, PP 동일)
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2b -e E1_2_alpha

CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_3 -e E1_2_alpha

# H1: M3_3 vs M4 (PP 영향, Coverage 유사)
CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M4 -e E1_2_alpha
```

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

## 관련 문서

- [[../00_MoC_INDEX]] - 프로젝트 문서 허브
- [[../practical/MOUSE_QUICK_REFERENCE]] - 빠른 참조
- [[../EXPERIMENT_REGISTRY]] - 실험 레지스트리

---

*Dataset Documentation v3.1 | 2026-01-26 | M3 Variants Added*
