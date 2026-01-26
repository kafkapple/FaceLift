# FaceLift Dataset Master Reference

> **SSOT (Single Source of Truth)** - 모든 데이터셋 정보의 중앙 문서
> **Updated**: 2026-01-27

---

## Quick Reference

### 현재 권장 데이터셋

| 순위 | Dataset | PP | fx | Clipping | Coverage | 상태 | 용도 |
|------|---------|----|----|----------|----------|------|------|
| **1** | **M3_2** | 256 | 549 | 6.5% | ~5% | ⭐ **권장** | Per-sample zoom |
| **2** | **M3_1** | 256 | 549 | 0% | ~4-5% | ⭐ 안전 | Global zoom |
| 3 | D7_1 (M1) | 256 | 549 | 0% | ~3% | 기준선 | Affine |
| 4 | D8 (M2) | 256 | 549 | 0% | ~3% | 기준선 | Homography |

### 가설 검증용 데이터셋 (진행 중)

| Dataset | 목적 | PP | zoom_range | 상태 |
|---------|------|-----|------------|------|
| **M3_2b** | H2 baseline (낮은 zoom) | 256 | [1.0, 1.5] | 🔬 실험 중 |
| **M3_3** | H2 test (safe zoom) | 256 | [1.0, 2.5] + safe | 🔬 실험 중 |
| **M4** | H1 test (PP correction) | 가변 | [1.0, 2.5] + safe | 🔬 실험 중 |

### 실패/Deprecated 데이터셋 (사용 금지)

| Dataset | Ray Error | 실패 원인 |
|---------|-----------|-----------|
| M4 | 11.8° | Object-centered zoom → PP 가변 |
| M3_persample | 17.4° | zoom_center_mode: object |
| M3_norm | 15.0° | zoom_center_mode: object |
| D10.3 (M3) | ~8° | fx=739 버그 |
| D1/D4/D6-* | ~13° | PP 미보정 |

### 역사적 참고 (재현 불가)

| Dataset | PSNR | Coverage | 비고 |
|---------|------|----------|------|
| D3_normalized | 27.09 | ~6% | 수동 전처리, preprocessing_info 없음 |

---

## 핵심 원칙

### 1. PP (Principal Point) 규칙 ⭐

```
✅ 성공: zoom_center_mode = "image" → PP = 256 고정
❌ 실패: zoom_center_mode = "object" → PP 가변 → Ray Error → Ghosting
```

### 2. 카메라 파라미터 (Pretrained 기준)

```
fx = 549
cx = cy = 256 (이미지 중앙)
translation_norm ≈ 2.7
```

### 3. Coverage 효과

```
Coverage ↑ → PSNR ↑
- D7_1/D8: ~3% → PSNR ~20-21
- M3_1/M3_2: ~5% → PSNR TBD (실험 중)
- D3_normalized: ~6% → PSNR 27 (참고용)
```

---

## 성공/실패 원인 분석

### zoom_center_mode 결정적 영향

| zoom_center_mode | PP | Ray Error | 결과 |
|------------------|-----|-----------|------|
| **"image"** (center-aligned) | 256 고정 | 0° | ✅ 성공 |
| **"object"** (object-centered) | 가변 | 11-17° | ❌ 실패 |

### 검증된 사실

| 비교 | 결과 | 결론 |
|------|------|------|
| M3_2 vs M4 | PP 256 vs 가변 | PP 가변 → 실패 |
| D7_1 vs D8 | Affine vs Homography | 성능 차이 미미 (0.72 PSNR) |
| Coverage 3% vs 6% | D7_1 vs D3_norm | Coverage ↑ → PSNR ↑ |

---

## M-Series Alias

| Alias | Preset | Transform | Zoom | PP | 상태 |
|-------|--------|-----------|------|-----|------|
| M1 | D7.1 | Affine | 없음 | 256 | ✅ 기준선 |
| M2 | D8 | Homography | 없음 | 256 | ✅ 기준선 |
| M3 | D10.3 | Homography | Global | 가변 | ⛔ fx 버그 |
| - | **M3_1** | Homography | Global | 256 | ✅ 안전 |
| - | **M3_2** | Homography | Per-sample | 256 | ⭐ **권장** |
| - | M3_2b | Homography | Per-sample (낮음) | 256 | 🔬 H2 baseline |
| - | M3_3 | Homography | Per-sample + safe | 256 | 🔬 H2 test |
| - | M4 | Homography | Per-sample + PP corr | 가변 | 🔬 H1 test |

---

## M3_1 vs M3_2 핵심 차이

| 설정 | M3_1 | M3_2 |
|------|------|------|
| zoom_scope | global | **per_sample** |
| 장점 | Clipping 없음 | 샘플별 최적화 |
| 단점 | 일부 샘플 저 coverage | 6.5% clipping |
| 권장 | 안전한 선택 | 최적 성능 |

---

## 실험 명령어

### 권장 실험 (M3_2)

```bash
torchrun --standalone --nproc_per_node=1 train_gslrm.py \
    -b gslrm_mouse -d M3_2 -e E1_2_alpha
```

### 가설 검증 실험

```bash
# H2 baseline (M3_2b)
torchrun ... -b gslrm_mouse -d M3_2b -e E1_2_alpha

# H2 test (M3_3)
torchrun ... -b gslrm_mouse -d M3_3 -e E1_2_alpha

# H1 test (M4)
torchrun ... -b gslrm_mouse -d M4 -e E1_2_alpha
```

---

## 전처리 Evolution

```
v5 (원본) → 생쥐 위치/크기 불균일
    ↓
v2 (pixel_based) → 뷰별 독립 변환 → 복수 생쥐
    ↓
D1-D6 (PP 강제) → Ray 방향 오류
    ↓
D7/D8 (M1/M2) → PP 보정 + fx 정규화 → PSNR 20-21
    ↓
M3_1/M3_2 (Zoom 추가) → Coverage 향상 → ⭐ 현재 권장
    ↓
M3_2b/M3_3/M4 → 가설 검증 실험 중
```

---

## 관련 문서

| 문서 | 내용 |
|------|------|
| `PREPROCESSING_REGISTRY.md` | 프리셋 정의, 버그 이력 |
| `M3_SERIES_SPEC.md` | M3 계열 상세 |
| `HYPOTHESIS_VERIFICATION.md` | 가설 검증 매트릭스 |

---

*FaceLift Dataset SSOT | Updated: 2026-01-27*
