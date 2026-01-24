# Preprocessing Registry (전처리 레지스트리)

> **SSOT**: 데이터셋 전처리 설정 중앙 관리
> **최종 업데이트**: 2026-01-24

---

## 분류 체계

### 기하학적 변환 기준

| 카테고리 | 변환 | 특징 | 권장 |
|----------|------|------|------|
| **affine** | Affine | 회전, 스케일, 이동 | M1 (D7.1) |
| **homography** | Homography | affine + skew 보정 | M2 (D8) |
| **homography_zoom** | Homography + Zoom | + coverage 최적화 | M3 (D10.3) ⭐ |

### 기타

| 카테고리 | 설명 | 상태 |
|----------|------|------|
| **experimental** | up_alignment 시도 (93도 회전 문제) | ⚠️ |
| **native** | 변환 없음, 원본 유지 | 특수 용도 |
| **geometry_broken** | centering만, PP 미보정 | ❌ 사용 금지 |

---

## M-Series (권장)

| Alias | Preset | 카테고리 | 샘플 수 | 경로 |
|-------|--------|----------|---------|------|
| **M1** | D7.1 | affine | ~1600 | `/home/joon/data/preprocessed/FaceLift_mouse/D7_1/` |
| **M2** | D8 | homography | ~1600 | `/home/joon/data/preprocessed/FaceLift_mouse/D8/` |
| **M3** | D10.3 | homography_zoom | 3597 | `/home/joon/data/preprocessed/FaceLift_mouse/M3/` |

### M3 (D10.3) 상세

| 항목 | 값 |
|------|-----|
| Transform | homography (skew 보정) |
| Coverage | adaptive zoom (목표 5%, 실제 ~9.7%) |
| Zoom Factor | 1.35x |
| Split | Train 3238, Val 359 (9:1) |
| up_alignment | **False** (문제 해결됨) |

---

## 프리셋별 설정

### affine 계열 (M1)

| Preset | transform | scale_mode | pp_method | 비고 |
|--------|-----------|------------|-----------|------|
| D7 | affine | fx_only | shift_to_256 | 기본 |
| **D7.1** | affine | individual | shift_to_256 | ✅ 권장 |
| D7.2 | affine | average | shift_to_256 | 평균 스케일 |

### homography 계열 (M2)

| Preset | transform | skew_correction | pp_method | 비고 |
|--------|-----------|-----------------|-----------|------|
| **D8** | homography | ✅ | shift_to_256 | ✅ 권장 |
| D8.1 | homography | ✅ | shift_to_256 | + 1.3x zoom |
| D8.2 | homography | ✅ | shift_to_256 | 특수 용도 |

### homography_zoom 계열 (M3)

| Preset | transform | adaptive_zoom | zoom_method | 비고 |
|--------|-----------|---------------|-------------|------|
| **D10.3** | homography | ✅ | coverage_based | ⭐ Production |

### geometry_broken (❌ 사용 금지)

| Preset | 문제점 |
|--------|--------|
| D1 | centering만, PP 미보정 → ray error |
| D4 | PP=256 강제 → 37px 오차 |
| D6-1~3 | 다양한 PP 문제 |

---

## 전처리 명령어

### M1 (D7.1)
```bash
python -m mouse_extensions.preprocessing.preprocess \
    --preset D7.1 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1
```

### M2 (D8)
```bash
python -m mouse_extensions.preprocessing.preprocess \
    --preset D8 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D8
```

### M3 (D10.3)
```bash
python -m mouse_extensions.preprocessing.preprocess \
    --preset D10.3 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3
```

---

## 관련 문서

| 문서 | 위치 |
|------|------|
| 실험 레지스트리 | [`../EXPERIMENT_REGISTRY.md`](../EXPERIMENT_REGISTRY.md) |
| Quick Reference | [`../MOUSE_QUICK_REFERENCE.md`](../MOUSE_QUICK_REFERENCE.md) |
| 전처리 분석 | [`../../analysis/preprocessing/`](../../analysis/preprocessing/) |

---

*Preprocessing Registry v5.0 | 2026-01-24*
