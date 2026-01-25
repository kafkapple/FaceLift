# Preprocessing Registry (전처리 레지스트리)

> **Navigation**: [← Index](./00_INDEX.md) | [MoC](../00_MoC_INDEX.md)
> **SSOT**: 데이터셋 전처리 설정 중앙 관리
> **최종 업데이트**: 2026-01-26

---

## 1. 분류 체계

### 1.1 기하학적 변환 기준

| 카테고리 | 변환 | 특징 | 권장 |
|----------|------|------|------|
| **affine** | Affine | 회전, 스케일, 이동 | M1 (D7.1) |
| **homography** | Homography | affine + skew 보정 | M2 (D8) |
| **homography_zoom** | Homography + Zoom | + coverage 최적화 | M3_2 ⭐ |
| **geometry_broken** | centering만 | PP 미보정 | ⛔ 사용 금지 |
| **experimental** | up_alignment | 93° 회전 문제 | ⚠️ |

### 1.2 VERSION_HIERARCHY

| 분류 | 프리셋 | 설명 |
|------|--------|------|
| **affine** | D7, D7.1, D7.2 | 기본 변환 |
| **homography** | D8, D8.1, D8.2 | skew 보정 |
| **homography_zoom** | D10.3, M3_1, M3_2 | + adaptive zoom |
| **geometry_broken** | D1, D4, D6-* | ⛔ PP 미보정 |
| **experimental** | D10, D10.1, D10.2 | up_alignment 문제 |

---

## 2. M-Series (권장)

| Alias | Preset | 카테고리 | PP | fx | 상태 |
|-------|--------|----------|-----|-----|------|
| **M1** | D7.1 | affine | 256 | 549 | ✅ 기준선 |
| **M2** | D8 | homography | 256 | 549 | ✅ 정밀 |
| **M3** | D10.3 | homography_zoom | 가변 | 739 | ⚠️ H2 검증용 |
| **M3_1** | M3_1 | homography_zoom | **256** | 549 | ✅ MVG-correct |
| **M3_2** | M3_2 | homography_zoom | **256** | 549 | ✅ **권장** ⭐ |

---

## 3. 프리셋별 설정

### 3.1 affine 계열 (M1)

| Preset | transform | scale_mode | pp_method | 비고 |
|--------|-----------|------------|-----------|------|
| D7 | affine | fx_only | shift_to_256 | 기본 |
| **D7.1** | affine | individual | shift_to_256 | ✅ 권장 |
| D7.2 | affine | average | shift_to_256 | 평균 스케일 |
| D7_1_t | affine | individual | shift_to_256 | + temporal split |

### 3.2 homography 계열 (M2)

| Preset | transform | skew_correction | pp_method | 비고 |
|--------|-----------|-----------------|-----------|------|
| **D8** | homography | ✅ | shift_to_256 | ✅ 권장 |
| D8.1 | homography | ✅ | shift_to_256 | + 1.3x zoom |
| D8.2 | homography | ✅ | shift_to_256 | 특수 용도 |

### 3.3 homography_zoom 계열 (M3)

| Preset | zoom_scope | zoom_center_mode | PP 결과 | 비고 |
|--------|------------|------------------|---------|------|
| D10.3 (M3) | global | object | 가변 | H2 검증용 |
| **M3_1** | global | **image** | **256** | ✅ MVG-correct |
| **M3_2** | per_sample | **image** | **256** | ⭐ **권장** |

### 3.4 geometry_broken (⛔ 사용 금지)

| Preset | 문제점 |
|--------|--------|
| D1 | centering만, PP 미보정 → ray error |
| D4 | PP=256 강제 → 37px 오차 |
| D6-1~3 | 다양한 PP 문제 |

---

## 4. MVG-Correct Presets (2026-01-25)

### 4.1 문제 발견

기존 Object-Centered Zoom이 MVG 부정합 야기:
- M3_norm: Ray Error 13.62° (cx=197±46)
- M3_persample: Ray Error 16.15° (cx=235±32)

### 4.2 해결책: Center-Aligned Zoom

| 프리셋 | zoom_center_mode | PP 결과 | Ray Error |
|--------|------------------|---------|-----------|
| M3_1 | image | 256±0 | 0° |
| M3_2 | image | 256±0 | 0° |

### 4.3 검증

```bash
python mouse_extensions/scripts/diagnostics/verify_pp_mvg_consistency.py \
    --datasets M3_1,M3_2 --verbose

# 예상 결과:
# M3_1: PP=256.0±0.0, fx=549.0±0.0 ✓
# M3_2: PP=256.0±0.0, fx=549.0±0.0 ✓
```

---

## 5. 버그 수정 이력

### 5.1 normalize_after_zoom PP 버그 (2026-01-25)

**영향**: M3_1, M3_2 (center-aligned zoom 사용)

**문제**: `normalize_after_zoom` 활성화 시 center-aligned zoom에서도 PP가 스케일링됨 (256 → 190)

**수정**: `preprocess.py:656-663` - zoom_center_mode 조건 추가

```python
# Before (버그)
if self.normalize_after_zoom:
    cx = cx * scale
    cy = cy * scale

# After (수정)
if self.normalize_after_zoom and zoom_center_mode != "image":
    cx = cx * scale
    cy = cy * scale
```

**재전처리 필요**: M3_1, M3_2 데이터셋 삭제 후 재생성

**상세**: [[../theory/PP_FX_MVG_ANALYSIS#버그-수정]]

---

## 6. Raw Data Notes

### 6.1 Frame Discontinuity

원본 비디오에서 일부 위치에 프레임 불연속(갭)이 존재:

```python
DISCONTINUITY_FRAMES = {5900, 11800, 17700}
```

**중요**:
- 이 프레임들 자체는 **정상** (제외 불필요)
- 전체 **3,600 샘플 사용** (이전 3,597은 잘못된 제외)
- Temporal 연속성 가정 알고리즘만 주의

### 6.2 샘플 수 계산

```
Raw: 18,000 frames
frame_interval: 5
Total samples: 3,600 (전체 사용)
```

---

## 7. 전처리 명령어

### 7.1 M3_2 (권장)

```bash
cd /home/joon/dev/FaceLift

python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_2 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_2
```

### 7.2 M3_1 (Global zoom)

```bash
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_1 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_1
```

### 7.3 D7_1 (기준선)

```bash
python -m mouse_extensions.preprocessing.preprocess \
    --preset D7.1 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1
```

---

## 8. 데이터 위치

### 8.1 Raw Data

```
/home/joon/data/raw/markerless_mouse_1_nerf/
├── raw_videos/           # 6개 MP4 비디오
├── simpleclick_undist/   # 마스크 MP4 비디오
└── new_cam.pkl           # 카메라 파라미터
```

### 8.2 Preprocessed Data

```
/home/joon/data/preprocessed/FaceLift_mouse/
├── D7_1/      # M1 (기준선)
├── D8/        # M2 (homography)
├── M3_1/      # Global zoom + MVG-correct
└── M3_2/      # Per-sample zoom + MVG-correct ⭐
```

---

## 9. 관련 문서

| 문서 | 위치 |
|------|------|
| Dataset Index | [[./00_INDEX]] |
| Version Schema | [[./VERSION_SCHEMA]] |
| M3 Series Spec | [[./M3_SERIES_SPEC]] |
| Experiment Results | [[./EXPERIMENT_RESULTS]] |
| Quick Reference | [[../practical/MOUSE_QUICK_REFERENCE]] |
| PP/MVG Analysis | [[../theory/PP_FX_MVG_ANALYSIS]] |

---

*Preprocessing Registry v6.0 | 2026-01-26 | Unified from multiple sources*
