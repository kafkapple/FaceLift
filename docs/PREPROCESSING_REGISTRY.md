# FaceLift Mouse Preprocessing Registry

> **Version**: v4.0 (2026-01-22)
> **Single Source of Truth** for all preprocessing configurations

---

## 전처리 패러다임 개요

| 패러다임 | 프리셋 | PP 처리 | 이미지 변환 | Ray Error | 상태 |
|----------|--------|---------|-------------|-----------|------|
| **pp_centered_shift** | D7, D7.1, D7.2 | 256으로 shift | 이미지도 shift | ~0° | ⭐ 권장 |
| geometry_preserving | D6-1, D6-2, D6-3 | 정확한 값 유지 | 최소 변환 | 0° | 실험적 |
| precision_homography | D8, D8.1 | 256으로 shift | Homography+skew | ~0° | 실험적 |
| native | D9, D9_norm | 원본 유지 | 없음 | 0° | 고해상도용 |
| object_centered | D1-D4 | 256 강제 | Crop | 5-13° | ❌ DEPRECATED |

---

## 상세 프리셋 비교

### 이미지 처리

| 프리셋 | 입력 해상도 | 출력 해상도 | Crop | Shift | 파일 크기 |
|--------|-------------|-------------|------|-------|-----------|
| **D7.1** | 1152×1024 | **512×512** | ❌ | ✅ | ~22% |
| D6-2 | 1152×1024 | **512×512** | ❌ | ❌ | ~22% |
| D4 | 1152×1024 | 512×512 | ✅ | - | ~22% |
| D9 | 1152×1024 | **1152×1024** | ❌ | ❌ | 100% |
| D9_resized | 1152×1024 | **512×512** | ❌ | ✅ | ~22% |

### 카메라 파라미터

| 프리셋 | fx | fy | cx | cy | translation |
|--------|-----|-----|-----|-----|-------------|
| **D7.1** | **549** | **549** | **256** | **256** | **2.7** |
| D6-2 | 549 | 549 | **가변** | **가변** | 2.7 |
| D4 | 원본 | 원본 | **256 (강제)** | **256 (강제)** | 원본 |
| D9 | 1632 | 1607 | 576 | 512 | 246 |
| D9_norm | 1632 | 1607 | 576 | 512 | **2.7** |
| D9_resized | **549** | **549** | **256** | **256** | **2.7** |

### Pretrained 호환성

| 프리셋 | fx 일치 | trans 일치 | 호환성 | 권장 용도 |
|--------|---------|------------|--------|-----------|
| **D7.1** | ✅ 549 | ✅ 2.7 | **완벽** | 일반 학습 |
| D6-2 | ✅ 549 | ✅ 2.7 | 좋음 | 기하학 연구 |
| D4 | ❌ | ❌ | **불가** | DEPRECATED |
| D9 | ❌ 1632 | ❌ 246 | **불가** | - |
| D9_norm | ❌ 1632 | ✅ 2.7 | 실험적 | 고해상도 연구 |
| D9_resized | ✅ 549 | ✅ 2.7 | **완벽** | D7.1과 동일 |

---

## D7.1 vs D9_resized

**결론: 사실상 동일**

| 항목 | D7.1 | D9_resized |
|------|------|------------|
| 패러다임 | pp_centered_shift | pp_centered_shift |
| 출력 해상도 | 512×512 | 512×512 |
| fx, fy | 549, 549 | 549, 549 |
| cx, cy | 256, 256 | 256, 256 |
| translation | 2.7 | 2.7 |
| **차이점** | - | 없음 |

→ **D9_resized는 불필요**. D7.1 사용 권장.

---

## D6-2 vs D7.1 핵심 차이

| 항목 | D6-2 | D7.1 |
|------|------|------|
| **PP 처리** | 가상 shift (이미지 그대로) | 실제 shift (이미지 이동) |
| **cx, cy 값** | 실제 값 유지 (가변) | 256 고정 |
| **이미지 품질** | 100% 보존 | 가장자리 ~50px 손실 가능 |
| **기하학 정확도** | 완벽 (0°) | 거의 완벽 (~0°) |
| **Pretrained 호환** | 좋음 | 완벽 |

---

## GPU 메모리 요구사항

| 프리셋 | 해상도 | 픽셀 수 | 상대 메모리 |
|--------|--------|---------|-------------|
| D7.1/D6-2/D9_resized | 512×512 | 262K | **1x** (~16GB) |
| D9/D9_norm | 1152×1024 | 1.18M | **4.5x** (~72GB) |

---

## 권장 프리셋

| 목적 | 권장 프리셋 | 이유 |
|------|-------------|------|
| **일반 학습** | **D7.1** | Pretrained 완벽 호환, 검증됨 |
| 기하학 연구 | D6-2 | 정확한 PP, 이미지 품질 보존 |
| 고해상도 실험 | D9_norm | 원본 해상도 (메모리 4.5x 필요) |

---

## 전처리 명령어

### D7.1 (권장)
```bash
python -m mouse_extensions.preprocessing.preprocess \
    --preset D7.1 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1
```

### D6-2 (기하학 정확)
```bash
python -m mouse_extensions.preprocessing.preprocess \
    --preset D6-2 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D6-2
```

### D9 (원본 해상도)
```bash
python -m mouse_extensions.preprocessing.preprocess \
    --preset D9 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D9
```

---

## 데이터 위치

### 전처리 완료
```
/home/joon/data/preprocessed/FaceLift_mouse/
├── D7_1/      # 3238 train, 359 val (권장)
├── D9/        # 3238 train, 359 val (고해상도, 미정규화)
└── ...
```

### Raw 데이터
```
/home/joon/data/raw/markerless_mouse_1_nerf/
├── raw_videos/           # 6개 MP4
├── simpleclick_undist/   # 마스크 MP4
└── new_cam.pkl           # 카메라 파라미터
```

---

## DEPRECATED 프리셋

| 프리셋 | 문제점 |
|--------|--------|
| D1-D4 | PP=256 강제 → Ray error 5-13° |
| D5 | 실험적, 미완성 |

---

## Changelog

### v4.0 (2026-01-22)
- D9, D9_norm, D9_resized 추가
- 상세 프리셋 비교표 추가
- D7.1 vs D9_resized 동일성 확인

### v3.0 (2026-01-20)
- D7.1, D7.2 추가
- 모듈화 config 시스템

---

*Last updated: 2026-01-22*
