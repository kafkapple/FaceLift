# 260130 Debug: Turntable Visualization Bugs

> **Date**: 2026-01-30
> **Type**: Debug / Visualization
> **Status**: ✅ 수정 완료 (260205)

---

## 이슈 요약

| # | 이슈 | 심각도 | 상태 |
|---|------|--------|------|
| A | Train/Val 입력 스트립 불일치 | Medium | ✅ 수정 |
| B | 카메라 인덱스 ↔ 이미지 불일치 | High | ✅ 수정 |
| C | Grid에 카메라 라벨 중복 | Low | ✅ 수정 |

---

## 이슈 A: Train/Val 입력 스트립 불일치

### 현상
- **Train**: 하단에 카메라 라벨 + In/Pred 구분 표시
- **Val**: 단순 입력 이미지만 표시 (라벨 없음)

### 원인
- Train: `create_labeled_input_strip()` 사용 (gslrm.py)
- Val: 단순 리사이즈 + 패딩 (validator.py)

### 해결
Validation에서도 `create_labeled_input_strip` 사용하도록 통일

---

## 이슈 B: 카메라 인덱스 ↔ 이미지 불일치

### 현상
- 영상 상단: "Cam 0 -> 4" 순서로 회전
- 하단 이미지: 라벨은 정확하나 실제 이미지가 다름

### 원인
`create_labeled_input_strip` 함수에서 카메라 순서 ↔ 실제 이미지 매핑 오류

### 해결
view_indices 파라미터로 정확한 매핑 전달

---

## 이슈 C: Grid 카메라 라벨 중복 (260205 수정)

### 현상
- Train turntable: 라벨 없음 (정상)
- Val turntable: 각 프레임에 라벨 중복 표시

### 원인
- Train: `show_overlay=cfg.get("show_frame_overlay", False)`
- Val: `show_overlay=True` **하드코딩**

### 해결 (validator.py:297)
```python
# Before
show_overlay=True, original_resolution=input_res,

# After
show_overlay=cfg.get("show_frame_overlay", False), original_resolution=input_res,
```

---

*FaceLift Debug Notes | 2026-01-30 (Updated: 2026-02-05)*
