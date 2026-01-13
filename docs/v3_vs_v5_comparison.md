# v3 vs v5 전처리 코드 비교

## 1. 이미지 스케일 계산

### v3 (버그)
```python
# 거리 보정 없음!
total_image_scale = target_fx / orig_fx
# View 0: 549/1632 = 0.336
# View 1: 549/1557 = 0.353
```

**문제**: 원본 거리가 뷰마다 다른데 (246~414mm), 
스케일에 반영되지 않아 뷰마다 생쥐 크기가 다름.

### v5 (수정)
```python
# 단위 변환
current_dist_norm = current_dist_mm / UNIT_SCALE  # mm → normalized

# 거리 보정 포함!
fx_ratio = target_fx / orig_fx
dist_ratio = current_dist_norm / target_distance
total_image_scale = fx_ratio * dist_ratio
# View 0: (549/1632) * (2.46/2.7) = 0.306
# View 1: (549/1557) * (4.14/2.7) = 0.541
```

**효과**: 멀리 있던 카메라(View 1)의 이미지를 더 크게 스케일링하여 모든 뷰에서 동일한 크기.

---

## 2. Centering 방식

### v3 (Principal Point 기반)
```python
# 카메라 광학 중심 기반
scaled_cx = orig_cx * total_image_scale
scaled_cy = orig_cy * total_image_scale
center_offset_x = target_size / 2 - scaled_cx
center_offset_y = target_size / 2 - scaled_cy
```

**문제**: 생쥐 위치 무시, 꼬리로 인한 오프셋 문제.

### v5 (마스크 Centroid 기반)
```python
# 마스크 무게중심 계산
fg_coords = np.where(mask > 127)
centroid_x = fg_coords[1].mean()
centroid_y = fg_coords[0].mean()

# Centroid가 이미지 중앙에 오도록
scaled_centroid_x = centroid_x * total_image_scale
scaled_centroid_y = centroid_y * total_image_scale
center_offset_x = target_size / 2 - scaled_centroid_x
center_offset_y = target_size / 2 - scaled_centroid_y
```

**효과**: 생쥐 몸통이 이미지 중앙에 위치, 꼬리 영향 최소화.

---

## 3. 거리 정규화 (Extrinsics)

### v3
```python
distance_scale = target_distance / current_distance  # 2.7 / 246 = 0.011
new_cam_pos = cam_pos * distance_scale
```

**문제**: 단위 불일치 (mm vs normalized).
카메라가 거의 원점으로 이동됨.

### v5
```python
# 단위 고려
current_dist_mm = np.linalg.norm(cam_pos)
cam_direction = cam_pos / current_dist_mm  # 방향 벡터

# normalized unit으로 새 위치 설정
new_cam_pos = cam_direction * target_distance  # 2.7 방향으로
```

**효과**: 올바른 단위로 카메라 위치 설정.

---

## 4. 수치 비교

### 이미지 스케일
| View | orig_dist_mm | v3 scale | v5 scale | 차이 |
|------|--------------|----------|----------|------|
| 0 | 246 | 0.336 | 0.306 | -9% |
| 1 | 414 | 0.353 | 0.541 | +54% |
| 2 | 364 | 0.337 | 0.454 | +35% |
| 3 | 340 | 0.342 | 0.430 | +26% |
| 4 | 318 | 0.339 | 0.400 | +18% |
| 5 | 306 | 0.335 | 0.380 | +13% |

### 예상 생쥐 크기 (투영)
| View | v3 (불일치) | v5 (일치) |
|------|-------------|-----------|
| 0 | 크게 보임 | 정상 |
| 1 | 너무 작게 보임 | 정상 |
| 2~5 | 다양 | 정상 |

---

## 5. 함수 시그니처 변경

### v3
```python
def compute_camera_transform(
    K, R, T,
    target_distance, target_fx, target_size, orig_size
)
```

### v5
```python
def compute_camera_transform(
    K, R, T,
    mask,  # 새로 추가: centroid 계산용
    target_distance, target_fx, target_size, orig_size
)
```

---

## 6. 메타데이터 비교

### v3 _transform
```json
{
    "image_scale": 0.336,
    "target_distance": 2.7,
    "target_fx": 549
}
```

### v5 _transform
```json
{
    "image_scale": 0.306,
    "fx_ratio": 0.336,
    "dist_ratio": 0.911,
    "target_distance": 2.7,
    "target_fx": 549,
    "centroid_orig": [576.0, 512.0],
    "centroid_scaled": [176.3, 156.6],
    "center_offset": [79.7, 99.4]
}
```

**v5 추가 정보**: 디버깅 및 검증에 유용.

---

*작성일: 2025-01-12*
