# 카메라 투영 모델 (Camera Projection)

> FaceLift 전처리 이해를 위한 카메라 모델 기초 이론

## 1. Pinhole Camera Model

### 기본 투영 공식

카메라 좌표계에서 3D 점 `(X, Y, Z)`를 이미지 좌표 `(u, v)`로 투영:

```
u = fx * (X / Z) + cx
v = fy * (Y / Z) + cy
```

### 파라미터 설명

| 파라미터 | 설명 | 단위 |
|----------|------|------|
| `fx, fy` | 초점 거리 (Focal Length) | pixels |
| `cx, cy` | 주점 (Principal Point) | pixels |
| `X, Y, Z` | 카메라 좌표계 3D 점 | 임의 단위 |
| `u, v` | 이미지 좌표 | pixels |

### Intrinsic Matrix (K)

```
K = | fx   0  cx |
    |  0  fy  cy |
    |  0   0   1 |
```

---

## 2. 투영 크기 관계

### 물체 크기와 이미지 크기

```
projected_size = fx * (object_size / distance)
```

**의미**: 
- `fx` 클수록 → 이미지에서 크게 보임
- `distance` 멀수록 → 이미지에서 작게 보임

### fx / distance 비율의 중요성

모든 뷰에서 **동일한 물체 크기**로 보이려면:

```
fx / distance = constant (모든 뷰에서)
```

**FaceLift 기준값:**
```
fx / distance = 549 / 2.7 = 203.3
```

---

## 3. FaceLift 카메라 설정

### Pretrained Model 가정

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| fx = fy | 549 | 정사각 픽셀 |
| cx = cy | 256 | 512x512 이미지 중앙 |
| distance | 2.7 | 카메라-원점 거리 |
| image_size | 512 | 픽셀 |

### 좌표계

```
      Z (위)
      |
      |
      +---- Y (오른쪽)
     /
    /
   X (앞)
```

**Z-up 좌표계**: Z축이 위쪽 방향

---

## 4. Extrinsic Matrix

### World-to-Camera (w2c)

```
w2c = | R  t |  (4x4)
      | 0  1 |

R: 3x3 회전 행렬
t: 3x1 이동 벡터
```

### Camera-to-World (c2w)

```
c2w = inverse(w2c)

cam_position = c2w[:3, 3]  # 카메라 위치
cam_direction = c2w[:3, 2]  # 카메라 방향 (Z축)
```

### 카메라 거리 계산

```python
c2w = np.linalg.inv(w2c)
cam_pos = c2w[:3, 3]
distance = np.linalg.norm(cam_pos)  # 원점에서 카메라까지
```

---

## 5. 전처리 수식 유도

### 목표

원본 카메라 설정을 FaceLift 기대값으로 변환

### 이미지 스케일 공식

```
원본: pixel_orig = orig_fx * (X / orig_dist)
타겟: pixel_target = target_fx * (X / target_dist)

변환: pixel_target = pixel_orig * scale

따라서:
scale = (target_fx / orig_fx) * (orig_dist / target_dist)
```

### 예시 계산

| 뷰 | orig_fx | orig_dist | scale | 결과 |
|----|---------|-----------|-------|------|
| 0 | 1632 | 2.46 | 0.306 | 축소 |
| 1 | 1557 | 4.14 | 0.541 | 중간 |
| 2 | 1630 | 3.64 | 0.454 | 중간 |

**변환 후**: 모든 뷰에서 `fx/dist = 549/2.7 = 203.3`

---

## 6. Principal Point Centering

### 문제

원본 이미지의 주점(cx, cy)이 뷰마다 다름

### PP-Centered 방식 (권장)

```python
# 스케일링 후 주점 위치
scaled_cx = orig_cx * scale
scaled_cy = orig_cy * scale

# 이미지 중앙으로 이동
offset_x = 256 - scaled_cx
offset_y = 256 - scaled_cy

# 이미지 shift
shifted_image = shift(scaled_image, (offset_y, offset_x))

# 결과: cx = cy = 256 (정확히 중앙)
```

### Why PP-Centered?

1. Ray direction 계산이 기하학적으로 정확
2. 모든 뷰에서 일관된 cx, cy
3. 모델이 기대하는 설정과 일치

---

## 7. Plücker 좌표

### 정의

3D 공간의 광선(ray)을 표현하는 6차원 좌표

```
Plücker = (d, m)

d: Ray direction (3D 단위 벡터)
m: Moment = origin × direction (3D 벡터)
```

### FaceLift에서의 사용

```python
# 각 픽셀에 대한 ray 계산
ray_origin = cam_position
ray_direction = normalize(pixel_3d - cam_position)

# Plücker 좌표
moment = cross(ray_origin, ray_direction)
plucker = concat(ray_direction, moment)  # [6]
```

### Why 고정 뷰 순서?

뷰 순서가 바뀌면:
- Ray direction이 달라짐
- Plücker 좌표가 달라짐
- 모델이 일관된 3D 관계를 학습하기 어려움

**해결**: 항상 동일한 순서 [0,1,2,3,4,5] 사용

---

## 8. 참고 자료

- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Plücker Coordinates](https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates)
- [GS-LRM Paper](https://sai-bi.github.io/project/gs-lrm/)

---

*Created: 2026-01-13*
