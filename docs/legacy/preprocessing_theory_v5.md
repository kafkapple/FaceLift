# FaceLift Mouse 전처리 이론 및 수식 (v5)

## 1. 투영 기본 공식

### 1.1 Pinhole Camera Model
```
pixel_x = fx * (X / Z) + cx
pixel_y = fy * (Y / Z) + cy
```

여기서:
- `(X, Y, Z)`: 카메라 좌표계에서의 3D 점
- `(fx, fy)`: 초점 거리 (pixels)
- `(cx, cy)`: 주점 (principal point)
- `Z`: 카메라에서 물체까지 거리

### 1.2 물체 크기와 투영 관계
```
projected_size = fx * (object_size / distance)
```

**핵심**: 같은 물체가 모든 뷰에서 **동일한 크기**로 투영되려면,
`fx / distance` 비율이 모든 뷰에서 동일해야 함.

---

## 2. 원본 데이터 분석

### 2.1 원본 카메라 파라미터 (markerless_mouse_1_nerf)

| View | fx (px) | fy (px) | distance (mm) | fx/dist |
|------|---------|---------|---------------|---------|
| 0 | 1632 | 1639 | 246 | 6.63 |
| 1 | 1557 | 1580 | 414 | 3.76 |
| 2 | 1630 | 1633 | 364 | 4.48 |
| 3 | 1606 | 1618 | 340 | 4.72 |
| 4 | 1618 | 1629 | 318 | 5.09 |
| 5 | 1637 | 1642 | 306 | 5.35 |

**관찰**:
- `fx/dist` 비율이 뷰마다 다름 (3.76 ~ 6.63)
- View 0에서 생쥐가 가장 크게 보임 (가까움)
- View 1에서 생쥐가 가장 작게 보임 (멂)

### 2.2 FaceLift 타겟 설정
```
target_fx = 549
target_fy = 549  (정사각 픽셀)
target_distance = 2.7
target_fx/dist = 549/2.7 = 203.3
```

---

## 3. 전처리 변환 수식

### 3.1 단위 변환
원본 거리는 mm 단위, FaceLift는 normalized unit 사용.

```python
UNIT_SCALE = 100.0
orig_dist_norm = orig_dist_mm / UNIT_SCALE
# View 0: 246mm → 2.46
# View 1: 414mm → 4.14
```

### 3.2 이미지 스케일 계산 (핵심!)

**목표**: 변환 후 모든 뷰에서 동일한 투영 크기

**수식 유도**:
```
원본 투영: pixel_orig = orig_fx * (X / orig_dist_norm)
타겟 투영: pixel_target = target_fx * (X / target_dist)

변환: pixel_target = pixel_orig * scale

따라서:
target_fx * (X / target_dist) = orig_fx * (X / orig_dist_norm) * scale

scale = (target_fx / orig_fx) * (orig_dist_norm / target_dist)
      = (target_fx * orig_dist_norm) / (orig_fx * target_dist)
```

**최종 수식**:
```python
image_scale = (target_fx / orig_fx) * (orig_dist_norm / target_distance)
```

### 3.3 뷰별 스케일 계산 예시

| View | orig_fx | orig_dist_norm | target_fx | target_dist | scale |
|------|---------|----------------|-----------|-------------|-------|
| 0 | 1632 | 2.46 | 549 | 2.7 | 0.306 |
| 1 | 1557 | 4.14 | 549 | 2.7 | 0.541 |
| 2 | 1630 | 3.64 | 549 | 2.7 | 0.454 |
| 3 | 1606 | 3.40 | 549 | 2.7 | 0.430 |
| 4 | 1618 | 3.18 | 549 | 2.7 | 0.400 |
| 5 | 1637 | 3.06 | 549 | 2.7 | 0.380 |

**검증**: 스케일 적용 후 fx/dist 비율
```
View 0: 549 / 2.7 = 203.3
View 1: 549 / 2.7 = 203.3
...
모든 뷰에서 동일! ✓
```

---

## 4. Centering 방식

### 4.1 기존 방식 (v3): Principal Point 기반
```python
center_offset_x = target_size/2 - orig_cx * scale
center_offset_y = target_size/2 - orig_cy * scale
```
**문제**: 생쥐 위치 무시, 꼬리로 인한 오프셋

### 4.2 새 방식 (v5): 마스크 Centroid 기반
```python
# 마스크에서 foreground 픽셀의 무게중심 계산
mask_coords = np.where(mask > 127)
centroid_y = mask_coords[0].mean()
centroid_x = mask_coords[1].mean()

# 스케일링 후 centroid가 이미지 중앙에 오도록
scaled_centroid_x = centroid_x * scale
scaled_centroid_y = centroid_y * scale
center_offset_x = target_size/2 - scaled_centroid_x
center_offset_y = target_size/2 - scaled_centroid_y
```

**장점**:
- 생쥐 몸통 중심이 이미지 중앙에 위치
- 꼬리의 영향 최소화 (무게중심은 몸통 쪽으로 치우침)

---

## 5. fy/fx 비율 처리

### 5.1 원본 비율
```
View 0: fy/fx = 1.0043 (+0.43%)
View 1: fy/fx = 1.0152 (+1.52%)  ← 가장 큰 차이
...
평균: 1.0064 (+0.64%)
```

### 5.2 v5 방식: fy = fx (정사각 픽셀)
```python
target_fx = 549
target_fy = 549  # fy = fx
```

**이유**:
- FaceLift pre-trained 모델이 동일 intrinsics 가정
- 뷰간 일관성 확보
- 1.5% 기하학적 오차는 감수

---

## 6. 카메라 extrinsics 변환

### 6.1 거리 정규화
```python
# 원본 카메라 위치
c2w = np.linalg.inv(w2c)
cam_pos = c2w[:3, 3]
current_dist_mm = np.linalg.norm(cam_pos)
current_dist_norm = current_dist_mm / UNIT_SCALE

# 거리 스케일
distance_scale = target_distance / current_dist_norm

# 새 카메라 위치 (방향 유지, 거리만 변경)
new_cam_pos = cam_pos * (distance_scale / (UNIT_SCALE if already_mm else 1))
```

### 6.2 최종 w2c 매트릭스
```python
new_c2w = c2w.copy()
new_c2w[:3, 3] = new_cam_pos / UNIT_SCALE  # mm → normalized
new_w2c = np.linalg.inv(new_c2w)
```

---

## 7. v3 vs v5 비교

| 항목 | v3 (버그) | v5 (수정) |
|------|-----------|-----------|
| 이미지 스케일 | `target_fx / orig_fx` | `(target_fx / orig_fx) * (orig_dist_norm / target_dist)` |
| 거리 보정 | ❌ 없음 | ✅ 포함 |
| Centering | Principal point | 마스크 Centroid |
| fy/fx | 1.0 (OK) | 1.0 (OK) |
| 뷰별 크기 | 불일치 (최대 54%) | 일치 |

---

## 8. 검증 체크리스트

1. **스케일 검증**:
   - [ ] 변환 후 fx/dist 비율이 모든 뷰에서 동일한가?
   - [ ] 뷰별 마스크 면적이 비슷한가? (각도 차이 제외)

2. **Centering 검증**:
   - [ ] 마스크 centroid가 이미지 중앙 근처인가?
   - [ ] 뷰간 centroid 위치 편차가 작은가?

3. **카메라 검증**:
   - [ ] distance = 2.7 (모든 뷰)
   - [ ] fx = fy = 549 (모든 뷰)

4. **시각적 검증**:
   - [ ] 6개 뷰 이미지 나란히 비교
   - [ ] 생쥐 크기가 비슷해 보이는가?

---

*작성일: 2025-01-12*
*버전: v5 (거리 보정 + 마스크 centroid)*

---

## 9. 중요 업데이트: v5 cx,cy 오류 발견 (2026-01-13)

### 9.1 발견된 문제

v5의 centroid centering에서 **critical bug** 발견:

```python
# v5 버그 코드
offset = 256 - scaled_centroid
shifted_image = shift(image, offset)
cx = cy = 256  # ← 잘못! shift 후에도 256 고정
```

**문제**: Image shift 후 actual principal point는 256이 아님

### 9.2 오류 크기

| View | 실제 cx | 주장 cx | 오류 | Ray 오류 |
|------|---------|---------|------|----------|
| 0 | 192 | 256 | 64px | 6.6° |
| 1 | **415** | 256 | **159px** | **16.2°** |
| 2 | 365 | 256 | 109px | 11.2° |

### 9.3 해결책: v10, v11

| 버전 | 방식 | cx,cy | 권장 |
|------|------|-------|------|
| v5 | Object centered, cx=256 고정 | **오류** | ✗ Deprecated |
| v10 | Object centered, cx 보정 | 정확 (변동) | △ |
| **v11** | **PP centered** | **정확 (256)** | **✓ 권장** |

### 9.4 v11 핵심 로직

```python
# v11: Principal Point centered
offset = 256 - scaled_cx  # PP 기준 centering
shifted_image = shift(image, offset)
cx = cy = 256  # 수학적으로 정확히 256
```

**결과**: Object는 중앙 아니지만, cx=cy=256 정확+일관

---

*업데이트: 2026-01-13*
*v5는 deprecated, v11 사용 권장*
