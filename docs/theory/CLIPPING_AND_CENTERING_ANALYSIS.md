# Clipping & Centering Analysis: MVG-Safe Preprocessing

## 1. 현재 클리핑 로직

### 1.1 Zoom 계산 ()

```python
def compute_persample_zoom_coverage(mask, target_coverage=0.05, zoom_range=(1.0, 2.5)):
    """
    Coverage-based zoom 계산
    - coverage = fg_pixels / total_pixels
    - zoom = sqrt(target_coverage / current_coverage)
    - coverage scales with zoom² (area relationship)
    """
    current_coverage = (mask > 0).sum() / mask.size
    zoom = np.sqrt(target_coverage / current_coverage)
    return np.clip(zoom, zoom_range[0], zoom_range[1])  # ★ Hard clipping
```

### 1.2 Zoom 적용 ()

```python
def apply_zoom(image, mask, zoom, zoom_center_mode='image'):
    crop_size = int(output_size / zoom)
    
    if zoom_center_mode == 'image':
        # ★ Center-aligned: PP=256 보존, 클리핑 위험
        crop_x = (size - crop_size) // 2
        crop_y = (size - crop_size) // 2
    else:
        # Object-centered: PP 가변, 클리핑 없음
        cx, cy = object_center(mask)
        crop_x = cx - crop_size/2
        crop_y = cy - crop_size/2
```

### 1.3 현재 문제점

| Mode | PP | 클리핑 위험 | MVG 정확성 |
|------|-----|------------|-----------|
| **center-aligned** | 256 (고정) | ★ **높음** | ✅ 정확 |
| **object-centered** | 가변 | 없음 | ⚠️ PP 업데이트 필요 |

---

## 2. 클리핑 발생 조건

### 2.1 수학적 분석

Center-aligned zoom 시 클리핑 조건:

```
이미지 크기: S = 512
Crop 크기: C = S / zoom
Crop 시작: (S-C)/2 ~ (S+C)/2

Object bbox: [x_min, x_max] × [y_min, y_max]

클리핑 발생 조건:
  x_min < (S-C)/2  또는  x_max > (S+C)/2
  y_min < (S-C)/2  또는  y_max > (S+C)/2
```

### 2.2 안전 zoom 계산

클리핑 없는 최대 zoom:

```python
def compute_safe_zoom(mask, output_size=512):
    """클리핑 없는 최대 zoom 계산"""
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return 1.0
    
    # Object bbox
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    
    # Center로부터의 최대 거리
    center = output_size / 2
    max_dist_x = max(center - x_min, x_max - center)
    max_dist_y = max(center - y_min, y_max - center)
    max_dist = max(max_dist_x, max_dist_y)
    
    # 안전 zoom: crop이 object를 포함하도록
    # crop_half = output_size / (2 * zoom)
    # 조건: crop_half >= max_dist
    # => zoom <= output_size / (2 * max_dist)
    
    safe_zoom = output_size / (2 * max_dist) if max_dist > 0 else float('inf')
    return min(safe_zoom, 2.5)  # 상한
```

---

## 3. 해결 방안 비교

### 3.1 Option A: Dynamic Safe Zoom

**아이디어**: 클리핑 발생 시 zoom을 safe_zoom으로 제한

```python
def compute_clipping_safe_zoom(mask, target_coverage=0.05, zoom_range=(1.0, 2.5)):
    # 1. Target coverage 기반 이상적 zoom
    ideal_zoom = compute_coverage_zoom(mask, target_coverage)
    
    # 2. 클리핑 없는 최대 zoom
    safe_zoom = compute_safe_zoom(mask)
    
    # 3. 둘 중 작은 값 선택
    final_zoom = min(ideal_zoom, safe_zoom)
    return np.clip(final_zoom, zoom_range[0], zoom_range[1])
```

**장점**: 샘플별 최적화, 클리핑 0%
**단점**: 가변 zoom/fx → 배치 내 불균일

### 3.2 Option B: Zoom Range 축소

**아이디어**: 보수적 zoom_range로 통계적 클리핑 방지

| zoom_range | 예상 클리핑 | Coverage |
|------------|------------|----------|
| [1.0, 2.5] | ~6.5% | 높음 |
| [1.0, 1.8] | ~0.5% | 중간 |
| **[1.0, 1.5]** | ~0% | 낮음 |

**장점**: 단순, 일관성
**단점**: 모든 샘플에 비효율적 zoom

### 3.3 Option C: Object-Centered + PP Correction (★ 권장)

**핵심 통찰**: Object-centered crop은 PP를 올바르게 업데이트하면 **MVG-correct**

```
Original:
- Image center: (256, 256)
- Object center: (obj_cx, obj_cy)
- Principal point: (cx, cy)

After object-centered crop:
- Crop offset: (offset_x, offset_y) = (obj_cx - 256, obj_cy - 256)
- New PP: (cx - offset_x, cy - offset_y)
```

**수학적 증명**:

3D 점 P → 2D 점 p의 projection:
```
p = K @ [R|T] @ P
p_u = fx * X/Z + cx
p_v = fy * Y/Z + cy
```

Crop 후 (offset으로 이동):
```
p'_u = p_u - offset_x = fx * X/Z + (cx - offset_x)
p'_v = p_v - offset_y = fy * Y/Z + (cy - offset_y)
```

새로운 intrinsics:
```
cx' = cx - offset_x
cy' = cy - offset_y
```

**결론**: PP를 crop offset만큼 이동하면 ray direction 보존 ✓

---

## 4. 이전 방식의 문제점

### D1, D4, D6 실패 원인

```
1. Object-centered crop 수행 (올바름)
2. PP를 256으로 강제 (❌ 잘못됨!)
   - 실제 PP: (cx - offset_x, cy - offset_y) ≠ 256
   - 강제 PP=256 → ray direction ~13° 오차
3. 결과: Multi-view inconsistency → Ghosting
```

### M3_1, M3_2의 접근 (회피)

```
1. Center-aligned crop (PP=256 자동 보존)
2. Object가 edge에 있으면 클리핑 발생
3. zoom_range 제한으로 클리핑 최소화
```

---

## 5. 제안: M4 Preprocessing

### 5.1 알고리즘

```python
def preprocess_m4(images, masks, cameras):
    """Object-centered + MVG-correct preprocessing"""
    
    # 1. 3D center triangulation (모든 뷰에서 일관된 기준점)
    center_3d = triangulate_center(masks, cameras)
    
    for view_idx, (img, mask, cam) in enumerate(zip(images, masks, cameras)):
        # 2. 3D center를 2D로 projection
        center_2d = project_point(center_3d, cam)
        
        # 3. Object-centered crop
        offset_x = center_2d[0] - 256
        offset_y = center_2d[1] - 256
        cropped_img = crop_centered(img, center_2d)
        
        # 4. ★ PP correction (MVG 보존 핵심)
        cam['cx'] = cam['cx'] - offset_x
        cam['cy'] = cam['cy'] - offset_y
        
        # 5. Zoom 적용 (object 중심이므로 클리핑 없음)
        zoom = compute_coverage_zoom(mask, target=0.05)
        cam['fx'] *= zoom
        cam['fy'] *= zoom
        cam['cx'] *= zoom  # zoom도 반영
        cam['cy'] *= zoom
        
        # 6. Post-zoom normalization (fx=549로 정규화)
        if normalize_after_zoom:
            scale = 549 / cam['fx']
            cam['fx'] = 549
            cam['fy'] *= scale
            cam['cx'] *= scale
            cam['cy'] *= scale
```

### 5.2 특성 비교

| 방식 | PP | fx | 클리핑 | MVG | GS-LRM 호환 |
|------|-----|-----|--------|-----|------------|
| M3_1 | 256 | 549 | 0% | ✅ | ✅ (pretrained 일치) |
| M3_2 | 256 | 549 | 0.5% | ✅ | ✅ |
| **M4** | **가변** | 549 | **0%** | ✅ | ⚠️ (적응 필요) |

### 5.3 GS-LRM 호환성

```
GS-LRM은 fxfycxcy 텐서를 데이터에서 읽어 사용 (hardcode 아님)
- Pretrained: cx=cy=256 분포
- M4: cx, cy 가변 (뷰/샘플마다 다름)

예상:
- 초기 학습 불안정 가능 (distribution shift)
- Fine-tuning 시 적응 예상
- LR warmup / longer training 권장
```

---

## 6. 권장 사항

### 6.1 단기 (현재 실험)

**M3_1 사용** (클리핑 0%, pretrained 일치)

### 6.2 중기 (품질 개선)

**Option A: Dynamic Safe Zoom 구현**
- M3_2에 safe_zoom 로직 추가
- 클리핑 0% + 최대 coverage

### 6.3 장기 (근본 해결)

**M4 구현**
- Object-centered + PP correction
- 클리핑 0% + 최대 coverage + MVG 정확

---

## 7. 구현 우선순위

| 우선순위 | 작업 | 효과 |
|---------|------|------|
| P0 | M3_2에 safe_zoom 추가 | 클리핑 0% |
| P1 | M4 prototype | MVG + 최적 coverage |
| P2 | M4 검증 실험 | GS-LRM 호환성 확인 |

---

*Created: 2026-01-26 | FaceLift Mouse Preprocessing*
