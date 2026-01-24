# 전처리 Zoom 방식 비교 분석

> Created: 2026-01-24 | Purpose: Uniform Scale vs M3 Coverage Zoom 이론적 비교

---

## 1. 두 방식 개요

### 1.1 Uniform Scale (구버전, Archived)

```
목적: 모든 뷰에서 마우스가 동일 비율(60%)로 보이도록
방법: 뷰별로 BBox 측정 → 뷰별 개별 스케일 적용
```

**핵심 로직:**
```python
# 뷰별 개별 처리
for view in views:
    bbox = get_bbox(alpha_mask)
    current_ratio = bbox_size / image_size
    scale = target_ratio / current_ratio
    
    # Fit-in-Frame 안전 장치
    com = get_center_of_mass(mask)
    max_dist = max(distance_from_com_to_edges)
    safe_scale = (output_size/2 - margin) / max_dist
    final_scale = min(scale, safe_scale)  # 클리핑 방지
```

### 1.2 M3 Coverage Zoom (현재)

```
목적: 변환 후 FG coverage가 5%가 되도록
방법: 전체 뷰 평균 coverage → 단일 글로벌 zoom 적용
```

**핵심 로직:**
```python
# 글로벌 단일 처리
avg_coverage = mean([mask.sum()/mask.size for mask in masks])
zoom = sqrt(target_coverage / avg_coverage)
zoom = clip(zoom, 1.0, 2.5)  # 범위 제한만
# 클리핑 방지 없음
```

---

## 2. 이론적 비교

### 2.1 Multi-view Consistency

| 방식 | Cross-view 일관성 | 3D 재구성 영향 |
|------|-----------------|---------------|
| **Uniform Scale** | ❌ 뷰마다 다른 스케일 | ⚠️ 카메라 파라미터 불일치 |
| **M3 Coverage** | ✅ 단일 글로벌 zoom | ✅ 기하학적 일관성 유지 |

**분석**: Uniform Scale은 뷰마다 다른 스케일을 적용하므로, 실제 intrinsics(fx, fy)와 이미지 스케일이 불일치할 수 있음. 이는 ray error를 유발할 가능성.

### 2.2 클리핑 안전성

| 방식 | 클리핑 방지 | 최악 케이스 |
|------|----------|-----------|
| **Uniform Scale** | ✅ Fit-in-Frame | 마우스 작아짐 (29-47%) |
| **M3 Coverage** | ⚠️ zoom_range 제한만 | 마우스 잘림 가능 |

**분석**: M3의 zoom_range [1.0, 2.5]에서 2.5x zoom 시:
- 원본에서 가장자리에 있는 마우스가 프레임 밖으로 나갈 수 있음
- 특히 꼬리 등 얇은 부분이 잘릴 위험

### 2.3 기하학적 정확성

| 방식 | Ray Error | Pretrained 호환 |
|------|----------|----------------|
| **Uniform Scale** | ⚠️ 가변 (스케일마다 다름) | ❌ fx 불일치 |
| **M3 Coverage** | ✅ ~0° (homography 적용) | ✅ fx=549 정규화 |

**분석**: M3는 homography 변환 후 fx 정규화를 수행하므로 기하학적으로 정확.

### 2.4 목표 달성률

| 방식 | 목표 | 실제 달성 |
|------|------|---------|
| **Uniform Scale** | 60% BBox fill | 29-47% (안전 제한으로 감소) |
| **M3 Coverage** | 5% FG coverage | ~5% (클리핑 시 손실 가능) |

---

## 3. 위험성 분석

### 3.1 Uniform Scale 위험

1. **Multi-view 불일치**
   - 각 뷰의 스케일이 다름 → 3D 재구성 시 크기 불일치
   - 카메라 intrinsics와 이미지 스케일 mismatch

2. **카메라 파라미터 무효화**
   - 이미지만 스케일하면 fx, fy가 실제와 다르게 됨
   - Plucker ray encoding에 오류 전파

### 3.2 M3 Coverage 위험

1. **클리핑 발생 가능**
   - zoom 2.5x에서 가장자리 마우스가 잘릴 수 있음
   - 꼬리, 귀 등 extremity 손실

2. **프레임별 변동성 미고려**
   - 첫 프레임 기준으로 zoom 계산
   - 이후 프레임에서 마우스가 더 가장자리로 이동하면 클리핑

---

## 4. 권장 해결책

### Option A: M3 + Safe Zoom 제한

```python
# M3에 안전 장치 추가
coverage_zoom = sqrt(target / current)
safe_zoom = compute_safe_zoom(masks, margin=0.05)  # 클리핑 방지
final_zoom = min(coverage_zoom, safe_zoom)
```

**장점**: 기하학적 일관성 유지 + 클리핑 방지
**단점**: 목표 coverage 미달 가능

### Option B: Zoom 범위 보수적 설정

```python
zoom_range = [1.0, 1.5]  # 2.5 → 1.5로 축소
```

**장점**: 간단, 클리핑 위험 감소
**단점**: Coverage 개선 효과 제한적

### Option C: Per-frame 동적 검증

```python
for frame in frames:
    zoom = compute_zoom(frame)
    if will_clip(frame, zoom):
        zoom = compute_safe_zoom(frame)
```

**장점**: 프레임별 최적화
**단점**: 복잡성 증가, 프레임 간 zoom 변동

---

## 5. 테스트 계획

### 5.1 테스트 샘플
- 10개 프레임 (처음, 중간, 끝)
- 6개 뷰 모두

### 5.2 측정 지표
1. **클리핑 발생률**: 마스크 경계가 이미지 경계에 닿는 비율
2. **FG Coverage**: 변환 후 실제 coverage
3. **Center Offset**: 마우스 중심과 이미지 중심 거리

### 5.3 비교 조건
- A: M3 현재 설정 (zoom_range [1.0, 2.5])
- B: M3 보수적 (zoom_range [1.0, 1.5])
- C: M3 + Safe Zoom

---

*Analysis Report | 2026-01-24*
