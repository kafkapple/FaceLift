---
date: 2026-01-04
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, preprocessing, scaling, clipping, centering]
project: Mouse-FaceLift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# 2026-01-04 연구 일지: 전처리 종합 분석 (Scaling & Clipping)

> **통합 문서**: 이전 2개 문서를 핵심 내용 중심으로 통합
> - `260104_mouse_data_clipping_analysis.md` (삭제됨)
> - `260104_preprocessing_scaling_analysis.md` (삭제됨)

---

## 1. 문제 요약

### 1.1 뷰별 객체 크기 불일치
`data_mouse_uniform` 전처리 후에도 뷰별 pixel ratio **2.4x 차이**:
- 최소: 8.8% (위에서 본 뷰)
- 최대: 25% (측면 뷰)

### 1.2 이미지 Clipping 문제
- 300개 뷰 중 **290개(97%)** 에서 clipping 발생
- 238개(79%)가 severe (200+ pixels at edge)

---

## 2. 원인 분석

### 2.1 뷰별 크기 불일치 원인

1. **3D 투영의 본질**: 마우스의 비대칭 형태(긴 꼬리) → 시점에 따라 다른 2D 투영 크기
2. **버그**: 뷰마다 다른 `safe_scale` 적용하면서 camera intrinsics 미업데이트

### 2.2 Clipping 원인

**현재 스케일링 로직**:
```python
# 이미지 기하학적 중심 기준 스케일링
scaled_center_x = new_w / 2
offset_x = int(output_size / 2 - scaled_center_x)
```

**문제점**: 객체의 **Center of Mass(CoM)가 이미지 중심에서 벗어나면**, 스케일 확대 시 반대쪽이 프레임 밖으로 나감

---

## 3. Scaling 전략 비교

### Option A: 뷰별 균일 Pixel Ratio
각 뷰를 독립적으로 스케일링하여 모든 뷰가 동일한 pixel ratio(60%) 달성

| 항목 | 내용 |
|------|------|
| 장점 | 모든 뷰에서 일관된 시각적 크기, attention 균등 분배 |
| **단점** | 뷰마다 다른 effective focal length, **Plücker 좌표 불균일**, pretrained 분포 벗어남 |

### Option B: 샘플별 균일 Scale (권장 ⭐)
샘플 내 모든 뷰에 **동일한 scale factor** 적용

| 항목 | 내용 |
|------|------|
| **장점** | 3D 일관성 유지, pretrained 분포 내, 카메라 파라미터 간단 |
| 단점 | 뷰별 시각적 크기 차이 존재 |

### 결론: **Option B (샘플별 균일 scale)** 권장
- 뷰별 크기 차이는 **3D 형상의 자연스러운 특성**
- GS-LRM은 3D 일관성을 더 중요시

---

## 4. Clipping 해결 방안

### 4.1 CoM 기반 스케일링 (권장)

```python
def scale_image_com_based(image, alpha, scale_factor, output_size):
    """Center of Mass 기준 스케일링"""
    # 1. 객체의 Center of Mass 계산
    y_coords, x_coords = np.where(alpha > 0.5)
    com_y = np.mean(y_coords)
    com_x = np.mean(x_coords)

    # 2. CoM이 출력 이미지 중앙에 오도록 offset 계산
    scaled_com_x = com_x * scale_factor
    scaled_com_y = com_y * scale_factor
    offset_x = output_size / 2 - scaled_com_x
    offset_y = output_size / 2 - scaled_com_y

    # 3. Affine transform 적용
    ...
```

### 4.2 Safe Scale 계산 개선

```python
def compute_safe_scale(alpha, output_size, target_ratio):
    """Clipping 방지 safe scale 계산"""
    # 현재 객체의 bounding box
    y_coords, x_coords = np.where(alpha > 0.5)
    bbox_w = x_coords.max() - x_coords.min()
    bbox_h = y_coords.max() - y_coords.min()

    # CoM에서 bbox 경계까지 최대 거리
    com_x, com_y = np.mean(x_coords), np.mean(y_coords)
    max_dist_x = max(com_x - x_coords.min(), x_coords.max() - com_x)
    max_dist_y = max(com_y - y_coords.min(), y_coords.max() - com_y)

    # 스케일 후에도 이미지 내에 있도록 제한
    safe_scale_x = (output_size / 2) / max_dist_x
    safe_scale_y = (output_size / 2) / max_dist_y

    return min(safe_scale_x, safe_scale_y, target_scale)
```

---

## 5. 권장 전처리 파이프라인

```bash
# Step 1: CoM 기반 Centering + Scaling
python scripts/preprocess_com_based.py \
    --input_dir data_mouse_raw \
    --output_dir data_mouse_preprocessed \
    --target_ratio 0.6 \
    --scale_mode per_sample  # 샘플별 균일 scale

# Step 2: 카메라 거리 정규화
python scripts/normalize_cameras_to_facelift.py \
    --input_dir data_mouse_preprocessed \
    --output_dir data_mouse_final \
    --target_distance 2.7
```

---

## 6. 품질 검증 체크리스트

- [ ] Clipping 없음: 모든 뷰에서 객체가 프레임 내에 있음
- [ ] 3D 일관성: 동일 샘플 내 scale factor 일치
- [ ] CoM 중앙: 객체 중심이 이미지 중앙 근처
- [ ] 카메라 동기화: intrinsics가 scale 변환 반영

---

## 7. 핵심 교훈

1. **샘플별 균일 scale**이 뷰별 균일보다 3D 일관성 유지에 유리
2. **CoM 기반** 스케일링이 clipping 방지에 효과적
3. **Intrinsics 동기화**: 이미지 변환 시 camera intrinsics도 반드시 함께 업데이트

---

## 관련 파일

- `scripts/preprocess_uniform_scale.py` - 현재 스케일링 스크립트
- `scripts/preprocess_center_align_all_views.py` - 중앙 정렬 스크립트
- `reports/preprocessing_comparison/` - 전처리 비교 시각화

---

*🤖 Generated with Claude Code - 2026-01-04*
*📝 통합 정리: 2026-01-05*
