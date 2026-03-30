# M5 Series Specification (M5 계열 상세 명세)

> **Navigation**: [← Registry](./PREPROCESSING_REGISTRY.md) | [Commands](../experiments/COMMANDS.md)
> **목적**: M5, M5_4, M5_5 차이점 및 카메라 정규화 상세 설명
> **최종 업데이트**: 2026-01-29

---

## 1. 개요

M5 계열은 **카메라 정규화 방식**에 따른 ablation study를 위한 프리셋입니다.

| Preset | PP Centering | Translation Norm | 목적 |
|--------|--------------|------------------|------|
| **M5** | ✅ PP=256 | Batch Uniform | 기준선 |
| **M5_4** | ✅ PP=256 | None | Norm 필요성 검증 |
| **M5_5** | ✅ PP=256 | Per-view | Norm 방식 비교 |

---

## 2. 핵심 설정 비교

### 2.1 Preset 정의 (presets.py)

```python
"M5": {
    "pp_method": "shift_to_256",      # PP centering
    "recenter_cameras": True,         # ★ Batch uniform norm
    "normalize_translation": True,    # (recenter_cameras=True 시 무시됨)
}

"M5_4": {
    "pp_method": "shift_to_256",      # PP centering
    "recenter_cameras": False,        # No batch norm
    "normalize_translation": False,   # ★ No per-view norm either
}

"M5_5": {
    "pp_method": "shift_to_256",      # PP centering
    "recenter_cameras": False,        # No batch norm
    "normalize_translation": True,    # ★ Per-view norm
}
```

### 2.2 설정-동작 매핑

| 설정 조합 | 실제 동작 | Preset |
|-----------|----------|--------|
| `recenter_cameras=True` | Batch Uniform | M5 |
| `recenter=False, norm=False` | No Norm | M5_4 |
| `recenter=False, norm=True` | Per-view | M5_5 |

---

## 3. 정규화 방식 상세

### 3.1 Batch Uniform (M5) - 권장

**"Batch"의 의미: 샘플 내 6개 카메라** (전체 데이터셋 아님)

```python
def _normalize_cameras_batch(cam_params_list, target_distance=2.7):
    """
    샘플 하나의 6개 카메라에 대해:
    1. Centroid → Origin 이동
    2. 동일한 scale factor로 모든 카메라 조정
    """
    # Step 1: 6개 카메라 위치의 중심점 계산
    positions = [c2w[:3, 3] for each camera]  # 6개
    centroid = positions.mean(axis=0)
    centered = positions - centroid

    # Step 2: 평균 거리 기준 단일 scale factor
    mean_dist = mean(||centered||)
    scale = 2.7 / mean_dist  # 모든 카메라에 동일하게 적용

    # Step 3: 적용
    for i, cam in enumerate(cameras):
        cam.position = centered[i] * scale
```

**핵심 특징:**
- ✅ Centroid = Origin (pretrained 모델 호환)
- ✅ Average distance = 2.7
- ✅ **거리 비율 보존** (parallax 정확)

### 3.2 Per-view Normalization (M5_5)

```python
# preprocess.py line 918
for each camera:
    cam_pos = c2w[:3, 3]
    dist_scale = 2.7 / ||cam_pos||  # 각 카메라 개별 scale
    cam_pos = cam_pos * dist_scale   # 모든 카메라가 정확히 2.7m
```

**핵심 특징:**
- ⚠️ Centroid ≠ Origin
- ✅ 모든 카메라 = 정확히 2.7m
- ❌ **거리 비율 왜곡** (parallax 오류)

### 3.3 No Normalization (M5_4)

```python
# normalize_translation=False
cam_pos = c2w[:3, 3]  # 원본 그대로 사용
```

**핵심 특징:**
- ❌ Centroid ≠ Origin
- ❌ Average ≠ 2.7 (pretrained scale 불일치)
- ✅ **거리 비율 보존**

---

## 4. 시각적 비교

### 4.1 원본 카메라 배치 (Top View)

```
        cam2 (2.8m)
           ●
          /|\
    cam1 ● | ● cam3
   (2.6m)  |  (2.5m)
          [M]  ← Mouse (not at origin)
    cam5 ● | ● cam4
   (2.6m)  |  (2.7m)
           ●
        cam0 (2.4m)

Centroid: (0.1, 0.05, 0.02)  ← Origin이 아님
Average distance: 2.6m
```

### 4.2 정규화 후 비교

```
┌─────────────────────────────────────────────────────────────────────────────┐
│         M5 (Batch Uniform)        │     M5_4 (None)    │   M5_5 (Per-view) │
├─────────────────────────────────────────────────────────────────────────────┤
│        cam2 (2.91m)               │    cam2 (2.8m)     │    cam2 (2.7m)    │
│           ●                       │       ●            │       ●           │
│          /|\                      │      /|\           │      /|\          │
│    cam1 ● | ● cam3                │ cam1 ● | ● cam3    │ cam1 ● | ● cam3   │
│   (2.70)  |  (2.60)               │ (2.6)  |  (2.5)    │ (2.7)  |  (2.7)   │
│        [ORIGIN]                   │      [M]           │      [M]          │
│    cam5 ● | ● cam4                │ cam5 ● | ● cam4    │ cam5 ● | ● cam4   │
│   (2.70)  |  (2.81)               │ (2.6)  |  (2.7)    │ (2.7)  |  (2.7)   │
│           ●                       │       ●            │       ●           │
│        cam0 (2.49m)               │    cam0 (2.4m)     │    cam0 (2.7m)    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Centroid: (0, 0, 0) ✅            │ (0.1, 0.05, 0.02)  │ (0.1, 0.05, 0.02) │
│ Avg dist: 2.7m ✅                 │ 2.6m ❌            │ 2.7m ✅           │
│ Ratio: 0.92:1:1.08 ✅             │ 0.92:1:1.08 ✅     │ 1:1:1 ❌          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 수치 예시

| Camera | 원본 | M5 (Batch) | M5_4 (None) | M5_5 (Per-view) |
|--------|------|------------|-------------|-----------------|
| cam0 | 2.4m | 2.49m | 2.4m | 2.7m |
| cam1 | 2.6m | 2.70m | 2.6m | 2.7m |
| cam2 | 2.8m | 2.91m | 2.8m | 2.7m |
| cam3 | 2.5m | 2.60m | 2.5m | 2.7m |
| cam4 | 2.7m | 2.81m | 2.7m | 2.7m |
| cam5 | 2.6m | 2.70m | 2.6m | 2.7m |
| **Avg** | 2.6m | **2.7m** | 2.6m | **2.7m** |
| **비율 보존** | - | ✅ | ✅ | ❌ |

---

## 5. 왜 Batch Uniform이 중요한가?

### 5.1 Multi-view 3D Reconstruction 원리

```
Depth estimation = f(parallax)
Parallax = camera baseline / object distance

카메라 거리 비율이 바뀌면 → parallax 계산 오류 → depth 오류
```

### 5.2 Per-view Norm의 문제 (M5_5)

```
원본: cam0=2.4m, cam2=2.8m → baseline 비율 = 2.8/2.4 = 1.17
Per-view: cam0=2.7m, cam2=2.7m → baseline 비율 = 1.0

→ 16.7% parallax 오류
→ 3D 재구성 시 depth 왜곡
```

### 5.3 No Norm의 문제 (M5_4)

```
Pretrained GS-LRM: avg distance = 2.7m 기대
M5_4 실제: avg distance = 2.6m

→ Scale mismatch
→ 학습 초기 불안정 예상
```

### 5.4 Batch Uniform의 장점 (M5)

| 측면 | 효과 |
|------|------|
| Centroid = Origin | Pretrained 모델과 좌표계 일치 |
| Avg = 2.7m | Pretrained scale과 일치 |
| 비율 보존 | Parallax 정확 → 3D 정확 |

---

## 6. 검증 가설

### 6.1 Ablation 매트릭스

|  | No Norm | Per-view | Batch Uniform |
|--|---------|----------|---------------|
| **Center (PP=256)** | M5_4 | M5_5 | **M5** ✅ |
| **No Center** | M0 | M0_n | (N/A) |

### 6.2 예상 결과

| 비교 | 예상 | 검증 내용 |
|------|------|----------|
| M5 vs M5_4 | M5 >> M5_4 | Pretrained scale 매칭 중요 |
| M5 vs M5_5 | M5 >> M5_5 | 기하학 보존 중요 |
| M5_4 vs M5_5 | M5_4 ≈ M5_5 또는 ? | 어떤 오류가 더 치명적? |

### 6.3 결과 해석 시나리오

| 결과 패턴 | 의미 | 결론 |
|----------|------|------|
| M5 ≈ M5_5 >> M5_4 | Scale 매칭이 핵심 | Norm 필수, 방식은 유연 |
| M5 >> M5_5 ≈ M5_4 | 기하학 보존이 핵심 | Batch Uniform만 가능 |
| M5 ≈ M5_4 ≈ M5_5 | PP centering만으로 충분 | Norm 불필요 |
| M5 >> M5_4, M5 >> M5_5 | 둘 다 중요 | 현재 M5가 최적 |

---

## 7. 코드 흐름

```
preprocess.py: process_sample()
    │
    ├── for each camera (0-5):
    │       compute_camera_params(..., skip_distance_norm=recenter_cameras)
    │       │
    │       └── if not skip_distance_norm:  # M5_5: Per-view norm
    │               dist_scale = 2.7 / ||cam_pos||
    │               cam_pos *= dist_scale
    │
    └── if recenter_cameras:  # M5: Batch uniform
            _normalize_cameras_batch(all 6 cameras)
            │
            ├── Step 1: centroid → origin
            └── Step 2: uniform scale (same for all)
```

---

## 8. Quick Reference

| 질문 | 답변 |
|------|------|
| Batch = 전체 데이터셋? | ❌ 샘플 내 6개 카메라 |
| M5와 M5_4 차이? | M5: Batch Uniform, M5_4: No Norm |
| M5와 M5_5 차이? | M5: Batch Uniform, M5_5: Per-view |
| 권장 설정? | **M5** (Batch Uniform) |
| M5_4, M5_5 용도? | Ablation 실험용 |

---

## Related Documents

- ↑ [[PREPROCESSING_REGISTRY]] — Preset 중앙 관리
- ↔ [[RAW_DATA]] — M5t2 데이터 명세 + 원본 출처
- ↔ Obsidian [[CAMERA_NORMALIZATION]] — Batch Uniform 설계 근거 + MVG 이론 (Why)

---

*FaceLift Mouse Project | M5 Series Spec v1.1 | 2026-03-30*
