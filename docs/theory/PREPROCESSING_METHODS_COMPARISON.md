# 전처리 방식 종합 비교 분석

> **작성일**: 2026-01-25
> **버전**: v1.0
> **관련 문서**: [[PP_FX_MVG_ANALYSIS]], [[PREPROCESSING_REGISTRY]], [[M3_SERIES_SPEC]]

---

## 1. 핵심 요약 (TL;DR)

| 방식 | Transform | Zoom | PP | 권장 |
|------|-----------|------|-----|------|
| D7.1 (M1) | Affine | - | 256 | ✅ 안정적 |
| D8 (M2) | Homography + Skew | - | 256 | ✅ 정밀 |
| M3_norm | Homography | Object-centered | 가변 | ⚠️ 특수 용도 |
| **M3_1** | Homography | **Center-aligned** | **256** | **★ 권장** |
| **M3_2** | Homography | **Center-aligned (per-sample)** | **256** | **★ 최고 권장** |

---

## 2. Transform 방식 비교

### 2.1 Affine Transform (D7.x, M1)

$$T_{affine} = \begin{bmatrix} a & b & t_x \\ c & d & t_y \end{bmatrix}$$

**특징**:
- 6 DoF: scale, rotation, shear, translation
- 평행선 유지 ✅
- 계산 빠름 ✅

**한계**:
- 원근 왜곡 미보정
- 카메라 틸트 미반영

### 2.2 Homography Transform (D8, M2, M3)

$$T_{homography} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & 1 \end{bmatrix}$$

**특징**:
- 8 DoF: affine + 원근 변환
- 기하학적 정확성 ✅
- Skew 보정 가능 ✅

**적용**:
```python
# Skew 보정 (D8 방식)
H = compute_homography_with_skew_correction(K, R, T)
img_corrected = cv2.warpPerspective(img, H, (512, 512))
```

### 2.3 Transform 선택 기준

| 상황 | 권장 Transform |
|------|----------------|
| 빠른 기준선 | Affine (M1) |
| 정밀 기하학 | Homography (M2) |
| 최대 coverage | Homography + Zoom (M3) |

---

## 3. Zoom 방식 비교

### 3.1 Object-Centered Zoom

```python
# 객체 중심 기준 crop
ys, xs = np.where(mask > 127)
cx = (xs.min() + xs.max()) / 2  # 객체 중심 (예: 300)
cy = (ys.min() + ys.max()) / 2  # 객체 중심 (예: 280)

crop_x = int(cx - crop_size/2)  # offset = 44 (300 - 256)
crop_y = int(cy - crop_size/2)  # offset = 24 (280 - 256)

# PP 변화
cx_new = (256 - 44) * 1.35 = 286  # ≠ 256
cy_new = (256 - 24) * 1.35 = 313  # ≠ 256
```

**특징**:
- 객체가 crop 중앙에 위치 ✅
- PP 가변 (pretrained 불일치) ❌
- Ray 방향 정확 ✅

**사용 사례**: MVG 정확성이 필수인 연구용

### 3.2 Center-Aligned Zoom (★ 권장)

```python
# 이미지 중심 기준 crop
crop_x = (512 - crop_size) // 2  # 항상 대칭
crop_y = (512 - crop_size) // 2  # 항상 대칭

# PP 자동 유지
cx_new = (256 - crop_x) * zoom = 256  # ★ 자동 유지
cy_new = (256 - crop_y) * zoom = 256  # ★ 자동 유지
```

**특징**:
- 이미지 중심 기준 crop
- PP = 256 자동 유지 ✅
- Pretrained 호환 ✅
- 객체가 crop 중앙이 아닐 수 있음 (offset)

**사용 사례**: GS-LRM pretrained 호환 필수

### 3.3 Zoom 방식 시각화

```
Object-Centered:          Center-Aligned:
┌─────────────┐           ┌─────────────┐
│    ┌───┐    │           │    ┌───┐    │
│    │ 🐭│    │           │    │ 🐭│    │
│    └───┘    │           │    └───┘    │
│      ↓      │           │      ↓      │
│  [객체 중심]  │           │  [이미지 중심] │
└─────────────┘           └─────────────┘
     ↓                         ↓
┌─────────┐               ┌─────────┐
│  ┌───┐  │               │   ┌───┐ │
│  │ 🐭│  │               │   │ 🐭│ │  ← 객체 offset
│  └───┘  │               │   └───┘ │
│ PP≠256  │               │ PP=256  │  ← PP 자동 유지
└─────────┘               └─────────┘
```

---

## 4. PP (Principal Point) 처리

### 4.1 PP 문제 정의

카메라 intrinsics:
$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

**문제**: Crop/Resize 시 PP 변화
$$c_x^{new} = (c_x - offset_x) \times zoom$$

### 4.2 PP 처리 전략

| 전략 | 설명 | 결과 PP | 권장 |
|------|------|---------|------|
| **shift_to_256** | PP-shift transform 적용 | 256 (변환 후) | M1, M2 |
| **preserve** | PP 변화 없음 | 원본 유지 | - |
| **force_to_target** | 강제 256 설정 | 256 (거짓) | ❌ |
| **center_aligned_zoom** | Zoom 방식으로 256 유지 | 256 (진실) | **★ M3** |

### 4.3 PP 오류 영향

$$\theta_{error} = \arctan\left(\frac{|c_x^{actual} - c_x^{used}|}{f_x}\right)$$

| PP 오류 | Ray Error | 영향 |
|---------|-----------|------|
| 0 px | 0° | 없음 |
| 10 px | ~1° | 미세 |
| 50 px | ~5° | Ghosting |
| 90 px | ~9° | 심각 |

---

## 5. Skew 보정

### 5.1 Skew 정의

카메라 skew (s): 이미지 축이 직교하지 않는 정도

$$K_{skew} = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

### 5.2 D8 Skew 보정

```python
# Homography로 skew 보정
K_ideal = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
K_actual = np.array([[fx, skew, cx], [0, fy, cy], [0, 0, 1]])

H = K_ideal @ np.linalg.inv(K_actual)
img_corrected = cv2.warpPerspective(img, H, (512, 512))
```

### 5.3 Skew 영향

| 데이터셋 | Skew 보정 | 영향 |
|----------|-----------|------|
| D7.1 (M1) | ❌ | 미세 오차 |
| **D8 (M2)** | ✅ | 정밀 기하학 |
| M3 | ✅ (homography 포함) | 정밀 + zoom |

---

## 6. fx 정규화

### 6.1 Pretrained 분포

GS-LRM (Objaverse):
- $f_x = 549$
- translation_norm = 2.7
- $f_x / trans \approx 203$

### 6.2 정규화 필요성

Zoom 적용 시:
$$f_x^{zoomed} = f_x \times zoom = 549 \times 1.35 = 741$$

**문제**: Pretrained 분포와 불일치 → 학습 불안정

### 6.3 normalize_after_zoom

```python
if normalize_after_zoom and zoom > 1.0:
    renorm_scale = target_fx / fx  # 549 / 741 = 0.74
    fx = target_fx  # 549
    fy = fy * renorm_scale
    
    # Center-aligned zoom만 PP 유지
    if zoom_center_mode == "image":
        cx, cy = 256, 256  # ★ 유지
    else:
        cx = cx * renorm_scale  # Object-centered: 스케일링
        cy = cy * renorm_scale
```

---

## 7. 데이터셋 버전 계보

```
Raw Data
    │
    ├── D0~D4 (Legacy) ────────────────── ❌ PP 버그
    │
    ├── D7.1 (M1) ─── Affine ─────────── ✅ 안정적
    │       └── pp_method: shift_to_256
    │
    ├── D8 (M2) ──── Homography + Skew ── ✅ 정밀
    │       └── skew_correction: true
    │
    └── M3 Series
            │
            ├── M3 (base) ── Object-centered ── PP 가변, fx=741
            │
            ├── M3_norm ─── Object-centered ── PP 가변, fx=549
            │
            ├── M3_persample ── Per-sample zoom ── PP~253
            │
            ├── M3_1 ★ ── Center-aligned (global) ── PP=256, fx=549
            │
            └── M3_2 ★★ ── Center-aligned (per-sample) ── PP=256, fx=549
```

---

## 8. 권장 설정

### 8.1 Production 권장

| 용도 | 데이터셋 | 실험 설정 |
|------|----------|-----------|
| **최고 성능** | M3_2 | E0_1_facelift |
| 안정적 기준 | M1 (D7.1) | E1_2_gt_alpha |
| 정밀 기하학 | M2 (D8) | E1_2_gt_alpha |

### 8.2 전처리 명령어

```bash
# M3_2 (★★ 최고 권장)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_2 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_2

# M1 (안정적)
python -m mouse_extensions.preprocessing.preprocess \
    --preset D7.1 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M1
```

---

## 9. 관련 문서

- **이론**: [[PP_FX_MVG_ANALYSIS]] - PP/fx MVG 수학적 분석
- **레지스트리**: [[PREPROCESSING_REGISTRY]] - 버전 관리 및 명령어
- **M3 상세**: [[M3_SERIES_SPEC]] - M3 시리즈 상세 명세
- **실험**: [[EXPERIMENT_REGISTRY]] - 실험 설정 레지스트리
- **빠른 참조**: [[MOUSE_QUICK_REFERENCE]] - 명령어 및 권장 설정

---

*PREPROCESSING_METHODS_COMPARISON v1.0 | 2026-01-25*

---

## 10. 버그 수정 이력

### 2026-01-25: Per-sample zoom_after_transform 버그

**문제**: Per-sample zoom이 원본 마스크로 계산됨 (transform 전)

**영향**: M3_2, M3_persample (zoom ≈ 1.0으로 거의 효과 없음)

**수정**: `preprocess.py:955-980` - transform 후 마스크로 coverage 계산

**관련**: [[PP_FX_MVG_ANALYSIS#11-8-per-sample-zoom-버그]]
