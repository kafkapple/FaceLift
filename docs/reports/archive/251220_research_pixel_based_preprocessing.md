---
date: 2024-12-20
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, preprocessing, pixel-based, center-of-mass]
project: mouse-facelift
status: in_progress
generator: ai-assisted
generator_tool: claude-code
---

# Pixel-Based Preprocessing for Mouse-FaceLift

## 개요

Mouse-FaceLift 파이프라인에서 **bounding box 기반 전처리의 근본적 문제**를 해결하기 위한 완전 pixel-based 전처리 방식 도입.

## 문제 상황

### 기존 전처리 단계별 문제

| 단계 | 기존 방식 | 문제점 |
|------|----------|--------|
| **Centering** | Bbox 중심 | 꼬리 방향에 따라 중심점이 왜곡됨 |
| **Scaling** | Bbox 비율 → 픽셀 비율 (v2에서 해결) | v1에서는 bbox 크기 불균일 |

### 꼬리가 중심점에 미치는 영향 예시

```
Case 1: 꼬리가 왼쪽                  Case 2: 꼬리가 오른쪽
  ┌─────────────────┐                ┌─────────────────┐
  │      ●●●        │                │        ●●●      │
  │     ●●●●●       │                │       ●●●●●     │
  │    ●●●●●●●      │                │      ●●●●●●●    │
  │   ●●●●●●●●      │                │      ●●●●●●●●   │
  │  ●●●●●●●●●      │                │     ●●●●●●●●●●  │
  │ ●●●●    ●●●     │                │     ●●●    ●●●● │
  │●●               │                │               ●●│
  └─────────────────┘                └─────────────────┘
   Bbox 중심: 왼쪽으로 편향            Bbox 중심: 오른쪽으로 편향
```

**실제 "몸통 중심"은 동일하지만 bbox 중심은 크게 달라짐**

## 해결 방법: Pixel-Based Preprocessing

### 1. Center of Mass (CoM) 기반 Centering

```python
def get_center_of_mass(alpha: np.ndarray, threshold: int = 10):
    """
    CoM = Σ(position × weight) / Σ(weight)

    꼬리는 픽셀 수가 적어 중심에 미치는 영향이 작음
    """
    mask = (alpha > threshold).astype(np.float32)
    if not mask.any():
        return None

    h, w = alpha.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]

    total_mass = np.sum(mask)
    cx = np.sum(x_coords * mask) / total_mass
    cy = np.sum(y_coords * mask) / total_mass

    return float(cx), float(cy)
```

**왜 CoM이 더 나은가?**

- 꼬리: 픽셀 수 적음 (전체의 ~10-15%)
- 몸통: 픽셀 수 많음 (전체의 ~85-90%)
- 결과: CoM은 몸통 쪽으로 자연스럽게 가중됨

### 2. Pixel Count 기반 Scaling

```python
def get_pixel_size_ratio(alpha: np.ndarray, threshold: int = 10):
    """
    size_ratio = sqrt(pixel_count / total_pixels)

    sqrt 변환: 면적(2D) → 선형 크기(1D)
    """
    mask = alpha > threshold
    pixel_count = np.sum(mask)
    total_pixels = alpha.shape[0] * alpha.shape[1]

    area_ratio = pixel_count / total_pixels
    size_ratio = np.sqrt(area_ratio)

    return float(size_ratio)
```

### 3. 통합 변환

```python
# 단일 affine transform으로 centering + scaling 동시 적용
M = np.array([
    [scale, 0, tx],
    [0, scale, ty]
], dtype=np.float32)

output = cv2.warpAffine(image, M, (output_size, output_size), ...)
```

## 기대 효과

### Centering 개선

| 지표 | Bbox 기반 | CoM 기반 (예상) |
|------|----------|----------------|
| 뷰 간 중심 편차 | 높음 | 낮음 |
| 꼬리 방향 의존성 | 높음 | 낮음 |

### Scaling 일관성 (이미 검증됨)

| 지표 | Bbox 기반 | Pixel 기반 |
|------|----------|-----------|
| Size ratio Std | 1.32% | **0.00%** |
| Size ratio CV | 5.2% | **0.00%** |

## 데이터셋 히스토리

```
data_mouse (원본)
    ↓
data_mouse_centered (bbox centering + bbox scaling) ← 문제 있음
    ↓
data_mouse_uniform_v2 (bbox centering + pixel scaling) ← 부분 해결
    ↓
data_mouse_pixel_based (CoM centering + pixel scaling) ← 완전 해결 (진행 중)
```

## 현재 진행 상황

### 1. MVDiffusion 학습 (uniform_v2 데이터)
- Config: `mouse_mvdiffusion_uniform_v2.yaml`
- 시작: pretrained (pipeckpts)에서 finetune
- WandB: `mvdiff_uniform_v2`
- 상태: 진행 중

### 2. Pixel-Based 전처리
- 스크립트: `scripts/preprocess_pixel_based.py`
- 입력: `data_mouse` (원본)
- 출력: `data_mouse_pixel_based`
- 상태: 진행 중 (~30분 예상)

## 다음 단계

1. ~~pixel-based 전처리 스크립트 작성~~ (완료)
2. ~~data_mouse_pixel_based 생성~~ (진행 중)
3. 전처리 결과 검증
   - CoM vs Bbox 중심 차이 통계
   - 출력 이미지 시각적 검증
4. MVDiffusion config 작성 (`mouse_mvdiffusion_pixel_based.yaml`)
5. MVDiffusion 학습 (pixel-based 데이터)
6. 결과 비교 (uniform_v2 vs pixel_based)

## 관련 파일

- `scripts/preprocess_pixel_based.py`: 완전 pixel-based 전처리 스크립트
- `scripts/preprocess_uniform_scale.py`: 픽셀 스케일링만 적용 (centering은 기존 유지)
- `configs/mouse_mvdiffusion_uniform_v2.yaml`: 현재 학습 중인 config
- `data_mouse_pixel_based/`: 생성 중인 데이터셋

## 핵심 인사이트

### Why Pixel-Based Preprocessing?

1. **Domain Shift 최소화**: 원본 데이터의 특성을 최대한 보존하면서 정규화
2. **꼬리 독립성**: 마우스의 변동성 높은 꼬리가 전처리에 미치는 영향 최소화
3. **물리적 의미**: Center of Mass는 실제 "무게 중심"에 가까움
4. **재현성**: 모든 단계가 deterministic하고 검증 가능

### Trade-off

- **장점**: 더 일관된 학습 데이터, 꼬리 방향에 덜 민감
- **단점**: 카메라 파라미터와의 불일치 (image-space only transform)
- **대응**: MVDiffusion은 2D 이미지 생성 → 영향 적음. GS-LRM에서는 추가 검토 필요

---

## Pose-Splatter vs FaceLift 비교 분석

### 왜 FaceLift에서는 전처리가 까다로운가?

두 파이프라인의 근본적 차이:

```
Pose-Splatter:
  Real 6-views → Visual Hull (3D) → 3DGS
                      ↓
                3D 공간에서 기하학적 처리
                카메라 파라미터가 크기 차이를 자동 보정

FaceLift:
  1-view → MVDiffusion (2D 생성) → 5 synthetic views → GS-LRM
                ↓
           이미지 레벨에서 "같은 객체의 다른 뷰" 패턴 학습
           크기/위치 불균일 → 패턴 학습 실패
```

### 핵심 차이점

| 측면 | Pose-Splatter | FaceLift |
|------|---------------|----------|
| **입력** | Real multi-view (6개) | Single-view |
| **3D 추정** | Visual Hull → 직접 3D 복원 | 2D diffusion → 합성 뷰 생성 |
| **크기 정규화** | 불필요 (3D에서 자동) | **필수** (2D 학습에 영향) |
| **전처리** | 간단 | 민감 |

### Pose-Splatter의 Shape Carving

```python
# shape_carver.py: 3D 공간에서 직접 처리
def get_volume_torch(images, intrinsic_matrices, extrinsic_matrices, grid_points):
    # 1. 3D 그리드 포인트 생성
    # 2. 각 카메라로 투영
    # 3. 모든 카메라에서 visible한 복셀 = 객체
    all_projected_coords = project_points_torch(grid_points, ...)
    sampled_values = sample_nearest_pixels_torch(images, ...)
    return averaged_values
```

→ 이미지 크기 차이는 카메라 파라미터(intrinsic/extrinsic)가 자동 보정

### 장단점 비교

| | Pose-Splatter | FaceLift |
|--|---------------|----------|
| **장점** | 실제 데이터 직접 활용, 전처리 간단, 3D 정확성 | 단일 뷰에서 3D 복원 가능, 카메라 setup 자유로움 |
| **단점** | 항상 multi-view 카메라 필요 | 전처리 민감, 합성 품질 의존 |
| **적합** | 고정 카메라 setup | sparse view, 단일 카메라 |

---

## 하이브리드 전처리 제안

### 아이디어: Visual Hull 기반 CoM 추정

Pose-Splatter의 3D 접근과 FaceLift의 2D 처리를 결합:

```
기존 (bbox 기반):
  2D 이미지 → Bbox 중심 → 문제 (꼬리 영향)

개선 (CoM 기반, 현재):
  2D 이미지 → Pixel CoM → 개선 (꼬리 영향 감소)

하이브리드 (제안):
  Multi-view → Visual Hull (3D) → 3D CoM → 2D 투영
                                      ↓
                               뷰 간 일관된 중심점
```

### 구현 방안

```python
def get_3d_com_from_visual_hull(images, cameras):
    """
    1. Visual Hull로 대략적 3D 볼륨 추정
    2. 3D 볼륨의 CoM 계산
    3. 각 카메라로 투영하여 2D 중심점 획득
    """
    # Step 1: Visual Hull
    voxel_grid = create_3d_grid(...)
    occupied = shape_carving(images, cameras, voxel_grid)

    # Step 2: 3D CoM
    com_3d = compute_3d_center_of_mass(occupied, voxel_grid)

    # Step 3: 각 뷰로 투영
    centers_2d = []
    for cam in cameras:
        center_2d = project_point(com_3d, cam)
        centers_2d.append(center_2d)

    return centers_2d
```

### 기대 효과

| 방식 | 뷰 간 일관성 | 계산 비용 | 구현 복잡도 |
|------|-------------|----------|-----------|
| Bbox | 낮음 | 낮음 | 낮음 |
| 2D CoM | 중간 | 낮음 | 낮음 |
| **3D CoM (제안)** | **높음** | 중간 | 중간 |

---

## Downstream Task 활용 가능성 분석

### 제안: Multi-View Feature Embedding for Behavior Analysis

Pose-Splatter 논문의 "회전-불변 시각 임베딩"을 FaceLift에 적용:

```
FaceLift Output:
  Single-view → 6 canonical views (합성)
                    ↓
         고른 시점의 2D 이미지 세트

Feature Extraction:
  6 views → CNN/ViT Encoder → [feat_1, ..., feat_6]
                    ↓
         시점 불변 특징 (concat or pool)

Downstream:
  Multi-view Features → Action Recognition
                     → Keypoint Prediction
                     → Behavior Classification
```

### 비판적 검토

**장점**:
1. **시점 불변성**: 고정된 6개 canonical view → 카메라 방향에 독립적 표현
2. **풍부한 정보**: 단일 뷰보다 3D 구조 정보 포함
3. **End-to-end 학습 가능**: 3D 복원 + downstream task 동시 최적화

**단점/우려**:
1. **합성 품질 의존**: MVDiffusion 품질이 낮으면 feature도 부정확
2. **Computational cost**: 프레임마다 6개 뷰 생성 → inference 비용 증가
3. **Domain gap**: 합성 뷰와 실제 뷰 간 차이가 feature에 영향
4. **Temporal consistency**: 프레임 간 합성 뷰의 일관성 미보장

### Pose-Splatter 참고 (arxiv:2505.18342)

논문의 핵심 접근:
- **64 novel views 렌더링** → 충분한 3D 커버리지
- **Spherical Harmonic 확장**: 방향성 정보 인코딩
- **AAPCA 50D 임베딩**: 저차원 자세 표현
- **4-way 행동 분류**: Walk/Head Up/Still/Groom

### 모듈형 구현 계획

```
Phase 1: 기반 검증
  - FaceLift 출력 품질 확인 (현재 진행 중)
  - 6-view 합성 일관성 평가

Phase 2: Feature Extraction
  - ResNet/ViT 기반 특징 추출 모듈
  - Multi-view fusion 전략 실험 (concat vs attention)

Phase 3: Downstream Tasks
  - Behavior classification (Pose-Splatter 4-class)
  - Keypoint prediction (lifted from 2D predictions)

Phase 4: Temporal Extension
  - Frame 간 일관성 향상
  - Action recognition with temporal modeling
```

### 관련 코드 참조

- `pose-splatter/src/modules/deform/behavior_features.py`: 행동 특징 추출
- `pose-splatter/scripts/preprocessing/calculate_visual_features.py`: 시각 특징 계산
- `pose-splatter/scripts/preprocessing/calculate_visual_embedding.py`: AAPCA 임베딩

---

## 전처리 품질 검증 방법론

### 핵심 문제: 2D 전처리의 3D 일관성 검증

전처리의 최종 목표는 **2단계 모델 학습에 적합한 데이터**를 생성하는 것.
문제는 2D에서 수행한 전처리가 **3D 기하학적으로 일관성**이 있는지 확인하기 어려움.

### 검증 방법론 설계

#### 방법 1: 3D CoM 기반 Ground Truth 생성

**원리**: Visual Hull에서 추정한 3D CoM을 각 뷰에 투영하여 "이상적인" 2D center 획득

```
6개 뷰 실루엣 → Visual Hull (3D) → 3D CoM → 각 뷰 투영
                                          ↓
                              Ground Truth 2D centers
```

**검증 지표**:
- **Center Error**: 전처리에 사용된 center와 3D CoM 투영점 간 거리
- **Cross-View Consistency**: 6개 뷰의 center error 분산

**구현**:
```python
def compute_3d_com_centers(alpha_masks, intrinsics, extrinsics, grid_size=64):
    """
    1. Visual Hull 생성
    2. 3D CoM 계산
    3. 각 뷰에 투영
    """
    # Step 1: Create 3D grid
    grid = create_3d_grid(ellipsoid_size, grid_size)  # [n1, n2, n3, 3]

    # Step 2: Shape carving (Visual Hull)
    volume = get_volume_torch(alpha_masks, intrinsics, extrinsics, grid)
    occupied = volume > 0.5  # threshold

    # Step 3: Compute 3D CoM
    occupied_points = grid[occupied]  # [N, 3]
    com_3d = occupied_points.mean(dim=0)  # [3]

    # Step 4: Project to each view
    centers_2d = []
    for i in range(len(intrinsics)):
        pt_2d = project_point(com_3d, intrinsics[i], extrinsics[i])
        centers_2d.append(pt_2d)

    return com_3d, centers_2d
```

#### 방법 2: Epipolar Consistency 검증

**원리**: 기하학적으로 일관된 center라면 epipolar constraint를 만족해야 함

```
View A의 center → View B의 epipolar line
                      ↓
               View B의 center가 이 line 위에 있어야 함
```

**검증 지표**:
- **Epipolar Distance**: center에서 epipolar line까지 거리 (픽셀)
- 이상적인 경우 = 0

**구현**:
```python
def check_epipolar_consistency(centers, intrinsics, extrinsics):
    """
    모든 뷰 쌍에 대해 epipolar consistency 검사
    """
    n_views = len(centers)
    errors = []

    for i in range(n_views):
        for j in range(i+1, n_views):
            # Compute fundamental matrix F_ij
            F = compute_fundamental_matrix(intrinsics[i], extrinsics[i],
                                           intrinsics[j], extrinsics[j])

            # Epipolar line in view j for center i
            line_j = F @ np.array([*centers[i], 1])

            # Distance from center j to this line
            dist = point_to_line_distance(centers[j], line_j)
            errors.append(dist)

    return np.mean(errors), np.std(errors)
```

#### 방법 3: Triangulation 수렴성 검증

**원리**: 6개 뷰의 center가 동일한 3D 점을 가리키면, triangulation 결과가 수렴해야 함

**검증 지표**:
- **Reprojection Error**: triangulation 결과를 각 뷰에 투영했을 때 원래 center와의 거리
- **3D Point Variance**: 다양한 뷰 조합에서 triangulation 결과의 분산

**구현**:
```python
def triangulate_centers(centers, intrinsics, extrinsics):
    """
    모든 뷰의 center를 사용하여 3D point 추정
    """
    # Build projection matrices
    P_matrices = []
    for i in range(len(centers)):
        P = intrinsics[i] @ extrinsics[i][:3, :]
        P_matrices.append(P)

    # DLT triangulation
    A = []
    for i, (center, P) in enumerate(zip(centers, P_matrices)):
        x, y = center
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    point_3d = Vt[-1]
    point_3d = point_3d[:3] / point_3d[3]

    # Compute reprojection error
    reproj_errors = []
    for i, P in enumerate(P_matrices):
        proj = P @ np.array([*point_3d, 1])
        proj = proj[:2] / proj[2]
        error = np.linalg.norm(proj - centers[i])
        reproj_errors.append(error)

    return point_3d, reproj_errors
```

### 시각화 도구 설계

#### 1. Center Point Overlay

```python
def visualize_centers(images, centers_bbox, centers_com, centers_3d=None):
    """
    각 이미지에 다양한 center 표시
    - 빨간색: Bbox center
    - 파란색: 2D CoM
    - 초록색: 3D CoM 투영 (ground truth)
    """
    for i, img in enumerate(images):
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.scatter(*centers_bbox[i], c='r', label='Bbox')
        plt.scatter(*centers_com[i], c='b', label='2D CoM')
        if centers_3d:
            plt.scatter(*centers_3d[i], c='g', label='3D CoM')
        plt.legend()
```

#### 2. Error Histogram

```python
def plot_center_error_distribution(errors_by_method):
    """
    Bbox, 2D CoM, 3D CoM 각각의 error 분포 비교
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (method, errors) in zip(axes, errors_by_method.items()):
        ax.hist(errors, bins=50)
        ax.set_title(f'{method}: μ={np.mean(errors):.2f}, σ={np.std(errors):.2f}')
```

#### 3. Cross-View Consistency Matrix

```python
def plot_consistency_matrix(centers, intrinsics, extrinsics):
    """
    뷰 쌍별 epipolar distance heatmap
    이상적인 경우 모든 값이 0에 가까움
    """
    n_views = len(centers)
    matrix = np.zeros((n_views, n_views))
    for i in range(n_views):
        for j in range(n_views):
            if i != j:
                matrix[i, j] = compute_epipolar_distance(centers[i], centers[j], ...)

    plt.imshow(matrix)
    plt.colorbar(label='Epipolar Distance (pixels)')
```

### 품질 지표 정의

| 지표 | 설명 | 목표값 |
|------|------|--------|
| **Mean Center Error** | 2D center와 3D CoM 투영점 간 평균 거리 | < 5 pixels |
| **Center Error Std** | 6개 뷰 간 center error 표준편차 | < 2 pixels |
| **Epipolar Consistency** | 평균 epipolar distance | < 1 pixel |
| **Reprojection Error** | Triangulation 후 reprojection error | < 2 pixels |
| **Cross-View CV** | 뷰 간 center 위치 변동계수 | < 5% |

### 구현 계획

```
Phase 1: 기본 검증 도구 (1-2일)
  ├── scripts/evaluate_preprocessing.py
  │     - Bbox vs CoM 비교
  │     - 기본 통계량 출력
  └── scripts/visualize_centers.py
        - 6개 뷰 center overlay 시각화

Phase 2: 3D 기반 검증 (2-3일)
  ├── scripts/visual_hull_center.py
  │     - Visual Hull 생성
  │     - 3D CoM 계산 및 투영
  └── scripts/check_epipolar_consistency.py
        - Epipolar constraint 검증

Phase 3: 자동화 파이프라인 (1일)
  └── scripts/preprocess_with_validation.py
        - 전처리 + 자동 품질 검증
        - 품질 미달 샘플 플래깅
```

### Downstream Task를 위한 메타데이터 저장

전처리 후 역변환 정보 저장 (downstream task에서 원본 좌표계로 복원 시 필요):

```python
preprocessing_metadata = {
    'original_size': (H, W),
    'output_size': 512,
    'transform': {
        'scale': float,
        'center_original': (cx, cy),  # 전처리 전 center
        'center_output': (256, 256),  # 전처리 후 center
    },
    'quality_metrics': {
        'center_error': float,        # 3D CoM 대비 error
        'epipolar_consistency': float,
        'reprojection_error': float,
    },
    'method': 'pixel_com',  # 'bbox', 'pixel_com', '3d_com'
}
```

### 하이브리드 전처리 개선 방안

품질 검증 결과를 바탕으로 전처리 방식 개선:

```
Level 1: 2D CoM (현재)
  ├── 장점: 빠름, 간단
  └── 한계: 뷰 간 일관성 부족 가능

Level 2: 3D CoM (하이브리드)
  ├── Visual Hull → 3D CoM → 2D 투영
  ├── 장점: 뷰 간 완벽한 일관성
  └── 한계: 계산 비용 증가, Visual Hull 품질 의존

Level 3: Iterative Refinement (제안)
  ├── 1) 2D CoM으로 초기 전처리
  ├── 2) 품질 지표 계산
  ├── 3) 품질 미달 시 3D CoM 방식으로 보정
  └── 4) 최종 데이터셋 생성

Adaptive Selection:
  - epipolar_consistency < 1px → 2D CoM 유지
  - epipolar_consistency >= 1px → 3D CoM 적용
```

### 관련 참조 코드

- `pose-splatter/src/shape_carver.py`: Visual Hull 및 3D 투영
  - `get_volume_torch()`: Shape carving
  - `project_points_torch()`: 3D → 2D 투영
  - `ray_cast_visibility_torch()`: 가시성 계산

---

## 실험 계획 요약

### 단기 (현재 진행)
1. pixel-based 전처리 완료
2. MVDiffusion uniform_v2 학습
3. 기본 품질 검증 도구 구현

### 중기 (다음 단계)
1. 3D CoM 기반 ground truth 생성
2. 전처리 품질 정량 평가
3. 하이브리드 전처리 구현

### 장기 (확장)
1. Downstream task 모듈 설계
2. Multi-view feature extraction
3. Behavior classification/Keypoint prediction 실험
