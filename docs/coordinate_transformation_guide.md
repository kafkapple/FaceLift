# GS-LRM 좌표계 변환 및 전처리 가이드

> **목적**: GS-LRM/FaceLift 학습을 위한 카메라 파라미터 변환의 수학적 기초와 전처리 필수 요소 정리

---

## 1. 좌표계 정의

### 1.1 네 가지 좌표계

| 좌표계 | 기호 | 차원 | 설명 |
|--------|------|------|------|
| **World** | $\mathbf{P}_w$ | 3D | 월드 공간의 절대 좌표 |
| **Camera** | $\mathbf{P}_c$ | 3D | 카메라 중심 기준 좌표 |
| **Normalized** | $\mathbf{p}_n$ | 2D | 초점거리 1인 정규화 평면 |
| **Pixel** | $\mathbf{p}$ | 2D | 이미지 픽셀 좌표 |

### 1.2 좌표 표기법

```
World:      P_w = [X_w, Y_w, Z_w]^T
Camera:     P_c = [X_c, Y_c, Z_c]^T
Normalized: p_n = [x_n, y_n]^T
Pixel:      p   = [u, v]^T
```

---

## 2. 변환 단계별 수식

### 2.1 World → Camera (Extrinsic Transform)

#### 변환 행렬

World-to-Camera 변환 행렬 $\mathbf{W}$ (4×4):

$$
\mathbf{W} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}
$$

여기서 $\mathbf{R}$: 3×3 회전 행렬, $\mathbf{t}$: 3×1 이동 벡터

#### 수식

$$
\mathbf{P}_c = \mathbf{R} \cdot \mathbf{P}_w + \mathbf{t}
$$

또는 동차좌표(homogeneous coordinates)로:

$$
\begin{bmatrix} X_c \\ Y_c \\ Z_c \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
r_{11} & r_{12} & r_{13} & t_x \\ 
r_{21} & r_{22} & r_{23} & t_y \\ 
r_{31} & r_{32} & r_{33} & t_z \\ 
0 & 0 & 0 & 1 
\end{bmatrix} 
\begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
$$

#### 카메라 위치 계산

Camera-to-World 변환 $\mathbf{C} = \mathbf{W}^{-1}$에서:

$$
\mathbf{C}_{position} = -\mathbf{R}^T \cdot \mathbf{t}
$$

카메라 거리:

$$
d = \|\mathbf{C}_{position}\| = \sqrt{C_x^2 + C_y^2 + C_z^2}
$$

---

### 2.2 Camera → Normalized (Perspective Projection)

#### 핵심 원리: Pinhole Camera Model

카메라 좌표를 깊이($Z_c$)로 나누어 2D 평면에 투영:

$$
x_n = \frac{X_c}{Z_c}, \quad y_n = \frac{Y_c}{Z_c}
$$

#### 의미

- 초점거리(focal length) = 1인 가상 이미지 평면의 좌표
- 원근감(perspective)이 적용된 정규화 좌표
- **단위 없음** (dimensionless)

---

### 2.3 Normalized → Pixel (Intrinsic Transform)

#### Intrinsic Matrix K

$$
\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

#### 파라미터 의미

| 파라미터 | 의미 | 단위 |
|----------|------|------|
| $f_x$ | 수평 초점거리 | pixels |
| $f_y$ | 수직 초점거리 | pixels |
| $c_x$ | Principal point X | pixels |
| $c_y$ | Principal point Y | pixels |

#### 변환 수식

$$
u = f_x \cdot x_n + c_x = f_x \cdot \frac{X_c}{Z_c} + c_x
$$

$$
v = f_y \cdot y_n + c_y = f_y \cdot \frac{Y_c}{Z_c} + c_y
$$

---

### 2.4 역변환: Pixel → Normalized

Pixel 좌표에서 Normalized 좌표로:

$$
x_n = \frac{u - c_x}{f_x}, \quad y_n = \frac{v - c_y}{f_y}
$$

**이것이 "초점거리로 나누는" 연산의 의미:**

$$
\mathbf{p}_n = \mathbf{K}^{-1} \mathbf{p}
$$

---

## 3. 전체 투영 파이프라인

### 3.1 World → Pixel (순방향)

$$
\mathbf{p} = \pi(\mathbf{P}_w) = \mathbf{K} \cdot \frac{1}{Z_c} \cdot [\mathbf{R} | \mathbf{t}] \cdot \mathbf{P}_w^{(h)}
$$

전개하면:

$$
\begin{bmatrix} u \\ v \end{bmatrix} = 
\begin{bmatrix} 
f_x \frac{X_c}{Z_c} + c_x \\ 
f_y \frac{Y_c}{Z_c} + c_y 
\end{bmatrix}
$$

여기서 $[X_c, Y_c, Z_c]^T = \mathbf{R} \cdot \mathbf{P}_w + \mathbf{t}$

### 3.2 Pixel → Ray (역방향, GS-LRM에서 사용)

픽셀 좌표 $(u, v)$에서 3D ray 생성:

**Ray Origin** (카메라 위치):

$$
\mathbf{o} = -\mathbf{R}^T \cdot \mathbf{t}
$$

**Ray Direction** (정규화된 방향):

$$
\mathbf{d} = \mathbf{R}^T \cdot \begin{bmatrix} \frac{u - c_x}{f_x} \\ \frac{v - c_y}{f_y} \\ 1 \end{bmatrix}
$$

정규화:

$$
\hat{\mathbf{d}} = \frac{\mathbf{d}}{\|\mathbf{d}\|}
$$

---

## 4. GS-LRM의 Plücker Ray Encoding

### 4.1 Plücker 좌표계

GS-LRM은 각 픽셀의 ray를 6D Plücker 좌표로 인코딩:

$$
\mathbf{L} = (\mathbf{d}, \mathbf{m}) \in \mathbb{R}^6
$$

여기서:
- $\mathbf{d}$: Ray direction (3D)
- $\mathbf{m} = \mathbf{o} \times \mathbf{d}$: Moment vector (3D)

### 4.2 Plücker 좌표의 특성

$$
\mathbf{d} \cdot \mathbf{m} = 0 \quad \text{(직교 조건)}
$$

**중요**: Plücker ray는 **절대적인 카메라 위치**와 **ray 방향**을 모두 인코딩

---

## 5. 실제 데이터 변환 계산 예시

### 5.1 원본 데이터 (Camera 0)

```
원본 파라미터:
  fx = 1632.3 px
  fy = 1639.3 px  (fx ≠ fy, 비정사각)
  cx = 601.3 px
  cy = 491.2 px
  이미지 크기: 1152 × 1024 px
  카메라 거리: 246.1 mm (normalized: 2.46)
```

### 5.2 Step 1: fy = fx 강제

**문제**: 원본 $f_y/f_x = 1639.3/1632.3 = 1.0043$ (0.43% 차이)

**해결**: $f_y$를 $f_x$로 강제 대체

$$
f_x^{new} = f_y^{new} = f_x^{orig} = 1632.3
$$

### 5.3 Step 2: 거리 정규화

**목표**: 모든 카메라를 동일 거리($d = 2.7$)에 배치

**거리 스케일 계산**:

$$
s_{dist} = \frac{d_{target}}{d_{orig}} = \frac{2.7}{2.46} = 1.097
$$

**카메라 위치 변환**:

$$
\mathbf{C}_{new} = \mathbf{C}_{orig} \times s_{dist}
$$

```python
# 실제 계산
orig_cam_pos = [-0.084, 0.541, 1.929]
new_cam_pos = [-0.092, 0.594, 2.116]  # × 1.097
# 검증: sqrt(0.092² + 0.594² + 2.116²) = 2.7 ✓
```

### 5.4 Step 3: 이미지 스케일 계산

**v5 방식 (거리 보정 포함)**:

$$
s_{v5} = \frac{f_x^{target}}{f_x^{orig}} \times \frac{d^{orig}}{d^{target}} = \frac{549}{1632.3} \times \frac{2.46}{2.7} = 0.336 \times 0.911 = 0.307
$$

**v8 방식 (거리 보정 없음)**:

$$
s_{v8} = \frac{f_x^{target}}{f_x^{orig}} = \frac{549}{1632.3} = 0.336
$$

**v9 방식 (스케일 보정 없음)**:

$$
s_{v9} = \frac{512}{1152} = 0.444 \quad \text{(단순 리사이즈)}
$$

### 5.5 전체 카메라별 스케일 비교

| Camera | 거리 (norm) | v5 scale | v8 scale | v9 scale | 
|--------|-------------|----------|----------|----------|
| 0 | 2.46 | 0.307 | 0.336 | 0.444 |
| 1 | 4.14 | 0.541 | 0.353 | 0.444 |
| 2 | 3.64 | 0.454 | 0.337 | 0.444 |
| 3 | 3.40 | 0.430 | 0.342 | 0.444 |
| 4 | 3.18 | 0.400 | 0.339 | 0.444 |
| 5 | 3.06 | 0.380 | 0.335 | 0.444 |

---

## 6. Ray 계산 상세

### 6.1 Ray의 정의

**Ray**: 카메라 중심 $\mathbf{o}$에서 픽셀 $(u,v)$를 통과하는 직선

$$
\mathbf{r}(t) = \mathbf{o} + t \cdot \hat{\mathbf{d}}
$$

### 6.2 실제 계산 예시

```python
# Camera 0, 픽셀 (256, 256) - 이미지 중앙
u, v = 256, 256
fx, fy, cx, cy = 549, 549, 256, 256

# 1. Normalized 좌표
x_n = (256 - 256) / 549 = 0
y_n = (256 - 256) / 549 = 0

# 2. 카메라 좌표계 방향
d_cam = [0, 0, 1]  # 정면 방향

# 3. 회전 행렬 R
R = [[0.577, -0.817, -0.014],
     [0.003,  0.019, -1.000],
     [0.817,  0.577,  0.013]]

# 4. 월드 좌표계 방향
d_world = R^T @ [0, 0, 1] = [-0.014, -1.000, 0.013]

# 5. Ray Origin (카메라 위치)
o = [-0.092, 0.594, 2.116]  # after distance normalization

# 6. Plücker Moment
m = o × d_world = [0.594×0.013 - 2.116×(-1.000), 
                   2.116×(-0.014) - (-0.092)×0.013,
                   (-0.092)×(-1.000) - 0.594×(-0.014)]
    = [2.124, -0.029, 0.100]
```

---

## 7. 투영 비율 일관성

### 7.1 투영 비율 (Projection Ratio)

$$
\rho = \frac{f_x}{d}
$$

**의미**: 단위 거리당 픽셀 크기. 같은 $\rho$면 같은 3D 크기가 같은 픽셀 크기로 투영.

### 7.2 원본 데이터의 문제

```
Cam 0: ρ = 1632 / 246 = 6.63
Cam 1: ρ = 1557 / 414 = 3.76  ← 크게 다름!
```

→ 같은 물체가 다른 크기로 투영됨

### 7.3 정규화 후

$$
\rho_{new} = \frac{549}{2.7} = 203.3 \quad \text{(모든 뷰 동일)}
$$

---

## 8. 전처리 버전 비교

| 항목 | v5 | v8 | v9 |
|------|-----|-----|-----|
| $f_x = f_y$ | ✅ 549 | ✅ 549 | ✅ ~725 |
| 거리 정규화 | ✅ → 2.7 | ✅ → 2.7 | ✅ → 2.7 |
| 이미지 스케일 | fx비율 × 거리비율 | fx비율만 | 단순 리사이즈 |
| Centering | Centroid | Principal Point | ❌ 없음 |
| $c_x, c_y$ | 256 (강제) | 256 (강제) | 원본 비율 |

---

## 9. GS-LRM 전처리 필수 요소 (결론)

### 9.1 필수 (Must Have)

1. **$f_x = f_y$**: 정사각 픽셀 강제 (Ray 방향 일관성)
2. **거리 정규화**: Pretrained 분포와 일치 ($d = 2.7$)

### 9.2 권장 (Should Have)

1. **일관된 $f_x$**: 모든 뷰에서 동일한 FOV (549)
2. **이미지 스케일 보정**: 투영 크기 일관성

### 9.3 선택 (Nice to Have)

1. **Centering**: Object를 이미지 중앙에 배치
2. **Centroid 기반**: 마스크 무게중심 사용

---

*문서 버전: 2.0*
*작성일: 2026-01-13*
*프로젝트: FaceLift Mouse 3D Reconstruction*
