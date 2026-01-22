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

---

## 6. 스케일 변환 원리

### 6.1 핀홀 카메라 모델

43364
u = f_x \cdot \frac{X}{Z} + c_x, \quad v = f_y \cdot \frac{Y}{Z} + c_y
43364

### 6.2 3D 장면의 스케일 불변성

**스케일 변환 적용**:
- 3D 점: $\mathbf{P}' = s \cdot \mathbf{P}$
- 초점거리: '_x = f_x / s$



---

## 6. 스케일 변환 원리

### 6.1 핀홀 카메라 모델

$$
u = f_x \cdot \frac{X}{Z} + c_x, \quad v = f_y \cdot \frac{Y}{Z} + c_y
$$

### 6.2 3D 장면의 스케일 불변성

**스케일 변환 적용**:
- 3D 점: $\mathbf{P}' = s \cdot \mathbf{P}$
- 초점거리: $f'_x = f_x / s$

**증명**:
$$
u' = \frac{f_x}{s} \cdot \frac{sX}{sZ} + c_x = f_x \cdot \frac{X}{Z} + c_x = u
$$

> **결론**: 거리와 fx를 같은 비율로 조정하면 **이미지 불변**. 회전(R)은 유지, Translation(T)만 스케일링.

### 6.3 FaceLift 정규화 예시

| 항목 | 원본 | 정규화 | 비율 |
|------|------|--------|------|
| Translation | [10.34, 66.41, 236.70] | [0.11, 0.73, 2.60] | ×0.011 |
| fx | 1632 | 549 | ×0.336 |
| 평균 거리 | ~330 | 2.7 | ×0.008 |

**스케일 팩터**: $s = 2.7 / 330 \approx 0.008$

---

## 7. Affine vs Homography 변환

### 7.1 변환 비교

| 항목 | Affine (D7.1) | Homography (D8) |
|------|---------------|-----------------|
| **자유도** | 6 (scale, rotation, translation, shear) | 8 (perspective 포함) |
| **행렬** | 2×3 | 3×3 |
| **평행선** | 보존 ✅ | 보존 안됨 |
| **Skew 보정** | ❌ 불가 | ✅ 가능 |
| **OpenCV** | warpAffine | warpPerspective |

### 7.2 D7.1 Affine 변환

$$
\begin{bmatrix} u' \\ v' \end{bmatrix} =
\begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}
\begin{bmatrix} u \\ v \end{bmatrix} +
\begin{bmatrix} t_x \\ t_y \end{bmatrix}
$$

### 7.3 D8 Homography 변환 (Skew 보정)

$$
\mathbf{H} = \mathbf{K}_{target} \cdot \mathbf{K}_{orig}^{-1}
$$

**K_target (목표)**:
$$
\mathbf{K}_{target} = \begin{bmatrix}
548.99 & 0 & 256 \\
0 & 548.99 & 256 \\
0 & 0 & 1
\end{bmatrix}
$$

**K_orig (원본, skew 포함)**:
$$
\mathbf{K}_{orig} = \begin{bmatrix}
548.7 & 0.003 & 252.1 \\
0 & 551.5 & 257.8 \\
0 & 0 & 1
\end{bmatrix}
$$

**한 번에 보정되는 항목**:
1. Skew (s): 0.003 → 0
2. 비등방성: fx/fy ≈ 0.995 → 1.0
3. PP 이동: (cx, cy) → (256, 256)

### 7.4 D7.1 vs D8 정밀도 비교

| 항목 | D7.1 | D8 |
|------|------|-----|
| fx, fy | 549.0 (반올림) | 548.9937744 (정밀값) |
| Skew | 무시 (~0.9px) | Homography로 보정 |
| 변환 방식 | Affine (scale+shift) | Perspective (K_target × K_orig⁻¹) |
| Skew ray error | ~0.02° | 0° |

---

## 8. OpenCV 카메라 Convention

### 8.1 좌표축 정의

- **+X**: 오른쪽
- **+Y**: 아래쪽
- **+Z**: 카메라가 바라보는 방향 (전방)

### 8.2 C2W 행렬 해석

$$
\mathbf{C2W} = \begin{bmatrix}
\mathbf{R} & \mathbf{t} \\
\mathbf{0}^T & 1
\end{bmatrix}
$$

- `c2w[:3, 0]` = 카메라 X축 (world 좌표)
- `c2w[:3, 1]` = 카메라 Y축 (world 좌표)
- `c2w[:3, 2]` = 카메라 Z축 = **시선 방향** (world 좌표)
- `c2w[:3, 3]` = 카메라 위치 (world 좌표)

> **시각화 시**: `-c2w[:3, 2]`를 사용하면 카메라가 바라보는 방향을 화살표로 표시

---

## 9. 데이터 파이프라인 요약

```
MAMMAL 원본 (new_cam.pkl)
         │
         ▼
[1] 평균 거리 계산 (avg ≈ 330mm)
         │
         ▼
[2] 스케일 팩터 (2.7 / 330 ≈ 0.008)
         │
         ▼
[3] Translation 정규화 (T × scale)
         │
         ▼
[4] FaceLift Intrinsics 적용
    - fx = fy = 549
    - cx = cy = 256
         │
         ▼
[5] 이미지 변환
    - D7.1: Affine (warpAffine)
    - D8: Homography (warpPerspective)
         │
         ▼
[6] opencv_cameras.json 저장
```

---

*Updated: 2026-01-23*

---

## 10. GS-LRM 카메라 정규화 이론

> **출처**: GS-LRM 논문 분석 기반 정리

### 10.1 Object-level vs Scene-level 정규화

| 구분 | Object-level | Scene-level |
|------|--------------|-------------|
| **데이터** | Objaverse (합성) | RealEstate10K (실사) |
| **거리 설정** | 훈련: [1.5, 2.8] 무작위 | 정규화 필수 |
| **정규화** | 렌더링 시 고정 가능 | 복잡한 변환 필요 |
| **스케일** | 단일 오브젝트 | 다양한 장면 크기 |

### 10.2 Object-level: 카메라 거리 [1.5, 2.8]

#### 훈련 시 무작위 거리 사용 이유

1. **일반화 능력 향상**
   - $1.5$: 상대적으로 가까운 거리 → 디테일 학습
   - $2.8$: 상대적으로 먼 거리 → 전체 형태 학습
   - 다양한 스케일/관점에서 인식 및 재구성 능력 강화

2. **과적합 방지**
   - 고정 거리 훈련 → 해당 조건에 과적합 위험
   - 무작위성 → 일반적인 3D 재구성 원리 학습

3. **데이터 증강 효과**
   - 제한된 3D 오브젝트로 다양한 2D 입력 생성

#### 추론 시 거리 설정

> **GS-LRM 논문에서 명시적 언급 없음**

- 일반적 관행: 평가 시 **고정 거리** 사용 (예: $2.7$)
- 목적: 공정성(fairness) 및 재현성(reproducibility) 확보
- Instant3D [32] 등 관련 논문에서 표준화된 조건 사용

### 10.3 Scene-level: 카메라 포즈 정규화

#### 이론적 배경

NeRF 계열 모델이 정규화를 필요로 하는 이유:

1. **스케일 일관성**
   - 실제 장면: 스케일 매우 다양 (작은 방 vs 넓은 공원)
   - 일관된 스케일 범위(단위 스케일)로 정규화 필요

2. **좌표계 중심**
   - NeRF: 장면 중심이 원점 $[0,0,0]$ 근처에서 최적 작동
   - 위치 인코딩, 공간 분할(복셀/해시 그리드)이 특정 범위 최적화

3. **훈련 안정성**
   - 일관된 스케일/중심 → 안정적 학습, 빠른 수렴
   - 매 샘플 다른 조건 → 학습 불안정

#### 정규화 방법 (2단계)

**Step 1: 중심 이동 (Translation to Origin)**

$$
\mathbf{t}_{center} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{t}_i
$$

$$
\mathbf{t}'_i = \mathbf{t}_i - \mathbf{t}_{center}
$$

- 모든 카메라 위치의 평균(또는 중앙값) 계산
- 장면 중심을 월드 좌표계 원점으로 이동

**Step 2: 스케일 조정 (Scaling to Bounding Box)**

$$
d_{max} = \max_i \| \mathbf{t}'_i \|_2
$$

$$
s = \frac{\text{target\_radius}}{d_{max}}
$$

$$
\mathbf{t}''_i = s \cdot \mathbf{t}'_i
$$

- 목표: 모든 카메라가 $[-1, 1]^3$ 바운딩 박스 내 위치
- 최종 외부 파라미터: $\mathbf{P}''_{ext,i} = \begin{pmatrix} \mathbf{R}_i & \mathbf{t}''_i \\ \mathbf{0}^T & 1 \end{pmatrix}$

### 10.4 정규화 핵심 요약

| 항목 | Object-level | Scene-level |
|------|--------------|-------------|
| **목표 거리** | ~2.7 (추론 시) | 단위 구/박스 내 |
| **중심 이동** | 불필요 (렌더링 시 설정) | 필수 (평균 위치 → 원점) |
| **스케일** | 거리 범위로 조정 | 바운딩 박스로 정규화 |
| **회전** | 유지 | 유지 |

---

## 11. FaceLift Mouse 구현 비교 분석

### 11.1 현재 구현 방식 (camera_normalizer.py)

```python
# 핵심 로직
fx_scale = target_fx / avg_original_fx          # 549 / 1632
distance_scale = target_distance / avg_distance  # 2.7 / 330

# Intrinsics: fx, fy 스케일링
new_frame["fx"] = target_fx  # 549
new_frame["fy"] = frame["fy"] * fx_scale

# Extrinsics: Translation만 스케일링 (Rotation 유지)
c2w[:3, 3] = c2w[:3, 3] * distance_scale
```

### 11.2 GS-LRM 이론 vs FaceLift 구현 비교

| 항목 | GS-LRM 이론 | FaceLift 구현 | 일치 |
|------|-------------|---------------|------|
| **거리 정규화** | $d \rightarrow$ 단위 범위 | $d \rightarrow 2.7$ | ✅ |
| **중심 이동** | 평균 위치 → 원점 | ❌ 미구현 | ⚠️ |
| **스케일 팩터** | $s = \frac{target}{d_{max}}$ | $s = \frac{2.7}{\bar{d}}$ | ⚠️ |
| **Rotation 보존** | ✅ R 유지 | ✅ R 유지 | ✅ |
| **fx, fy 정규화** | 언급 없음 | ✅ fx=fy=549 | 추가 구현 |

### 11.3 비판적 분석

#### ✅ 장점 (현재 구현)

1. **Pretrained 호환성 우선**
   - GS-LRM pretrained 모델 분포(fx=549, d=2.7)와 일치
   - 즉시 fine-tuning 가능

2. **fx=fy 강제**
   - 정사각 픽셀 보장 → Ray 방향 일관성
   - GS-LRM 이론에서 미언급이나 실용적 필수

3. **단순한 스케일링**
   - 평균 거리 기반 단일 스케일 팩터
   - 구현 간단, 디버깅 용이

#### ⚠️ 개선 여지

1. **중심 이동 미구현**
   ```python
   # GS-LRM 이론 권장
   t_center = np.mean(camera_positions, axis=0)
   c2w[:3, 3] = c2w[:3, 3] - t_center  # 중심 이동
   c2w[:3, 3] = c2w[:3, 3] * scale     # 스케일링
   ```
   - 현재: Object 위치 그대로 유지
   - 문제: 원점에서 벗어난 장면에서 잠재적 이슈

2. **평균 vs 최대 거리**
   ```
   현재: s = 2.7 / mean(distances)
   이론: s = target / max(distances)
   ```
   - 평균 사용 시 일부 카메라가 범위 초과 가능
   - Mouse 데이터: 카메라 거리 분산 작아 실질적 차이 미미

3. **훈련 시 거리 다양성 미적용**
   - GS-LRM: [1.5, 2.8] 무작위 거리로 일반화 강화
   - FaceLift Mouse: 고정 거리 2.7
   - 데이터 증강 기회 상실

### 11.4 Mouse 데이터 특성 고려

| 특성 | 값 | 영향 |
|------|-----|------|
| **카메라 수** | 6대 (고정) | 중심 추정 안정적 |
| **거리 분산** | 낮음 (std ~0.5) | 평균/최대 차이 미미 |
| **Object 위치** | 원점 근처 | 중심 이동 불필요 |
| **스케일** | 이미 정규화됨 | 추가 처리 불필요 |

**결론**: Mouse 데이터 특성상 현재 구현이 **실용적으로 충분**.
Scene-level 일반화가 필요한 경우(다양한 장면)에만 중심 이동 추가 권장.

### 11.5 권장 사항

| 상황 | 권장 |
|------|------|
| **현재 Mouse 학습** | 현재 구현 유지 (fx=549, d=2.7) |
| **새로운 Scene 데이터** | 중심 이동 + 바운딩 박스 정규화 추가 |
| **일반화 강화** | 훈련 시 거리 augmentation [2.0, 3.0] 고려 |

---

*Updated: 2026-01-23*
*Reference: GS-LRM Paper Analysis*


---

## 12. 카메라 시각화 구현

### 12.1 시각화 구성 요소

| 요소 | 설명 | C2W 행렬 사용 |
|------|------|---------------|
| **카메라 위치** | 3D 공간에서 카메라 원점 | `c2w[:3, 3]` |
| **시선 방향** | 카메라가 바라보는 방향 화살표 | `c2w[:3, 2]` |
| **카메라 Frustum** | 시야각(FOV) 표현 피라미드 | intrinsics + c2w |
| **이미지 Billboard** | 카메라 앞에 배치된 RGB 이미지 | c2w + focal length |

### 12.2 구현 상세

#### 12.2.1 카메라 위치 (Position)

```python
# C2W에서 직접 추출
camera_position = c2w[:3, 3]  # [x, y, z]

# W2C에서 계산 (역변환)
R = w2c[:3, :3]
t = w2c[:3, 3]
camera_position = -R.T @ t
```

#### 12.2.2 시선 방향 (Viewing Direction)

```python
# OpenCV convention: +Z가 카메라 전방
viewing_direction = c2w[:3, 2]  # 카메라 Z축

# 시각화용 화살표 (카메라 위치에서 시작)
arrow_start = camera_position
arrow_end = camera_position + viewing_direction * arrow_length
```

> **주의**: OpenCV에서 카메라는 +Z 방향을 바라봄. 일부 렌더러(OpenGL)는 -Z.

#### 12.2.3 카메라 Frustum (FOV 피라미드)

```python
def compute_frustum_corners(c2w, fx, fy, cx, cy, width, height, near=0.1, far=1.0):
    """
    Frustum 8개 꼭짓점 계산 (near/far plane 각 4개)
    """
    # 이미지 모서리의 normalized 좌표
    corners_2d = [
        [0, 0],           # top-left
        [width, 0],       # top-right
        [width, height],  # bottom-right
        [0, height],      # bottom-left
    ]

    frustum_points = []
    for z in [near, far]:
        for u, v in corners_2d:
            # Pixel → Normalized → Camera → World
            x_n = (u - cx) / fx
            y_n = (v - cy) / fy

            # Camera 좌표 (depth = z)
            p_cam = np.array([x_n * z, y_n * z, z, 1])

            # World 좌표
            p_world = c2w @ p_cam
            frustum_points.append(p_world[:3])

    return np.array(frustum_points)
```

#### 12.2.4 이미지 Billboard

```python
def place_image_billboard(c2w, fx, image, distance=0.5):
    """
    카메라 앞 distance 위치에 이미지 평면 배치
    """
    H, W = image.shape[:2]

    # Billboard 크기 (실제 투영 크기)
    billboard_width = W * distance / fx
    billboard_height = H * distance / fx

    # Billboard 중심 (카메라 앞 distance 위치)
    center = c2w[:3, 3] + c2w[:3, 2] * distance

    # Billboard 좌표축
    right = c2w[:3, 0] * billboard_width / 2
    up = -c2w[:3, 1] * billboard_height / 2  # Y 반전 (이미지 좌표계)

    # 4개 모서리
    corners = [
        center - right + up,   # top-left
        center + right + up,   # top-right
        center + right - up,   # bottom-right
        center - right - up,   # bottom-left
    ]

    return corners, image
```

### 12.3 시각화 예시 코드 (Matplotlib)

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_cameras(c2w_list, colors=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, c2w in enumerate(c2w_list):
        pos = c2w[:3, 3]
        direction = c2w[:3, 2]

        # 카메라 위치 (점)
        color = colors[i] if colors else 'blue'
        ax.scatter(*pos, c=color, s=50, label=f'Cam {i}')

        # 시선 방향 (화살표)
        ax.quiver(*pos, *direction, length=0.3, color=color, arrow_length_ratio=0.2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
```

### 12.4 FaceLift Mouse 카메라 배치

```
Top View (XY plane):
         Y
         ^
         |
    [2]  |  [1]
      \  |  /
       \ | /
    [3]--*--[0]  → X
       / | \
      /  |  \
    [4]  |  [5]
         |

Side View (XZ plane):
         Z (up)
         ^
         |   [5] elevation +30.8° ⚠️
        /|\
       / | \
      /  |  \
   [0-4] |   → X
         |
```

**View 5 특이점**: 다른 뷰(elevation ~0°) 대비 +30.8° 높은 각도

---
