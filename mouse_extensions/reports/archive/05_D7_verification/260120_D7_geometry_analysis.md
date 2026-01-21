# D7 Preprocessing: Strict Multi-View Geometry Analysis

> **Version**: v2.0 (2026-01-20)
> **Scope**: 엄격한 multi-view geometry 관점에서 D4, D7, D7.1, D7.2 비교 분석

---

## Executive Summary

### 핵심 발견

| 버전 | PP 처리 | fy 처리 | 총 Ray Error | 권장 |
|------|---------|---------|--------------|------|
| **D4** | 256 강제 (오류) | 비례 스케일 (정확) | **5-13°** | X |
| **D7** | PP-shift (정확) | 549 강제 (오류) | **~0.4°** | 현재 사용 |
| **D7.1** | PP-shift (정확) | 개별 scale (정확) | **~0°** | **권장** |
| **D7.2** | PP-shift (정확) | 평균 scale (근사) | **~0.2°** | 대안 |

**결론**: D7.1이 기하학적으로 가장 정확

---

## 1. 데이터셋별 핵심 설정 비교

### 1.1 Intrinsics 처리

| 항목 | D4 | D6-1 | D7 | D7.1 | D7.2 |
|------|-----|------|-----|------|------|
| **fx** | 549 (고정) | 549±9.3 (변동) | 549 (고정) | 549 (고정) | ~548.1 (계산) |
| **fy** | 비례 (537~558) | 552±7 (변동) | **549 (강제!)** | 549 (정확) | ~550.0 (계산) |
| **cx** | 256 (강제!) | 208±6 (실제) | 256 (정확) | 256 (정확) | 256 (정확) |
| **cy** | 256 (강제!) | 187±15 (실제) | 256 (정확) | 256 (정확) | 256 (정확) |

### 1.2 Transform 방식

| 버전 | Scale 방식 | 수식 |
|------|------------|------|
| **D4** | fx 기준 단일 | `scale = target_fx / avg_orig_fx` |
| **D7** | fx 기준 단일 | `scale = target_fx / orig_fx` |
| **D7.1** | 개별 scale | `scale_x = target_fx/orig_fx`, `scale_y = target_fy/orig_fy` |
| **D7.2** | 평균 scale | `scale = (scale_x + scale_y) / 2` |

---

## 2. 기하학적 정확성 분석

### 2.1 Pinhole Camera Model

```
u = fx * X/Z + cx
v = fy * Y/Z + cy
```

**Ray Direction (pixel → 3D)**:
```
ray_x = (u - cx) / fx
ray_y = (v - cy) / fy
ray_z = 1.0
```

### 2.2 오류 유형별 영향

#### Type 1: PP (cx, cy) 오류 (D4의 문제)

```
실제 PP: (312, 287)
기록 PP: (256, 256)
오차: Δcx=56, Δcy=31

Ray error at pixel (u, v):
  Δray_x = Δcx / fx = 56 / 549 ≈ 0.102 rad ≈ 5.8°
  Combined: sqrt(5.8² + 3.2²) ≈ 6.6°
```

**영향**: 모든 픽셀에서 일정한 각도 오류 → **심각한 Ghosting**

#### Type 2: fy 오류 (D7의 문제)

```
원본: fx=1632.3, fy=1639.3 (aspect ratio 1.0043)
D7 scale: scale = 549 / 1632.3 = 0.3363

실제 변환된 fy: 1639.3 * 0.3363 = 551.3
기록된 fy: 549.0
오차: Δfy = 2.3

Ray error at edge (v=256 from center):
  Expected: ray_y = 256 / 551.3 = 0.4645
  Recorded: ray_y = 256 / 549.0 = 0.4663
  Error: 0.4% → ~0.24°
```

**영향**: 이미지 가장자리에서 작은 오류 → **미미한 Ghosting (LOW risk)**

### 2.3 정량적 비교

| 버전 | PP Error | fy Error | Max Ray Error | Ghosting Risk |
|------|----------|----------|---------------|---------------|
| **D4** | 37px avg | 0 | **5-13°** | **HIGH** |
| **D6-1** | 48-69px | 0 | **5-8°** | **HIGH** |
| **D7** | 0px | 2.5 avg | **0.4°** | LOW |
| **D7.1** | 0px | 0 | **~0°** | NONE |
| **D7.2** | 0px | ~1.2 | **~0.2°** | VERY LOW |

---

## 3. D7.1 vs D7.2 상세 비교

### 3.1 D7.1: Individual Scale (권장)

**방법**:
```python
scale_x = target_fx / orig_fx  # 549 / 1632 = 0.3363
scale_y = target_fy / orig_fy  # 549 / 1639 = 0.3350

affine_matrix = [
    [scale_x, 0, shift_x],
    [0, scale_y, shift_y]  # 다른 scale!
]

# 결과
fx = 549 (정확)
fy = 549 (정확)
```

**장점**:
- 기하학적으로 완벽
- fx, fy 모두 정확히 549
- GS-LRM pretrained 분포와 완전 일치

**단점**:
- 비등방성(anisotropic) scaling
- 원본 aspect ratio 1.0064 → 1.0으로 변환
- 수직 방향 ~0.6% 압축 (시각적으로 거의 감지 불가)

### 3.2 D7.2: Average Scale (대안)

**방법**:
```python
scale_x = target_fx / orig_fx  # 0.3363
scale_y = target_fy / orig_fy  # 0.3350
scale = (scale_x + scale_y) / 2  # 0.3356

affine_matrix = [
    [scale, 0, shift_x],
    [0, scale, shift_y]  # 같은 scale
]

# 결과
fx = orig_fx * scale = 1632.3 * 0.3356 = 547.9
fy = orig_fy * scale = 1639.3 * 0.3356 = 550.3
```

**장점**:
- 등방성(isotropic) scaling 유지
- 이미지 왜곡 없음
- Aspect ratio 보존

**단점**:
- fx, fy가 정확히 549가 아님
- GS-LRM pretrained 분포와 약간 불일치 (~0.2%)
- 하지만 GS-LRM은 fxfycxcy 텐서를 실제로 읽어 사용하므로 문제없을 수 있음

### 3.3 권장 사항

| 우선순위 | 방안 | 이유 |
|----------|------|------|
| **1** | D7.1 | 기하학적 완벽성, pretrained 일치 |
| **2** | D7.2 | 이미지 품질 보존, 기하학 근사 |
| **3** | D7 현재 | 실용적으로 충분 (0.4° error) |

---

## 4. 프레임 일관성 분석

### 4.1 뷰 간 일관성

| 항목 | D4 | D7 | D7.1/D7.2 |
|------|-----|-----|-----------|
| **같은 3D점 → 다른 뷰** | PP 오류로 불일치 | PP 정확, fy 미세 불일치 | 완전 일치 |
| **Multi-view triangulation** | 오류 | 근사 정확 | 정확 |
| **Epipolar constraint** | 위반 | 근사 만족 | 만족 |

### 4.2 프레임 간 일관성 (시간)

| 항목 | 모든 버전 |
|------|-----------|
| **카메라 파라미터** | 프레임마다 동일 (6개 뷰 고정) |
| **Transform** | 프레임마다 독립 적용 |
| **Temporal consistency** | 뷰별 처리이므로 보장됨 |

---

## 5. GS-LRM 관점 분석

### 5.1 Pretrained 모델 기대값

```
fx = fy = 549.36 (Objaverse)
cx = cy = 256.0
distance ~ 2.7 units
```

### 5.2 호환성 분석

| 버전 | fx 일치 | fy 일치 | PP 일치 | 종합 |
|------|---------|---------|---------|------|
| **D4** | O | X (변동) | O (강제) | 기하학 오류 |
| **D7** | O | O (강제) | O | 기하학 근사 |
| **D7.1** | O | O | O | **완벽** |
| **D7.2** | X (~548) | X (~550) | O | 근사 (GS-LRM이 실제 값 사용) |

---

## 6. 결론 및 권장 사항

### 6.1 핵심 결론

1. **D4의 PP 강제 설정은 치명적** (5-13° ray error → Ghosting)
2. **D7의 fy 강제 설정은 경미** (0.4° ray error → LOW risk)
3. **D7.1이 기하학적으로 완벽** (0° ray error)
4. **D7.2는 이미지 품질 보존 대안** (~0.2° ray error)

### 6.2 최종 권장

```
Production: D7.1 (기하학적 완벽성 우선)
Alternative: D7.2 (이미지 품질 보존 우선)
Current: D7 (실용적으로 충분, 0.4° error)
Deprecated: D4, D6-1 (PP 오류)
```

### 6.3 다음 단계

1. [ ] D7.1 전처리 구현 완성
2. [ ] D7.1 vs D7 실험 비교 (PSNR, Ghosting)
3. [ ] D7.2 전처리 구현 (선택적)
4. [ ] Background loss weight 설정 검토

---

*Generated: 2026-01-20*
