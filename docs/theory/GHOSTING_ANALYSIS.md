> Parent: [[INDEX]] > Theory

# Ghosting Analysis & Solution Strategy

**Version**: 2.0
**Date**: 2026-01-27
**Merged from**: GHOSTING_ANALYSIS_REPORT.md, GHOSTING_SOLUTION_STRATEGY.md

---

## 1. 현상 정의

### Ghost Type A: 근접 고스트
- GT 형상 근처에 약간 다른 크기/위치의 이미지 중첩
- **원인**: Intrinsics scaling 불일치, depth ambiguity, PP 미세 불일치

### Ghost Type B: 원거리 고스트
- 90도 회전 위치에 동일 포즈 이미지 배치
- **원인**: View-dependent SH 과적합, Gaussian 3D 위치 오류

---

## 2. 근본 원인: Plucker Ray 분포 불균형

**GS-LRM 입력**: `[RGB(3ch) | Plucker(6ch)] = 9ch per pixel`

**Plucker Coordinates**:
```
P = (d, m) = (d, o × d)
- d: ray 방향 (normalized)
- m: moment (원점에서 ray까지의 수직거리 × 방향)
```

**Mouse 데이터 문제**: Object coverage 2-3%
- 97% 픽셀이 배경 ray → View 간 Plucker 분포 거의 동일
- Transformer가 view 구분 실패 → 독립적 Gaussian 생성 → Ghosting

### PP Mismatch 영향
```
ray_error ≈ arctan(PP_error / focal_length)
예: PP 37px, f=549 → ~3.9° → Ghosting
```

---

## 3. Training vs Validation 차이 분석

| 항목 | Training | Validation | 영향 |
|------|----------|------------|------|
| Turntable 해상도 | input 해상도 사용 | input 해상도 사용 | ✅ 동일 (검증됨) |
| Augmentation | 활성화 | 비활성화 | ⚪ 정상 차이 |
| View selection | 고정 | 고정 | ✅ 동일 |

> **Note**: 해상도 불일치 가설은 코드 분석 후 기각됨 (2026-01-28). Training/Validation 모두 input 해상도 사용.

---

## 4. 데이터 전처리 영향

### zoom_center_mode 영향

| Mode | PP | 특성 |
|------|-----|------|
| `"image"` | 256 고정 | ✅ Pretrained 호환, MVG 정확 |
| `"object"` | 가변 | ⚠️ PP 분산 → Ray 오류 → Ghosting |

**필수**: `zoom_center_mode: "image"`

### M3 시리즈 특성

| Dataset | Zoom | Clipping | 상태 |
|---------|------|----------|------|
| M3_1 | Global | 0% | ✅ 안전 |
| M3_2 | Per-sample [1.0, 1.8] | 6.5% | ⚠️ |

---

## 5. 해결 전략

### 5.1 Object-Centered Zoom (핵심)

```
D7_1 기하학적 정확성 + Object Centering/Scaling
= D7_1_centered (새 프리셋)
```

**목표**:
- FG Coverage: 2% → **5%+**
- Center Offset: 101px → **<15px**
- Ray Error: ~0° 유지
- fx=549, dist=2.7 유지

**파이프라인**:
1. Object Center Detection (3D triangulation)
2. Object-Centered Crop (각 뷰에서 back-project)
3. Intrinsics Adjustment (crop/zoom 반영)
4. Camera Normalization (fx→549, cx/cy→256)

### 5.2 Loss 함수 최적화 (문헌 기반)

```yaml
training:
  losses:
    mask_mode: gt                  # GT mask (안정적)
    normalize_by_mask: true        # Pose Splatter
    alpha_loss_weight: 0.2         # LGM (MSE)
    alpha_loss_type: mse
    bg_loss_weight: 0.3            # Object-Centric 2DGS
    perceptual_loss_weight: 0.5
```

### 5.3 추가 수단

- **Opacity Regularization**: `opacity_reg_weight: 0.01, type: entropy`
- **Input Views 증가**: 4 → 5-6
- **View Augmentation** 활성화

---

## 6. 문헌 근거

| 방법론 | Object Coverage | Centering | Loss |
|--------|-----------------|-----------|------|
| LRM | - | 이미지 중앙 crop | - |
| Objaverse Render | 50-80% | 중앙 정렬 | - |
| LGM | - | - | MSE α |
| Pose Splatter | - | - | Normalized L1 |
| Object-Centric 2DGS | - | - | BG penalty |

### 핵심 인용
- **LRM**: "We crop and resize images to center the object with adjusted camera parameters."
- **LGM**: "MSE loss on both RGB and alpha for faster convergence of the shape."
- **Pose Splatter**: "Division by mask sum normalizes gradients by foreground area."

---

## 7. 예상 결과

| Metric | D7_1 현재 | D7_1_centered 예상 |
|--------|-----------|-------------------|
| Val PSNR | ~20 | 24-26 |
| FG Coverage | 2.16% | 5%+ |
| Center Offset | 101px | <15px |
| Ghosting | 심각 | 경미/없음 |

---

## See Also

- [[MASK_GUIDE]] - 마스크 시스템 상세
- [[FLOATER_ARTIFACT_ANALYSIS]] - Floater 분석
- [[PP_FX_MVG_ANALYSIS]] - PP/MVG 이론
- [[CENTER_ESTIMATION]] - 3D Triangulation

---

*FaceLift Mouse | Ghosting Analysis v2.0 | Last updated: 2026-01-27*
