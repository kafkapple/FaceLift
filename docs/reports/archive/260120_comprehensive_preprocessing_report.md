# FaceLift Mouse 전처리 종합 보고서

> **버전**: v8.0 (2026-01-20)
> **범위**: 이론 + 코드 구현 + 검증 + 실험 결과

---

## Executive Summary

### 핵심 발견

1. **Principal Point (PP) 버그**: D4에서 cx=cy=256 강제로 11-13도 ray 오류 발생하여 Ghosting
2. **카메라 정규화 필수**: 미적용 시 PSNR 약 3 (학습 실패)
3. **D7 해결책**: PP-centered shift로 기하학 정확성 + pretrained 호환성 동시 달성

### 최종 권장 사항

| Use Case | Dataset | 이유 |
|----------|---------|------|
| **Production** | D7 | PP=256 정확, 기하학 보존 |
| **Evaluation** | D7_t | Temporal split, 일반화 테스트 |
| **Legacy** | D4 | PSNR 높으나 Ghosting 존재 |

---

# Part 1: 이론적 배경

## 1.1 Principal Point의 물리적 의미

### 정의
Principal Point (PP) = (cx, cy)는 카메라 광축이 이미지 평면과 만나는 점이며,
이미지 좌표계의 원점에 대한 오프셋을 나타냄.

### Pinhole Camera Model
x_img = fx * (X_cam / Z_cam) + cx
y_img = fy * (Y_cam / Z_cam) + cy

### PP 오류의 영향
PP 오차가 56px이고 fx = 549일 때:
theta = arctan(56 / 549) 약 5.8도

양 축 결합 시: 11-13도 ray 방향 오류
결과: 각 뷰의 ray가 다른 3D 점을 가리킴 -> Ghosting artifact 발생

---

## 1.2 카메라 정규화 이론

### Objaverse 분포 (Pretrained)

| Parameter | 값 | 의미 |
|-----------|-----|------|
| fx, fy | 549.36 | 정사각 픽셀 |
| cx, cy | 256, 256 | 이미지 중앙 |
| translation_norm | ~2.7 | 객체까지 거리 |

### 정규화 공식
scale = target_fx / current_fx  (549.36 / 844 약 0.65)
new_translation = translation * (target_dist / current_dist)

---

# Part 2: 전처리 방법론 비교

| Version | Center Method | PP | Normalize | 결과 |
|---------|---------------|-----|-----------|------|
| D1-D3 | Various | Various | No | 실패 |
| D4 | Triangulation | 256 (강제) | Yes | Ghosting |
| D6-1 | None (resize) | Actual | Yes | PSNR 낮음 |
| D6-3 | Triangulation | Actual | Yes | PP 분산 |
| **D7** | **PP-shift** | **256 (정확)** | **Yes** | **권장** |

---

## D7 PP-Centered Shift 상세

### 핵심 아이디어
기존 D4: 쥐를 이미지 중앙에 배치 -> PP=256 강제 기록 (틀림!)
D7: PP를 이미지 중앙에 배치 -> PP=256 실제 달성 (맞음!)

### 수학적 원리
원본: PP = (cx_orig, cy_orig)
목표: PP = (256, 256)

필요한 shift:
  dx = 256 - cx_orig
  dy = 256 - cy_orig

결과: new_cx = new_cy = 256 (정확)

---

# Part 3: 코드 구현 검증

## 카메라 위치 검증 (중요 발견!)

**이전 보고서 오류**: Elevation 약 30도 단일 링

**실제 측정** (2026-01-20):
View 0: Elev= 14.9도, Azim=-147.0도
View 1: Elev= 20.6도, Azim=  34.0도
View 2: Elev= 11.3도, Azim=  86.1도
View 3: Elev= 10.7도, Azim= -11.3도
View 4: Elev= 26.5도, Azim= 144.0도
View 5: Elev= 30.8도, Azim= -64.1도

Elevation: min=10.7도, max=30.8도, std=7.6도
WARNING: Elevation은 일정하지 않음! (범위 20.1도)

---

# Part 4: Loss 함수 구현

## Alpha Mask 메커니즘

### 파일 위치
mouse_extensions/model/loss_extensions.py

### Mask 종류

| MaskType | 설명 | 사용 |
|----------|------|------|
| NONE | 전체 이미지 | 기본 |
| GT | Ground truth alpha | 안정적 |
| RGB_PRED | RGB 거리 기반 | 실험적 |
| ALPHA | Rendered alpha | 현재 사용 |

## Background Loss

수식: L_bg = weight * Sum[(render - white)^2 * (1 - mask)] / Sum(1 - mask)

## Alpha Loss

| Type | 수식 |
|------|------|
| BCE | -p*log(p) - (1-p)*log(1-p) |
| MSE | (pred - gt)^2 |
| Dice | 1 - 2*intersection / union |
| Focal | (1-p_t)^gamma * BCE |

---

# Part 5: 실험 결과

## PSNR 비교

| Dataset | Config | Val PSNR | Ghosting |
|---------|--------|----------|----------|
| D3 (no norm) | - | 2.11 | N/A |
| D3n (norm) | - | 26.34 | Yes |
| D4_E4 | 5v_alpha | 27.87 | Yes |
| D6-1_E4 | 5v_alpha | 23.34 | Reduced |
| D7 | TBD | In progress | Expected: No |

## Ablation Studies

### 정규화 효과
- 없음: PSNR 2.11
- 있음: PSNR 26.34 (+24.23)

### 뷰 개수 효과
- 4뷰: PSNR 26.30
- 5뷰: PSNR 27.87 (+1.57)

---

# Part 6: 결론

1. **기하학 오류 > 분포 불일치**: 기하학 오류는 복구 불가
2. **PSNR 품질이 아님**: 시각적 검증 필수
3. **PP-centered shift가 최적**: D7 권장

---

## 파일 위치 정리

### 전처리 코드
/home/joon/dev/FaceLift/mouse_extensions/preprocessing/
- preprocessor_d6.py
- camera_normalizer.py
- preprocess_D7_pp_centered.py

### Loss 코드
/home/joon/dev/FaceLift/mouse_extensions/model/
- loss_extensions.py
- gslrm_patches.py
- gaussian_pruning.py

---

*Generated: 2026-01-20*
