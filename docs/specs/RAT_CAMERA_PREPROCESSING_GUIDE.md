# Rat Camera Preprocessing Guide

> **Version**: 2.0 | **Date**: 2026-04-06
> **Audience**: Research team — camera geometry + GS-LRM preprocessing 이해 목적
> **Prerequisite**: GS-LRM 기본 구조 이해 (multi-view → 3D Gaussians)

---

## 1. GS-LRM이 카메라 파라미터를 사용하는 방법

### 1.1 Plucker Ray: 모델의 눈

GS-LRM은 이미지 픽셀마다 **Plucker ray**를 계산하여 "이 픽셀이 3D 공간에서 어디를 보고 있는가"를 인코딩합니다.

```
각 픽셀 (u, v)에 대해:

  ray_direction = normalize([ (u - cx) / fx,     ← fx가 클수록 ray가 좁게 퍼짐
                               (v - cy) / fy,
                               1.0 ])

  ray_origin = camera_position                     ← 카메라 위치 (distance에 의존)

  plucker = (ray_direction, ray_origin × ray_direction)
```

**핵심**: 모델은 `fx`와 `camera_position` 두 정보를 모두 사용합니다.
- `fx` → ray가 퍼지는 각도 (FOV) 결정
- `camera_position` → ray의 시작점 결정

### 1.2 Pretrained 모델의 기대값

GS-LRM은 인간 얼굴 데이터로 pretrained 되었습니다:

```
Pretrained 학습 환경:
  - fx = 549          (중간 정도의 FOV)
  - distance = 2.7    (카메라-물체 거리)
  - cx = cy = 256     (이미지 중심)
  - 이미지 = 512 × 512 px
```

이 값들은 서로 **물리적으로 일관**됩니다:
- 거리 2.7에서 fx=549로 찍으면 → 특정 크기의 물체가 특정 픽셀 수를 차지
- 모델은 이 조합의 Plucker ray 패턴을 "정상"으로 학습함

---

## 2. Mouse 전처리 파이프라인 (M5)

### 2.1 원본 데이터

```
Mouse 원본 이미지:
  - 해상도: 1920 × 1200 px
  - fx_original ≈ 1936 (좁은 FOV, telephoto 렌즈)
  - 카메라 6대, 이미 mouse 중심 주변에 배치
  - mouse FG coverage: 원본에서 ~2.5%
```

### 2.2 Affine Transform (핵심 단계)

Mouse M5는 **이미지와 카메라 파라미터를 동시에** 변환합니다:

```
Step 1: fx 정규화
  scale = target_fx / original_fx = 549 / 1936 = 0.284

Step 2: 이미지 축소 (같은 scale 적용)
  new_image = affine_transform(original_image, scale=0.284)
  → 1920×1200이 ~545×340 영역으로 축소
  → 512×512 프레임 안에 배치

Step 3: PP (Principal Point) 이동
  cx, cy → 256, 256 (이미지 중심으로)

결과:
  - fx = 549 ✅ (pretrained와 일치)
  - 이미지 내 mouse 크기도 0.284배 축소 ✅
  - → 물리적으로 일관: "렌즈를 바꾼 것"과 동일한 효과
```

**왜 일관적인가**: 실제로 telephoto 렌즈(fx=1936)를 normal 렌즈(fx=549)로 교체하면,
같은 거리에서 물체가 0.284배 작게 보입니다. M5는 이것을 **이미지와 fx를 동시에 변환**하여
시뮬레이션합니다.

```
[Mouse Affine Transform — 이미지+fx 동시 변환]

  Original (fx=1936)          After Affine (fx=549)
  ┌──────────────┐            ┌──────────────┐
  │              │            │              │
  │    ██████    │  scale     │              │
  │    ██████    │  ────→     │     ██       │
  │    ██████    │  0.284x    │              │
  │              │            │              │
  └──────────────┘            └──────────────┘
  mouse = ~2.5%               mouse = ~2.5% (비율 유지)
  fx = 1936                   fx = 549
```

### 2.3 카메라 위치

Mouse 카메라는 전처리 과정에서 이미 원점 중심으로 정규화됩니다:

```
Mouse 카메라 배치:
  - 6대 카메라가 원점(mouse 위치) 주변에 배치
  - 원점에서 카메라까지 거리 ≈ 2.7 (pretrained와 일치)
  - centroid = [0, 0, 0] (이미 원점)
```

### 2.4 최종 결과

```
Mouse M5 최종:
  fx = 549      ✅ pretrained 일치
  distance = 2.7  ✅ pretrained 일치
  coverage = 2.5% (이미지 대비 mouse 크기)
  → Plucker ray 분포가 pretrained와 거의 동일
  → Fine-tuning이 빠르고 안정적
```

---

## 3. Rat 전처리 파이프라인 (현재)

### 3.1 원본 데이터

```
Rat 원본 이미지:
  - 해상도: 1920 × 1200 px (s-DANNCE 카메라)
  - fx_original ≈ 2268 (telephoto)
  - 카메라 6대, 넓은 arena를 촬영
  - rat은 arena 바닥에 있음 → 프레임에서 매우 작음
```

### 3.2 Zero-Pad + Resize (현재 방식)

```
Step 1: 정사각형 패딩
  1920 × 1200 → 1920 × 1920 (위아래 흰색 패딩)

Step 2: 리사이즈
  1920 × 1920 → 512 × 512

Step 3: fx 재계산
  fx_new = 2268 × (512 / 1920) = 605

결과:
  - fx = 605 ❌ (pretrained 549 대비 +10%)
  - 이미지 내 rat 크기 = 원본 비율 그대로
  - coverage = 1.5% (mouse 2.5%보다 작음)
```

**핵심 차이**: Mouse는 fx를 549로 **정규화**하면서 이미지도 함께 축소했지만,
Rat은 fx를 **그대로 유지** (605)하고 이미지도 원본 비율 그대로입니다.

```
[Rat Zero-Pad — 이미지 변환 없이 fx도 그대로]

  Original (1920×1200)        After Pad+Resize (512×512)
  ┌──────────────┐            ┌──────────────┐
  │              │            │   (padding)  │
  │       ·      │  pad+      │              │
  │    (rat)     │  resize    │       ·      │
  │              │  ────→     │    (rat)     │
  │              │            │              │
  └──────────────┘            │   (padding)  │
  rat = 작음                  └──────────────┘
  fx = 2268                   rat = 여전히 작음 (1.5%)
                              fx = 605
```

### 3.3 카메라 배치 (Mouse와의 결정적 차이)

```
[Mouse 카메라 배치]                 [Rat 카메라 배치]

        C1   C2                            C1 C2 C3
       / | / |                              |||
      /  |/  |                              |||
     ●  mouse                               ||| ← 좁은 baseline (~0.4)
      \  |\  |                              |||
       \ | \ |                              |||
        C3   C4                             |||
                                            ↓↓↓
  카메라 거리: ~2.7                          ↓↓↓  (~6.0 거리)
  카메라 간격: ~2.0 (넓은 baseline)          ↓↓↓
  convergence ≈ origin                      ●  rat (arena 바닥)

  → 원점에서 카메라까지: 2.7             → 원점에서 카메라까지: 2.6
  → 카메라에서 mouse까지: ~2.7            → 카메라에서 rat까지: ~6.0
  → centroid ≈ [0,0,0]                  → centroid ≈ [0, 2.6, 0.5]
                                         → convergence ≈ [0.1, -3.1, -1.0]
```

**핵심 발견**: 
- Mouse: 카메라 centroid ≈ 원점 ≈ mouse 위치 → 모든 것이 원점 중심
- Rat: 카메라 centroid는 원점 근처(2.6)이지만, rat(convergence point)은 원점 반대편(3.3)
- 카메라-rat 실제 거리: **~6.0** (pretrained 2.7의 2.2배)

---

## 4. 실험 히스토리: v1→v5 (7회 실험, 6회 실패)

### 전체 버전 매트릭스

| Ver | 데이터 | fx | Despill | Distance | 핵심 변경 | 결과 | 교훈 |
|:---:|--------|:---:|:---:|:---:|------|:---:|------|
| **v1** | RAT1 1001fr | 605 | ❌ | 2.7 (전처리) | baseline | **17.49 dB** | — |
| v2 | RAT2 2967fr | 605 | ❌ | 2.7 (전처리) | +clip_xyz | ❌ 3D 손상 | clip_xyz 위험 |
| v3 | RAT2 despill | 605 | ✅ | centroid recenter | +recenter | ❌ dist 붕괴 | 밀집 카메라 recenter 금지 |
| v4 | RAT2 despill | 605 | ✅ | convergence+2.7 | +convergence | ❌ IoU=0 | clip_xyz 확정 금지 |
| v4b | RAT2 despill | 605 | ✅ | convergence+2.7 | -clip_xyz | ❌ PSNR 0.9 | **fx-distance 정합성 필수** |
| v5a | RAT2 fxnorm | **549** | ❌ | 2.7 (전처리) | +fxnorm, -despill | ❌ 녹색 | **despill 누락 금지** |
| **v5** | RAT2 despill+fxnorm | **549** | ✅ | 2.7 (전처리) | despill→fxnorm | **준비 완료** | 전 교훈 반영 |

### 4.1 v3: Centroid Recentering → 거리 붕괴

```
v3 접근: "카메라 centroid를 원점으로 옮기자"

  Before:                           After centroid shift:
  카메라 centroid = [0, 2.6, 0.5]   카메라 centroid = [0, 0, 0]
  카메라 간 거리: 0.3~0.5           카메라 간 거리: 0.3~0.5 (변함없음)
  원점에서 카메라: 2.6              원점에서 카메라: 0.435 ← ❌ !!!

문제: 6대 카메라가 좁게 모여있어서 (inter-camera ~0.4),
      centroid를 원점으로 옮기면 카메라가 원점에 붙어버림.
      pretrained 기대(2.7) 대비 6.2배 가까움 → 모델 붕괴
```

### 4.2 v4: Convergence Recentering + clip_xyz → 즉시 붕괴

```
v4 접근: "카메라 시선이 교차하는 점(convergence)을 원점으로 옮기자"

  After convergence shift:
  원점 = convergence point (rat 위치)
  원점에서 카메라: 5.96 ← 자연스러운 거리 보존!
  
  + distance normalization: 5.96 → 2.7 (fx는 유지)
  
  문제: clip_xyz=true가 Gaussian 위치를 [-1, 1]로 제한.
        scene이 올바르게 원점에 있지만, depth prediction이
        fx=605와 distance=2.7의 불일치로 인해 
        Gaussian이 [-1,1] 밖에 예측됨 → clipping → IoU=0
```

### 4.3 v4b: clip_xyz=false → 초기 안정 후 붕괴

```
v4b 접근: v4 + clip_xyz 비활성화

  convergence recentering ✅ (scene at origin)
  distance = 2.7 ✅
  fx = 605 (10% 불일치, but no clipping)
  clip_xyz = false ✅

  결과: step 1-13 IoU 0.22-0.27 → step ~60 IoU=0.000 붕괴
  → fx-distance 불일치가 clip_xyz 없이도 학습을 방해
  → 3모델 audit에서 "fx-distance 정합성"이 근본 원인으로 확정
```

### 4.4 v5: despill + fx 정규화 (mouse 방식 적용) — ✅ 준비 완료

```
v5 접근: 2-stage 전처리
  Stage 1: despill.py (green spill 제거, G/R 0.977→0.798)
  Stage 2: normalize_fx.py (fx=605→549 affine, 이미지 0.908x 축소)

  ⚠️ v5a 실패: fxnorm WITHOUT despill → 녹색 재구성
  → Stage 순서 중요: despill(색상) → fxnorm(기하) (MoA 6/6 합의)

  fx = 549 ✅ (pretrained와 정확히 일치)
  recenter_cameras = false (v1과 동일, v3 실패 교훈)
  clip_xyz = false (v2 3D 손상 방지)
  target_camera_distance = 0 (sdannce_to_gslrm.py가 이미 2.7 정규화)
  green G/R = 0.810 ✅ (despill 보존 확인)

  데이터셋: rat2_s1despill_s2fxnorm (2967 frames)
  상태: 전처리 완료, 학습 준비 완료
```

### 4.5 v5a: fxnorm WITHOUT despill → 녹색 재구성

```
v5a 접근: sdannce_to_gslrm.py --normalize_fx로 원본에서 직접 재생성
  → 디렉토리명 gslrm_format_rat2_despilled_fxnorm (이름에 "despilled" 포함)
  → 실제로 despill.py 미적용, G/R = 0.975 (원본 수준)
  
  결과: 학습 시 녹색 tinted 3D 재구성
  원인: 2-stage 파이프라인에서 stage 1 (despill) 누락
  → MoA 6/6 합의: despill(색상) → fxnorm(기하) 순서 필수
  → 잘못된 데이터셋 삭제, rat2_s1despill_s2fxnorm으로 재생성
```

---

## 5. fx-distance 정합성 문제 (3모델 Audit 핵심 지적)

### 5.1 물리적 의미

실제 카메라에서:
```
  가까이 가면 (distance ↓) → 물체가 크게 보임 → 넓은 FOV 필요 (fx ↓)
  멀어지면   (distance ↑) → 물체가 작게 보임 → 좁은 FOV 충분 (fx ↑)
```

**정합적 조합** (물리적으로 일관):
```
  distance=2.7, fx=549  ← pretrained (기준)
  distance=6.0, fx=605  ← rat 원본 (자연스러운 관계)
```

**부정합 조합** (v4b 현재):
```
  distance=2.7, fx=605  ← ❌ "가까이 있는데 FOV가 좁다"
```

### 5.2 Plucker Ray에 미치는 영향

```
Pretrained (fx=549, d=2.7):
  이미지 가장자리 픽셀의 ray angle = arctan(256/549) = 25.0°

v4b (fx=605, d=2.7):
  이미지 가장자리 픽셀의 ray angle = arctan(256/605) = 22.9°

차이: 2.1° (약 8%)
```

이 2.1° 차이는 모델이 "이 카메라의 FOV가 생각보다 좁다"고 해석합니다.
Fine-tuning으로 적응 가능한 범위이지만, 모델의 depth prediction 초기 정확도에 영향을 줍니다.

### 5.3 Mouse는 왜 이 문제가 없는가

```
Mouse M5:
  원본 fx = 1936 → 이미지와 함께 축소 → fx = 549
  이미지 축소 비율 = 549/1936 = 0.284

  이것은 "렌즈를 교체한 것"과 동일:
  - 이미지 내 mouse 크기도 0.284배 축소 ✅
  - fx도 0.284배 축소 ✅
  - 물리적으로 일관 ✅

Rat v4b:
  원본 fx = 605 → 이미지는 변경 없음 → fx = 605 (그대로)
  좌표계만 스케일링 (distance 6.0 → 2.7)

  이것은 "카메라를 물리적으로 이동한 것":
  - 이미지 내 rat 크기는 변경 없음 ❌ (변하지 않음)
  - fx도 변경 없음 ❌
  - 하지만 distance는 변경됨 → 물리적 불일관 ⚠️
```

---

## 6. v5 vs M5 비교 분석 (2026-04-06 검증)

### 6.1 정량 비교

| 항목 | Mouse M5 (기준) | Rat v5 | 일치? | 비고 |
|------|:---:|:---:|:---:|------|
| **fx** | 549.0 | 549.0 | ✅ | affine 정규화 |
| **fy** | 549.0 | 549.0 | ✅ | square pixel |
| **cx, cy** | 256.0, 256.0 | 256.0, 256.0 | ✅ | 이미지 중심 |
| **avg distance** | 2.700 | 2.700 | ✅ | sdannce_to_gslrm.py 내부 정규화 |
| **distance range** | 2.593-2.799 | 2.456-2.896 | ⚠️ | rat 분산 더 큼 |
| **centroid** | [0, 0, 0] | [0.04, 2.61, 0.55] | ⚠️ | 의도적 (§4.1 참조) |
| **FG coverage** | 2.54% | 1.14% | ⚠️ | rat arena 넓어 FG 작음 |
| **Green G/R** | N/A | 0.810 | ✅ | despill 적용 확인 |

### 6.2 차이점 분석

**centroid ≠ origin**: Mouse M5는 batch uniform recentering으로 centroid=origin.
Rat v5는 recentering 없음 — v3에서 centroid shift가 거리 붕괴(2.6→0.4)를 일으켜 실패.
v1(no recenter, val PSNR 17.49)이 이 구조의 유효성을 실증.

**coverage 1.14% vs 2.54%**: rat이 넓은 arena에 있어 FG 비율이 낮음.
M5 mouse는 좁은 cage에서 촬영. 이 차이는 전처리로 해결할 수 없고 (zoom-in 하면
per-frame fx 변동 위험 = D4 bug), 모델이 적응해야 하는 domain gap.

**distance range**: rat 카메라 간 거리 편차(0.44)가 mouse(0.21)보다 큼.
이는 s-DANNCE arena의 비대칭 카메라 배치 때문. 학습에 큰 영향 없음.

### 6.3 v5 전처리 파이프라인 (최종)

```
Stage 0: sdannce_to_gslrm.py (원본 비디오 → GS-LRM format)
  - 1920×1200 → zero-pad → 512×512
  - SAM2 mask → alpha channel
  - 카메라 정규화: normalize_cameras() → distance=2.7
  → gslrm_format_rat2 (fx=605, G/R=0.977)

Stage 1: despill.py (녹색 제거)
  - FG 픽셀의 green spill 제거 (VFX 2-stage)
  - G/R: 0.977 → 0.798
  → gslrm_format_rat2_despilled (fx=605, G/R=0.798)

Stage 2: normalize_fx.py (기하 정규화)
  - affine scaling: fx=605→549 (scale=0.908)
  - mask: INTER_NEAREST (이진성 보존)
  - RGB: INTER_LANCZOS4 (품질 보존)
  - BG: white compositing 후 재적용
  → rat2_s1despill_s2fxnorm (fx=549, G/R=0.810)
```

### 6.4 가설 (v5 학습 목적)

**H1: fx 정규화 + despill → v1 대비 PSNR 개선**

| 조건 | v1 (baseline) | v5 (현재) | 변경 이유 |
|------|:---:|:---:|------|
| 데이터 | RAT1 (1001fr) | RAT2 (2967fr) | 3× 데이터 |
| fx | 605 (10% off) | 549 (정확히 일치) | Plucker ray 정확성 |
| despill | ❌ | ✅ | 녹색 artifact 제거 |
| distance norm | 2.7 (전처리) | 2.7 (전처리) | 동일 |
| recenter | ❌ | ❌ | 동일 (v3 교훈) |

**예상 결과**:
- v1 (17.49 dB) 대비 **+2~5 dB** 개선 기대
  - 3× 데이터 → overfitting 감소
  - fx=549 → Plucker ray 정확성 → 초기 수렴 가속
  - despill → FG 색상 정확성 → PSNR 직접 기여

**성공 기준**:
- Tier 1 (최소): val PSNR ≥ 17.49 (v1 이상)
- Tier 2 (기대): val PSNR ≥ 20.0 (mouse 6v 수준)
- Tier 3 (이상): val PSNR ≥ 22.0 (mouse 6v best 근접)

---

## 7. 용어 정리

| 용어 | 설명 |
|------|------|
| **fx (focal length)** | 카메라의 초점거리 (px 단위). 클수록 telephoto, 작을수록 wide-angle |
| **FOV (Field of View)** | 카메라가 보는 각도. `FOV = 2 × arctan(image_size / (2 × fx))` |
| **distance** | 카메라에서 원점(scene center)까지 거리 |
| **coverage** | 이미지 전체 면적 대비 물체(FG)가 차지하는 비율 (%) |
| **Plucker ray** | 각 픽셀의 3D 방향+위치를 인코딩한 6D 벡터 |
| **convergence point** | 여러 카메라의 viewing direction이 교차하는 3D 점 |
| **centroid** | 카메라 위치들의 평균점 (≠ convergence point) |
| **clip_xyz** | Gaussian 위치를 [-1, 1] 범위로 제한하는 옵션 |
| **affine transform** | 이미지와 카메라 파라미터를 동시에 스케일링하는 변환 |
| **Pareto-optimal** | 하나를 개선하면 다른 하나가 악화되는 최적 트레이드오프 상태 |

---

## 8. 참고 문서

| 문서 | 위치 | 내용 |
|------|------|------|
| RAT_PREPROCESSING_STRATEGY.md | `docs/specs/` | fx-coverage Pareto 분석 |
| PREPROCESSING_REGISTRY.md | `docs/datasets/` | 전처리 버전 이력 |
| RAT2_V3_SUCCESS_CRITERIA.md | `docs/specs/` | v3 성공 기준 (v4b에도 적용) |
| COORDINATE_SYSTEMS.md | Obsidian `docs/theory/` | 카메라 좌표계 SSOT |
| camera_geometry.py | `mouse_extensions/data/` | convergence point 계산 코드 |

---

*FaceLift Project | RAT Camera Preprocessing Guide v1.0 | 2026-04-05*
