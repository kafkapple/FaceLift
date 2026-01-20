# FaceLift Mouse Preprocessing: Unified Technical Report

> **Version**: v1.0 (2026-01-21)
> **Author**: Claude Code Analysis
> **Scope**: 전처리 버전(v13, D1~D7.2) 통합 비교 및 권장 사항

---

## Executive Summary

### 전처리 진화 타임라인

```
v12/v13 (초기) → D1 (PP crop) → D4 (정규화) → D7 (PP-shift) → D7.1/D7.2 (완성)
   ↓               ↓              ↓            ↓               ↓
화이트닝      Ghosting       PP 오류      fy 근사       기하학 정확
```

### 데이터셋 Quick Reference

| Dataset | PP 처리 | fy 처리 | Ray Error | 권장 |
|---------|---------|---------|-----------|------|
| **v13** | 256 강제 | 549 강제 | 5-13° | ❌ Deprecated |
| **D1** | crop (뷰별 다름) | 549 강제 | ~13° | ❌ Deprecated |
| **D4** | 256 강제 | 비례 스케일 | 5-13° | ❌ Deprecated |
| **D7** | PP-shift | 549 강제 | ~0.4° | ⚠️ Current |
| **D7.1** | PP-shift | 개별 scale | ~0° | ✅ **Recommended** |
| **D7.2** | PP-shift | 평균 scale | ~0.2° | ⚪ Alternative |

---

## Part 1: Preprocessing Evolution

### 1.1 v12/v13 (초기 - 화이트닝 현상)

**특징**:
- `force_center_cxcy=true`: PP를 (256, 256)으로 강제
- 카메라 정규화: fx=fy=549, distance=2.7

**문제점**:
```
실제 PP: 뷰마다 다름 (예: cam0: (312, 287), cam1: (298, 301))
기록 PP: 모두 (256, 256) 강제

결과:
- 모든 픽셀의 ray direction이 5-13° 오류
- 학습 시 gaussian이 잘못된 위치에 배치
- 학습이 진행될수록 배경(흰색)으로 수렴 (화이트닝)
```

**데이터셋 위치**: `/home/joon/data/preprocessed/FaceLift_mouse/data_mouse_v13_original_ratio/`

---

### 1.2 D1 (PP-Centered Crop - Ghosting)

**특징**:
- 각 뷰의 PP를 (256, 256)에 맞추기 위해 이미지 crop
- 뷰마다 다른 crop offset 적용

**문제점**:
```
View 0: shift=(+72, +105) → 쥐가 우측 하단으로 이동
View 1: shift=(-81, +30)  → 쥐가 좌측으로 이동
...

결과:
- 같은 프레임에서 뷰마다 쥐의 위치가 다름
- Multi-view consistency 완전히 깨짐
- 심각한 Ghosting artifact
```

**데이터셋 위치**: `/home/joon/data/preprocessed/FaceLift_mouse/D1_pp_centered/`

---

### 1.3 D4 (카메라 정규화 - PP 오류 지속)

**특징**:
- 3D triangulation 기반 center estimation
- 카메라 정규화 (fx=549, distance=2.7)
- **PP=256 강제 (버그)**

**문제점**:
```
PP offset: 평균 37px 오차
Ray error: 5-13°
여전히 Ghosting 발생
```

**데이터셋 위치**: `/home/joon/data/preprocessed/FaceLift_mouse/D4/`

---

### 1.4 D7 (PP-Shift - 현재 운영)

**특징**:
- **PP-shift 방식**: 이미지를 shift하여 PP를 (256, 256)에 위치
- 카메라 정규화 유지
- **fy=549 강제** (실제 ~551.3)

**장점**:
- PP 오류 해결 (0px)
- Cross-view consistency 유지

**단점**:
- fy 강제로 ~0.4° ray error (미미하지만 존재)

**데이터셋 위치**: `/home/joon/data/preprocessed/FaceLift_mouse/D7/`

---

### 1.5 D7.1 (개별 스케일 - 권장)

**특징**:
- PP-shift 방식 유지
- **개별 scale**: scale_x와 scale_y를 따로 계산
- fx=fy=549 **정확히** 일치

**수식**:
```python
scale_x = target_fx / orig_fx  # 549 / 1632 = 0.3363
scale_y = target_fy / orig_fy  # 549 / 1639 = 0.3350

# Affine transform
new_img = cv2.warpAffine(img, [[scale_x, 0, shift_x],
                                [0, scale_y, shift_y]], ...)
```

**장점**:
- 기하학적으로 완벽 (Ray error ~0°)
- GS-LRM pretrained 분포와 완전 일치

**단점**:
- ~0.6% 비등방성 압축 (시각적으로 감지 불가)

**데이터셋 위치**: `/home/joon/data/preprocessed/FaceLift_mouse/D7_1/`

---

### 1.6 D7.2 (평균 스케일 - 대안)

**특징**:
- PP-shift 방식 유지
- **평균 scale**: (scale_x + scale_y) / 2
- 등방성 스케일링 (이미지 왜곡 없음)

**수식**:
```python
scale_avg = (scale_x + scale_y) / 2  # 0.3356
new_fx = orig_fx * scale_avg  # ~547.3 (≠549)
new_fy = orig_fy * scale_avg  # ~550.7 (≠549)
```

**장점**:
- 이미지 품질 완벽 보존
- 등방성 스케일링

**단점**:
- fx, fy가 정확히 549가 아님 (~0.2% 오차)

**데이터셋 위치**: `/home/joon/data/preprocessed/FaceLift_mouse/D7_2/`

---

## Part 2: Geometry Analysis

### 2.1 Pinhole Camera Model

```
3D point (X, Y, Z) → 2D pixel (u, v):
  u = fx * X/Z + cx
  v = fy * Y/Z + cy

Inverse (pixel → ray):
  ray_x = (u - cx) / fx
  ray_y = (v - cy) / fy
  ray_z = 1.0
```

### 2.2 Ray Error 계산

**PP Error (cx, cy 오류)**:
```
θ_PP = arctan(sqrt((Δcx/fx)² + (Δcy/fy)²))

예: Δcx=56, Δcy=31, fx=fy=549
θ_PP = arctan(sqrt((56/549)² + (31/549)²))
     = arctan(0.116) ≈ 6.6°
```

**fy Error (D7)**:
```
실제 fy = 551.3, 기록 fy = 549
이미지 가장자리 (v=256 from center):
  Expected ray_y = 256/551.3 = 0.4645
  Recorded ray_y = 256/549.0 = 0.4663
  Error ≈ 0.4% → ~0.24°
```

### 2.3 정량적 비교 표

| 버전 | PP Error (px) | fy Error | Max Ray Error | Ghosting Risk |
|------|---------------|----------|---------------|---------------|
| **v13** | 37-56 avg | 0 | **5-13°** | **CRITICAL** |
| **D1** | ~0 (but shifted) | 0 | **~13°** | **CRITICAL** |
| **D4** | 37 avg | 0 | **5-13°** | **HIGH** |
| **D7** | 0 | 2.5 avg | **0.4°** | LOW |
| **D7.1** | 0 | 0 | **~0°** | NONE |
| **D7.2** | 0 | ~1.2 | **~0.2°** | VERY LOW |

---

## Part 3: Experiment Comparison Framework

### 3.1 비교 실험 설계

| 실험 | Dataset | Schema | 목적 |
|------|---------|--------|------|
| Baseline-v13 | v13 | E1_1_paper_random | 화이트닝 재현 |
| Baseline-D1 | D1 | E1_1_paper_random | Ghosting 재현 |
| **Recommended** | D7_1_t | E1_1_paper_random | 정확한 기준선 |

### 3.2 실행 명령어

```bash
cd /home/joon/dev/FaceLift

# D7_1_t (Recommended)
CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1_t -e E1_1_paper_random \
    > logs/d7_1_t_e1_1_paper_random.log 2>&1 &

# v13 (화이트닝 baseline)
CUDA_VISIBLE_DEVICES=1 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d v13 -e E1_1_paper_random \
    > logs/v13_e1_1_paper_random.log 2>&1 &

# D1 (Ghosting baseline)
CUDA_VISIBLE_DEVICES=2 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D1 -e E1_1_paper_random \
    > logs/d1_e1_1_paper_random.log 2>&1 &
```

### 3.3 평가 지표

| 지표 | 설명 | 예상 결과 |
|------|------|-----------|
| **PSNR** | 이미지 품질 | D7_1 > D7 > D4 > v13/D1 |
| **SSIM** | 구조 유사성 | D7_1 > D7 > D4 > v13/D1 |
| **mask_iou** | 마스크 일치도 | D7_1 > D7 >> v13/D1 |
| **Ghosting** | 시각적 artifact | D7_1: None, v13/D1: Severe |

---

## Part 4: Recommendations

### 4.1 Production 권장

```
1순위: D7.1 (기하학적 완벽성)
2순위: D7.2 (이미지 품질 보존)
3순위: D7   (현재 운영, 실용적)

Deprecated: v13, D1, D4, D6-* (PP 오류)
```

### 4.2 전처리 명령어

```bash
# D7.1 전처리
python -m mouse_extensions.scripts.preprocess_D7_pp_centered \
    --data-dir /home/joon/data/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --camera-pkl /home/joon/data/markerless_mouse_1_nerf/new_cam.pkl \
    --frame-interval 5 \
    --scale-mode individual
```

### 4.3 Config 생성 (모듈화 시스템)

```bash
cd /home/joon/dev/FaceLift/configs/mouse/_modular

# 사용 가능한 옵션 확인
python generate_config.py --list

# Config 생성
python generate_config.py --dataset D7_1 --schema 5v_alpha
```

---

## Appendix: File Locations

| 항목 | 경로 |
|------|------|
| **데이터셋 루트** | `/home/joon/data/preprocessed/FaceLift_mouse/` |
| **Config 루트** | `/home/joon/dev/FaceLift/configs/mouse/` |
| **모듈화 Config** | `configs/mouse/_modular/{datasets,schemas}/` |
| **전처리 스크립트** | `mouse_extensions/scripts/preprocess_D7_*.py` |
| **레포트** | `mouse_extensions/reports/` |
| **문서** | `docs/PREPROCESSING_REGISTRY.md` |

---

## Related Reports

기존 레포트 통합 (05_D7_verification/):
- 260120_D7_geometry_analysis.md → Part 2 통합
- 260120_D7_variants_comparison.md → Part 1.5-1.6 통합
- 260120_session_summary.md → Part 4 통합

---

*Generated: 2026-01-21 | Claude Code Analysis*
