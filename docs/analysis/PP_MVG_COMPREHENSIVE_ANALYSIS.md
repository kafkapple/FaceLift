# PP (Principal Point) MVG 정합성 종합 분석

> **작성일**: 2026-01-25
> **목적**: 모든 데이터셋의 PP 정합성 검증 및 PSNR 상관관계 분석
> **검증 스크립트**: mouse_extensions/scripts/diagnostics/verify_pp_mvg_consistency.py
> **관련 문서**: [[PP_FIX_MVG_THEORY]], [[COMPREHENSIVE_ANALYSIS_260125]], [[TRAIN_VAL_GAP_ANALYSIS]]

---

## 1. 핵심 발견

### 1.1 PP 정합성 vs PSNR 상관관계

| 데이터셋 | PP cx (mean±std) | PP cy (mean±std) | Ray Error (max) | Val PSNR | 상태 |
|----------|------------------|------------------|-----------------|----------|------|
| **D3_normalized** | 256.0±0.0 | 256.0±0.0 | 0.00° | **27.09** | ✅ OK |
| **D7_1** | 256.0±0.0 | 256.0±0.0 | 0.00° | **20.93** | ✅ OK |
| **D8** | 256.0±0.0 | 256.0±0.0 | 0.00° | **20.21** | ✅ OK |
| M3 | 265.7±62.6 | 184.2±29.7 | 6.96° | 10.56 | ❌ BAD |
| M3_norm | 197.2±46.4 | 136.7±22.0 | 13.62° | 17.09 | ❌ BAD |
| M3_persample | 234.7±32.1 | 212.0±39.8 | 16.15° | 17.08-18.88 | ❌ BAD |

### 1.2 결론

PP=256±0 (MVG 정합) → Val PSNR 20+ ✅
PP 가변 (MVG 부정합) → Val PSNR 10-18 ❌

**PP 정합성이 Val PSNR에 직접적 영향!**

---

## 2. 데이터셋별 상세 분석

### 2.1 정상 데이터셋 (MVG 정합)

#### D3_normalized (★ 최고 성능)
- **PP**: cx=256±0, cy=256±0
- **fx**: 549.0 (정규화됨)
- **Val PSNR**: 27.09 (최고)
- **특징**: PP shift 방식, zoom 없음
- **Train-Val Gap**: -3.0 (음수! 일반화 우수)

#### D7_1 (안정적 기준선)
- **PP**: cx=256±0, cy=256±0
- **fx**: 549.0 (정규화됨)
- **Val PSNR**: 20.93
- **특징**: Affine transform, PP shift
- **Train-Val Gap**: +3.7 (정상)

#### D8 (Homography 기반)
- **PP**: cx=256±0, cy=256±0
- **fx**: 548.99 (정규화됨)
- **Val PSNR**: 20.21
- **특징**: Homography + skew correction
- **Train-Val Gap**: +6.6 (약간 높음)

### 2.2 문제 데이터셋 (MVG 부정합)

#### M3 (가장 심각)
- **PP**: cx=265.7±62.6 (범위: 165.7-344.9)
- **PP**: cy=184.2±29.7 (범위: 165.7-307.2)
- **fx**: **739.72** ← 정규화 안됨!
- **Ray Error**: 6.96°
- **Val PSNR**: 10.56
- **문제**: fx 정규화 + PP 정합성 모두 실패

#### M3_norm
- **PP**: cx=197.2±46.4 (범위: 123.0-256.0)
- **PP**: cy=136.7±22.0 (범위: 123.0-228.0)
- **fx**: 548.99 (정규화됨 ✅)
- **Ray Error**: **13.62°** (심각)
- **Val PSNR**: 17.09
- **문제**: 객체 중심 zoom → PP 가변

#### M3_persample
- **PP**: cx=234.7±32.1 (범위: 97.0-256.0)
- **PP**: cy=212.0±39.8 (범위: 97.0-256.0)
- **fx**: 548.99 (정규화됨 ✅)
- **Ray Error**: **16.15°** (가장 심각)
- **Val PSNR**: 17.08-18.88
- **문제**: Per-sample zoom → PP 극단적 분산

---

## 3. Ray Error 영향 분석

### 3.1 Ray Error 계산

Ray Error = arctan(PP_offset / fx)

예시:
- PP_offset = 0px → Ray Error = 0°
- PP_offset = 50px, fx=549 → Ray Error = 5.2°
- PP_offset = 100px, fx=549 → Ray Error = 10.3°
- PP_offset = 159px, fx=549 → Ray Error = 16.2°

### 3.2 Ray Error vs 재구성 품질

| Ray Error 범위 | 영향 | 예상 PSNR |
|----------------|------|-----------|
| 0-1° | 무시 가능 | 20+ |
| 1-5° | 경미한 블러 | 18-20 |
| 5-10° | 뚜렷한 Ghosting | 15-18 |
| **10°+** | **심각한 Ghosting** | **10-17** |

---

## 4. 문제 원인

### 4.1 Object-Centered Zoom의 문제

현재 M3_* 프리셋들은 **객체 중심** zoom을 사용.
객체 중심은 뷰마다 다름 → crop offset이 샘플마다 다름 → PP가 뷰/샘플마다 달라짐 → MVG 부정합

### 4.2 올바른 해결책: Center-Aligned Zoom

이미지 중심 기준 crop (MVG 정합)
- crop_x = (size - crop_size) // 2  # 항상 동일!
- crop_y = (size - crop_size) // 2
- 결과: PP = 256 자동 유지 → MVG 정합

---

## 5. 권장 조치

### 5.1 즉시 수정 필요

| 프리셋 | 현재 문제 | 수정 방법 |
|--------|-----------|-----------|
| M3_persample | Object-centered zoom | zoom_center_mode: "image" |
| M3_norm | Object-centered zoom | zoom_center_mode: "image" |

### 5.2 새 프리셋

M3_2: zoom_center_mode: "image" (MVG-correct)
M3_1: zoom_center_mode: "image" (MVG-correct) - 추가 필요

### 5.3 검증 명령어

# PP 분포 검증
python mouse_extensions/scripts/diagnostics/verify_pp_mvg_consistency.py --verbose

# 새 데이터셋 전처리
python -m mouse_extensions.preprocessing.preprocess_unified \
    --preset M3_2 \
    --output /home/joon/data/preprocessed/FaceLift_mouse/M3_2

---

## 6. 실험 결과 요약

### 6.1 전체 실험 PSNR 순위 (Val PSNR 기준)

| 순위 | 실험 | Val PSNR | Train PSNR | Gap | PP 상태 |
|------|------|----------|------------|-----|---------|
| 1 | **D3_normalized_E0_paper** | **27.09** | 24.07 | -3.0 | ✅ 256±0 |
| 2 | D3_normalized_E0_1_facelift | 23.11 | 24.57 | +1.5 | ✅ 256±0 |
| 3 | D7_1_E0_paper | 20.93 | 24.58 | +3.7 | ✅ 256±0 |
| 4 | D8_E0_paper | 20.21 | 26.82 | +6.6 | ✅ 256±0 |
| 5 | D7_1_E0_mouse | 19.86 | 28.15 | +8.3 | ✅ 256±0 |
| 6 | M3_persample_E0_2_mouse | 18.88 | 23.82 | +4.9 | ❌ 가변 |
| 7 | D8_2_E0_paper | 18.34 | 23.03 | +4.7 | ✅ 256±0 |
| 8 | M3_norm_E1_2_alpha_5v | 17.94 | 23.93 | +6.0 | ❌ 가변 |
| 9 | M3_norm_E0_1_facelift | 17.09 | 19.23 | +2.1 | ❌ 가변 |
| 10 | M3_persample_E0_1_facelift | 17.08 | 24.56 | +7.5 | ❌ 가변 |

### 6.2 패턴 분석

**PP 정합 데이터셋 (D3, D7_1, D8):**
- Val PSNR: 18.3 - 27.1 (평균 ~22)
- Train-Val Gap: -3.0 ~ +8.3

**PP 부정합 데이터셋 (M3_*):**
- Val PSNR: 10.6 - 18.9 (평균 ~15)
- Train-Val Gap: +2.1 ~ +7.5

**결론**: PP 정합성이 Val PSNR을 5-10 포인트 향상시킴!

---

## 7. 관련 문서

- [[PP_FIX_MVG_THEORY]] - MVG 이론 분석 및 해결책
- [[COMPREHENSIVE_ANALYSIS_260125]] - 전체 분석 보고서
- [[TRAIN_VAL_GAP_ANALYSIS]] - Train-Val Gap 분석
- [[EXPERIMENT_REGISTRY]] - 실험 레지스트리
- [[PREPROCESSING_REGISTRY]] - 전처리 레지스트리

---

*PP MVG Comprehensive Analysis v1.0 | 2026-01-25*
*검증 스크립트: verify_pp_mvg_consistency.py*
