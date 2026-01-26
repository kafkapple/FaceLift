# Dataset Analysis (데이터셋 분석)

> **SSOT**: 데이터셋 성공/실패 요인 종합 분석
> **최종 업데이트**: 2026-01-26

---

## 1. Coverage 정의

### 계산 공식
```
Coverage = foreground_pixels / total_pixels × 100%
```

- **foreground_pixels**: 마스크/알파 > 127인 픽셀 수
- **total_pixels**: 이미지 전체 픽셀 (512×512 = 262,144)

### Adaptive Zoom 목표
- **target_fg_coverage**: 5% (presets.py 설정)
- Zoom 계수: `zoom = sqrt(target / current)`
- Zoom 범위: [1.0, 1.8] (클리핑 방지)

### 실측 Coverage
| 데이터셋 | Zoom | Coverage (실측) |
|---------|------|-----------------|
| D7_1 (M1) | ❌ 없음 | ~2-3% (추정) |
| D8 (M2) | ❌ 없음 | ~2-3% (추정) |
| M3_2 | ✅ per-sample | **6.26%** |
| M3_3 | ✅ per-sample | **6.26%** |

> ⚠️ 이전 문서의 50%, 74% 수치는 검증되지 않음

---

## 2. 핵심 요인 분석

### 2.1 기하학적 정합성 (PP)

| 설정 | PP 결과 | Ray Error | 영향 |
|------|---------|-----------|------|
| `zoom_center_mode: "image"` | **256 (고정)** | ~0° | ✅ Pretrained 호환 |
| `zoom_center_mode: "object"` | 가변 (185±52) | 13-16° | ❌ Ghosting |
| `pp_method: "shift_to_256"` (no zoom) | **256 (고정)** | ~0° | ✅ |

**핵심**: GS-LRM pretrained 모델은 **PP=256** 분포에서 학습됨

### 2.2 Transform 비교 (Affine vs Homography)

| 데이터셋 | Transform | Skew Correction | Val PSNR |
|---------|-----------|-----------------|----------|
| **D7_1** (M1) | Affine | ❌ | **20.93** |
| **D8** (M2) | Homography | ✅ | 20.21 |

**결론**: 
- 차이 = **-0.72 PSNR** (Homography가 약간 낮음)
- 통계적 유의성 불확실
- **Affine이 더 안정적**일 수 있음

### 2.3 Coverage (Zoom) 효과

| 데이터셋 | Zoom | Coverage | PSNR | 비고 |
|---------|------|----------|------|------|
| D7_1 | ❌ | ~2-3% | 20.93 | 기준선 |
| M3_2 | ✅ | ~6% | TBD | 실험 필요 |

**가설**: Coverage 증가 → PSNR 향상
- 근거: D3_normalized (PSNR 27.09) - 하지만 전처리 방법 불명

---

## 3. 데이터셋별 상태

### 3.1 검증된 (Active)

| 데이터셋 | 분류 | PP | Zoom | 상태 |
|---------|------|-----|------|------|
| **D7_1** (M1) | affine | 256 | ❌ | ✅ 안정 기준선 |
| **D8** (M2) | homography | 256 | ❌ | ✅ 검증됨 |
| **M3_1** | homo+zoom | 256 | global | ✅ MVG-correct |
| **M3_2** | homo+zoom | 256 | per-sample | ✅ **권장** |
| **M3_3** | homo+zoom | 256 | per-sample | ✅ 권장 |

### 3.2 실패 (Deprecated)

| 데이터셋 | 문제 | PP | 원인 |
|---------|------|-----|------|
| M4 | PP 가변 | 가변 | `zoom_center_mode: object` |
| M3_norm | PP 가변 | 197±46 | object-centered |
| M3_persample | PP 가변 | 235±32 | object-centered |
| D1~D6 | 기하학 손상 | 가변 | PP 미보정 |

---

## 4. 검증 필요 가설

| ID | 가설 | 비교 실험 | 상태 |
|----|------|----------|------|
| H1 | Coverage↑ → PSNR↑ | D7_1 vs M3_2 (PP 동일) | ⏳ 대기 |
| H2 | Affine ≈ Homography | D7_1 vs D8 | ✅ 차이 미미 |
| H3 | PP 정합 필수 | M3_norm vs M3_2 | ⏳ 대기 |

---

## 5. 권장 사용

### 학습용
1. **M3_2** ⭐: Per-sample zoom + PP=256 + fx=549
2. **M3_3**: M3_2 변형 (동일 설정)
3. **D7_1**: 안정적 기준선 (no zoom)

### Temporal Split (Hold-out Test)
- **M3_3t**: M3_3 + Pose-Splatter 1/3 split
  - Train: frames 0-1199 (1200)
  - Val: frames 1200-2399 (1200)
  - Test: frames 2400-3599 (1200)

---

## 6. 관련 문서

| 문서 | 내용 |
|------|------|
| [[PREPROCESSING_REGISTRY]] | 전처리 프리셋 상세 |
| [[M3_SERIES_SPEC]] | M3 계열 명세 |
| [[../theory/PP_FX_MVG_ANALYSIS]] | PP/MVG 이론 |

---

*Dataset Analysis v1.0 | 2026-01-26*
