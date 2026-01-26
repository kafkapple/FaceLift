# FaceLift Mouse: 종합 분석 보고서

> **작성일**: 2026-01-25
> **목적**: 정량 지표 vs 정성적 관찰(Ghosting) 불일치 분석
> **관련 문서**: [[HYPOTHESIS_EXPERIMENT_PLAN]], [[TRAIN_VAL_GAP_ANALYSIS]]

---

## 1. 현재 실험 결과 요약

### 1.1 최신 Validation 결과 (2026-01-25 기준)

| 실험 | Val PSNR | Val SSIM | Val LPIPS | Train PSNR | **Gap** | Step |
|------|----------|----------|-----------|------------|---------|------|
| **D3_normalized_E0_paper** | **27.09** | 0.9881 | 0.0210 | ~24.07 | **-3.0** | 3001 |
| D3_normalized_E0_1_facelift | 23.11 | 0.9824 | 0.0217 | ~24+ | ~+1 | 2801 |
| **M3_persample_E0_2_mouse** | **18.88** | 0.9830 | 0.0204 | ~28+ | +9+ | 2801 |
| M3_persample_E0_1_facelift | 17.24 | 0.9798 | 0.0195 | ~28+ | +11+ | 2601 |
| M3_persample_E1_2_alpha | 15.97 | 0.9785 | 0.0222 | ~25+ | +9+ | 3001 |
| M3_norm_E0_1_facelift | 16.93 | 0.9777 | 0.0215 | ~22+ | +5+ | 2201 |
| M3_norm_E1_2_alpha_5v | 17.94 | 0.9786 | 0.0220 | ~22+ | +4+ | 2201 |
| D8_E0_paper | 20.21 | 0.9895 | 0.0170 | ~27+ | +7+ | 4301 |
| D7_1_E0_paper | 20.93 | 0.9902 | 0.0140 | ~24.58 | +3.7 | 4901 |

### 1.2 핵심 발견

```
🔴 문제: D3_normalized 27.09 PSNR에서도 Ghosting 관찰
🔴 문제: M3_persample 18.88 PSNR에서도 Ghosting 관찰
⚠️ 주의: 정량 지표(PSNR)와 정성적 품질(Ghosting)이 불일치
```

---

## 2. Train-Val Gap 심층 분석

### 2.1 Gap 패턴 분류

| Gap 유형 | 범위 | 실험 | 해석 |
|----------|------|------|------|
| **음수** | < 0 | D3_normalized_E0_paper (-3.0) | ★ 이례적 |
| **정상** | 0~5 | D7_1_E0_paper (+3.7) | 양호한 일반화 |
| **경고** | 5~10 | D8_E0_paper (+7), M3_norm (+5) | 약간 과적합 |
| **과적합** | > 10 | M3_persample_E0_1 (+11) | 심한 과적합 |

### 2.2 D3_normalized 음수 Gap의 원인

**가설 검토**:

| # | 가설 | 근거 | 가능성 |
|---|------|------|--------|
| 1 | **Val 샘플이 더 쉬움** | UID 44 (D3) vs UID 202 (M3) 다름 | ★★★ |
| 2 | **Train 데이터 다양성** | 3240 train 샘플의 난이도 분포 | ★★☆ |
| 3 | **전처리 파이프라인 차이** | D3는 별도 preprocessor 사용 | ★★☆ |
| 4 | **시간적 범위 차이** | 프레임 interval 다름 가능 | ★☆☆ |

**검증 필요**:
```bash
# D3 vs M3 validation 샘플 비교
# UID 44 (D3) vs UID 202 (M3)의 특성 분석 필요
```

---

## 3. Ghosting 현상 분석

### 3.1 Ghost 관련 메트릭 비교

| 실험 | ghost_alpha_std | ghost_fg_coverage | gaussians_usage |
|------|-----------------|-------------------|-----------------|
| D3_normalized | **0.15~0.20** | ~0.97 | ~0.17 |
| M3_persample | **0.38~0.43** | ~0.60 | ~0.05 |
| M3_norm | **0.21~0.25** | ~0.92 | ~0.10 |

**해석**:
- **ghost_alpha_std**: Alpha 예측의 불확실성 (높을수록 ghosting 가능성↑)
- D3가 M3보다 ghost_alpha_std가 **2배 낮음** → 더 확실한 alpha 예측
- M3_persample이 가장 높은 ghost_alpha_std → Ghosting 원인

### 3.2 Ghosting 근본 원인 분석

```
┌─────────────────────────────────────────────────────────────────┐
│                    Ghosting 발생 메커니즘                        │
├─────────────────────────────────────────────────────────────────┤
│  1. Multi-view 기하학 불일치                                     │
│     └─ 카메라 파라미터 오차 → 3D 위치 추정 오류                    │
│                                                                 │
│  2. Alpha/Opacity 불확실성                                       │
│     └─ ghost_alpha_std↑ → 반투명 Gaussian → 잔상                 │
│                                                                 │
│  3. 뷰 간 일관성 부족                                            │
│     └─ Train에서 학습한 패턴이 Val에 전이 안됨 (Gap↑)             │
│                                                                 │
│  4. 객체 크기/위치 분산                                          │
│     └─ Per-sample 처리해도 완벽한 정규화 어려움                    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 D3 vs M3: 왜 D3가 더 나은가?

| 요소 | D3_normalized | M3_persample | 차이 |
|------|---------------|--------------|------|
| **원본 데이터** | 다른 전처리 버전 | 최신 전처리 | 파이프라인 다름 |
| **PP 처리** | 256 고정 | 256 고정 | 동일 |
| **fx 정규화** | 549 | 549 | 동일 |
| **Coverage** | ~74% | ~80% | M3가 더 높음 |
| **ghost_alpha_std** | **0.15~0.20** | **0.38~0.43** | D3가 2배 낮음 |
| **Val PSNR** | **27.09** | 17~19 | D3가 +10 |

**결론**: Coverage가 높아도 Ghosting 해결 안됨 → **기하학적 정확성**이 핵심

---

## 4. Train/Val Split 분석

### 4.1 데이터셋 구성

| 데이터셋 | Train | Val | 비율 | Val UID |
|----------|-------|-----|------|---------|
| D3_normalized | 3240 | 360 | 90/10 | 44 |
| M3_persample | 3237 | 358 | 90/10 | 202 |
| M3_norm | 3237 | 358 | 90/10 | 202 |
| D7_1 | 3237 | 358 | 90/10 | 202 |

### 4.2 Split 방식의 문제점

```
⚠️ 문제: 모든 데이터셋이 동일한 random split 사용
→ 특정 프레임이 항상 Val에 포함
→ Val 난이도가 데이터셋마다 다를 수 있음

예: UID 44 (D3) vs UID 202 (M3)
   - 다른 시간대의 프레임
   - 쥐의 pose가 다름
   - 재구성 난이도가 다름
```

---

## 5. 가설 상태 업데이트

### 5.1 검증된 가설

| 가설 | 내용 | 상태 | 근거 |
|------|------|------|------|
| **H2** | fx=549 정규화 필수 | ✅ 검증됨 | M3(fx=739) 실패 |
| **H7** | mask_mode=none 유리 | ✅ 검증됨 | E0 > E1/E2 |

### 5.2 부분 검증

| 가설 | 내용 | 상태 | 발견 |
|------|------|------|------|
| **H1** | Coverage↑ → PSNR↑ | ⚠️ 부분 지지 | M3(80%) < D3(74%)로 반증 |
| **H5** | Per-sample zoom 유리 | ⚠️ 불분명 | M3_persample ≈ M3_norm |
| **H8** | PP=256 고정 유리 | ⚠️ 불분명 | 둘 다 256이지만 결과 다름 |

### 5.3 새로운 가설 (H9)

**H9: Validation 샘플 특성이 결과를 좌우한다**

| 근거 | 데이터 |
|------|--------|
| D3 Val UID 44 | PSNR 27+ (쉬운 샘플?) |
| M3 Val UID 202 | PSNR 17~19 (어려운 샘플?) |

**검증 방법**: 동일 Val UID로 D3와 M3 비교

---

## 6. 핵심 문제: PSNR ≠ 시각적 품질

### 6.1 PSNR의 한계

```
PSNR = 10 * log10(MAX^2 / MSE)

문제점:
├─ 전체 이미지 평균 → 국소적 artifact 무시
├─ 배경 품질도 포함 → 객체 품질 희석
└─ 구조적 일관성 미반영 → Ghosting 감지 불가
```

### 6.2 Ghosting 감지 지표 제안

| 지표 | 설명 | 현재 로깅 |
|------|------|-----------|
| **ghost_alpha_std** | Alpha 예측 불확실성 | ✅ 있음 |
| **view_consistency** | 뷰 간 재구성 일관성 | ❌ 없음 |
| **edge_sharpness** | 객체 경계 선명도 | ❌ 없음 |
| **temporal_coherence** | 시간적 일관성 | ❌ 없음 |

---

## 7. 결론

### 7.1 핵심 발견

1. **D3_normalized가 여전히 SOTA** (Val PSNR 27.09)
   - 하지만 여전히 Ghosting 존재

2. **M3 데이터셋은 D3 재현 실패**
   - 동일한 PP=256, fx=549 설정에도
   - Val PSNR 10 낮음 (27 vs 17~19)
   - ghost_alpha_std 2배 높음

3. **Train-Val Gap이 클수록 Ghosting 심함**
   - D3: Gap -3 → 낮은 ghost_alpha_std
   - M3: Gap +10+ → 높은 ghost_alpha_std

4. **Coverage 증가가 만능 해결책 아님**
   - M3 (80%) < D3 (74%)
   - 기하학적 정확성이 더 중요

### 7.2 근본 원인 추정

```
D3_normalized의 성공 요인 (추정):
├─ 1. 전처리 파이프라인의 차이 (다른 preprocessor)
├─ 2. 원본 데이터 품질의 차이
├─ 3. Validation 샘플 특성 (UID 44 vs 202)
└─ 4. 카메라 캘리브레이션 정밀도
```

---

## 8. 제안

### 8.1 즉시 실행 (P0)

| # | 작업 | 목적 |
|---|------|------|
| 1 | **D3 전처리 코드 분석** | D3와 M3의 차이점 파악 |
| 2 | **Val UID 통일 실험** | UID 44로 M3 검증 |
| 3 | **시각화 비교** | D3 vs M3 렌더링 결과 비교 |

### 8.2 단기 (P1)

| # | 작업 | 목적 |
|---|------|------|
| 4 | **view_consistency 지표 추가** | Ghosting 정량화 |
| 5 | **Temporal validation** | 시간적 일관성 검증 |
| 6 | **Multi-view calibration 재검증** | 카메라 파라미터 정확도 |

### 8.3 중기 (P2)

| # | 작업 | 목적 |
|---|------|------|
| 7 | **Opacity regularization 강화** | Ghosting 억제 |
| 8 | **Depth supervision 추가** | 기하학적 일관성 |
| 9 | **Multi-scale training** | 다양한 해상도 학습 |

### 8.4 권장 실험 명령어

```bash
# 1. D3 전처리 코드 위치 확인
ls /home/joon/dev/FaceLift/mouse_extensions/preprocessing/

# 2. D3와 M3의 카메라 파라미터 비교
python -c "
import json
d3 = json.load(open('/home/joon/data/preprocessed/FaceLift_mouse/D3_normalized/000/metadata.json'))
m3 = json.load(open('/home/joon/data/preprocessed/FaceLift_mouse/M3_persample/000/metadata.json'))
print('D3:', d3['cameras'][0])
print('M3:', m3['cameras'][0])
"

# 3. UID 통일 실험 (M3_persample에서 UID 44 검증)
# → validation 데이터셋 수정 필요
```

---

## 9. ⚠️ 핵심 발견: M3_persample PP 버그

### 9.1 PP 분포 분석 결과

```
M3_persample (first 100 samples):
  cx: mean=238.3, std=19.5, min=179.0, max=256.0
  cy: mean=237.1, std=21.9, min=169.0, max=256.0

D3_normalized (first 100 samples):
  cx: mean=256.0, std=0.0, min=256.0, max=256.0  ← 완벽히 고정!
  cy: mean=256.0, std=0.0, min=256.0, max=256.0
```

### 9.2 문제점

| 데이터셋 | PP 설정 | 실제 cx | 실제 cy | 문제 |
|----------|---------|---------|---------|------|
| **D3_normalized** | 256 고정 | **256±0** | **256±0** | ✅ 정상 |
| **M3_persample** | 256 고정 (예상) | 238±20 | 237±22 | ❌ **버그** |
| **M3_norm** | 가변 (예상) | **155±17** | **137±20** | ❌ **심각** |

**M3_norm이 더 심각**: cx/cy가 256이 아닌 155/137 근처!

### 9.3 원인 확인 (코드 분석)

**presets.py Line 364**:
```python
"M3_persample": {
    "pp_method": "shift_to_256",
    "force_pp_to_target": False,  # ❌ 이것이 버그!
    ...
}
```

**preprocess.py Line 646**:
```python
# force_pp_to_target = True 일 때만 PP를 256으로 강제
if getattr(cfg, 'force_pp_to_target', False):
    cx, cy = cfg.target_pp  # (256, 256)
# False면 cx, cy가 renorm_scale로 스케일되어 가변값 유지
```

**결론**: `force_pp_to_target: False` 설정 때문에 PP가 256으로 고정되지 않음

### 9.4 해결 방안

**즉시 수정 (presets.py)**:
```python
# mouse_extensions/preprocessing/presets.py Line 364
"M3_persample": {
    ...
    "force_pp_to_target": True,  # ✅ False → True 로 변경
    ...
}
```

**데이터 재생성**:
```bash
cd /home/joon/dev/FaceLift
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_persample \
    --output /home/joon/data/preprocessed/FaceLift_mouse/M3_persample_fixed
```

**Option B**: D3_normalized 방식 채택
- D3_normalized 전처리 파이프라인 사용
- 이미 검증된 방식

### 9.5 가설 수정

| 가설 | 기존 | 수정 |
|------|------|------|
| **H8** | PP=256 고정이 유리할 수 있음 | **PP=256 고정이 필수** (D3 성공 요인) |

---

## 10. 결론 및 즉시 조치사항

### 10.1 핵심 결론

```
🔴 근본 원인 발견:
   M3_persample/M3_norm의 PP가 256으로 고정되지 않음

   D3_normalized: cx=cy=256 (완벽 고정) → PSNR 27.09
   M3_persample:  cx~238, cy~237 (가변) → PSNR 17.24
   M3_norm:       cx~155, cy~137 (가변) → PSNR 16.93

   PP 오프셋이 클수록 → Ghosting 증가, PSNR 감소
```

### 10.2 즉시 조치 (P0)

| # | 작업 | 담당 | 예상 효과 |
|---|------|------|-----------|
| 1 | **M3 전처리 PP 버그 수정** | 코드 수정 | PSNR +10 예상 |
| 2 | **M3_fixed 데이터셋 재생성** | 전처리 재실행 | PP=256 고정 |
| 3 | **D3 전처리 코드 분석** | 코드 리뷰 | 참조 구현 확보 |

### 10.3 전처리 수정 방향

```python
# 현재 (버그):
# Per-sample zoom 후 PP가 변경됨
cx_new = original_cx * zoom_factor  # 가변
cy_new = original_cy * zoom_factor  # 가변

# 수정 필요:
# Zoom 후 PP를 256으로 재매핑
cx_new = 256.0  # 고정
cy_new = 256.0  # 고정
# 이미지도 이에 맞게 crop/shift
```

### 10.4 예상 결과

| 수정 후 | 예상 Val PSNR | 근거 |
|---------|--------------|------|
| M3_fixed (PP=256) | **25+** | D3와 유사 조건 |
| M3_fixed + Coverage 80% | **27+** | D3 능가 가능 |

---

## 11. 가설 검증 로드맵 (수정)

```
Phase 1: D3 분석 (현재)
├─ D3 vs M3 전처리 차이 분석
├─ Val UID 특성 분석
└─ 카메라 파라미터 비교

Phase 2: Ghosting 정량화
├─ view_consistency 지표 구현
├─ 시각화 자동 비교 도구
└─ Temporal coherence 측정

Phase 3: 개선 실험
├─ Depth supervision
├─ Stronger regularization
└─ Calibration refinement
```

---

## 부록: 실험 설정 요약

### A. 진행 중인 실험 (8 GPU 사용)

| GPU | 실험 | 상태 |
|-----|------|------|
| 0 | D3_normalized_E0_1_1_facelift_alpha | 진행중 |
| 1 | D3_normalized_E0_1_3_facelift_alpha_fixed | 진행중 |
| 2 | M3_norm_E0_1_facelift | 진행중 |
| 3 | M3_norm_E1_2 | 완료 |
| 4 | M3_persample_E0_1_facelift | 진행중 |
| 5 | M3_persample_E0_2_mouse | 완료 |
| 6 | M3_persample_E1_2 | 완료 |
| 7 | M3_persample_E0_1_1_facelift_alpha | 진행중 |

### B. 데이터셋 비교

| 항목 | D3_normalized | M3_persample | M3_norm |
|------|---------------|--------------|---------|
| Train | 3240 | 3237 | 3237 |
| Val | 360 | 358 | 358 |
| Val UID | 44 | 202 | 202 |
| Coverage | ~74% | ~80% | ~80% |
| PP | 가변→256 | 256 고정 | 가변 |
| fx | 549 | 549 | 549 |

---

*Comprehensive Analysis v1.0 | 2026-01-25*
*FaceLift Mouse Project*
