# Preprocessing Comparison: M5t2 (Mouse) vs RAT2 (Rat)

> **Navigation**: [← INDEX](../INDEX.md) | [PREPROCESSING_REGISTRY](PREPROCESSING_REGISTRY.md) | [RAT_PREPROCESSING_STRATEGY](../specs/RAT_PREPROCESSING_STRATEGY.md)
> **Purpose**: M5t2와 RAT2 전처리 파라미터 정량 비교 — FT 학습 시 domain gap 원인 분석
> **Created**: 2026-03-30
> **Data Source**: gpu03 실측 (100 UID 샘플링)

---

## 1. Summary

| Parameter | M5t2 (Mouse) | RAT2 (Rat) | Gap | Impact |
|-----------|:------------:|:----------:|:---:|:------:|
| **Total UIDs** | 3,600 | 2,967 | -18% | Moderate |
| **Split (train/val/test)** | 2880/360/360 (80/10/10) | 2385/290/292 (80/10/10) | ~same ratio | Low |
| **Image size** | 512×512 RGBA | 512×512 RGBA | None | None |
| **fx** | 549.0 (uniform) | 595–612 (per-camera) | +8–11% | Low (GS-LRM explicit) |
| **fy** | 549.0 (uniform) | 595–612 (per-camera) | +8–11% | Low |
| **cx** | 256.0 (centered) | 251–261 (per-camera) | ±5px | Low |
| **cy** | 256.0 (centered) | 233–260 (per-camera) | -23px max offset | Low–Moderate |
| **Camera distance** | 2.59–2.80 | 2.46–2.90 | Wider range | Low |
| **FG coverage** | **2.52%** ± 0.88% | **1.55%** ± 0.62% | **-38%** | **High** |
| **FG pixels** (per 512²) | ~6,600 | ~4,060 | -38% | **High** |
| **Preprocessing** | M5 recentered_affine | Zero-pad | Different | See §3 |
| **UID spacing** | Consecutive (step=1) | Stride-30 (HLAC-stratified) | Different | See §4 |

---

## 2. Camera Intrinsics

### M5t2 (Mouse)
- **Normalized**: fx=fy=549, cx=cy=256 (all 6 cameras identical)
- **Method**: M5 recentered_affine — skew 제거 + affine transform으로 canonical 형태 생성
- Camera distance ~2.7 (pretrained GS-LRM 호환)

### RAT2 (Rat)
- **Raw calibration**: fx=595–612, fy=595–612 (per-camera 고유값)
- **PP offset**: cx ≈ 중앙이나, cy = 233–260 (최대 23px 하방 편향)
- **Method**: Zero-pad — 원본 해상도를 512×512으로 zero-padding, intrinsics 보존
- Camera distance 2.46–2.90 (mouse보다 넓은 범위)

### fx 차이 영향도 (Low)
- GS-LRM은 Plücker ray에 fx를 explicit하게 사용 → 모델이 geometry 인식
- `/deliberate --audit --loop 3` 결과: fx 조정 시 FG 감소 trade-off (Pareto-suboptimal)
- 상세: `docs/specs/RAT_PREPROCESSING_STRATEGY.md` §3

---

## 3. Foreground Coverage (Primary Gap)

| Metric | M5t2 | RAT2 | Ratio |
|--------|:----:|:----:|:-----:|
| Mean FG ratio | 2.52% | 1.55% | 0.62× |
| Std | 0.88% | 0.62% | — |
| Min | 1.09% | 0.65% | — |
| Max | 5.66% | 3.55% | — |

**해석**:
- RAT2의 foreground가 M5t2 대비 38% 작음
- **원인**: Rat 몸체가 상대적으로 작고, SAM2 마스크가 tighter
- **영향**: Pretrained 모델이 더 작은 foreground에 적응해야 함 → FT 초기 domain gap
- **완화책**: Zero-pad 전처리가 Pareto-optimal (RAT_PREPROCESSING_STRATEGY.md 결론)

---

## 4. Split Strategy

### M5t2: Temporal Split
- UID 0–2879 (train), 2880–3239 (val), 3240–3599 (test)
- 시간순 연속 — temporal leakage 방지
- 모든 행동 패턴이 train에 충분히 포함

### RAT2: HLAC-Stratified 2-Phase Split
- Phase 1: HLAC 8-class 균등 분포로 test set 선택
- Phase 2: 나머지에서 val 분리
- UID stride=30 (30fps → 1초 간격 = temporal independence)
- 최대 step=1290 (43초 gap) — 일부 긴 gap은 미사용 구간

---

## 5. RAT1 → RAT2 FT Baseline

| Metric | RAT1 (100-UID) | RAT2 (in progress) | Notes |
|--------|:--------------:|:-------------------:|-------|
| Zero-shot | 2.77 dB | 2.97 dB | Consistent |
| Val PSNR (FT) | 17.49 dB | TBD | RAT2: 2385 train UIDs |
| Train PSNR | ~35 dB | ~19.5 dB (step 150) | Early stage |
| Data size | 800 train | 2385 train | 3× more |

**기대**: RAT2는 데이터 3배 → RAT1의 17.5 dB overfitting gap (35-17.5) 축소 예상.

---

## 6. Key Takeaways

1. **FG coverage gap이 가장 큰 domain gap 요인** (38% 감소)
2. fx 차이는 GS-LRM이 explicit하게 처리하므로 저영향
3. cy offset (최대 23px)은 minor but nonzero — bottom-view artifact 가능성
4. RAT2의 HLAC-stratified split은 행동 다양성 보장 면에서 M5t2보다 우수
5. Zero-pad 전처리는 3-model 3-loop audit 결과 Pareto-optimal로 확인

---

*Created: 2026-03-30 | Cross-ref: RAT_PREPROCESSING_STRATEGY.md, PREPROCESSING_REGISTRY.md §9*
