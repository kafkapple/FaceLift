# Hypothesis Verification Plan: M3 Variants

## 핵심 가설

### H1: PP Distribution 중요성
> **Pretrained PP=256 분포와 일치할수록 학습 안정성/품질 향상**

- GS-LRM pretrained: cx=cy=256 (고정)
- PP 가변 시: distribution shift → 적응 시간 필요

### H2: Coverage (생쥐 크기) 중요성
> **Coverage가 클수록 (생쥐가 프레임에 크게 차지) 재구성 품질 향상**

- 높은 zoom → 더 많은 pixel → 더 상세한 정보
- 단, 클리핑은 절대 금지 (정보 손실)

---

## 실험 설계

### 실험 Matrix

```
                    PP=256 (고정)              PP=가변
                    ─────────────              ─────────
Coverage 낮음      M3_2b (baseline)           -
                   zoom [1.0, 1.5]

Coverage 높음      M3_3                       M4
                   zoom [1.0, 2.5] + safe     object-centered
                   PP=256 유지                PP correction
```

### 비교 쌍

| 비교 | 프리셋 | 검증 가설 | 차이 변수 |
|------|--------|----------|-----------|
| **A** | M3_2b vs M3_3 | H2 (Coverage) | zoom range (PP 동일) |
| **B** | M3_3 vs M4 | H1 (PP) | PP 고정 vs 가변 (Coverage 유사) |

---

## 프리셋 상세

### M3_2b: Baseline (PP=256, 낮은 Coverage)

```
┌──────────────────────────────────────┐
│                                      │
│        ┌────────────────┐            │
│        │                │            │
│        │      🐭        │  zoom 1.0~1.5
│        │   (작은 크기)  │  PP = 256 고정
│        │                │            │
│        └────────────────┘            │
│                                      │
│          Coverage: ~3%               │
└──────────────────────────────────────┘

특성:
- zoom_range: [1.0, 1.5] (보수적)
- zoom_center_mode: image (중앙 정렬)
- PP: 256 고정
- Clipping: ~0% (통계적)
```

### M3_3: H2 테스트 (PP=256, 높은 Coverage)

```
┌──────────────────────────────────────┐
│                                      │
│     ┌────────────────────────┐       │
│     │                        │       │
│     │         🐭             │  zoom 1.0~2.5
│     │      (큰 크기)         │  + safe_zoom
│     │                        │  PP = 256 고정
│     │                        │       │
│     └────────────────────────┘       │
│                                      │
│          Coverage: ~5% (목표)        │
└──────────────────────────────────────┘

특성:
- zoom_range: [1.0, 2.5] (넓은 범위)
- safe_zoom: True (클리핑 방지)
- zoom_center_mode: image (중앙 정렬)
- PP: 256 고정
- Clipping: 0% (safe_zoom으로 보장)

Safe Zoom 원리:
  safe_zoom = output_size / (2 * max_dist_from_center)
  final_zoom = min(coverage_zoom, safe_zoom)
```

### M4: H1 테스트 (PP 가변, 최대 Coverage)

```
원본 이미지:
┌──────────────────────────────────────┐
│                          🐭          │  객체가 edge 근처
│                     (off-center)     │
└──────────────────────────────────────┘
           ↓ object-centered crop
┌──────────────────────────────────────┐
│                                      │
│     ┌────────────────────────┐       │
│     │                        │       │
│     │         🐭             │  항상 중앙
│     │      (최대 크기)       │  최대 zoom 가능
│     │                        │       │
│     └────────────────────────┘       │
│                                      │
│          Coverage: ~5% (목표)        │
└──────────────────────────────────────┘

PP Correction:
  crop_offset = object_center - image_center
  new_cx = original_cx - crop_offset.x
  new_cy = original_cy - crop_offset.y

  → Ray direction 보존 (MVG 정확성 유지)

특성:
- zoom_center_mode: object (객체 중심)
- pp_correction: True (PP 동적 조정)
- PP: 가변 (crop offset 반영)
- Clipping: 0% (객체가 항상 중앙)
- Coverage: 최대 (객체 위치 무관)
```

---

## 예상 결과

### H2 검증: M3_2b vs M3_3

| 지표 | M3_2b | M3_3 | 예상 |
|------|-------|------|------|
| PSNR | 기준 | +0.5~1.0? | M3_3 승 (높은 coverage) |
| SSIM | 기준 | +? | M3_3 승 |
| fg_coverage | ~0.05 | ~0.05 | 동일 (정상) |
| Clipping | ~0% | 0% | 동일 |

**예상**: H2 참 → **Coverage 클수록 품질 향상**

### H1 검증: M3_3 vs M4

| 지표 | M3_3 | M4 | 예상 |
|------|------|-----|------|
| PSNR | 기준 | ? | PP shift 영향 |
| 초기 수렴 | 빠름 | 느림? | PP distribution shift |
| 최종 품질 | 기준 | ≥M3_3? | 적응 후 동등 이상 |

**예상 시나리오**:
- H1 참 (PP 중요): M3_3 승 (pretrained 일치)
- H1 거짓 (PP 덜 중요): M4 승 또는 동등 (최대 coverage)

---

## 실험 명령어

### 1. 전처리

```bash
cd /home/joon/dev/FaceLift

# 모든 variant 전처리 (M3_3, M3_2b, M4)
./scripts/preprocess_m3_variants.sh all

# 또는 개별
./scripts/preprocess_m3_variants.sh M3_2b
./scripts/preprocess_m3_variants.sh M3_3
./scripts/preprocess_m3_variants.sh M4
```

### 2. 클리핑 검증

```bash
for ds in M3_2b M3_3 M4; do
    python mouse_extensions/scripts/analysis/clipping_analyzer.py \
        /home/joon/data/preprocessed/FaceLift_mouse/$ds -r 50 -m 10
done
```

### 3. 학습

```bash
# M3_2b (Baseline)
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_gslrm.py \
    -d M3_2b -e E1_2_alpha_M3_2b

# M3_3 (H2 Test)
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 train_gslrm.py \
    -d M3_3 -e E1_2_alpha_M3_3

# M4 (H1 Test)
CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=1 train_gslrm.py \
    -d M4 -e E1_2_alpha_M4
```

---

## 평가 지표

### 정량적
| 지표 | 의미 | 목표 |
|------|------|------|
| **PSNR** | 픽셀 정확도 | 높을수록 좋음 |
| **SSIM** | 구조적 유사도 | 높을수록 좋음 |
| **LPIPS** | 지각적 유사도 | 낮을수록 좋음 |
| **fg_coverage** | Ghosting 지표 | <0.05 정상 |

### 정성적
- Turntable 비교 (360도 회전)
- Alpha comparison (FN/FP 시각화)
- GT vs Pred 비교

---

## 결과 기록 템플릿

### 전처리 검증

| Dataset | Samples | Clipping | Avg Coverage | PP |
|---------|---------|----------|--------------|-----|
| M3_2b | - | -% | -% | 256 |
| M3_3 | - | -% | -% | 256 |
| M4 | - | -% | -% | 가변 |

### 학습 결과

| Exp | Dataset | PSNR | SSIM | LPIPS | fg_coverage |
|-----|---------|------|------|-------|-------------|
| E1 | M3_2b | - | - | - | - |
| E2 | M3_3 | - | - | - | - |
| E3 | M4 | - | - | - | - |

---

## 결론 템플릿

### H2 결론 (Coverage 영향)
- M3_2b vs M3_3 비교:
- Coverage 증가 효과:
- 결론:

### H1 결론 (PP 영향)
- M3_3 vs M4 비교:
- PP distribution shift 영향:
- 결론:

### 최종 권장 프리셋
- 단기 (안정성 우선):
- 장기 (품질 우선):

---

*Created: 2026-01-26 | FaceLift Mouse Hypothesis Verification*
