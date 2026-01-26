# Hypothesis Verification Plan: M3 Variants

> **목적**: PP Distribution과 Coverage가 재구성 품질에 미치는 영향 검증
> **기준선**: M3_2 + E1_2_alpha (현재 권장 설정)

---

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

## 실험 매트릭스

| 우선순위 | 가설 | 대조 비교 | 차이 변수 | 고정 변수 |
|----------|------|-----------|-----------|-----------|
| **P1** | H2: Coverage 영향 | M3_2b vs M3_3 | zoom_range | PP=256 동일 |
| **P2** | H1: PP Distribution | M3_3 vs M4 | PP (256 vs 가변) | Coverage 유사 |

---

## 데이터셋 종합 비교표

| Setting | D7.1 (M1) | D8 (M2) | M3_1 | M3_2 | M3_2b | M3_3 | M4 | D10.3 |
|---------|-----------|---------|------|------|-------|------|-----|-------|
| **paradigm** | pp_centered | precision_homo | precision_homo | precision_homo | precision_homo | precision_homo | **object_centered** | precision_homo |
| **transform** | affine | homography | homography | homography | homography | homography | homography | homography |
| **pp_method** | shift_to_256 | shift_to_256 | shift_to_256 | shift_to_256 | shift_to_256 | shift_to_256 | **pp_correction** | shift_to_256 |
| **skew_correction** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **adaptive_zoom** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **zoom_scope** | - | - | global | **per_sample** | per_sample | per_sample | per_sample | global |
| **zoom_center_mode** | - | - | image | image | image | image | **object** | object |
| **zoom_range** | - | - | [1.0,1.8] | [1.0,1.8] | **[1.0,1.5]** | **[1.0,2.5]** | [1.0,2.5] | [1.0,2.5] |
| **safe_zoom** | - | - | ❌ | ❌ | ❌ | **✅** | ✅ | ❌ |
| **pp_correction** | - | - | ❌ | ❌ | ❌ | ❌ | **✅** | ❌ |
| **normalize_after_zoom** | - | - | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **target_fx** | 549 | 549 | 549 | 549 | 549 | 549 | 549 | **739** |
| **PP 결과** | 256 | 256 | 256 | 256 | 256 | 256 | **가변** | **가변** |
| **Coverage** | ~50% | ~50% | ~78% | ~78% | ~60% | ~78% | ~78% | ~78% |
| **Clipping** | 0% | 0% | 0% | 0% | 0% | **0%** | 0% | ~6% |
| **상태** | ✅ 기준선 | ✅ 정밀 | ✅ 안정 | ⭐ **권장** | 🔬 H2 baseline | 🔬 H2 test | 🔬 H1 test | ⚠️ fx 버그 |

---

## M3 Variants 핵심 차이

```
M3_2 (권장 기준선)
├── zoom_scope: per_sample
├── zoom_center_mode: image
├── zoom_range: [1.0, 1.8]
├── PP: 256 (고정)
└── Coverage: ~78%

M3_2b (H2 baseline - 낮은 coverage)
├── 차이: zoom_range [1.0, 1.5] (보수적)
├── PP: 256 (고정)
└── Coverage: ~60% (낮음)

M3_3 (H2 test - 높은 coverage + safe zoom)
├── 차이: zoom_range [1.0, 2.5], safe_zoom=True
├── PP: 256 (고정)
└── Coverage: ~78% (0% clipping 보장)

M4 (H1 test - PP correction)
├── 차이: zoom_center_mode=object, pp_correction=True
├── PP: 가변 (crop offset 반영)
└── Coverage: ~78% (최대)
```

---

## E 실험 설정 비교

| 설정 | E0_1_facelift | E1_2_alpha | E1_3_lgm | E2_1_alpha |
|------|---------------|------------|----------|------------|
| **mask_mode** | none | **gt** | gt | none |
| **normalize_by_mask** | ❌ | **✅** | ✅ | ❌ |
| **alpha_loss_weight** | 0.0 | **0.1** | 1.0 | 0.1 |
| **alpha_loss_type** | - | mse | mse | mse |
| **num_input_views** | 4 | 4 | 4 | 4 |

---

## 실험 실행 명령어

### Phase 1: 전처리 (병렬 실행 가능)

```bash
cd /home/joon/dev/FaceLift

# M3_2b (H2 baseline)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_2b \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_2b

# M3_3 (H2 test)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_3 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_3

# M4 (H1 test)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M4 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M4
```

### Phase 2: 클리핑 검증

```bash
for ds in M3_2b M3_3 M4; do
    python mouse_extensions/scripts/analysis/clipping_analyzer.py \
        /home/joon/data/preprocessed/FaceLift_mouse/$ds -r 50 -m 10
done
```

### Phase 3: 학습 (순차 또는 병렬)

```bash
# H2 검증: Coverage 영향
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2b -e E1_2_alpha  # H2 baseline

CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_3 -e E1_2_alpha   # H2 test

# H1 검증: PP 영향 (H2 완료 후)
CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M4 -e E1_2_alpha     # H1 test

# 기준선 (이미 완료된 경우 생략)
CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E1_2_alpha   # Baseline
```

---

## 예상 결과

### H2 검증: M3_2b vs M3_3

| 지표 | M3_2b | M3_3 | 예상 |
|------|-------|------|------|
| PSNR | 기준 | +0.5~1.0? | M3_3 승 (높은 coverage) |
| SSIM | 기준 | +? | M3_3 승 |
| fg_coverage | ~0.03 | ~0.05 | 정상 범위 |
| Clipping | ~0% | 0% | 동일 |

### H1 검증: M3_3 vs M4

| 지표 | M3_3 | M4 | 예상 |
|------|------|-----|------|
| PSNR | 기준 | ? | PP shift 영향 불확실 |
| 초기 수렴 | 빠름 | 느림? | PP distribution shift |
| 최종 품질 | 기준 | ≥M3_3? | 적응 후 동등 이상 가능 |

### 결과 해석

**H1 참 (PP 중요)**: M3_3 승 → PP=256 유지 권장
**H1 거짓 (PP 덜 중요)**: M4 승 또는 동등 → 최대 coverage 추구 가능

---

## 평가 지표

### 정량적
| 지표 | 의미 | 목표 |
|------|------|------|
| **PSNR** | 픽셀 정확도 | 높을수록 좋음 |
| **SSIM** | 구조적 유사도 | 높을수록 좋음 |
| **LPIPS** | 지각적 유사도 | 낮을수록 좋음 |
| **fg_coverage** | Ghosting 지표 | <0.05 정상, >0.05 ghosting |

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
| Baseline | M3_2 | - | - | - | - |
| H2-base | M3_2b | - | - | - | - |
| H2-test | M3_3 | - | - | - | - |
| H1-test | M4 | - | - | - | - |

---

*Created: 2026-01-26 | FaceLift Mouse Hypothesis Verification v2.0*
