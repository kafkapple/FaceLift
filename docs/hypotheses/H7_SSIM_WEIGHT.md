# H7: SSIM Loss Weight Ablation

> **가설**: SSIM loss weight를 높이면 구조적 보존이 개선되어 mouse 재구성 품질이 향상될 것이다.
>
> ← [RESEARCH_HYPOTHESES.md](../RESEARCH_HYPOTHESES.md) | **상태**: ⏳ 대기 | **Updated**: 2026-02-09

---

## 1. 배경

### 1.1 현상 관찰
- H4 View Ablation 실험에서 `train/ssim_loss` 패턴 관찰:
  - 초기 500 steps: SSIM loss 빠르게 감소 (구조 학습)
  - 이후: SSIM loss 천천히 증가 (L2/perceptual에 밀림)
- **해석**: 현재 SSIM weight 0.1이 다른 loss (L2=1.0, perceptual=0.5) 대비 너무 낮아
  L2 최적화에 의해 구조적 일관성이 희생됨

### 1.2 문헌 근거
| 논문 | SSIM Weight | Loss 조합 | 결과 |
|------|------------|-----------|------|
| GS-LRM (원본) | 0.1 | L2 + SSIM + LPIPS | 보수적 |
| 3DGS (Kerbl) | 0.2 | 0.8×L1 + 0.2×SSIM | 구조 중시 |
| Instant-3D | 0.5 | MSE + SSIM | 균형 |
| Splatter Image | ~0.5 | L1 + SSIM | 50:50 비율 |
| NeRF (Mildenhall) | - | MSE only | SSIM 미사용 |

### 1.3 이론적 근거
- **L2 Loss**: 픽셀 단위 정확도, 블러 경향
- **SSIM Loss**: 구조적 유사도 (luminance, contrast, structure)
  - Window 기반 (11×11) → 지역 패턴 보존
  - Edge, texture 보존에 효과적
- **Trade-off**: SSIM↑ → 구조 선명도↑ but PSNR slightly↓ (known trade-off)

---

## 2. 실험 설계

### 2.1 변수

| Config | ssim_weight | L2:SSIM 비율 | 목적 |
|--------|-------------|-------------|------|
| 4view_v2 (baseline) | **0.1** | 10:1 | 기준선 (GS-LRM 기본) |
| 4view_ssim03_v2 | **0.3** | 3.3:1 | 적당한 구조 강조 |
| 4view_ssim05_v2 | **0.5** | 2:1 | 균형점 (Instant-3D 수준) |
| 4view_ssim10_v2 | **1.0** | 1:1 | 극단적 구조 강조 |

### 2.2 고정 조건
- 4-view input, random_view_selection=true
- 나머지 loss weight 동일 (base_uniform_v2.yaml)
- 15K fwdbwd_passes (~15840 actual steps), seed=42, M5t2 dataset

### 2.3 평가 메트릭

| 메트릭 | 기대 변화 | 의미 |
|--------|----------|------|
| val/ssim | ↑ | 직접 최적화 대상 |
| val/psnr | ↓ (소폭) | L2 vs SSIM trade-off |
| val/mask_iou | ↑ (약간) | 구조 보존 → boundary 개선 |
| train/ssim_loss | 단조 감소 유지 | 높은 weight → 지속 최적화 |

### 2.4 핵심 비교

**Primary**: `ssim_loss 곡선 형태`
- baseline (0.1): 감소 → 증가 (✗ 구조 희생)
- 0.3-0.5: 감소 → 안정 (◯ 균형)
- 1.0: 단조 감소 (△ PSNR 희생?)

**Secondary**: `PSNR vs SSIM Pareto Frontier`
- 이상적: PSNR 유지 + SSIM 향상
- 현실적: PSNR 소폭 감소 + SSIM 명확 향상

---

## 3. 실행 명령어

```bash
cd /home/joon/dev/FaceLift

# SSIM 0.3
export CUDA_VISIBLE_DEVICES=4 && nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/4view_ssim03_v2.yaml \
    > logs/uniform_4view_ssim03_v2.log 2>&1 &

# SSIM 0.5
export CUDA_VISIBLE_DEVICES=5 && nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/4view_ssim05_v2.yaml \
    > logs/uniform_4view_ssim05_v2.log 2>&1 &

# SSIM 1.0
export CUDA_VISIBLE_DEVICES=6 && nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/4view_ssim10_v2.yaml \
    > logs/uniform_4view_ssim10_v2.log 2>&1 &
```

---

## 4. 예상 결과

### 4.1 시나리오 분석

| 시나리오 | ssim_weight | PSNR | SSIM | 판단 |
|----------|------------|------|------|------|
| A: 0.3 최적 | 0.3 | ~-0.3dB | +0.01 | ✅ 권장 |
| B: 0.5 최적 | 0.5 | ~-0.5dB | +0.02 | ✅ 균형 |
| C: 높을수록 좋음 | 1.0 | ~-1.0dB | +0.03 | ⚠️ trade-off |
| D: 기본이 최적 | 0.1 | 최고 | 기준 | 변경 불필요 |

### 4.2 후속 실험

| 결과 | 후속 |
|------|------|
| 0.3이 Pareto 최적 | H6+H7 조합 (alpha 0.1 + ssim 0.3) |
| 높을수록 좋음 | 더 높은 weight 테스트 (2.0) |
| PSNR 크게 감소 | L2 weight도 함께 조정 |

---

## 5. Config 파일 위치

```
configs/mouse/uniform/
├── 4view_ssim03_v2.yaml    # ssim_weight=0.3
├── 4view_ssim05_v2.yaml    # ssim_weight=0.5
└── 4view_ssim10_v2.yaml    # ssim_weight=1.0
```

---

## 6. 현황

| 상태 | 내용 |
|------|------|
| ✅ 완료 | 문헌 조사 |
| ✅ 완료 | Config 생성 (3개) |
| ⏳ 대기 | H4 Round 1 완료 후 진행 |

---

## 7. 참고 문헌

1. Kerbl et al. "3D Gaussian Splatting" (SIGGRAPH 2023) - L1+SSIM (0.8:0.2)
2. Wang et al. "Image Quality Assessment: From Error Visibility to Structural Similarity" (IEEE TIP 2004)
3. Li et al. "Instant3D" (ICLR 2024) - MSE+SSIM (1:0.5)
4. Szymanowicz et al. "Splatter Image" (CVPR 2024)

---

*H7 SSIM Weight | v1.0 | 2026-02-07*
