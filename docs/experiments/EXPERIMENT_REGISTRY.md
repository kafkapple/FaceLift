# Experiment Registry

> **목적**: 실험 설정 및 결과 기록
> **명령어**: → [[COMMANDS]]
> **Updated**: 260211

---

## 1. 명명 규칙

```
{카테고리}_{번호}_{이름}
예: E0_1_facelift, uniform_v2/E0_1_4view
```

---

## 2. 실험 카테고리

| 카테고리 | mask_mode | 설명 |
|----------|-----------|------|
| **E0** | none | FaceLift Baseline ⭐ |
| **E1** | gt | GT Mask 기반 |
| **uniform_v2/** | none | View 수 ablation (uniform sampling) |

---

## 3. View Ablation 결과

### R1: Inference-time (260205, 비균일 샘플링)

| Views | PSNR | 비고 |
|-------|------|------|
| **3** | **21.12** | ⭐ Best (비균일) |
| 2 | 20.20 | |
| 4-6 | 19.58 | Baseline |

> ⚠️ R1은 비균일 뷰 샘플링. v2 (uniform)에서 결과 반전됨

### R2: Uniform v2 Training (✅ 최종, 260210)

| Views | Val PSNR (dB) | Best Step | Δ vs Baseline | 상태 |
|:-----:|:-------------:|:---------:|:-------------:|:----:|
| **6** | **24.49** | 4,201 | +8.50 | ✅ |
| 5 | 23.02 | 13,101 | +7.03 | ✅ |
| 4 | 21.71 | 9,201 | +5.72 | ✅ |
| 3 | 20.01 | 10,701 | +4.02 | ✅ |
| 2 | 17.75 | 11,801 | +1.76 | ✅ |
| 1 | 11.08 | 2,401 | -4.91 | ✅ |
| baseline | 15.99 | 0 | — | ✅ |

**결론**: 뷰 수 ↑ = PSNR ↑ (**단조 증가**, diminishing returns 없음). 6-view 최적.

---

## 4. MV-Diffusion 실험

| Config | sparse_mv | ref_view | 상태 | 결과 |
|--------|:---------:|:--------:|:----:|:----:|
| **M5t2** (baseline) | true | 0 | ✅ ckpt-5000 | PSNR_wh=21.29 |
| M5t2_cfgr | false | 0 | ✅ 완료 | PSNR_wh=20.81 (-0.48 dB) |
| M5t2_3view | true | 0 | 🔄 step 7600/10000 | H8용 |
| ~~M5t2_cyclic~~ | false | all | ❌ 폐기 | 2변수 동시 변경 |
| M5t2_randref_sparse | true | random | ⏳ 대기 | |
| M5t2_20k_sparse | true | 0 | ⏳ 대기 | |

---

## 5. H3 진단 결과

### H3 (260205)

| Dataset | GS-LRM Test | E2E Test | Gap |
|---------|-------------|----------|-----|
| M5t | 20.56 | 19.15 | **+1.41** |
| M5t2 | 19.58 | 19.63 | -0.05 |

### H3-bis 재검증 (260209, metrics_v2)

| Condition | PSNR_wh | PSNR_fg | SSIM | IoU |
|-----------|---------|---------|------|-----|
| A: GS-LRM only | 35.01 | 20.92 | 0.9929 | 0.946 |
| B: E2E M5t2 | 21.21 | 7.91 | 0.9662 | 0.521 |
| C: E2E M5t | 18.28 | 4.66 | 0.9566 | 0.263 |

**결론**: MVDiffusion은 항상 병목 (13-17 dB gap)

---

## 6. MV-Diffusion Ablation (H5: Paper-Aligned)

> **목표**: 원 논문 (Lyu et al., 2024) 설정 재현 + 개별 변수 효과 분리
> **Baseline**: M5t2 (aug=ON, lr=5e-5, 10K steps) → PSNR_wh=21.29

### LR Scaling Note

| | Paper | Ours | Ratio |
|--|-------|------|-------|
| GPUs | 8× A100 | 1× A6000 | 1/8 |
| Effective batch | 64 | 16 | 1/4 |
| LR (paper) | 1e-4 | 1e-4 | ❌ diverge |
| LR (sqrt scaling) | 1e-4 | **5e-5** | ✅ stable |

> `lr_scaled = lr_paper × √(batch_ours / batch_paper) = 1e-4 × √(16/64) = 5e-5`

### Ablation Table

| ID | Config | 변경사항 | LR | Steps | 상태 | 결과 |
|----|--------|---------|:--:|:-----:|:----:|------|
| **E0** | M5t2_paper_aligned | aug=OFF, lr=1e-4, 20K | 1e-4 | 20K | ❌ diverge @3956 | loss→2.5, silent crash |
| **E0v2** | M5t2_E0v2 | aug=OFF, lr=5e-5(scaled), 20K | 5e-5 | 20K | 🔄 | paper-aligned (LR scaled) |
| **E2** | M5t2_noaug | aug=OFF only | 5e-5 | 10K | 🔄 step ~7K | noaug isolation test |
| E6 | (planned) | lr=7.5e-5 only | 7.5e-5 | 10K | ⏳ | intermediate LR test |

### E0 Divergence Analysis (260211)

```
Step ~3600: loss=0.01-0.02 (normal)
Step ~3800: loss=0.039→0.139→0.164 (onset)
Step ~3900: loss=0.345→1.2→2.09→2.51 (explode)
Step  3956: process killed (NaN→CUDA crash)
```

**Root cause**: lr=1e-4 with effective batch=16 (paper uses batch=64).
4× smaller batch → 4× higher gradient variance → diverge.
**Resolution**: Use sqrt-scaled lr=5e-5 for E0v2.

---

## 7. GS-LRM Experiments

### Uniform v2 (View Ablation)

| Views | Val PSNR | Best Step | 상태 |
|-------|----------|-----------|------|
| **6** | **24.49** | 4201 | ✅ completed |
| 5 | 22.63 | 2901 | ✅ completed |
| 4 | 21.50 | 3801 | ✅ completed |
| 3 | 19.92 | 3801 | ✅ completed |
| 2 | 17.70 | 8801 | ✅ completed |
| 1 | 11.08 | 2401 | ✅ completed |

### Paper-Aligned

| Config | Views | LR | Loss | 상태 |
|--------|:-----:|:--:|:----:|:----:|
| paper_aligned_4view | 4 | 1e-4 | no LPIPS/SSIM | 🔄 step ~8K/20K |
| paper_aligned_6view | 6 | 1e-4 | no LPIPS/SSIM | ✅ PSNR=24.49 |

---

## 8. 체크포인트

| 모델 | 경로 |
|------|------|
| GS-LRM Pretrained | `checkpoints/gslrm/ckpt_*.pt` |
| GS-LRM Best | `checkpoints/gslrm/{exp}/best_psnr.pt` |
| MVDiffusion Baseline | `checkpoints/mvdiffusion/mouse_M5t2/` |
| MV-Diffusion noaug (E2) | `/node_data/.../mouse_M5t2_noaug/` |
| MV-Diffusion E0v2 | `/node_data/.../mouse_M5t2_E0v2/` |

---

*Registry v5.0 | 260211*
