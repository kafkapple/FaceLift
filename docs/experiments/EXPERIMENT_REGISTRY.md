# Experiment Registry

> **목적**: 실험 설정 및 결과 기록
> **명령어**: → [[COMMANDS]]
> **Updated**: 260209

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

### R2: Uniform v2 Training (260209, 균일 샘플링)

| Views | Val PSNR | Best Step | GPU | 상태 |
|-------|----------|-----------|-----|------|
| **6** | **23.46** | 601 | 7 | 🔄 step 5800/15840 |
| 5 | 22.63 | 2901 | 6 | 🔄 step 8100/15840 |
| 4 | 21.50 | 3801 | 4 | 🔄 step 8000/15840 |
| 3 | 19.92 | 3801 | 5 | 🔄 step 12750/15840 |
| 2 | 17.70 | 8801 | - | ✅ completed |
| 1 | 11.08 | 2401 | - | ✅ completed |
| baseline | 15.99 | 0 | - | ✅ zero-shot |

**결론 (잠정)**: 뷰 수 ↑ = PSNR ↑ (단조 증가). R1과 반전.

---

## 4. MVDiffusion 실험

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

## 6. 체크포인트

| 모델 | 경로 |
|------|------|
| GS-LRM Pretrained | `checkpoints/gslrm/ckpt_*.pt` |
| GS-LRM Best | `checkpoints/gslrm/{exp}/best_psnr.pt` |
| MVDiffusion Baseline | `checkpoints/mvdiffusion/mouse_M5t2/` |
| MVDiffusion cfgr | `checkpoints/mvdiffusion/mouse_M5t2_cfgr/` |

---

*Registry v4.0 | 260209*
