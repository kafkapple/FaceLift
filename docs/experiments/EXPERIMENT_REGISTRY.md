# Experiment Registry

> **목적**: 실험 설정 및 결과 기록
> **명령어**: → [[COMMANDS]]
> **Updated**: 260205

---

## 1. 명명 규칙

```
{카테고리}_{번호}_{이름}
예: E0_1_facelift, view_ablation/E0_1_3view
```

---

## 2. 실험 카테고리

| 카테고리 | mask_mode | 설명 |
|----------|-----------|------|
| **E0** | none | FaceLift Baseline ⭐ |
| **E1** | gt | GT Mask 기반 |
| **view_ablation/** | none | View 수 ablation |

---

## 3. View Ablation 결과

### Inference-time (260205)

| Views | PSNR | 비고 |
|-------|------|------|
| **3** | **21.12** | ⭐ Best |
| 2 | 20.20 | |
| 4-6 | 19.58 | Baseline |

### Training (진행 중)

| Config | Views | GPU | 상태 |
|--------|-------|-----|------|
| E0_1_2view | 2 | 6 | 🔄 |
| E0_1_3view | 3 | 5 | 🔄 |
| E0_1_6view | 6 | 7 | 🔄 |

---

## 4. MVDiffusion 실험

| Config | 설명 | 상태 |
|--------|------|------|
| M5t2 | Baseline (CFG=0.05) | ✅ ckpt-5000 |
| M5t2_cyclic | 6x Cyclic Aug | 🔄 GPU 4 |
| M5t2_consistent | CFG=0, FullAttn | ⚠️ 검증필요 |

---

## 5. H1 진단 결과 (260205)

| Dataset | GS-LRM Test | E2E Test | Gap |
|---------|-------------|----------|-----|
| M5t | 20.56 | 19.15 | **+1.41** |
| M5t2 | 19.58 | 19.63 | -0.05 |

**결론**: MVDiffusion이 병목 (undertrained 시)

---

## 6. 체크포인트

| 모델 | 경로 |
|------|------|
| GS-LRM Pretrained | `checkpoints/gslrm/ckpt_*.pt` |
| GS-LRM Best | `checkpoints/gslrm/{exp}/best_psnr.pt` |
| MVDiffusion | `checkpoints/mvdiffusion/mouse_M5t2_cfgr/` |

---

*Registry v3.1 | 260205*
