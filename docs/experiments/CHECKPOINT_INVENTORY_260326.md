# Checkpoint Inventory & Experiment Plan

> **Navigation**: [← Index](../INDEX.md) | [← Experiment Registry](EXPERIMENT_REGISTRY.md)
> **Date**: 2026-03-26
> **Status**: P0 실행 중, P1 계획 수립

---

## 1. Checkpoint Inventory

### Category A: View Ablation (논문 backbone)

| Views | α | Checkpoint | PSNR (val) | Status |
|:-----:|:-:|-----------|:----------:|:------:|
| 1 | 0.0 | — | (10.47 WandB) | ❌ 재학습 필요 |
| 2 | 0.0 | — | (15.95 WandB) | ❌ 재학습 필요 |
| 3 | 0.0 | — | (18.56 WandB) | ❌ 재학습 필요 |
| **4** | 0.0 | `base_uniform_v2_4view_v2` | **21.71** | ✅ |
| 5 | 0.0 | — | (22.16 WandB) | ❌ 재학습 필요 |
| **6** | 0.0 | `base_uniform_v2_6view_v2` | **24.49** | ✅ |

> ⚠️ WandB 숫자는 논문 보고 불가 — 체크포인트 없으면 재현 불가 (NeurIPS 기준)

### Category B: Alpha Ablation — 4-view (✅ 완료)

| α | Checkpoint | PSNR (val) | Step |
|:-:|-----------|:----------:|:----:|
| 0.0 | `base_uniform_v2_4view_v2` | 21.71 | 9201 |
| 0.3 | `base_uniform_v2_4view_alpha03_v3` | 21.34 | 10201 |
| 0.5 | `base_uniform_v2_4view_alpha05_v3` | 21.20 | 8901 |
| 1.0 | `base_uniform_v2_4view_alpha10_v3` | 20.84 | 8101 |

### Category C: Alpha Ablation — 6-view (⚠️ eval 필요)

| α | Checkpoint | PSNR_gt | PSNR_int | IoU | Sil.Prec | N_gs | Status |
|:-:|-----------|:-------:|:--------:|:---:|:--------:|:----:|:------:|
| 0.0 | `base_uniform_v2_6view_v2` | **20.12** | 31.09 | 0.886 | 0.886 | 18,027 | ✅ |
| **0.3** | `M5t2_6view_alpha03_v3` | **20.01** | 32.51 | **0.913** | **0.914** | 17,274 | ✅ **best trade-off** |
| 0.5 | `M5t2_6view_alpha05_v3` | 19.81 | 32.77 | 0.918 | 0.919 | 17,165 | ✅ |
| 1.0 | `M5t2_6view_alpha10_v3` | 19.55 | **33.14** | **0.925** | **0.926** | **17,096** | ✅ best IoU/Prec |

> **3-Best Checkpoint Rule** (용도별 체크포인트 선택 기준):
>
> | 기준 | Best α | Checkpoint | 용도 |
> |------|:------:|-----------|------|
> | **FG Quality** (PSNR_gt) | **0.0** | `base_uniform_v2_6view_v2` | 정량 비교, 논문 main table |
> | **Trade-off** (IoU/dB knee) | **0.3** | `M5t2_6view_alpha03_v3` | Practical recommendation, 시각화 기본 |
> | **Artifact** (IoU, Sil.Prec) | **1.0** | `M5t2_6view_alpha10_v3` | 데모, 정성 비교, 깔끔한 렌더링 |
>
> **Marginal analysis**: α=0.3에서 IoU gain/PSNR cost = 0.245 (10× 효율). α>0.3은 diminishing returns.

### Category D: 기타

| Experiment | Views | PSNR | 용도 |
|-----------|:-----:|:----:|------|
| M5t2_E0_1_facelift | 4 | 22.34 | 원본 config |
| hp_M5_4 / hp_M5_5 | 4 | 21.83/21.70 | HP search |
| domain_adapt_E2 | 4 | 21.42 | DA |
| SSIM 0.3/0.5/1.0 | 4 | 21.10-21.30 | SSIM weight |

## 2. Priority Plan

| Priority | 작업 | 비용 | 시간 | 근거 |
|:--------:|------|:----:|:----:|------|
| **P0** | 6v alpha eval (4개) | GPU 1시간 | 즉시 | 체크포인트 있음, eval만 |
| **P1** | 1/2/3/5v 재학습 (α=0.0) | GPU 2-3일 | P0 후 | 논문 핵심 |
| **P2** | Artifact 정성 비교 + metric 선정 | 낮음 | P0 후 | P0 결과 기반 |
| **P3** | Pruning ablation 설계 | 중간 | P1 병행 | 최적 filter 탐색 |

## 3. Execution Log

- 2026-03-26: P0 시작 (6v alpha fair eval)
- 2026-03-26: P0 완료 ✅ — comprehensive eval (PSNR_gt/int, IoU, Sil.Prec/Rec, SSIM, LPIPS, per-cam, Gaussian stats)
- 2026-03-26: α=0.3 best trade-off 확인 (marginal analysis: 0.245 IoU/dB)
- 2026-03-26: Cinematic 768px/30fps 생성 중 (GPU 5)
- [ ] P1: 1/2/3/5v 재학습 (α=0.0)
- [ ] Pruning ablation E1-E5

---

*Checkpoint Inventory v1.0 | 2026-03-26*
