# 6-View Alpha Comprehensive Evaluation Report

> **Navigation**: [← Index](../INDEX.md) | [← Checkpoint Inventory](CHECKPOINT_INVENTORY_260326.md)
> **Date**: 2026-03-26
> **Code**: `mouse_extensions/scripts/eval/comprehensive_eval.py`
> **Data**: `outputs/eval/mouse/comprehensive_6v/eval_summary.json`

---

## 1. Experiment Overview

### Goal
6-view GS-LRM에서 alpha supervision weight (α=0.0, 0.3, 0.5, 1.0)가 재구성 품질과 artifact에 미치는 영향을 **masked foreground** 및 **artifact-specific metric** 관점에서 종합 평가.

### Setup

| Item | Value |
|------|-------|
| Model | GS-LRM 6-view (configs/base/gslrm_mouse.yaml) |
| Input resolution | 512×512 |
| Patch size | 8 |
| n_gaussians | 2 per ray |
| Total Gaussians (raw) | (512/8)² × 6 × 2 = 49,152 |
| Evaluation set | M5t2 test split (360 frames, indices 3240-3599) |
| Evaluation views | All 6 cameras per frame (360 × 6 = 2,160 evaluations) |
| Filter params | opacity=0.04, scaling=0.1, floater=0.6, bbox=±0.91 |
| PSNR_gt mask | GT alpha > 0.5 binary (from RGBA PNG) |

### Checkpoints

| α | Checkpoint | Training Step |
|:-:|-----------|:------------:|
| 0.0 | `base_uniform_v2_6view_v2/best_psnr.pt` | 4,201 |
| 0.3 | `M5t2_6view_alpha03_v3/ckpt_15840.pt` (symlink) | 15,840 |
| 0.5 | `M5t2_6view_alpha05_v3/ckpt_15840.pt` (symlink) | 15,840 |
| 1.0 | `M5t2_6view_alpha10_v3/ckpt_15840.pt` (symlink) | 15,840 |

---

## 2. Results

### 2.1 Full Comparison Table

| α | PSNR_gt↑ | PSNR_int↑ | IoU↑ | Sil.Prec↑ | Sil.Rec | SSIM↑ | LPIPS↓ | N_gs | Floater |
|:-:|:--------:|:---------:|:----:|:---------:|:-------:|:-----:|:------:|:----:|:-------:|
| **0.0** | **20.12 ± 1.88** | 31.09 | 0.886 | 0.886 | 1.000 | 0.989 | 0.012 | 18,027 | 0.001 |
| **0.3** | **20.01 ± —** | 32.51 | **0.913** | **0.914** | — | 0.991 | 0.011 | 17,274 | 0.000 |
| 0.5 | 19.81 | 32.77 | 0.918 | 0.919 | — | 0.991 | 0.011 | 17,165 | 0.001 |
| **1.0** | 19.55 | **33.14** | **0.925** | **0.926** | — | **0.991** | **0.011** | **17,096** | 0.001 |

### 2.2 Per-Camera PSNR_gt (α=0.0)

| cam_0 | cam_1 | cam_2 | cam_3 | cam_4 | cam_5 |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 19.54 | 20.40 | 19.92 | **21.91** | 20.56 | **18.35** |

> cam_5 (18.35) = 최저 → bottom/side view. cam_3 (21.91) = 최고 → 가장 직접적 뷰.

### 2.3 Gaussian Statistics (α=0.0)

| Metric | Value |
|--------|:-----:|
| N_gaussians (filtered) | 18,027 |
| Opacity mean | 0.258 |
| Anisotropy % flat (ratio≥30) | ~94% |
| Floater fraction (DBSCAN) | 0.001 |

---

## 3. Key Findings

### 3.1 PSNR_gt vs PSNR_int: 반대 방향 Trade-off

```
α 증가 →  PSNR_gt ↓ (전경 세부 표현 약간 희생)
          PSNR_int ↑ (배경 깨끗 → white-BG MSE 감소)
```

**원인**: α supervision이 opacity 분포를 양극화하여 배경 Gaussian을 투명하게 만듦. 이는 PSNR_int (배경 포함)에서 유리하지만, 전경 세부 텍스처가 약간 뭉개져 PSNR_gt (전경만)에서 불리.

> ⚠️ **이전 P0 eval (PSNR_int only)에서 α=1.0이 "best"로 보였던 이유**: PSNR_int만 측정했기 때문. PSNR_gt를 함께 보면 trade-off가 명확.

### 3.2 Marginal Analysis: α=0.3 = Knee Point

| 구간 | ΔPSNR_gt | ΔIoU | **IoU gain / PSNR cost** |
|:----:|:--------:|:----:|:------------------------:|
| 0.0→0.3 | -0.11 dB | +0.027 | **0.245** (최고 효율) |
| 0.3→0.5 | -0.20 dB | +0.005 | 0.025 (10× 하락) |
| 0.5→1.0 | -0.26 dB | +0.007 | 0.027 |

α=0.3에서 IoU 이득/PSNR 비용 비율이 **10배 급락** → diminishing returns 시작 = knee point.

### 3.3 Silhouette Precision

| α | Sil.Precision | 해석 |
|:-:|:------------:|------|
| 0.0 | 0.886 | 배경에 11.4% 불필요 Gaussian |
| 0.3 | 0.914 | 배경 Gaussian 8.6%로 감소 |
| 1.0 | 0.926 | 배경 Gaussian 7.4%로 최소 |

α 증가가 배경 artifact (floater)를 체계적으로 감소시킴.

### 3.4 CLAUDE.md PSNR과의 차이

| 출처 | PSNR | 기준 | 비고 |
|------|:----:|------|------|
| CLAUDE.md §11 | 23.84 | "fair eval" | Holdout view? Train set? 정확한 eval 조건 불명 |
| 이 보고서 | 20.12 | PSNR_gt (masked FG, test set, all 6 views) | Comprehensive eval |
| P0 eval | 31.09 | PSNR_int (white-BG, test set) | 배경 포함 |

> 차이 원인: (1) eval 프로토콜 차이 (mask 방식), (2) dataset split (train vs test), (3) view selection (holdout vs all). 향후 CLAUDE.md 수치는 본 보고서 기준으로 통일 필요.

---

## 4. Recommendation

### Best Trade-off: α=0.3

- PSNR_gt: -0.11 dB (최소 손실, 거의 무시 가능)
- IoU: +0.027 (실루엣 3% 개선)
- Sil.Precision: +0.028 (배경 artifact 25% 감소)
- 논문 practical recommendation으로 적합

### Best Artifact Suppression: α=1.0

- 모든 artifact metric에서 최고
- PSNR_gt -0.57 dB 비용이 허용 가능한 경우 선택
- 시각화/데모 목적에 적합

### 논문 발표 전략

1. **Table**: α=0.0, 0.3, 0.5, 1.0 전체 보고 (본 보고서 §2.1)
2. **Figure**: PSNR_gt vs IoU scatter (4 points) + marginal efficiency bar chart
3. **Main result**: α=0.3 기준으로 보고, α=1.0은 supplementary에서 비교
4. **PSNR metric 명시**: "PSNR_gt (masked foreground)" vs "PSNR_int (white-BG)" 반드시 구분

---

## 5. Related

- [[CHECKPOINT_INVENTORY_260326]] — 전체 체크포인트 현황
- [[../specs/METRICS_PROTOCOL]] §7 — Comprehensive metric 정의
- [[../hypotheses/H8_opacity_anisotropy_analysis]] — Opacity 분포 분석
- Obsidian `analysis/PRUNING_ABLATION_DESIGN.md` — Pruning 실험 설계

---

*Alpha Comprehensive Eval v1.0 | 2026-03-26 | 4 checkpoints × 360 test frames × 6 views = 8,640 evaluations*
