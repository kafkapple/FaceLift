# FaceLift: Per-Stage Optimal Training Settings & Results

> Version 1.0 | 2026-02-18 | Based on Phase 3 + Analysis Phase experiments

---

## Overview

FaceLift E2E pipeline: **Image → Stage 1 (MVDiffusion) → Stage 2 (GS-LRM) → 3DGS renders**

This document covers optimal training settings for each stage, all experiments conducted, and quantitative/qualitative results.

---

## Stage 1: Multi-view Diffusion (SD2.1-UnCLIP + Era3D) (Multi-View Generation)

### 1.1 Architecture

| Component | Value |
|-----------|-------|
| Base model | SD 2.1-UnCLIP |
| Task | Single image → 6 multi-view images (60° apart) |
| Input | 1 reference image (512×512) + CLIP embedding |
| Output | 6 views (512×512 RGB) |
| Attention | Sparse multi-view attention (cross-view consistency) |
| Training | Finetune from pretrained Era3D-style pipeline |

### 1.2 Common Training Settings (All Experiments)

| Parameter | Value |
|-----------|-------|
| **Dataset** | M5t2 (Train: 2880 frames, Val: 360 frames) |
| **Resolution** | 512×512 |
| **Batch size** | 4 (per GPU) |
| **Gradient accumulation** | 4 (effective batch = 16) |
| **Learning rate** | 5e-5 |
| **Optimizer** | AdamW (β1=0.9, β2=0.999, wd=0.01) |
| **Mixed precision** | fp16 |
| **EMA** | Enabled |
| **Gradient checkpointing** | Enabled |
| **SNR gamma** | 5.0 |
| **Max grad norm** | 1.0 |
| **Warmup** | 100 steps |
| **CFG** | Enabled (condition_drop_rate=0.05) |
| **Augmentation** | brightness=[0.9,1.1], contrast=[0.9,1.1] |
| **Background** | three_choices (train), white (val) |
| **Seed** | 42 |

### 1.3 Experiment Variants

| ID | Name | Key Differences | Steps | Checkpoint |
|----|------|----------------|:-----:|-----------|
| **Baseline** | mouse_M5t2 | Sparse attn, fixed ref (v0), piecewise LR | 5K→10K | `checkpoint-5000` |
| **cfgr** | mouse_M5t2_cfgr | **Full attention** (sparse=false), fixed ref | 10K | `checkpoint-10000` |
| **E1** | mouse_M5t2_20k_cosine | **Cosine LR**, sparse attn, fixed ref | 20K | `checkpoint-20000` |
| **E2** | mouse_M5t2_randref_sparse | **Random ref view**, sparse, **resume from 5K**, piecewise (1:10K,0.2×) | 20K | `checkpoint-20000` |
| **E3** | mouse_M5t2_pose_extrinsic_add | **Pose conditioning** (extrinsic+add), **init from baseline UNet**, cosine LR, random ref | 10K | `checkpoint-10000` |
| E6 | mouse_M5t2_E6_lr75 | Higher LR (**7.5e-5**), no augmentation, piecewise | 10K | — |

#### Detailed Config Differences

| Setting | Baseline | cfgr | E1 | E2 | E3 |
|---------|:--------:|:----:|:--:|:--:|:--:|
| `lr_scheduler` | piecewise | piecewise | **cosine** | piecewise | **cosine** |
| `step_rules` | 1:100K,0.5 | 1:100K,0.5 | null | **1:10K,0.2** | null |
| `max_train_steps` | 10K | 10K | **20K** | **20K** | 10K |
| `sparse_mv_attention` | true | **false** | true | true | true |
| `reference_view_idx` | 0 (fixed) | 0 (fixed) | 0 (fixed) | **random** | **random** |
| `resume_from_checkpoint` | null | null | null | **latest (5K)** | null |
| `pretrained_unet_path` | null | null | null | null | **baseline/ckpt-5K/unet** |
| `pose_conditioning` | null | null | null | null | **extrinsic+add** |

### 1.4 Quantitative Results (E2E, 360 test frames)

> MVDiff 단독 val metric 대신, **E2E pipeline에서의 최종 결과**가 중요합니다.

| Experiment | PSNR_gt | PSNR_int | IoU | Coverage | Best at |
|-----------|:-------:|:--------:|:---:|:--------:|---------|
| Baseline 5K | 7.93 | 13.70 | 0.474 | 74.0% | — |
| cfgr 10K | 7.75 | 15.75 | 0.491 | 67.5% | — |
| E1 cosine 11K | 7.90 | 16.12 | 0.522 | 70.4% | — |
| **E1 cosine 20K** | 7.90 | **16.15** | **0.528** | 70.5% | **Color accuracy** |
| **E2 resume 20K** | **8.20** | 15.63 | 0.521 | **71.5%** | **Overall PSNR** |
| E3 pose 10K | 8.10 | 15.88 | 0.523 | 71.6% | — |

### 1.5 MVDiff Direct Quality (A1 Analysis)

MVDiff 생성 뷰를 GT와 직접 비교 (E1 cosine 20K 기준):

| Metric | MVDiff Output | Interpretation |
|--------|:---:|------|
| Silhouette IoU | 0.582 | Shape accuracy 58% (primary bottleneck) |
| Coverage | 70.8% | Misses 29% of mouse area |
| Precision | 69.9% | 30% false foreground |
| PSNR_intersection | 18.61 dB | Decent color where overlap exists |

#### Per-View Quality (E1)

| View | Angular Distance | Sil IoU | PSNR_int | Coverage |
|------|:---:|:---:|:---:|:---:|
| 0 (input/reference) | 0° | 0.975 | 25.13 | 99.9% |
| 1 (adjacent) | 60° | 0.621 | 18.90 | 78.9% |
| 2 | 120° | 0.500 | 16.56 | 64.9% |
| 3 | 180° | 0.509 | 17.09 | 65.3% |
| 4 | 240° | 0.464 | 17.02 | 60.5% |
| 5 (farthest) | 300° | 0.423 | 16.95 | 55.0% |

### 1.6 Qualitative Observations

- **Baseline → E1/E2/E3**: 수렴 범위 동일 (PSNR_gt 7.9-8.2). 학습 전략 변경으로 병목 극복 불가
- **Cosine LR (E1)**: Color accuracy 최고 (PSNR_int=16.15), 그러나 PSNR_gt는 baseline과 동일
- **Resume + Random ref (E2)**: 전체 PSNR 최고, coverage 최대. 다양한 reference view로 robustness 향상
- **Pose conditioning (E3)**: E2와 유사 성능. Extrinsic camera info는 유의미한 개선 없음
- **Full attention (cfgr)**: Sparse attention 대비 열등. Cross-view consistency가 오히려 하락
- **실루엣 오류가 지배적**: FG 면적은 GT와 유사 (2.37% vs 2.39%)하지만 **위치가 어긋남** → 낮은 IoU

### 1.7 Stage 1 Optimal Settings

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| **LR scheduler** | Cosine (best color) or Piecewise + resume (best overall) | E1 vs E2 trade-off |
| **Steps** | 20K | 11K→20K: marginal gain (+0.03 PSNR_int) |
| **Sparse attention** | Yes | cfgr (full) < sparse in all metrics |
| **Reference view** | Random (if resuming) or Fixed (v0) | E2 random + resume = best PSNR_gt |
| **Pose conditioning** | Skip | No benefit (E3 ≈ E2) |
| **Augmentation** | Yes | Standard brightness/contrast |

**Conclusion**: **현재 아키텍처 내에서는 최적 도달**. E2E PSNR_gt 7.9-8.2 수렴. 돌파를 위해서는 MVDiff 아키텍처 변경 필요 (실루엣 예측 개선, view consistency 강화).

---

## Stage 2: GS-LRM (Gaussian Splatting Large Reconstruction Model)

### 2.1 Architecture

| Component | Value |
|-----------|-------|
| Model | GS-LRM (Gaussian Splatting + Transformer) |
| Task | Multi-view images → 3D Gaussian Splatting |
| Input | N GT/generated views (default: 4) + camera poses |
| Output | 3DGS representation → rendered novel views |
| Backbone | ViT-based (d=1024, heads=64, 24 layers, patch=8) |
| Gaussians | 2 per patch, SH degree 0 |

### 2.2 Common Training Settings

| Parameter | Value |
|-----------|-------|
| **Dataset** | M5t2 (Train: 2880 frames, Val: 360 frames) |
| **Resolution** | 512×512 |
| **Batch size** | 2 (per GPU) |
| **Learning rate** | 1e-6 |
| **Optimizer** | AdamW (β1=0.9, β2=0.95, wd=0.05) |
| **Mixed precision** | bf16 |
| **Gradient checkpoint** | Every 1 layer |
| **Grad clip norm** | 50.0 |
| **Warmup** | 500 steps |
| **Max forward-backward** | 15,000 passes |
| **Early stop patience** | 10 validations (val_every=200 → ~2000 steps) |
| **Losses** | L2 (1.0) + Perceptual (0.5) |
| **Pretrained** | `ckpt_0000000000021125.pt` (pretrained on Objaverse) |
| **Seed** | 42 |

### 2.3 Experiment Variants

| ID | Name | num_input_views | Config Change | Val PSNR | Checkpoint |
|----|------|:---:|------|:---:|-----------|
| **4v baseline** | M5t2_E0_1_facelift | 4 | Standard | **22.34** | `best_psnr.pt` (step 8001) |
| 6v | base_uniform_v2_6view_v2 | 6 | num_input_views=6 | **24.49** | `best_psnr.pt` |
| 5v | base_uniform_v2_5view_v2 | 5 | num_input_views=5 | ~23.5 | `best_psnr.pt` |
| 3v | base_uniform_v2_3view_v2 | 3 | num_input_views=3 | ~19.5 | `best_psnr.pt` |
| 2v | base_uniform_v2_2view_v2 | 2 | num_input_views=2 | ~15.0 | `best_psnr.pt` |
| 1v | base_uniform_v2_1view_v2 | 1 | num_input_views=1 | **11.08** | `best_psnr.pt` |
| **E5 alpha** | base_uniform_v2_4view_alpha03_v3 | 4 | alpha_loss_weight=0.3 | **21.34** | `best_psnr.pt` (step 15840) |

### 2.4 Quantitative Results

#### Val PSNR by Input Views (H4 View Ablation)

| Input Views | Val PSNR | Delta from 4v |
|:-----------:|:--------:|:---:|
| 6 | 24.49 | +2.15 |
| 5 | ~23.5 | +1.2 |
| **4 (default)** | **22.34** | **(baseline)** |
| 3 | ~19.5 | -2.8 |
| 2 | ~15.0 | -7.3 |
| 1 | 11.08 | -11.26 |

#### Test-Set Fair Metrics (360 frames, views 1-5)

| Model | PSNR_gt | PSNR_int | IoU | Coverage |
|-------|:-------:|:--------:|:---:|:--------:|
| **GS-LRM 6v GT** | **21.02** | **22.36** | **0.943** | 98.9% |
| GS-LRM 4v GT | 20.66 | 21.29 | 0.926 | **99.3%** |
| GS-LRM 1v GT | 10.47 | 10.47 | 0.028 | 100.0% |

#### Per-View Quality (GS-LRM 4v GT)

| View | PSNR | Interpretation |
|------|:----:|------|
| 2 (near input) | ~24 dB | Close to input → well reconstructed |
| 3 (near input) | ~24 dB | Close to input → well reconstructed |
| 4 (far) | ~16.6 dB | Far from 4 inputs → interpolation only |
| 5 (far) | ~16.6 dB | Far from 4 inputs → interpolation only |

Spread: 7.9 dB (vs PS: 1.8 dB). GS-LRM 4v는 input 근접 뷰에서 매우 강하지만, far views에서 급격히 하락.

#### E5 Alpha Regularization

| Metric | 4v Baseline | E5 Alpha (0.3) | Delta |
|--------|:---:|:---:|:---:|
| Val PSNR | **22.34** | 21.34 | **-1.00** |
| Training steps | 8,001 | 15,840 | +7,839 |
| E2E PSNR_gt (predicted) | 7.9-8.2 | ~7.9-8.2 | **0** (F3) |

> **E5 결론**: Opacity regularization은 GS-LRM val 성능을 하락시킴 (-1.0 dB). Stage 2 개선은 E2E로 전이되지 않으므로 (F3: 0% transfer), E2E 관점에서 무의미.

### 2.5 E2E Transfer Analysis

| Source of Improvement | Val Improvement | E2E Improvement | Transfer Rate |
|----------------------|:---:|:---:|:---:|
| MVDiff val (Stage 1) | +3.59 dB | +0.27 dB | ~7.5% |
| **GS-LRM val (Stage 2)** | **+2.3 dB** | **0.0 dB** | **0%** |
| Cosine LR (PSNR_int) | — | +0.52 dB (color) | ~15% |

### 2.6 Sensitivity Analysis (A2: Oracle MVDiff)

GS-LRM 4v 모델에 GT/MVDiff 혼합 입력을 제공하여 뷰별 민감도 측정:

| Level | Config | PSNR_gt | IoU | Key Finding |
|:-----:|--------|:-------:|:---:|-------------|
| 0 | 6 GT (upper bound) | 21.02 | 0.943 | Upper bound |
| 1 | Replace v5 | **21.02** | **0.943** | **v5 미사용 (영향 0)** |
| 2 | Replace v4,5 | **21.02** | **0.943** | **v4도 미사용 (영향 0)** |
| 3 | Replace v3,4,5 | 15.82 | 0.780 | **v3 교체 시 -5.2 dB 급락** |
| 4 | Replace v2,3,4,5 | 10.44 | 0.614 | v2 추가 교체 -5.4 dB |
| 5 | Replace v1-5 (≈ E2E) | 7.90 | 0.528 | v1 교체 -2.5 dB |

**Key Insight**: `num_input_views=4` 설정에서 실제 선택되는 뷰는 **0, 1, 2, 3번** (처음 4개). Views 4, 5는 완전히 무시됨. MVDiff가 6개 뷰를 생성하지만 2개는 사용되지 않음.

### 2.7 Qualitative Observations

- **View 수 증가의 효과**: 4v→6v에서 +2.15 dB val. 추가 뷰가 원형 커버리지 개선
- **Far view 약점**: 4v 모델의 far views (~16.6 dB)는 PS 수준과 유사. GS-LRM의 강점은 near views에 집중
- **Alpha regularization 실패**: Opacity entropy loss가 오히려 3DGS 표현력을 제한. Floater 감소 효과 < 품질 하락
- **1v의 실패**: IoU=0.028, 사실상 의미 있는 3D 구조 복원 불가. 최소 3-4뷰 필요
- **Stage 2 개선은 E2E에 전이 불가**: MVDiff 출력 품질이 bottleneck이므로, GS-LRM을 아무리 개선해도 E2E 결과 불변

### 2.8 Stage 2 Optimal Settings

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| **num_input_views** | 4 (현재) or 6 (향후) | 6v=+2.15 dB val, 그러나 E2E에서는 views 4,5 미사용 |
| **Losses** | L2 + Perceptual (standard) | Alpha regularization은 성능 하락 |
| **opacity_reg_weight** | 0.0 (disabled) | E5: -1.0 dB val (harmful) |
| **Learning rate** | 1e-6 | Default, no improvement from changes |
| **Early stopping** | patience=10, val_every=200 | ~8K steps에서 수렴 |
| **Pretrained** | Objaverse ckpt (`ckpt_0000000000021125.pt`) | 필수 (from-scratch 불가) |

**Conclusion**: **4v baseline이 최적**. Stage 2 개선은 E2E에 0% 전이되므로, 추가 실험 불필요. 향후 MVDiff 아키텍처 변경 시 num_input_views=6 전환 검토.

---

## Combined E2E Pipeline: Optimal Configuration

### Recommended Configuration

| Component | Setting | Checkpoint |
|-----------|---------|-----------|
| **MVDiff** | E2 (resume + random ref + sparse + piecewise) | `mouse_M5t2_randref_sparse/checkpoint-20000` |
| **GS-LRM** | 4v baseline (standard losses) | `M5t2_E0_1_facelift/best_psnr.pt` |

**Alternative** (color accuracy 중시):
| Component | Setting | Checkpoint |
|-----------|---------|-----------|
| **MVDiff** | E1 (cosine LR + sparse + fixed ref) | `mouse_M5t2_20k_cosine/checkpoint-20000` |
| **GS-LRM** | 4v baseline | `M5t2_E0_1_facelift/best_psnr.pt` |

### E2E Inference Command

```bash
cd /home/joon/dev/FaceLift
export CUDA_VISIBLE_DEVICES=7

# MVDiff + GS-LRM combined inference
python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt \
    --mvdiffusion_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_randref_sparse/checkpoint-20000 \
    --gslrm_checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/e2e_optimal \
    --input_view_idx 0 \
    --no_turntable --no_mesh --no_metrics \
    --prefer_ema --skip_preprocess
```

### Best E2E Results

| Metric | E2 (Best Overall) | E1 (Best Color) | Baseline (5K) |
|--------|:---:|:---:|:---:|
| PSNR_gt_masked | **8.20** | 7.90 | 7.93 |
| PSNR_intersection | 15.63 | **16.15** | 13.70 |
| IoU | 0.521 | **0.528** | 0.474 |
| Coverage | **71.5%** | 70.5% | 74.0% |

---

## Bottleneck Summary & Improvement Roadmap

### Current Pipeline Bottleneck

```
GS-LRM 6v GT:  PSNR_gt=21.02, IoU=0.943  ← Upper bound
       ↓
MVDiff:         Sil IoU=0.582              ← 86% of quality loss HERE
       ↓
E2E Best:       PSNR_gt=8.20, IoU=0.521   ← Final output
```

**Stage 2 (GS-LRM)은 bottleneck이 아님.** GT 입력 시 이미 PS를 +4.22 dB 초과.

### Improvement Priorities

| Priority | Target | Action | Expected Impact |
|:--------:|--------|--------|:---:|
| **P1** | MVDiff silhouette | 실루엣 예측 정확도 개선 | High |
| **P2** | MVDiff views 1-3 | 인접 뷰 일관성 강화 (A2: 이 뷰들이 critical) | High |
| **P3** | GS-LRM num_input_views | 4→6으로 증가 (현재 v4,5 미사용) | Medium |
| Low | Training strategy | 더 긴 학습, 다른 LR → 이미 수렴 | Negligible |
| Low | GS-LRM regularization | Alpha/opacity loss → 성능 하락 (E5) | Negative |

---

## File Locations

### Checkpoints

| Model | Path | Notes |
|-------|------|-------|
| MVDiff Baseline 5K | `/node_data/.../mvdiffusion/mouse_M5t2/checkpoint-5000/` | sparse, fixed ref |
| MVDiff E1 Cosine 20K | `/node_data/.../mvdiffusion/mouse_M5t2_20k_cosine/checkpoint-20000/` | cosine LR |
| MVDiff E2 Resume 20K | `/node_data/.../mvdiffusion/mouse_M5t2_randref_sparse/checkpoint-20000/` | random ref, resume |
| MVDiff E3 Pose 10K | `/node_data/.../mvdiffusion/mouse_M5t2_pose_extrinsic_add/checkpoint-10000/` | extrinsic+add |
| GS-LRM 4v Baseline | `/node_data/.../gslrm/M5t2_E0_1_facelift/best_psnr.pt` | **Optimal** |
| GS-LRM 6v | `/node_data/.../gslrm/base_uniform_v2_6view_v2/best_psnr.pt` | Upper bound |
| GS-LRM E5 Alpha | `/node_data/.../gslrm/base_uniform_v2_4view_alpha03_v3/best_psnr.pt` | Regularization (bad) |

### Config Files

| Config | Path |
|--------|------|
| GS-LRM base | `configs/base/gslrm_mouse.yaml` |
| GS-LRM 4v | `configs/mouse/uniform/4view_v2.yaml` |
| GS-LRM 6v | `configs/mouse/uniform/6view_v2.yaml` |
| GS-LRM E5 | `configs/mouse/uniform/4view_alpha03_v3.yaml` |
| MVDiff configs | `/node_data/.../mvdiffusion/{experiment}/config.yaml` |

### Results & Analysis

| File | Purpose |
|------|---------|
| `experiments/comparison/fair/facelift_fair.json` | E2E fair metrics |
| `experiments/comparison/tier/gslrm_*_fair.json` | GS-LRM standalone fair metrics |
| `experiments/analysis/mvdiff_quality/` | A1 MVDiff diagnostic |
| `outputs/analysis/oracle_mvdiff/` | A2 sensitivity analysis |
| `experiments/comparison/FL_vs_PS/FL_vs_PS_comparison_v7.md` | Full comparison document |

---

*FaceLift Training Optimal Settings v1.0 | 2026-02-18*
