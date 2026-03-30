# Deformation V3: FG-Aware Rendering Loss

> **Navigation**: [← INDEX](../INDEX.md) | [DEFORMATION_4DGS_EXPERIMENT_PLAN](DEFORMATION_4DGS_EXPERIMENT_PLAN.md)
> **Purpose**: V2(param MSE, all Gaussians) → V3(rendering loss, FG-aware) 전환 계획
> **Created**: 2026-03-31
> **Method**: 4× `/deliberate --moa --audit` (Deform 전략 + RAFT + FG-only + 중단 판정)
> **Status**: V2 중단 (epoch 26), 237GB cache 삭제 완료

---

## 1. Why V3 (V2 실패 분석)

### V2의 구조적 문제 (3/3 audit consensus)

| 문제 | Severity | 설명 |
|------|:--------:|------|
| **Param MSE ≠ rendering quality** | 🔴 Critical | Gaussian 파라미터 거리가 visual 품질과 직접 상관 없음 |
| **97.5% BG 학습 낭비** | 🔴 Critical | 배경 Gaussians deformation=0 → MLP 용량 낭비, 237GB 캐시 |
| **Pixel-identity approximate** | 🔴 Critical | Correspondence 미확보 상태로 pair 학습 |
| **ARAP topology 미정의** | 🟡 Major | 1.57M unstructured points에 k-NN 방식 불명 |

### V2 → V3 핵심 변경

| | V2 (중단) | V3 (계획) |
|---|---|---|
| **Loss** | param MSE (PRIMARY) | **Rendering loss** (PRIMARY) |
| **Data** | ALL 1.57M Gaussians | **FG ~39K + BG 5% sparse** |
| **Cache** | 237GB (85MB/frame) | **~8GB (~3MB/frame)** |
| **Correspondence** | Pixel-identity only | **RAFT + CoTracker hybrid** (Phase 2) |
| **ARAP** | Undefined topology | **k-NN k=8 on FG (world-space)** |
| **Training view** | N/A | **Lock loss (preservation)** |

---

## 2. FG-Aware Gaussian Cache

### 2.1 FG Identification Strategy

**Soft filtering** (hard filtering 금지 — audit 3/3 합의):

```python
# Alpha mask → FG identification
# GS-LRM RGBA images: alpha channel = FG mask
# Pixel-aligned: alpha[u,v] > threshold → Gaussian at (u,v) is FG

# 3-tier classification:
#   FG core:    alpha > 0.8     → weight = 1.0 (full training)
#   FG boundary: 0.2 < alpha ≤ 0.8 → weight = alpha (soft contribution)
#   BG:          alpha ≤ 0.2    → 5% random sample, weight = 0.1
```

**Why soft, not hard**:
- Hard filtering 시 경계 Gaussians(모피, 수염) 손실
- BG sparse sample은 "δ=0" negative signal 제공 → 일반화 유지
- Alpha-weighted loss로 경계 자연스럽게 처리

### 2.2 Cache Format

```python
# Per-frame cache (V3 format):
{
    "fg_xyz": tensor[N_fg, 3],        # FG core + boundary positions
    "fg_features": tensor[N_fg, 1, 3], # SH features
    "fg_scaling": tensor[N_fg, 3],
    "fg_rotation": tensor[N_fg, 4],
    "fg_opacity": tensor[N_fg, 1],
    "fg_weights": tensor[N_fg],        # alpha-based weights
    "fg_indices": tensor[N_fg],        # original pixel indices (correspondence용)
    "bg_sample_xyz": tensor[N_bg, 3],  # 5% BG sample
    "bg_sample_indices": tensor[N_bg],
}
# Estimated: ~3MB/frame (float16) × 2880 = ~8GB (vs 237GB)
```

### 2.3 Cache Generation Script

```bash
# Step 1: GS-LRM inference → full Gaussians + alpha masks
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.cache_fg_gaussians \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_6view_alpha03_v3/best_psnr.pt \
    --data_list /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_train.txt \
    --output_dir /node_data/joon/checkpoints/FaceLift/deformation/v3/fg_cache \
    --fg_threshold 0.2 \
    --bg_sample_ratio 0.05 \
    --dtype float16
# Expected: ~2h, output ~8GB
```

---

## 3. V3 Loss Design

### 3.1 Loss Components

```
L_total = 1.0·L_render + 0.1·L_arap + 0.05·L_vel

Phase 2 additions (after RAFT integration):
  + 0.1·L_corr (correspondence)
  + 0.1·L_tOF (temporal consistency)
```

### 3.2 L_render (PRIMARY — 최대 변경)

```python
def rendering_loss(gs_deformed, cameras, gt_images):
    """
    Differentiable rendering → image space loss.
    Stochastic: 2-3 random views per step (memory 절약).
    """
    selected_cams = random.sample(cameras, k=min(3, len(cameras)))
    loss = 0
    for cam in selected_cams:
        rendered = differentiable_render(gs_deformed, cam)
        loss += 0.8 * L1(rendered, gt_images[cam])
        loss += 0.2 * (1 - SSIM(rendered, gt_images[cam]))
    return loss / len(selected_cams)
```

**Why rendering loss > param MSE**:
- PSNR은 rendered image에서 측정 → loss도 rendering space에서 최적화해야
- 작은 param 변화가 큰 visual 변화 가능 (역도 가능) → MSE는 proxy로 부적합
- Differentiable GS rendering은 이미 코드베이스에 존재 (gsplat)

### 3.3 L_arap (FG-aware)

```python
def fg_arap_loss(g_t, g_deformed, fg_indices, k=8):
    """
    k-NN graph on FG Gaussians (world-space XYZ).
    BG로 graph 확장하되 loss는 FG만 계산.
    """
    # Build k-NN on FG positions
    knn_graph = build_knn(g_t.xyz[fg_indices], k=k)

    # Edge length preservation
    for (i, j) in knn_graph.edges:
        e_before = g_t.xyz[i] - g_t.xyz[j]
        e_after = g_deformed.xyz[i] - g_deformed.xyz[j]
        loss += (e_after.norm() - e_before.norm()) ** 2

    return loss / len(knn_graph.edges)
```

### 3.4 Alpha-Weighted Training

```python
def weighted_loss(loss_per_gaussian, weights):
    """FG core=1.0, boundary=alpha, BG=0.1"""
    return (loss_per_gaussian * weights).sum() / weights.sum()
```

---

## 4. Architecture Changes

### 4.1 Network (Minimal Change)

V2의 8-layer MLP 유지, input/output 변경만:

```
V2: [xyz(3) + PE(60) + time(32) + features(15)] → [dx,dy,dz,dα,ds] (5)
V3: same input → [dx,dy,dz, dq0,dq1,dq2,dq3, ds0,ds1,ds2, dα] (11)
                   position    rotation(quat)    scale        opacity
```

**Why rotation delta 추가**: Audit에서 rotation 누락 지적 (o3-mini). Mouse 사지 회전이 큼.

### 4.2 Variable Batch Size

FG Gaussian 수가 프레임마다 다름 (~30K-50K):

```python
# Dynamic batching: pad to max in batch, mask out padding
def collate_fg_pairs(batch):
    max_fg = max(b.fg_xyz.shape[0] for b in batch)
    padded = [pad_to(b, max_fg) for b in batch]
    masks = [create_mask(b.fg_xyz.shape[0], max_fg) for b in batch]
    return padded, masks
```

---

## 5. Implementation Priority

| Step | 작업 | 시간 | 의존성 |
|:----:|------|:----:|:------:|
| **S0** | `cache_fg_gaussians.py` 스크립트 작성 | 2h | — |
| **S1** | FG cache 생성 (2880 frames) | 2h GPU | S0 |
| **S2** | `train_deform_v3.py` — rendering loss + FG-aware | 4h | S0 |
| **S3** | 10-frame smoke test (수렴 확인) | 1h GPU | S1+S2 |
| **S4** | Full training (2880 frames, 100 epochs) | ~4h GPU | S3 pass |
| **S5** | tOF + PSNR 측정 (temporal eval) | 1h | S4 |

**Total: ~2일** (구현 1일 + 학습 1일)

### Go/No-Go Gates

| Gate | 기준 | 실패 시 |
|------|------|---------|
| S3 smoke test | Loss 수렴 (10 frames, 20 epochs) | Loss 설계 재검토 |
| S5 tOF | tOF < V2 baseline (or < 0.15) | Correspondence 추가 (RAFT) |
| S5 PSNR | PSNR_gt ≥ 20 dB 유지 | Rendering loss weight 조정 |

---

## 6. Compute Savings

| | V2 | V3 | 절감 |
|---|---:|---:|:----:|
| Cache | 237GB | ~8GB | **97%** |
| Per-epoch time | ~24min | ~2min (est.) | **92%** |
| Total training (100ep) | ~40h | ~3-4h | **90%** |
| GPU memory | ~17GB | ~4GB (est.) | **76%** |

---

## 7. Migration from V2

### 삭제 완료 (2026-03-31)

- ✅ V2 training stopped (epoch 26/100)
- ✅ V2 checkpoints deleted (15MB — flawed loss, no reuse value)
- ✅ V1 checkpoints deleted (46MB)
- ✅ 237GB gaussian_cache deleted
- ✅ temporal_inference_v2 deleted

### 보존

- ✅ `train_deform_v2.py` (코드 참조용, git history에 보존)
- ✅ `configs/mouse/deform_v2.yaml` (V3 config 기반)
- ✅ `mouse_extensions/model/deformation.py` (MLP architecture 재사용)

---

## 8. V3 → V3.5 (RAFT Phase)

V3 baseline 확보 후 RAFT correspondence 추가:

```
V3.0: FG-aware + rendering loss (이 문서)
V3.5: + RAFT/CoTracker correspondence + L_corr + L_tOF
V4.0: + Hierarchical Transformer (장기)
```

상세: Obsidian `theory/DEFORMATION_STRATEGY.md` §4-7

---

*Created: 2026-03-31 | V2 중단 + 237GB 삭제 후 V3 전환*
*Cross-ref: DEFORMATION_STRATEGY.md (Obsidian theory), DEFORMATION_4DGS_EXPERIMENT_PLAN.md*
