# 4D-GS Deformation Experiment Plan

Created: 2026-03-29 | Based on FaceLift paper arXiv:2412.17812 Appendix 3.5

---

## 1. Objective

FaceLift 논문 deformable module 기반으로 temporal consistency 확보.
- Canonical Gaussians + 8-layer MLP deformation → 프레임 간 Gaussian identity 유지
- 3D scene flow 암묵적 학습 (rendering supervision)

## 2. Paper Architecture (Target)

```
Frame t (canonical)     Frame t+1 (target pseudo-GT)
    |                        |
    v                        v (독립 GS-LRM inference)
G_t [1,572,866 x 14]        G_{t+1} [1,572,866 x 14]
    |
    v 8-layer MLP
D(G_t) = [dx,dy,dz,dα,ds]
    |
    v Apply deformation
G'_{t+1} = G_t + D(G_t)
    |
    v Rendering (6-view)
rendered images
    |
    v Loss vs rendered(G_{t+1})
L = MSE + Perceptual
```

## 3. Current Implementation Status

| Component | Status | File |
|-----------|:------:|------|
| DeformationNetwork V2 (8-layer MLP) | ✅ | `model/deformation/deformation_network.py` |
| DeformationTrainer V2 | ✅ | `model/deformation/deformation_trainer_v2.py` |
| ARAP + Velocity losses | ✅ | `model/deformation/temporal_regularization.py` |
| GS-LRM integration | ✅ | `model/deformation/gslrm_integration.py` |
| Pre-computed Gaussian cache | ✅ (code) | `model/deformation/gslrm_integration.py:GaussianCache` |
| **Trained checkpoint** | **❌** | 미실행 |

## 4. Experiment Plan

### Phase 0: Gaussian 캐시 생성 (1회성)

```bash
ssh gpu03
cd /home/joon/dev/FaceLift
conda activate facelift

# GS-LRM으로 전체 프레임 Gaussian 추론 + 캐시
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.train_deformation \
    --config configs/deformation/default.yaml \
    --precompute_cache

# 예상 소요: ~2시간 (3600 frames × ~2s/frame)
# 출력: /node_data/joon/checkpoints/FaceLift/deformation/default/gaussian_cache/frame_*.pt
```

**Config 확인 필요 (configs/deformation/default.yaml)**:
- `gslrm.checkpoint`: `/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_6view_alpha03_v3/best_psnr.pt`
- `data.data_dir`: `/home/joon/data/preprocessed/FaceLift_mouse/M5`
- `data.frame_range`: `0:3600` (전체) 또는 `1500:1860` (cinematic range)

### Phase 1: Paper 설정 기준 학습 (V2)

```bash
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.train_deform_v2 \
    --config configs/mouse/deform_v2.yaml

# Config (deform_v2.yaml):
#   num_layers: 8
#   hidden_dim: 256
#   use_positional_encoding: true
#   use_time_embedding: true
#   loss: param(1.0) + ARAP(0.1) + velocity(0.01)
#   epochs: 200
#   lr: 1e-4
#   batch_size: 10000 Gaussians
```

### Phase 2: Rendering Supervision 추가 (논문 일치)

현재 V2는 Gaussian parameter loss (MSE). 논문은 **rendering loss** 사용.
→ 렌더링 loss 추가 구현 필요:

```python
# Pseudo-code for rendering supervision
deformed_gaussians = canonical + mlp(canonical.xyz)
rendered = render_6views(deformed_gaussians, cameras)
target_rendered = render_6views(target_gaussians, cameras)
loss = MSE(rendered, target_rendered) + LPIPS(rendered, target_rendered)
```

### Phase 3: Temporal Evaluation

```bash
# Deformed 프레임 생성
python -m mouse_extensions.scripts.inference.run_temporal_deform \
    --checkpoint /path/to/deform_v2/best.pt \
    --frame_range 3300:3320 \
    --output_dir outputs/datasets/temporal_eval/deform_v2

# EMA와 비교
python -m mouse_extensions.scripts.eval.temporal_comparison \
    --data_root outputs/datasets/temporal_eval \
    --output_dir outputs/viz/comparison/mouse/deform_vs_ema
```

## 5. Key Design Decisions

### Gaussian Identity 문제

- GS-LRM per-pixel → 프레임마다 동일 수 (1,572,866) 보장
- Pixel-position identity (index N = 같은 pixel 위치) ≠ semantic identity
- 0.05s 간격에서는 spatial ≈ semantic (작은 모션)
- **빠른 동작 구간**: explicit correspondence 필요 → future work

### Frame Range 선택

| Range | Frames | 용도 |
|-------|:------:|------|
| `1500:1860` | 360 | Cinematic range (활발한 행동) |
| `3300:3320` | 21 | Dense temporal eval (표준) |
| `0:3600` | 3600 | 전체 학습 (최대 데이터) |

**권장**: Phase 1은 `1500:1860` (360 frames, 빠른 실험). 성공 시 전체로 확대.

### Visibility Filtering

- Deformation은 **필터링 전** 전체 Gaussians에 적용
- 렌더링 시: vis_mask 적용 (foreground만)
- 배경 Gaussians도 deformation 학습에 포함 (consistency 유지)

## 6. Success Criteria

| Metric | Baseline (Original) | Target (Deformed) | Best (EMA α=0.1) |
|--------|:-------------------:|:-----------------:|:-----------------:|
| tOF | 0.2052 | < 0.10 | 0.0230 |
| PSNR_gt | ~20 dB | >= 20 dB | ~19.5 dB (blur) |
| Visual | Flickering | Smooth + sharp | Smooth but blurry |

**핵심 목표**: EMA의 blur 없이 temporal consistency 확보.

## 7. Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Pixel-identity != semantic | Deformation 실패 | 작은 motion 구간 우선 |
| Rendering loss 비용 | 학습 느림 | Parameter loss로 warmup → rendering loss fine-tune |
| Overfitting (360 frames) | 일반화 실패 | Val split 사용, augmentation |

---

> Related: [[260329_GS-LRM_Architecture_and_Temporal_Analysis]] | [[TEMPORAL_CONSISTENCY_STUDY]]
