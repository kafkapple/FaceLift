# FaceLift Code Walkthrough

전체 코드 흐름을 한 문서로 정리합니다. 각 섹션은 `파일:라인번호`로 소스를 참조합니다.

**관련 문서**: [[GS-LRM_ARCHITECTURE_GUIDE]], [[GS-LRM_Loss_Formula]], [[METRICS_GUIDE]], [[TRAINING_PIPELINE_GUIDE]]

---

## 1. Directory Structure

```
FaceLift/
├── train_gslrm.py                          # Entry point (1502 lines)
├── inference_mouse.py                      # Inference entry
├── gslrm/
│   └── model/
│       ├── gslrm.py                        # GSLRM model (1910 lines)
│       ├── gaussians_renderer.py           # Gaussian splatting (1649 lines)
│       └── utils_losses.py                 # VGG19, PerceptualLoss, SSIM
├── mouse_extensions/                       # Mouse-specific extensions
│   ├── data/mouse_dataset.py              # MouseViewDataset (664 lines)
│   ├── model/
│   │   ├── loss_extensions.py             # Mask/alpha loss (823 lines)
│   │   ├── gaussian_pruning.py            # Ghost gaussian regularizer
│   │   └── visualization.py               # Threshold comparison
│   ├── preprocessing/
│   │   ├── presets.py                     # VERSION_HIERARCHY, preset configs
│   │   ├── preprocess.py                  # Unified preprocessor
│   │   ├── camera_normalizer.py           # fx→549, trans→2.7
│   │   └── center_estimation.py           # 3D triangulation
│   └── scripts/                           # Analysis, inference, diagnostics
└── configs/
    ├── base/gslrm_mouse.yaml              # Base config
    ├── datasets/                           # M1~M5, D7_1, D8, etc.
    └── experiments/                        # E0~E2, debug, overfit
```

**핵심 파일 3개**: `train_gslrm.py` (학습), `gslrm/model/gslrm.py` (모델), `gaussians_renderer.py` (렌더링)

---

## 2. Entry Point: `train_gslrm.py`

### 2.1 CLI → Config → DDP

```
main() (L1432)
  ├── parse_arguments() (L1355)
  │     --config (Legacy) 또는 -d/-e (Modular)
  ├── load_modular_config() (L1287) 또는 load_and_process_config() (L1389)
  │     OmegaConf: base + dataset + experiment merge
  └── GSLRMTrainer(config, args).__init__() (L82)
        ├── setup_distributed() (L104)     # DDP, rank, world_size
        ├── setup_cuda() (L132)            # device, mixed precision
        ├── load_datasets() (L189)         # MouseViewDataset or RandomViewDataset
        ├── setup_model() (L310)           # GSLRM + DDP wrap
        ├── setup_optimization() (L395)    # AdamW + lr scheduler
        ├── load_checkpoint() (L444)       # Pretrained weights
        └── setup_wandb() (L485)           # Logging
```

### 2.2 Training Loop

```
trainer.train() (L1220)
  for epoch in range(max_epochs):
    for batch in dataloader:
      ├── train_step(batch) (L667)
      │     model.set_training_step(...)
      │     result = model(batch, create_visual=...)
      │     # result contains: loss, metrics, visual
      ├── optimizer_step(result) (L763)
      │     loss.backward() → clip_grad_norm → optimizer.step()
      ├── log_training_metrics(...) (L827)
      ├── save_checkpoint_if_needed() (L888)
      ├── save_visuals_if_needed(...) (L908)
      └── run_validation() (L1069)   # every val_every steps
```

---

## 3. Config System

### 3.1 Modular Mode (권장)

```bash
torchrun train_gslrm.py -d M3_1 -e E1_1_base
```

`load_modular_config()` (L1287):
```
configs/base/gslrm_mouse.yaml    # 공통 설정 (pretrained checkpoint 포함)
  + configs/datasets/M3_1.yaml   # 데이터 경로, 뷰 수, 전처리 preset
  + configs/experiments/E1_1_base.yaml  # 학습 하이퍼파라미터
  → OmegaConf.merge() → edict
```

### 3.2 Legacy Mode

```bash
torchrun train_gslrm.py --config configs/mouse/complete_config.yaml
```

단일 YAML에 모든 설정 포함. 구조 실수 위험 높음.

### 3.3 주요 Config 키

| 키 | 설명 | 예시 |
|---|---|---|
| `model.resolution` | 렌더링 해상도 | `384` or `null` (input 따름) |
| `training.lr` | 학습률 | `4e-5` |
| `training.num_views` | 입력 뷰 수 | `5` |
| `losses.mask_mode` | 마스크 타입 | `"gt"`, `"alpha"`, `"none"` |
| `losses.alpha_loss_weight` | Alpha supervision 가중치 | `0.1` |
| `data.dataset_path` | 전처리 데이터 경로 | `/home/joon/data/preprocessed/...` |

---

## 4. Data Pipeline

### 4.1 MouseViewDataset

`mouse_extensions/data/mouse_dataset.py`

```
__init__(config, split) (L77)
  ├── Load uid list from data_train.txt / data_val.txt
  ├── Parse preprocessing_info.json (if exists)
  └── Setup augmentation transforms

__getitem__(idx) (L290)
  ├── Load uid directory (images/, cameras.json)
  ├── _select_views(total_views) (L253)
  │     Random sample → sorted indices → split input/target
  ├── For each view:
  │     ├── Load image (PIL) → _process_image_channels() (L226)
  │     ├── Load camera: c2w (4×4), fxfycxcy (4,)
  │     └── Load mask (if available)
  └── Return dict:
        image: (V, 3, H, W)       # V = num_views
        c2w: (V, 4, 4)            # camera-to-world
        fxfycxcy: (V, 4)          # fx, fy, cx, cy
        mask: (V, 1, H, W)        # optional
```

### 4.2 Plücker Ray Encoding

`gslrm/model/gslrm.py` → `_create_posed_images_with_plucker()` (L905)

각 픽셀에 대해 6D Plücker 좌표 생성:
```
ray_origin (o) = camera position from c2w
ray_direction (d) = pixel → 3D direction via intrinsics
Plücker = (d, o × d)   # 6 channels

Output: (B, V_in, 9, H, W)  = 3(RGB) + 6(Plücker)
```

뷰 순서와 무관한 per-pixel encoding. `view_type_embeddings` (L950)로 reference 뷰 구분.

---

## 5. Model: GSLRM

`gslrm/model/gslrm.py` — `class GSLRM(nn.Module)` (L669)

### 5.1 Architecture

```
__init__(config) (L684)
  ├── _init_data_processors (L701)      # Image normalization
  ├── _init_tokenizer (L707)            # ViT patch embedding (14×14 patches)
  ├── _init_positional_embeddings (L727) # Learnable pos embeds + view_type
  ├── _init_transformer (L746)          # Transformer blocks (24 layers)
  ├── _init_gaussian_modules (L759)     # MLP head → 14D gaussian params
  │     xyz(3) + features(3) + scaling(3) + rotation(4) + opacity(1)
  ├── _init_rendering_modules (L784)    # Renderer, LossComputer
  └── _init_training_state (L789)       # Step tracking, warmup
```

### 5.2 Forward Pass

```
forward(batch_data, create_visual, split_data) (L1146)
  │
  ├─ [1] Split input/target views
  │     input_data: V_in views (e.g., 4)
  │     target_data: V_out views (e.g., 1)
  │
  ├─ [2] Plücker encoding
  │     _create_posed_images_with_plucker() → (B, V_in, 9, H, W)
  │
  ├─ [3] Tokenize
  │     ViT patchify: (B, V_in, 9, H, W) → (B, N_tokens, D)
  │     + positional embeddings + view_type embeddings
  │
  ├─ [4] Transformer
  │     _process_through_transformer() (L971)
  │     24 layers, gradient checkpointing optional
  │
  ├─ [5] Gaussian prediction
  │     MLP head: (B, N_tokens, D) → (B, N_gaussians, 14)
  │     _create_gaussian_models_and_stats() (L1081)
  │     → GaussianModel per sample
  │
  ├─ [6] Render target views
  │     Renderer.forward() → rendered images + alpha
  │
  └─ [7] Compute losses
        LossComputer.forward() (L325)
        → L2 + perceptual + SSIM + mask + alpha + background
```

### 5.3 Key Submodules

| Class | File:Line | Role |
|---|---|---|
| `Renderer` | `gslrm.py:100` | Gaussian rendering wrapper |
| `GaussiansUpsampler` | `gslrm.py:194` | Optional gaussian upsampling |
| `LossComputer` | `gslrm.py:279` | All loss computation |

---

## 6. Rendering

`gslrm/model/gaussians_renderer.py`

### 6.1 Camera

`class Camera(nn.Module)` (L505)
```python
Camera(C2W, fxfycxcy, h, w)
  # C2W: 4×4 camera-to-world matrix
  # fxfycxcy: (fx, fy, cx, cy) intrinsics
  # Computes: world_view_transform, projection_matrix, full_proj_transform
  getProjectionMatrix(W, H, fx, fy, cx, cy, znear, zfar)  # (L524)
```

### 6.2 GaussianModel

`class GaussianModel` (L548)
```
Stores per-gaussian attributes:
  _xyz:        (N, 3)     # position
  _features:   (N, C)     # color (SH coefficients)
  _scaling:    (N, 3)     # log-scale
  _rotation:   (N, 4)     # quaternion
  _opacity:    (N, 1)     # logit-opacity

Key methods:
  set_data()              (L580) — set all attributes
  filter(valid_mask)      (L609) — mask-based filtering
  prune(opacity_thres)    (L647) — remove low-opacity
  prune_by_scaling(thres) (L653) — remove too-large
  apply_all_filters()     (L688) — chain all filters
```

### 6.3 Deferred Rendering

`class DeferredGaussianRender(torch.autograd.Function)` (L1086)

Custom autograd function for differentiable gaussian splatting:
```
forward(ctx, gaussians, camera, bg_color, ...) (L1088)
  → rasterize gaussians → rendered_image, rendered_alpha

backward(ctx, grad_renders, grad_alphas) (L1156)
  → compute gradients for gaussian parameters
```

### 6.4 Turntable & Visualization

| Function | Line | Description |
|---|---|---|
| `render_turntable()` | L1211 | 360° orbit rendering |
| `render_dataset_trajectory()` | L1301 | Dataset camera path animation |
| `render_dataset_views()` | L1381 | Static per-view rendering |
| `get_turntable_cameras()` | L228 | Generate orbit camera poses |
| `get_dataset_camera_trajectory()` | L61 | Dataset camera interpolation (hold_frames) |

---

## 7. Loss & Metrics

### 7.1 Core Losses (`gslrm/model/utils_losses.py`)

| Class | Line | Description |
|---|---|---|
| `VGG19` | L105 | Feature extractor (layers 3,8,17,26,35) |
| `PerceptualLoss` | L221 | L1 on VGG features |
| `SsimLoss` | L345 | Structural similarity |

### 7.2 LossComputer (`gslrm/model/gslrm.py:279`)

```
forward(rendering, target, input, ...) (L325)
  ├── _compute_l2_loss()          (L466) — pixel-wise MSE
  ├── _compute_perceptual_loss()  (L498) — VGG feature loss
  ├── _compute_ssim_loss()        (L517) — 1 - SSIM
  ├── _compute_lpips_loss()       (L489) — optional LPIPS
  ├── _compute_pixelalign_loss()  (L553) — 3D alignment
  └── _compute_total_loss()       (L589) — weighted sum
```

### 7.3 Mouse Loss Extensions (`mouse_extensions/model/loss_extensions.py`)

| Component | Line | Description |
|---|---|---|
| `MaskConfig` | L104 | Mask mode configuration |
| `compute_mask_from_config()` | L113 | GT/alpha/none mask selection |
| `LossExtensions` | L362 | Extends LossComputer with mask support |
| `compute_alpha_loss()` | L430 | Alpha supervision (MSE/BCE/focal) |
| `AlphaLossComputer` | L540 | Stateful alpha loss with warmup |
| `compute_opacity_regularization()` | L619 | Entropy/L1/L2 opacity reg |
| `OpacityRegularizer` | L666 | Stateful opacity reg |
| `compute_ghost_metrics()` | L318 | Ghost gaussian detection |

### 7.4 Loss Formula Summary

```
L_total = w_l2 · L_l2
        + w_perceptual · L_perceptual
        + w_ssim · L_ssim
        + w_background · L_background
        + w_alpha · L_alpha
        + w_opacity · L_opacity_reg
```

→ 상세: [[GS-LRM_Loss_Formula]]

---

## 8. Mouse Extensions

`mouse_extensions/` — 원본 GS-LRM을 mouse 데이터에 적응시키는 확장 모듈.

### 8.1 Preprocessing

| File | Key Function | Description |
|---|---|---|
| `presets.py` | `get_preset()` (L742) | VERSION_HIERARCHY: D7→M1, D8→M2, D10.3→M3 등 |
| `preprocess.py` | main preprocessor | Affine/homography transform + zoom |
| `camera_normalizer.py` | `normalize_cameras()` (L30) | fx→549, translation→2.7 정규화 |
| `center_estimation.py` | `CenterEstimator` | 3D DLT triangulation (ray error 2.7mm) |
| `split_manager.py` | — | Train/val/test temporal split |

**전처리 흐름**:
```
Raw video/images → Mask extraction → Center estimation (3D triangulation)
→ Crop & transform (preset-dependent) → Camera normalization → Output
```

### 8.2 Model Extensions

| File | Key Class | Description |
|---|---|---|
| `loss_extensions.py` | `LossExtensions` (L362) | Mask-aware loss, alpha loss, opacity reg |
| `gaussian_pruning.py` | `GhostGaussianRegularizer` (L107) | Ghost gaussian penalty |
| `mask_losses.py` | — | Normalized masked L1 (pose-splatter 방식) |
| `gslrm_patches.py` | — | Runtime model patches |
| `visualization.py` | — | Threshold comparison plots |

### 8.3 Data

| File | Key Class | Description |
|---|---|---|
| `mouse_dataset.py` | `MouseViewDataset` (L62) | Multi-view mouse dataset |
| `mouse_dataset.py` | `MouseSingleViewDataset` (L531) | Single-view inference |
| `preprocessing.py` | — | Runtime data transforms |

### 8.4 Visualization & Inference

```
mouse_extensions/visualization/
├── turntable_config.py        # Turntable rendering parameters
├── video_generator.py         # MP4 output
├── alpha_visualization.py     # Alpha channel debug
├── error_annotation.py        # Error overlay on renders
└── inference_viz.py           # Inference-time visualization

mouse_extensions/scripts/inference/
├── render_from_checkpoint.py  # Checkpoint → renders
├── temporal_turntable.py      # Temporal sequence rendering
└── prune_gaussians.py         # Post-hoc pruning
```

---

## 9. Inference & Visualization

### 9.1 Checkpoint Loading

`GSLRM.load_from_checkpoint()` (`gslrm.py:1042`):
```python
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model'], strict=False)
# Pretrained: checkpoints/gslrm/ckpt_0000000000021125.pt
```

### 9.2 Turntable Rendering

`gslrm.py` → `save_visualization_outputs()` (L1341):
```
1. Forward pass → gaussian_params
2. Create GaussianModel from params
3. render_turntable(pc, resolution, num_views=60, elevation=20)
4. Grid layout → wandb upload / save to disk
```

**해상도 주의**: `config.resolution: null` → input 해상도 사용 (512). 값 지정 시 해당 해상도로 렌더.

### 9.3 Dataset View Rendering

`render_dataset_views()` (`gaussians_renderer.py:1381`):
각 데이터셋 카메라 위치에서 렌더링 → GT와 비교.

`render_dataset_trajectory()` (`gaussians_renderer.py:1301`):
카메라 간 보간 + `hold_frames` 정지 → 부드러운 영상.

---

## 10. Cross-References

| 주제 | 문서 |
|---|---|
| GS-LRM 아키텍처 상세 | `docs/architecture/GS-LRM_ARCHITECTURE_GUIDE.md` |
| Loss 수식 유도 | `docs/architecture/GS-LRM_Loss_Formula.md` |
| 평가 지표 (PSNR, SSIM, IoU) | `docs/architecture/METRICS_GUIDE.md` |
| 학습 파이프라인 상세 | `docs/architecture/TRAINING_PIPELINE_GUIDE.md` |
| 전처리 레지스트리 (D1~M5) | `docs/datasets/PREPROCESSING_REGISTRY.md` |
| PP/MVG 이론 분석 | `docs/theory/PP_FX_MVG_ANALYSIS.md` |
| 마스크 시스템 | `docs/theory/MASK_GUIDE.md` |
| Ghosting 분석 | `docs/theory/GHOSTING_ANALYSIS.md` |
| Config 설정 튜토리얼 | `docs/tutorials/Step4_Config_Setup.md` |

---

*FaceLift Code Walkthrough | Created: 2026-01-27*
