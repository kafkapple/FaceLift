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

#### Camera View Ordering (2 systems)

Visualization에서 카메라 뷰 순서는 **2가지 체계**가 공존합니다:

| 체계 | 순서 | 용도 | 라벨 위치 |
|------|------|------|-----------|
| **Data Order** | `[0, 1, 2, 3, 4, 5]` | Supervision (GT vs Pred), Input strip | 상단 "Cam 0, Cam 1, ..." |
| **Spatial Order** | Azimuth 정렬 (동적 계산) | Turntable grid, Trajectory video | 좌측 "0→4, 4→2, ..." |

**Data Order**: 데이터셋 로딩 순서. Supervision 이미지의 각 열은 카메라 인덱스 순서.
카메라 물리적 배치와 무관하며, loss 디버깅용으로 일관된 비교 제공.

**Spatial Order**: `get_dynamic_camera_order(c2ws)` — 카메라 extrinsics에서 azimuth 각도를
계산하여 CCW(반시계) 순서로 정렬. Turntable grid/video에서 연속적 회전 궤적 표현.
이전 하드코딩 `MOUSE_CAMERA_ORDER = [0,4,2,1,3,5]`을 대체.

```
Data Order:     Cam 0 | Cam 1 | Cam 2 | Cam 3 | Cam 4 | Cam 5
                (front) (R-front) (front-center) (right) (left) (back-right)

Spatial Order:  Cam 0 → Cam 4 → Cam 2 → Cam 1 → Cam 3 → Cam 5 → (back to 0)
                (azimuth sorted: -123° → -54° → +4° → +56° → +101° → +154°)
```

**주의**: Supervision Col[1] = Cam 1 ≠ Turntable Col[1] = Cam 4.
각 시각화에 라벨이 포함되어 있으므로 직접 비교 시 라벨을 확인할 것.

#### Visualization Output Catalog

**Config** (`configs/base/gslrm_mouse.yaml`):
```yaml
training.logging.vis_every: 100      # Training 시각화 주기
validation.val_every: 200             # Validation 주기
validation.visual_first_batch: true   # Val 첫 배치만 시각화

visualization.turntable:
  smooth_trajectory: true       # Dataset camera interpolation
  hold_frames: 15               # 각 카메라 위치 정지 프레임
  trajectory_fps: 10            # Trajectory 비디오 FPS
  save_orbit_turntable: true    # 360° orbit 별도 저장
  orbit_views: 120 / orbit_fps: 15
  video_views: 144              # 총 프레임 수
  grid_rows: 6, grid_cols: 6   # 6×6 = 36 grid
  resolution: null              # null = input resolution (512)
  save_video: true              # MP4 저장
```

##### Training Outputs (`iter_{step:08d}/`)

| # | 파일명 | 형식 | 카메라 순서 | 내용 | WandB |
|---|--------|------|-------------|------|-------|
| 1 | `supervision_{uids}.jpg` | Image | Data [0-5] | GT vs Pred (행: GT, Pred, [Mask], Error) | `images/train/supervision_0` |
| 2 | `input_{uids}.jpg` | Image | Data [0-5] | Input 뷰 (num_input_views) | `images/train/input_0` |
| 3 | `alpha_comparison_{uids}.jpg` | Image | Data [0-5] | GT Mask / Rendered α / α>0.5 / Diff | `images/train/alpha_comparison_0` |
| 4 | `turntable_{uid}.jpg` | Image | **Spatial** (azimuth) | 6×6 Grid (hold 제외, transition만) | `images/train/turntable_0` |
| 5 | `turntable_{uid}.mp4` | Video | Spatial | Dataset camera trajectory (144f, 10fps) | ❌ |
| 6 | `turntable_with_input_{uid}.mp4` | Video | Spatial + Data | Trajectory + 하단 input strip | ❌ |
| 7 | `turntable_orbit_{uid}.mp4` | Video | 360° orbit | Convergence center 기준 회전 (120f, 15fps) | ❌ |
| 8 | `turntable_orbit_with_input_{uid}.mp4` | Video | 360° + Data | Orbit + 하단 input strip | ❌ |
| 9 | `aligned_gs_opacity_depth_{uid}.jpg` | Image | Data [0-5] | Gaussian opacity + depth map | `images/train/gaussian_vis_0` |
| 10 | `input_{uid}.jpg` (inference only) | Image | Data | Individual input views | ❌ |
| 11 | `gaussians_{uid}.ply` | PLY | - | Filtered Gaussian model | ❌ |

**트리거**: `step == 0` 또는 `step % vis_every == 0` (`train_gslrm.py:670`)
**코드**: `gslrm.py:save_visualization_outputs()` (L1340)

##### Validation Outputs (`val/iter_{step:08d}/{uid:08d}/`)

| # | 파일명 | 형식 | 카메라 순서 | 내용 | WandB |
|---|--------|------|-------------|------|-------|
| 1 | `input.png` | Image | Data [0-5] | 6뷰 입력 이미지 | ❌ |
| 2 | `gt_vs_pred.png` | Image | Data [0-5] | GT vs Pred + error heatmap | `images/val/gt_vs_pred_0` |
| 3 | `alpha_comparison_{uid}.jpg` | Image | Data [0-5] | Alpha 비교 (GT/Pred/Diff) | `images/val/alpha_comparison_0` |
| 4 | `turntable_{uid}.jpg` | Image | **Spatial** (azimuth) | 6×6 Grid (hold 제외) | `images/val/turntable_0` |
| 5 | `turntable.mp4` | Video | Spatial | Dataset camera trajectory | ❌ |
| 6 | `turntable_with_input.mp4` | Video | Spatial + Data | Trajectory + input 오버레이 | ❌ |
| 7 | `turntable_orbit_{uid}.mp4` | Video | 360° orbit | Convergence center 기준 회전 | ❌ |
| 8 | `turntable_orbit_with_input_{uid}.mp4` | Video | 360° + Data | Orbit + input strip | ❌ |
| 9 | `gaussians.ply` | PLY | - | Filtered Gaussian model | ❌ |
| 10 | `perview_metrics.txt` | Text | - | Per-view PSNR/LPIPS/SSIM | ❌ |
| 11 | `metrics.txt` | Text | - | Averaged metrics | ❌ |
| 12 | `alpha_metrics.txt` | Text | - | IoU, precision, recall | ❌ |

**트리거**: `step == 0` 또는 `step % val_every == 0`, 첫 배치만 시각화 (`train_gslrm.py:1089`)
**코드**: `validator.py:ValidationRunner._save_visualizations()` (L172)

##### WandB Metrics (숫자)

| Prefix | 항목 | 트리거 |
|--------|------|--------|
| `train/` | loss, psnr, l2_loss, perceptual_loss, ssim_loss, lpips, bg_loss, alpha_loss | 매 log_every step |
| `val/` | loss, psnr, ssim, lpips, mask_iou, l2_loss | 매 val_every step |
| `val_view/` | view{N}_psnr, view{N}_lpips, view{N}_ssim | 매 val_every (첫 배치) |
| `meta/` | current_step, total_steps, progress | 매 val_every step |

##### 코드 경로 요약

```
train_gslrm.py
  ├── training_step()
  │   ├── model(batch, create_visual=True)         # vis_every마다
  │   └── save_visuals_if_needed()                  # L908
  │       ├── model.save_visuals() → gslrm.py:save_visualization_outputs()
  │       └── _log_visuals_to_wandb(prefix="train") # L923
  │
  └── run_validation()                              # val_every마다
      ├── model.save_validations() → validator.py:save_validation_results()
      │   └── _save_visualizations() → _create_turntable(), _save_grid(), etc.
      ├── wandb.log(val_metrics)                    # L1166
      └── _log_visuals_to_wandb(prefix="val")       # L1175

  # 독립 평가 (training loop에서 호출 안 됨)
  save_evaluation_results()                           # gslrm.py:L1737
    └── per-sample: input.png, gt_vs_pred.png, metrics.txt,
        perview_metrics.txt, gaussians.ply, turntable.mp4,
        turntable_preview.png, turntable_with_input.mp4
```

> **Note**: `save_evaluation_results()`는 독립 evaluation/inference 스크립트에서만 호출됩니다.
> Training 중 validation은 `save_validations()` → `ValidationRunner`를 사용합니다.

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
├── simple_temporal.py         # ✅ 권장 temporal inference
├── temporal_turntable.py      # v3: Gaussian filtering + per-frame center (실험적)
├── temporal_turntable_v4.py   # v4: render_turntable 기반 (안정)
├── render_from_checkpoint.py  # Checkpoint → single-frame renders
└── prune_gaussians.py         # Post-hoc pruning
```

#### Temporal Inference 스크립트 비교

| 스크립트 | 상태 | 렌더링 방식 | Center | 특징 |
|----------|------|------------|--------|------|
| **simple_temporal.py** | ✅ 권장 | `render_turntable()` (검증됨) | Gaussian mean | auto-discovery, multi-angle, grid |
| temporal_turntable.py (v3) | ⚠️ 실험적 | 자체 `render_opencv_cam()` 루프 | Per-frame foreground | Gaussian 필터링, 자체 카메라 생성 |
| temporal_turntable_v4.py | ✅ 안정 | `render_turntable()` (검증됨) | First frame fixed | v3의 안정화 버전 |

**v3 불안정 원인**:
1. 자체 카메라 생성 (`get_centered_turntable_cameras`) — hFOV 기반, 검증된 `render_turntable`과 다른 경로
2. Per-frame center 변동 → 시점 jitter
3. Gaussian 필터링 시 `GaussianModel` 수동 복사 → SH band 호환성 리스크
4. `target_camera_distance` config에서 읽음 → 0.0이면 정규화 안 됨 (수정 완료: fallback 2.7)

#### Temporal Inference 사용법

**기본 사용 (simple_temporal.py 권장)**:
```bash
cd /home/joon/dev/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh && conda activate facelift

CUDA_VISIBLE_DEVICES=6 python mouse_extensions/scripts/inference/simple_temporal.py \
  --checkpoint checkpoints/gslrm/M5h_2_E1_2_alpha/ckpt_0000000000001100.pt \
  --config configs/base/gslrm_mouse.yaml \
  --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5h_2 \
  --start_frame 0 --end_frame 120 --frame_step 5 \
  --num_views 36 --resolution 384 --elevation 20.0 --radius 2.7 \
  --fps 24 --output_dir outputs/temporal_M5h_2/simple
```

**출력 영상 (5종)**:
| 파일 | 내용 | 용도 |
|------|------|------|
| `turntable_first.mp4` | 첫 프레임 360° 회전 | 3D 품질 확인 |
| `time_fixed.mp4` | 고정 시점, 시간 변화 | 동작 분석 |
| `time_rotating.mp4` | 시간+회전 동시 | 시네마틱 |
| `full_all.mp4` | 전체 T×V 프레임 | 완전 탐색 |
| `grid_first.jpg` | 첫 프레임 6×6 그리드 | Quick reference |

**주요 파라미터**:
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--frame_step` | 1 | 프레임 샘플링 간격 (5=매 5번째) |
| `--num_views` | 36 | 360°를 몇 등분 (36=10° 간격) |
| `--resolution` | 384 | 렌더링 해상도 |
| `--elevation` | 20.0 | 카메라 고도각 |
| `--radius` | 2.7 | 카메라 거리 (정규화 단위) |
| `--fixed_angles` | [0] | 고정 시점 영상에 사용할 각도 인덱스 |
| `--start/end_frame` | None | 자동 감지 (미지정 시) |

**GPU 주의**: gpu03에서 GPU 0-3은 Blackwell (sm_120, PyTorch 미지원). **GPU 4-7 (A6000)** 사용 필수.

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
