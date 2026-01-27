# FaceLift GS-LRM Training Pipeline Guide

> **목적**: `train_gslrm.py` 실행 시 전체 학습 파이프라인을 단계별로 따라갈 수 있는 상세 가이드
> **대상**: FaceLift 코드베이스에 처음 접근하는 사람
> **Created**: 2026-01-27

---

## 1. Overview (전체 흐름도)

```
┌──────────────────────────────────────────────────────────────────┐
│  train_gslrm.py (Entry Point)                                    │
│                                                                  │
│  [1] Config Loading ─── Modular: base ← dataset ← experiment    │
│           │                                                      │
│  [2] GSLRMTrainer.__init__()                                     │
│           ├── setup_distributed()   DDP / torchrun               │
│           ├── setup_cuda()          Device, seed, TF32           │
│           ├── load_datasets()       MouseViewDataset             │
│           ├── setup_model()         GSLRM (dynamic import)       │
│           ├── setup_optimization()  Adam + cosine scheduler      │
│           ├── load_checkpoint()     Pretrained or resume         │
│           └── setup_wandb()         Logging                      │
│                                                                  │
│  [3] GSLRMTrainer.train()                                        │
│           while step < max_steps:                                │
│             batch = next(dataloader)                             │
│             result = train_step(batch)     ── GSLRM.forward()   │
│             optimizer_step(result)         ── grad clip + step   │
│             log / checkpoint / visualize / validate              │
└──────────────────────────────────────────────────────────────────┘
```

### 핵심 파일 위치

| Component | File | Lines |
|-----------|------|-------|
| **Trainer** | `train_gslrm.py` | 1502 |
| **GSLRM Model** | `gslrm/model/gslrm.py` | 1910 |
| **Gaussian Renderer** | `gslrm/model/gaussians_renderer.py` | 1649 |
| **RandomViewDataset** | `gslrm/data/dataset.py` | 307 |
| **MouseViewDataset** | `mouse_extensions/data/mouse_dataset.py` | 664 |
| **Loss Extensions** | `mouse_extensions/model/loss_extensions.py` | 823 |
| **Base Config** | `configs/base/gslrm_mouse.yaml` | - |
| **Pretrained Checkpoint** | `checkpoints/gslrm/ckpt_0000000000021125.pt` | - |

---

## 2. Entry Point (train_gslrm.py)

### 2.1 Config 로딩

두 가지 모드를 지원한다:

**Legacy Mode** — 단일 YAML 파일 지정:
```bash
torchrun ... train_gslrm.py --config configs/mouse/D7_1_E3.yaml
```
- `train_gslrm.py:1389` `load_and_process_config()` 호출

**Modular Mode** (권장) — base + dataset + experiment 3단계 merge:
```bash
torchrun ... train_gslrm.py --dataset D7_1 --experiment E3_2_5v_alpha
```
- `train_gslrm.py:1287` `load_modular_config()` 호출
- Merge 순서: `base/gslrm_mouse.yaml` ← `datasets/D7_1.yaml` ← `experiments/E3_2_5v_alpha.yaml`
- 자동 생성: `checkpoint_dir`, `wandb.group`, `wandb.exp_name`

### 2.2 GSLRMTrainer 초기화 순서

```python
class GSLRMTrainer:                    # train_gslrm.py:79
    def __init__(self, config, args):  # train_gslrm.py:82
```

| 순서 | Method | Line | 역할 |
|------|--------|------|------|
| 1 | `setup_distributed()` | 104 | DDP 초기화, `ddp_rank`, `ddp_world_size` 설정 |
| 2 | `setup_cuda()` | 132 | Device 설정, seed=777+rank, TF32 활성화 |
| 3 | `load_datasets()` | 189 | MouseViewDataset 또는 RandomViewDataset 생성 |
| 4 | `_setup_dataloaders()` | 257 | DataLoader + DistributedSampler, fixed vis sample 저장 |
| 5 | `setup_model()` | 310 | GSLRM class dynamic import, DDP wrap |
| 6 | `_apply_layer_freeze()` | 331 | Finetuning 시 초기 transformer layer + patch embedder 동결 |
| 7 | `setup_optimization()` | 395 | Adam optimizer, cosine scheduler, GradScaler |
| 8 | `load_checkpoint()` | 444 | Pretrained 또는 resume checkpoint 로드 |
| 9 | `setup_wandb()` | 485 | W&B 초기화 (rank 0 only) |

### 2.3 Train Loop 상세

```python
def train(self):  # train_gslrm.py:1220
    while self.fwdbwd_pass_step <= max_steps:
        # [1] Batch loading
        batch = next(self.dataloader_iter)                    # :1241
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # [2] Forward + Backward
        result, create_visual, create_val = self.train_step(batch)  # :1246

        # [3] Optimizer step
        total_grad_norm = self.optimizer_step(result)         # :1251

        # [4] Logging
        self.log_training_metrics(result, total_grad_norm)    # :1256

        # [5] Checkpoint
        self.save_checkpoint_if_needed()                      # :1260

        # [6] Visualization (turntable 등)
        self.save_visuals_if_needed(result, batch)            # :1263

        # [7] Validation
        if create_val:
            self.run_validation()                             # :1271
```

### 2.4 Train Step (train_gslrm.py:667)

```python
def train_step(self, batch):
    # Vis/val 플래그 결정
    create_visual = (step % vis_every == 0)                   # :669
    create_val = (step % val_every == 0)                      # :675

    # Gradient accumulation context
    ctx = nullcontext() if last_accum_step else no_sync()     # :683

    # AMP forward
    with torch.autocast("cuda", dtype=bf16):                  # :688
        result = self.model(batch, create_visual=create_visual)  # :704

    # Backward
    loss = result.loss_metrics.loss / grad_accum_steps
    self.scaler.scale(loss).backward()                        # :721
    self.fwdbwd_pass_step += 1
```

### 2.5 Optimizer Step (train_gslrm.py:763)

```python
def optimizer_step(self, result):
    # NaN check
    if torch.isnan(result.loss_metrics.loss):                 # :771
        skip = True

    # Every grad_accum_steps:
    total_grad_norm = clip_grad_norm_(params, max_norm=50.0)  # :795
    self.scaler.step(self.optimizer)                          # :809
    self.param_update_step += 1
    self.lr_scheduler.step()                                  # :817
    self.optimizer.zero_grad(set_to_none=True)                # :820
```

---

## 3. Data Pipeline

### 3.1 Dataset 선택

`train_gslrm.py:189` `load_datasets()`:
- `config.mouse.use_mouse_dataset = true` → **MouseViewDataset**
- Otherwise → **RandomViewDataset**

### 3.2 MouseViewDataset

**File**: `mouse_extensions/data/mouse_dataset.py:62`

핵심 처리 단계 (`__getitem__`, line 290-483):

```
[1] Load cameras (opencv_cameras.json)              :304
[2] View selection (fixed order [0,1,2,...])         :321
[3] Image load + resize → image_size (512)           :369
[4] Augmentation (per-view consistent)               :377
[5] RGBA channel processing                          :390
[6] PP correction (v12/v13 intrinsics fix)           :395
[7] Camera pose: w2c → c2w conversion                :411
[8] Auto alpha mask generation                       :419
[9] Camera normalization (Z-up / distance)           :447
[10] Distance scaling → target_camera_distance       :461
```

### 3.3 RandomViewDataset (원본)

**File**: `gslrm/data/dataset.py:84`

MouseViewDataset과의 차이:
- View order: **random** (`random.sample()`) vs **fixed** (`sorted()`)
- Camera normalization: 없음 (원본 데이터 그대로)
- Auto mask: 없음

### 3.4 Output Tensor Shapes

`__getitem__` 반환값 (single sample, batch 전):

| Key | Shape | Description |
|-----|-------|-------------|
| `image` | `[V, 4, H, W]` = `[6, 4, 512, 512]` | RGBA multi-view images |
| `c2w` | `[V, 4, 4]` = `[6, 4, 4]` | Camera-to-world matrices |
| `fxfycxcy` | `[V, 4]` = `[6, 4]` | Intrinsics (fx, fy, cx, cy) |
| `index` | `[V, 2]` = `[6, 2]` | `[:,0]`=camera idx, `[:,1]`=scene idx |
| `bg_color` | `[4]` | Background color (RGBA) |

DataLoader 이후 (batched):

| Key | Shape | Notes |
|-----|-------|-------|
| `image` | `[B, V, 4, H, W]` | B=batch_size_per_gpu |
| `c2w` | `[B, V, 4, 4]` | |
| `fxfycxcy` | `[B, V, 4]` | |

### 3.5 카메라 파일 포맷 (opencv_cameras.json)

```json
{
  "frames": [
    {
      "file_path": "images/cam0/frame_00000.png",
      "w": 512, "h": 512,
      "fl_x": 549.0, "fl_y": 549.0,
      "cx": 256.0, "cy": 256.0,
      "transform_matrix": [[...], [...], [...], [...]]
    }
  ]
}
```

---

## 4. Model Architecture (gslrm.py)

### 4.1 주요 클래스

**File**: `gslrm/model/gslrm.py`

| Class | Line | Purpose |
|-------|------|---------|
| `Renderer` | 100 | Gaussian → 2D rendering |
| `GaussiansUpsampler` | 194 | Latent tokens → Gaussian params |
| `LossComputer` | 279 | Loss 계산 |
| `GSLRM` | 669 | End-to-end model |

### 4.2 GSLRM 초기화

```python
class GSLRM(nn.Module):                       # gslrm.py:669
    def __init__(self, config):                # gslrm.py:684
        self._init_data_processors(config)     # :701  SplitData, InputTransformer
        self._init_tokenizer(config)           # :707  PatchEmbedder
        self._init_positional_embeddings(config) # :727  View type + Gaussian embeddings
        self._init_transformer(config)         # :746  24-layer Transformer
        self._init_gaussian_modules(config)    # :759  Upsampler + PixelDecoder
        self._init_rendering_modules(config)   # :784  Renderer
        self._init_training_state(config)      # :789  LossComputer, counters
```

### 4.3 Forward Pass (gslrm.py:1146)

```
                          GSLRM.forward() Pipeline
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  [1] Data Split & Transform (no_grad)                    :1161       │
│      batch_data → input_data + target_data                           │
│      input_data → posed_images (RGB + Plücker coords)               │
│      Shape: [B, V, 9, H, W]  (3 RGB + 6 Plücker)                   │
│                                                                      │
│  [2] Patch Tokenization                                  :1179       │
│      posed_images → image_patch_tokens                               │
│      [B, V, 9, 512, 512] → [B*V, 4096, 1024]                       │
│      Reshape → [B, V*4096=24576, 1024]                              │
│                                                                      │
│  [3] View Type Embeddings (optional)                     :1190       │
│      ref_marker / src_marker 추가 (2D params)                        │
│                                                                      │
│  [4] Gaussian Position Tokens                            :1196       │
│      [B, 2, 1024] learnable embeddings                              │
│                                                                      │
│  [5] Transformer (24 layers)                             :1200       │
│      Input:  [B, 2+24576=24578, 1024]                               │
│      Output: [B, 24578, 1024]                                       │
│                                                                      │
│  [6] Token Split                                         :1207       │
│      gaussian_tokens: [B, 2, 1024]                                  │
│      image_tokens:    [B, 24576, 1024]                              │
│                                                                      │
│  [7] Gaussian Upsampler                                  :1212       │
│      gaussian_tokens → [B, 2, 14]                                   │
│                                                                      │
│  [8] Pixel-Aligned Gaussian Decoder                      :1216       │
│      image_tokens → [B, 24576, 14]                                  │
│                                                                      │
│  [9] Combine All Gaussians                               :1227       │
│      [B, 2+24576, 14] = [B, 24578, 14]                             │
│                                                                      │
│  [10] to_gs(): Split into params                         :1230       │
│       xyz:      [B, N, 3]                                           │
│       features: [B, N, 1, 3]   (SH degree 0)                       │
│       scaling:  [B, N, 3]                                           │
│       rotation: [B, N, 4]                                           │
│       opacity:  [B, N, 1]                                           │
│                                                                      │
│  [11] Hard Pixel Alignment (optional)                    :1250       │
│       Pixel-aligned xyz → unproject to 3D                            │
│                                                                      │
│  [12] Gaussian Rendering                                 :1266       │
│       → rendered_images: [B, V, 3, H, W]                            │
│       → rendered_alpha:  [B, V, 1, H, W]                            │
│                                                                      │
│  [13] Loss Computation                                   :1286       │
│       L2 + Perceptual + Alpha + Background + ...                     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.4 Tokenizer (Patch Embedding)

```python
# gslrm.py:707-725
patch_size = 8
in_channels = 9  # RGB(3) + Plücker(6)
hidden_dim = 1024

self.patch_embedder = nn.Sequential(
    Rearrange("b v c (h ph) (w pw) -> (b v) (h w) (ph pw c)", ph=8, pw=8),
    nn.Linear(9 * 8 * 8, 1024, bias=False),
)
```

Input: `[B, V, 9, 512, 512]` → Output: `[B*V, 4096, 1024]`

### 4.5 Plücker 좌표

Per-pixel ray encoding (6D). 뷰 순서에 의존하지 않음.

```
Pixel (u,v) + Camera (K, c2w):
  Ray origin:    o = c2w[:3, 3]
  Ray direction: d = K^{-1} @ [u, v, 1]^T (then normalize)
  Plücker:       L = (d, o × d)  (6D vector)
```

Input image `[B, V, 3, H, W]` + Plücker `[B, V, 6, H, W]` → Posed `[B, V, 9, H, W]`

### 4.6 Transformer

```python
# gslrm.py:746-757
hidden_dim = 1024
head_dim = 64     # → 16 heads
num_layers = 24

self.input_layer_norm = nn.LayerNorm(1024, bias=False)
self.transformer_layers = nn.ModuleList([
    TransformerBlock(1024, 64) for _ in range(24)
])
```

### 4.7 GaussiansUpsampler (gslrm.py:194)

```python
class GaussiansUpsampler(nn.Module):
    # gaussian_param_dim = 3(xyz) + 3(sh) + 3(scale) + 4(rot) + 1(opacity) = 14
    self.layernorm = nn.LayerNorm(1024, bias=False)
    self.linear = nn.Linear(1024, 14, bias=False)

    def to_gs(self, gaussians):
        xyz, features, scaling, rotation, opacity = gaussians.split([3,3,3,4,1], dim=2)
        scaling = (scaling - 2.3).clamp(max=-1.20)  # → exp() activation
        opacity = opacity - 2.0                       # → sigmoid() activation
        return xyz, features, scaling, rotation, opacity
```

---

## 5. Loss Computation

### 5.1 LossComputer (gslrm.py:279)

```python
class LossComputer(nn.Module):       # gslrm.py:279
    def forward(self, rendering, target, img_aligned_xyz, input,
                create_visual, rendered_alpha, view_indices):  # :325
```

### 5.2 핵심 Loss 종류

| Loss | Weight Config | Line | 수식 |
|------|---------------|------|------|
| **L2** | `l2_loss_weight: 1.0` | 466 | MSE(rendered, target) |
| **LPIPS** | `lpips_loss_weight: 0.0` | 489 | LPIPS perceptual (VGG) |
| **Perceptual** | `perceptual_loss_weight: 0.5` | 498 | Custom VGG feature loss |
| **SSIM** | `ssim_loss_weight: 0.0` | 517 | 1 - SSIM |
| **Pixel Align** | `pixelalign_loss_weight` | 553 | Depth ordering consistency |
| **Points Dist** | `pointsdist_loss_weight` | 572 | 3D position regularization |

### 5.3 Warmup 동작

```
Step 0 ~ l2_warmup_steps (500):
  → L2 loss ONLY (perceptual 등 비활성)

Step 500+:
  → L2 + Perceptual + (기타 loss 추가)
```

### 5.4 Mask 계산 우선순위

```
[1] GT mask (mask_mode: "gt")     → target alpha channel 사용
[2] Alpha mask (mask_mode: "alpha") → rendered alpha 사용 (⚠️ 피드백 루프 주의)
[3] None (mask_mode: "none")      → 전체 이미지 loss
```

**권장**: `mask_mode: gt` — GT mask가 가장 안정적.

### 5.5 Mouse-Specific Loss (loss_extensions.py)

**File**: `mouse_extensions/model/loss_extensions.py`

| Class | Line | Purpose |
|-------|------|---------|
| `LossExtensions` | 362 | Mask handling helper |
| `AlphaLossComputer` | 540 | Alpha channel supervision (MSE) |
| `OpacityRegularizer` | 666 | Entropy / L1 sparse / L2 binary |
| `DepthRegularizer` | 754 | Depth regularization |

**bf16 NaN 버그 수정** (line 388):
```python
# Before: opacity.clamp(1e-6, 1-1e-6)
# After:  opacity.float().clamp(1e-4, 1-1e-4)
```

---

## 6. Rendering & Visualization

### 6.1 Renderer (gslrm.py:100)

```python
class Renderer(nn.Module):
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)  # Always float32
    def forward(self, xyz, features, scaling, rotation, opacity,
                height, width, C2W, fxfycxcy, deferred=True):  # :122
```

- **Deferred mode** (기본): `DeferredGaussianRender.apply()` — custom autograd Function
- **Sequential mode**: Per-batch, per-view 순차 렌더링

### 6.2 DeferredGaussianRender

**File**: `gslrm/model/gaussians_renderer.py:1086`

```python
class DeferredGaussianRender(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, features, scaling, rotation, opacity,
                height, width, C2W, fxfycxcy, ...):
        for i in range(B):
            for j in range(V):
                result = render_opencv_cam(pc, H, W, C2W[i,j], fxfycxcy[i,j])
                renders.append(result["render"])   # [3, H, W]
                alphas.append(result["alpha"])      # [1, H, W]

        return renders.reshape(B, V, 3, H, W), alphas.reshape(B, V, 1, H, W)
```

### 6.3 Turntable 시각화

- `vis_every` step마다 turntable 렌더링 생성
- Dataset camera trajectory 순회 또는 360° orbit
- W&B에 `turntable_{uid}.jpg` 업로드

**해상도 주의**: `config.resolution: null` 설정 시 input resolution 사용 (권장)

### 6.4 W&B 로깅

| Metric | 시점 | 설명 |
|--------|------|------|
| `train/loss` | 매 step | Total loss |
| `train/psnr` | 매 step | Peak SNR |
| `train/l2_loss` | 매 step | L2 component |
| `train/perceptual_loss` | warmup 후 | VGG feature loss |
| `train/background_loss` | warmup 후 | Background penalty |
| `val/psnr` | val_every | Validation PSNR |

---

## 7. Config System

### 7.1 3단계 Merge (Modular Mode)

```
configs/
├── base/
│   └── gslrm_mouse.yaml       ← [1] Base (모든 기본값)
├── datasets/
│   └── D7_1.yaml               ← [2] Dataset (데이터 경로, 뷰 수)
└── experiments/
    └── E3_2_5v_alpha.yaml       ← [3] Experiment (loss, mask, lr)
```

Merge: `base` ← `dataset` ← `experiment` (나중이 우선)

### 7.2 주요 설정값

#### Model

| Key | Default | Description |
|-----|---------|-------------|
| `model.class_name` | `gslrm.model.gslrm.GSLRM` | Dynamic import path |
| `model.image_tokenizer.patch_size` | `8` | Patch size for tokenization |
| `model.image_tokenizer.in_channels` | `9` | RGB(3) + Plücker(6) |
| `model.transformer.d` | `1024` | Hidden dimension |
| `model.transformer.d_head` | `64` | Head dimension |
| `model.transformer.n_layer` | `24` | Transformer depth |
| `model.gaussians.n_gaussians` | `2` | Learnable Gaussian tokens |
| `model.gaussians.sh_degree` | `0` | Spherical harmonics degree |
| `model.num_views` | `6` | Total views per sample |
| `model.num_input_views` | `4` | Input views (rest = target) |
| `model.hard_pixelalign` | `true` | Unproject pixel Gaussians to 3D |

#### Training

| Key | Default | Description |
|-----|---------|-------------|
| `training.runtime.use_amp` | `true` | Mixed precision |
| `training.runtime.amp_dtype` | `bf16` | AMP dtype |
| `training.runtime.grad_clip_norm` | `50.0` | Max gradient norm |
| `training.dataloader.batch_size_per_gpu` | `2` | Batch size |
| `training.optimizer.lr` | `1e-6` | Learning rate |
| `training.schedule.warmup` | `500` | LR warmup steps |
| `training.schedule.l2_warmup_steps` | `500` | L2-only warmup |
| `training.schedule.max_fwdbwd_passes` | `15000` | Max training steps |

#### Losses

| Key | Default | Description |
|-----|---------|-------------|
| `training.losses.l2_loss_weight` | `1.0` | L2 loss weight |
| `training.losses.perceptual_loss_weight` | `0.5` | VGG perceptual |
| `training.losses.opacity_reg_weight` | `0.0` | Opacity regularization |
| `training.losses.alpha_loss_weight` | `0.0` | Alpha supervision |
| `training.losses.background_loss_weight` | `0.0` | Background penalty |

#### Mouse-Specific

| Key | Default | Description |
|-----|---------|-------------|
| `mouse.use_mouse_dataset` | `true` | Use MouseViewDataset |
| `mouse.normalize_cameras` | `false` | Camera normalization |
| `mouse.normalize_to_z_up` | `true` | Z-up coordinate system |
| `mouse.auto_generate_mask` | `false` | Auto alpha mask |

---

## 8. Tensor Shape Reference (Quick Reference)

파이프라인 전체에서 텐서가 어떻게 변환되는지 한눈에:

```
Stage                    Tensor              Shape
─────────────────────────────────────────────────────────────────
DataLoader output        batch["image"]      [B, V=6, C=4, H=512, W=512]
                         batch["c2w"]        [B, V=6, 4, 4]
                         batch["fxfycxcy"]   [B, V=6, 4]

Data Split               input_data.image    [B, Vi=4, 4, 512, 512]
                         target_data.image   [B, Vt=2, 4, 512, 512]

Plücker + RGB            posed_images        [B, Vi=4, 9, 512, 512]

Patch Tokenization       image_patch_tokens  [B, Vi×P=4×4096=16384, 1024]

Gaussian Tokens          gaussian_tokens     [B, 2, 1024]

Transformer Input        combined            [B, 2+16384=16386, 1024]

Transformer Output       combined            [B, 16386, 1024]

Token Split              gaussian_tokens     [B, 2, 1024]
                         image_tokens        [B, 16384, 1024]

Gaussian Upsampler       gaussian_params     [B, 2, 14]

Pixel Decoder            pixel_params        [B, 16384, 14]

All Gaussians            all_params          [B, 16386, 14]

to_gs()                  xyz                 [B, N, 3]
                         features            [B, N, 1, 3]
                         scaling             [B, N, 3]
                         rotation            [B, N, 4]
                         opacity             [B, N, 1]

Rendered Output          rendered_images     [B, Vt=2, 3, 512, 512]
                         rendered_alpha      [B, Vt=2, 1, 512, 512]
```

**Notes**:
- `V=6` (total), `Vi=4` (input), `Vt=2` (target) — 기본 설정 기준
- `P=4096` = (512/8)² patches per view
- `N=16386` = 2 (learnable) + 16384 (pixel-aligned) Gaussians
- `14` = 3(xyz) + 3(sh) + 3(scale) + 4(rot) + 1(opacity)

---

## 9. 실행 명령어

```bash
# 단일 GPU
cd /home/joon/dev/FaceLift
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --dataset D7_1 --experiment E3_2_5v_alpha

# Multi-GPU (4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    train_gslrm.py --dataset D7_1 --experiment E3_2_5v_alpha

# Legacy mode
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E3.yaml
```

---

## 10. 관련 문서

| 문서 | 위치 | 내용 |
|------|------|------|
| Dataset Master | `docs/datasets/DATASET_MASTER_REFERENCE.md` | M-series 데이터셋 비교 |
| Preprocessing | `docs/datasets/PREPROCESSING_REGISTRY.md` | 전처리 프리셋 상세 |
| Experiment Quick | `docs/EXPERIMENT_QUICKSTART.md` | 실험 설정 빠른 참조 |
| PP/MVG Analysis | `docs/theory/PP_FX_MVG_ANALYSIS.md` | PP 버그, 이론 |
| Visualization | `docs/VISUALIZATION_SETTINGS.md` | Turntable 설정 |

---

*FaceLift Training Pipeline Guide v1.0 | Created: 2026-01-27*
