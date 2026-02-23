# CH2: GS-LRM Code Flow

> Config 시스템, 학습 루프, 모델 Forward Pass, Loss 계산, Validation을 코드 레벨로 상세히 설명합니다.
>
> ← [[CH1_ENVIRONMENT_AND_DATA]] | [[EXPERIMENT_MASTER_GUIDE]] | [[CH3_EXPERIMENTS_AND_RESULTS]] →

---

## 1. Config System (3-Layer Merge)

FaceLift는 3개 YAML을 계층적으로 병합하여 최종 설정을 만듭니다.

```
Layer 1: Base Config          configs/base/gslrm_mouse.yaml     (모델, 학습 공통)
Layer 2: Dataset Config       configs/datasets/M5t2.yaml         (데이터 경로, split)
Layer 3: Experiment Config    configs/mouse/uniform/4view_v2.yaml (실험별 차이)
```

### 1.1 실행 명령어

```bash
# Modular config (3-layer)
python train_gslrm.py -d M5t2 -e E0_1_facelift

# Uniform config (2-layer: base + experiment overlay)
python train_gslrm.py \
    -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/4view_v2.yaml
```

### 1.2 Base Config (`base_uniform_v2.yaml`) 핵심 설정

```yaml
# ── 학습 ──
learning_rate: 1.0e-6           # Fine-tuning LR (pretrained에서)
max_fwdbwd_passes: 15000        # = 15840 actual steps (batch 2)
gradient_accumulation_steps: 1
max_grad_norm: 50.0
seed: 42

# ── 모델 ──
num_views: 6                    # Total views per sample
num_input_views: 4              # Input to transformer
random_view_selection: true

# ── Loss ──
l2_loss_weight: 1.0             # MSE loss
perceptual_loss_weight: 0.5     # VGG19 multi-scale
lpips_loss_weight: 0.05         # LPIPS (VGG)
ssim_loss_weight: 0.1           # 1 - SSIM
alpha_loss_weight: 0.0          # Alpha supervision (off by default)
mask_mode: none                 # no mask (none / gt)

# ── Validation ──
val_every: 200                  # steps
early_stopping_patience: 10     # = 2000 steps of no improvement

# ── Resume ──
pretrained: checkpoints/gslrm/ckpt-21125.pt  # Objaverse pretrained
```

### 1.3 Experiment Overlay 예시 (`4view_v2.yaml`)

```yaml
# 4view_v2는 base와 동일 (baseline)
# 변경 없음 — overlay로 base 값을 그대로 사용

# 반면 6view_v2.yaml:
num_input_views: 6              # 4 → 6으로 변경

# 또는 4view_ssim03_v2.yaml:
ssim_loss_weight: 0.3           # 0.1 → 0.3으로 변경
```

### 1.4 OmegaConf Merge 로직

```python
# train_gslrm.py → load_modular_config()
from omegaconf import OmegaConf

def load_modular_config(args):
    base = OmegaConf.load(args.base_config)        # Layer 1
    dataset = OmegaConf.load(args.dataset_config)   # Layer 2
    experiment = OmegaConf.load(args.experiment)     # Layer 3

    # 나중 layer가 이전 layer 값을 override
    config = OmegaConf.merge(base, dataset, experiment)

    # 자동 경로 생성
    config.checkpoint_dir = f"checkpoints/gslrm/{experiment_name}"
    config.wandb.exp_name = experiment_name

    return config
```

---

## 2. Training Loop (GSLRMTrainer)

**파일**: `train_gslrm.py` (~1739줄)

### 2.1 초기화 순서

```python
class GSLRMTrainer:
    def __init__(self, config, args):
        # 1. DDP 설정 (multi-GPU시)
        self.setup_distributed()

        # 2. CUDA 최적화
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # 3. 데이터셋 로드
        self.train_dataset = MouseViewDataset(config, split='train')
        self.val_dataset = MouseViewDataset(config, split='val')
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=2,           # GPU 메모리 제약
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )

        # 4. 모델 생성
        self.model = GSLRM(config)
        # Layer freezing: 첫 N개 transformer block 동결 (optional)
        if config.get('freeze_layers', 0) > 0:
            freeze_first_n_layers(self.model, config.freeze_layers)

        # 5. Optimizer + Scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,    # 1e-6
            weight_decay=0.05
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_fwdbwd_passes
        )
        self.scaler = GradScaler()  # AMP (bf16)

        # 6. Checkpoint 로드 (pretrained or resume)
        self.load_checkpoint()

        # 7. WandB 초기화
        wandb.init(project="FaceLift", name=config.wandb.exp_name)
```

### 2.2 학습 루프 핵심

```python
def train(self):
    self.model.train()
    dataloader_iter = iter(self.train_loader)

    while self.step <= self.total_steps:
        # ── Batch 로드 ──
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            self.epoch += 1
            dataloader_iter = iter(self.train_loader)
            batch = next(dataloader_iter)

        batch = {k: v.cuda() for k, v in batch.items()}

        # ── Forward Pass ──
        with autocast(dtype=torch.bfloat16):
            result = self.model(
                batch,
                step=self.step,
                create_visual=(self.step % 100 == 0)
            )

        # ── Backward Pass ──
        loss = result.loss_metrics.loss / self.grad_accum
        self.scaler.scale(loss).backward()

        # ── Optimizer Step (매 grad_accum번째) ──
        if self.step % self.grad_accum == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=50.0                # Gradient clipping
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

        # ── Logging (매 step) ──
        wandb.log({
            "train/loss": loss.item(),
            "train/l2_loss": result.loss_metrics.l2_loss,
            "train/psnr": result.loss_metrics.psnr,
            "train/ssim_loss": result.loss_metrics.ssim_loss,
            "train/lr": self.scheduler.get_last_lr()[0],
        }, step=self.step)

        # ── Validation (매 200 step) ──
        if self.step % self.val_every == 0:
            val_metrics = self.run_validation()
            # best_psnr.pt 저장 (현재 best보다 높으면)
            if val_metrics['psnr'] > self.best_psnr:
                self.best_psnr = val_metrics['psnr']
                self.save_best_checkpoint()

            # Early stopping 체크
            self.patience_counter += 1
            if val_metrics['psnr'] > self.best_psnr:
                self.patience_counter = 0
            if self.patience_counter >= 10:
                break  # 2000 steps 무개선

        # ── Regular Checkpoint (매 100 step) ──
        if self.step % 100 == 0:
            self.save_checkpoint()  # ckpt_{step}.pt

        self.step += 1
```

---

## 3. GSLRM Model Forward Pass (14 Steps)

**파일**: `gslrm/model/gslrm.py` (~1766줄)

```python
class GSLRM(nn.Module):
    def forward(self, batch_data, step=0, create_visual=False):
        """
        Input:  batch_data = {image: [B,V,4,H,W], c2w: [B,V,4,4], fxfycxcy: [B,V,4]}
        Output: result with rendered images + loss metrics
        """

        # ═══ Step 1: Input/Target 분리 ═══
        # SplitData: first num_input_views → input, all num_views → target
        input_data, target_data = self.data_splitter(batch_data)
        # input:  [B, 4, 4, 512, 512]  (4 input views)
        # target: [B, 6, 4, 512, 512]  (6 target views = ALL)

        # ═══ Step 2: Target Transform (optional crop) ═══
        target_data = self.target_transformer(target_data)

        # ═══ Step 3: Input Transform (ray 계산) ═══
        input_data = self.input_transformer(batch_data)
        # ray_o: [B, 4, 3, 64, 64]  (ray origins at patch centers)
        # ray_d: [B, 4, 3, 64, 64]  (ray directions)

        # ═══ Step 4: Posed Images + Plucker Coordinates ═══
        # RGB[-1,1] + cross(ray_o, ray_d) + ray_d = 9 channels
        posed_images = self._create_posed_images_with_plucker(input_data)
        # [B, 4, 9, 512, 512]  (custom plucker: RGB + ray_d + nearest_points)

        # ═══ Step 5: Patch Embedding ═══
        # patch_size=8 → 512/8 = 64 → 64×64 = 4096 patches per view
        tokens = self.patch_embedder(posed_images)
        # [B, 4×4096, 1024] = [B, 16384, 1024]

        # ═══ Step 6: View Type Embeddings (optional) ═══
        # Reference view vs source view marker
        if self.use_view_type_embed:
            tokens += self.view_type_embeddings

        # ═══ Step 7: Gaussian Position Embeddings ═══
        # 2개의 learnable global Gaussian tokens
        gauss_tokens = self.gaussian_position_embeddings  # [B, 2, 1024]
        all_tokens = torch.cat([gauss_tokens, tokens], dim=1)
        # [B, 16386, 1024]

        # ═══ Step 8: Transformer Processing ═══
        # 24 TransformerBlock layers (d=1024, heads=16, d_head=64)
        # Gradient checkpointing 사용 (메모리 절약)
        output_tokens = self._process_through_transformer(all_tokens)
        # [B, 16386, 1024]

        # ═══ Step 9: Global Gaussian Upsampler ═══
        # 2 global tokens → LayerNorm → Linear → Gaussian params
        global_gaussians = self.gaussian_upsampler(output_tokens[:, :2])

        # ═══ Step 10: Pixel-Aligned Gaussian Decoder ═══
        # 16384 image tokens → LayerNorm → Linear → Gaussian params
        pixel_gaussians = self.pixel_gaussian_decoder(output_tokens[:, 2:])

        # ═══ Step 11: Gaussian Parameters Decoding ═══
        all_params = torch.cat([global_gaussians, pixel_gaussians], dim=1)
        # Split into: xyz(3) + SH(3) + scaling(3) + rotation(4) + opacity(1)
        xyz, sh, scaling, rotation, opacity = self.to_gs(all_params)

        # Activation functions:
        # scaling = exp(x - 2.3), clamped max=-1.20
        # opacity = sigmoid(x - 2.0)
        # rotation = normalize(quaternion)

        # ═══ Step 12: Hard Pixel Alignment (optional) ═══
        if self.hard_pixel_alignment:
            # sigmoid(depth) → ray_o + depth * ray_d
            depth = torch.sigmoid(xyz[:, :, 2:3])
            xyz = ray_origins + depth * ray_directions

        # ═══ Step 13: Gaussian Rendering ═══
        # Deferred Gaussian Splatting
        rendered = self.gaussian_renderer(
            xyz, sh, scaling, rotation, opacity,
            target_c2w, target_fxfycxcy,
            image_size=512, bg_color=[1,1,1]
        )
        # rendered_images: [B, 6, 3, 512, 512]
        # rendered_alpha:  [B, 6, 1, 512, 512]

        # ═══ Step 14: Loss Computation ═══
        loss_result = self.loss_calculator(
            rendered_images, target_images,
            rendered_alpha, target_masks,
            step=step
        )

        return ModelOutput(
            rendered_images=rendered_images,
            rendered_alpha=rendered_alpha,
            loss_metrics=loss_result,
            gaussians=(xyz, sh, scaling, rotation, opacity)
        )
```

### 3.1 Gaussian 수 분석

| Component | 수량 | 역할 |
|-----------|------|------|
| Global Gaussians | 2 | 전체 장면 배경/구조 |
| Pixel-aligned Gaussians | 4 × 4096 = 16,384 | 각 input view의 패치별 Gaussian |
| **Total** | **16,386** | |

### 3.2 Transformer 구조

```
Input: [B, 16386, 1024]
       ↓
24 × TransformerBlock:
  ├─ LayerNorm
  ├─ Multi-Head Self-Attention (heads=16, d_head=64)
  ├─ Residual connection
  ├─ LayerNorm
  ├─ FFN (1024 → 4096 → 1024, GELU)
  └─ Residual connection
       ↓
Output: [B, 16386, 1024]
```

---

## 4. Loss Computation

**파일**: `gslrm/model/gslrm.py`, class `LossComputer`

### 4.1 Total Loss

```python
total_loss = (
    l2_weight(1.0)     * l2_loss           # Pixel-wise MSE
  + perceptual_w(0.5)  * perceptual_loss   # VGG19 multi-scale L1
  + lpips_w(0.05)      * lpips_loss        # VGG perceptual distance
  + ssim_w(0.1)        * ssim_loss         # 1 - SSIM
  + alpha_w(0.0)       * alpha_loss        # MSE(rendered_alpha, gt_mask)
  + bg_w(0.0)          * bg_loss           # Background penalty
)
```

### 4.2 각 Loss 구현

**L2 Loss** (MSE):
```python
def _compute_l2_loss(self, rendered, target, mask=None):
    if mask is not None:  # mask_mode='gt'일 때
        rendered = rendered * mask
        target = target * mask
    return F.mse_loss(rendered, target)
```

**Perceptual Loss** (`gslrm/model/utils_losses.py`):
```python
class PerceptualLoss(nn.Module):
    """VGG19 multi-scale feature matching"""
    def __init__(self):
        # VGG19 layers: conv2, relu3, relu5, relu9, relu14
        self.weights = [1.0, 1/2.6, 1/4.8, 1/3.7, 1/5.6]
        self.raw_weight = 10 / 1.5  # Raw image L1 weight

    def forward(self, pred, target):
        loss = 0
        for layer_feat_pred, layer_feat_target, w in zip(...):
            loss += w * F.l1_loss(layer_feat_pred, layer_feat_target)
        loss += self.raw_weight * F.l1_loss(pred, target)
        return loss
```

**SSIM Loss** (`gslrm/model/utils_losses.py`):
```python
class SsimLoss(nn.Module):
    def __init__(self):
        self.ssim = pytorch_msssim.SSIM(win_size=11, data_range=1.0)

    def forward(self, pred, target):
        return 1 - self.ssim(pred, target)  # 1-SSIM, 작을수록 좋음
```

**Alpha Loss** (`mouse_extensions/model/mask_losses.py`):
```python
def compute_alpha_supervision_loss(rendered_alpha, gt_mask, method='mse'):
    """Rendered alpha와 GT mask의 차이"""
    if method == 'mse':
        return F.mse_loss(rendered_alpha, gt_mask)
    elif method == 'bce':
        return F.binary_cross_entropy(rendered_alpha, gt_mask)
```

### 4.3 Loss Warmup

```python
# 처음 500 step: L2 only (perceptual, LPIPS off)
if step < 500:
    perceptual_loss = 0
    lpips_loss = 0
# 이후: 모든 loss 활성화
```

**Why**: 초기에 perceptual loss를 켜면 gradient 불안정. L2로 대략적 형태를 먼저 학습한 후 perceptual로 세부 디테일 개선.

---

## 5. Validation & Checkpoint

### 5.1 Validation Flow

```python
def run_validation(self):
    self.model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch in self.val_loader:
            result = self.model(batch)

            # Per-sample metrics 계산
            metrics = compute_per_view_metrics(
                result.rendered_images,
                batch['image'][:, :, :3],   # GT RGB
                batch['image'][:, :, 3:4],  # GT alpha (mask)
            )
            all_metrics.append(metrics)

    # 평균 metrics
    avg = {k: np.mean([m[k] for m in all_metrics]) for k in metrics}

    # WandB logging
    wandb.log({
        "val/psnr": avg['psnr'],
        "val/ssim": avg['ssim'],
        "val/lpips": avg['lpips'],
        "val/mask_iou": avg['mask_iou'],
    }, step=self.step)

    return avg
```

### 5.2 Metrics 계산

**파일**: `gslrm/model/utils_metrics.py`

```python
def compute_psnr(pred, target, mask=None):
    """PSNR 계산 (optional foreground masking)"""
    if mask is not None:
        pred = pred * mask
        target = target * mask
    mse = F.mse_loss(pred, target)
    return -10 * torch.log10(mse)

def compute_mask_iou(pred_image, gt_mask, threshold=0.98):
    """Silhouette IoU: predicted mask vs GT mask"""
    # pred_image에서 mask 추출: white background에서 벗어난 픽셀
    pred_mask = (pred_image.mean(dim=-3) < threshold).float()
    intersection = (pred_mask * gt_mask).sum()
    union = (pred_mask + gt_mask).clamp(max=1).sum()
    return intersection / (union + 1e-6)
```

### 5.3 Checkpoint 저장

```python
def save_best_checkpoint(self):
    """Best PSNR 기준 체크포인트 저장"""
    path = self.checkpoint_dir / "best_psnr.pt"
    torch.save({
        'model': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'scheduler': self.scheduler.state_dict(),
        'step': self.step,
        'best_psnr': self.best_psnr,
    }, path)

    # JSON도 함께 저장 (빠른 조회용)
    json_path = self.checkpoint_dir / "best_psnr.json"
    json.dump({
        "value": self.best_psnr,
        "step": self.step,
        "metric": "psnr"
    }, open(json_path, 'w'))
```

---

## 6. Multi-view Diffusion Training — Stage 1 (요약)

> **용어**: 코드 폴더 `mvdiffusion/`과 클래스명은 Era3D에서 상속. Tang et al. "MVDiffusion"과는 **별개**.
> FaceLift Stage 1 = SD2.1-UnCLIP + Era3D RMA.

**파일**: `train_diffusion.py` (~1254줄)

### 6.1 Architecture

```
Input Image → CLIP Encoder → image_embeds
            → VAE Encoder  → conditional_latents

Target Views → VAE Encoder → latents
             → Add noise   → noisy_latents

UNet(noisy_latents, timestep, prompt_embeds, image_embeds, conditional_latents)
  └─ Multi-view attention (6개 뷰 간 cross-attention)
     → noise prediction → MSE loss
```

### 6.2 Training Step

```python
def process_training_batch(batch):
    # 1. Input image → CLIP image embeddings
    image_embeds = clip_encoder(batch['input_image'])

    # 2. All views → VAE encoding
    latents = vae.encode(batch['target_views'])      # [B*6, 4, 64, 64]
    cond_latents = vae.encode(batch['input_image'])   # [B, 4, 64, 64]

    # 3. Add noise (same timestep for all views of same object)
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, 1000, (B,))
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    # 4. Classifier-free guidance dropout (10% probability)
    if random() < 0.1:
        image_embeds = torch.zeros_like(image_embeds)

    # 5. UNet prediction
    model_pred = unet(noisy_latents, timesteps, prompt_embeds,
                      image_embeds, cond_latents)

    # 6. Loss (MSE or v-prediction)
    loss = F.mse_loss(model_pred, noise)  # epsilon prediction
    return loss
```

### 6.3 Key Config Differences

| Config | LR | Steps | Ref View | Sparse Attn | 차이점 |
|--------|:--:|:-----:|:--------:|:-----------:|--------|
| M5t2 (baseline) | 5e-5 | 10K | fixed(0) | ✅ | 기본 |
| M5t2_randref_sparse | 5e-5 | 10K | **random** | ✅ | H5: generalization |
| M5t2_H3_resume_pose | 5e-5 | 10K | random | ✅ | +pose conditioning |

---

## 7. mouse_extensions Integration Points

### 7.1 Patches Applied

**파일**: `mouse_extensions/apply_patches.py`

원본 FaceLift 코드에 최소한의 수정:

| 패치 대상 | 변경 내용 |
|-----------|-----------|
| `gaussians_renderer.py` | `diff_gauss` import fallback 추가 |
| `gslrm.py` | `mouse_extensions` import + alpha loss hook |
| `train_gslrm.py` | Enhanced logging + validation runner |

### 7.2 Key Extension Modules

| 모듈 | 파일 | 역할 |
|------|------|------|
| **Dataset** | `data/mouse_dataset.py` | MouseViewDataset (CH1 참조) |
| **Mask Loss** | `model/mask_losses.py` | 6가지 mask loss mode |
| **Alpha Renderer** | `model/alpha_renderer.py` | Gaussian alpha rendering |
| **Pose Conditioning** | `model/pose_conditioning.py` | 카메라 포즈 인코딩 |
| **Validation** | `validation/validator.py` | PSNR/SSIM/LPIPS + turntable |
| **Fair Eval** | `scripts/eval/fair_comparison.py` | FL vs PS 공정 비교 |
| **DA Datagen** | `scripts/domain_adapt/generate_mvdiff_train_data.py` | MVDiff data 생성 |

---

## Navigation

| Link | Document |
|------|----------|
| ← Environment & Data | [[CH1_ENVIRONMENT_AND_DATA]] |
| ← Master Guide | [[EXPERIMENT_MASTER_GUIDE]] |
| → Experiments & Results | [[CH3_EXPERIMENTS_AND_RESULTS]] |
| Config Guide | [[EXPERIMENT_CONFIG_GUIDE]] |
| Pipeline Architecture | [[theory/PIPELINE_ARCHITECTURE]] |

---

*CH2 GS-LRM Code Flow v1.0 | 2026-02-23*
