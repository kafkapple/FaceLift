# CH3: Experiments & Results

> 모든 실험의 가설, 근거, 설정, 명령어, 코드 흐름, 결과, 후속 실험 도출까지 시간순으로 상세히 설명합니다.
>
> ← [[CH2_GSLRM_CODE_FLOW]] | [[EXPERIMENT_MASTER_GUIDE]] | → 개별 실험 문서 참조

---

## Phase 1: GS-LRM Baseline (E0_1)

### 1.1 가설

> "FaceLift GS-LRM을 M5t2 마우스 데이터로 fine-tuning하면 양질의 3D reconstruction이 가능하다."

### 1.2 근거

- FaceLift 논문: Objaverse 데이터로 pretrained된 GS-LRM을 domain-specific 데이터로 fine-tuning
- GS-LRM은 feed-forward 모델 → per-scene optimization 불필요, inference 속도 빠름

### 1.3 데이터셋

**M5t2**: M5 preprocessing (PP centering + uniform distance normalization) + 80:10:10 temporal split

```yaml
# configs/datasets/M5t2.yaml
data:
  data_root: /home/joon/data/preprocessed/FaceLift_mouse/M5
  train_split: data_mouse_t2_train.txt   # 2880 frames
  val_split: data_mouse_t2_val.txt       # 360 frames
  test_split: data_mouse_t2_test.txt     # 360 frames
```

### 1.4 실행 명령어

```bash
cd /home/joon/dev/FaceLift

# E0_1: 논문 원본 config
export CUDA_VISIBLE_DEVICES=5 && nohup python \
    train_gslrm.py -e configs/experiments/E0_1_facelift.yaml \
    > logs/E0_1_facelift.log 2>&1 &
```

### 1.5 Config (E0_1 — 논문 원본)

```yaml
# configs/experiments/E0_1_facelift.yaml
model:
  num_views: 6
  num_input_views: 4
training:
  dataset:
    random_view_selection: true
  losses:
    mask_mode: none
    alpha_loss_weight: 0.0
    bg_loss_weight: 0.0
```

> **⚠️ Config 전환 주의**: E0_1은 논문 원본 config (`E0_1_facelift.yaml`)를 사용.
> Phase 2 (H4 View Ablation)부터는 통제된 ablation을 위해 `base_uniform_v2.yaml` 기반으로 전환됨.
> E0_1 (22.34 dB) vs H4-4v (21.71 dB)의 ~0.63 dB 차이는 config 차이에 기인하며, 직접 비교 불가.

### 1.6 코드 흐름

```
train_gslrm.py main()
  → load_modular_config(-e E0_1_facelift.yaml)
  → GSLRMTrainer(config)
    → MouseViewDataset: M5/의 2880 train 프레임 로드
    → GSLRM model: ckpt-21125.pt에서 resume
    → 매 step: batch(2) × 4-view input → 16,386 Gaussians → 6-view 렌더링
    → Loss: L2(1.0) + Perceptual(0.5) + SSIM(0.1) + LPIPS(0.05)
    → 매 200 step: validation → best_psnr.pt 저장
    → Early stopping: 10 × 200 = 2000 steps 무개선 시 종료
```

### 1.7 결과

| Metric | Value |
|--------|:-----:|
| Best val/psnr | **22.34 dB** |
| Best step | ~4600 |
| val/mask_iou | ~0.93 |

### 1.8 분석 & 후속

- 22.34 dB는 reasonable하지만 6-view input이면 더 높을 수 있음 → **H4 (View Ablation)** 설계
- Alpha mask supervision이 도움될 수 있음 → **H6** 고려
- SSIM weight 조정 가능 → **H7** 고려

---

## Phase 2: GS-LRM Ablation Studies

### 2.1 H4: View Ablation (입력 뷰 수 변화)

#### 가설
> "입력 뷰 수가 증가하면 3D reconstruction 품질이 단조 증가할 것이다."

#### 근거
- 더 많은 시점 → 더 적은 ambiguity → 더 정확한 3D
- 하지만 diminishing returns 예상 (6v vs 4v 차이 < 2v vs 1v)

#### Config diff (6개 실험)

```yaml
# 1view_v2.yaml
num_input_views: 1

# 2view_v2.yaml
num_input_views: 2

# 3view_v2.yaml
num_input_views: 3

# 4view_v2.yaml (baseline)
num_input_views: 4

# 5view_v2.yaml
num_input_views: 5

# 6view_v2.yaml (upper bound)
num_input_views: 6
```

나머지 설정은 `base_uniform_v2.yaml`과 동일.

#### 실행 명령어

```bash
# GPU 4-7에 분배하여 병렬 실행
export CUDA_VISIBLE_DEVICES=4 && nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/1view_v2.yaml \
    > logs/uniform_1view_v2.log 2>&1 &

export CUDA_VISIBLE_DEVICES=5 && nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/2view_v2.yaml \
    > logs/uniform_2view_v2.log 2>&1 &

# ... 3view, 5view, 6view 동일 패턴
```

#### 코드 흐름 (차이점만)

```python
# MouseViewDataset.__getitem__() 에서:
# num_input_views 값에 따라 input 수 변경
# target은 항상 6-view 전체

# GSLRM.forward() 에서:
# SplitData가 첫 num_input_views를 input으로 분리
# 1-view: [B, 1, 9, 512, 512] → 4096 tokens
# 6-view: [B, 6, 9, 512, 512] → 24576 tokens
# Transformer가 다른 수의 tokens 처리
```

#### 결과

| Views | Best PSNR | IoU | Marginal Gain | 해석 |
|:-----:|:---------:|:---:|:-------------:|------|
| 1 | 10.47 | 0.028 | — | Near collapse (단일 시점으로 3D 불가능) |
| 2 | 15.95 | 0.858 | **+5.48** | **Phase transition**: stereo → 3D 가능 |
| 3 | 18.56 | 0.899 | +2.61 | 120° spread 효과 |
| 4 | 20.66 | 0.926 | +2.10 | Solid baseline |
| 5 | 22.16 | 0.942 | +1.50 | Diminishing returns 시작 |
| 6 | **23.84** | **0.954** | +1.68 | Upper bound |

#### 핵심 발견

- **F5**: 1→2 view = phase transition (+5.48 dB, IoU 0.03→0.86)
- **F2**: View 감소가 E2E gap의 83.5% 차지
- **F4**: Break-even ≥ 3 GT views (18.56 dB vs PS 13.78)
- Marginal gain 곡선은 log-like → 6v 이상은 큰 효과 없을 것

#### 후속 → H6, H7, HP (4-view baseline에서 loss/preprocessing 최적화)

---

### 2.2 H6: Alpha Mask Supervision

#### 가설
> "Rendered alpha를 GT mask로 supervision하면 foreground shape 품질이 개선될 것이다."

#### 근거

| 논문 | 방식 | 효과 |
|------|------|------|
| LGM | MSE alpha loss | "faster convergence of the shape" |
| Pose Splatter | normalized masked L1 | Mouse 데이터 검증 |
| Object-Centric 2DGS | background penalty | foreground 집중 |

#### Config diff

```yaml
# 4view_alpha03_v3.yaml (v3 = bugfix 후)
alpha_loss_weight: 0.3          # 0.0 → 0.3

# 4view_alpha05_v3.yaml
alpha_loss_weight: 0.5

# 4view_alpha10_v3.yaml
alpha_loss_weight: 1.0
```

#### 코드 흐름

```python
# gslrm/model/gslrm.py → LossComputer._compute_loss()
if self.alpha_loss_weight > 0:
    # mouse_extensions/model/mask_losses.py
    alpha_loss = compute_alpha_supervision_loss(
        rendered_alpha,    # [B, V, 1, H, W] - Gaussian splatting output
        gt_mask,           # [B, V, 1, H, W] - from RGBA alpha channel
        method='mse'
    )
    total_loss += alpha_loss_weight * alpha_loss

# compute_alpha_supervision_loss:
def compute_alpha_supervision_loss(rendered_alpha, gt_mask, method='mse'):
    rendered_alpha = rendered_alpha.clamp(0, 1)
    if method == 'mse':
        return F.mse_loss(rendered_alpha, gt_mask)
```

#### 실행 명령어

```bash
export CUDA_VISIBLE_DEVICES=5 && nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/4view_alpha03_v3.yaml \
    > logs/uniform_4view_alpha03_v3.log 2>&1 &
# alpha 0.5, 1.0도 동일 패턴 (GPU 6, 7)
```

#### 결과

| Alpha Weight | Best PSNR | vs Baseline (21.71) |
|:------------:|:---------:|:-------------------:|
| 0.0 (baseline) | **21.71** | — |
| 0.3 | 21.34 | -0.37 |
| 0.5 | 21.20 | -0.51 |
| 1.0 | 20.84 | -0.87 |

#### 결론: **❌ 기각**

Alpha weight 증가에 따른 **monotonic PSNR 하락**. Alpha loss가 L2/perceptual loss와 gradient conflict 발생.

**원인**: Rendered alpha는 Gaussian opacity의 결과물인데, 이를 직접 supervision하면 opacity 분포가 왜곡되어 RGB 렌더링 품질이 저하됨.

→ 상세: [[H6_ALPHA_MASK]]

---

### 2.3 H7: SSIM Weight Ablation

#### 가설
> "SSIM loss weight를 높이면 구조적 보존이 개선되어 reconstruction 품질이 향상될 것이다."

#### 근거
- 학습 중 `train/ssim_loss`가 초기 감소 후 다시 증가 (L2에 밀림)
- 문헌: 3DGS (0.2), Instant-3D (0.5), Splatter Image (~0.5)
- Baseline 0.1은 L2(1.0) 대비 너무 낮을 수 있음

#### Config diff

```yaml
# 4view_ssim03_v2.yaml
ssim_loss_weight: 0.3    # baseline 0.1의 3배

# 4view_ssim05_v2.yaml
ssim_loss_weight: 0.5    # 5배

# 4view_ssim10_v2.yaml
ssim_loss_weight: 1.0    # 10배 (L2와 동일)
```

#### 코드 흐름

```python
# gslrm/model/utils_losses.py → SsimLoss
class SsimLoss(nn.Module):
    def __init__(self):
        self.ssim = pytorch_msssim.SSIM(
            win_size=11,       # 11×11 window
            data_range=1.0,
            size_average=True
        )

    def forward(self, pred, target):
        return 1 - self.ssim(pred, target)

# gslrm/model/gslrm.py → LossComputer
ssim_loss = self.ssim_loss_fn(rendered, target)
total_loss += ssim_loss_weight * ssim_loss
```

#### 결과

| SSIM Weight | Best PSNR | Final PSNR | Status |
|:-----------:|:---------:|:----------:|:------:|
| 0.1 (baseline) | **21.71** | **21.71** | ✅ 최적 |
| 0.3 | 21.10 | 19.69 | ↓ 지속 하락 |
| 0.5 | 21.30 | **10.17** | ☠️ Collapse |
| 1.0 | 21.20 | **4.72** | ☠️ Collapse |

#### 결론: **❌ 기각**

0.5/1.0에서 학습 후반 **mode collapse** (PSNR 급락). SSIM loss의 11×11 window가 고해상도 Gaussian rasterization과 gradient conflict 발생.

→ 상세: [[H7_SSIM_WEIGHT]]

---

### 2.4 HP: Preprocessing Ablation

#### 가설
> "M5 전처리의 각 구성요소 (centering, normalization)가 GS-LRM 학습에 필수적인가?"

#### 근거
- M0 (raw)로 학습 시 발산 관찰 → 전처리가 필수
- 하지만 어떤 전처리 단계가 핵심인지 분리 실험 필요

#### 데이터셋 설계

| Dataset | PP Centering | Distance Norm | 실험 의도 |
|---------|:------------:|:-------------:|-----------|
| M0 | ❌ | ❌ | Control: 전처리 없음 |
| M5_4 | ✅ | ❌ | Centering만 효과 |
| M5_5 | ✅ | ✅ (per-view) | Centering + per-view norm |
| M5 | ✅ | ✅ (uniform) | Full preprocessing |

#### 전처리 실행 (오프라인, 1회)

```bash
# M5_4 생성 (centering only)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M5_4 \
    --input /path/to/raw \
    --output /home/joon/data/preprocessed/FaceLift_mouse/M5_4

# M5_5 생성 (centering + per-view norm)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M5_5 \
    --input /path/to/raw \
    --output /home/joon/data/preprocessed/FaceLift_mouse/M5_5
```

#### Config diff

```yaml
# hp_M0.yaml
data:
  data_root: /home/joon/data/preprocessed/FaceLift_mouse/M0
  train_split: data_mouse_train.txt   # M0 기본 split

# hp_M5_4.yaml
data:
  data_root: /home/joon/data/preprocessed/FaceLift_mouse/M5_4
  train_split: data_mouse_t2_train.txt  # M5t2 split (같은 프레임 범위)

# hp_M5_5.yaml (동일 패턴, M5_5 경로)
```

#### 실행 명령어

```bash
# M0 (발산 예상)
export CUDA_VISIBLE_DEVICES=6 && nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/hp_M0.yaml \
    > logs/hp_M0.log 2>&1 &

# M5_4 (GPU 7)
export CUDA_VISIBLE_DEVICES=7 && nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/hp_M5_4.yaml \
    > logs/hp_M5_4.log 2>&1 &

# M5_5 (GPU 4)
export CUDA_VISIBLE_DEVICES=4 && nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/hp_M5_5.yaml \
    > logs/hp_M5_5.log 2>&1 &
```

#### 결과 (진행중)

| Dataset | Best PSNR | Step | Status |
|---------|:---------:|:----:|:------:|
| M0 (raw) | 발산 | — | ❌ 중단 |
| M5_4 (center) | **21.80** | 9101 | 🔄 학습중 (step ~9700) |
| M5_5 (center+norm) | 21.58 | 3801 | 🔄 학습중 (step ~4400) |
| M5 (full, baseline) | 21.71 | 4600 | ✅ 완료 |

#### 핵심 발견

- **F8**: M0 (raw) → 발산. PP centering 없이 학습 불가능.
- **HP-M5_4 = 21.80**: Centering만으로 baseline(21.71) 돌파! Distance normalization이 필수가 아닐 수 있음.
- M5_5 (per-view norm): 아직 학습 초기, 최종 결과 대기

→ 상세: [[PREPROCESSING_REGISTRY]], [[M5_SERIES_SPEC]]

---

## Phase 3: Multi-view Diffusion Optimization (H5)

> **용어 주의**: 코드 폴더 `mvdiffusion/`과 클래스명은 Era3D 코드에서 상속된 명칭입니다.
> Tang et al.의 "MVDiffusion" 논문과는 **별개**입니다. FaceLift Stage 1 = **SD2.1-UnCLIP + Era3D RMA**.
> 약칭 "MVDiff"는 "multi-view diffusion"의 줄임말로 사용합니다.

### 3.1 배경

GS-LRM이 안정적으로 동작함을 확인한 후, E2E pipeline의 첫 단계인 multi-view diffusion 최적화로 전환.

### 3.2 가설 (H5)
> "Multi-view diffusion (Stage 1)의 생성 품질을 개선하면 E2E reconstruction 품질이 향상될 것이다."

### 3.3 Multi-view Diffusion Config 발전

```
Baseline (M5t2):      fixed ref(0), sparse attn, 10K steps
    │
    ├─ P0: randref_sparse    random ref, sparse attn → generalization 향상
    │   └─ Best MVDiff 이 시점: val PSNR ~24.0
    │
    ├─ P1: resume 20K        P0에서 resume, 20K steps → val 27.70
    │   └─ E2 = 최종 best MVDiff
    │
    ├─ Phase 3 P1: Pose conditioning (spherical)
    │   └─ Concat/add injection → 미미한 효과
    │
    └─ Phase 3 E3: Pose conditioning (extrinsic 6D)
        └─ val 개선되었지만 E2E에 반영 미미
```

### 3.4 핵심 Config: E2 (Best Stage 1)

```yaml
# M5t2_randref_20k_resume.yaml
random_reference_view: true    # 고정 view 0 → 랜덤
use_sparse_attention: true     # Multi-view sparse attention
max_train_steps: 20000         # 10K → 20K (resume)
learning_rate: 1.0e-5          # Resume LR (기본 5e-5의 1/5)
lr_scheduler: piecewise_constant
lr_decay_steps: [10000]        # 10K 시점에 LR decay
pretrained_unet: checkpoints/mvdiffusion/mouse_M5t2_randref_sparse/checkpoint-5000
```

### 3.5 Pose Conditioning 코드

**파일**: `mouse_extensions/model/pose_conditioning.py`

```python
class ExtrinsicPoseEncoder(nn.Module):
    """카메라 extrinsic을 연속 표현으로 인코딩"""

    def __init__(self, d_model=768):
        # 6D rotation (Zhou et al.) + 3D translation = 9D
        self.rotation_encoder = FourierEncoder(6, 128)   # 6D → 128D
        self.translation_encoder = FourierEncoder(3, 64)  # 3D → 64D
        self.projection = nn.Linear(128 + 64, d_model)   # 192D → 768D

    def forward(self, c2w_matrices):
        # c2w: [B, N_views, 4, 4]

        # 6D rotation representation (first 2 columns of R)
        rot_6d = c2w[:, :, :3, :2].reshape(B, N, 6)  # [B, N, 6]

        # Translation
        trans = c2w[:, :, :3, 3]  # [B, N, 3]

        # Fourier encoding + projection
        rot_feat = self.rotation_encoder(rot_6d)     # [B, N, 128]
        trans_feat = self.translation_encoder(trans)  # [B, N, 64]
        pose_embed = self.projection(
            torch.cat([rot_feat, trans_feat], dim=-1)
        )  # [B, N, 768]
        return pose_embed
```

**파일**: `mouse_extensions/model/pose_conditioning_integration.py`

```python
class PoseConditioningInjector:
    """UNet forward에 비침습적 pose 주입"""

    def __init__(self, unet, method='add'):
        self.method = method  # 'add', 'concat', 'replace_last'

    def inject(self, unet_hidden_states, pose_embeddings):
        if self.method == 'add':
            # Pose embedding을 hidden states에 더함
            return unet_hidden_states + pose_embeddings
        elif self.method == 'concat':
            # Extra cross-attention token으로 추가
            return torch.cat([unet_hidden_states, pose_embeddings], dim=1)
```

### 3.6 Multi-view Diffusion 실행 명령어

```bash
# E2: Best Stage 1 (random ref + sparse + resume 20K)
export CUDA_VISIBLE_DEVICES=5 && nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py \
    --config configs/mvdiffusion/M5t2_randref_20k_resume.yaml \
    > logs/mvdiff_M5t2_randref_20k_resume.log 2>&1 &

# H3: E2 + Pose conditioning
export CUDA_VISIBLE_DEVICES=5 && nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py \
    --config configs/mvdiffusion/M5t2_H3_resume_pose.yaml \
    > logs/mvdiff_H3_resume_pose.log 2>&1 &
```

### 3.7 Multi-view Diffusion 결과 요약

| Config | Val PSNR | E2E PSNR | Transfer |
|--------|:--------:|:--------:|:--------:|
| M5t2 (baseline) | 24.0 | 7.93 | — |
| E2 (randref+resume 20K) | **27.70** | **8.44** | +0.51 for +3.7 |
| Phase 3 Pose (extrinsic) | ~28.5 | ~8.5 | marginal |

**핵심 발견 (F3)**: Stage 1 +3.7 dB → E2E +0.51 dB. **Transfer rate ~14%**. 이는 multi-view diffusion 개선의 대부분이 E2E에 전달되지 않음을 의미.

→ 상세: [[hypothesis_roadmap]], [[mvdiffusion_bottleneck_analysis]]

---

## Phase 4: E2E Pipeline & Bottleneck Analysis

### 4.1 E2E Pipeline 코드

**파일**: `mouse_extensions/inference/end_to_end.py`

```python
class EndToEndPipeline:
    """Single image → MVDiff → GS-LRM → 3D Gaussians"""

    def __init__(self, mvdiff_ckpt, gslrm_ckpt):
        self.mvdiff = MVDiffusionPipeline(mvdiff_ckpt)
        self.gslrm = GSLRMPipeline(gslrm_ckpt)

    def run(self, input_image, cameras):
        # Step 1: MVDiffusion inference
        # 1장 → 6장 multi-view 생성
        generated_views = self.mvdiff.generate(
            input_image,
            num_inference_steps=50,
            guidance_scale=3.0,
        )  # [6, 3, 512, 512]

        # Step 2: GS-LRM inference
        # 6장 → 16,386 Gaussians
        gaussians = self.gslrm.predict(
            generated_views, cameras
        )

        return gaussians
```

### 4.2 Bottleneck Analysis

**핵심 질문**: "왜 MVDiff 개선이 E2E에 반영되지 않는가?"

**분석 결과** (→ [[mvdiffusion_bottleneck_analysis]]):

```
GS-LRM trained on GT views (M5)
    ↓ 추론 시 MVDiff-generated views 입력
Distribution Mismatch!
    - GT: sharp, accurate, consistent
    - MVDiff: blurry, color shift, view inconsistency
    ↓
GS-LRM이 out-of-distribution input에 대해 성능 저하
    = 14% transfer rate
```

### 4.3 3가지 가설

| ID | 가설 | 검증 방법 |
|----|------|-----------|
| **H_T1** | Distribution mismatch가 주 원인 | DA1: Stage 1 생성 데이터로 GS-LRM fine-tune |
| **H_T2** | View inconsistency가 주 원인 | DA2: Consistency regularization |
| **H_T3** | Input quality가 주 원인 (MVDiff 한계) | H3: Pose conditioning negative control |

---

## Phase 5: Domain Adaptation (DA1)

### 5.1 가설 (H_T1)
> "GS-LRM을 MVDiff-generated views로 fine-tuning하면 distribution gap이 줄어 E2E 성능이 향상될 것이다."

### 5.2 근거
- GS-LRM은 GT views (sharp, accurate)로만 학습됨
- Inference 시 MVDiff 생성 이미지 (blurry, shifted)를 입력받으면 distribution shift
- Fine-tuning으로 domain gap 축소 가능

### 5.3 3-Step Pipeline

```
Step 1: MVDiff Datagen
  Input:  M5 GT cam_000.png (2880 train frames)
  Model:  E2 checkpoint (checkpoint-20000)
  Output: M5_mvdiff/ (2880 frames × 6 views = 17,280 images)

Step 2: Data Formatting
  M5_mvdiff/{frame}/opencv_cameras.json ← GT에서 복사
  M5_mvdiff/{frame}/images/cam_000~005.png ← MVDiff 생성 RGB

Step 3: GS-LRM Fine-tuning
  Train: M5_mvdiff (MVDiff-generated)
  Val:   M5 GT (실제 성능 측정)
  Resume: M5t2_E0_1_facelift/best_psnr.pt (GT-trained)
```

### 5.4 Step 1: Multi-view Diffusion Data Generation

**스크립트**: `mouse_extensions/scripts/domain_adapt/generate_mvdiff_train_data.py`

```python
"""Multi-view diffusion (Stage 1)으로 GS-LRM 학습 데이터 생성

Usage:
    CUDA_VISIBLE_DEVICES=6 python mouse_extensions/scripts/domain_adapt/\
        generate_mvdiff_train_data.py \
        --checkpoint checkpoints/mvdiffusion/mouse_M5t2_randref_sparse/checkpoint-20000 \
        --data_root /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --split_file configs/splits/data_mouse_t2_train.txt \
        --output_dir /home/joon/data/preprocessed/FaceLift_mouse/M5_mvdiff \
        --num_inference_steps 50 \
        --guidance_scale 3.0 \
        --seed 42
"""

def generate_for_frame(frame_dir, mvdiff_pipeline, output_dir):
    # 1. GT input image 로드 (cam_000.png)
    input_image = load_image(frame_dir / "images/cam_000.png")

    # 2. GT 카메라 파라미터 로드
    cameras = load_cameras(frame_dir / "opencv_cameras.json")

    # 3. Multi-view diffusion inference
    generated_views = mvdiff_pipeline(
        image=input_image,
        num_inference_steps=50,
        guidance_scale=3.0,
        generator=torch.Generator().manual_seed(42)
    )  # [6, 512, 512, 3]

    # 4. 저장 (GT와 동일 구조)
    output_frame = output_dir / frame_dir.name
    output_frame.mkdir(exist_ok=True)
    (output_frame / "images").mkdir(exist_ok=True)

    # 카메라: GT에서 복사 (동일 카메라 배치)
    shutil.copy(frame_dir / "opencv_cameras.json",
                output_frame / "opencv_cameras.json")

    # 이미지: Stage 1 생성 RGB (alpha 없음!)
    for i, view in enumerate(generated_views):
        view.save(output_frame / f"images/cam_{i:03d}.png")
```

### 5.5 Step 2: Auto-Launch Script

**스크립트**: `mouse_extensions/scripts/domain_adapt/launch_da1_finetune.sh`

```bash
#!/bin/bash
# DA1 datagen 완료 대기 → GS-LRM fine-tune 자동 실행

DATA_DIR="/home/joon/data/preprocessed/FaceLift_mouse/M5_mvdiff"
TARGET_FRAMES=2880

while true; do
    CURRENT=$(ls -d "$DATA_DIR"/*/images 2>/dev/null | wc -l)
    echo "$(date): $CURRENT / $TARGET_FRAMES frames"

    if [ "$CURRENT" -ge "$TARGET_FRAMES" ]; then
        echo "Datagen complete! Launching GS-LRM fine-tune..."

        cd /home/joon/dev/FaceLift
        CUDA_VISIBLE_DEVICES=6 python train_gslrm.py \
            -b configs/mouse/uniform/base_uniform_v2.yaml \
            -e configs/mouse/uniform/domain_adapt_E2_v1.yaml

        break
    fi
    sleep 300  # 5분마다 확인
done
```

### 5.6 Step 3: GS-LRM Fine-tune Config

```yaml
# domain_adapt_E2_v1.yaml
data:
  data_root: /home/joon/data/preprocessed/FaceLift_mouse/M5_mvdiff
  train_split: data_mouse_t2_train.txt   # Same frames as M5t2
  val_split: data_mouse_t2_val.txt       # But val uses M5 GT!
  remove_alpha: true                      # MVDiff output은 RGB (alpha 없음)

learning_rate: 1.0e-6
max_fwdbwd_passes: 7500                  # 15K의 절반 (fine-tune이므로)
warmup_steps: 200

# Resume from GT-trained best model
pretrained: /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt
```

### 5.7 코드 흐름 (DA1 차이점)

```python
# MouseViewDataset에서 remove_alpha=true 처리:
def __getitem__(self, idx):
    img = Image.open(path)  # RGB (not RGBA!)
    if self.remove_alpha:
        # Alpha channel이 없으므로 자동 생성
        # White background threshold로 mask 추정
        mask = (img_tensor > 250/255).all(dim=0)
        alpha = (~mask).float()
        img_tensor = torch.cat([img_tensor[:3], alpha.unsqueeze(0)])

# 학습 데이터: MVDiff-generated views (blurry, shifted)
# 검증 데이터: M5 GT views (sharp, accurate)
# → GS-LRM이 blurry input에도 reasonable output을 생성하도록 학습
```

### 5.8 DA1 실행 명령어

```bash
# Step 1: Stage 1 datagen (GPU 6, ~7h)
export CUDA_VISIBLE_DEVICES=6 && nohup python \
    mouse_extensions/scripts/domain_adapt/generate_mvdiff_train_data.py \
    --checkpoint checkpoints/mvdiffusion/mouse_M5t2_randref_sparse/checkpoint-20000 \
    --data_root /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split_file configs/splits/data_mouse_t2_train.txt \
    --output_dir /home/joon/data/preprocessed/FaceLift_mouse/M5_mvdiff \
    > logs/da1_datagen.log 2>&1 &

# Step 2: Auto-launcher (datagen 완료 후 fine-tune 자동 시작)
export CUDA_VISIBLE_DEVICES=6 && nohup bash \
    mouse_extensions/scripts/domain_adapt/launch_da1_finetune.sh \
    > /tmp/da1_launcher.log 2>&1 &
```

### 5.9 DA1 결과 (진행중)

| Metric | Baseline E2E | DA1 | 개선 |
|--------|:----------:|:---:|:----:|
| Best PSNR (val) | 8.44 | **10.08** | **+1.64 dB** |
| Best step | — | 901 | |
| Status | ✅ | 🔄 step ~1450/7500 | |

**성공 기준**: E2E PSNR > 10.0 dB → **H_T1 확인!**

DA1 best PSNR 10.08 at step 901로 이미 threshold 초과. Distribution mismatch가 주 bottleneck임을 확인.

→ 상세: [[domain_adaptation_DA1]]

---

## Phase 5b: H3 (E2+Pose as Negative Control)

### 가설
> "Multi-view diffusion에 pose conditioning을 추가하면 view consistency가 개선되어 E2E 향상될 것이다."

### 역할
H3는 DA1의 **negative control**: "Stage 1만 개선해도 E2E가 오르지 않음"을 확인

### Config

```yaml
# M5t2_H3_resume_pose.yaml
pretrained_unet: checkpoints/mvdiffusion/mouse_M5t2_randref_sparse/checkpoint-20000
pose_conditioning:
  method: extrinsic         # 6D rotation + 3D translation
  integration: add          # Additive injection
  encoder_dim: 768
random_reference_view: true
use_sparse_attention: true
lr_scheduler: cosine        # cosine decay (기존 piecewise와 다름)
max_train_steps: 10000
```

### 현재 상태
- GPU 5, step 4200/10000
- Checkpoints at 2000, 3000, 4000
- 완료 후 E2E eval로 H_T1 negative control 확인 예정

---

## Phase 6: Fair Comparison (FL vs PS)

> **데이터 출처 주의**: PS 논문(Goffinet et al. 2025)은 **자체 녹화한 Duke 데이터**를 사용합니다
> (1536×2048, 30fps, 324K frames). DANNCE/MAMMAL 데이터와 **완전히 별개**입니다.
> 본 비교에서는 PS 코드를 M5 데이터(`m5_baseline_gs`)에 적용하여 동일 데이터 조건에서 비교합니다.
> 따라서 PS 논문의 수치(PSNR 33.5 등)와 본 비교의 수치(PSNR_fg 13.78 등)는 **직접 비교 불가**합니다.

### 6.1 5가지 공정성 이슈

FL과 PS를 직접 비교할 때 발견된 불공정 요소:

| # | Issue | 해결 |
|---|-------|------|
| 1 | PS `paper_standard_evaluation`이 80% 학습 데이터 포함 | Test-only (frames 3240-3599) |
| 2 | 모델 타입 비대칭 (feed-forward vs per-scene) | 명시적 공시 |
| 3 | Mask source 비대칭 (GT alpha vs white-BG extraction) | 통일된 mask protocol |
| 4 | Metric protocol 불일치 | Unified metrics |
| 5 | FL silhouette threshold 민감도 | Fixed threshold |

### 6.2 Fair Evaluation Script

**파일**: `mouse_extensions/scripts/eval/fair_comparison.py`

```python
"""FL vs PS 공정 비교 평가

Usage:
    python mouse_extensions/scripts/eval/fair_comparison.py \
        --render_dir outputs/tier_comparison/gslrm_6view_test/samples \
        --gt_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --test_frames 3240-3599 \
        --output experiments/comparison/tier/gslrm_6view_fair.json
"""

def evaluate_fair(render_dir, gt_dir, frame_range):
    metrics = {}
    for frame_id in frame_range:
        for view_id in range(6):
            # GT 로드 (RGBA)
            gt = load_rgba(gt_dir / f"{frame_id:06d}/images/cam_{view_id:03d}.png")
            gt_rgb = gt[:3]     # RGB
            gt_mask = gt[3:]    # Alpha as mask

            # Render 로드 (RGB)
            pred = load_rgb(render_dir / f"{frame_id}/render_view_{view_id:02d}.png")

            # 1. PSNR (GT-masked foreground)
            fg_pred = pred * gt_mask
            fg_gt = gt_rgb * gt_mask
            psnr_fg = compute_psnr(fg_pred, fg_gt)

            # 2. IoU (silhouette overlap)
            pred_mask = extract_mask(pred, threshold=0.98)
            iou = compute_iou(pred_mask, gt_mask)

            # 3. Coverage (pred mask가 GT mask를 얼마나 커버)
            coverage = (pred_mask * gt_mask).sum() / gt_mask.sum()

            # 4. PSNR intersection (둘 다 foreground인 영역만)
            intersection = pred_mask * gt_mask
            if intersection.sum() > 0:
                psnr_inter = compute_psnr(
                    pred * intersection,
                    gt_rgb * intersection
                )

            metrics[f"{frame_id}_{view_id}"] = {
                'psnr_gt_masked': psnr_fg,
                'iou': iou,
                'coverage': coverage,
                'psnr_intersection': psnr_inter,
            }

    return aggregate(metrics)
```

### 6.3 PS Fair Eval Script (joon 서버)

**파일**: `fair_test_only_eval.py` (joon 서버에 배포)

```python
"""PS test-only evaluation (frames 3240-3599)

PS는 per-scene optimization이므로 별도 렌더링 → eval 필요.
동일한 unified metrics 사용.
"""
# PS renders → fair_comparison과 동일 메트릭 계산
```

### 6.4 최종 비교 결과

| Metric | FL GS-LRM 6v | FL E2E (1v) | PS M5 6v |
|--------|:----------:|:----------:|:--------:|
| PSNR_fg | **23.84** | 8.20 | 13.78 |
| IoU | **0.954** | 0.521 | 0.846 |
| Coverage | ~1.0 | 0.715 | 0.893 |
| PSNR_inter | ~23.84 | — | 20.47 |

**핵심 발견**:
- **F1**: FL GS-LRM 6v >> PS per-scene by +9.62 dB (GT input 기준)
- **F9**: PS coverage 89.3%가 낮은 PSNR_fg 주 원인. PSNR_intersection으로 순수 색상 비교 필요.
- E2E의 8.20 dB는 Stage 1 bottleneck (14% transfer) 때문

→ 상세: [[FL_vs_PS_comparison]], [[evaluation_protocol_v1]]

---

## 9-Experiment Matrix (진행중)

### Protocol A: Temporal (동일 프레임, 다른 시점)

| Views | FL GS-LRM | FL E2E | PS M5 |
|:-----:|:---------:|:------:|:-----:|
| 6v | **23.84** | 8.20 | 13.78 |
| 5v | **22.16** | 8.20 | 🔄 학습중 |
| 4v | **20.66** | 8.20 | ⏳ 대기 |

### Protocol B: Spatial (Novel View Synthesis)

| Views | FL GS-LRM | FL E2E | PS M5 |
|:-----:|:---------:|:------:|:-----:|
| 6v | **16.81** | 9.10 | 13.16 |
| 5v | **16.70** | 8.15 | 🔄 학습중 |
| 4v | **15.50** | 7.14 | ⏳ 대기 |

→ 상세: [[evaluation_protocol_v1]]

---

## Summary: Experiment Dependencies

```
E0_1 (baseline, val PSNR 22.34, 논문 원본 config)
  │
  ├─ H4 (view ablation, uniform config) → F2, F4, F5
  │   └─ 6v: val=24.49/PSNR_fg=23.84, 1→2v phase transition
  │
  ├─ H6 (alpha) → ❌ F6
  │
  ├─ H7 (SSIM) → ❌ F7
  │
  ├─ HP (preprocessing) → F8
  │   └─ M5_4=21.80 beats baseline!
  │
  ├─ H5 (MVDiff optimization)
  │   └─ E2 = best MVDiff (val 27.70)
  │       │
  │       ├─ H3 (E2+Pose) → 🔄 negative control
  │       │
  │       └─ E2E eval → F3 (14% transfer)
  │           │
  │           └─ DA1 (domain adaptation)
  │               └─ 10.08 dB → H_T1 confirmed!
  │
  └─ Fair Comparison (FL vs PS)
      └─ F1, F9 (FL >> PS with GT input)
```

---

## Navigation

| Link | Document |
|------|----------|
| ← Code Flow | [[CH2_GSLRM_CODE_FLOW]] |
| ← Environment | [[CH1_ENVIRONMENT_AND_DATA]] |
| ← Master Guide | [[EXPERIMENT_MASTER_GUIDE]] |
| H4 View Ablation | [[H4_VIEW_ABLATION]] |
| H6 Alpha Mask | [[H6_ALPHA_MASK]] |
| H7 SSIM Weight | [[H7_SSIM_WEIGHT]] |
| MVDiff Bottleneck | [[mvdiffusion_bottleneck_analysis]] |
| DA1 | [[domain_adaptation_DA1]] |
| FL vs PS | [[FL_vs_PS_comparison]] |
| Eval Protocol | [[evaluation_protocol_v1]] |
| Hypothesis Roadmap | [[hypothesis_roadmap]] |

---

*CH3 Experiments & Results v1.0 | 2026-02-23*
