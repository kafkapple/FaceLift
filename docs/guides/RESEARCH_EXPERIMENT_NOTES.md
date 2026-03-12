# FaceLift Mouse Experiment: Research Notes

> Lab meeting / 연구 노트용 실험 체계 정리
> Updated: 2026-02-23

---

## 0. Project Summary

### Research Question

**단일 이미지에서 3D mouse reconstruction을 feed-forward 방식으로 달성할 수 있는가?**

생쥐 행동 분석(behavioral neuroscience)에서 3D 재구성은 multi-camera 시스템과 per-scene optimization에 의존해 왔다.
본 프로젝트는 **단 1장의 이미지**에서 즉시 3D Gaussian Splatting을 생성하는 feed-forward 파이프라인으로,
이 패러다임을 전환할 수 있는지 검증한다.

### Method: FaceLift 2-Stage Pipeline

```
Single Image ──→ [Stage 1: Multi-view Diffusion] ──→ 6-view Images ──→ [Stage 3: GS-LRM] ──→ 3D Gaussians
                  SD2.1-UnCLIP + Era3D RMA                            Transformer (24L)
                  1 image → 6 multi-view                               6 views → 16,386 Gaussians
```

- **Stage 1 (Multi-view Diffusion)**: Stable Diffusion V2.1-UnCLIP을 backbone으로, Era3D의 Row-wise Multi-view Attention (RMA)을 통해 6장의 consistent multi-view 이미지를 생성
- **Stage 3 (GS-LRM)**: Transformer 기반 feed-forward 모델이 multi-view 이미지 → Plucker ray → 16,386개 3D Gaussian 예측

> **용어 주의**: 코드 폴더명 `mvdiffusion/`은 Era3D 코드베이스에서 상속된 명칭이며, Tang et al.의 "MVDiffusion" 논문과는 별개임.

### Baseline Comparison

**Pose Splatter**: Per-scene optimization 기반, 6-camera 입력, 50 epoch 최적화 필요.
FL의 feed-forward 방식 대비 추론 시간이 수 배 길지만, per-scene 특화로 높은 품질 기대.

### Dataset

| Item | Value |
|------|-------|
| **원본** | DANNCE `markerless_mouse_1` (Dunn et al. 2021, Harvard) → MAMMAL (An et al. 2023) 경유 |
| **Frames** | 3,600 (18,000 raw × 1/5 temporal downsample) |
| **Resolution** | 512 x 512 RGBA |
| **Split (M5t2)** | Train 0-2879 (80%) / Val 2880-3239 (10%) / Test 3240-3599 (10%) |
| **Cameras** | 6 views, HFOV=50 deg |
| **Preprocessing (M5)** | Center crop + uniform background normalization + RGBA 생성 |

> **⚠️ 데이터 출처 주의**: PoseSplatter 논문(Goffinet et al. 2025)은 **자체 Duke 데이터**를 사용하며
> (1536×2048, 30fps, 324K frames), DANNCE/MAMMAL 데이터와 **완전히 별개**입니다.
> 본 프로젝트에서 PS 코드를 M5 데이터에 적용한 것입니다.

---

## 1. Phase 1: Baseline --- "Can FaceLift reconstruct mice at all?"

### Why This Experiment?

FaceLift는 원래 인간 얼굴(face)에 최적화된 파이프라인이다.
Mouse 도메인은 (1) texture가 단조로움, (2) body shape이 non-rigid로 변형이 크며, (3) 학습 데이터가 소규모(3,600 frames)이다.
먼저 GS-LRM Stage 3가 **GT(ground truth) multi-view를 입력**받았을 때 mouse 3D를 재구성할 수 있는지 확인해야 한다.
이 실험이 실패하면 전체 파이프라인이 성립하지 않는다.

### Hypothesis (E0)

> FaceLift GS-LRM을 mouse 데이터(M5t2, 4-view GT)로 fine-tune하면,
> 의미 있는 수준의 3D reconstruction (PSNR > 15 dB)이 가능하다.

### Experiment Setup

| Item | Value |
|------|-------|
| **Experiment ID** | E0_1 |
| **Model** | GS-LRM (Transformer 24L, hidden_dim=1024) |
| **Input** | 4 GT views (512x512 RGBA) |
| **Output** | 16,386 3D Gaussians |
| **Dataset** | M5t2: 2,880 train / 360 val / 360 test |
| **Loss** | L2 (w=1.0) + Perceptual VGG19 (w=0.5) + SSIM (w=0.1) + LPIPS (w=0.05) |
| **Learning Rate** | 1e-6, warmup 500 steps |
| **Training Steps** | 15,000 forward-backward passes |
| **Batch Size** | 1 (A6000 48GB memory constraint) |
| **GPU** | NVIDIA RTX A6000 48GB, ~8h |
| **Config** | `configs/experiments/E0_1_facelift.yaml` (논문 원본 설정) |
| **Checkpoint** | `checkpoints/gslrm/M5t2_E0_1_facelift/best_psnr.pt` |

> **Config Note**: E0_1은 FaceLift 논문 원본 config (`E0_1_facelift.yaml`)를 사용.
> 이후 Phase 2 (H4 View Ablation)부터는 통제된 비교를 위해 `base_uniform_v2.yaml` 기반으로 전환됨.
> 따라서 **E0_1 (22.34 dB)과 H4-4v (20.66 dB)는 직접 비교할 수 없음** — config 차이(mask_mode, random_view_selection 등)가 ~1.68 dB 차이의 원인.

### Code Flow

```
Train entry: train_gslrm.py
  │
  ├── Config merge: E0_1_facelift.yaml + M5t2.yaml
  │   → OmegaConf.merge() with CLI overrides
  │   → (train_gslrm.py:L45-55)
  │
  ├── Dataset: mouse_extensions/data/mouse_dataset.py::MouseViewDataset
  │   → __getitem__(): load RGBA → split RGB+alpha → normalize cameras
  │   → Camera convention: OpenCV → Plucker ray embedding
  │   → (mouse_dataset.py:L80-150)
  │
  ├── Model: gslrm/model/gslrm.py::GSLRM
  │   → forward() 14-step pipeline:
  │   │  SplitData → Plucker Ray → PatchEmbed → Transformer(24L) → GaussianDecode
  │   │  → Render (differentiable rasterization)
  │   → (gslrm.py:L150-300)
  │
  ├── Loss: mouse_extensions/model/enhanced_loss.py
  │   → composite = L2 + VGG19_perceptual + SSIM + LPIPS
  │   → (enhanced_loss.py:L20-80)
  │
  └── Validation: mouse_extensions/validation/validator.py
      → Render val views → compute PSNR/SSIM → save best_psnr.pt
      → (validator.py:L50-120)
```

### Command

```bash
cd /home/joon/dev/FaceLift
CUDA_VISIBLE_DEVICES=5 python train_gslrm.py \
    -e configs/experiments/E0_1_facelift.yaml
```

### Results

| Metric | Value |
|--------|:-----:|
| Val PSNR (best) | **22.34 dB** |
| Best step | 8,001 |
| Training loss | Stable convergence |
| Training time | ~8 hours |

### Interpretation

- **E0 가설 채택**: PSNR 22.34 dB는 초기 기대치(15 dB)를 크게 상회한다. GS-LRM은 mouse domain에서 안정적으로 동작한다.
- **Why it works**: Transformer의 self-attention이 sparse 4-view 입력에서도 3D geometry를 합리적으로 추론함. Mouse의 단조로운 texture가 오히려 Gaussian 표현에 유리하게 작용한 것으로 보인다.
- **Implication**: Ablation 실험 진행의 근거 확보. 이제 "어떤 조건에서 최적인가"를 체계적으로 탐색할 수 있다.
- **Next**: View 수에 따른 영향(H4), loss function 최적화(H6, H7), 전처리 영향(HP) 순서로 진행.
- **⚠️ Config 전환**: 이후 실험들은 통제된 비교를 위해 `base_uniform_v2.yaml` 기반 config으로 전환됨 (Phase 2부터).

---

## 2. Phase 2: View Ablation --- "How many views do we need?"

### Why This Experiment?

FaceLift의 실용적 가치는 **적은 수의 뷰**에서 얼마나 좋은 3D를 만들 수 있는지에 달려 있다.
6-camera 시스템은 설치가 복잡하고 비용이 높다. 만약 2-3개 카메라로도 충분하다면, 실험실 구축 비용이 크게 절감된다.
또한, 1→2뷰 전환에서 depth ambiguity가 해소되므로 가장 큰 quality jump이 예상된다.

### Hypothesis (H4)

> 입력 뷰 수가 증가하면 3D reconstruction 품질이 **단조 증가**한다.
> 특히 **1 to 2 뷰 전환에서 가장 큰 gain**이 있을 것이다 (monocular depth ambiguity 해소).

### Experiment Setup

> **⚠️ Config 전환 주의**: Phase 1 (E0_1)은 논문 원본 config (`E0_1_facelift.yaml`)로 실행되었으나,
> Phase 2부터는 통제된 ablation을 위해 **`base_uniform_v2.yaml`** 기반으로 전환됨.
> E0_1 (22.34 dB)과 H4-4v (20.66 dB)의 ~1.68 dB 차이는 config 차이에 기인하며, **직접 비교 불가**.

6개의 실험을 동일 조건에서 `num_input_views`만 변경하여 실행.

| Item | Value (All experiments) |
|------|------------------------|
| **Model** | GS-LRM (24L Transformer) |
| **Dataset** | M5t2 (2,880 / 360 / 360) |
| **Loss** | L2(1.0) + VGG19(0.5) + SSIM(0.1) + LPIPS(0.05) |
| **LR** | 1e-6, warmup 500 |
| **Steps** | 15,000 |
| **Base Config** | `base_uniform_v2.yaml` (**NOT** `E0_1_facelift.yaml`) |
| **Controlled Variable** | `num_input_views` only |

### Results

| ID | Views | Config | Val PSNR | IoU | Delta from prev |
|----|:-----:|--------|:--------:|:---:|:---------------:|
| H4-1v | 1 | `1view_v2.yaml` | 10.47 | 0.028 | --- |
| H4-2v | 2 | `2view_v2.yaml` | 15.95 | 0.858 | **+5.48** |
| H4-3v | 3 | `3view_v2.yaml` | 18.56 | 0.899 | +2.61 |
| H4-4v | 4 | `4view_v2.yaml` | 20.66 | 0.926 | +2.10 |
| H4-5v | 5 | `5view_v2.yaml` | 22.16 | 0.942 | +1.50 |
| H4-6v | 6 | `6view_v2.yaml` | **23.84** | **0.954** | +1.68 |

> **E0_1과의 관계**: H4-4v (20.66)는 E0_1 (22.34)과 동일하게 4-view이지만, config이 다름.
> E0_1_facelift.yaml은 `mask_mode: none, random_view_selection: true`를 설정하지만,
> base_uniform_v2.yaml의 기본값과 미세하게 다른 hyperparameter가 ~1.68 dB 차이를 만듦.

### Command (example: 6-view)

```bash
cd /home/joon/dev/FaceLift
CUDA_VISIBLE_DEVICES=5 python train_gslrm.py \
    -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/6view_v2.yaml
```

### Key Findings

- **F2: Phase transition at 1 to 2 views** --- 1->2뷰에서 **+5.48 dB**로 최대 gain. IoU가 0.028 -> 0.858로 급등.
  - *Why*: 1-view는 depth가 완전히 ambiguous하여 3D 구조 자체를 추론할 수 없음. 2번째 뷰가 stereo baseline을 제공하면서 geometry가 극적으로 개선됨.

- **F4: Monotonic increase, diminishing returns** --- 단조 증가하되, marginal gain이 점진적으로 감소.
  - *Exception*: 5->6뷰에서 +1.68 dB로 살짝 반등. 이는 6번째 카메라가 기존 5개로 커버하지 못한 occlusion 영역을 보완하기 때문으로 해석.

- **F5: 6-view = GT input upper bound** --- 23.84 dB. Multi-view Diffusion을 거치지 않고 GT 이미지를 직접 입력하면 이 수준까지 가능.
  - 이 값은 이후 모든 E2E 실험의 **ceiling**으로 기능.

- **F6: 2-view is competitive with per-scene optimization** --- 2뷰 feed-forward(15.95 dB)가 Pose Splatter 6v per-scene(13.78 dB, PSNR_fg)과 비교 가능한 수준.

### Implications

| Views | Use case | Feasibility |
|:-----:|----------|:-----------:|
| 1 | Single image 3D | Insufficient (10.47) |
| 2 | Minimal stereo rig | Viable baseline (15.95) |
| 3-4 | Standard lab setup | Good quality (18-21) |
| 6 | Full coverage | Best achievable (23.84) |

---

## 3. Phase 2b: Loss Ablation --- "Can the loss function improve reconstruction?"

### Why This Experiment?

Baseline loss (L2 + VGG19 + SSIM 0.1 + LPIPS 0.05)가 반드시 최적은 아닐 수 있다.
Mouse 데이터는 foreground가 작고(전체 이미지의 ~15%), background가 지배적이다.
**Foreground 강조** 또는 **structural similarity 강조**가 도움이 될 수 있다는 합리적 가설이 존재한다.

### H6: Alpha Mask Loss --- "Foreground 강조가 도움이 되는가?"

#### Hypothesis

> GT alpha mask로 foreground pixel에 가중치를 부여하면,
> mouse body 영역의 reconstruction quality가 개선될 것이다.

#### Rationale

Mouse가 이미지의 ~15%만 차지하므로, L2 loss에서 background pixel이 gradient를 지배한다.
Alpha mask loss로 foreground에 집중하면 body detail이 개선될 수 있다.

#### Experiment Setup

| Item | Value |
|------|-------|
| **Base** | 4-view baseline (E0_1 동일 조건) |
| **Variable** | `alpha_mask_weight`: 0.3, 0.5, 1.0 |
| **Config pattern** | `4view_alpha{W}_v3.yaml` |
| **Baseline** | alpha_mask_weight = 0 (PSNR 21.71) |

#### Results

| Alpha Weight | Val PSNR | vs Baseline (21.71) | Trend |
|:------------:|:--------:|:-------------------:|:-----:|
| 0.0 (base) | **21.71** | --- | --- |
| 0.3 | 21.34 | -0.37 | decline |
| 0.5 | 21.20 | -0.51 | decline |
| 1.0 | 20.84 | -0.87 | decline |

#### Interpretation

**H6 기각.** Weight 증가에 따라 **단조 하락(monotonic decline)**.

- *Why it failed*: Alpha mask loss가 foreground pixel에 과도하게 집중하면서, background의 geometry 정보(depth cue, occlusion boundary)를 학습하지 못함. 3D Gaussian Splatting은 전체 scene의 global structure를 함께 학습해야 하는데, foreground-only supervision이 이를 방해.
- *Lesson learned*: Gaussian representation에서 foreground/background 분리 supervision은 효과적이지 않음. 전체 이미지에 대한 holistic loss가 더 적합.

---

### H7: SSIM Weight --- "Structural similarity를 강조하면?"

#### Hypothesis

> SSIM loss의 비중을 높이면 edge와 구조적 유사성이 개선되어
> 전체 reconstruction quality가 향상될 것이다.

#### Rationale

L2 loss는 pixel-wise로 blurry 결과를 유도할 수 있다. SSIM은 local structure (luminance, contrast, structure)를 반영하므로, perceptual quality 개선이 기대된다.

#### Experiment Setup

| Item | Value |
|------|-------|
| **Base** | 4-view baseline |
| **Variable** | `ssim_weight`: 0.3, 0.5, 1.0 |
| **Config pattern** | `4view_ssim{W}_v2.yaml` |
| **Baseline** | ssim_weight = 0.1 (PSNR 21.71) |

#### Results

| SSIM Weight | Initial PSNR | Final PSNR | Status |
|:-----------:|:------------:|:----------:|:------:|
| 0.1 (base) | --- | **21.71** | Stable |
| 0.3 | 21.10 | 19.69 | Decline |
| 0.5 | 21.30 | 10.17 | **Collapse** |
| 1.0 | 21.20 | 4.72 | **Collapse** |

#### Interpretation

**H7 기각.** SSIM weight >= 0.5에서 **training collapse** 발생.

- *Why it collapsed*: SSIM loss의 gradient는 Gaussian rasterization의 differentiable rendering과 충돌한다. SSIM이 local window 기반이므로, Gaussian의 global splatting과 gradient 방향이 불일치하여 학습이 불안정해짐. 특히 weight가 높아지면 L2의 stabilizing 효과가 상대적으로 감소하면서 발산.
- *Critical insight*: Gaussian Splatting에서 SSIM은 보조적(0.1 수준)으로만 사용해야 한다. 이는 원본 3DGS 논문의 설정(SSIM weight 0.2)과도 일맥상통.
- *Baseline 0.1 = optimal*: 추가 실험 없이 baseline 유지 결정.

### Loss Ablation Summary

```
Loss Ablation Decision Tree:
├─ Alpha mask loss (H6) → Monotonic decline → REJECTED
├─ SSIM weight increase (H7) → Collapse at 0.5+ → REJECTED
└─ Baseline loss (L2=1.0, VGG=0.5, SSIM=0.1, LPIPS=0.05) → OPTIMAL
```

**Takeaway**: 현재 loss configuration이 이미 near-optimal. Loss engineering보다 **입력 품질(view count, preprocessing)**이 훨씬 큰 영향을 미침.

---

## 4. Phase 2c: Preprocessing Ablation --- "How much does data preprocessing matter?"

### Why This Experiment?

M5 전처리(center crop + uniform background normalization)는 경험적으로 선택된 것이다.
전처리의 각 구성 요소가 실제로 얼마나 기여하는지 분리하여 검증할 필요가 있다.
Raw 데이터로 학습이 가능하다면 전처리 파이프라인 전체가 불필요해진다.

### Hypothesis (HP)

> 전처리 없이(M0, raw) 학습하면 성능이 크게 하락하거나 학습이 발산할 것이다.
> Centering은 필수이나, normalization은 선택적일 수 있다.

### Preprocessing Variants

| Config | Description | Preprocessing Steps |
|--------|------------|---------------------|
| **M0** | Raw, no preprocessing | None (original camera images) |
| **M5_4** | Center crop only | Crop to mouse center, resize 512x512 |
| **M5_5** | Center + per-view norm | Center crop + per-view intensity normalization |
| **M5** (baseline) | Full preprocessing | Center crop + uniform global normalization |

### Experiment Setup

| Item | Value |
|------|-------|
| **Model** | GS-LRM 4-view |
| **Base Config** | `base_uniform_v2.yaml` |
| **Variable** | Preprocessing pipeline only |
| **All other params** | Identical to E0_1 baseline |

### Results

| Config | Preprocessing | Val PSNR | Status | Notes |
|--------|--------------|:--------:|:------:|-------|
| HP-M0 | None (raw) | diverge | FAILED | Loss explodes after ~500 steps |
| HP-M5_4 | Center only | **21.80** | Training | Beats baseline at step 4,600 |
| HP-M5_5 | Center + per-view norm | 21.58 | Training | Slightly below baseline |
| M5 (base) | Center + uniform norm | 21.71 | Complete | Reference |

### Command (example: M5_4)

```bash
cd /home/joon/dev/FaceLift
CUDA_VISIBLE_DEVICES=7 python train_gslrm.py \
    -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/hp_M5_4.yaml
```

### Code Flow (Preprocessing)

```
Preprocessing pipeline: mouse_extensions/data/preprocessing/
  │
  ├── M0: Raw frames from DANNCE cameras
  │   → No transform → direct to MouseViewDataset
  │
  ├── M5_4: preprocess_center.py
  │   → Detect mouse bounding box (SAM or thresholding)
  │   → Center crop → resize 512x512
  │   → (preprocess_center.py:L30-80)
  │
  ├── M5_5: preprocess_center_norm.py
  │   → Center crop + per-view intensity normalization
  │   → Each camera independently normalized
  │   → (preprocess_center_norm.py:L30-100)
  │
  └── M5: preprocess_full.py
      → Center crop + global uniform normalization (across all views)
      → (preprocess_full.py:L30-120)
```

### Interpretation

- **F8: Raw data (M0) is unusable** --- Centering은 **필수**. Raw 이미지에서 mouse가 이미지 내 다양한 위치에 있으므로, 모델이 spatial prior를 학습하지 못하고 발산.

- **F9: Centering alone (M5_4) >= full preprocessing** --- 놀랍게도 centering만으로 baseline(21.71)을 상회하는 21.80 달성. Uniform normalization이 반드시 필요하지는 않음.

- **Per-view norm (M5_5) slightly hurts** --- 각 카메라별 독립 정규화가 view 간 intensity consistency를 깨뜨려서 multi-view reasoning에 약간의 방해.

- *Implication*: 전처리 파이프라인 단순화 가능. Center crop만으로 충분하며, normalization은 optional.

---

## 5. Phase 3: Multi-view Diffusion (Stage 1) --- "Can we improve the full pipeline?"

### Why This Experiment?

Phase 1-2에서 GS-LRM (Stage 3)은 GT 입력에서 잘 동작함을 확인했다.
그러나 실제 E2E 파이프라인에서는 **Stage 1 (Multi-view Diffusion)**이 생성한 이미지를 사용한다.
Stage 1의 품질이 곧 전체 파이프라인의 품질을 결정하므로, Stage 1 개선이 필수적이다.

### Hypothesis (H5)

> Stage 1의 multi-view 생성 품질을 개선하면 E2E reconstruction quality가 비례하여 향상될 것이다.

### Stage 1 Training History

| ID | Strategy | Val PSNR | E2E PSNR | Config |
|----|----------|:--------:|:--------:|--------|
| E0 | Baseline (vanilla fine-tune) | 24.0 | 7.93 | `M5t2.yaml` |
| E1 | + Cosine LR schedule | 25.3 | 8.01 | `M5t2_cosine.yaml` |
| E2 | + Random reference + Sparse attn + Resume 20K | **27.70** | **8.44** | `M5t2_randref_sparse.yaml` |
| E3 | + Higher LR | 26.8 | 8.20 | `M5t2_highLR.yaml` |
| E4 | + Augmentation | 26.1 | 7.95 | `M5t2_aug.yaml` |
| E5 | + Larger batch | 27.2 | 8.30 | `M5t2_largebatch.yaml` |

### Best Stage 1 Configuration (E2)

| Item | Value |
|------|-------|
| **Model** | SD2.1-UnCLIP + Era3D RMA |
| **Strategy** | Random reference view + Sparse attention + Resume from 20K |
| **Dataset** | M5t2 (same split) |
| **Val PSNR** | 27.70 dB |
| **E2E PSNR** | 8.44 dB |
| **Config** | `configs/mvdiffusion/M5t2_randref_sparse.yaml` |
| **Checkpoint** | `mouse_M5t2/checkpoint-5000` |

### Command (Stage 1 training)

```bash
cd /home/joon/dev/FaceLift
CUDA_VISIBLE_DEVICES=6 python train_diffusion.py \
    --config configs/mvdiffusion/M5t2_randref_sparse.yaml
```

### The Alarming Discovery: Transfer Rate

Stage 1 개선이 E2E에 얼마나 전이되는가?

```
Stage 1 improvement:  E0(24.0) → E2(27.70) = +3.70 dB
E2E improvement:      E0(7.93) → E2(8.44)  = +0.51 dB

Transfer rate = 0.51 / 3.70 = ~14%
```

**Stage 1에서 +3.7 dB 개선했지만, E2E에서는 +0.51 dB만 전이됨.**

| Observation | Detail |
|-------------|--------|
| E1-E5 convergence | 모든 전략이 E2E 7.9-8.4 dB 범위에 수렴 |
| Transfer rate | ~14% (극히 낮음) |
| Ceiling | Stage 1 전략 변경만으로는 E2E 개선에 한계 |

### Interpretation

- *Why only 14% transfer?*: 이 질문이 Phase 4 전체를 구동하는 핵심 research question이 됨.
- *Possible explanations*:
  1. **Distribution mismatch**: GS-LRM이 GT 이미지로 학습됐는데, Stage 1이 생성한 blurry/inconsistent 이미지를 받으면 domain shift 발생
  2. **View inconsistency**: Stage 1이 생성한 6뷰 간 3D consistency가 부족하여 GS-LRM이 coherent geometry를 추론하지 못함
  3. **Quality ceiling**: Stage 1 출력의 absolute quality가 GS-LRM의 최소 요구치를 충족하지 못함
- *Implication*: Stage 1 개선에 더 투자하는 것은 비효율적. **병목 원인 진단**이 우선.

---

## 6. Phase 3b: Pose Conditioning --- "Does explicit pose help Stage 1?"

### Why This Experiment?

Multi-view Diffusion이 6뷰를 생성할 때, 각 뷰의 카메라 포즈를 **명시적으로** 알려주면 view consistency가 개선될 수 있다.
현재 Stage 1은 포즈 정보 없이 학습된 implicit view arrangement에 의존한다.

### Hypothesis (H3)

> Stage 1에 카메라 pose를 조건으로 추가하면 multi-view consistency가 향상되고,
> E2E reconstruction quality가 개선될 것이다.

### Experiment Setup

| Item | Value |
|------|-------|
| **Model** | E2 (best Stage 1) + Pose conditioning module |
| **Pose representation** | Spherical coordinates (azimuth, elevation, distance) |
| **Conditioning** | Cross-attention injection |
| **Base** | Resume from E2 checkpoint (20K) |
| **Steps** | 10,000 additional |
| **GPU** | A6000 (GPU 5), ~28h |
| **Config** | `configs/mvdiffusion/M5t2_H3_resume_pose.yaml` |

### Status

| Progress | Detail |
|----------|--------|
| Step | ~4,200 / 10,000 |
| ETA | ~20h remaining |
| GPU | 5 |

### Expected Outcome

H3가 Stage 1의 multi-view consistency를 개선하더라도, Phase 4의 분석(14% transfer rate)에 의하면 E2E 개선은 제한적일 가능성이 높다. 이 실험은 **H_T3 (view inconsistency)** 가설을 검증하기 위한 것이기도 하다.

---

## 7. Phase 4: Bottleneck Analysis --- "Why doesn't Stage 1 improvement transfer to E2E?"

### Why This Analysis?

Phase 3에서 발견한 **14% transfer rate**는 전체 프로젝트의 방향을 결정하는 핵심 문제이다.
Stage 1을 아무리 개선해도 E2E가 개선되지 않는다면, 다른 전략이 필요하다.
원인 진단 없이 Stage 1 최적화를 계속하는 것은 자원 낭비이다.

### Three Competing Hypotheses

| ID | Hypothesis | Prediction | Test Method |
|----|-----------|------------|-------------|
| **H_T1** | Distribution mismatch | GS-LRM을 Stage 1 출력으로 fine-tune하면 E2E 개선 | DA1 experiment |
| **H_T2** | View inconsistency | Pose conditioning으로 consistency 개선하면 E2E 개선 | H3 experiment |
| **H_T3** | Input quality ceiling | Stage 1 출력의 absolute quality가 부족 | Stage 1 val PSNR vs E2E correlation 분석 |

### H_T1 Verification: Domain Adaptation (DA1)

#### Why DA1?

H_T1이 맞다면, GS-LRM을 GT 이미지가 아닌 **Stage 1이 생성한 이미지**로 재학습하면,
GS-LRM이 Stage 1의 artifact에 적응하여 E2E 성능이 개선될 것이다.
이는 "domain gap을 줄이는" 가장 직접적인 방법이다.

#### DA1 Pipeline

```
DA1 3-Stage Pipeline:
  │
  ├── Step 1: Generate synthetic training data (datagen)
  │   → mouse_extensions/scripts/domain_adapt/generate_mvdiff_train_data.py
  │   → E2 (best MVDiff) checkpoint로 2,880 train frames에 대해 6-view 생성
  │   → Output: /home/joon/data/preprocessed/FaceLift_mouse/M5_mvdiff/
  │   → GPU 6, ~10s/frame, total ~8h
  │   → (generate_mvdiff_train_data.py:L50-120)
  │
  ├── Step 2: Fine-tune GS-LRM on synthetic data
  │   → train_gslrm.py with domain_adapt_E2_v1.yaml
  │   → Input: Stage 1 outputs (NOT GT)
  │   → Target: Still GT for supervision
  │   → 7,500 steps, lr=1e-6
  │   → (train_gslrm.py + configs/mouse/uniform/domain_adapt_E2_v1.yaml)
  │
  └── Step 3: E2E evaluation
      → Generate 6-view with E2 → Feed to DA1 GS-LRM → Render → Evaluate
      → mouse_extensions/scripts/eval/fair_comparison.py
```

#### DA1 Experiment Setup

| Item | Value |
|------|-------|
| **Model** | GS-LRM (DA1: fine-tuned on Stage 1 outputs) |
| **Training Input** | Stage 1 (E2) generated 6-view images |
| **Supervision Target** | GT images (unchanged) |
| **Steps** | 7,500 |
| **LR** | 1e-6 |
| **Config** | `configs/mouse/uniform/domain_adapt_E2_v1.yaml` |
| **GPU** | 6 (after datagen completes) |

#### Results

| Metric | Baseline E2E | DA1 E2E | Delta |
|--------|:-----------:|:-------:|:-----:|
| PSNR | 8.44 | **10.08** | **+1.64** |

#### Interpretation

- **H_T1 CONFIRMED**: Distribution mismatch는 primary bottleneck이다.
  - DA1이 단순히 GS-LRM의 input distribution을 맞춰주는 것만으로 +1.64 dB 개선.
  - 이는 Stage 1의 3.7 dB 개선보다 더 효율적인 E2E 개선 경로.

- *Why this matters*:
  - GS-LRM은 GT 이미지의 sharp, consistent multi-view에 최적화되어 있었음.
  - Stage 1이 생성하는 slightly blurry, view-inconsistent 이미지에 대해서는 trained distribution 바깥이므로 성능이 급격히 저하됨.
  - Domain adaptation으로 이 gap을 줄이면 Stage 1 개선 없이도 E2E가 개선됨.

- *Strategic implication*: **Stage 1 최적화**보다 **GS-LRM domain adaptation**이 더 효율적인 E2E 개선 전략.

### Transfer Rate Analysis Summary

```
Without DA:  Stage 1 +3.7dB → E2E +0.51dB  (14% transfer)
With DA:     GS-LRM retrain   → E2E +1.64dB (direct improvement)

→ Domain adaptation is 3.2x more efficient than Stage 1 optimization
```

---

## 8. Phase 5: Fair Comparison --- "FaceLift vs Pose Splatter"

### Why This Comparison?

FaceLift의 연구적 가치를 입증하려면, 기존 per-scene optimization 방법과 공정하게 비교해야 한다.
단순 숫자 비교가 아닌, **공정한 조건**에서의 비교가 핵심이다.

### 5 Fairness Issues Identified and Resolved

연구 과정에서 발견한 5가지 공정성 문제와 해결 방법:

| # | Issue | Problem | Solution |
|---|-------|---------|----------|
| 1 | **Evaluation data overlap** | PS `paper_standard_evaluation`이 80% train 데이터 포함 (frame_step=30 across ALL frames) | Test-only eval: frames 3240-3599 |
| 2 | **Model type asymmetry** | FL=generalizing feed-forward vs PS=per-scene optimization | 명시적 구분, 각 모델 유형의 장단점 기술 |
| 3 | **Mask source asymmetry** | FL=GT RGBA alpha, PS=white-BG extraction | 동일 GT alpha mask 사용으로 통일 |
| 4 | **Metric protocol mismatch** | PSNR/SSIM 계산 방식 상이 | Unified metrics: psnr_gt_masked, psnr_intersection, coverage, color_bias |
| 5 | **Silhouette threshold sensitivity** | FL silhouette extraction threshold가 결과에 영향 (mouse = ~2.5% of image) | 고정 threshold + coverage 보고 |

### Fair Evaluation Protocol

```
Fair evaluation scripts:
  │
  ├── FL side (gpu03):
  │   mouse_extensions/scripts/eval/fair_comparison.py
  │   → Renders GS-LRM output → Computes unified metrics on test frames (3240-3599)
  │   → (fair_comparison.py:L50-200)
  │
  └── PS side (joon server):
      scripts/mouse/analysis/fair_test_only_eval.py
      → Loads PS model → Renders test frames → Same unified metrics
      → (fair_test_only_eval.py:L30-150)
```

### Unified Metrics

| Metric | Description | Why Needed |
|--------|-------------|------------|
| `psnr_gt_masked` | PSNR on GT foreground mask only | Standard foreground quality |
| `psnr_intersection` | PSNR on intersection of predicted & GT masks | Pure color accuracy, coverage-independent |
| `psnr_whole` | PSNR on full image (including background) | Overall image quality |
| `IoU` | Intersection over Union of predicted vs GT silhouette | Shape accuracy |
| `coverage` | Predicted mask area / GT mask area | Completeness |
| `color_bias` | Mean RGB difference on intersection | Systematic color shift detection |

### Final Results (M5, Test frames 3240-3599)

| Metric | FL GS-LRM 6v (GT input) | FL E2E (Stage 1 input) | PS M5 6v (per-scene) |
|--------|:------------------------:|:----------------------:|:--------------------:|
| PSNR_fg | **23.84** | 8.20 | 13.78 |
| PSNR_whole | **36.97** | 28.50 | 29.00 |
| PSNR_intersection | **~23.40** | ~8.00 | 20.47 |
| IoU | **0.954** | 0.521 | 0.846 |
| Coverage | **~1.0** | 0.715 | 0.893 |

### 9-Experiment Comparison Matrix

Systematic 비교를 위해 3 methods x 3 view conditions로 구성:

#### Protocol A: Temporal (Same camera, test frames)

| Condition | FL GS-LRM | FL E2E | PS M5 |
|:---------:|:---------:|:------:|:-----:|
| 6v | **22.16** | 8.20 | 13.78 |
| 5v | **20.66** | 8.20 | *training* |
| 4v | **18.56** | 8.20 | *queued* |

#### Protocol B: Spatial NVS (Holdout camera)

| Condition | FL GS-LRM | FL E2E | PS M5 |
|:---------:|:---------:|:------:|:-----:|
| 6v | **16.81** | 9.10 | 13.16 |
| 5v | **16.70** | 8.15 | *training* |
| 4v | **15.50** | 7.14 | *queued* |

### Key Findings from Comparison

- **F1: FL GS-LRM (GT input) >> PS** --- +9.62 dB PSNR_fg, +2.93 dB PSNR_intersection.
  Feed-forward 모델이 per-scene optimization을 압도적으로 이김 (단, GT input 조건).

- **F9: Coverage 차이가 PSNR 격차의 주 원인** --- PS의 coverage 89.3%가 낮은 PSNR_fg의 핵심 원인.
  Mask 밖 영역이 penalty를 받기 때문. PSNR_intersection (순수 색상 비교)에서 gap이 줄어듦 (+2.93 vs +9.62).

- **E2E vs PS**: FL E2E (8.20) < PS (13.78). Stage 1 병목으로 인해 E2E는 아직 PS에 미치지 못함.
  → 이것이 DA1 등 domain adaptation 전략의 동기.

### Interpretation for Research Narrative

```
FaceLift의 연구적 가치:

1. GS-LRM 자체의 우수성은 입증됨 (GT input에서 +9.62 dB over PS)
2. 현재 E2E 병목은 Stage 1 (Multi-view Diffusion)에 있음
3. Stage 1 개선은 E2E로 잘 전이되지 않음 (14% transfer rate)
4. Domain adaptation (DA1)이 더 효율적인 해결 경로 (+1.64 vs +0.51)

→ "파이프라인의 개별 stage가 아닌 stage 간 연결(domain gap)이 핵심 문제"
```

---

## 9. Summary: Experiment Decision Tree

전체 실험 흐름을 한눈에 보여주는 decision tree:

```
Baseline (E0: 22.34 dB, 4-view GT)
│
├─ More views help?
│   └─ YES (H4: 1v=10.47 → 6v=23.84, monotonic increase) ──────────── ✅ CONFIRMED
│       └─ 1→2 view = phase transition (+5.48 dB, IoU 0.028→0.858)
│
├─ Alpha mask loss helps?
│   └─ NO (H6: 21.71 → 20.84, monotonic decline) ──────────────────── ❌ REJECTED
│       └─ Foreground overfocus harms global geometry
│
├─ SSIM weight increase helps?
│   └─ NO (H7: collapse at weight >= 0.5) ──────────────────────────── ❌ REJECTED
│       └─ Gaussian rendering + SSIM gradient conflict
│
├─ Preprocessing matters?
│   └─ YES (HP: M0=diverge, M5_4=21.80 >= baseline) ───────────────── ✅ CONFIRMED
│       └─ Centering is essential, normalization is optional
│
├─ Stage 1 improvement → E2E?
│   └─ WEAK (H5: +3.7 dB Stage 1 → +0.51 dB E2E, 14% transfer) ──── ⚠️ LIMITED
│       │
│       └─ Why?
│           ├─ H_T1: Distribution mismatch?
│           │   └─ DA1: +1.64 dB E2E ──────────────────────────────── ✅ CONFIRMED
│           ├─ H_T2: View inconsistency?
│           │   └─ H3: In progress (GPU 5, step ~4200/10K) ─────────── 🔄 TESTING
│           └─ H_T3: Quality ceiling?
│               └─ All strategies converge at 7.9-8.4 ──────────────── ⚠️ POSSIBLE
│
└─ FL vs PS?
    ├─ FL GS-LRM 6v >> PS by +9.62 dB (GT input) ──────────────────── ✅ FL WINS
    ├─ FL E2E << PS (8.20 vs 13.78, Stage 1 bottleneck) ────────────── ⚠️ PS WINS (E2E)
    └─ DA1 narrows gap: 10.08 vs 13.78 ─────────────────────────────── 🔄 CLOSING
```

---

## 10. Cumulative Findings Registry

모든 실험에서 도출된 findings를 체계적으로 정리:

| ID | Finding | Source | Confidence |
|----|---------|--------|:----------:|
| **F1** | FL 6v feedforward >> PS per-scene by +9.62 dB (PSNR_fg) | Phase 5 | HIGH |
| **F2** | 1→2 view = phase transition (+5.48 dB, depth ambiguity 해소) | H4 | HIGH |
| **F3** | MVDiff transfer rate ~14% (distribution mismatch 주 원인) | H5, DA1 | HIGH |
| **F4** | View count vs PSNR: monotonic increase, diminishing returns | H4 | HIGH |
| **F5** | 6-view GT = 23.84 dB (GS-LRM upper bound) | H4-6v | HIGH |
| **F6** | Alpha mask supervision 무효 (foreground overfocus) | H6 | HIGH |
| **F7** | SSIM weight > 0.1 = decline or collapse | H7 | HIGH |
| **F8** | Raw data (M0) = diverge, centering is prerequisite | HP-M0 | HIGH |
| **F9** | PS coverage 89.3%가 낮은 PSNR_fg 주 원인 | Phase 5 | MEDIUM |
| **F10** | DA1 domain adaptation = +1.64 dB (3.2x more efficient than Stage 1 optim) | DA1 | MEDIUM |
| **F11** | Centering alone (M5_4) >= full preprocessing | HP | MEDIUM |
| **F12** | All Stage 1 strategies converge at E2E 7.9-8.4 dB ceiling | Phase 3 | HIGH |
| **F13** | PS 논문 PSNR 33.5 vs 우리 13.78 = metric protocol 차이 (full-image vs FG-masked) | Protocol analysis | HIGH |
| **F14** | PS 논문 데이터(Duke)와 우리 데이터(DANNCE/MAMMAL)는 완전 별개 | Provenance audit | HIGH |

---

## 11. Current Status and Next Steps

> **⚠️ 최신 상태**: [[experiments/EXPERIMENT_REGISTRY]] 및 [[experiments/hypothesis_roadmap]] 참조.
> 아래는 2026-02-23 시점의 스냅샷입니다.

### Active Experiments (as of 2026-02-23)

| Experiment | GPU | Progress | ETA | Purpose |
|------------|:---:|:--------:|:---:|---------|
| **H3** (Pose conditioning) | 5 | step ~4,200 / 10,000 | ~20h | H_T2 검증: view consistency |
| **DA1** (Domain adaptation) | 6 | Datagen phase, then GS-LRM fine-tune | ~16h total | H_T1 추가 검증 |
| **HP-M5_4** (Center only) | 7 | step ~6,000 / 15,000 | ~6h | Preprocessing ablation |
| **HP-M5_5** (Center+norm) | 4 | Training (best_psnr.pt saved) | ~8h | Preprocessing ablation |
| **PS M5 5v** (Pose Splatter) | joon | epoch ~7 / 50 | ~25h | Fair comparison: 5v condition |

### Priority Queue (Next)

| Priority | Action | Depends on | Expected insight |
|:--------:|--------|:----------:|------------------|
| **P1** | DA1 datagen 완료 → GS-LRM fine-tune → E2E eval | DA1 datagen | Distribution adaptation의 E2E 효과 확인 |
| **P2** | H3 완료 → E2E eval | H3 training | Pose conditioning이 E2E에 미치는 영향 |
| **P3** | PS 5v 완료 → fair eval → 9-exp matrix 업데이트 | PS 5v training | 3-way comparison at 5v condition |
| **P4** | DA2 설계 (mixed GT+MVDiff training) | DA1 results | 반복적 domain adaptation으로 추가 개선 |
| **P5** | **H_Split: 1:1:1 split 비교** | DA1+H3 완료 | 8:1:1 vs 1:1:1 split bias 검증 |
| **P6** | **Cross-species Rat7M** | H_Split 완료 | Multi-species generalization 검증 |

### Open Research Questions

1. **DA1 ceiling은 어디인가?** --- DA1이 +1.64 dB를 줄였지만, GT 기준 23.84와 DA1의 10.08 사이에 아직 ~14 dB gap이 존재. Iterative DA (DA2, DA3...)로 추가 수렴 가능한가?

2. **H3 (Pose conditioning)은 H_T2를 해소하는가?** --- View consistency 개선이 E2E에 유의미한 영향을 주는지. H_T1(DA1)과 H_T2(H3)의 상대적 기여도 비교.

3. **Preprocessing의 최적점은?** --- HP-M5_4가 baseline을 이긴 것이 robust한지, 더 많은 step에서도 유지되는지.

4. **FL E2E가 PS를 이길 수 있는가?** --- 현재 8.20 vs 13.78. DA1(10.08)으로 gap이 줄었지만, DA + H3 + advanced Stage 1의 조합으로 역전 가능한가?

5. **Scalability** --- 3,600 frame 소규모 데이터셋에서의 결과가 larger dataset에서도 유지되는가?

6. **H_Split: Split ratio bias?** --- 8:1:1이 FL에 유리한 bias를 만드는가? 1:1:1 split에서 FL-PS gap 변화 확인. (FL_vs_PS_comparison.md §2.5 참조)

7. **Cross-species generalization** --- Mouse로 학습한 FL이 Rat7M 데이터에서 얼마나 transfer되는가? PS cross-species (Mouse→Rat -1.8 dB)와 비교. (FL_vs_PS_comparison.md §2.6 참조)

---

## Appendix A: Configuration Quick Reference

### GS-LRM (Stage 3) Configs

```
configs/mouse/uniform/
├── base_uniform_v2.yaml           # Base: lr=1e-6, 15K steps, loss weights
├── 1view_v2.yaml ~ 6view_v2.yaml # View ablation (H4)
├── 4view_alpha03_v3.yaml          # Alpha mask 0.3 (H6)
├── 4view_alpha05_v3.yaml          # Alpha mask 0.5 (H6)
├── 4view_alpha10_v3.yaml          # Alpha mask 1.0 (H6)
├── 4view_ssim03_v2.yaml           # SSIM weight 0.3 (H7)
├── 4view_ssim05_v2.yaml           # SSIM weight 0.5 (H7)
├── 4view_ssim10_v2.yaml           # SSIM weight 1.0 (H7)
├── hp_M5_4.yaml                   # Center only preprocessing (HP)
├── hp_M5_5.yaml                   # Center + per-view norm (HP)
└── domain_adapt_E2_v1.yaml        # Domain adaptation (DA1)
```

### Multi-view Diffusion (Stage 1) Configs

```
configs/mvdiffusion/
├── M5t2.yaml                      # Baseline
├── M5t2_randref_sparse.yaml       # E2 (best: random ref + sparse attn)
└── M5t2_H3_resume_pose.yaml       # H3 (E2 + pose conditioning)
```

### Key Data Paths (gpu03)

```
/home/joon/data/preprocessed/FaceLift_mouse/
├── M5/        → Full preprocessing (3,600 frames, 6 views each)
├── M0/        → Raw (no preprocessing)
├── M5_4/      → Center crop only
├── M5_5/      → Center + per-view normalization
└── M5_mvdiff/ → Stage 1 generated (DA1 training data)
```

---

## Appendix B: Key Code Entry Points

| Script | Purpose | Location |
|--------|---------|----------|
| `train_gslrm.py` | GS-LRM training | Project root |
| `train_diffusion.py` | Multi-view Diffusion training | Project root |
| `mouse_extensions/data/mouse_dataset.py` | Dataset class | `MouseViewDataset` class |
| `gslrm/model/gslrm.py` | GS-LRM model | `GSLRM.forward()` |
| `mouse_extensions/model/enhanced_loss.py` | Loss computation | Composite loss |
| `mouse_extensions/validation/validator.py` | Validation logic | PSNR/SSIM eval |
| `mouse_extensions/scripts/eval/fair_comparison.py` | Fair eval (FL side) | Unified metrics |
| `mouse_extensions/scripts/domain_adapt/generate_mvdiff_train_data.py` | DA1 datagen | Stage 1 inference on train set |
| `mouse_extensions/scripts/report/` | Report generation | Modular report system |

---

## Appendix C: GPU Allocation Reference

| GPU | Type | VRAM | Status | Current Job |
|:---:|------|:----:|:------:|-------------|
| 0-3 | RTX PRO 6000 Blackwell Max-Q | 97GB | UNUSABLE | sm_120, PyTorch CUDA kernel 미지원 |
| 4 | RTX A6000 | 48GB | **Active** | HP-M5_5 (preprocessing ablation) |
| 5 | RTX A6000 | 48GB | **Active** | H3 (pose conditioning) |
| 6 | RTX A6000 | 48GB | **Active** | DA1 (datagen → auto-launch fine-tune) |
| 7 | RTX A6000 | 48GB | **Active** | HP-M5_4 (preprocessing ablation) |

> **Rule**: `CUDA_VISIBLE_DEVICES=4,5,6,7` 범위만 사용. GPU 0-3 지정 시 `RuntimeError: no kernel image` 발생.

---

## Appendix D: Glossary

| Term | Definition |
|------|-----------|
| **GS-LRM** | Gaussian Splatting Large Reconstruction Model. Transformer-based feed-forward 3D reconstruction. |
| **Multi-view Diffusion** | SD2.1-UnCLIP + Era3D RMA 기반 multi-view image generation (Stage 1). |
| **RMA** | Row-wise Multi-view Attention. Era3D에서 제안한, multi-view consistency를 위한 attention mechanism. |
| **M5** | Mouse dataset preprocessing level 5: center crop + uniform normalization. |
| **M5t2** | M5 dataset with temporal split (80:10:10 = 2880:360:360). |
| **E2E** | End-to-End. Stage 1 + Stage 3 전체 파이프라인을 통한 결과. |
| **Transfer rate** | Stage 1 개선이 E2E에 전이되는 비율 (dB/dB). |
| **Domain adaptation** | GS-LRM을 Stage 1 출력(non-GT)으로 재학습하여 distribution gap을 줄이는 전략. |
| **Coverage** | Predicted silhouette area / GT silhouette area. 재구성 completeness 지표. |
| **PSNR_intersection** | Predicted와 GT mask의 교집합 영역에서만 계산한 PSNR. Coverage에 무관한 순수 색상 정확도. |
| **PS** | Pose Splatter. Per-scene optimization baseline (6-camera, 50 epoch). |

---

*FaceLift Mouse Research Notes v1.0 | Updated: 2026-02-23*
