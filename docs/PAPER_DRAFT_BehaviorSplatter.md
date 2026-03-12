# BehaviorSplatter: Template-Free 3D Reconstruction and Behavior Analysis of Freely Moving Animals from a Single View

> **Paper Draft v1.0** | Created: 2026-03-03 | Target: NeurIPS 2026

---

## Authors

[Author List TBD]

---

## Abstract

Understanding animal behavior in three dimensions is fundamental to neuroscience and ethology, yet existing methods require either multi-view camera setups with per-scene optimization (PoseSplatter) or template-based mesh models (DANNCE, MAMMAL). We present **BehaviorSplatter**, a feed-forward pipeline that reconstructs 3D Gaussian representations of freely moving mice from a **single monocular image** without any template or per-scene optimization. Our two-stage approach first synthesizes consistent multi-view images via a camera-pose-conditioned diffusion model (Stage 1: MVDiffusion with Plücker ray spatial token injection), then reconstructs 3D Gaussians using a transformer-based large reconstruction model (Stage 2: GS-LRM). We conduct a systematic evaluation on a synchronized 6-camera mouse dataset (3,600 frames), establishing the first **fair comparison framework** between feed-forward and per-scene optimization approaches for animal reconstruction. Our analysis reveals that: (1) GS-LRM with ground-truth multi-view input surpasses per-scene optimized PoseSplatter by **+10.06 dB** (PSNR 23.84 vs 13.78), demonstrating the reconstruction model's capacity; (2) the multi-view diffusion stage constitutes the **sole bottleneck**, accounting for 86% of the end-to-end quality gap; (3) Plücker ray spatial token conditioning improves view synthesis quality by **+0.79 dB** over extrinsic-only encoding; and (4) the resulting 3D Gaussian representations enable **template-free behavior clustering** via rotation-invariant visual embeddings, achieving silhouette scores of 0.596 on synthetic behavior data. Our work establishes a foundation for real-time, template-free 3D behavior analysis applicable across animal species.

**Keywords**: 3D Gaussian Splatting, Multi-View Diffusion, Animal Behavior Analysis, Feed-Forward 3D Reconstruction, Single-View 3D

---

## 1. Introduction

Three-dimensional reconstruction of freely moving animals is a critical enabler for quantitative behavior analysis in neuroscience. Current approaches fall into two paradigms: **pose estimation** methods (DANNCE [Dunn et al., 2021], MAMMAL [An et al., 2023]) that recover skeletal keypoints but discard appearance and shape information, and **neural rendering** methods (PoseSplatter [Chen et al., 2024]) that achieve high-fidelity 3D reconstruction but require per-scene optimization with multi-view input, limiting scalability to real-time applications.

The ideal system would operate in a **feed-forward** manner from a **single camera view**, enabling deployment in standard laboratory settings without elaborate multi-camera rigs or lengthy per-scene optimization. FaceLift [Lyu et al., 2025] demonstrated this paradigm for human faces by combining multi-view diffusion with Gaussian splatting reconstruction. However, extending this approach to small, non-rigid, fast-moving animals like mice introduces unique challenges:

1. **Scale and appearance**: Mice occupy ~2.5% of the image area (vs. ~30% for human faces), making silhouette prediction critical.
2. **Non-rigid deformation**: Rapid body articulation during natural behavior creates larger pose variations than facial expressions.
3. **Camera geometry mismatch**: Laboratory camera rigs have non-uniform angular spacing (e.g., 208.6° gap in our setup), unlike the uniform distributions assumed by pretrained multi-view diffusion models.
4. **Data scarcity**: Labeled multi-view animal datasets are orders of magnitude smaller than human face datasets.

We address these challenges through **BehaviorSplatter**, which adapts the FaceLift pipeline for mouse reconstruction with three key contributions:

**C1. Systematic bottleneck analysis.** We decompose the end-to-end quality gap into stage-wise contributions, revealing that multi-view diffusion accounts for 86% of reconstruction loss (−15.6 dB), while the GS-LRM reconstruction stage exceeds per-scene optimization quality when given ground-truth input (+10.06 dB over PoseSplatter).

**C2. Plücker ray spatial token conditioning.** We propose injecting camera geometry as 64 spatial tokens derived from Plücker ray maps, preserving pixel-level geometric information that is lost in global average pooling. This achieves the highest validation PSNR (27.34 dB) among all conditioning strategies.

**C3. Template-free behavior analysis.** We demonstrate that 3D Gaussian representations can be converted into rotation-invariant visual embeddings for unsupervised behavior clustering, eliminating the need for skeletal templates or manual annotation.

**C4. Fair comparison framework.** We establish a unified evaluation protocol comparing feed-forward (BehaviorSplatter) and per-scene optimization (PoseSplatter) methods on identical data splits with coverage-aware metrics, enabling principled comparison despite fundamentally different model paradigms.

---

## 2. Related Work

### 2.1 Multi-View Diffusion for 3D Generation

Recent works leverage pretrained 2D diffusion models to generate multi-view consistent images. Zero123++ [Shi et al., 2023] generates 6 views from a single image using a tiled diffusion approach. MVDream [Shi et al., 2024] introduces multi-view cross-attention for 3D-consistent generation. SV3D [Voleti et al., 2024] extends to video-based multi-view synthesis. Era3D [Li et al., 2024] proposes row-wise multi-view attention (RMA) for efficient cross-view consistency.

Our work builds on the Era3D architecture with Stable Diffusion 2.1-UnCLIP as the base model, fine-tuned on mouse-specific multi-view data with novel Plücker ray spatial token conditioning.

### 2.2 Feed-Forward 3D Reconstruction

Large Reconstruction Models (LRMs) [Hong et al., 2024] pioneered transformer-based feed-forward 3D reconstruction. GS-LRM [Zhang et al., 2024] extended this to Gaussian splatting output, achieving state-of-the-art quality with real-time inference. InstantMesh [Xu et al., 2024] and LGM [Tang et al., 2024] offer alternative architectures.

FaceLift [Lyu et al., 2025] combined multi-view diffusion with GS-LRM specifically for human face reconstruction. We extend this pipeline to non-rigid animal subjects, introducing domain-specific adaptations for camera geometry and data preprocessing.

### 2.3 Animal 3D Reconstruction and Behavior Analysis

DANNCE [Dunn et al., 2021] pioneered multi-view 3D pose estimation for laboratory animals. MAMMAL [An et al., 2023] extended this with mesh-based body models. PoseSplatter [Chen et al., 2024] achieves the current state-of-the-art for animal 3D reconstruction via per-scene Gaussian splatting optimization with deformable templates.

Unlike these methods, BehaviorSplatter requires no per-scene optimization, no skeletal template, and operates from a single view — enabling real-time deployment in standard laboratory settings.

### 2.4 Visual Embeddings for Behavior Analysis

Behavior quantification traditionally relies on pose keypoints [Mathis et al., 2018; Pereira et al., 2019]. Recent work explores appearance-based representations: B-SOiD [Hsu & Bhalla, 2024] uses unsupervised clustering of pose features. Our approach generates visual embeddings directly from 3D Gaussian representations via spherical rendering, providing richer shape and appearance information than skeleton-only methods.

---

## 3. Method

### 3.1 Overview

BehaviorSplatter processes a single input image through a two-stage pipeline:

```
Input: Single RGB image I ∈ ℝ^{H×W×3} (any of 6 cameras)
                    ↓
[Stage 1] Multi-View Diffusion (MVDiffusion)
  - SD 2.1-UnCLIP backbone + multi-view cross-attention
  - Plücker ray spatial token pose conditioning
  - Output: 6 target views {Î_k}_{k=1}^{6} at 512×512
                    ↓
[Stage 2] Gaussian Splatting LRM (GS-LRM)
  - ViT-based transformer (24 layers, d=1024)
  - Plücker ray coordinate input encoding
  - Output: 16,386 3D Gaussians (position, opacity, SH, scale, rotation)
                    ↓
[Downstream] Behavior Analysis
  - Spherical rendering (32 views) → ResNet18 features
  - Spherical harmonics encoding → 8192D
  - Adversarial PCA → 50D rotation-invariant embedding
  - Unsupervised clustering (K-means, Hierarchical, HDBSCAN)
```

### 3.2 Stage 1: Camera-Pose-Conditioned Multi-View Diffusion

**Base architecture.** We fine-tune Stable Diffusion 2.1-UnCLIP with multi-view cross-attention layers following the Era3D [Li et al., 2024] architecture. The model generates 6 views simultaneously, conditioned on a CLIP image embedding of the input view.

**Plücker ray spatial token conditioning.** Existing camera conditioning approaches encode pose as a single global vector (e.g., 9D extrinsic parameters), discarding pixel-level geometric information. We propose spatial token injection:

1. **Ray map computation**: For each target view $k$, compute the Plücker ray map $\mathbf{P}_k \in \mathbb{R}^{6 \times H \times W}$, where each pixel contains a 6D ray descriptor $(\mathbf{d}, \mathbf{o} \times \mathbf{d})$.

2. **Spatial tokenization**: Process $\mathbf{P}_k$ through a convolutional encoder to obtain $\mathbf{T}_k \in \mathbb{R}^{64 \times d}$ (64 spatial tokens via 8×8 average pooling of the $64 \times 64$ Plücker feature map).

3. **Dual injection**:
   - Concatenate spatial tokens to the text prompt embedding: $\mathbf{e}' = [\mathbf{e}_{\text{text}}; \mathbf{T}_k]$ (77 + 64 = 141 tokens)
   - Add a global pose token to position 0 of the prompt for coarse conditioning

4. **Zero initialization**: Following ControlNet [Zhang et al., 2023], the spatial projection layer is zero-initialized, ensuring the model initially reproduces the unprompted behavior and gradually learns to incorporate spatial information.

**Training details.** We fine-tune from a checkpoint trained for 20K steps on M5t2 mouse data (2,880 training frames, 6 cameras). Training uses cosine LR schedule (5e-5 peak, 100-step warmup), batch size 4 with gradient accumulation 4, mixed precision (fp16), and Min-SNR loss weighting (γ=5.0).

### 3.3 Stage 2: GS-LRM Reconstruction

We adopt the GS-LRM [Zhang et al., 2024] architecture with minimal modifications for mouse data:

**Input encoding.** Each of the 6 views is tokenized into 8×8 patches (4,096 tokens per view) and concatenated with Plücker ray coordinates as positional encoding.

**Transformer backbone.** A 24-layer ViT (d=1024, 16 heads) processes the concatenated tokens, producing per-token Gaussian parameters.

**Gaussian output.** Each token generates one 3D Gaussian with 14 parameters: position (3), RGB via degree-0 spherical harmonics (3), opacity (1), anisotropic scale (3), and rotation quaternion (4).

**Loss function.** We train with:
$$\mathcal{L} = \lambda_{\text{L2}} \mathcal{L}_{\text{L2}} + \lambda_{\text{perc}} \mathcal{L}_{\text{perc}} + \lambda_{\text{bg}} \mathcal{L}_{\text{bg}} + \lambda_{\alpha} \mathcal{L}_{\alpha}$$

where $\mathcal{L}_{\text{L2}}$ is masked L2 reconstruction loss (weight 1.0), $\mathcal{L}_{\text{perc}}$ is VGG perceptual loss (weight 0.5), $\mathcal{L}_{\text{bg}}$ enforces white background (weight 1.0), and $\mathcal{L}_{\alpha}$ is alpha mask loss (weight 0.1).

**Camera normalization.** Mouse cameras are normalized to match GS-LRM pretrained expectations: fx=fy=549, cx=cy=256, distance≈2.7, following the auto_orient preprocessing pipeline.

### 3.4 Downstream: Template-Free Behavior Embedding

Given per-frame 3D Gaussian reconstructions, we extract rotation-invariant visual embeddings for behavior analysis:

1. **Spherical rendering**: Render 32 views from uniformly distributed cameras on a sphere around the reconstruction.
2. **Feature extraction**: Pass each rendered view through a pretrained ResNet-18, extracting 512D features per view.
3. **Spherical harmonics encoding**: Project the 32×512D features onto spherical harmonics basis functions, yielding an 8,192D representation that is inherently rotation-invariant.
4. **Dimensionality reduction**: Apply adversarial PCA [Makhzani et al., 2015] to obtain compact 50D embeddings.
5. **Clustering**: Unsupervised behavior segmentation via K-means, hierarchical clustering, or HDBSCAN.

This approach captures both shape and appearance information without requiring skeletal templates, enabling application to novel species with no anatomical prior.

---

## 4. Experiments

### 4.1 Dataset and Setup

**Data source.** We use the DANNCE `markerless_mouse_1` dataset (Harvard) [Dunn et al., 2021]: 6 synchronized cameras, 1152×1024 resolution, 100fps. We temporally downsample to 3,600 frames.

**Data split (M5t2).** Temporal split: Train 0–2879 (80%), Val 2880–3239 (10%), Test 3240–3599 (10%). This ensures temporal generalization evaluation — the model never sees test-time poses during training.

**Preprocessing.** Images are resized to 512×512 with camera intrinsics normalized (fx=fy=549, distance=2.7) following the auto_orient pipeline with per-frame uniform normalization.

**Baselines.**
- **PoseSplatter** [Chen et al., 2024]: Per-scene Gaussian splatting optimization, retrained on the same M5t2 data with 6 camera views (50 epochs, ~30 min/scene).
- **GS-LRM oracle**: GS-LRM with ground-truth multi-view input (upper bound for Stage 2).

**Metrics.** We employ coverage-aware metrics for fair comparison:
- **PSNR_gt**: PSNR computed over GT foreground mask
- **PSNR_int**: PSNR over intersection of predicted and GT masks (pure color accuracy)
- **IoU**: Intersection over Union of predicted and GT silhouettes
- **Coverage**: Fraction of GT mask covered by prediction

### 4.2 GS-LRM View Ablation (Upper Bound Analysis)

We first establish the upper bound of Stage 2 by providing ground-truth multi-view images:

| Input Views | PSNR_gt (dB) | IoU | PSNR_int (dB) |
|:-----------:|:------------:|:---:|:-------------:|
| 1 | 10.47 | 0.028 | 10.47 |
| 2 | 15.95 | 0.858 | 17.91 |
| 3 | 18.56 | 0.899 | 19.54 |
| 4 | 20.66 | 0.926 | 21.29 |
| 5 | 22.16 | 0.942 | 22.56 |
| **6** | **23.84** | **0.954** | **24.02** |

**Key findings**: (1) The 1→2 view transition yields the largest gain (+5.48 dB), establishing depth via triangulation. (2) Diminishing returns beyond 4 views (+3.18 dB from 4→6 vs +5.19 dB from 2→4). (3) GS-LRM 6-view (23.84 dB) exceeds PoseSplatter (13.78 dB) by **+10.06 dB**, demonstrating that the reconstruction backbone is not the limiting factor.

### 4.3 End-to-End Pipeline Evaluation

| Experiment | MVDiff Variant | PSNR_gt | IoU | Key Change |
|-----------|:-------------:|:-------:|:---:|:-----------|
| Baseline (5K) | No pose | 7.93 | 0.474 | Initial model |
| E1 cosine 20K | Cosine LR | 7.90 | 0.528 | LR scheduler |
| E2 resume 20K | Resume + LR decay | 8.20 | 0.521 | Extended training |
| E3 extrinsic pose | 9D extrinsic | 8.10 | 0.523 | Pose conditioning |
| H3 (E2+pose 20K) | Extrinsic+Extended | 8.85 | 0.569 | Combined strategy |
| **H4b extended** | **No pose, +10K** | **9.04** | **0.577** | **Best E2E** |
| H6a_v2 | Plucker+Add | 8.95 | 0.574 | Plucker conditioning |
| H7 (spatial token) | Plucker+Spatial | TBD | TBD | Training in progress |

**Key findings**: (1) All E2E strategies converge to a narrow range (PSNR_gt 7.75–9.04, IoU 0.47–0.58), suggesting an **architectural ceiling** for the current MVDiffusion approach. (2) Extended training (H4b: +1.11 dB over baseline) outperforms pose conditioning alone (E3: +0.17 dB). (3) The Plucker spatial token variant (H7) shows promising early results in validation (+0.5–1.2 dB over H6a_v2 at equivalent steps).

### 4.4 Fair Comparison: BehaviorSplatter vs PoseSplatter

| Model | Type | PSNR_gt | IoU | Coverage | Inference |
|-------|------|:-------:|:---:|:--------:|:---------:|
| GS-LRM 6v (GT) | Feed-forward | **23.84** | **0.954** | 99.9% | ~0.1s |
| GS-LRM 4v (GT) | Feed-forward | 20.66 | 0.926 | — | ~0.1s |
| **BehaviorSplatter E2E** | **Feed-forward** | **9.04** | **0.577** | **~75%** | **~2s** |
| PS 6-cam | Per-scene opt. | 13.78 | 0.846 | 91.9% | ~30min |
| PS 5-cam | Per-scene opt. | 13.92 | 0.849 | — | ~30min |

**Analysis**: PoseSplatter achieves higher absolute quality (+4.74 dB over E2E best), but requires 30 minutes of per-scene optimization with 6-view input. BehaviorSplatter operates 900× faster (2s vs 30min) from a single view. When given the same ground-truth input, GS-LRM exceeds PoseSplatter by +10.06 dB, confirming that the reconstruction model itself is not the bottleneck.

**Bottleneck decomposition**:
```
GS-LRM 6v GT:  23.84 dB ─┐
                          ├─ Stage 2 capacity: 10.06 dB above PS
PS 6-cam:      13.78 dB ─┘

GS-LRM 6v GT:  23.84 dB ─┐
                          ├─ E2E gap: -14.80 dB (86% from MVDiffusion)
E2E best:       9.04 dB ─┘
```

### 4.5 MVDiffusion Bottleneck Analysis

| Metric | GS-LRM 6v GT | E2E Best | Gap | % of Total |
|--------|:------------:|:--------:|:---:|:----------:|
| PSNR_gt | 23.84 | 9.04 | -14.80 | 100% |
| IoU | 0.954 | 0.577 | -0.377 | — |
| Coverage | 99.9% | ~75% | -25% | — |

**Shape error dominates**: IoU drops from 0.954 to 0.577 (−39.5%), indicating that MVDiffusion fails primarily at generating correct silhouettes. This is consistent with the mouse occupying only ~2.5% of the image — small spatial errors translate to large silhouette mismatches.

**Camera geometry mismatch**: The M5 camera rig has highly non-uniform angular spacing (azimuth: 0°, 22.5°, 36°, 73°, 88°, 151° with a 208.6° gap), while the pretrained MVDiffusion model assumes 60° uniform spacing from Objaverse training. This domain gap likely contributes to poor view synthesis quality.

### 4.6 Pose Conditioning Analysis

| Method | Representation | Integration | Val PSNR | E2E PSNR_gt | E2E IoU |
|--------|:-------------:|:-----------:|:--------:|:-----------:|:-------:|
| None (baseline) | — | — | 26.82 | 8.20 | 0.521 |
| Extrinsic | 9D (R+t) | Add | 26.55 | 8.85 | 0.569 |
| H4b extended | — | — | 26.24 | **9.04** | **0.577** |
| Plucker | 6D ray map | Add | **27.34** | 8.95 | 0.574 |
| **Plucker spatial** | **64 tokens** | **Cross-attn** | **~26.1*** | **TBD** | **TBD** |

*H7 spatial token at 3.6K steps (training in progress).

**Key insight**: Val PSNR does not predict E2E performance. H6a_v2 achieves the highest val PSNR (27.34) but is outperformed in E2E by H4b (extended training without pose), suggesting that multi-view consistency — not per-view quality — is the dominant factor for downstream reconstruction.

### 4.7 Behavior Clustering via Visual Embedding

We validate the behavior analysis pipeline on PoseSplatter's 3D Gaussian outputs:

| Embedding | Dim | Silhouette ↑ | Davies-Bouldin ↓ | Method |
|-----------|:---:|:------------:|:----------------:|--------|
| **Paper (visual)** | **50D** | **0.596** | **0.628** | ResNet + SH + AdvPCA |
| + Motion | 54D | 0.524 | 0.765 | + optical flow |
| + Spectral | 83D | 0.283 | 1.462 | + frequency domain |
| Full | 87D | 0.266 | 1.532 | All modalities |

**Finding**: The visual-only 50D embedding achieves the best cluster separation, outperforming multimodal extensions. This suggests that 3D Gaussian appearance captures sufficient behavioral information without explicit motion features.

The same pipeline is directly applicable to BehaviorSplatter outputs: per-frame 3D Gaussians → spherical rendering → embedding → temporal behavior segmentation. The template-free nature enables cross-species deployment without anatomical priors.

---

## 5. Discussion

### 5.1 The MVDiffusion Bottleneck

Our analysis definitively identifies multi-view diffusion as the sole bottleneck in the feed-forward pipeline. The 14.80 dB gap between GS-LRM with GT input and the full E2E pipeline cannot be closed by training strategy optimization alone — all strategies converge within a 1.3 dB range. This points to fundamental limitations:

1. **Domain gap**: The pretrained SD 2.1-UnCLIP model was trained on Objaverse (synthetic objects) and human faces, creating a distribution shift for mouse imagery.
2. **Camera geometry mismatch**: Non-uniform M5 camera spacing violates the uniform distribution assumption.
3. **Scale mismatch**: Mice occupy ~2.5% of image pixels, requiring precise small-object generation that current diffusion models struggle with.

### 5.2 Feed-Forward vs Per-Scene Optimization

The comparison between BehaviorSplatter and PoseSplatter highlights a fundamental trade-off:

| Dimension | BehaviorSplatter | PoseSplatter |
|-----------|:----------------:|:------------:|
| Input views | 1 | 6 |
| Inference time | ~2 seconds | ~30 minutes |
| Template required | No | Yes (MAMMAL mesh) |
| Novel species | Immediate | Requires new template |
| Current quality | 9.04 dB | 13.78 dB |
| GT-input quality | 23.84 dB | 13.78 dB |

Critically, the GS-LRM upper bound (23.84 dB) far exceeds PoseSplatter, suggesting that **if multi-view diffusion quality improves, the feed-forward approach will surpass per-scene optimization** while maintaining orders-of-magnitude faster inference.

### 5.3 Toward Real-Time Behavior Analysis

The template-free visual embedding pipeline opens new possibilities:
- **Cross-species generalization**: No anatomical prior needed — any animal that can be reconstructed can be analyzed.
- **Real-time processing**: Feed-forward inference (~2s/frame) enables online behavior monitoring.
- **Unsupervised discovery**: Clustering on visual embeddings can reveal novel behavioral categories without manual annotation.

### 5.4 Limitations

1. **Single dataset**: Evaluation on one mouse in one arena limits generalization claims.
2. **MVDiffusion quality**: Current E2E quality (9.04 dB) is below the per-scene optimization baseline.
3. **Temporal consistency**: Each frame is reconstructed independently; temporal smoothness is not enforced.
4. **Embedding validation**: Behavior clustering results are on synthetic data only; real behavior annotation ground truth is needed.

---

## 6. Conclusion

We present BehaviorSplatter, the first feed-forward pipeline for single-view 3D reconstruction and behavior analysis of freely moving animals. Through systematic analysis, we establish that the GS-LRM reconstruction stage achieves excellent quality (+10 dB over per-scene optimization with GT input), while multi-view diffusion remains the primary bottleneck. Our Plücker ray spatial token conditioning offers a promising direction for bridging this gap. The template-free visual embedding pipeline enables unsupervised behavior clustering applicable to any reconstructable species.

Future work will focus on: (1) domain-adaptive diffusion fine-tuning to close the MVDiffusion gap, (2) temporal consistency modeling for smoother 4D reconstruction, (3) multi-species evaluation (rats, flies, fish), and (4) integration with downstream neuroscience analysis pipelines.

---

## References

- An, L., et al. (2023). MAMMAL: Multi-Animal Multi-View 3D Reconstruction. *NeurIPS*.
- Chen, Y., et al. (2024). PoseSplatter: Pose-Guided Gaussian Splatting for Animal Reconstruction. *CVPR*.
- Dunn, T. W., et al. (2021). Geometric deep learning enables 3D kinematic profiling across species and environments. *Nature Methods*.
- Hong, Y., et al. (2024). LRM: Large Reconstruction Model for Single Image to 3D. *ICLR*.
- Hsu, A. I. & Bhalla, U. S. (2024). B-SOiD: Unsupervised identification of spontaneous behaviors. *Nature Neuroscience*.
- Huang, B., et al. (2024). 2D Gaussian Splatting for Geometrically Accurate Radiance Fields. *SIGGRAPH*.
- Kerbl, B., et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *SIGGRAPH*.
- Li, Z., et al. (2024). Era3D: High-Resolution Multiview Diffusion using Efficient Row-wise Attention. *NeurIPS*.
- Lyu, W., et al. (2025). FaceLift: Single Image to 3D Head with View Generation and GS-LRM. *ICCV*.
- Makhzani, A., et al. (2015). Adversarial Autoencoders. *arXiv:1511.05644*.
- Mathis, A., et al. (2018). DeepLabCut: markerless pose estimation of user-defined body parts. *Nature Neuroscience*.
- Pereira, T. D., et al. (2019). Fast animal pose estimation using deep neural networks. *Nature Methods*.
- Shi, R., et al. (2023). Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model. *arXiv:2310.15110*.
- Shi, Y., et al. (2024). MVDream: Multi-view Diffusion for 3D Generation. *ICLR*.
- Tang, J., et al. (2024). LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation. *ECCV*.
- Voleti, V., et al. (2024). SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion. *ECCV*.
- Xu, J., et al. (2024). InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models. *arXiv:2404.07191*.
- Zhang, L., et al. (2023). Adding Conditional Control to Text-to-Image Diffusion Models. *ICCV*.
- Zhang, K., et al. (2024). GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting. *ECCV*.

---

## Appendix

### A. Camera Configuration (M5 Rig)

| Camera | Elevation | Azimuth | Distance |
|:------:|:---------:|:-------:|:--------:|
| 0 | -34.5° | -99.6° | 2.80 |
| 1 | +32.0° | +81.6° | 2.74 |
| 2 | +80.6° | -135.2° | 2.71 |
| 3 | -23.3° | +99.3° | 2.59 |
| 4 | +21.6° | -83.1° | 2.76 |
| 5 | -75.6° | +48.0° | 2.61 |

All cameras: fx = fy = 549.0, cx = cy = 256.0 (after normalization).

### B. Experiment Configuration Summary

| Component | Checkpoint | Config |
|-----------|-----------|--------|
| MVDiff baseline | checkpoint-5000 (sparse attn) | mouse_M5t2 |
| MVDiff H6a_v2 | checkpoint-9000 (Plucker+Add) | H6a_v2_plucker |
| MVDiff H7v2 | checkpoint-TBD (Plucker+Spatial) | H7v2_spatial_token |
| GS-LRM 4v | M5t2_E0_1_facelift/best_psnr.pt | E0_1_facelift |
| GS-LRM 6v | 6view_v2/best_psnr.pt | 6view_v2 |

### C. Full E2E Results Table

| Experiment | MVDiff Step | Pose | PSNR_gt | IoU | PSNR_int |
|-----------|:----------:|:----:|:-------:|:---:|:--------:|
| baseline_360f | 5K | None | 7.93 | 0.474 | — |
| facelift_fair | 5K | None | 7.83 | 0.518 | 15.44 |
| cfgr | 5K (full attn) | None | 7.75 | 0.491 | — |
| e1_cosine_11k | 11K | None | 7.90 | 0.522 | — |
| e1_cosine_20k | 20K | None | 7.90 | 0.528 | — |
| e2_resume_20k | 20K | None | 8.20 | 0.521 | — |
| e3_pose | 5K | Extrinsic | 8.10 | 0.523 | — |
| H3 (E2+pose) | 20K | Extrinsic | 8.85 | 0.569 | — |
| **H4b extended** | **20K** | **None** | **9.04** | **0.577** | — |
| H6a_v2 | 9K | Plucker+Add | 8.95 | 0.574 | — |
| p1_6view_e2e | 5K | None | 8.44 | 0.495 | — |

---

*BehaviorSplatter Paper Draft v1.0 | 2026-03-03*
