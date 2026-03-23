# Neural Texture for MAMMAL Mesh: MoReMouse-Inspired Analysis

> 3-Model Deliberation + Audit + PoC Results | 2026-03-22 ~ 2026-03-23

---

## 1. Problem Statement

**Goal**: Improve MAMMAL mesh rendering quality for DiFix pseudo-GT pair generation.

| Current State | Issue |
|---------------|-------|
| GS-LRM 6v GT view | PSNR 23.84 (excellent, but novel view has needle/pancake artifacts) |
| MAMMAL Blender render | Procedural fur texture (diffuse + noise) — visually unrealistic |
| DiFix Type 3 pairs | 43.2K pairs (N-view→6-view, self-supervised) — in progress |
| DiFix Type 2 pairs | 6v GS-LRM ↔ MAMMAL mesh pseudo-GT — **blocked by mesh quality** |
| DiFix zero-shot | FAILED (domain gap) → fine-tuning required |

**Root cause**: MAMMAL mesh procedural texture creates large domain gap vs real images,
making Type 2 DiFix pairs (novel view supervision) unusable.

---

## 2. MoReMouse Paper Analysis (arxiv 2507.04258)

### 2.1 Overview

| Item | Detail |
|------|--------|
| Authors | Yuan Zhong, Jingxiang Sun, Zhongbin Zhang, Liang An, **Yebin Liu** (Tsinghua) |
| Dataset | **markerless_mouse_1** (same as ours) — 6 cameras, 18000 frames @100FPS |
| Code | **NOT available**, no release announced |
| Web | https://zyyw-eric.github.io/MoreMouse-webpage/ |

### 2.2 Architecture

```
[Gaussian Avatar Construction]
  markerless_mouse_1 (800 frames) + MAMMAL mesh (K=140 joints, 14522 vertices)
  → StyleUNet: UV coords → (position offset, color, opacity, rotation, scale)
  → LBS deformation: mu' = D(mu_0 + delta_mu)
  → Gaussian Splatting render
  → 400k training steps, L1+SSIM+LPIPS+TV losses

[Single-Image Reconstruction Network]
  Single RGB (378x378) → DINOv2-base (768-dim)
  → 12-layer Transformer (16 heads, 64-dim)
  → Triplane decoder (3x64x64x512)
  → MultiHeadMLP (64 neurons, 10 shared layers)
  → NeRF (60 epochs) → DMTet (100 epochs)

[Geodesic Embedding]
  13059 mesh vertices → pairwise geodesic distances
  → 3D continuous correspondence embedding
  → HSV color encoding via PCA (auxiliary supervision)
```

### 2.3 Results

| Method | Synthetic PSNR | Real PSNR | Real SSIM | Real LPIPS |
|--------|:-:|:-:|:-:|:-:|
| **MoReMouse** | **22.03** | **18.42** | **0.948** | **0.087** |
| TripoSR | 13.67 | 11.52 | 0.811 | 0.197 |
| Triplane-GS | 18.05 | 16.79 | 0.930 | 0.110 |
| TripoSR-Tuned | 22.00 | 18.16 | — | — |

**Critical notes on comparison**:
- Real PSNR 18.42 measured on **4-view PointGray cameras** (NOT 6-view arena cameras)
- Different evaluation protocol from FaceLift's fair eval → **direct comparison invalid**
- TripoSR-Tuned nearly matches MoReMouse → dataset quality matters more than architecture

### 2.4 Key Takeaways for Our Use

| Component | Relevance | Adoptability |
|-----------|-----------|:------------:|
| **StyleUNet** | UV→Gaussian params (NOT simple texture) | Needs redesign |
| **Gaussian Avatar** | Per-subject, 400k steps optimization | Too slow for pipeline |
| **Triplane decoder** | DINOv2→triplane→NeRF/DMTet | Separate system (not mesh texture) |
| **Geodesic embedding** | Semantic prior for consistency | Nice-to-have, not critical |
| **Synthetic dataset** | Avatar renders as training data | Our GS-LRM 6v serves same role |
| **Pose conditioning** | MAMMAL LBS deformation | We have this infrastructure |

---

## 3. 3-Model Deliberation Results

### 3.1 Opinions Summary

| Model | Strategy | Confidence | Key Insight |
|-------|----------|:----------:|-------------|
| 🔵 Claude | UV+Pose MLP (500K params, 1 week) | 3.5/5 | Quick to implement, uncertain if sufficient |
| 🟢 Gemini | Per-scene NeRF/GS (bypass mesh) | 2/5 | Neural texture fundamentally limited by mesh alignment |
| 🟠 GPT | StyleUNet-inspired (long-term trend) | 4/5 | Direction correct but lacks implementation detail |

### 3.2 Consensus Points

1. Full MoReMouse reimplementation is impractical (no code, 4-6 weeks)
2. Current Blender procedural texture definitely needs improvement
3. Staged approach required — no big-bang implementation
4. DiFix improvement ≠ guaranteed even with better pseudo-GT

### 3.3 Critical Divergence

| Issue | Resolution |
|-------|------------|
| Neural texture vs direct rendering | **Hybrid**: Neural texture first (lower compute), fall back to per-scene if fails |
| Mesh alignment dependency | **P0 diagnostic**: Quantify alignment before committing |
| StyleUNet extraction feasibility | **Redesign**: Don't copy StyleUNet; build nvdiffrast-compatible UV→RGB model |

---

## 4. Audit Findings

### 4.1 Critical Issues (Must Address)

| # | Finding | Impact | Resolution |
|---|---------|--------|------------|
| A1 | PSNR 18.42 vs 23.84 incomparable (different cameras, protocol) | Misleading ROI estimate | Remove comparison; note protocol difference |
| A2 | StyleUNet output ≠ standard texture (it's Gaussian params) | Architecture mismatch if copied | Design our own UV→RGB for nvdiffrast |
| A3 | Circular logic: "better texture will help" is unproven | Wasted effort if bottleneck elsewhere | Baseline diagnostic FIRST |

### 4.2 Major Issues (Should Address)

| # | Finding | Impact | Resolution |
|---|---------|--------|------------|
| A4 | MAMMAL tracking accuracy unquantified | Blurry texture from misalignment | ✅ **RESOLVED**: IoU=0.793 (PASS) |
| A5 | 6-view UV coverage may leave large gaps | Hallucinated novel view texture | ✅ **RESOLVED**: 98% total, 88% extreme bottom |
| A6 | "Better pseudo-GT → better DiFix" causal chain unproven | Wrong optimization target | DiFix bottleneck ablation |
| A7 | GPU availability uncertain (concurrent session) | Compute conflict | nvidia-smi check before training |

### 4.3 Minor Issues (Acknowledged)

| # | Finding |
|---|---------|
| A8 | P1 timeline assumes no code dependency (correct — pure self-implementation) |
| A9 | NeurIPS deadline not explicitly specified in plan |

---

## 5. Recommended Strategy (Post-Audit)

### Phase 0: Diagnostic — COMPLETED (2026-03-22) ✅

**Results** (script: `/tmp/phase0_diag_v3.py`, report: `outputs/diagnostics/phase0/`):

```
[D0] Camera Configuration
  - 6 cameras at elevation [-8.5°, +9.6°] — near-horizontal
  - All cameras roughly at arena wall height

[D1] MAMMAL Mesh Alignment Quality — ✅ PASS
  - Frame 0 (matched): Mean IoU = 0.793
  - Frame 24 (cross-check): Mean IoU = 0.744
  - MAMMAL tracking accuracy is SUFFICIENT for neural texture supervision

[D2] UV Coverage — ✅ PASS (exceeded expectations)
  - Total: 98.0% vertices visible from at least 1 camera
  - Ventral (bottom 50%): 96.1%
  - Bottom 20%: 91.8%
  - Extreme bottom 10%: 87.8%
  - KEY: Even near-horizontal cameras see ventral surface (mouse rearing)

[D3/D4] Deferred — baseline measurement and DiFix bottleneck check
  moved to Phase 1 integration stage
```

**Gate decision**:
- D1+D2 PASS → proceed to Phase 1
- D1 FAIL → investigate per-frame pose refinement first
- D2 FAIL → neural texture will hallucinate too much → consider per-scene NeRF
- D4 shows DiFix saturated → redirect effort elsewhere

### Phase 1: Differentiable Neural Texture (1 week)

**Architecture** (nvdiffrast-based, NOT MoReMouse copy):

```python
class NeuralMeshTexture(nn.Module):
    """
    Learns RGB texture for MAMMAL mesh via differentiable rendering.
    Supervised by 6-view GT images.
    """
    def __init__(self, hidden_dim=256, num_layers=6):
        # Input: UV coords (2D) + pose embedding (optional)
        # Output: RGB (3D)
        self.uv_encoder = FourierFeatures(2, num_frequencies=10)  # 42-dim
        self.pose_encoder = nn.Linear(280, 64)  # K=140 joints × 2 (sin/cos)
        self.mlp = nn.Sequential(
            nn.Linear(42 + 64, hidden_dim),
            *[ResBlock(hidden_dim) for _ in range(num_layers)],
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

    def forward(self, uv_coords, pose_params):
        uv_feat = self.uv_encoder(uv_coords)
        pose_feat = self.pose_encoder(pose_params)
        rgb = self.mlp(torch.cat([uv_feat, pose_feat], -1))
        return rgb
```

**Training pipeline**:
```
For each frame:
  1. Load MAMMAL mesh + pose → apply LBS deformation
  2. Rasterize mesh from 6 GT cameras (nvdiffrast) → UV maps per pixel
  3. Query NeuralMeshTexture(UV, pose) → predicted RGB per pixel
  4. Loss: L1 + 0.1*LPIPS vs GT images (masked by mesh silhouette)
  5. Optional: per-frame translation/rotation refinement (Δt, Δr)
```

**Key design decisions**:
- Fourier features for UV → captures high-frequency fur patterns
- Pose conditioning → appearance changes with body deformation
- Per-frame refinement → compensates MAMMAL tracking errors (Audit A4)
- Silhouette masking → avoids background contamination

**Training config**:
- 800 frames × 6 views = 4800 training images
- Batch: 4 frames × 6 views per step
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-5
- Epochs: ~200 (~6 hours on single GPU)
- Loss: L1(1.0) + LPIPS(0.1) + mask_IoU(0.3)

**Success criteria**:
- GT view: LPIPS < 0.10 (>50% improvement over Blender baseline)
- Visual: Novel view renders look plausibly realistic (qualitative)

### Phase 2: DiFix Integration (1-2 weeks, if Phase 1 succeeds)

1. Render neural-textured MAMMAL mesh at novel view cameras
2. Generate Type 2 DiFix pairs: GS-LRM(N-view) ↔ neural_texture_render
3. Fine-tune DiFix with Type 2 + Type 3 mixed pairs
4. Evaluate: Novel view PSNR improvement > 1.0 dB

### Phase 3: Enhancement (if Phase 2 succeeds)

Options (pick based on remaining time to NeurIPS):
- View-dependent texture (add view direction input)
- Temporal consistency (shared latent across frames)
- Multi-species transfer (rat mesh + neural texture)

---

## 6. Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| MAMMAL mesh misalignment | High | High | Phase 0 D1 gate + per-frame refinement |
| UV coverage gaps | Medium | High | Phase 0 D2 gate + fallback to per-scene |
| Neural texture overfits to GT views | Medium | Medium | Held-out frame validation |
| DiFix doesn't improve with better pseudo-GT | Medium | High | Phase 0 D4 diagnostic |
| GPU conflict with concurrent session | Low | Medium | nvidia-smi check, GPU isolation |
| NeurIPS timeline exceeded | Medium | High | Strict 1-week Phase 1 timebox |

---

## 7. Comparison: Our Approach vs MoReMouse

| Aspect | MoReMouse | Our Plan |
|--------|-----------|----------|
| **Goal** | Monocular 3D recon | Mesh texture for DiFix pseudo-GT |
| **Architecture** | Gaussian Avatar + Triplane + DINOv2 | nvdiffrast + UV MLP |
| **Training** | 400k steps, per-subject | ~200 epochs, ~6 hours |
| **Pose conditioning** | StyleUNet (UV→Gaussian params) | MLP (UV+pose→RGB) |
| **Novel view** | Triplane inference | Direct mesh render |
| **Complexity** | Very high (multi-stage) | Low-medium (single stage) |
| **Code available** | No | Self-implemented |
| **Expected quality** | Synthetic PSNR 22.03 | Target: LPIPS < 0.10 on GT views |

---

## 8. Decision Tree

```
Phase 0 Diagnostic (1-2 days)
├── D1 FAIL (mesh alignment poor) → Per-frame pose refinement module first
├── D2 FAIL (UV coverage < 50%) → Abort neural texture; try per-scene Instant-NGP
├── D4 FAIL (DiFix saturated) → Redirect to different improvement path
└── ALL PASS
    └── Phase 1: Neural Texture (1 week, strict timebox)
        ├── Success (LPIPS < 0.10) → Phase 2: DiFix Integration
        ├── Partial (LPIPS 0.10-0.15) → Add per-frame refinement, retry
        └── Fail (LPIPS > 0.15) → Evaluate: mesh geometry issue → try per-scene NeRF
```

---

## Backlinks

- ↑ [INDEX](../INDEX.md) | [NOVEL_VIEW_QUALITY_STRATEGY](../experiments/NOVEL_VIEW_QUALITY_STRATEGY.md)
- ↔ [MAMMAL_MESH_RENDERING_PIPELINE](MAMMAL_MESH_RENDERING_PIPELINE.md) | [DIFIX_TRAINING_STRATEGY](../experiments/DIFIX_TRAINING_STRATEGY.md)
- ↓ Implementation: `mouse_extensions/model/neural_texture.py` (to be created)

---

*Created: 2026-03-22 | FaceLift Phase 2 — Neural Texture Analysis*
