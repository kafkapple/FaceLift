# Visual Embedding SSOT (Single Source of Truth)

> **Created**: 2026-04-18 | **Scope**: All per-frame visual/shape/motion feature extractors in FaceLift
> **Role**: Implementation reference — I/O specs, dimensions, call sites, status
> **Theory/analysis**: Obsidian `docs/research/260418_visual_embedding_ps_vs_fl.md`
> **Usage**: Any module consuming feature NPZs MUST cite this file.

---

## 0. Canonical Dimension Table (⚠️ drift-prone)

| Feature | Dim | Shape | File | Canonical NPZ |
|---|:---:|---|---|---|
| Gaussian raw summary | **22-31** | `(N_frames, D)` | `extract_gaussian_features.py` | `outputs/features/gaussians/gaussian_features.npz` |
| Bodypart delta (velocity) | **88** | `(N_frames, 22*3 + 22)` | `extract_temporal_features.py` | key `bodypart_delta` in `temporal_features.npz` |
| Temporal concat (centroid+bp+rigid+window) | **129** | `(N_frames, 7+88+25+9)` | `extract_temporal_features.py` | `temporal_features.npz` full |
| **Covariance static** (per-joint shape) | **154** | `(N_frames, 22*7)` | `extract_covariance_features.py:326` (stack), `:348` (save) | `covariance_static.npy` |
| Covariance temporal | **198** | `(N_frames, 22*9)` | `extract_covariance_features.py:341` (compute), `:349` (save) | `covariance_temporal.npy` |
| PointNet global | **256** | `(N_frames, 256)` | `pointnet_gaussian.py:46-62` | `pointnet_features.npz` |
| Bodypart static (per-joint) | **308** | `(N_frames, 22*14)` | `extract_bodypart_features.py:279` | `bodypart_features.npy` |
| Bodypart flow (+flow option) | **308+132** | `(N_frames, 22*14 + 22*6)` | same | same + flow key |
| DINOv2 CLS (multi-view mean-pooled) | **768** | `(N_frames, 768)` | `extract_dinov2_features.py:135` (mean-pool), `:138-146` (save) | `outputs/features/dinov2/dinov2_features.npz` |
| DINOv2 masked (3 strategies) | **768** | `(N_frames, 768)` × 3 | `extract_dinov2_masked.py` | `dinov2_{masked,cropped,full}_features.npz` |
| DINOv2 patch concat (experimental) | **4608** | `(N_frames, 6*768)` | `run_dinov2_patch.py` | — (clustering eval only) |

### ⚠️ Drift alert — "COV 88d" is WRONG

**Incident (2026-04-14)**: `260414_decomposition_report.html` claims "COV 88d = Static 96.7% + Dynamic 3.3%".
**Reality**: Covariance feature is **154d** (22×7). The **88d** is `bodypart_delta` (keypoint velocity), NOT Gaussian covariance.
**Action for report authors**: either (a) rename to "temporal velocity decomposition" and justify 88d, or (b) regenerate with actual 154d covariance.
**Rule**: Every future doc citing a dimension must link its row in §0 table above.

---

## 1. Extractors — Input / Method / Status

### 1.1 DINOv2 pathway (2D image)

| File | Input | Method | Output | Status |
|---|---|---|---|---|
| `extract_dinov2_features.py` | 6-cam RGB 224×224 (frame-jump masked) | DINOv2-ViT-B/14 forward → per-view 768 CLS → mean-pool 6 views | `dinov2_features.npz` (768d) | Active |
| `extract_dinov2_masked.py` | RGBA with 3 masking strategies (masked/cropped/full) | Same backbone + alpha preprocessing | `dinov2_{strategy}_features.npz` (768d × 3) | Active |
| `run_dinov2_patch.py` | Same RGB | Patch-level pool + temporal windowing | Not persisted | Experimental |

**Dependencies**: HuggingFace `facebook/dinov2-base`. Frozen (no fine-tune).

### 1.2 Gaussian shape pathway (3D)

| File | Input | Method | Output | Status |
|---|---|---|---|---|
| `extract_gaussian_features.py` | Per-frame PLY (xyz, opacity, scale, rot) | Summary stats: means, std, pct, PCA eigvals | `gaussian_features.npz` (22-31d) | Active baseline |
| `extract_covariance_features.py` | Per-frame NPZ (xyz, opacity, scale_raw, rotation_raw) | Σ = R·diag(exp(s))²·R^T → eigvals + per-joint aggregate | `covariance_{static,temporal}.npy` (154d, 198d) | Active |
| `extract_bodypart_features.py` | Gaussian NPZ + keypoints (NN assignment) | Per-joint stats (shape+opacity+scale+spread) | `bodypart_features.npy` (308d, +132 flow) | Active |
| `pointnet_gaussian.py` | Point cloud (xyz+scale+opacity, ≤4096) | Deterministic PointNet (no training) + max-pool | 256d | Experimental |

**Dependencies**: GS-LRM inference checkpoint per experiment.

### 1.3 Keypoint kinematics pathway

| File | Input | Method | Output | Status |
|---|---|---|---|---|
| `extract_temporal_features.py` | MAMMAL-fit keypoints `(N, 22, 3)` | Centroid (7) + bodypart Δ (88) + Kabsch rigid/nonrigid (25) + sliding window (9) | `temporal_features.npz` (129d) | Active |

**Boundary masking**: single-frame forward-diff invalidation — `extract_temporal_features.py:227-248` zeroes the frame immediately before each `np.diff > 1` gap (not a ±N window). Covariance uses an explicit 15-frame `FRAME_JUMPS` set (3 session boundaries × 5 frames each, `extract_covariance_features.py:31-32`). Intent similar (exclude boundary-contaminated kinematics); implementations divergent in scope (forward-diff 1 frame vs. fixed ±2 frames around each of 3 boundaries). Candidate for unification.

---

## 2. Consumer Sites

| Consumer | Features Used | Classifier | File |
|---|---|---|---|
| HLAC comprehensive | KP centered (66) + Temporal (129) + Gauss raw (31) + Cov static (154) + Cov temporal (198) | LinearSVM + RF + MLP (sklearn) | `hlac_comprehensive.py:120-137` |
| s-DANNCE H1 probe | Raw sDANNCE KP (69) + derived kinematics | LogisticRegression L2 + GroupKFold | `h1_sdannce_probe.py:291` |
| Unified pipeline (clustering) | Bodypart subset (variable 7-66d), z-scored | KMeans + UMAP + (optional) HDBSCAN | `unified_pipeline.py:44-80` |

### Classifier head implementations

- **sklearn shallow** (dominant): `LinearSVC(C=1.0)`, `RandomForestClassifier(n_estimators=200, max_depth=15)`, `MLPClassifier(hidden_layer_sizes=(128,64))` — all StandardScaler-wrapped, 5-fold StratifiedKFold.
- **LogisticRegression probe**: L2, `class_weight="balanced"`, `max_iter=1000`, GroupKFold for leakage-safe CV.
- **PyTorch MLP head**: only inside `pointnet_gaussian.py` (feature extractor, not classifier). No end-to-end PyTorch classifier on extracted features.

---

## 3. Clustering Inventory (pre-ICML)

| Method | Files | Notes |
|---|---|---|
| KMeans | ~25 call sites across `behavior/` | K ∈ {4,6,8,10,12}, silhouette auto-select, n_init=5-10 |
| HDBSCAN | `analyze_bams_embeddings.py` only | Not in core extractors |
| UMAP | `analyze_bams_embeddings.py`, `unified_pipeline.py` | Visualization + optional DR |
| AgglomerativeClustering | **None** | — |
| GMM | **None** | — |

**Gap**: no unified clustering driver — every extractor embeds ad-hoc KMeans.

---

## 4. Frame Alignment & Masking

| Alignment concern | Rule |
|---|---|
| Per-frame vs per-clip | All features listed are per-frame (shape `(N_frames, D)`). Temporal features derive from neighbors but output per-frame. |
| Frame-jump masking | DINOv2, covariance, temporal — all apply frame-jump masks before feature computation. 5-frame (temporal) vs 15-frame (covariance) divergence flagged above. |
| N≥2 visibility filter | Covariance recommends `min_valid_views ≥ 2` (argparse `--n_filter` at `extract_covariance_features.py:253-254`; applied at `:308-309` via `counts >= args.n_filter`) → ~16% frame loss. |
| Common frame index | Consumers (`hlac_comprehensive.py:116`) compute intersection of frame indices across all feature files before concat. |

---

## 5. Output Paths SSOT

All output paths resolved via `mouse_extensions/behavior/paths.py`. Do NOT hardcode paths — use factory functions `get_feature_dir(species, feature_type)`.

```
outputs/features/{species}/
├── dinov2/dinov2_features.npz                    # 768d
├── dinov2_masked/dinov2_{strategy}_features.npz  # 768d × 3
├── gaussians/gaussian_features.npz               # 22-31d
├── covariance/covariance_{static,temporal}.npy   # 154d, 198d
├── bodypart/bodypart_features.npy                # 308d (+132 flow)
├── temporal/temporal_features.npz                # 129d (88d via key)
└── pointnet/pointnet_features.npz                # 256d (experimental)
```

---

## 6. Analysis Methods Inventory (feature comparison + evaluation)

Analysis methods operate on feature NPZs from §5. Currently implemented vs planned:

### 6.1 Clustering (unsupervised discovery)

| Method | Status | Call sites |
|---|:-:|---|
| KMeans + silhouette auto-k | Implemented | ~25 in `behavior/` |
| HDBSCAN | Implemented (limited) | `analyze_bams_embeddings.py` |
| UMAP (viz/DR) | Implemented | `analyze_bams_embeddings.py`, `unified_pipeline.py` |
| GMM | Not implemented | — |
| Agglomerative (hierarchical) | Not implemented | — |

### 6.2 Representation similarity (feature-pair comparison)

| Method | Status | Code |
|---|:-:|---|
| **CCA** (canonical correlation) | Implemented | `hlac_comprehensive.py:321` (kp ↔ cov), `h1_sdannce_probe.py:361` (generic pair independence test, criterion: ≥2 components with r<0.7) |
| **CKA** (centered kernel alignment, Kornblith 2019) | **Planned** | — (add to ICML module, linear + RBF variants) |
| **HSIC** (Hilbert-Schmidt Independence) | Not implemented | — |
| Pearson correlation matrix | ad-hoc | — |
| Procrustes distance | Not implemented | — |

**CCA thresholds** (from `hlac_comprehensive.py:336-341`): max_r > 0.7 = HIGH shared info, 0.4-0.7 = MODERATE overlap, < 0.4 = LARGELY INDEPENDENT.

### 6.3 Supervised probe (feature quality measurement)

| Probe | Status | Code |
|---|:-:|---|
| LinearSVC + RF + MLP (StratifiedKFold) | Implemented | `hlac_comprehensive.py:120-137` |
| LogReg L2 + GroupKFold (leakage-safe) | Implemented | `h1_sdannce_probe.py:291` |
| PyTorch end-to-end classifier | Not implemented | — |

### 6.4 Cluster-quality metrics

All standard `sklearn.metrics.cluster`: silhouette, davies_bouldin, calinski_harabasz, adjusted_rand_score, normalized_mutual_info_score. Scattered use across `visualize_clusters.py`, `pilot_3c_3d.py`, `hlac_classification.py`, `analyze_bams_embeddings.py`. **Recommended**: unify in `mouse_extensions/behavior/clustering/metrics.py` (per module plan).

---

## 7. Cross-References

- **Module plan (ICML workshop)**: [`docs/experiments/ICML_CLUSTERING_MODULE_PLAN.md`](../experiments/ICML_CLUSTERING_MODULE_PLAN.md)
- **Theory / PS comparison**: Obsidian `docs/research/260418_visual_embedding_ps_vs_fl.md`
- **PS baseline metrics**: `baselines/pose_splatter/{paper_standard_evaluation,posesplatter_fair}.json`
- **Canonical project entry**: [`docs/FACELIFT_SSOT.md`](../FACELIFT_SSOT.md)
- **Index**: [`docs/INDEX.md`](../INDEX.md) (register this file under specs)

---

*FaceLift | Visual Embedding SSOT | 2026-04-18*
