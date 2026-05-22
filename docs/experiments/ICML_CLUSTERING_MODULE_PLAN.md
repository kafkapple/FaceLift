# ICML Workshop — Unsupervised Clustering Module Plan

> **Created**: 2026-04-18 | **Status**: PLAN (no scaffold/code yet)
> **Deadline reference**: Handoff `260416_1805.md` — ICML 2026 workshop D-7, deadline 2026-04-24 AoE
> **Scope**: Spec-only (per Q2=A from plan approval 2026-04-18)
> **Feature SSOT**: [`../specs/VISUAL_EMBEDDING_SSOT.md`](../specs/VISUAL_EMBEDDING_SSOT.md)

---

## Venue (confirmed 2026-04-18)

**ICML 2026 Workshop AI4Science** — *"AI Scientists – Tools, Co-authors, or Founders?"*
- Deadline: **2026-04-24 AoE** (D-6 from 2026-04-18)
- Format: 4–8 pages, ICML style, double-blind, non-archival (NeurIPS resubmission preserved)
- Track (chosen): **Original Research Track** — "new algorithms, systems, or scientific findings"
- Alt track if needed: **Highlight Track** — "surveys, benchmarks, or synthesis work"
- Scope: biology explicitly included (alongside physics/chem/climate/eng)
- Reviewers: 300+ pool, 50+ ACs, mixed backgrounds → **narrative must be accessible to non-neuroscience ML reviewer**
- OpenReview: `ICML.cc/2026/Workshop/AI4Science`
- CFP: https://ai4sciencecommunity.github.io/icml26/call

## 0. Motivation (why this module)

### 0.1 Research question (revised 2026-04-18 per MoA + devil audit)

> ~~"Is rotation-invariance necessary for behavior clustering?"~~ **DROPPED** — confounded by deployment assumption (our fixed 6-cam rig vs PS portable orientation). Research question re-framed to what our data can honestly answer.

**Revised question**: *"In a fixed multi-camera rig, which feature family (image-derived vs geometry-derived) best supports unsupervised discovery of behavior structure?"*

### 0.2 Empirical gap (PS paper)

PS (arXiv:2505.18342, NeurIPS 2025 poster) uses 50d visual embedding **only** for:
1. k-NN keypoint regression (R²)
2. Logistic regression on 4 mouse classes (Walk/HeadUp/Still/Groom), 2×binary finch
3. Human-preference study (k=1 pairwise retrieval)

**No unsupervised clustering evaluation** (silhouette / ARI / NMI / k-means) in the paper. This workshop fills that gap while also introducing geometry-derived features from 3DGS that PS does not consider.

### 0.3 Drift warning

> ⚠️ **COV 88d ≠ covariance.** The `260414_decomposition_report.html` claim ("COV 88d = Static 96.7% + Dynamic 3.3%") uses 88d velocity features (`bodypart_delta`), NOT Gaussian covariance (which is 154d). See [`../specs/VISUAL_EMBEDDING_SSOT.md`](../specs/VISUAL_EMBEDDING_SSOT.md) §0 drift alert. Any ICML figure using "COV" must cite the 154d row.

---

## 1. Target Behavior Label Set

### 1.1 HLAC 8-class (SSOT for supervised eval)

| # | Class | Source |
|:-:|---|---|
| 1-8 | s-DANNCE HLAC Table S2 canonical mapping | Project memory `project_rat_hlac_discovery.md` + `mouse_extensions/behavior/hlac_m5t2.py` |

- Labels aligned to frame index via `hlac_comprehensive.py` infrastructure.
- Available for both rat (2968 frames, RAT2 v6 retracted → wait for v7/v9 camera-holdout) and mouse (M5t2 3600 frames).

### 1.2 PS 4-class replication track (apples-to-apples)

For strict PS baseline comparison: mouse only, {Walk, HeadUp, Still, Groom}, 60/40 random split, logistic regression. Fair comparison requires re-labeling our data to PS's 4 classes (or restricting HLAC 8 → 4 superset).

---

## 2. Module Architecture (spec)

```
mouse_extensions/behavior/clustering/            # NEW (to be created in separate session)
├── __init__.py
├── driver.py              # unified CLI: python -m ...clustering.driver --feature FEAT_NAME
├── pipeline.py            # PCA → cluster → eval
├── methods.py             # KMeans, HDBSCAN, GMM, AgglomerativeClustering adapters
├── metrics.py             # silhouette, DB, ARI, NMI, purity, per-cluster entropy
├── probe.py               # linear/MLP probe for supervised comparison (wraps sklearn)
└── configs/
    ├── default.yaml       # K sweep {4,6,8,10,12}, PCA dims {10,25,50,100}, seed list
    └── icml_workshop.yaml # fixed config for workshop results
```

### 2.1 Interface contract

```python
# pipeline.py
def cluster_features(
    feature_npz_path: str,              # from VISUAL_EMBEDDING_SSOT §5
    method: Literal["kmeans","hdbscan","gmm","agglom"],
    k: int | None,                      # None → auto via silhouette
    pca_dim: int | None,                # None → no PCA
    seed: int = 42,
    mask_frame_jumps: bool = True,
) -> ClusterResult:
    """
    Returns:
        ClusterResult with .labels (N,), .metrics dict, .embedding_2d (N,2 for viz)
    """
```

### 2.2 Inputs (all feature NPZs listed in VISUAL_EMBEDDING_SSOT §0)

Supported via duck-typed NPZ loader — detects shape `(N_frames, D)`, dispatches to PCA + cluster.

### 2.3 Outputs

```
outputs/analysis/{species}/clustering/{exp_id}/
├── cluster_labels.npy                 # (N_frames,)
├── metrics.json                       # {silhouette, db, ari, nmi, ...}
├── viz_umap.html                      # interactive plotly
├── confusion_vs_hlac.png              # cluster × HLAC heatmap
└── config_snapshot.yaml               # full config + git commit
```

---

## 3. Experiment Design (for ICML workshop paper)

### 3.1 MVP scope (revised per MoA 3/3 consensus, 2026-04-18)

**3 features only** for workshop submission (5-feature full benchmark deferred to NeurIPS full paper):

| Feature | Dim | Role | Represents |
|---|:-:|---|---|
| **PCA-pose-auto** (PS-inspired, non-adv ⚠️) | 50 | Prior-art baseline | Rotation-invariant image embedding |
| **DINOv2 CLS** | 768 → 50 (PCA) | Foundation-model baseline | Modern frozen ViT |
| **Gaussian covariance** | 154 → 50 (PCA) | Our contribution | 3D geometric shape from 3DGS |

**Deferred to appendix / future work**: Bodypart 308d, PointNet 256d (experimental), Temporal 129d (label-leakage risk §5.2).

**Species**: Mouse M5t2 only (N=3,600). Rat deferred — RAT2 v6 retracted; v7+ pending reliability.

### 3.2 Primary track — **matched PCA-50d** (headline)

All features projected to 50d via vanilla PCA before clustering (addresses S2 scale/dim confound). This is the main result table.

| Axis | Values |
|---|---|
| Feature | PCA-pose-auto 50d (native) · DINOv2→50d · Cov→50d |
| Method | KMeans (primary), HDBSCAN (secondary robustness) |
| k (KMeans) | {4, 6, 8, 10, 12} + silhouette auto |
| Labels | HLAC **4-class** (PS-matched headline) + HLAC 8-class (supplementary) |
| Seeds | 5 seeds × 5-fold |

### 3.3 Supplementary track — native dimensions

Same features at native dim (50/768/154). Reported in second table with explicit note on sample-per-dim ratio confound.

### 3.4 Metrics

- **Internal** (no label): silhouette ↑, Davies-Bouldin ↓, Calinski-Harabasz ↑
- **External** (vs HLAC): ARI ↑, NMI ↑, cluster purity ↑
- **Linear probe** (supplementary): logistic regression + macro-F1 matching PS 60/40 split

### 3.5 Declared headline metric (C2 mitigation)

> **Pre-registered primary claim**: ARI on HLAC **4-class** with **PCA-50d matched** dimensionality. Any other numbers are secondary/ablation. This is declared BEFORE running experiments to prevent cherry-pick accusation.

### 3.6 Representation-similarity analysis (paired with clustering)

For AI4Science reviewer accessibility, add a **complementary angle**: "do these feature families encode the same behavior structure or distinct views of it?" This is answered by feature-pair similarity rather than absolute clustering performance.

| Method | Pair | Interpretation |
|---|---|---|
| **CCA** (reuse `hlac_comprehensive.py:321`) | PCA-pose-auto ↔ DINOv2 PCA-50d; PCA-pose-auto ↔ Cov PCA-50d; DINOv2 ↔ Cov | max canonical r > 0.7 = HIGH overlap, < 0.4 = LARGELY INDEPENDENT (thresholds from existing code) |
| **Linear CKA** (Kornblith 2019) | All 3 pairs | Orthogonal-invariant similarity 0-1 |
| **RBF CKA** | All 3 pairs | Nonlinear counterpart |

**Why CKA in addition to CCA**: CCA is sensitive to rotation/scaling within each space; CKA is rotation-invariant and the modern standard for representation analysis in ML literature. Both reported to triangulate.

**Deliverable**: 3×3 similarity matrix (one per metric) → paper Figure 2 (post-MVP if time). CKA implementation: ~50 LOC (reuse `sklearn.metrics.pairwise` kernel + centering), 1 day.

---

## 4. Deliverables & Timeline

### 4.1 Revised MVP timeline (D-6 from 2026-04-18, MoA Q5 consensus: **headline table first**)

| Day | Deliverable | Gate |
|:-:|---|---|
| **D-6 (Apr-19)** ⭐ | **Headline table**: ARI on HLAC 4-class, PCA-50d, 3 features, KMeans k=4. M5t2 only. | GO/NO-GO: if top feature ARI < 0.15, pivot to position paper |
| D-5 | Extend to supplementary metrics (silhouette, DB, NMI) + k-sweep {4,6,8,10,12} + **CCA pairs (reuse existing code)** | |
| D-4 | Surrogate-label shuffle control (C1) + HDBSCAN robustness + UMAP viz + **CKA linear+RBF impl** (~50 LOC) | |
| D-3 | Figure generation (bar plot ARI, UMAP grid, confusion heatmap, **3×3 CCA/CKA similarity matrix**) | |
| D-2 | Paper writeup (4-8 pages workshop format, **AI4Science biology framing**) | |
| D-1 | `/audit --full` + polish + surplus ablation if time | |
| **D-0 (2026-04-24 AoE)** | Submit to `ICML.cc/2026/Workshop/AI4Science` | |

### 4.1a Paper narrative for AI4Science (biology track)

**Title draft (v2, post-deliberation)**: "Feature Classes for Unsupervised Animal Behavior Discovery: A Benchmark of Linear, Foundation-Model, and 3D-Geometric Embeddings"

**Opening sentence** (per Claude F4 deliberation): *"Recovering structured behavioral repertoires from raw video is a geometric inverse problem: which latent embedding of animal posture best preserves the manifold structure that separates biologically distinct actions?"*

**AI4Science theme alignment** (per OpenAI V2 + Gemini A2): frame the geometric feature pipeline as an **automated ethogram-discovery tool** — an "AI co-author" in behavioral science that proposes cluster structure for human ethologists to validate. This fits the workshop question "AI Scientists: Tools, Co-authors, or Founders?" by positioning our pipeline as a *tool* that scales human behavior annotation.

**Framing** (accessible to non-neuroscience ML reviewer):
1. Biology problem: unsupervised discovery of ethogram structure from video (no hand labels needed for production use)
2. ML gap: PoseSplatter (NeurIPS 2025) provides 50d embedding but no clustering evaluation; it is unknown whether image-derived 50d outperforms 3D-geometry-derived features for behavior discovery
3. Contribution: (a) first benchmark of **3 feature families** (linear-reconstructive, foundation-model, 3D-geometric) on unsupervised behavior clustering, (b) representation-similarity analysis via CCA + CKA showing whether geometry features encode complementary structure, (c) pre-registered methodology (matched PCA-50d, surrogate-shuffle control, dual 4/8-class eval)

**Track decision tree**:
- If headline ARI > 0.25 and CKA shows orthogonal information → **Original Research Track**
- If headline ARI marginal but CCA/CKA reveals interesting structure → **Highlight Track** (reframe as "benchmark/synthesis")

### 4.2 Day-1 priority (per MoA Q5)

**Before writing a single sentence of the paper**: generate the headline table. If PCA-pose-auto, DINOv2-PCA-50d, or Cov-PCA-50d ARI < 0.15 on HLAC 4-class, the paper has no empirical claim — pivot to position paper (discussing clustering gap as open problem) rather than empirical benchmark.

### 4.3 Gating decision tree

```
Headline table (D-6)
├── Top feature ARI ≥ 0.25 → strong empirical claim, Original Research Track, proceed full plan
├── 0.15 ≤ ARI < 0.25   → modest claim, emphasize methodology, consider Highlight Track
└── ARI < 0.15          → Highlight Track (negative-result benchmark) — pre-written fallback abstract ready
```

### 4.4 D-6 FIRST ACTIONS (must complete before experiments) — per final deliberation

Three **blocking** tasks at start of D-6 (Apr-19), estimated total ~3 hours:

1. **30 min — OpenReview `ICML.cc/2026/Workshop/AI4Science`** submission system check + account setup.
2. **30 min — PS OpenReview `KuXnKedjAj` check** (devil RD1): read reviewer comments + author responses. If authors mention attempting clustering → narrative collapses, must reframe immediately.
3. **2 hours — Pre-write Highlight Track fallback abstract** (per Claude F3): 200-word abstract framed as "negative-result benchmark: none of current feature families achieve ARI > 0.25 for unsupervised mouse behavior discovery at N=3,600, indicating open problem." Keep in `_backup/` directory; promote if headline gate fails.

Only AFTER these 3 complete, run headline table.

### 4.5 CKA is optional — do not let it block submission

Per Claude F1: CKA (~50 LOC) is scheduled D-4 but is *decorative* if headline clustering claim holds. If D-4 slips, **drop CKA entirely** and rely on CCA (existing). Paper can make a clean claim without CKA.

---

## 5. Scientific Risks & Falsification Tests

### 5.1 Risk: "clustering quality is a reconstruction artifact"

From 260414 audit + `project_view_ablation_methodology_slip.md`: reconstruction noise may drive apparent "dynamic" signal.

**Falsification**: run clustering on **synthetic 3DGS with known pose noise model** (planned in `project_view_ablation_methodology_slip`). If cluster structure emerges from noise-only synthetic, kill claim.

### 5.2 Risk: Temporal 129d ↔ HLAC label leakage (devil C1)

HLAC 8-class labels are derived from keypoint trajectories via s-DANNCE canonical clustering. Temporal 129d is derived from the same keypoint trajectories (centroid + bodypart Δ + Kabsch + window). ARI-vs-HLAC for Temporal features measures keypoint → HLAC invertibility, NOT clustering quality.

**Mitigation** (MoA 2/3 consensus):
1. Exclude Temporal 129d from MVP primary table (§3.1) — keep PCA-pose-auto, DINOv2, Cov only.
2. Report Temporal as **upper-bound oracle** in appendix with explicit warning.
3. **Surrogate-label shuffle control** (MoA Q3, spec per Phase 7a I3):
   - Define window = consecutive run of identical HLAC label (= one "behavior bout").
   - Shuffle: permute the order of bout-labels globally (preserving each bout's internal frame identity but randomizing which bout belongs to which class).
   - n_permutations = 1000. Report `ARI_shuffled_mean ± std`.
   - Headline metric = `ARI_observed − ARI_shuffled_mean` ("surplus ARI" — signal beyond label-derivation artifact).
   - Reference impl: `mouse_extensions/behavior/clustering/controls.py` (to be created in module scaffold).

### 5.3 Risk: HLAC labels correlate with camera artifacts

If HLAC class boundaries correlate with camera exposure / compression / frame-jumps, external ARI inflates trivially.

**Test**: compute ARI between HLAC labels and feature-agnostic baselines (mean pixel intensity per frame, mean opacity per frame). If |ARI| > 0.1, feature-agnostic variable is a confounder → flag in paper.

### 5.3 Risk: 4-class vs 8-class unfair comparison

Extending PS 4-class to HLAC 8-class makes our numbers look stricter; reviewers may suspect cherry-pick.

**Mitigation**: report **both** 4-class (matched PS) and 8-class (our extension) side-by-side. Never report only 8-class.

### 5.4 Risk: Rat2 dataset still unreliable

Per `project_rat_audit_260407`: Rat2 v6 RETRACTED (frame-holdout ≠ camera-holdout). v7+ status unclear.

**Mitigation**: use mouse M5t2 (known-good) as primary; rat as sensitivity analysis only.

---

## 6. Open Questions (needs user decision)

| # | Question | Status / Default |
|:-:|---|---|
| Q1 | Paper scope — methods (clustering benchmark) or application (specific finding)? | **Methods paper** (safer) |
| Q2 | PS 50d replication feasibility — `jackgoffinet/pose-splatter` is public ✅. Adv-PCA needs rotation-angle concomitant variable; fixed-rig has no such label. | **DROPPED as "replicate"** (Gemini A1). Reframed as **PCA-pose-auto**: PS-inspired front-end (32 renders + ResNet18 + SH) + vanilla PCA. Paper explicitly states this is **NOT** a PoseSplatter reproduction (§3.1a). |
| Q3 | Rat or mouse-only? | **Mouse M5t2 only** for submission (RAT2 v6 retracted, v7+ pending). |
| Q4 | Workshop venue | ✅ **CONFIRMED 2026-04-18**: AI4Science @ ICML 2026, "AI Scientists – Tools, Co-authors, or Founders?", Original Research Track (primary) / Highlight Track (fallback). Deadline Apr 24 AoE, non-archival, 4-8p. |
| Q5 | Check PS OpenReview for authors' stance on clustering omission (devil RD1) | 30-min task for D-6 — frames narrative: "deliberate omission" vs "oversight". |
| Q6 | ~~Adv-PCA vs vanilla PCA for PS 50d replicate~~ **RESOLVED via Gemini A1**: do NOT call it a "replicate". Use name **PCA-pose-auto** with explicit §3.1a caveat. See final deliberation record (Phase 6e). |

---

## 7. Cross-References

- **Feature SSOT**: [`../specs/VISUAL_EMBEDDING_SSOT.md`](../specs/VISUAL_EMBEDDING_SSOT.md)
- **Theory / PS paper extraction**: Obsidian `docs/research/260418_visual_embedding_ps_vs_fl.md`
- **Prior ICML scope** (project memory): `project_neurips_scope_v3.md`
- **ICML Workshop handoff**: `~/.agent/logging/handoffs/260416_1805.md`
- **Methodology slip precedent**: `REPORT_260408_VIEW_ABLATION_METHODOLOGY_SLIP.md`, `project_view_ablation_methodology_slip.md`

---

*FaceLift | ICML Clustering Module Plan | 2026-04-18*
