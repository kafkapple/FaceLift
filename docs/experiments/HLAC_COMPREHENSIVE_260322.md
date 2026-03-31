# HLAC Comprehensive Analysis Report — 260322

> **Script**: `mouse_extensions/behavior/hlac_comprehensive.py`
> **Data**: 2992 aligned frames (KP ∩ Gaussian ∩ Temporal)
> **Labels**: K-means (K=8) on body-centered keypoints
> **Created**: 2026-03-22

---

## Hypothesis

**H**: Foreground-masked Gaussian covariance features complement keypoints for behavior classification.

**Motivation**: If Gaussian features capture information that keypoints miss (e.g., body shape, fur texture dynamics), combining them should improve behavior classification beyond keypoints alone.

## Method

4-part analysis addressing circular bias concerns from previous HLAC sessions:

1. **Multi-classifier comparison** — LinearSVM, RF, MLP (not just LinearSVM)
2. **Paired t-test** — statistical complementarity test
3. **Body-part analysis** — per-joint-group feature subsets
4. **CCA** — canonical correlation between KP and Gaussian features

## Results

### Part 1: Multi-classifier + PCA Sweep

| Feature | Dim | LinearSVM | RF | MLP | Best |
|---------|:---:|:---------:|:--:|:---:|:----:|
| **KP_centered** | 66 | 0.960 | 0.971 | **0.980** | **0.980** |
| Gaussian_raw | 31 | 0.833 | 0.928 | 0.928 | 0.928 |
| Cov_N2_static (fg) | 154 | 0.867 | 0.924 | 0.933 | 0.933 |
| Cov_N2_temporal (fg) | 198 | 0.791 | **0.955** | 0.847 | 0.955 |
| Cov_N2_combined (fg) | 352 | 0.915 | 0.949 | 0.934 | 0.949 |
| PCA(10)_Cov | 10 | 0.525 | 0.875 | 0.880 | 0.880 |
| KP + PCA(10)_Cov | 76 | 0.961 | 0.972 | 0.978 | 0.978 |
| PCA(20)_Cov | 20 | 0.648 | 0.893 | 0.899 | 0.899 |
| KP + PCA(20)_Cov | 86 | 0.957 | 0.972 | 0.974 | 0.974 |
| PCA(50)_Cov | 50 | 0.782 | 0.906 | 0.917 | 0.917 |
| KP + PCA(50)_Cov | 116 | 0.962 | 0.971 | 0.970 | 0.970 |
| KP + Cov_static | 220 | 0.955 | 0.969 | 0.967 | 0.969 |

**Key finding**: KP alone (0.980 MLP) consistently outperforms all combinations. Adding Gaussian features never improves over KP baseline.

### Part 2: Complementarity Test (Paired t-test)

Best PCA k = 10 (by RF).

| Classifier | Combined F1 | Delta vs KP | p-value | Complementary? |
|-----------|:-----------:|:-----------:|:-------:|:--------------:|
| LinearSVM | 0.961 | +0.001 | 0.735 | NO |
| RF | 0.972 | +0.001 | 0.268 | NO |
| MLP | 0.978 | -0.002 | 0.294 | NO |

**All 3 classifiers agree**: no statistically significant improvement from adding Gaussian features.

### Part 3: Body-part Analysis (RF)

| Body Part | KP F1 | Cov F1 | KP+Cov F1 | Delta |
|-----------|:-----:|:------:|:---------:|:-----:|
| head | 0.963 | 0.927 | 0.965 | +0.002 |
| **torso** | 0.928 | 0.917 | **0.950** | **+0.022** |
| forelimb | 0.966 | 0.916 | 0.962 | -0.004 |
| hindlimb | 0.970 | 0.913 | 0.970 | -0.000 |
| **tail** | 0.940 | 0.934 | **0.952** | **+0.012** |

**Interpretation**: Torso and tail show modest complementarity — these are regions with fewer keypoints (body_middle + tail_root only for torso, 2 joints for tail) where Gaussian shape information adds marginal value.

### Part 4: CCA (Canonical Correlation Analysis)

Top 10 correlations: `0.856, 0.841, 0.814, 0.743, 0.738, 0.643, 0.602, 0.560, 0.417, 0.145`

| Metric | Value |
|--------|:-----:|
| Mean correlation | 0.636 |
| Max correlation | 0.856 |
| Interpretation | **HIGH** — significant shared information |

CCA confirms that Gaussian features largely encode the same behavioral information as keypoints (redundant, not complementary).

## Conclusions

1. **Gaussian covariance features do NOT complement keypoints** for behavior classification (K-means labels)
2. **Exception**: torso (+2.2%) and tail (+1.2%) show marginal improvement — likely due to sparse keypoint coverage in these regions
3. **CCA**: high canonical correlations (mean 0.636) confirm feature spaces are largely overlapping
4. **Classifier consistency**: all 3 classifiers (LinearSVM, RF, MLP) agree — result is robust to classifier choice

## Caveats

- **Circular bias**: Labels are K-means clusters on keypoints → structurally favors KP features
- **Ground truth needed**: s-DANNCE HLAC 13-class behavior labels would provide unbiased evaluation
- **Limited scope**: Only tested covariance (N>=2) features, not raw Gaussian parameters or learned embeddings

## Impact on Paper Narrative

- ~~"Gaussian features complement keypoints for behavior analysis"~~ → **Rejected**
- **Revised narrative**: Gaussian 3D reconstruction enables novel view synthesis and spatial analysis that keypoints alone cannot provide (rendering, not classification)
- The value of GS-LRM is in **3D rendering capability**, not in behavior feature complementarity

---

## Related

- Previous HLAC: `outputs/m5t2_hlac_analysis/` (initial), `outputs/m5t2_hlac_analysis_n2/` (N>=2), `outputs/m5t2_hlac_fg_n2/` (foreground)
- Bias correction: `docs/experiments/PAST_SESSION_AUDIT_260322.md` _(archived)_
- Memory: `feedback_confirmation_bias.md`

---

*Created: 2026-03-22 | HLAC Comprehensive Analysis*
