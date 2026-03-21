"""Comprehensive behavior clustering metrics module.

Includes:
- Unsupervised: Silhouette, Calinski-Harabasz, Davies-Bouldin
- Temporal: TPI, Entropy Rate, Bout Duration, Transition Rate, Temporal Consistency
- Statistical: Permutation test, Bootstrap CI, Temporal shuffle baseline
- Supervised (when GT available): ARI, NMI, MI

Usage:
    from mouse_extensions.behavior.metrics import compute_all_metrics, permutation_test
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    mutual_info_score,
)


@dataclass
class ClusteringMetrics:
    """Complete metrics for a clustering result."""

    # Unsupervised
    n_clusters: int = 0
    n_frames: int = 0
    n_noise: int = 0
    noise_ratio: float = 0.0
    silhouette: float = float("nan")
    calinski_harabasz: float = float("nan")
    davies_bouldin: float = float("nan")

    # Temporal
    bout_mean_sec: float = 0.0
    bout_median_sec: float = 0.0
    bout_std_sec: float = 0.0
    n_bouts: int = 0
    transition_rate: float = 0.0
    tpi_mean: float = 0.0
    tpi_per_cluster: list = field(default_factory=list)
    entropy_rate: float = 0.0
    temporal_consistency: float = 0.0  # 1 - (transitions / max_transitions)

    # Supervised (when GT available)
    ari: float = float("nan")
    nmi: float = float("nan")
    mi: float = float("nan")

    # Flickering / Distribution
    short_bout_ratio: float = 0.0  # fraction of bouts < threshold (flickering proxy)
    short_bout_threshold_sec: float = 0.3  # default 300ms
    bout_cv: float = 0.0  # coefficient of variation of bout durations
    label_autocorrelation: float = float("nan")  # lag-1 autocorrelation

    # Statistical
    silhouette_ci_lower: float = float("nan")
    silhouette_ci_upper: float = float("nan")
    p_value_vs_shuffle: float = float("nan")

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (list, np.ndarray)):
                d[k] = [round(float(x), 4) for x in v]
            elif isinstance(v, float):
                d[k] = round(v, 4) if not np.isnan(v) else None
            else:
                d[k] = v
        return d


def compute_all_metrics(
    labels: np.ndarray,
    features: np.ndarray,
    fps: float = 20.0,
    gt_labels: Optional[np.ndarray] = None,
    bootstrap_n: int = 100,
    compute_significance: bool = True,
) -> ClusteringMetrics:
    """Compute all available metrics for a clustering result.

    Args:
        labels: (T,) cluster assignments (-1 = noise)
        features: (T, D) feature matrix used for clustering
        fps: frames per second
        gt_labels: optional ground truth labels for supervised metrics
        bootstrap_n: number of bootstrap iterations for CI
        compute_significance: whether to run permutation test
    """
    m = ClusteringMetrics()
    m.n_frames = len(labels)

    valid = labels >= 0
    n_valid = valid.sum()
    m.n_noise = int((~valid).sum())
    m.noise_ratio = round(m.n_noise / len(labels), 4) if len(labels) > 0 else 0
    m.n_clusters = len(set(labels[valid])) if n_valid > 0 else 0

    if m.n_clusters < 2 or n_valid < 10:
        return m

    feat_v, lab_v = features[valid], labels[valid]
    sample = min(5000, n_valid)

    # === Unsupervised ===
    m.silhouette = float(silhouette_score(feat_v, lab_v, sample_size=sample))
    m.calinski_harabasz = float(calinski_harabasz_score(feat_v, lab_v))
    m.davies_bouldin = float(davies_bouldin_score(feat_v, lab_v))

    # === Temporal ===
    m.bout_mean_sec, m.bout_median_sec, m.bout_std_sec, m.n_bouts = _bout_stats(lab_v, fps)
    m.transition_rate = _transition_rate(lab_v, fps)
    m.tpi_per_cluster, m.tpi_mean = _compute_tpi(lab_v)
    m.entropy_rate = _entropy_rate(lab_v)
    m.temporal_consistency = _temporal_consistency(lab_v)

    # === Flickering / Distribution ===
    m.short_bout_ratio, m.bout_cv = _flickering_stats(lab_v, fps, threshold_sec=0.3)
    m.short_bout_threshold_sec = 0.3
    m.label_autocorrelation = _label_autocorrelation(lab_v)

    # === Supervised (if GT available) ===
    if gt_labels is not None:
        gt_v = gt_labels[valid]
        m.ari = float(adjusted_rand_score(gt_v, lab_v))
        m.nmi = float(normalized_mutual_info_score(gt_v, lab_v))
        m.mi = float(mutual_info_score(gt_v, lab_v))

    # === Statistical ===
    if bootstrap_n > 0:
        m.silhouette_ci_lower, m.silhouette_ci_upper = _bootstrap_silhouette(
            feat_v, lab_v, n_iterations=bootstrap_n
        )

    if compute_significance:
        m.p_value_vs_shuffle = _temporal_shuffle_test(
            features, labels, fps, n_permutations=200
        )

    return m


def _bout_stats(labels: np.ndarray, fps: float) -> tuple:
    changes = np.where(np.diff(labels) != 0)[0]
    bout_lengths = np.diff(np.concatenate([[0], changes + 1, [len(labels)]]))
    bouts = bout_lengths / fps
    if len(bouts) == 0:
        return 0.0, 0.0, 0.0, 0
    return (
        float(bouts.mean()),
        float(np.median(bouts)),
        float(bouts.std()),
        int(len(bouts)),
    )


def _transition_rate(labels: np.ndarray, fps: float) -> float:
    if len(labels) < 2:
        return 0.0
    changes = np.where(np.diff(labels) != 0)[0]
    duration = len(labels) / fps
    return float(len(changes) / duration) if duration > 0 else 0.0


def _compute_tpi(labels: np.ndarray) -> tuple:
    """Temporal Persistence Index (SUBTLE metric)."""
    unique = sorted(set(labels))
    K = len(unique)
    if K < 2:
        return [], 0.0

    label_map = {l: i for i, l in enumerate(unique)}
    T = np.zeros((K, K))
    for t in range(len(labels) - 1):
        T[label_map[labels[t]], label_map[labels[t + 1]]] += 1

    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = T / row_sums

    self_trans = np.diag(P)
    tpi = 1.0 / (1.0 - self_trans + 1e-6)
    tpi = np.clip(tpi, 0, 10000)  # cap at 10K to avoid overflow
    return [float(t) for t in tpi], float(tpi.mean())


def _entropy_rate(labels: np.ndarray) -> float:
    """Behavioral entropy rate (Shannon-based Markov chain)."""
    unique = sorted(set(labels))
    K = len(unique)
    if K < 2:
        return 0.0

    label_map = {l: i for i, l in enumerate(unique)}
    T = np.zeros((K, K))
    for t in range(len(labels) - 1):
        T[label_map[labels[t]], label_map[labels[t + 1]]] += 1

    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = T / row_sums

    # Stationary distribution
    counts = np.zeros(K)
    for l in labels:
        counts[label_map[l]] += 1
    pi = counts / counts.sum()

    H = 0.0
    for i in range(K):
        for j in range(K):
            if P[i, j] > 0:
                H -= pi[i] * P[i, j] * np.log2(P[i, j])
    return float(H)


def _temporal_consistency(labels: np.ndarray) -> float:
    """Temporal consistency: 1 - (actual transitions / max possible transitions).

    Range [0, 1]: 1 = all frames same label, 0 = every frame different.
    """
    if len(labels) < 2:
        return 1.0
    transitions = np.sum(np.diff(labels) != 0)
    max_transitions = len(labels) - 1
    return float(1.0 - transitions / max_transitions)


def _flickering_stats(
    labels: np.ndarray, fps: float, threshold_sec: float = 0.3
) -> tuple[float, float]:
    """Compute flickering ratio and bout CV.

    Args:
        labels: cluster assignments
        fps: frames per second
        threshold_sec: bouts shorter than this are 'flickering'

    Returns:
        (short_bout_ratio, bout_cv)
    """
    changes = np.where(np.diff(labels) != 0)[0]
    bout_lengths = np.diff(np.concatenate([[0], changes + 1, [len(labels)]]))
    bouts_sec = bout_lengths / fps

    if len(bouts_sec) == 0:
        return 0.0, 0.0

    short_ratio = float(np.mean(bouts_sec < threshold_sec))
    cv = float(bouts_sec.std() / bouts_sec.mean()) if bouts_sec.mean() > 0 else 0.0
    return short_ratio, cv


def _label_autocorrelation(labels: np.ndarray, lag: int = 1) -> float:
    """Label autocorrelation at given lag.

    Measures temporal structure: high = labels persist over time.
    Uses indicator match: corr = P(label[t] == label[t+lag]).
    """
    if len(labels) <= lag:
        return float("nan")
    matches = (labels[:-lag] == labels[lag:]).astype(float)
    return float(matches.mean())


def _bootstrap_silhouette(
    features: np.ndarray,
    labels: np.ndarray,
    n_iterations: int = 100,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval for silhouette score."""
    n = len(features)
    sils = []
    rng = np.random.RandomState(42)

    for _ in range(n_iterations):
        idx = rng.choice(n, n, replace=True)
        feat_b, lab_b = features[idx], labels[idx]
        if len(set(lab_b)) < 2:
            continue
        try:
            s = silhouette_score(feat_b, lab_b, sample_size=min(2000, n))
            sils.append(s)
        except Exception:
            continue

    if not sils:
        return float("nan"), float("nan")

    alpha = (1 - ci) / 2
    lower = float(np.percentile(sils, alpha * 100))
    upper = float(np.percentile(sils, (1 - alpha) * 100))
    return lower, upper


def _temporal_shuffle_test(
    features: np.ndarray,
    labels: np.ndarray,
    fps: float,
    n_permutations: int = 200,
) -> float:
    """Temporal shuffle significance test.

    Shuffles the temporal order of features, re-clusters, and compares
    temporal consistency. If original >> shuffled, the temporal structure
    is significant (not an artifact of feature-space clustering alone).

    Returns p-value: probability that shuffled data has equal or better
    temporal consistency than original.
    """
    from sklearn.cluster import KMeans

    valid = labels >= 0
    lab_v = labels[valid]
    orig_tc = _temporal_consistency(lab_v)

    K = len(set(lab_v))
    feat_v = features[valid]
    rng = np.random.RandomState(42)

    n_better = 0
    for _ in range(n_permutations):
        # Shuffle temporal order
        perm = rng.permutation(len(feat_v))
        feat_shuffled = feat_v[perm]

        # Re-cluster with same K
        km = KMeans(n_clusters=K, random_state=42, n_init=3)
        labels_shuffled = km.fit_predict(feat_shuffled)

        # Unshuffle labels back to original temporal order
        labels_unshuffled = np.empty_like(labels_shuffled)
        labels_unshuffled[perm] = labels_shuffled

        tc_shuffled = _temporal_consistency(labels_unshuffled)
        if tc_shuffled >= orig_tc:
            n_better += 1

    return float(n_better / n_permutations)


def permutation_test_two_methods(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    features_a: np.ndarray,
    features_b: np.ndarray,
    metric: str = "silhouette",
    n_permutations: int = 1000,
) -> dict:
    """Permutation test comparing two clustering methods.

    Tests H0: the two methods produce equally good clusterings.
    Returns observed difference, p-value, and CI.
    """
    def _get_metric(features, labels, metric_name):
        valid = labels >= 0
        if valid.sum() < 10 or len(set(labels[valid])) < 2:
            return 0.0
        f, l = features[valid], labels[valid]
        if metric_name == "silhouette":
            return silhouette_score(f, l, sample_size=min(3000, len(f)))
        elif metric_name == "temporal_consistency":
            return _temporal_consistency(l)
        elif metric_name == "tpi":
            _, tpi_mean = _compute_tpi(l)
            return tpi_mean
        return 0.0

    obs_a = _get_metric(features_a, labels_a, metric)
    obs_b = _get_metric(features_b, labels_b, metric)
    obs_diff = obs_a - obs_b

    # Permutation: randomly swap method labels
    rng = np.random.RandomState(42)
    perm_diffs = []
    for _ in range(n_permutations):
        swap = rng.random(len(labels_a)) > 0.5
        lab_perm_a = np.where(swap, labels_b, labels_a)
        lab_perm_b = np.where(swap, labels_a, labels_b)
        feat_perm_a = np.where(swap[:, np.newaxis], features_b, features_a)
        feat_perm_b = np.where(swap[:, np.newaxis], features_a, features_b)

        d = _get_metric(feat_perm_a, lab_perm_a, metric) - \
            _get_metric(feat_perm_b, lab_perm_b, metric)
        perm_diffs.append(d)

    perm_diffs = np.array(perm_diffs)
    p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))

    return {
        "metric": metric,
        "method_a": round(obs_a, 4),
        "method_b": round(obs_b, 4),
        "observed_diff": round(obs_diff, 4),
        "p_value": round(p_value, 4),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def compare_methods_comprehensive(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    fps: float = 20.0,
) -> dict:
    """Compare multiple methods on all metrics.

    Args:
        results: {method_name: (labels, features)} dict
    """
    all_metrics = {}
    for name, (labels, features) in results.items():
        m = compute_all_metrics(labels, features, fps, compute_significance=True)
        all_metrics[name] = m.to_dict()

    return all_metrics


# === Advanced Statistical Tests ===


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation. Interpretation:
    |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large.
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a, var_b = group_a.var(ddof=1), group_b.var(ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((group_a.mean() - group_b.mean()) / pooled_std)


def mcnemar_test(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    """McNemar test for comparing two clusterings on the same data.

    Tests whether the two methods disagree symmetrically. Useful when
    both methods cluster the same frames and you want to know if the
    disagreement pattern is significant.

    Returns dict with chi2 statistic and p-value.
    """
    from scipy.stats import chi2 as chi2_dist

    agree = labels_a == labels_b
    # Build contingency: count frames where methods agree/disagree
    # For behavior clustering: compare whether each pair of frames
    # is co-clustered (same cluster) or not
    n = min(len(labels_a), 2000)  # subsample for efficiency
    rng = np.random.RandomState(42)
    idx = rng.choice(len(labels_a), n, replace=False) if len(labels_a) > n else np.arange(len(labels_a))

    la, lb = labels_a[idx], labels_b[idx]

    # Pairwise co-clustering agreement
    b_count = 0  # A says same, B says different
    c_count = 0  # A says different, B says same
    sample_pairs = min(5000, n * (n - 1) // 2)
    pair_idx = rng.choice(n, (sample_pairs, 2), replace=True)

    for i, j in pair_idx:
        if i == j:
            continue
        a_same = la[i] == la[j]
        b_same = lb[i] == lb[j]
        if a_same and not b_same:
            b_count += 1
        elif not a_same and b_same:
            c_count += 1

    # McNemar statistic with continuity correction
    if b_count + c_count == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b_count": b_count, "c_count": c_count}

    chi2 = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
    p_value = 1.0 - chi2_dist.cdf(chi2, df=1)

    return {
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p_value), 4),
        "b_count": b_count,
        "c_count": c_count,
        "significant_005": p_value < 0.05,
    }


def bout_duration_distribution_test(
    labels: np.ndarray, fps: float = 20.0
) -> dict:
    """Test whether bout durations follow an exponential distribution.

    Exponential durations suggest memoryless (Poisson) transitions.
    Non-exponential suggests structured/hierarchical behavior.

    Returns KS test statistic and p-value.
    """
    from scipy.stats import kstest, expon

    changes = np.where(np.diff(labels) != 0)[0]
    bout_lengths = np.diff(np.concatenate([[0], changes + 1, [len(labels)]]))
    bouts_sec = bout_lengths / fps

    if len(bouts_sec) < 10:
        return {"ks_statistic": float("nan"), "p_value": float("nan"), "is_exponential": False}

    # Fit exponential and test
    stat, p_value = kstest(bouts_sec, "expon", args=(0, bouts_sec.mean()))

    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 4),
        "is_exponential": p_value > 0.05,
        "mean_bout_sec": round(float(bouts_sec.mean()), 4),
        "n_bouts": len(bouts_sec),
    }


def label_autocorrelation_multi_lag(
    labels: np.ndarray, max_lag_frames: int = 40, fps: float = 20.0
) -> dict:
    """Compute label autocorrelation at multiple lags.

    Returns autocorrelation curve and decay time constant.
    """
    lags = list(range(1, min(max_lag_frames + 1, len(labels))))
    autocorrs = []

    for lag in lags:
        matches = (labels[:-lag] == labels[lag:]).astype(float)
        autocorrs.append(float(matches.mean()))

    autocorrs = np.array(autocorrs)
    lag_sec = [l / fps for l in lags]

    # Estimate decay: find lag where autocorrelation drops below 1/e of initial
    if len(autocorrs) > 0 and autocorrs[0] > 0:
        threshold = autocorrs[0] / np.e
        decay_idx = np.where(autocorrs < threshold)[0]
        decay_time = lag_sec[decay_idx[0]] if len(decay_idx) > 0 else lag_sec[-1]
    else:
        decay_time = 0.0

    return {
        "lags_sec": [round(l, 3) for l in lag_sec],
        "autocorrelations": [round(a, 4) for a in autocorrs],
        "decay_time_sec": round(decay_time, 4),
        "lag1_autocorr": round(autocorrs[0], 4) if len(autocorrs) > 0 else float("nan"),
    }


def transition_matrix(labels: np.ndarray) -> dict:
    """Compute transition probability matrix and related statistics.

    Returns transition matrix, stationary distribution, and
    per-state self-transition probabilities.
    """
    unique = sorted(set(labels))
    K = len(unique)
    if K < 2:
        return {"n_clusters": K}

    label_map = {l: i for i, l in enumerate(unique)}
    T = np.zeros((K, K))
    for t in range(len(labels) - 1):
        T[label_map[labels[t]], label_map[labels[t + 1]]] += 1

    # Normalize rows
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = T / row_sums

    # Stationary distribution
    counts = np.zeros(K)
    for l in labels:
        counts[label_map[l]] += 1
    pi = counts / counts.sum()

    return {
        "n_clusters": K,
        "cluster_labels": unique,
        "transition_matrix": [[round(float(p), 4) for p in row] for row in P],
        "stationary_distribution": [round(float(x), 4) for x in pi],
        "self_transition_probs": [round(float(P[i, i]), 4) for i in range(K)],
        "count_matrix": [[int(c) for c in row] for row in T],
    }
