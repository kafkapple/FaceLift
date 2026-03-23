"""HLAC Classification PoC: Keypoint-defined HLACs → Feature Probe.

Generates Higher-Level Action Classes (HLACs) from raw keypoint clustering,
then evaluates S1, S3, and BAMS feature representations via linear probe.

Avoids circularity: HLACs are defined from raw keypoints (physical pose),
not from any learned representation.

Usage:
    cd /home/joon/dev/FaceLift
    python -m mouse_extensions.behavior.hlac_classification \
        --sparse-path outputs/analysis/mouse/bams/features/sparse_features.npz \
        --bams-path outputs/analysis/mouse/bams/bams_embeddings.npz \
        --output-dir outputs/analysis/mouse/behavior_clustering/hlac_analysis \
        --target-k 7
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# Keypoint names from SDAnnce (23 joints)
KEYPOINT_NAMES = [
    "L_ear", "R_ear", "nose", "neck", "body_middle", "tail_root",
    "tail_middle", "tail_end", "L_paw", "L_paw_end", "L_elbow", "L_shoulder",
    "R_paw", "R_paw_end", "R_elbow", "R_shoulder", "L_foot", "L_knee",
    "L_hip", "R_foot", "R_knee", "R_hip",
]

# Skeletal connections for visualization
SKELETON_CONNECTIONS = [
    # Head
    (2, 3),   # nose → neck
    (0, 3),   # L_ear → neck
    (1, 3),   # R_ear → neck
    # Spine
    (3, 4),   # neck → body_middle
    (4, 5),   # body_middle → tail_root
    (5, 6),   # tail_root → tail_middle
    (6, 7),   # tail_middle → tail_end
    # Left forelimb
    (11, 10), # L_shoulder → L_elbow
    (10, 8),  # L_elbow → L_paw
    (8, 9),   # L_paw → L_paw_end
    # Right forelimb
    (15, 14), # R_shoulder → R_elbow
    (14, 12), # R_elbow → R_paw
    (12, 13), # R_paw → R_paw_end
    # Left hindlimb
    (18, 17), # L_hip → L_knee
    (17, 16), # L_knee → L_foot
    # Right hindlimb
    (21, 20), # R_hip → R_knee
    (20, 19), # R_knee → R_foot
    # Torso connections
    (3, 11),  # neck → L_shoulder
    (3, 15),  # neck → R_shoulder
    (4, 18),  # body_middle → L_hip
    (4, 21),  # body_middle → R_hip
]


# =============================================================================
# Step 1: Data Loading & BAMS Alignment
# =============================================================================

def load_data(sparse_path: str, bams_path: str) -> Dict[str, np.ndarray]:
    """Load sparse features and BAMS embeddings."""
    print("Loading data...")
    sparse = np.load(sparse_path, allow_pickle=True)
    bams = np.load(bams_path, allow_pickle=True)

    data = {
        "kp1_raw": sparse["kp1_raw"],           # (90000, 23, 3)
        "kp2_raw": sparse["kp2_raw"],           # (90000, 23, 3)
        "s1_rat1": sparse["s1_single_rat1"],    # (90000, 69)
        "s1_dyadic": sparse["s1_dyadic"],       # (90000, 138)
        "s3_engineered": sparse["s3_engineered"],  # (90000, 12)
        "bams_short": bams["short_term"],       # (179000, 64)
        "bams_long": bams["long_term"],         # (179000, 64)
        "bams_combined": bams["combined"],      # (179000, 128)
        "fps": float(sparse["fps"]),
        "bams_seq_len": int(bams["seq_len"]),
    }

    for k in ["kp1_raw", "s1_rat1", "s3_engineered", "bams_combined"]:
        print(f"  {k}: {data[k].shape}")

    return data


def align_bams_embeddings(
    bams_combined: np.ndarray,
    num_frames: int,
    seq_len: int,
    stride: int = 500,
) -> np.ndarray:
    """Align BAMS overlapping window embeddings to per-frame via averaging.

    BAMS produces embeddings for overlapping sequences. Each frame appears
    in multiple windows. We average all embeddings for each frame.

    Args:
        bams_combined: (N_windows * seq_len, embed_dim) flattened embeddings
        num_frames: Total original frames (90000)
        seq_len: Window length (1000)
        stride: Window stride (500)

    Returns:
        (num_frames, embed_dim) averaged per-frame embeddings
    """
    print("Aligning BAMS embeddings (179K → 90K)...")
    embed_dim = bams_combined.shape[1]
    num_windows = (num_frames - seq_len) // stride + 1

    # Accumulator arrays
    frame_sum = np.zeros((num_frames, embed_dim), dtype=np.float64)
    frame_count = np.zeros(num_frames, dtype=np.int32)

    for w in range(num_windows):
        start = w * stride
        end = start + seq_len
        emb_start = w * seq_len
        emb_end = emb_start + seq_len

        if end > num_frames or emb_end > bams_combined.shape[0]:
            break

        frame_sum[start:end] += bams_combined[emb_start:emb_end]
        frame_count[start:end] += 1

    # Avoid division by zero for edge frames
    safe_count = np.maximum(frame_count, 1)
    aligned = (frame_sum / safe_count[:, np.newaxis]).astype(np.float32)

    covered = np.sum(frame_count > 0)
    print(f"  Aligned: {aligned.shape}, frames covered: {covered}/{num_frames}")
    print(f"  Avg overlaps per frame: {frame_count[frame_count > 0].mean():.1f}")

    return aligned


# =============================================================================
# Step 2: HLAC Generation (Keypoint Clustering)
# =============================================================================

def generate_hlacs(
    kp_raw: np.ndarray,
    k_range: range = range(4, 10),
    random_state: int = 42,
) -> Dict[int, Dict]:
    """Generate HLACs by clustering raw keypoints.

    Uses StandardScaler + KMeans on flattened keypoints.

    Returns:
        Dict mapping K → {labels, kmeans, metrics}
    """
    print("\nGenerating HLACs from raw keypoints...")
    n_frames = kp_raw.shape[0]
    features = kp_raw.reshape(n_frames, -1)  # (N, 69)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    results = {}
    for k in k_range:
        print(f"  K={k}...", end=" ")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(features_scaled)

        sil = silhouette_score(features_scaled, labels, sample_size=10000,
                               random_state=random_state)
        ch = calinski_harabasz_score(features_scaled, labels)
        db = davies_bouldin_score(features_scaled, labels)

        # Temporal metrics
        transitions = np.sum(labels[1:] != labels[:-1])
        tpi = 1.0 - transitions / (n_frames - 1)

        # Bout statistics
        bout_lengths = []
        current_len = 1
        for i in range(1, len(labels)):
            if labels[i] == labels[i - 1]:
                current_len += 1
            else:
                bout_lengths.append(current_len)
                current_len = 1
        bout_lengths.append(current_len)
        bout_arr = np.array(bout_lengths)

        results[k] = {
            "labels": labels,
            "kmeans": kmeans,
            "scaler": scaler,
            "metrics": {
                "silhouette": float(sil),
                "calinski_harabasz": float(ch),
                "davies_bouldin": float(db),
                "tpi": float(tpi),
                "n_transitions": int(transitions),
                "bout_median_frames": float(np.median(bout_arr)),
                "bout_mean_frames": float(np.mean(bout_arr)),
                "n_bouts": len(bout_lengths),
            },
        }

        print(f"Sil={sil:.3f}, TPI={tpi:.3f}, DB={db:.2f}, "
              f"bouts={len(bout_lengths)}, median_bout={np.median(bout_arr):.0f}f")

    return results


def select_best_k(hlac_results: Dict[int, Dict]) -> int:
    """Select best K based on composite score (silhouette + TPI)."""
    scores = {}
    for k, res in hlac_results.items():
        m = res["metrics"]
        # Normalize: higher silhouette better, higher TPI better, lower DB better
        scores[k] = m["silhouette"] * 0.4 + m["tpi"] * 0.3 - m["davies_bouldin"] * 0.1
    best_k = max(scores, key=scores.get)
    print(f"\nBest K by composite score: {best_k} (score={scores[best_k]:.3f})")
    return best_k


# =============================================================================
# Step 3: Classification PoC (Linear Probe)
# =============================================================================

def run_classification_probe(
    feature_sets: Dict[str, np.ndarray],
    labels: np.ndarray,
    k: int,
    random_state: int = 42,
    n_cv_folds: int = 5,
) -> Dict[str, Dict]:
    """Run linear probe classification for each feature set.

    Uses stratified train/test split + cross-validation.

    Returns:
        Dict mapping feature_name → {accuracy, f1_macro, cv_scores, report, cm, ...}
    """
    print(f"\n{'='*60}")
    print(f"Classification Probe (K={k} HLACs)")
    print(f"{'='*60}")

    results = {}
    for name, X in feature_sets.items():
        print(f"\n--- {name} ({X.shape[1]}d) ---")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, labels, test_size=0.2, random_state=random_state, stratify=labels
        )

        # Train logistic regression
        clf = LogisticRegression(
            max_iter=1000, solver="saga", multi_class="ovr",
            random_state=random_state, n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred, normalize="true")
        report = classification_report(y_test, y_pred, output_dict=True)

        # Cross-validation for robustness
        cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(clf, X_scaled, labels, cv=cv, scoring="accuracy")

        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  Test F1-macro: {f1:.4f}")
        print(f"  CV Accuracy:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        results[name] = {
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "cv_scores": cv_scores.tolist(),
            "confusion_matrix": cm,
            "classification_report": report,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    return results


# =============================================================================
# Step 4: Qualitative Analysis
# =============================================================================

def analyze_clusters_qualitatively(
    kp_raw: np.ndarray,
    labels: np.ndarray,
    k: int,
    fps: float,
) -> Dict:
    """Compute mean poses, cluster sizes, and transition matrix."""
    print(f"\nQualitative analysis (K={k})...")

    # Mean pose per cluster
    mean_poses = np.zeros((k, kp_raw.shape[1], kp_raw.shape[2]))
    std_poses = np.zeros_like(mean_poses)
    cluster_sizes = np.zeros(k, dtype=int)
    for i in range(k):
        mask = labels == i
        cluster_sizes[i] = mask.sum()
        if cluster_sizes[i] > 0:
            mean_poses[i] = kp_raw[mask].mean(axis=0)
            std_poses[i] = kp_raw[mask].std(axis=0)

    # Transition matrix (normalized by row)
    trans_matrix = np.zeros((k, k), dtype=int)
    for t in range(len(labels) - 1):
        trans_matrix[labels[t], labels[t + 1]] += 1
    trans_prob = trans_matrix / np.maximum(trans_matrix.sum(axis=1, keepdims=True), 1)

    # Bout duration per cluster
    bout_durations = {i: [] for i in range(k)}
    current_label = labels[0]
    current_len = 1
    for t in range(1, len(labels)):
        if labels[t] == current_label:
            current_len += 1
        else:
            bout_durations[current_label].append(current_len / fps)
            current_label = labels[t]
            current_len = 1
    bout_durations[current_label].append(current_len / fps)

    bout_stats = {}
    for i in range(k):
        bouts = np.array(bout_durations[i])
        bout_stats[i] = {
            "count": len(bouts),
            "median_s": float(np.median(bouts)) if len(bouts) > 0 else 0,
            "mean_s": float(np.mean(bouts)) if len(bouts) > 0 else 0,
            "max_s": float(np.max(bouts)) if len(bouts) > 0 else 0,
        }

    print("  Cluster distribution:")
    for i in range(k):
        pct = 100 * cluster_sizes[i] / len(labels)
        bs = bout_stats[i]
        print(f"    HLAC {i}: {cluster_sizes[i]:>6} frames ({pct:5.1f}%), "
              f"{bs['count']:>4} bouts, median={bs['median_s']:.2f}s")

    return {
        "mean_poses": mean_poses,
        "std_poses": std_poses,
        "cluster_sizes": cluster_sizes,
        "transition_matrix": trans_matrix,
        "transition_prob": trans_prob,
        "bout_stats": bout_stats,
    }


# =============================================================================
# Step 5: Visualizations
# =============================================================================

def set_neurips_style():
    """Set matplotlib style for NeurIPS-quality figures."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_k_sweep(hlac_results: Dict, output_dir: Path):
    """Plot clustering metrics across K values."""
    set_neurips_style()
    ks = sorted(hlac_results.keys())
    metrics = {k: hlac_results[k]["metrics"] for k in ks}

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

    axes[0].plot(ks, [metrics[k]["silhouette"] for k in ks], "o-", color="#2196F3")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_xlabel("K")
    axes[0].set_title("Cluster Separation")

    axes[1].plot(ks, [metrics[k]["tpi"] for k in ks], "s-", color="#4CAF50")
    axes[1].set_ylabel("TPI")
    axes[1].set_xlabel("K")
    axes[1].set_title("Temporal Persistence")

    axes[2].plot(ks, [metrics[k]["davies_bouldin"] for k in ks], "^-", color="#FF5722")
    axes[2].set_ylabel("Davies-Bouldin")
    axes[2].set_xlabel("K")
    axes[2].set_title("Cluster Overlap (lower=better)")

    axes[3].plot(ks, [metrics[k]["bout_median_frames"] for k in ks], "D-", color="#9C27B0")
    axes[3].set_ylabel("Median Bout (frames)")
    axes[3].set_xlabel("K")
    axes[3].set_title("Bout Duration")

    plt.tight_layout()
    plt.savefig(output_dir / "k_sweep_metrics.png")
    plt.close()
    print(f"  Saved: k_sweep_metrics.png")


def plot_classification_comparison(clf_results: Dict, k: int, output_dir: Path):
    """Bar chart comparing accuracy and F1-macro across feature sets."""
    set_neurips_style()
    names = list(clf_results.keys())
    acc = [clf_results[n]["accuracy"] for n in names]
    f1 = [clf_results[n]["f1_macro"] for n in names]
    cv_mean = [clf_results[n]["cv_accuracy_mean"] for n in names]
    cv_std = [clf_results[n]["cv_accuracy_std"] for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width, acc, width, label="Test Accuracy", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x, f1, width, label="F1-macro", color="#4CAF50", alpha=0.85)
    bars3 = ax.bar(x + width, cv_mean, width, yerr=cv_std, capsize=4,
                   label="CV Accuracy", color="#FF9800", alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title(f"HLAC Classification Performance (K={k})")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / f"classification_comparison_K{k}.png")
    plt.close()
    print(f"  Saved: classification_comparison_K{k}.png")


def plot_confusion_matrices(clf_results: Dict, k: int, output_dir: Path):
    """Normalized confusion matrices for each feature set."""
    set_neurips_style()
    n = len(clf_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, clf_results.items()):
        cm = res["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                    xticklabels=range(k), yticklabels=range(k),
                    vmin=0, vmax=1, cbar_kws={"shrink": 0.8})
        ax.set_title(f"{name} ({res['accuracy']:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.suptitle(f"Confusion Matrices — HLAC Classification (K={k})", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrices_K{k}.png")
    plt.close()
    print(f"  Saved: confusion_matrices_K{k}.png")


def plot_ethogram(labels: np.ndarray, fps: float, k: int, output_dir: Path,
                  duration_s: float = 120.0):
    """Ethogram showing temporal behavior sequence."""
    set_neurips_style()
    n_frames = min(len(labels), int(duration_s * fps))
    time_s = np.arange(n_frames) / fps

    fig, ax = plt.subplots(figsize=(14, 2))
    cmap = plt.colormaps.get_cmap("tab10").resampled(k)
    ax.imshow(labels[:n_frames][np.newaxis, :], aspect="auto", cmap=cmap,
              vmin=-0.5, vmax=k - 0.5,
              extent=[0, n_frames / fps, 0, 1], interpolation="nearest")
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Behavioral Ethogram (K={k}, first {duration_s:.0f}s)")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-0.5, k - 0.5))
    cbar = plt.colorbar(sm, ax=ax, ticks=range(k), shrink=0.8, pad=0.02)
    cbar.set_label("HLAC")

    plt.tight_layout()
    plt.savefig(output_dir / f"ethogram_K{k}.png")
    plt.close()
    print(f"  Saved: ethogram_K{k}.png")


def plot_transition_matrix(trans_prob: np.ndarray, k: int, output_dir: Path):
    """Transition probability matrix between HLACs."""
    set_neurips_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(trans_prob, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                xticklabels=range(k), yticklabels=range(k),
                vmin=0, vmax=1)
    ax.set_xlabel("Next HLAC")
    ax.set_ylabel("Current HLAC")
    ax.set_title(f"HLAC Transition Probabilities (K={k})")
    plt.tight_layout()
    plt.savefig(output_dir / f"transition_matrix_K{k}.png")
    plt.close()
    print(f"  Saved: transition_matrix_K{k}.png")


def plot_mean_poses(mean_poses: np.ndarray, k: int, output_dir: Path,
                    cluster_sizes: Optional[np.ndarray] = None):
    """3D mean pose visualization for each HLAC cluster."""
    set_neurips_style()
    cols = min(k, 4)
    rows = (k + cols - 1) // cols

    fig = plt.figure(figsize=(4 * cols, 4 * rows))

    # Global axis limits
    all_coords = mean_poses.reshape(-1, 3)
    pad = 0.1 * (all_coords.max() - all_coords.min())
    xlim = (all_coords[:, 0].min() - pad, all_coords[:, 0].max() + pad)
    ylim = (all_coords[:, 1].min() - pad, all_coords[:, 1].max() + pad)
    zlim = (all_coords[:, 2].min() - pad, all_coords[:, 2].max() + pad)

    for i in range(k):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        pose = mean_poses[i]

        # Plot skeleton connections
        for j1, j2 in SKELETON_CONNECTIONS:
            if j1 < len(pose) and j2 < len(pose):
                ax.plot([pose[j1, 0], pose[j2, 0]],
                        [pose[j1, 1], pose[j2, 1]],
                        [pose[j1, 2], pose[j2, 2]],
                        "gray", linewidth=1.5, alpha=0.7)

        # Plot joints
        ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2],
                   s=25, c="red", zorder=5, edgecolors="darkred", linewidth=0.5)

        title = f"HLAC {i}"
        if cluster_sizes is not None:
            pct = 100 * cluster_sizes[i] / cluster_sizes.sum()
            title += f" ({pct:.1f}%)"
        ax.set_title(title)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.view_init(elev=20, azim=45)

    plt.suptitle(f"Mean Keypoint Poses per HLAC (K={k})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"mean_poses_K{k}.png")
    plt.close()
    print(f"  Saved: mean_poses_K{k}.png")


def plot_tsne(
    feature_sets: Dict[str, np.ndarray],
    labels: np.ndarray,
    k: int,
    output_dir: Path,
    n_samples: int = 8000,
    random_state: int = 42,
):
    """t-SNE visualization colored by HLAC for each feature set."""
    from sklearn.manifold import TSNE

    set_neurips_style()
    n = len(feature_sets)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    # Subsample for speed
    np.random.seed(random_state)
    idx = np.random.choice(len(labels), min(n_samples, len(labels)), replace=False)
    labels_sub = labels[idx]

    cmap = plt.colormaps.get_cmap("tab10").resampled(k)

    for ax, (name, X) in zip(axes, feature_sets.items()):
        print(f"  t-SNE for {name}...", end=" ", flush=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[idx])

        tsne = TSNE(n_components=2, perplexity=50, random_state=random_state,
                     n_jobs=-1, learning_rate="auto", init="pca")
        embedding = tsne.fit_transform(X_scaled)

        scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                             c=labels_sub, cmap=cmap, s=3, alpha=0.5,
                             vmin=-0.5, vmax=k - 0.5)
        ax.set_title(f"{name} ({X.shape[1]}d)")
        ax.set_xticks([])
        ax.set_yticks([])
        print("done")

    plt.suptitle(f"t-SNE Projections Colored by HLAC (K={k})", y=1.02)
    plt.colorbar(scatter, ax=axes, ticks=range(k), label="HLAC", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / f"tsne_comparison_K{k}.png")
    plt.close()
    print(f"  Saved: tsne_comparison_K{k}.png")


def plot_bout_duration_boxplot(
    bout_stats: Dict,
    labels: np.ndarray,
    k: int,
    fps: float,
    output_dir: Path,
):
    """Boxplot of bout durations per HLAC cluster."""
    set_neurips_style()

    # Recompute bout durations per cluster
    bout_durations = {i: [] for i in range(k)}
    current_label = labels[0]
    current_len = 1
    for t in range(1, len(labels)):
        if labels[t] == current_label:
            current_len += 1
        else:
            bout_durations[current_label].append(current_len / fps)
            current_label = labels[t]
            current_len = 1
    bout_durations[current_label].append(current_len / fps)

    fig, ax = plt.subplots(figsize=(8, 4))
    data = [bout_durations[i] for i in range(k)]
    bp = ax.boxplot(data, labels=[f"HLAC {i}" for i in range(k)],
                    patch_artist=True, showfliers=False)

    cmap = plt.colormaps.get_cmap("tab10").resampled(k)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i))
        patch.set_alpha(0.7)

    ax.set_ylabel("Bout Duration (s)")
    ax.set_title(f"Bout Duration Distribution per HLAC (K={k})")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"bout_duration_K{k}.png")
    plt.close()
    print(f"  Saved: bout_duration_K{k}.png")


# =============================================================================
# Step 6: Hero Figure (Multi-panel)
# =============================================================================

def plot_hero_figure(
    clf_results: Dict,
    qual_analysis: Dict,
    labels: np.ndarray,
    k: int,
    fps: float,
    output_dir: Path,
):
    """Multi-panel hero figure for NeurIPS paper."""
    set_neurips_style()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    # Panel A: Classification comparison (top-left, spans 2 cols)
    ax_bar = fig.add_subplot(gs[0, :2])
    names = list(clf_results.keys())
    acc = [clf_results[n]["accuracy"] for n in names]
    f1 = [clf_results[n]["f1_macro"] for n in names]
    x = np.arange(len(names))
    width = 0.3
    ax_bar.bar(x - width/2, acc, width, label="Accuracy", color="#2196F3", alpha=0.85)
    ax_bar.bar(x + width/2, f1, width, label="F1-macro", color="#4CAF50", alpha=0.85)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("(a) Feature Probe Performance")
    ax_bar.legend(loc="lower right")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.grid(axis="y", alpha=0.3)

    # Panel B: Transition matrix (top-right, spans 2 cols)
    ax_trans = fig.add_subplot(gs[0, 2:])
    sns.heatmap(qual_analysis["transition_prob"], annot=True, fmt=".2f",
                cmap="YlOrRd", ax=ax_trans, vmin=0, vmax=0.8,
                xticklabels=range(k), yticklabels=range(k))
    ax_trans.set_xlabel("Next")
    ax_trans.set_ylabel("Current")
    ax_trans.set_title("(b) HLAC Transitions")

    # Panel C: Ethogram (middle, full width)
    ax_eth = fig.add_subplot(gs[1, :])
    n_frames = min(len(labels), int(60 * fps))
    time_s = np.arange(n_frames) / fps
    cmap = plt.colormaps.get_cmap("tab10").resampled(k)
    ax_eth.imshow(labels[:n_frames][np.newaxis, :], aspect="auto", cmap=cmap,
                  vmin=-0.5, vmax=k - 0.5,
                  extent=[0, n_frames / fps, 0, 1], interpolation="nearest")
    ax_eth.set_yticks([])
    ax_eth.set_xlabel("Time (s)")
    ax_eth.set_title("(c) Behavioral Ethogram (60s)")

    # Panel D: Mean poses (bottom, 4 subplots)
    n_show = min(k, 4)
    for i in range(n_show):
        ax_pose = fig.add_subplot(gs[2, i], projection="3d")
        pose = qual_analysis["mean_poses"][i]

        for j1, j2 in SKELETON_CONNECTIONS:
            if j1 < len(pose) and j2 < len(pose):
                ax_pose.plot([pose[j1, 0], pose[j2, 0]],
                             [pose[j1, 1], pose[j2, 1]],
                             [pose[j1, 2], pose[j2, 2]],
                             "gray", linewidth=1.2, alpha=0.7)

        ax_pose.scatter(pose[:, 0], pose[:, 1], pose[:, 2],
                        s=15, c="red", zorder=5)
        pct = 100 * qual_analysis["cluster_sizes"][i] / qual_analysis["cluster_sizes"].sum()
        ax_pose.set_title(f"(d{i+1}) HLAC {i} ({pct:.0f}%)", fontsize=10)
        ax_pose.set_xticklabels([])
        ax_pose.set_yticklabels([])
        ax_pose.set_zticklabels([])
        ax_pose.view_init(elev=20, azim=45)

    plt.suptitle(f"BehaviorSplatter: HLAC Analysis (K={k})", fontsize=15, y=1.01)
    plt.savefig(output_dir / f"hero_figure_K{k}.png", dpi=300)
    plt.close()
    print(f"  Saved: hero_figure_K{k}.png")


# =============================================================================
# Main Pipeline
# =============================================================================

def save_results(
    hlac_results: Dict,
    clf_results: Dict,
    qual_analysis: Dict,
    target_k: int,
    output_dir: Path,
):
    """Save all results as JSON."""
    # HLAC metrics (all K)
    hlac_metrics = {str(k): res["metrics"] for k, res in hlac_results.items()}
    with open(output_dir / "hlac_k_sweep_metrics.json", "w") as f:
        json.dump(hlac_metrics, f, indent=2)

    # Classification results
    clf_summary = {}
    for name, res in clf_results.items():
        clf_summary[name] = {
            "accuracy": res["accuracy"],
            "f1_macro": res["f1_macro"],
            "cv_accuracy_mean": res["cv_accuracy_mean"],
            "cv_accuracy_std": res["cv_accuracy_std"],
            "cv_scores": res["cv_scores"],
        }
    with open(output_dir / f"classification_results_K{target_k}.json", "w") as f:
        json.dump(clf_summary, f, indent=2)

    # Qualitative analysis
    qual_summary = {
        "cluster_sizes": qual_analysis["cluster_sizes"].tolist(),
        "bout_stats": qual_analysis["bout_stats"],
        "transition_matrix": qual_analysis["transition_matrix"].tolist(),
    }
    with open(output_dir / f"qualitative_analysis_K{target_k}.json", "w") as f:
        json.dump(qual_summary, f, indent=2)

    # Save HLAC labels
    np.save(output_dir / f"hlac_labels_K{target_k}.npy",
            hlac_results[target_k]["labels"])

    print(f"\nResults saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="HLAC Classification PoC")
    parser.add_argument("--sparse-path", type=str, required=True,
                        help="Path to sparse_features.npz")
    parser.add_argument("--bams-path", type=str, required=True,
                        help="Path to bams_embeddings.npz")
    parser.add_argument("--output-dir", type=str, default="outputs/analysis/mouse/behavior_clustering/hlac_analysis",
                        help="Output directory")
    parser.add_argument("--target-k", type=int, default=0,
                        help="Target K for detailed analysis (0=auto-select)")
    parser.add_argument("--k-min", type=int, default=4, help="Min K for sweep")
    parser.add_argument("--k-max", type=int, default=10, help="Max K for sweep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-tsne", action="store_true", help="Skip t-SNE (slow)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    data = load_data(args.sparse_path, args.bams_path)

    # 2. Align BAMS embeddings
    bams_aligned = align_bams_embeddings(
        data["bams_combined"],
        num_frames=data["kp1_raw"].shape[0],
        seq_len=data["bams_seq_len"],
    )

    # 3. Generate HLACs from raw keypoints
    k_range = range(args.k_min, args.k_max + 1)
    hlac_results = generate_hlacs(data["kp1_raw"], k_range=k_range,
                                  random_state=args.seed)

    # 4. Select best K
    target_k = args.target_k if args.target_k > 0 else select_best_k(hlac_results)
    labels = hlac_results[target_k]["labels"]
    print(f"\nUsing K={target_k} for detailed analysis")

    # 5. Classification probe
    feature_sets = {
        "S1_skeleton": data["s1_rat1"],       # 69d
        "S3_engineered": data["s3_engineered"],  # 12d
        "BAMS_aligned": bams_aligned,          # 128d
        "BAMS_short": align_bams_embeddings(   # 64d (short-term only)
            data["bams_short"], data["kp1_raw"].shape[0], data["bams_seq_len"],
        ),
        "BAMS_long": align_bams_embeddings(    # 64d (long-term only)
            data["bams_long"], data["kp1_raw"].shape[0], data["bams_seq_len"],
        ),
    }
    clf_results = run_classification_probe(feature_sets, labels, target_k,
                                           random_state=args.seed)

    # 6. Qualitative analysis
    qual_analysis = analyze_clusters_qualitatively(
        data["kp1_raw"], labels, target_k, data["fps"]
    )

    # 7. Visualizations
    print("\nGenerating visualizations...")
    plot_k_sweep(hlac_results, output_dir)
    plot_classification_comparison(clf_results, target_k, output_dir)
    plot_confusion_matrices(clf_results, target_k, output_dir)
    plot_ethogram(labels, data["fps"], target_k, output_dir)
    plot_transition_matrix(qual_analysis["transition_prob"], target_k, output_dir)
    plot_mean_poses(qual_analysis["mean_poses"], target_k, output_dir,
                    qual_analysis["cluster_sizes"])
    plot_bout_duration_boxplot(qual_analysis["bout_stats"], labels,
                               target_k, data["fps"], output_dir)
    plot_hero_figure(clf_results, qual_analysis, labels, target_k,
                     data["fps"], output_dir)

    if not args.skip_tsne:
        plot_tsne(feature_sets, labels, target_k, output_dir)

    # 8. Save results
    save_results(hlac_results, clf_results, qual_analysis, target_k, output_dir)

    print("\n" + "=" * 60)
    print("HLAC Classification PoC Complete!")
    print(f"Output: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
