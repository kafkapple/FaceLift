"""HLAC Analysis on M5t2 GS-LRM Data — Core BehaviorSplatter Experiment.

Applies the same HLAC framework to M5t2 (GS-LRM reconstructed) data,
comparing Gaussian-derived features vs Keypoint-derived features.

This is the CORE experiment for the BehaviorSplatter paper:
"Do 3D Gaussian properties capture behavioral information beyond keypoints?"

Usage:
    cd /home/joon/dev/FaceLift
    OPENBLAS_NUM_THREADS=16 python -u -m mouse_extensions.behavior.hlac_m5t2 \
        --output-dir outputs/m5t2_hlac_analysis
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Optional

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

# M5t2 data paths (gpu03)
KP_PATH = "/node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz"
TEMPORAL_PATH = "outputs/report/clustering/features/temporal/temporal_features.npz"
GAUSSIAN_RAW_PATH = "outputs/report/clustering/features/gaussian_raw_features.npz"

# MAMMAL 22 keypoint names
KP_NAMES = None  # Loaded from data

# Body-centered normalization reference joint
BODY_CENTER_JOINT = 4  # SpineM (middle of spine)


def load_m5t2_data() -> Dict[str, np.ndarray]:
    """Load all M5t2 feature sets and align to common frame indices."""
    print("Loading M5t2 data...")

    # Keypoints
    kp_data = np.load(KP_PATH, allow_pickle=True)
    keypoints = kp_data["keypoints"]  # (3600, 22, 3)
    kp_frame_idx = kp_data["frame_indices"]  # (3600,)
    kp_names = kp_data["keypoint_names"].tolist() if "keypoint_names" in kp_data.files else None
    print(f"  Keypoints: {keypoints.shape}, names: {kp_names}")

    # Temporal features (Strategy C)
    tf_data = np.load(TEMPORAL_PATH, allow_pickle=True)
    temporal_all = tf_data["all_features"]  # (3585, 129)
    temporal_centroid = tf_data["centroid_features"]  # (3585, 7)
    temporal_bodypart = tf_data["bodypart_delta_features"]  # (3585, 88)
    temporal_rigid = tf_data["rigid_nonrigid_features"]  # (3585, 25)
    temporal_window = tf_data["sliding_window_features"]  # (3585, 9)
    fps = float(tf_data["fps"]) if "fps" in tf_data.files else 20.0
    print(f"  Temporal (Strategy C): {temporal_all.shape}, fps={fps}")

    # Gaussian raw features
    gf_data = np.load(GAUSSIAN_RAW_PATH, allow_pickle=True)
    gaussian_raw = gf_data["features"]  # (3007, 31)
    gf_frame_idx = gf_data["frame_indices"]  # (3007,)
    print(f"  Gaussian raw: {gaussian_raw.shape}, frames {gf_frame_idx.min()}-{gf_frame_idx.max()}")

    # Align to common frames (intersection of all three)
    # Temporal: frames 0..3584
    # Gaussian: frames 578..3599 (variable)
    # Keypoints: frames from kp_frame_idx
    temporal_frames = set(range(temporal_all.shape[0]))
    gaussian_frames = set(gf_frame_idx.tolist())
    kp_frames = set(kp_frame_idx.tolist())

    common_frames = sorted(temporal_frames & gaussian_frames & kp_frames)
    print(f"  Common frames: {len(common_frames)} (range {min(common_frames)}-{max(common_frames)})")

    # Build index maps
    gf_idx_map = {f: i for i, f in enumerate(gf_frame_idx)}
    kp_idx_map = {f: i for i, f in enumerate(kp_frame_idx)}

    # Extract aligned data
    common_arr = np.array(common_frames)
    temporal_aligned = temporal_all[common_arr]
    temporal_centroid_aligned = temporal_centroid[common_arr]
    temporal_bodypart_aligned = temporal_bodypart[common_arr]
    temporal_rigid_aligned = temporal_rigid[common_arr]

    gaussian_indices = [gf_idx_map[f] for f in common_frames]
    gaussian_aligned = gaussian_raw[gaussian_indices]

    kp_indices = [kp_idx_map[f] for f in common_frames]
    keypoints_aligned = keypoints[kp_indices]

    # Body-centered normalization for keypoints
    center = keypoints_aligned[:, BODY_CENTER_JOINT:BODY_CENTER_JOINT+1, :]
    kp_centered = keypoints_aligned - center  # (N, 22, 3)
    kp_flat = kp_centered.reshape(len(common_frames), -1)  # (N, 66)

    print(f"  Aligned: keypoints={kp_flat.shape}, temporal={temporal_aligned.shape}, "
          f"gaussian={gaussian_aligned.shape}")

    return {
        "kp_raw": keypoints_aligned,
        "kp_centered": kp_centered,
        "kp_flat": kp_flat,  # body-centered, flattened (N, 66)
        "temporal_all": temporal_aligned,  # Strategy C 129d
        "temporal_centroid": temporal_centroid_aligned,  # 7d
        "temporal_bodypart": temporal_bodypart_aligned,  # 88d
        "temporal_rigid": temporal_rigid_aligned,  # 25d
        "gaussian_raw": gaussian_aligned,  # 31d
        "fps": fps,
        "n_frames": len(common_frames),
        "frame_indices": common_arr,
        "kp_names": kp_names,
    }


def generate_hlacs(
    features: np.ndarray,
    k_range: range,
    random_state: int = 42,
    label: str = "features",
) -> Dict[int, Dict]:
    """Generate HLACs via clustering. Body-centered keypoints recommended."""
    print(f"\nGenerating HLACs from {label} {features.shape}...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(features_scaled)

        sil = silhouette_score(features_scaled, labels, sample_size=min(5000, len(labels)),
                               random_state=random_state)
        ch = calinski_harabasz_score(features_scaled, labels)
        db = davies_bouldin_score(features_scaled, labels)

        # Temporal persistence
        transitions = np.sum(labels[1:] != labels[:-1])
        tpi = 1.0 - transitions / (len(labels) - 1)

        results[k] = {
            "labels": labels,
            "metrics": {
                "silhouette": float(sil),
                "calinski_harabasz": float(ch),
                "davies_bouldin": float(db),
                "tpi": float(tpi),
                "n_transitions": int(transitions),
            },
        }
        print(f"  K={k}: Sil={sil:.3f}, TPI={tpi:.3f}, DB={db:.2f}")

    return results


def run_classification(
    feature_sets: Dict[str, np.ndarray],
    labels: np.ndarray,
    k: int,
    random_state: int = 42,
) -> Dict[str, Dict]:
    """Linear probe classification for each feature set."""
    print(f"\n{'='*50}")
    print(f"Classification Probe (K={k})")
    print(f"{'='*50}")

    results = {}
    for name, X in feature_sets.items():
        print(f"\n--- {name} ({X.shape[1]}d) ---")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, labels, test_size=0.2, random_state=random_state, stratify=labels
        )

        clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="ovr",
                                 random_state=random_state, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred, normalize="true")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(clf, X_scaled, labels, cv=cv, scoring="accuracy")

        print(f"  Accuracy: {acc:.4f}, F1-macro: {f1:.4f}, CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")

        results[name] = {
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "confusion_matrix": cm,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    return results


def plot_comparison(clf_results: Dict, k: int, output_dir: Path, title_prefix: str = "M5t2"):
    """Bar chart comparing all feature sets."""
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 300})

    names = list(clf_results.keys())
    acc = [clf_results[n]["accuracy"] for n in names]
    f1 = [clf_results[n]["f1_macro"] for n in names]
    cv = [clf_results[n]["cv_accuracy_mean"] for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 5))
    ax.bar(x - width, acc, width, label="Accuracy", color="#2196F3", alpha=0.85)
    ax.bar(x, f1, width, label="F1-macro", color="#4CAF50", alpha=0.85)
    ax.bar(x + width, cv, width, label="CV Accuracy", color="#FF9800", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"{title_prefix}: Feature Probe Performance (K={k})")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for i, (a, f, c) in enumerate(zip(acc, f1, cv)):
        ax.text(i - width, a + 0.02, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(i, f + 0.02, f"{f:.3f}", ha="center", fontsize=7)
        ax.text(i + width, c + 0.02, f"{c:.3f}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / f"classification_comparison_K{k}.png")
    plt.close()
    print(f"  Saved: classification_comparison_K{k}.png")


def plot_confusion_matrices(clf_results: Dict, k: int, output_dir: Path):
    """Confusion matrices for selected feature sets."""
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": 300})

    # Show top 4 feature sets by accuracy
    sorted_names = sorted(clf_results.keys(), key=lambda n: clf_results[n]["accuracy"], reverse=True)[:4]
    n = len(sorted_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, sorted_names):
        cm = clf_results[name]["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax, vmin=0, vmax=1)
        ax.set_title(f"{name}\n(Acc={clf_results[name]['accuracy']:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.suptitle(f"M5t2 Confusion Matrices (K={k})", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrices_K{k}.png")
    plt.close()
    print(f"  Saved: confusion_matrices_K{k}.png")


def plot_k_sweep(hlac_results: Dict, output_dir: Path, label: str = ""):
    """K sweep metrics plot."""
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 300})
    ks = sorted(hlac_results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(ks, [hlac_results[k]["metrics"]["silhouette"] for k in ks], "o-")
    axes[0].set_ylabel("Silhouette")
    axes[0].set_title("Cluster Separation")

    axes[1].plot(ks, [hlac_results[k]["metrics"]["tpi"] for k in ks], "s-", color="green")
    axes[1].set_ylabel("TPI")
    axes[1].set_title("Temporal Persistence")

    axes[2].plot(ks, [hlac_results[k]["metrics"]["davies_bouldin"] for k in ks], "^-", color="red")
    axes[2].set_ylabel("Davies-Bouldin")
    axes[2].set_title("Overlap (lower=better)")

    for ax in axes:
        ax.set_xlabel("K")
        ax.grid(alpha=0.3)

    plt.suptitle(f"M5t2 K-Sweep {label}")
    plt.tight_layout()
    plt.savefig(output_dir / f"k_sweep_{label.replace(' ', '_')}.png")
    plt.close()
    print(f"  Saved: k_sweep_{label.replace(' ', '_')}.png")


def main():
    parser = argparse.ArgumentParser(description="M5t2 HLAC Analysis")
    parser.add_argument("--output-dir", type=str, default="outputs/m5t2_hlac_analysis")
    parser.add_argument("--target-k", type=int, default=0, help="0=auto-select")
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and align data
    data = load_m5t2_data()

    # 2. Generate HLACs from BODY-CENTERED keypoints
    k_range = range(args.k_min, args.k_max + 1)
    hlac_results = generate_hlacs(data["kp_flat"], k_range, args.seed,
                                  label="body-centered keypoints")

    # Also cluster with Gaussian features for comparison
    hlac_gauss = generate_hlacs(data["gaussian_raw"], k_range, args.seed,
                                label="Gaussian raw features")

    # 3. Select best K (from keypoint clustering)
    if args.target_k > 0:
        target_k = args.target_k
    else:
        scores = {k: r["metrics"]["silhouette"] * 0.5 - r["metrics"]["davies_bouldin"] * 0.2
                  for k, r in hlac_results.items()}
        target_k = max(scores, key=scores.get)
    print(f"\nUsing K={target_k}")

    labels_kp = hlac_results[target_k]["labels"]
    labels_gauss = hlac_gauss[target_k]["labels"]

    # 4. Classification probe — ALL feature sets against KEYPOINT-defined HLACs
    print("\n" + "="*60)
    print("EXPERIMENT A: Keypoint-defined HLACs → Feature Probe")
    print("="*60)
    feature_sets_a = {
        "KP_centered": data["kp_flat"],           # 66d (body-centered keypoints)
        "Temporal_all": data["temporal_all"],      # 129d (Strategy C)
        "Temporal_centroid": data["temporal_centroid"],  # 7d
        "Temporal_bodypart": data["temporal_bodypart"],  # 88d
        "Temporal_rigid": data["temporal_rigid"],  # 25d
        "Gaussian_raw": data["gaussian_raw"],      # 31d ← KEY comparison
    }
    clf_a = run_classification(feature_sets_a, labels_kp, target_k, args.seed)

    # 5. Classification probe — GAUSSIAN-defined HLACs → Feature Probe
    print("\n" + "="*60)
    print("EXPERIMENT B: Gaussian-defined HLACs → Feature Probe")
    print("="*60)
    clf_b = run_classification(feature_sets_a, labels_gauss, target_k, args.seed)

    # 6. Cross-agreement: how much do KP clusters and Gaussian clusters agree?
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(labels_kp, labels_gauss)
    nmi = normalized_mutual_info_score(labels_kp, labels_gauss)
    print(f"\nCluster Agreement (KP vs Gaussian): ARI={ari:.3f}, NMI={nmi:.3f}")

    # 7. Visualizations
    print("\nGenerating visualizations...")
    plot_k_sweep(hlac_results, output_dir, label="keypoint-defined")
    plot_k_sweep(hlac_gauss, output_dir, label="gaussian-defined")
    plot_comparison(clf_a, target_k, output_dir, title_prefix="M5t2 (KP-HLACs)")
    plot_comparison(clf_b, target_k, output_dir, title_prefix="M5t2 (Gauss-HLACs)")
    plot_confusion_matrices(clf_a, target_k, output_dir)

    # 8. Save results
    summary = {
        "dataset": "M5t2",
        "n_frames": data["n_frames"],
        "fps": data["fps"],
        "target_k": target_k,
        "hlac_source": "body-centered keypoints (22 joints)",
        "cluster_agreement": {"ARI": float(ari), "NMI": float(nmi)},
        "experiment_a_kp_hlacs": {
            name: {k: v for k, v in res.items()
                   if k not in ("confusion_matrix", "y_test", "y_pred")}
            for name, res in clf_a.items()
        },
        "experiment_b_gauss_hlacs": {
            name: {k: v for k, v in res.items()
                   if k not in ("confusion_matrix", "y_test", "y_pred")}
            for name, res in clf_b.items()
        },
        "k_sweep_kp": {str(k): r["metrics"] for k, r in hlac_results.items()},
        "k_sweep_gauss": {str(k): r["metrics"] for k, r in hlac_gauss.items()},
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    np.save(output_dir / f"hlac_labels_kp_K{target_k}.npy", labels_kp)
    np.save(output_dir / f"hlac_labels_gauss_K{target_k}.npy", labels_gauss)

    print(f"\n{'='*60}")
    print(f"M5t2 HLAC Analysis Complete!")
    print(f"Output: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
