"""H1 Probe: Do engineered features add value over raw keypoints for HLAC classification?

Tests whether derived kinematic features (S3) improve upon raw skeleton pose (S1)
for classifying s-DANNCE HLAC reference labels. Pre-registered decision boundary.

This is the BASELINE probe using s-DANNCE keypoints + labels.
Covariance features from GS-LRM will be added as a separate comparison
once GS-LRM reconstruction of s-DANNCE data is available.

Feature sets compared:
  - S1: Raw COM-centered keypoints (69d per animal)
  - S3_single: Single-animal kinematics (velocity, body_length, z_height, acceleration)
  - S3_dyadic: Dyadic kinematics (inter-animal distances, heading angle, proximity)
  - S1+S3: Combined

Pre-registered decision boundary:
  - CCA < 0.7 on >=2 components (feature independence)
  - Accuracy >= 2%p improvement (practical significance)
  - McNemar p < 0.01/N_comparisons (Bonferroni-corrected)

Usage:
    cd /home/joon/dev/FaceLift
    python -m mouse_extensions.behavior.h1_sdannce_probe \
        --mat-file /home/joon/data/sdannce/mouse/dataverse/MOUSE_B1_20240428_0061_S.mat \
        --output-dir outputs/sdannce_poc/h1_probe \
        --animal m1
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.io import loadmat
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# s-DANNCE 23 keypoint indices
IDX_NOSE = 0
IDX_SPINE_F = 3    # Neck
IDX_SPINE_M = 4    # Left Shoulder (approximate mid-spine)
IDX_TAIL_BASE = 6  # Left Elbow maps to... wait, need correct indices

# Correct indices from Dataset_sDannce.md keypoint table
# 0=Nose, 1=LEar, 2=REar, 3=Neck, 4=LShoulder, 5=RShoulder,
# 6=LElbow, 7=RElbow, 8=LWrist, 9=RWrist, 10=LHip, 11=RHip,
# 12=LKnee, 13=RKnee, 14=LAnkle, 15=RAnkle, 16=TailBase,
# 17=MidTail, 18=TipTail, 19=HeadCenter(derived), 20=BackCenter(derived),
# 21=MidBodyCenter(derived), 22=TailCenter(derived)
IDX_NOSE = 0
IDX_NECK = 3
IDX_L_SHOULDER = 4
IDX_R_SHOULDER = 5
IDX_L_HIP = 10
IDX_R_HIP = 11
IDX_TAIL_BASE = 16
IDX_MID_TAIL = 17
IDX_TIP_TAIL = 18

# Approximate body-center indices for velocity/height
IDX_BODY_MID = 21  # derived: mid body center
IDX_BACK_CENTER = 20  # derived: back center


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# Data Loading
# =============================================================================

def load_sdannce_mat(mat_path: str, animal: str = "m1") -> Dict:
    """Load s-DANNCE Dataverse .mat file.

    Args:
        mat_path: Path to .mat file
        animal: "m1" (primary) or "m2" (partner)

    Returns:
        dict with kp, hlac, metadata
    """
    m = loadmat(mat_path, squeeze_me=False)
    sd = m["sdannce"][0, 0]

    kp = sd[animal]  # (T, 3, 23)
    kp = kp.transpose(0, 2, 1)  # -> (T, 23, 3)

    hlac_field = "hlac" if animal == "m1" else "part_hlac"
    hlac = sd[hlac_field].flatten().astype(int)

    # Partner keypoints (for dyadic features)
    partner = "m2" if animal == "m1" else "m1"
    kp_partner = None
    is_social = int(sd["issoc"].flat[0]) == 1
    if is_social and sd[partner].size > 0:
        kp_partner = sd[partner].transpose(0, 2, 1)  # (T, 23, 3)

    return {
        "kp": kp.astype(np.float32),
        "kp_partner": kp_partner.astype(np.float32) if kp_partner is not None else None,
        "hlac": hlac,
        "is_social": is_social,
        "strain": str(sd["mousestrain"].flat[0]),
        "mouse_id": str(sd["mouseid"].flat[0]),
        "date": str(sd["date"].flat[0]),
        "n_frames": kp.shape[0],
    }


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_s1(kp: np.ndarray) -> np.ndarray:
    """S1: COM-centered raw keypoints. (T, 23, 3) -> (T, 69)."""
    com = kp.mean(axis=1, keepdims=True)  # (T, 1, 3)
    kp_centered = kp - com
    return kp_centered.reshape(kp.shape[0], -1).astype(np.float32)


def _velocity(kp_seq: np.ndarray, fps: float = 50.0) -> np.ndarray:
    """Speed of a keypoint trajectory. (T, 3) -> (T,)."""
    diff = np.zeros_like(kp_seq)
    diff[1:-1] = (kp_seq[2:] - kp_seq[:-2]) / 2.0
    diff[0] = kp_seq[1] - kp_seq[0]
    diff[-1] = kp_seq[-1] - kp_seq[-2]
    return np.linalg.norm(diff, axis=1) * fps


def _acceleration(kp_seq: np.ndarray, fps: float = 50.0) -> np.ndarray:
    """Acceleration magnitude. (T, 3) -> (T,)."""
    vel = np.zeros_like(kp_seq)
    vel[1:-1] = (kp_seq[2:] - kp_seq[:-2]) / 2.0
    vel[0] = kp_seq[1] - kp_seq[0]
    vel[-1] = kp_seq[-1] - kp_seq[-2]
    acc = np.zeros_like(vel)
    acc[1:-1] = (vel[2:] - vel[:-2]) / 2.0
    acc[0] = vel[1] - vel[0]
    acc[-1] = vel[-1] - vel[-2]
    return np.linalg.norm(acc, axis=1) * fps * fps


def extract_s3_single(kp: np.ndarray, fps: float = 50.0) -> Tuple[np.ndarray, List[str]]:
    """S3 single-animal kinematics. (T, 23, 3) -> (T, 8).

    Features:
        0: nose_velocity (mm/s)
        1: body_center_velocity (mm/s)
        2: body_length (nose-tail_base distance, mm)
        3: trunk_length (neck-tail_base distance, mm)
        4: z_height_spine (rearing indicator)
        5: nose_acceleration
        6: body_compactness (shoulder-hip distance)
        7: tail_curvature (angle: tail_base-mid_tail-tip_tail)
    """
    T = kp.shape[0]
    feats = np.zeros((T, 8), dtype=np.float32)

    feats[:, 0] = _velocity(kp[:, IDX_NOSE], fps)
    feats[:, 1] = _velocity(kp[:, IDX_NECK], fps)
    feats[:, 2] = np.linalg.norm(kp[:, IDX_NOSE] - kp[:, IDX_TAIL_BASE], axis=1)
    feats[:, 3] = np.linalg.norm(kp[:, IDX_NECK] - kp[:, IDX_TAIL_BASE], axis=1)
    feats[:, 4] = kp[:, IDX_NECK, 2]  # z-height of neck
    feats[:, 5] = _acceleration(kp[:, IDX_NOSE], fps)

    # Body compactness: mean shoulder-to-hip distance
    sh_dist = (np.linalg.norm(kp[:, IDX_L_SHOULDER] - kp[:, IDX_L_HIP], axis=1)
               + np.linalg.norm(kp[:, IDX_R_SHOULDER] - kp[:, IDX_R_HIP], axis=1)) / 2
    feats[:, 6] = sh_dist

    # Tail curvature: angle at mid-tail
    v1 = kp[:, IDX_TAIL_BASE] - kp[:, IDX_MID_TAIL]
    v2 = kp[:, IDX_TIP_TAIL] - kp[:, IDX_MID_TAIL]
    cos_a = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8)
    feats[:, 7] = np.arccos(np.clip(cos_a, -1, 1))

    names = [
        "nose_vel", "body_vel", "body_length", "trunk_length",
        "z_height", "nose_accel", "body_compactness", "tail_curvature",
    ]
    return feats, names


def extract_s3_dyadic(
    kp: np.ndarray, kp_partner: np.ndarray, fps: float = 50.0
) -> Tuple[np.ndarray, List[str]]:
    """S3 dyadic features (social interactions). (T, 23, 3) x2 -> (T, 6).

    Features:
        0: inter_nose_distance
        1: inter_centroid_distance
        2: relative_heading_angle
        3: nose_to_partner_tail_dist
        4: partner_nose_to_tail_dist
        5: approach_velocity (change in centroid distance)
    """
    T = kp.shape[0]
    feats = np.zeros((T, 6), dtype=np.float32)

    feats[:, 0] = np.linalg.norm(kp[:, IDX_NOSE] - kp_partner[:, IDX_NOSE], axis=1)

    com1 = kp.mean(axis=1)
    com2 = kp_partner.mean(axis=1)
    feats[:, 1] = np.linalg.norm(com1 - com2, axis=1)

    # Relative heading angle
    axis1 = kp[:, IDX_NOSE] - kp[:, IDX_TAIL_BASE]
    axis2 = kp_partner[:, IDX_NOSE] - kp_partner[:, IDX_TAIL_BASE]
    cos_a = np.sum(axis1 * axis2, axis=1) / (
        np.linalg.norm(axis1, axis=1) * np.linalg.norm(axis2, axis=1) + 1e-8
    )
    feats[:, 2] = np.arccos(np.clip(cos_a, -1, 1))

    feats[:, 3] = np.linalg.norm(kp[:, IDX_NOSE] - kp_partner[:, IDX_TAIL_BASE], axis=1)
    feats[:, 4] = np.linalg.norm(kp_partner[:, IDX_NOSE] - kp[:, IDX_TAIL_BASE], axis=1)

    # Approach velocity (negative = approaching)
    inter_dist = feats[:, 1]
    feats[1:, 5] = np.diff(inter_dist) * fps
    feats[0, 5] = feats[1, 5]

    names = [
        "inter_nose_dist", "inter_com_dist", "heading_angle",
        "nose_to_partner_tail", "partner_nose_to_tail", "approach_vel",
    ]
    return feats, names


# =============================================================================
# GroupKFold Classification
# =============================================================================

def make_groups(n_frames: int, fps: float = 50.0, group_duration_s: float = 15.0) -> np.ndarray:
    """Assign group IDs for GroupKFold (15-second blocks)."""
    frames_per_group = int(fps * group_duration_s)
    return np.arange(n_frames) // frames_per_group


def run_classification(
    feature_sets: Dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, Dict]:
    """Run GroupKFold L2 LogReg for each feature set.

    Returns per-feature-set results with accuracy, f1, predictions.
    """
    results = {}
    gkf = GroupKFold(n_splits=n_splits)

    for name, X in feature_sets.items():
        print(f"  {name} ({X.shape[1]}d)...", end=" ")

        all_preds = np.full(len(labels), -1, dtype=int)
        all_true = np.full(len(labels), -1, dtype=int)
        fold_accs = []

        for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, labels, groups)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            clf = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )
            clf.fit(X_train, labels[train_idx])

            preds = clf.predict(X_test)
            all_preds[test_idx] = preds
            all_true[test_idx] = labels[test_idx]

            acc = accuracy_score(labels[test_idx], preds)
            fold_accs.append(acc)

        overall_acc = accuracy_score(all_true, all_preds)
        overall_f1 = f1_score(all_true, all_preds, average="macro")

        print(f"acc={overall_acc:.4f}, f1_macro={overall_f1:.4f}")

        results[name] = {
            "accuracy": overall_acc,
            "f1_macro": overall_f1,
            "fold_accuracies": fold_accs,
            "predictions": all_preds,
            "true_labels": all_true,
            "n_features": X.shape[1],
        }

    return results


# =============================================================================
# Statistical Tests
# =============================================================================

def mcnemar_test(preds_a: np.ndarray, preds_b: np.ndarray, true: np.ndarray) -> Dict:
    """McNemar's test for paired classifier comparison."""
    correct_a = preds_a == true
    correct_b = preds_b == true

    # Contingency: b_wrong_a_right, b_right_a_wrong
    b01 = np.sum(correct_a & ~correct_b)  # A right, B wrong
    b10 = np.sum(~correct_a & correct_b)  # A wrong, B right

    n = b01 + b10
    if n == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n_discordant": 0}

    # McNemar with continuity correction
    chi2 = (abs(b01 - b10) - 1) ** 2 / n
    from scipy.stats import chi2 as chi2_dist
    p = chi2_dist.sf(chi2, df=1)

    return {"chi2": float(chi2), "p_value": float(p), "n_discordant": int(n)}


def cca_analysis(X_a: np.ndarray, X_b: np.ndarray, n_components: int = 5) -> Dict:
    """CCA between two feature sets.

    Returns canonical correlations and whether independence criterion is met.
    """
    n_comp = min(n_components, X_a.shape[1], X_b.shape[1])
    scaler_a = StandardScaler()
    scaler_b = StandardScaler()
    Xa = scaler_a.fit_transform(X_a)
    Xb = scaler_b.fit_transform(X_b)

    cca = CCA(n_components=n_comp, max_iter=1000)
    Xa_c, Xb_c = cca.fit_transform(Xa, Xb)

    correlations = []
    for i in range(n_comp):
        r = np.corrcoef(Xa_c[:, i], Xb_c[:, i])[0, 1]
        correlations.append(float(r))

    # Independence criterion: CCA < 0.7 on >= 2 components
    n_independent = sum(1 for r in correlations if r < 0.7)
    passes_criterion = n_independent >= 2

    return {
        "correlations": correlations,
        "n_independent_components": n_independent,
        "passes_criterion": passes_criterion,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_accuracy_comparison(results: Dict, output_dir: Path):
    """Bar chart of accuracy ± std across feature sets."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    stds = [np.std(results[n]["fold_accuracies"]) for n in names]
    dims = [results[n]["n_features"] for n in names]

    bars = ax.bar(range(len(names)), accs, yerr=stds, capsize=5,
                  color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"][:len(names)],
                  edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\n({d}d)" for n, d in zip(names, dims)], fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("H1 Probe: HLAC Classification Accuracy\n(GroupKFold 15s, L2 LogReg, 5-fold)")
    ax.set_ylim(0, 1)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "accuracy_comparison.png", dpi=150)
    plt.close()


def plot_confusion_matrices(results: Dict, output_dir: Path):
    """Normalized confusion matrices for each feature set."""
    n_sets = len(results)
    fig, axes = plt.subplots(1, n_sets, figsize=(5 * n_sets, 4))
    if n_sets == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(res["true_labels"], res["predictions"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(cm_norm, ax=ax, cmap="Blues", vmin=0, vmax=1,
                    square=True, cbar_kws={"shrink": 0.8},
                    xticklabels=True, yticklabels=True)
        ax.set_title(f"{name}\nacc={res['accuracy']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=150)
    plt.close()


def plot_cca_components(cca_result: Dict, output_dir: Path, pair_name: str):
    """CCA canonical correlations bar chart."""
    fig, ax = plt.subplots(figsize=(6, 4))
    corrs = cca_result["correlations"]
    colors = ["#C44E52" if r >= 0.7 else "#55A868" for r in corrs]

    ax.bar(range(len(corrs)), corrs, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0.7, color="red", linestyle="--", linewidth=1, label="Threshold (0.7)")
    ax.set_xlabel("Canonical Component")
    ax.set_ylabel("Correlation")
    ax.set_title(f"CCA: {pair_name}\n({cca_result['n_independent_components']} independent components)")
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / f"cca_{pair_name.replace(' ', '_').lower()}.png", dpi=150)
    plt.close()


def plot_hlac_distribution(hlac: np.ndarray, output_dir: Path):
    """HLAC class distribution bar chart."""
    unique, counts = np.unique(hlac, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(unique)), counts / len(hlac) * 100, color="#4C72B0", edgecolor="black")
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([f"HLAC {u}" for u in unique], rotation=45, fontsize=8)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"HLAC Distribution (N={len(hlac)}, {len(unique)} classes)")
    plt.tight_layout()
    fig.savefig(output_dir / "hlac_distribution.png", dpi=150)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="H1 Probe: engineered features vs raw keypoints")
    parser.add_argument("--mat-file", required=True, help="Path to s-DANNCE Dataverse .mat")
    parser.add_argument("--output-dir", default="outputs/sdannce_poc/h1_probe", help="Output directory")
    parser.add_argument("--animal", default="m1", choices=["m1", "m2"], help="Which animal to analyze")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--group-duration", type=float, default=15.0, help="Group duration (seconds)")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading {args.mat_file} (animal={args.animal})...")
    data = load_sdannce_mat(args.mat_file, args.animal)
    print(f"  Strain: {data['strain']}, ID: {data['mouse_id']}")
    print(f"  Frames: {data['n_frames']}, Social: {data['is_social']}")
    print(f"  HLAC classes: {np.unique(data['hlac'])}")

    # Extract features
    print("\nExtracting features...")
    s1 = extract_s1(data["kp"])
    s3_single, s3_single_names = extract_s3_single(data["kp"], args.fps)

    feature_sets = {
        "S1_raw": s1,
        "S3_single": s3_single,
        "S1+S3_single": np.concatenate([s1, s3_single], axis=1),
    }

    if data["is_social"] and data["kp_partner"] is not None:
        s3_dyadic, s3_dyadic_names = extract_s3_dyadic(data["kp"], data["kp_partner"], args.fps)
        feature_sets["S3_dyadic"] = s3_dyadic
        feature_sets["S1+S3_all"] = np.concatenate([s1, s3_single, s3_dyadic], axis=1)

    for name, X in feature_sets.items():
        print(f"  {name}: {X.shape}")

    # Groups for GroupKFold
    groups = make_groups(data["n_frames"], args.fps, args.group_duration)
    n_groups = len(np.unique(groups))
    print(f"\nGroups: {n_groups} (15s blocks)")

    # Plot HLAC distribution
    plot_hlac_distribution(data["hlac"], output_dir)

    # Classification
    print("\nRunning GroupKFold classification...")
    results = run_classification(feature_sets, data["hlac"], groups, args.n_splits)

    # Statistical tests
    print("\nStatistical tests...")
    comparisons = []
    set_names = list(results.keys())
    n_comparisons = len(set_names) * (len(set_names) - 1) // 2
    bonferroni_alpha = 0.01 / max(n_comparisons, 1)

    for i in range(len(set_names)):
        for j in range(i + 1, len(set_names)):
            a, b = set_names[i], set_names[j]
            mc = mcnemar_test(results[a]["predictions"], results[b]["predictions"], results[a]["true_labels"])
            acc_diff = results[b]["accuracy"] - results[a]["accuracy"]

            sig = mc["p_value"] < bonferroni_alpha and abs(acc_diff) >= 0.02
            comparisons.append({
                "pair": f"{a} vs {b}",
                "accuracy_diff": acc_diff,
                "mcnemar_p": mc["p_value"],
                "mcnemar_chi2": mc["chi2"],
                "n_discordant": mc["n_discordant"],
                "bonferroni_alpha": bonferroni_alpha,
                "significant": sig,
            })
            marker = "***" if sig else ""
            print(f"  {a} vs {b}: Δacc={acc_diff:+.4f}, p={mc['p_value']:.6f} {marker}")

    # CCA: S1 vs S3
    print("\nCCA analysis...")
    cca_pairs = [("S1_raw", "S3_single")]
    if "S3_dyadic" in feature_sets:
        cca_pairs.append(("S1_raw", "S3_dyadic"))

    cca_results = {}
    for fa, fb in cca_pairs:
        pair_name = f"{fa} vs {fb}"
        cca_res = cca_analysis(feature_sets[fa], feature_sets[fb])
        cca_results[pair_name] = cca_res
        print(f"  {pair_name}: correlations={[f'{r:.3f}' for r in cca_res['correlations']]}")
        print(f"    Independent components: {cca_res['n_independent_components']}, "
              f"passes criterion: {cca_res['passes_criterion']}")
        plot_cca_components(cca_res, output_dir, pair_name)

    # Decision boundary evaluation
    print("\n" + "=" * 60)
    print("PRE-REGISTERED DECISION BOUNDARY")
    print("=" * 60)

    # Check: does any combined set beat S1_raw?
    s1_acc = results["S1_raw"]["accuracy"]
    verdict_passed = False
    for name in ["S1+S3_single", "S1+S3_all"]:
        if name not in results:
            continue
        combined_acc = results[name]["accuracy"]
        acc_diff = combined_acc - s1_acc
        mc_pair = [c for c in comparisons if f"S1_raw vs {name}" in c["pair"]]
        mc_sig = mc_pair[0]["significant"] if mc_pair else False

        # CCA criterion (any pair)
        cca_pass = any(r["passes_criterion"] for r in cca_results.values())

        print(f"\n  {name}:")
        print(f"    Accuracy improvement: {acc_diff:+.4f} (threshold: >=0.02) {'PASS' if acc_diff >= 0.02 else 'FAIL'}")
        print(f"    McNemar significant: {mc_sig} (Bonferroni α={bonferroni_alpha:.6f})")
        print(f"    CCA independence: {cca_pass}")

        if acc_diff >= 0.02 and mc_sig and cca_pass:
            verdict_passed = True
            print(f"    → VERDICT: PASS — Engineered features add significant value")
        else:
            print(f"    → VERDICT: FAIL — Insufficient evidence for added value")

    # Visualizations
    print("\nGenerating visualizations...")
    plot_accuracy_comparison(results, output_dir)
    plot_confusion_matrices(results, output_dir)

    # Save results
    save_results = {
        "metadata": {
            "mat_file": str(args.mat_file),
            "animal": args.animal,
            "strain": data["strain"],
            "is_social": data["is_social"],
            "n_frames": data["n_frames"],
            "n_groups": int(n_groups),
            "fps": args.fps,
            "group_duration_s": args.group_duration,
            "n_splits": args.n_splits,
        },
        "feature_dimensions": {name: int(res["n_features"]) for name, res in results.items()},
        "accuracies": {name: res["accuracy"] for name, res in results.items()},
        "f1_macro": {name: res["f1_macro"] for name, res in results.items()},
        "fold_accuracies": {name: res["fold_accuracies"] for name, res in results.items()},
        "comparisons": comparisons,
        "cca": cca_results,
        "verdict": {
            "passed": verdict_passed,
            "s1_accuracy": s1_acc,
            "decision_boundary": {
                "accuracy_threshold": 0.02,
                "mcnemar_alpha": 0.01,
                "bonferroni_alpha": bonferroni_alpha,
                "cca_threshold": 0.7,
                "cca_min_independent": 2,
            },
        },
    }

    results_path = output_dir / "h1_probe_results.json"
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {results_path}")

    # Post-mortem checklist (confirmation bias prevention)
    print("\n" + "=" * 60)
    print("POST-MORTEM CHECKLIST (confirmation bias prevention)")
    print("=" * 60)
    print("[ ] Are the HLAC labels independent of the features tested?")
    print("    (HLACs from SocialMapper, features from raw keypoints — YES)")
    print("[ ] Could temporal autocorrelation inflate results?")
    print("    (GroupKFold 15s blocks should prevent — check fold variance)")
    fold_vars = {n: np.std(r["fold_accuracies"]) for n, r in results.items()}
    print(f"    Fold std: {fold_vars}")
    print("[ ] Is the accuracy improvement practically meaningful?")
    print(f"    Best vs S1: {max(r['accuracy'] for r in results.values()) - s1_acc:+.4f}")
    print("[ ] Are results consistent across folds?")


if __name__ == "__main__":
    main()
