"""Comprehensive HLAC complementarity analysis — addressing circular bias concerns.

Tests whether foreground-masked Gaussian covariance features complement keypoints
for behavior classification, with proper controls for:
1. Circular bias: uses K-means labels (limitation noted) — s-DANNCE HLAC labels TBD
2. Curse of dimensionality: PCA sweep (10, 20, 50 components)
3. Classifier diversity: Linear SVM, Random Forest, MLP
4. Body-part analysis: per-joint-group feature subsets
5. CCA: canonical correlation between KP and Gaussian features
6. Complementarity test: paired t-test on Combined vs individual

Usage on gpu03:
    python -m mouse_extensions.behavior.hlac_comprehensive \
        --output-dir outputs/features/mouse/hlac/hlac_comprehensive
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy import stats

warnings.filterwarnings("ignore")

# Data paths (SSOT)
from mouse_extensions.paths import KP_22, FEATURES_BASE, TEMPORAL_DIR, COV_N2_DIR, GAUSSIAN_RAW_FEATURES

KP_PATH = str(KP_22)
TEMPORAL_PATH = str(TEMPORAL_DIR / "temporal_features.npz")
GAUSSIAN_RAW_PATH = str(GAUSSIAN_RAW_FEATURES)
COV_N2_STATIC_PATH = str(COV_N2_DIR / "covariance_static.npy")
COV_N2_TEMPORAL_PATH = str(COV_N2_DIR / "covariance_temporal.npy")

BODY_CENTER_JOINT = 4

# Body-part joint groupings (22 MAMMAL joints)
JOINT_GROUPS = {
    "head":     [0, 1, 2, 3],       # Nose, LEar, REar, Neck
    "torso":    [4, 5, 6],           # SpineM, SpineF, TailBase
    "forelimb": [7, 8, 9, 10, 11],   # LShoulder, LElbow, LWrist, RShoulder, RElbow
    "hindlimb": [12, 13, 14, 15, 16, 17],  # RWrist, LHip, LKnee, RHip, RKnee, LAnkle
    "tail":     [18, 19, 20, 21],    # RAnkle, TailMid, TailEnd, + spare
}


def load_data():
    """Load and align all features."""
    print("Loading data...")

    # Keypoints
    kp_data = np.load(KP_PATH, allow_pickle=True)
    keypoints = kp_data["keypoints"]  # (3600, 22, 3)

    # Temporal
    tf_data = np.load(TEMPORAL_PATH, allow_pickle=True)
    temporal_all = tf_data["all_features"]  # (3585, 129)

    # Gaussian raw
    gf_data = np.load(GAUSSIAN_RAW_PATH, allow_pickle=True)
    gaussian_raw = gf_data["features"]  # (3007, 31)
    gf_frame_idx = gf_data["frame_indices"]

    # Covariance N>=2
    cov_static = np.load(COV_N2_STATIC_PATH)   # (3585, 154)
    cov_temporal = np.load(COV_N2_TEMPORAL_PATH)  # (3585, 198)

    print(f"  KP: {keypoints.shape}, Temporal: {temporal_all.shape}")
    print(f"  Gaussian raw: {gaussian_raw.shape}")
    print(f"  Cov N2: static={cov_static.shape}, temporal={cov_temporal.shape}")

    # Align frames
    temporal_frames = set(range(temporal_all.shape[0]))
    gaussian_frames = set(gf_frame_idx.tolist())
    kp_frames = set(range(keypoints.shape[0]))
    common = sorted(temporal_frames & gaussian_frames & kp_frames)
    idx_arr = np.array(common)

    gf_map = {f: i for i, f in enumerate(gf_frame_idx)}
    gi = [gf_map[f] for f in common]

    # Body-centered KP
    kp_aligned = keypoints[idx_arr]
    center = kp_aligned[:, BODY_CENTER_JOINT:BODY_CENTER_JOINT+1, :]
    kp_centered = (kp_aligned - center).reshape(len(common), -1)  # (N, 66)

    data = {
        "kp": kp_centered,
        "gaussian_raw": gaussian_raw[gi],
        "cov_static": cov_static[idx_arr],
        "cov_temporal": cov_temporal[idx_arr],
        "n_frames": len(common),
    }

    # Generate K-means labels (same as hlac_m5t2.py)
    from sklearn.cluster import KMeans
    scaler = StandardScaler()
    kp_scaled = scaler.fit_transform(kp_centered)
    km = KMeans(n_clusters=8, random_state=42, n_init=10)
    data["labels_kp_kmeans"] = km.fit_predict(kp_scaled)

    print(f"  Aligned: {len(common)} frames")
    return data


def run_all_classifiers(X_train, X_test, y_train, y_test, seed=42):
    """Run 3 classifiers, return dict of {name: {acc, f1, y_pred}}."""
    clfs = {
        "LinearSVM": LinearSVC(C=1.0, max_iter=10000, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=200, max_depth=15,
                                     random_state=seed, n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                             random_state=seed, learning_rate_init=0.001),
    }
    results = {}
    for name, clf in clfs.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        }
    return results


def run_cv(X, y, seed=42, n_splits=5):
    """Run 5-fold CV with 3 classifiers, return per-fold F1 scores."""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clfs = {
        "LinearSVM": lambda: LinearSVC(C=1.0, max_iter=10000, random_state=seed),
        "RF": lambda: RandomForestClassifier(n_estimators=200, max_depth=15,
                                              random_state=seed, n_jobs=-1),
        "MLP": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                      random_state=seed, learning_rate_init=0.001),
    }

    results = {name: [] for name in clfs}
    for train_idx, test_idx in kf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        X_tr = scaler.transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]
        for name, clf_fn in clfs.items():
            clf = clf_fn()
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            results[name].append(float(f1_score(y_te, y_pred, average="macro")))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/features/mouse/hlac/hlac_comprehensive")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = load_data()
    labels = data["labels_kp_kmeans"]
    all_results = {}

    # ================================================================
    # PART 1: Feature comparison with 3 classifiers + PCA sweep
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 1: Multi-classifier comparison + PCA sweep")
    print("=" * 60)

    feature_sets = {
        "KP_centered (66d)": data["kp"],
        "Gaussian_raw (31d, no fg mask)": data["gaussian_raw"],
        "Cov_N2_static (154d, fg)": data["cov_static"],
        "Cov_N2_temporal (198d, fg)": data["cov_temporal"],
        "Cov_N2_combined (352d, fg)": np.hstack([data["cov_static"], data["cov_temporal"]]),
    }

    # PCA variants
    for k in [10, 20, 50]:
        scaler = StandardScaler()
        cov_combined = np.hstack([data["cov_static"], data["cov_temporal"]])
        cov_scaled = scaler.fit_transform(cov_combined)
        pca = PCA(n_components=k).fit(cov_scaled)
        feature_sets[f"PCA({k})_Cov_N2 ({k}d, fg)"] = pca.transform(cov_scaled)

        # Combined with KP
        feature_sets[f"KP + PCA({k})_Cov_N2 ({66+k}d)"] = np.hstack([
            data["kp"], pca.transform(cov_scaled)
        ])

    # Full combined
    feature_sets["KP + Cov_N2_static (220d)"] = np.hstack([data["kp"], data["cov_static"]])

    part1 = {}
    for fname, X in feature_sets.items():
        print(f"\n--- {fname} ---")
        cv_results = run_cv(X, labels, seed=args.seed)
        part1[fname] = {}
        for clf_name, scores in cv_results.items():
            mean_f1 = np.mean(scores)
            std_f1 = np.std(scores)
            print(f"  {clf_name}: F1={mean_f1:.4f}±{std_f1:.4f}")
            part1[fname][clf_name] = {
                "mean_f1": round(mean_f1, 4),
                "std_f1": round(std_f1, 4),
                "per_fold": [round(s, 4) for s in scores],
            }
    all_results["part1_feature_comparison"] = part1

    # ================================================================
    # PART 2: Complementarity test (paired t-test)
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 2: Complementarity test (paired t-test)")
    print("=" * 60)

    # Best PCA k from part1
    best_k = max([10, 20, 50], key=lambda k:
        np.mean(part1[f"KP + PCA({k})_Cov_N2 ({66+k}d)"]["RF"]["per_fold"]))
    print(f"Best PCA k (by RF): {best_k}")

    comp_results = {}
    for clf_name in ["LinearSVM", "RF", "MLP"]:
        f1_kp = part1["KP_centered (66d)"][clf_name]["per_fold"]
        f1_gauss = part1[f"PCA({best_k})_Cov_N2 ({best_k}d, fg)"][clf_name]["per_fold"]
        f1_combined = part1[f"KP + PCA({best_k})_Cov_N2 ({66+best_k}d)"][clf_name]["per_fold"]

        t1, p1 = stats.ttest_rel(f1_combined, f1_kp)
        t2, p2 = stats.ttest_rel(f1_combined, f1_gauss)

        delta_kp = np.mean(f1_combined) - np.mean(f1_kp)
        delta_g = np.mean(f1_combined) - np.mean(f1_gauss)
        is_comp = (p1 < 0.05 and p2 < 0.05 and delta_kp > 0.02 and delta_g > 0.02)

        comp_results[clf_name] = {
            "kp_mean": round(np.mean(f1_kp), 4),
            "gauss_mean": round(np.mean(f1_gauss), 4),
            "combined_mean": round(np.mean(f1_combined), 4),
            "delta_vs_kp": round(delta_kp, 4),
            "delta_vs_gauss": round(delta_g, 4),
            "p_vs_kp": round(p1, 4),
            "p_vs_gauss": round(p2, 4),
            "complementary": bool(is_comp),
        }
        status = "✅ YES" if is_comp else "❌ NO"
        print(f"  {clf_name}: Combined={np.mean(f1_combined):.4f}, "
              f"Δ_KP={delta_kp:+.4f} (p={p1:.3f}), "
              f"Δ_Gauss={delta_g:+.4f} (p={p2:.3f}) → {status}")

    all_results["part2_complementarity"] = comp_results
    all_results["part2_best_pca_k"] = best_k

    # ================================================================
    # PART 3: Body-part analysis
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 3: Body-part analysis")
    print("=" * 60)

    bp_results = {}
    for bp_name, joints in JOINT_GROUPS.items():
        kp_cols = [j * 3 + c for j in joints for c in range(3)]
        cov_static_cols = [j * 7 + c for j in joints for c in range(7)]
        cov_temporal_cols = [j * 9 + c for j in joints for c in range(9)]

        X_kp_bp = data["kp"][:, kp_cols]
        X_cov_bp = np.hstack([
            data["cov_static"][:, cov_static_cols],
            data["cov_temporal"][:, cov_temporal_cols],
        ])
        X_combined_bp = np.hstack([X_kp_bp, X_cov_bp])

        bp_results[bp_name] = {}
        for feat_name, X in [("KP_only", X_kp_bp), ("Cov_only", X_cov_bp),
                              ("KP+Cov", X_combined_bp)]:
            cv = run_cv(X, labels, seed=args.seed)
            bp_results[bp_name][feat_name] = {
                clf: {"mean_f1": round(np.mean(s), 4), "std_f1": round(np.std(s), 4)}
                for clf, s in cv.items()
            }

        # Print body-part summary (RF)
        kp_f1 = bp_results[bp_name]["KP_only"]["RF"]["mean_f1"]
        cov_f1 = bp_results[bp_name]["Cov_only"]["RF"]["mean_f1"]
        comb_f1 = bp_results[bp_name]["KP+Cov"]["RF"]["mean_f1"]
        delta = comb_f1 - kp_f1
        marker = "⬆️" if delta > 0.01 else ("⬇️" if delta < -0.01 else "➡️")
        print(f"  {bp_name:12s}: KP={kp_f1:.3f}, Cov={cov_f1:.3f}, "
              f"KP+Cov={comb_f1:.3f} (Δ={delta:+.3f}) {marker}")

    all_results["part3_bodypart"] = bp_results

    # ================================================================
    # PART 4: CCA (Canonical Correlation Analysis)
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 4: CCA between KP and Gaussian features")
    print("=" * 60)

    kp_scaled = StandardScaler().fit_transform(data["kp"])
    cov_combined = np.hstack([data["cov_static"], data["cov_temporal"]])
    cov_scaled = StandardScaler().fit_transform(cov_combined)
    pca_cov = PCA(n_components=best_k).fit_transform(cov_scaled)

    n_comp = min(best_k, 66, 10)
    cca = CCA(n_components=n_comp)
    X_c, Y_c = cca.fit_transform(kp_scaled, pca_cov)

    corrs = [float(np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]) for i in range(n_comp)]
    print(f"  Top {n_comp} canonical correlations: {[f'{c:.3f}' for c in corrs]}")
    print(f"  Mean: {np.mean(corrs):.3f}, Max: {max(corrs):.3f}")

    all_results["part4_cca"] = {
        "canonical_correlations": [round(c, 4) for c in corrs],
        "mean": round(np.mean(corrs), 4),
        "max": round(max(corrs), 4),
        "n_components": n_comp,
    }

    # CCA interpretation
    if max(corrs) > 0.7:
        print("  → HIGH correlation: significant shared information")
    elif max(corrs) > 0.4:
        print("  → MODERATE correlation: partial overlap")
    else:
        print("  → LOW correlation: largely independent feature spaces")

    # ================================================================
    # Save results
    # ================================================================
    results_path = out / "comprehensive_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"⚠️  Labels: K-means on KP (circular bias — s-DANNCE HLAC TBD)")
    print(f"Best PCA k: {best_k}")
    for clf_name, r in comp_results.items():
        print(f"  {clf_name}: Complementary = {r['complementary']} "
              f"(Δ_KP={r['delta_vs_kp']:+.4f}, p={r['p_vs_kp']:.3f})")


if __name__ == "__main__":
    main()
