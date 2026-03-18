"""Phase 3A Full Ablation — Run on gpu03 with UMAP + B-SOiD + SUBTLE + Generic.

Fine-grained joint groups (4, 7, 10, 12, 15, 18, 22) with proper preprocessing.

Usage on gpu03:
    cd /home/joon/dev/FaceLift
    CUDA_VISIBLE_DEVICES="" python -m mouse_extensions.behavior.run_full_ablation_gpu03
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Frame jump masking
FRAME_JUMP_INDICES = [1180, 2360, 3540]
FRAME_JUMP_MARGIN = 2

# Fine-grained joint groups (Gemini-designed, ethologically motivated)
JOINT_GROUPS = {
    "ultra_4": {
        "indices": [2, 3, 4, 5],  # nose, neck, body_middle, tail_root
        "rationale": "Central spine only — global orientation, spine curvature",
    },
    "minimal_7": {
        "indices": [2, 3, 4, 5, 8, 12, 16],  # + L_paw, R_paw, L_foot
        "rationale": "Core locomotion — contact points for 3 limbs + head",
    },
    "loco_10": {
        "indices": [2, 3, 4, 5, 8, 12, 16, 19, 11, 15],  # + R_foot, L/R_shoulder
        "rationale": "Full locomotion — all 4 contact points + shoulder anchors",
    },
    "posture_12": {
        "indices": [2, 3, 4, 5, 8, 12, 16, 19, 11, 15, 18, 21],  # + L/R_hip
        "rationale": "Posture — all proximal joints (shoulders + hips) for stance",
    },
    "forelimb_15": {
        "indices": [2, 3, 4, 5, 6, 8, 10, 11, 12, 14, 15, 16, 18, 19, 21],  # + elbows, tail_mid
        "rationale": "Forelimb detail — elbows for grooming + tail dynamics",
    },
    "kinematic_18": {
        "indices": [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21],  # + knees, tail_end
        "rationale": "Full kinematics — all limb joints + complete tail",
    },
    "full_22": {
        "indices": list(range(22)),
        "rationale": "Complete skeleton — ears + paw_end for sensory/fine motor",
    },
}

KEYPOINT_NAMES = [
    "L_ear", "R_ear", "nose", "neck", "body_middle", "tail_root",
    "tail_middle", "tail_end", "L_paw", "L_paw_end", "L_elbow", "L_shoulder",
    "R_paw", "R_paw_end", "R_elbow", "R_shoulder", "L_foot", "L_knee",
    "L_hip", "R_foot", "R_knee", "R_hip",
]


def preprocess_keypoints(kp: np.ndarray, normalize_size: bool = True) -> np.ndarray:
    """Per-frame centering and optional body-size normalization.

    Args:
        kp: (T, K, 3) raw keypoints in mm
        normalize_size: if True, divide by per-frame nose-tail_root distance

    Returns:
        (T, K, 3) preprocessed keypoints
    """
    T, K, D = kp.shape

    # Per-frame center of mass subtraction
    com = kp.mean(axis=1, keepdims=True)  # (T, 1, 3)
    kp_centered = kp - com

    if normalize_size:
        # Normalize by nose-tail_root distance (body size proxy)
        nose_idx = 2
        tail_idx = 5
        body_size = np.linalg.norm(
            kp_centered[:, nose_idx] - kp_centered[:, tail_idx], axis=1, keepdims=True
        )  # (T, 1)
        body_size = np.clip(body_size, 10.0, None)  # min 10mm to avoid division issues
        kp_centered = kp_centered / body_size[:, :, np.newaxis]  # broadcast (T, K, 3)

    return kp_centered


def mask_frame_jumps(kp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = kp.shape[0]
    valid = np.ones(T, dtype=bool)
    for idx in FRAME_JUMP_INDICES:
        s, e = max(0, idx - FRAME_JUMP_MARGIN), min(T, idx + FRAME_JUMP_MARGIN + 1)
        valid[s:e] = False
    return kp[valid], valid


def compute_metrics(labels: np.ndarray, features: np.ndarray) -> dict:
    valid = labels >= 0
    n_valid, n_clusters = valid.sum(), len(set(labels[valid]))
    if n_clusters < 2 or n_valid < 10:
        return {"n_clusters": int(n_clusters), "silhouette": float("nan")}

    f, l = features[valid], labels[valid]
    sample = min(5000, n_valid)
    return {
        "n_clusters": int(n_clusters),
        "n_valid": int(n_valid),
        "silhouette": round(float(silhouette_score(f, l, sample_size=sample)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(f, l)), 2),
        "davies_bouldin": round(float(davies_bouldin_score(f, l)), 4),
    }


def temporal_metrics(labels: np.ndarray, fps: float = 20.0) -> dict:
    valid = labels >= 0
    lab = labels[valid]
    if len(lab) < 2:
        return {}
    changes = np.where(np.diff(lab) != 0)[0]
    bouts = np.diff(np.concatenate([[0], changes + 1, [len(lab)]])) / fps
    return {
        "bout_mean_sec": round(float(bouts.mean()), 3),
        "bout_median_sec": round(float(np.median(bouts)), 3),
        "n_bouts": int(len(bouts)),
        "transition_rate": round(float(len(changes) / (len(lab) / fps)), 3),
    }


def run_pca_kmeans(features: np.ndarray, k_range: list[int] = None) -> dict:
    """PCA + KMeans with optimal K selection."""
    if k_range is None:
        k_range = [4, 6, 8, 10, 12]

    pca = PCA(n_components=min(20, features.shape[1]))
    feat_pca = pca.fit_transform(features)

    best_sil, best_k = -1, k_range[0]
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(feat_pca)
        sil = silhouette_score(feat_pca, labels, sample_size=min(3000, len(feat_pca)))
        if sil > best_sil:
            best_sil, best_k = sil, k

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(feat_pca)

    return {
        "method": "PCA_KMeans",
        "labels": labels,
        "features_pca": feat_pca,
        "pca_var_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
        **compute_metrics(labels, feat_pca),
        **temporal_metrics(labels),
    }


def run_bsoid(kp_subset: np.ndarray, fps: int = 20) -> dict:
    """Run B-SOiD (3D-extended)."""
    try:
        from behavior_lab.models.discovery.bsoid import BSOiD
        model = BSOiD(fps=fps, min_cluster_size=30)
        result = model.fit_predict(kp_subset)
        labels = result.labels
        feat = kp_subset[:len(labels)].reshape(len(labels), -1)
        return {
            "method": "B-SOiD",
            "labels": labels,
            **compute_metrics(labels, feat),
            **temporal_metrics(labels),
        }
    except Exception as e:
        return {"method": "B-SOiD", "error": str(e)}


def run_subtle(kp_subset: np.ndarray, fps: int = 20) -> dict:
    """Run SUBTLE."""
    try:
        from behavior_lab.models.discovery.subtle_wrapper import SUBTLE
        model = SUBTLE(fps=fps, n_train_frames=min(120000, len(kp_subset)))
        result = model.fit([kp_subset])
        labels = result.labels if hasattr(result, "labels") else result["labels"]
        feat = kp_subset[:len(labels)].reshape(len(labels), -1)
        return {
            "method": "SUBTLE",
            "labels": labels,
            **compute_metrics(labels, feat),
            **temporal_metrics(labels),
        }
    except Exception as e:
        return {"method": "SUBTLE", "error": str(e)}


def run_umap_hdbscan(features: np.ndarray) -> dict:
    """UMAP + HDBSCAN."""
    try:
        import umap
        import hdbscan

        pca = PCA(n_components=min(30, features.shape[1]))
        feat_pca = pca.fit_transform(features)

        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.0, random_state=42)
        embedding = reducer.fit_transform(feat_pca)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10)
        labels = clusterer.fit_predict(embedding)

        return {
            "method": "UMAP_HDBSCAN",
            "labels": labels,
            "embedding_2d": embedding,
            **compute_metrics(labels, feat_pca),
            **temporal_metrics(labels),
        }
    except Exception as e:
        return {"method": "UMAP_HDBSCAN", "error": str(e)}


def main():
    output_dir = Path("outputs/clustering/sparse/results_full")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    kp_path = "/node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz"
    print(f"Loading {kp_path}")
    data = np.load(kp_path, allow_pickle=True)
    kp_raw = data["keypoints"]

    # Mask frame jumps
    kp_masked, valid_mask = mask_frame_jumps(kp_raw)
    print(f"Frame masking: {kp_raw.shape[0]} → {kp_masked.shape[0]}")

    # Preprocess: center + normalize
    kp_proc = preprocess_keypoints(kp_masked, normalize_size=True)
    print(f"Preprocessing: COM-centered + body-size normalized")
    print(f"  Post-proc COM mean: {kp_proc.mean(axis=(0,1)).round(4)}")
    print(f"  Post-proc std: {kp_proc.std(axis=(0,1)).round(4)}")

    # Also run without normalization for comparison
    kp_centered = preprocess_keypoints(kp_masked, normalize_size=False)

    all_results = {}

    for gname, ginfo in JOINT_GROUPS.items():
        indices = ginfo["indices"]
        n_joints = len(indices)
        joint_names = [KEYPOINT_NAMES[i] for i in indices]

        print(f"\n{'='*70}")
        print(f"Group: {gname} ({n_joints} joints) — {ginfo['rationale']}")

        kp_g = kp_proc[:, indices, :]
        feat_flat = kp_g.reshape(len(kp_g), -1)

        group_results = {}

        # 1. PCA + KMeans (always works)
        print(f"  PCA+KMeans...", end=" ", flush=True)
        t0 = time.time()
        r = run_pca_kmeans(feat_flat)
        r["time_sec"] = round(time.time() - t0, 1)
        group_results["PCA_KMeans"] = {k: v for k, v in r.items() if k not in ("labels", "features_pca")}
        np.save(output_dir / f"{gname}_PCA_KMeans_labels.npy", r["labels"])
        print(f"K={r.get('n_clusters','?')} Sil={r.get('silhouette','?')} ({r['time_sec']}s)")

        # 2. UMAP + HDBSCAN
        print(f"  UMAP+HDBSCAN...", end=" ", flush=True)
        t0 = time.time()
        r = run_umap_hdbscan(feat_flat)
        r["time_sec"] = round(time.time() - t0, 1)
        if "labels" in r:
            np.save(output_dir / f"{gname}_UMAP_HDBSCAN_labels.npy", r["labels"])
            if "embedding_2d" in r:
                np.save(output_dir / f"{gname}_UMAP_HDBSCAN_embed.npy", r["embedding_2d"])
        group_results["UMAP_HDBSCAN"] = {k: v for k, v in r.items() if k not in ("labels", "embedding_2d")}
        print(f"K={r.get('n_clusters','?')} Sil={r.get('silhouette','?')} ({r['time_sec']}s)")

        # 3. B-SOiD (3D)
        print(f"  B-SOiD (3D)...", end=" ", flush=True)
        t0 = time.time()
        r = run_bsoid(kp_g, fps=20)
        r["time_sec"] = round(time.time() - t0, 1)
        if "labels" in r:
            np.save(output_dir / f"{gname}_BSOID_labels.npy", r["labels"])
        group_results["B-SOiD"] = {k: v for k, v in r.items() if k != "labels"}
        print(f"K={r.get('n_clusters','err')} Sil={r.get('silhouette','err')} ({r['time_sec']}s)")

        # 4. SUBTLE (if 3D)
        print(f"  SUBTLE...", end=" ", flush=True)
        t0 = time.time()
        r = run_subtle(kp_g, fps=20)
        r["time_sec"] = round(time.time() - t0, 1)
        if "labels" in r:
            np.save(output_dir / f"{gname}_SUBTLE_labels.npy", r["labels"])
        group_results["SUBTLE"] = {k: v for k, v in r.items() if k != "labels"}
        print(f"K={r.get('n_clusters','err')} Sil={r.get('silhouette','err')} ({r['time_sec']}s)")

        all_results[gname] = {
            "n_joints": n_joints,
            "joint_names": joint_names,
            "rationale": ginfo["rationale"],
            "methods": group_results,
        }

    # Also run PCA+KMeans WITHOUT body-size normalization for comparison
    print(f"\n{'='*70}")
    print("COMPARISON: With vs Without body-size normalization (full_22)")
    feat_centered = kp_centered.reshape(len(kp_centered), -1)
    r_no_norm = run_pca_kmeans(feat_centered)
    print(f"  No normalization: K={r_no_norm['n_clusters']} Sil={r_no_norm['silhouette']}")
    all_results["_preprocessing_comparison"] = {
        "full_22_normalized": all_results["full_22"]["methods"]["PCA_KMeans"],
        "full_22_centered_only": {k: v for k, v in r_no_norm.items() if k not in ("labels", "features_pca")},
    }

    # Save summary
    summary = {
        "experiment": "Phase_3A_Full_Sparse_Ablation",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "preprocessing": "COM-centered + body-size normalized (nose-tail_root)",
        "frame_masking": f"Excluded indices ±{FRAME_JUMP_MARGIN} around {FRAME_JUMP_INDICES}",
        "total_frames": int(kp_raw.shape[0]),
        "frames_used": int(kp_masked.shape[0]),
        "results": all_results,
    }

    summary_path = output_dir / "full_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"\nSummary saved to {summary_path}")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Group':<16} {'Method':<16} {'K':>4} {'Sil':>8} {'CH':>10} {'DB':>8} {'Bout(s)':>8}")
    print("-" * 72)
    for gname, gdata in all_results.items():
        if gname.startswith("_"):
            continue
        for mname, mdata in gdata.get("methods", {}).items():
            sil = mdata.get("silhouette", "err")
            ch = mdata.get("calinski_harabasz", "")
            db = mdata.get("davies_bouldin", "")
            bout = mdata.get("bout_mean_sec", "")
            k = mdata.get("n_clusters", "")
            print(f"{gname:<16} {mname:<16} {k:>4} {sil:>8} {ch:>10} {db:>8} {bout:>8}")


if __name__ == "__main__":
    main()
