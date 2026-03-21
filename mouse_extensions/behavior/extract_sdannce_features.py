"""Extract sparse features (S1, S3) from s-DANNCE keypoint predictions.

Reads .mat files from s-DANNCE output (pred shape: T, 3, 23) and produces:
  - S1: Raw skeleton pose (COM-centered, per-animal and dyadic)
  - S3: Engineered kinematics (velocities, distances, angles, body length)

Usage:
    python -m mouse_extensions.behavior.extract_sdannce_features \
        --session /path/to/session_dir \
        --output outputs/sdannce_poc/features/ \
        --max_frames 90000
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio


# s-DANNCE 23 keypoint names (from dataset catalog)
KP_NAMES = [
    "Snout", "EarL", "EarR", "SpineF",
    "SpineM", "SpineL", "TailBase", "ShoulderL", "ShoulderR",
    "HipL", "HipR",
    "ElbowL", "WristL", "HandL", "ElbowR", "WristR", "HandR",
    "KneeL", "AnkleL", "FootL", "KneeR", "AnkleR", "FootR",
]

# Key indices
IDX_SNOUT = 0
IDX_SPINE_M = 4
IDX_TAIL_BASE = 6
IDX_SPINE_F = 3


def load_keypoints(session_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load 3D keypoints for both animals from SDANNCE predictions.

    Returns:
        kp1: (T, 23, 3) array for rat1
        kp2: (T, 23, 3) array for rat2
    """
    sdannce_dir = session_dir / "SDANNCE"
    rat_dirs = sorted([d for d in sdannce_dir.iterdir() if d.is_dir()])

    if len(rat_dirs) < 2:
        raise ValueError(f"Expected 2 rat dirs, found {len(rat_dirs)} in {sdannce_dir}")

    kps = []
    for rd in rat_dirs[:2]:
        mat_file = rd / "save_data_AVG0.mat"
        if not mat_file.exists():
            mat_file = rd / "save_data_AVG.mat"
        m = sio.loadmat(str(mat_file), squeeze_me=False)
        pred = m["pred"]  # (T, 3, 23)
        kps.append(pred.transpose(0, 2, 1))  # -> (T, 23, 3)

    return kps[0], kps[1]


def extract_s1(kp1: np.ndarray, kp2: np.ndarray) -> dict:
    """S1: Raw skeleton pose, COM-centered.

    Returns dict with:
        s1_single_rat1: (T, 69) COM-centered rat1
        s1_single_rat2: (T, 69) COM-centered rat2
        s1_dyadic: (T, 138) both rats, scene-COM-centered
    """
    T = kp1.shape[0]

    # Per-animal COM centering
    com1 = kp1.mean(axis=1, keepdims=True)  # (T, 1, 3)
    com2 = kp2.mean(axis=1, keepdims=True)

    kp1_c = kp1 - com1
    kp2_c = kp2 - com2

    # Dyadic: center by scene COM (mean of both COMs)
    scene_com = (com1 + com2) / 2
    kp1_sc = kp1 - scene_com
    kp2_sc = kp2 - scene_com

    return {
        "s1_single_rat1": kp1_c.reshape(T, -1).astype(np.float32),
        "s1_single_rat2": kp2_c.reshape(T, -1).astype(np.float32),
        "s1_dyadic": np.concatenate([
            kp1_sc.reshape(T, -1),
            kp2_sc.reshape(T, -1),
        ], axis=1).astype(np.float32),
    }


def extract_s3(kp1: np.ndarray, kp2: np.ndarray, fps: float = 50.0) -> dict:
    """S3: Engineered kinematics (12d minimal PoC).

    Features (per frame):
        0: nose_vel_rat1
        1: nose_vel_rat2
        2: spine_vel_rat1
        3: spine_vel_rat2
        4: body_length_rat1 (nose-tail_base distance)
        5: body_length_rat2
        6: inter_nose_distance
        7: inter_centroid_distance
        8: relative_heading_angle (radians)
        9: z_height_rat1 (spine_mid Z)
        10: z_height_rat2
        11: nose_to_tail_proximity (min of cross-animal)
    """
    T = kp1.shape[0]
    feats = np.zeros((T, 12), dtype=np.float32)

    # Velocities (mm/s) — use central difference, pad edges
    def velocity(kp_seq):
        """Compute speed of a single keypoint sequence (T, 3) -> (T,)."""
        diff = np.zeros_like(kp_seq)
        diff[1:-1] = (kp_seq[2:] - kp_seq[:-2]) / 2.0
        diff[0] = kp_seq[1] - kp_seq[0]
        diff[-1] = kp_seq[-1] - kp_seq[-2]
        return np.linalg.norm(diff, axis=1) * fps

    feats[:, 0] = velocity(kp1[:, IDX_SNOUT])
    feats[:, 1] = velocity(kp2[:, IDX_SNOUT])
    feats[:, 2] = velocity(kp1[:, IDX_SPINE_M])
    feats[:, 3] = velocity(kp2[:, IDX_SPINE_M])

    # Body length
    feats[:, 4] = np.linalg.norm(kp1[:, IDX_SNOUT] - kp1[:, IDX_TAIL_BASE], axis=1)
    feats[:, 5] = np.linalg.norm(kp2[:, IDX_SNOUT] - kp2[:, IDX_TAIL_BASE], axis=1)

    # Inter-animal distances
    feats[:, 6] = np.linalg.norm(kp1[:, IDX_SNOUT] - kp2[:, IDX_SNOUT], axis=1)

    com1 = kp1.mean(axis=1)  # (T, 3)
    com2 = kp2.mean(axis=1)
    feats[:, 7] = np.linalg.norm(com1 - com2, axis=1)

    # Relative heading angle (body axis = nose -> tail_base)
    axis1 = kp1[:, IDX_SNOUT] - kp1[:, IDX_TAIL_BASE]  # (T, 3)
    axis2 = kp2[:, IDX_SNOUT] - kp2[:, IDX_TAIL_BASE]
    cos_angle = np.sum(axis1 * axis2, axis=1) / (
        np.linalg.norm(axis1, axis=1) * np.linalg.norm(axis2, axis=1) + 1e-8
    )
    feats[:, 8] = np.arccos(np.clip(cos_angle, -1, 1))

    # Z-height (rearing indicator)
    feats[:, 9] = kp1[:, IDX_SPINE_M, 2]
    feats[:, 10] = kp2[:, IDX_SPINE_M, 2]

    # Nose-to-tail proximity (social sniffing indicator)
    d_nose1_tail2 = np.linalg.norm(kp1[:, IDX_SNOUT] - kp2[:, IDX_TAIL_BASE], axis=1)
    d_nose2_tail1 = np.linalg.norm(kp2[:, IDX_SNOUT] - kp1[:, IDX_TAIL_BASE], axis=1)
    feats[:, 11] = np.minimum(d_nose1_tail2, d_nose2_tail1)

    return {
        "s3_engineered": feats,
        "s3_feature_names": [
            "nose_vel_r1", "nose_vel_r2", "spine_vel_r1", "spine_vel_r2",
            "body_len_r1", "body_len_r2", "inter_nose_dist", "inter_com_dist",
            "heading_angle", "z_height_r1", "z_height_r2", "nose_tail_prox",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Extract sparse features from s-DANNCE")
    parser.add_argument("--session", required=True, help="Path to session directory")
    parser.add_argument("--output", default="outputs/sdannce_poc/features/", help="Output dir")
    parser.add_argument("--max_frames", type=int, default=90000, help="Max frames to process")
    parser.add_argument("--fps", type=float, default=50.0, help="Framerate")
    args = parser.parse_args()

    session_dir = Path(args.session)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading keypoints from {session_dir}...")
    kp1, kp2 = load_keypoints(session_dir)

    # Truncate if needed
    T = min(kp1.shape[0], args.max_frames)
    kp1 = kp1[:T]
    kp2 = kp2[:T]
    print(f"Frames: {T}, Keypoints: {kp1.shape[1]}, Animals: 2")

    # Extract S1
    print("Extracting S1 (Skeleton Pose)...")
    s1 = extract_s1(kp1, kp2)
    for k, v in s1.items():
        print(f"  {k}: shape={v.shape}")

    # Extract S3
    print("Extracting S3 (Engineered Kinematics)...")
    s3 = extract_s3(kp1, kp2, fps=args.fps)
    print(f"  s3_engineered: shape={s3['s3_engineered'].shape}")
    print(f"  features: {s3['s3_feature_names']}")

    # Save
    out_path = output_dir / "sparse_features.npz"
    np.savez_compressed(
        str(out_path),
        **s1,
        **{k: v for k, v in s3.items() if isinstance(v, np.ndarray)},
        s3_feature_names=s3["s3_feature_names"],
        kp1_raw=kp1.astype(np.float32),
        kp2_raw=kp2.astype(np.float32),
        fps=args.fps,
        n_frames=T,
    )
    print(f"\nSaved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Quick stats
    s3f = s3["s3_engineered"]
    print(f"\n--- Feature Statistics ---")
    for i, name in enumerate(s3["s3_feature_names"]):
        print(f"  {name:20s}: mean={s3f[:,i].mean():8.2f}  std={s3f[:,i].std():8.2f}  "
              f"min={s3f[:,i].min():8.2f}  max={s3f[:,i].max():8.2f}")


if __name__ == "__main__":
    main()
