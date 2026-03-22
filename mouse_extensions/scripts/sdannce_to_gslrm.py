#!/usr/bin/env python3
"""
DANNCE → GS-LRM format converter for s-DANNCE multi-view data.

Converts s-DANNCE session data (6-cam video + calibration + keypoints)
into GS-LRM-compatible format (512x512 RGBA PNG + opencv_cameras.json).

Usage:
    python -m mouse_extensions.scripts.sdannce_to_gslrm \
        --session_dir /path/to/2022_09_22_M3_M4 \
        --output_dir outputs/sdannce_smoke_test/gslrm_format \
        --animal_id 1 \
        --frame_indices 0 1000 5000 10000 20000

Requirements:
    conda activate facelift  (or any env with cv2, scipy, numpy)
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio


# === GS-LRM target parameters (must match pretrained model) ===
TARGET_IMG_SIZE = 512
TARGET_FX = 549.0
TARGET_CY = TARGET_CX = TARGET_IMG_SIZE / 2.0  # 256.0
TARGET_CAM_DIST = 2.7  # Normalized camera distance


def load_dannce_calibration(cal_dir: str, num_cams: int = 6) -> list[dict]:
    """Load DANNCE camera calibration and convert to OpenCV format."""
    cameras = []
    for i in range(1, num_cams + 1):
        cal = sio.loadmat(
            os.path.join(cal_dir, f"hires_cam{i}_params.mat"), squeeze_me=True
        )
        K_opencv = cal["K"].T  # MATLAB transposed → OpenCV
        R = cal["r"]  # 3x3 rotation matrix
        t = cal["t"]  # translation (mm)
        rdist = cal["RDistort"]
        tdist = cal["TDistort"]
        dist = np.array([rdist[0], rdist[1], float(tdist[0]), float(tdist[1]), 0.0])

        cameras.append({"K": K_opencv, "R": R, "t": t, "dist": dist, "cam_id": i})
    return cameras


def load_keypoints(session_dir: str, animal_id: int = 1) -> np.ndarray:
    """Load 3D keypoints from s-DANNCE predictions.

    Returns: (N, 3, 23) array of 3D keypoints in mm.
    """
    # Try standard SDANNCE output paths (lone + social)
    patterns = [
        "SDANNCE/bsl0.5_FM/save_data_AVG0.mat",  # lone session
        "SDANNCE/bsl0.5_FM/save_data_AVG.mat",  # lone session (alt)
        f"SDANNCE/bsl0.5_FM_rat{animal_id}/save_data_AVG0.mat",  # social
        f"SDANNCE/predict01/save_data_AVG.mat",  # legacy
    ]
    for pat in patterns:
        path = os.path.join(session_dir, pat)
        if os.path.exists(path):
            data = sio.loadmat(path, squeeze_me=True)
            kp = data["pred"]  # (N, 3, 23)
            print(f"Loaded keypoints from {pat}: shape={kp.shape}")
            return kp

    raise FileNotFoundError(f"No keypoint file found in {session_dir}/SDANNCE/")


def load_com(session_dir: str, animal_id: int = 1) -> np.ndarray:
    """Load Center-of-Mass from s-DANNCE COM predictions.

    Returns: (N, 3) array of COM positions in mm.
    """
    # Try lone session path first, then social pair path
    com_paths = [
        os.path.join(session_dir, "COM/predict00/com3d.mat"),  # lone
        os.path.join(session_dir, f"COM/predict01/instance{animal_id - 1}com3d.mat"),  # social
    ]
    com_path = None
    for cp in com_paths:
        if os.path.exists(cp):
            com_path = cp
            break
    if com_path is not None:
        data = sio.loadmat(com_path, squeeze_me=True)
        com = data["com"]  # (N, 3)
        print(f"Loaded COM from {com_path}: shape={com.shape}")
        return com

    # Fallback: compute COM from keypoints
    print("COM file not found, computing from keypoints mean")
    return None


def compute_crop_params(
    kp_3d: np.ndarray,
    camera: dict,
    img_w: int = 1920,
    img_h: int = 1200,
    padding_ratio: float = 0.3,
) -> tuple[int, int, int]:
    """Compute crop center and size from projected keypoints.

    Returns: (cx, cy, crop_size) in pixel coordinates.
    """
    K = camera["K"]
    R = camera["R"]
    t = camera["t"]
    dist = camera["dist"]

    rvec, _ = cv2.Rodrigues(R)
    pts_2d, _ = cv2.projectPoints(kp_3d.T, rvec, t, K, dist)
    pts_2d = pts_2d.reshape(-1, 2)

    # Filter to in-frame points
    valid = (
        (pts_2d[:, 0] >= 0)
        & (pts_2d[:, 0] < img_w)
        & (pts_2d[:, 1] >= 0)
        & (pts_2d[:, 1] < img_h)
    )
    if valid.sum() < 3:
        return img_w // 2, img_h // 2, min(img_w, img_h)

    pts_valid = pts_2d[valid]
    cx = int(np.mean(pts_valid[:, 0]))
    cy = int(np.mean(pts_valid[:, 1]))

    # Crop size: bounding box + padding
    x_range = pts_valid[:, 0].max() - pts_valid[:, 0].min()
    y_range = pts_valid[:, 1].max() - pts_valid[:, 1].min()
    crop_size = int(max(x_range, y_range) * (1 + padding_ratio))
    crop_size = max(crop_size, 200)  # Minimum crop size

    return cx, cy, crop_size


def extract_padded_frame(
    video_path: str,
    frame_idx: int,
    camera: dict,
    mask_img: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract frame, zero-pad to square, resize to 512x512.

    Strategy: Pad 1920x1200 → 1920x1920 (white), then resize to 512x512.
    - No information loss (no cropping)
    - fx ≈ 605 (close to GS-LRM training fx=549, ratio 1.1×)
    - cx ≈ 256, cy ≈ 256 (naturally centered by symmetric padding)

    Args:
        video_path: Path to camera video file.
        frame_idx: Frame index to extract.
        camera: Camera calibration dict with K, R, t, dist.
        mask_img: Optional pre-computed mask (1200×1920, bool/uint8).
            If None, a full-foreground mask is used.

    Returns: (rgba_img, mask_resized, intrinsics, pad_info)
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

    h, w = frame.shape[:2]  # 1200, 1920

    # Undistort
    K = camera["K"]
    dist = camera["dist"]
    frame = cv2.undistort(frame, K, dist)

    # --- Zero-pad to square (w × w) ---
    # Pad top and bottom symmetrically with white
    pad_top = (w - h) // 2        # (1920-1200)//2 = 360
    pad_bottom = w - h - pad_top  # 360
    frame_padded = cv2.copyMakeBorder(
        frame, pad_top, pad_bottom, 0, 0,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    # frame_padded is now w × w (1920 × 1920)

    # Pad mask similarly (with 0 = background for padded area)
    if mask_img is not None:
        if mask_img.dtype == bool:
            mask_img = (mask_img * 255).astype(np.uint8)
        mask_padded = cv2.copyMakeBorder(
            mask_img, pad_top, pad_bottom, 0, 0,
            cv2.BORDER_CONSTANT, value=0
        )
    else:
        # Full-frame foreground (no mask provided)
        mask_padded = np.zeros((w, w), dtype=np.uint8)
        mask_padded[pad_top:pad_top + h, :] = 255

    # --- Resize to 512×512 ---
    scale = TARGET_IMG_SIZE / w  # 512 / 1920 = 0.2667
    frame_resized = cv2.resize(frame_padded, (TARGET_IMG_SIZE, TARGET_IMG_SIZE),
                               interpolation=cv2.INTER_LANCZOS4)
    mask_resized = cv2.resize(mask_padded, (TARGET_IMG_SIZE, TARGET_IMG_SIZE),
                              interpolation=cv2.INTER_NEAREST)

    # --- White-background RGBA composite ---
    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    alpha_f = mask_resized.astype(np.float32) / 255.0
    bg_white = np.ones_like(rgb) * 255
    rgb_composite = (rgb * alpha_f[..., None] + bg_white * (1 - alpha_f[..., None])).astype(
        np.uint8
    )
    rgba = np.dstack([rgb_composite, mask_resized])

    # --- Compute intrinsics for padded+resized image ---
    # After padding: cy_padded = cy_orig + pad_top
    # After resize: all values × scale
    fx_new = K[0, 0] * scale
    fy_new = K[1, 1] * scale
    cx_new = K[0, 2] * scale
    cy_new = (K[1, 2] + pad_top) * scale

    intrinsics = {
        "fx": float(fx_new),
        "fy": float(fy_new),
        "cx": float(cx_new),
        "cy": float(cy_new),
    }

    return rgba, mask_resized, intrinsics, (pad_top, pad_bottom, w)


def build_w2c_matrix(camera: dict) -> np.ndarray:
    """Build 4x4 world-to-camera matrix."""
    R = camera["R"]
    t = camera["t"]
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c


def normalize_cameras(
    cameras_w2c: list[np.ndarray],
    intrinsics_list: list[dict],
    target_dist: float = TARGET_CAM_DIST,
) -> tuple[list[np.ndarray], list[dict]]:
    """Normalize cameras to GS-LRM conventions.

    Only normalizes spatial scale (camera distance → target_dist).
    Intrinsics (fx, fy, cx, cy) are preserved from PP-centering step.
    cx=cy=256 is guaranteed by extract_and_crop_frame.
    """
    # Compute current camera distances
    distances = []
    for w2c in cameras_w2c:
        c2w = np.linalg.inv(w2c)
        cam_pos = c2w[:3, 3]
        distances.append(np.linalg.norm(cam_pos))

    mean_dist = np.mean(distances)
    spatial_scale = target_dist / mean_dist  # mm → normalized units

    norm_w2c_list = []
    norm_intr_list = []

    for w2c, intr in zip(cameras_w2c, intrinsics_list):
        # Scale translation only
        w2c_norm = w2c.copy()
        w2c_norm[:3, 3] *= spatial_scale

        # Keep intrinsics as-is (cx=cy=256 from PP-centering)
        norm_w2c_list.append(w2c_norm)
        norm_intr_list.append(intr)

    print(f"Camera normalization: dist {mean_dist:.1f}mm → {target_dist:.3f}, scale={spatial_scale:.6f}")
    print(f"  Intrinsics preserved: fx={intrinsics_list[0]['fx']:.1f}, cx={intrinsics_list[0]['cx']:.1f}")
    return norm_w2c_list, norm_intr_list


def save_gslrm_frame(
    output_dir: str,
    frame_id: str,
    rgba_images: list[np.ndarray],
    w2c_matrices: list[np.ndarray],
    intrinsics_list: list[dict],
):
    """Save one frame in GS-LRM format."""
    frame_dir = os.path.join(output_dir, frame_id)
    img_dir = os.path.join(frame_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    frames_data = []
    for i, (rgba, w2c, intr) in enumerate(zip(rgba_images, w2c_matrices, intrinsics_list)):
        # Save RGBA PNG
        img_path = f"images/cam_{i:03d}.png"
        # Convert RGB→BGR for cv2, then add alpha
        bgra = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        bgra = np.dstack([bgra, rgba[:, :, 3]])
        cv2.imwrite(os.path.join(frame_dir, img_path), bgra)

        frames_data.append(
            {
                "w": TARGET_IMG_SIZE,
                "h": TARGET_IMG_SIZE,
                "fx": intr["fx"],
                "fy": intr["fy"],
                "cx": intr["cx"],
                "cy": intr["cy"],
                "w2c": w2c.tolist(),
                "file_path": img_path,
                "view_id": i,
            }
        )

    # Save camera JSON
    with open(os.path.join(frame_dir, "opencv_cameras.json"), "w") as f:
        json.dump({"frames": frames_data}, f, indent=2)


def load_sam2_masks(
    ann_dir: str,
    propagated_dir: str = None,
    animal_id: int = 0,
    merge_animals: bool = True,
) -> dict[int, np.ndarray]:
    """Load SAM2 annotation/propagated masks.

    Args:
        ann_dir: Directory with ann_frame_*.npz files (rat1/rat2 bool masks).
        propagated_dir: Directory with mask_frame_*.npz files from SAM2 propagation.
        animal_id: 0=merge both animals (union), 1=rat1 only, 2=rat2 only.
        merge_animals: If True and animal_id=0, use union of rat1+rat2.

    Returns: Dict mapping frame_idx → mask (H, W) uint8.
    """
    masks = {}

    # Load manual annotations first
    if ann_dir and os.path.exists(ann_dir):
        for f in sorted(os.listdir(ann_dir)):
            if f.startswith("ann_frame_") and f.endswith(".npz"):
                fi = int(f.split("_")[-1].split(".")[0])
                data = np.load(os.path.join(ann_dir, f))
                if merge_animals or animal_id == 0:
                    mask = data.get("rat1", np.zeros((1, 1), dtype=bool))
                    if "rat2" in data:
                        mask = mask | data["rat2"]
                elif animal_id == 1:
                    mask = data.get("rat1", np.zeros((1, 1), dtype=bool))
                else:
                    mask = data.get("rat2", np.zeros((1, 1), dtype=bool))
                masks[fi] = (mask * 255).astype(np.uint8)

    # Load propagated masks (override annotations if both exist)
    if propagated_dir and os.path.exists(propagated_dir):
        mask_dir = os.path.join(propagated_dir, "masks")
        if os.path.exists(mask_dir):
            for f in sorted(os.listdir(mask_dir)):
                if f.startswith("mask_frame_") and f.endswith(".npz"):
                    fi = int(f.split("_")[-1].split(".")[0])
                    data = np.load(os.path.join(mask_dir, f))
                    # Propagated masks may have different structure
                    if "rat1" in data:
                        if merge_animals or animal_id == 0:
                            mask = data["rat1"]
                            if "rat2" in data:
                                mask = mask | data["rat2"]
                        elif animal_id == 1:
                            mask = data["rat1"]
                        else:
                            mask = data.get("rat2", np.zeros((1, 1), dtype=bool))
                    else:
                        # Single mask
                        mask = list(data.values())[0]
                    masks[fi] = (mask * 255).astype(np.uint8) if mask.dtype == bool else mask

    if masks:
        print(f"Loaded {len(masks)} SAM2 masks (animal_id={animal_id}, merge={merge_animals})")
    return masks


def compute_background(video_paths: list[str], sample_indices: list[int] = None) -> list[np.ndarray]:
    """Compute background image per camera via median."""
    if sample_indices is None:
        sample_indices = [0, 10000, 30000, 60000, 89000]

    bg_images = []
    for vp in video_paths:
        cap = cv2.VideoCapture(vp)
        frames = []
        for fi in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                frames.append(frame.astype(np.float32))
        cap.release()
        if frames:
            bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
        else:
            bg = None
        bg_images.append(bg)
    return bg_images


def process_session(
    session_dir: str,
    output_dir: str,
    animal_id: int = 1,
    frame_indices: list[int] = None,
    num_cams: int = 6,
    sam2_ann_dir: str = None,
    sam2_prop_dir: str = None,
    merge_animals: bool = True,
):
    """Full DANNCE → GS-LRM conversion pipeline.

    Args:
        session_dir: Path to s-DANNCE session folder.
        output_dir: Output directory for GS-LRM format data.
        animal_id: Animal ID for keypoints (1 or 2).
        frame_indices: Frame indices to process.
        num_cams: Number of cameras.
        sam2_ann_dir: SAM2 annotation directory (ann_frame_*.npz).
        sam2_prop_dir: SAM2 propagation directory (mask_frame_*.npz).
        merge_animals: If True, use union of rat1+rat2 masks (recommended for social pairs).
    """
    print(f"=== DANNCE → GS-LRM Converter (v2: padding + SAM2) ===")
    print(f"Session: {session_dir}")
    print(f"Output: {output_dir}")
    print(f"Animal: {animal_id}, merge_masks: {merge_animals}")

    # Load calibration
    cal_dir = os.path.join(session_dir, "calibration")
    cameras = load_dannce_calibration(cal_dir, num_cams)

    # Load keypoints
    kp_3d = load_keypoints(session_dir, animal_id)
    total_frames = kp_3d.shape[0]

    if frame_indices is None:
        frame_indices = [0, 1000, 5000, 10000, 20000]

    print(f"Processing {len(frame_indices)} frames: {frame_indices}")

    # Video paths
    video_paths = [
        os.path.join(session_dir, f"videos/Camera{i}/0.mp4") for i in range(1, num_cams + 1)
    ]

    # Load SAM2 masks (Camera1 only for now — multi-cam SAM2 needs annotation)
    sam2_masks = None
    mask_animal = 0 if merge_animals else animal_id
    if sam2_ann_dir or sam2_prop_dir:
        sam2_masks = load_sam2_masks(sam2_ann_dir, sam2_prop_dir, mask_animal, merge_animals)
    else:
        print("No SAM2 masks provided — using full-frame foreground")

    # Process each frame
    for fi in frame_indices:
        if fi >= total_frames:
            print(f"Frame {fi} out of range ({total_frames}), skipping")
            continue

        frame_id = f"{fi:06d}"
        print(f"\n--- Frame {fi} ---")

        kp = kp_3d[fi]  # (3, 23)

        # Extract frames with zero-padding (no crop, no information loss)
        rgba_images = []
        w2c_matrices = []
        intrinsics_list = []

        for cam_idx, camera in enumerate(cameras):
            video_path = video_paths[cam_idx]

            # Load SAM2 mask if available (cam_idx 0 = Camera1 only for now)
            mask_img = None
            if sam2_masks is not None and cam_idx == 0:
                mask_img = sam2_masks.get(fi)

            try:
                rgba, mask, intr, pad_info = extract_padded_frame(
                    video_path, fi, camera, mask_img
                )
            except RuntimeError as e:
                print(f"  Camera {cam_idx + 1}: {e}")
                continue

            mask_pct = np.mean(mask > 0) * 100
            print(f"  Camera {cam_idx + 1}: pad=({pad_info[0]},{pad_info[1]}), "
                  f"fx={intr['fx']:.0f}, cx={intr['cx']:.1f}, cy={intr['cy']:.1f}, mask={mask_pct:.1f}%")

            rgba_images.append(rgba)
            w2c_matrices.append(build_w2c_matrix(camera))
            intrinsics_list.append(intr)

        if len(rgba_images) < 2:
            print(f"  Too few valid views ({len(rgba_images)}), skipping frame")
            continue

        # Normalize cameras
        norm_w2c, norm_intr = normalize_cameras(w2c_matrices, intrinsics_list)

        # Save
        save_gslrm_frame(output_dir, frame_id, rgba_images, norm_w2c, norm_intr)
        print(f"  Saved {len(rgba_images)} views → {output_dir}/{frame_id}/")

    # Write frame list
    list_path = os.path.join(output_dir, "data_sdannce_test.txt")
    with open(list_path, "w") as f:
        for fi in frame_indices:
            if fi < total_frames:
                f.write(f"{fi:06d}\n")
    print(f"\nFrame list: {list_path}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DANNCE → GS-LRM format converter")
    parser.add_argument(
        "--session_dir",
        type=str,
        required=True,
        help="Path to s-DANNCE session folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for GS-LRM format data",
    )
    parser.add_argument(
        "--animal_id",
        type=int,
        default=1,
        help="Animal ID (1 or 2 for social pairs)",
    )
    parser.add_argument(
        "--frame_indices",
        type=int,
        nargs="+",
        default=[0, 1000, 5000, 10000, 20000],
        help="Frame indices to process",
    )
    parser.add_argument(
        "--num_cams",
        type=int,
        default=6,
        help="Number of cameras",
    )
    parser.add_argument(
        "--sam2_ann_dir",
        type=str,
        default=None,
        help="SAM2 annotation directory (ann_frame_*.npz with rat1/rat2 masks)",
    )
    parser.add_argument(
        "--sam2_prop_dir",
        type=str,
        default=None,
        help="SAM2 propagation directory (mask_frame_*.npz)",
    )
    parser.add_argument(
        "--merge_animals",
        action="store_true",
        default=True,
        help="Merge rat1+rat2 masks (union) for social pairs",
    )

    args = parser.parse_args()
    process_session(
        session_dir=args.session_dir,
        output_dir=args.output_dir,
        animal_id=args.animal_id,
        frame_indices=args.frame_indices,
        num_cams=args.num_cams,
        sam2_ann_dir=args.sam2_ann_dir,
        sam2_prop_dir=args.sam2_prop_dir,
        merge_animals=args.merge_animals,
    )
