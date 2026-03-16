"""Convert DANNCE 2D keypoint labels to COCO format for MMPose training.

Reads DANNCE 2D detection results (pkl) and undistorted videos to produce
a COCO-format dataset suitable for MMPose fine-tuning.

Usage:
    conda activate mmpose
    python convert_dannce_to_coco.py \
        --data_dir ~/data/raw/markerless_mouse_1_nerf \
        --output_dir ~/data/processed/mmpose_mouse

    # Verify + visualize
    python convert_dannce_to_coco.py \
        --data_dir ~/data/raw/markerless_mouse_1_nerf \
        --output_dir ~/data/processed/mmpose_mouse \
        --verify --visualize
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# MAMMAL 22-keypoint definition (canonical order matching DANNCE 2D pkl)
# ---------------------------------------------------------------------------
KEYPOINT_NAMES = [
    "L_ear", "R_ear", "nose", "neck", "body_middle",
    "tail_root", "tail_middle", "tail_end",
    "L_paw", "L_paw_end", "L_elbow", "L_shoulder",
    "R_paw", "R_paw_end", "R_elbow", "R_shoulder",
    "L_foot", "L_knee", "L_hip",
    "R_foot", "R_knee", "R_hip",
]

# Skeleton: pairs of connected joint indices
SKELETON = [
    [2, 0],   # nose - L_ear
    [2, 1],   # nose - R_ear
    [2, 3],   # nose - neck
    [3, 4],   # neck - body_middle
    [4, 5],   # body_middle - tail_root
    [5, 6],   # tail_root - tail_middle
    [6, 7],   # tail_middle - tail_end
    [3, 11],  # neck - L_shoulder
    [11, 10], # L_shoulder - L_elbow
    [10, 8],  # L_elbow - L_paw
    [8, 9],   # L_paw - L_paw_end
    [3, 15],  # neck - R_shoulder
    [15, 14], # R_shoulder - R_elbow
    [14, 12], # R_elbow - R_paw
    [12, 13], # R_paw - R_paw_end
    [5, 18],  # tail_root - L_hip
    [18, 17], # L_hip - L_knee
    [17, 16], # L_knee - L_foot
    [5, 21],  # tail_root - R_hip
    [21, 20], # R_hip - R_knee
    [20, 19], # R_knee - R_foot
]

# Flip pairs for augmentation (L/R symmetry)
FLIP_PAIRS = [
    (0, 1),    # L_ear <-> R_ear
    (8, 12),   # L_paw <-> R_paw
    (9, 13),   # L_paw_end <-> R_paw_end
    (10, 14),  # L_elbow <-> R_elbow
    (11, 15),  # L_shoulder <-> R_shoulder
    (16, 19),  # L_foot <-> R_foot
    (17, 20),  # L_knee <-> R_knee
    (18, 21),  # L_hip <-> R_hip
]


def load_dannce_2d(data_dir: str, n_views: int = 6) -> np.ndarray:
    """Load DANNCE 2D keypoint predictions for all views.

    Returns:
        (n_views, n_frames, n_joints, 3) array with [x, y, confidence]
    """
    kp_dir = os.path.join(data_dir, "keypoints2d_undist")
    all_views = []
    for view_idx in range(n_views):
        pkl_path = os.path.join(kp_dir, f"result_view_{view_idx}.pkl")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        print(f"  View {view_idx}: shape={data.shape}, dtype={data.dtype}")
        all_views.append(data)
    return np.stack(all_views)


def get_m5_splits(n_m5_frames: int = 3600) -> dict:
    """M5t2 canonical train/val/test split (80/10/10)."""
    return {
        "train": list(range(0, 2880)),
        "val": list(range(2880, 3240)),
        "test": list(range(3240, n_m5_frames)),
    }


def compute_bbox(
    keypoints: np.ndarray,
    conf_threshold: float = 0.1,
    padding: float = 0.2,
    img_w: int = 0,
    img_h: int = 0,
) -> list:
    """Compute bounding box from visible keypoints with padding.

    Args:
        keypoints: (n_joints, 3) with [x, y, conf]
        padding: fractional padding around tight bbox
    Returns:
        [x, y, w, h] in COCO format
    """
    valid = keypoints[:, 2] > conf_threshold
    if valid.sum() < 2:
        valid = np.ones(len(keypoints), dtype=bool)

    xs = keypoints[valid, 0]
    ys = keypoints[valid, 1]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    w = x_max - x_min
    h = y_max - y_min

    # Ensure minimum bbox size
    w = max(w, 10.0)
    h = max(h, 10.0)

    x_min -= w * padding
    y_min -= h * padding
    w *= 1 + 2 * padding
    h *= 1 + 2 * padding

    if img_w > 0 and img_h > 0:
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        w = min(w, img_w - x_min)
        h = min(h, img_h - y_min)

    return [float(x_min), float(y_min), float(w), float(h)]


def build_coco_dataset(
    data_dir: str,
    output_dir: str,
    frame_step: int = 5,
    conf_threshold: float = 0.1,
    n_views: int = 6,
    save_images: bool = True,
):
    """Build COCO-format dataset from DANNCE 2D labels and undistorted videos.

    Reads videos sequentially (efficient for H.264) and saves every
    `frame_step`-th frame as JPEG with COCO annotations.

    Args:
        data_dir: Path to markerless_mouse_1_nerf/
        output_dir: Output directory for COCO dataset
        frame_step: Ratio between video fps and M5 fps (100/20 = 5)
        conf_threshold: Min confidence for COCO visibility=2
        n_views: Number of camera views (6)
        save_images: If True, extract and save video frames
    """
    print("Loading DANNCE 2D keypoints...")
    kp_2d = load_dannce_2d(data_dir, n_views)  # (6, 18000, 22, 3)
    n_total_frames = kp_2d.shape[1]
    n_m5_frames = n_total_frames // frame_step
    print(f"Total DANNCE frames: {n_total_frames}, M5 frames: {n_m5_frames}")

    splits = get_m5_splits(n_m5_frames)

    # Pre-create output directories
    ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    for split_name in splits:
        os.makedirs(os.path.join(output_dir, "images", split_name), exist_ok=True)

    # Open video captures
    video_dir = os.path.join(data_dir, "videos_undist")
    caps = {}
    video_w, video_h = None, None

    if save_images:
        for view_idx in range(n_views):
            video_path = os.path.join(video_dir, f"{view_idx}.mp4")
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")
            caps[view_idx] = cap
            if video_w is None:
                video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video resolution: {video_w}x{video_h}")

    # Build M5 frame -> split mapping
    m5_to_split = {}
    for split_name, indices in splits.items():
        for idx in indices:
            m5_to_split[idx] = split_name

    # Accumulators per split
    split_data = {
        s: {"images": [], "annotations": [], "img_id": 0, "ann_id": 0}
        for s in splits
    }

    # Sequential video reading (all views simultaneously)
    print("\nExtracting frames and building annotations...")
    for view_idx in range(n_views):
        print(f"\n  View {view_idx}/{n_views - 1}")

        if save_images:
            cap = caps[view_idx]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start

        m5_idx = 0
        for dannce_idx in tqdm(range(n_total_frames), desc=f"  cam{view_idx}"):
            if dannce_idx % frame_step != 0:
                if save_images:
                    cap.read()  # Skip frame
                continue

            if m5_idx >= n_m5_frames:
                break

            split_name = m5_to_split.get(m5_idx)
            if split_name is None:
                m5_idx += 1
                if save_images:
                    cap.read()
                continue

            sd = split_data[split_name]
            kp = kp_2d[view_idx, dannce_idx]  # (22, 3)

            img_filename = f"{split_name}/{m5_idx:06d}_cam{view_idx}.jpg"

            # Extract frame
            w, h = video_w or 1152, video_h or 1024
            if save_images:
                ret, frame = cap.read()
                if not ret:
                    print(f"    Warning: Cannot read frame {dannce_idx}")
                    m5_idx += 1
                    continue
                save_path = os.path.join(output_dir, "images", img_filename)
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                h, w = frame.shape[:2]

            # COCO image entry
            img_id = sd["img_id"]
            sd["images"].append({
                "id": img_id,
                "file_name": img_filename,
                "width": w,
                "height": h,
            })

            # COCO keypoint annotation: [x1, y1, v1, x2, y2, v2, ...]
            coco_kps = []
            num_visible = 0
            for j in range(22):
                x, y, conf = kp[j]
                if conf > conf_threshold:
                    v = 2  # Labeled and visible
                    num_visible += 1
                else:
                    v = 1  # Labeled but occluded
                coco_kps.extend([float(x), float(y), int(v)])

            bbox = compute_bbox(kp, conf_threshold, padding=0.2, img_w=w, img_h=h)

            ann_id = sd["ann_id"]
            sd["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "keypoints": coco_kps,
                "num_keypoints": num_visible,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            })

            sd["img_id"] += 1
            sd["ann_id"] += 1
            m5_idx += 1

    # COCO category definition
    categories = [{
        "id": 1,
        "name": "mouse",
        "supercategory": "animal",
        "keypoints": KEYPOINT_NAMES,
        "skeleton": SKELETON,
    }]

    # Save COCO JSON per split
    for split_name, sd in split_data.items():
        coco_json = {
            "images": sd["images"],
            "annotations": sd["annotations"],
            "categories": categories,
        }
        ann_path = os.path.join(ann_dir, f"{split_name}.json")
        with open(ann_path, "w") as f:
            json.dump(coco_json, f)
        n_imgs = len(sd["images"])
        n_anns = len(sd["annotations"])
        print(f"\n  {split_name}: {n_imgs} images, {n_anns} annotations -> {ann_path}")

    # Save keypoint info
    kp_info = {
        "keypoint_names": KEYPOINT_NAMES,
        "skeleton": SKELETON,
        "flip_pairs": FLIP_PAIRS,
        "num_keypoints": 22,
    }
    kp_info_path = os.path.join(output_dir, "mouse_keypoint_info.json")
    with open(kp_info_path, "w") as f:
        json.dump(kp_info, f, indent=2)

    for cap in caps.values():
        cap.release()

    print(f"\nDataset saved to {output_dir}")
    return output_dir


def verify_coco_dataset(output_dir: str):
    """Validate COCO JSON with pycocotools."""
    from pycocotools.coco import COCO

    ann_dir = os.path.join(output_dir, "annotations")
    for split in ["train", "val", "test"]:
        ann_path = os.path.join(ann_dir, f"{split}.json")
        if not os.path.exists(ann_path):
            continue

        coco = COCO(ann_path)
        n_imgs = len(coco.imgs)
        n_anns = len(coco.anns)

        # Validate samples
        errors = 0
        img_ids = list(coco.imgs.keys())[:50]
        for img_id in img_ids:
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                kps = np.array(ann["keypoints"]).reshape(-1, 3)
                n_vis = int((kps[:, 2] == 2).sum())
                if ann["num_keypoints"] != n_vis:
                    errors += 1
                if ann["area"] <= 0:
                    errors += 1

        status = "PASS" if errors == 0 else f"FAIL ({errors} errors)"
        print(f"  {split}: {n_imgs} imgs, {n_anns} anns -> {status}")


def visualize_samples(output_dir: str, n_samples: int = 10, split: str = "train"):
    """Draw keypoints and skeleton on sample images for visual verification."""
    from pycocotools.coco import COCO

    save_dir = os.path.join(output_dir, "verification")
    os.makedirs(save_dir, exist_ok=True)

    ann_path = os.path.join(output_dir, "annotations", f"{split}.json")
    coco = COCO(ann_path)

    # Deterministic sample selection
    img_ids = sorted(coco.imgs.keys())
    rng = np.random.RandomState(42)
    sample_ids = rng.choice(img_ids, min(n_samples, len(img_ids)), replace=False)

    # Color palette
    cmap = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (200, 100, 0), (100, 200, 0), (0, 100, 200), (200, 0, 100),
        (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0),
        (64, 0, 64), (0, 64, 64),
    ]

    for img_id in sample_ids:
        img_info = coco.imgs[img_id]
        img_path = os.path.join(output_dir, "images", img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: Cannot read {img_path}")
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            kps = np.array(ann["keypoints"]).reshape(-1, 3)

            # Draw skeleton
            for j1, j2 in SKELETON:
                if kps[j1, 2] > 0 and kps[j2, 2] > 0:
                    pt1 = (int(kps[j1, 0]), int(kps[j1, 1]))
                    pt2 = (int(kps[j2, 0]), int(kps[j2, 1]))
                    cv2.line(img, pt1, pt2, (180, 180, 180), 1)

            # Draw keypoints with labels
            for j in range(22):
                if kps[j, 2] > 0:
                    x, y = int(kps[j, 0]), int(kps[j, 1])
                    color = cmap[j % len(cmap)]
                    cv2.circle(img, (x, y), 4, color, -1)
                    cv2.putText(
                        img, KEYPOINT_NAMES[j][:4], (x + 5, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1,
                    )

            # Draw bbox
            bx, by, bw, bh = ann["bbox"]
            cv2.rectangle(
                img, (int(bx), int(by)), (int(bx + bw), int(by + bh)),
                (0, 255, 0), 2,
            )

        fname = img_info["file_name"].replace("/", "_")
        cv2.imwrite(os.path.join(save_dir, f"verify_{fname}"), img)

    print(f"  Saved {len(sample_ids)} verification images to {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DANNCE 2D keypoint labels to COCO format"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=os.path.expanduser("~/data/raw/markerless_mouse_1_nerf"),
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.expanduser("~/data/processed/mmpose_mouse"),
    )
    parser.add_argument("--frame_step", type=int, default=5)
    parser.add_argument("--conf_threshold", type=float, default=0.1)
    parser.add_argument("--no_images", action="store_true",
                        help="Skip image extraction (annotations only)")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--n_vis_samples", type=int, default=10)
    args = parser.parse_args()

    build_coco_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        frame_step=args.frame_step,
        conf_threshold=args.conf_threshold,
        save_images=not args.no_images,
    )

    if args.verify:
        print("\nVerifying COCO dataset...")
        verify_coco_dataset(args.output_dir)

    if args.visualize:
        print("\nVisualizing samples...")
        visualize_samples(args.output_dir, n_samples=args.n_vis_samples)


if __name__ == "__main__":
    main()
