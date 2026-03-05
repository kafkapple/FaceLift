"""Visualize keypoint detections on rendered images.

Creates side-by-side visualizations of rendered images with detected
keypoints overlaid, for quality inspection and debugging.

Usage:
    python visualize_detections.py \
        --render_dir ~/outputs/neural_triangulation/renders/12views \
        --detection_dir ~/outputs/neural_triangulation/results \
        --output_dir ~/outputs/neural_triangulation/comparison/plots
"""

import argparse
import json
import os

import cv2
import numpy as np

JOINT_NAMES = [
    "L_ear", "R_ear", "nose", "neck", "body_middle",
    "tail_root", "tail_middle", "tail_end",
    "L_paw", "L_paw_end", "L_elbow", "L_shoulder",
    "R_paw", "R_paw_end", "R_elbow", "R_shoulder",
    "L_foot", "L_knee", "L_hip",
    "R_foot", "R_knee", "R_hip",
]

SKELETON = [
    [2, 0], [2, 1], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
    [3, 11], [11, 10], [10, 8], [8, 9],
    [3, 15], [15, 14], [14, 12], [12, 13],
    [5, 18], [18, 17], [17, 16],
    [5, 21], [21, 20], [20, 19],
]

# Color palette per joint
JOINT_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (200, 100, 0), (100, 200, 0), (0, 100, 200), (200, 0, 100),
    (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0),
    (64, 0, 64), (0, 64, 64),
]


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    conf_threshold: float = 0.3,
    draw_skeleton: bool = True,
    radius: int = 4,
    thickness: int = 1,
):
    """Draw keypoints and skeleton on image.

    Args:
        image: BGR image (will be modified in-place)
        keypoints: (22, 3) with [x, y, confidence]
        conf_threshold: Min confidence to draw
    """
    vis = image.copy()

    # Draw skeleton first (behind keypoints)
    if draw_skeleton:
        for j1, j2 in SKELETON:
            if keypoints[j1, 2] > conf_threshold and keypoints[j2, 2] > conf_threshold:
                pt1 = (int(keypoints[j1, 0]), int(keypoints[j1, 1]))
                pt2 = (int(keypoints[j2, 0]), int(keypoints[j2, 1]))
                cv2.line(vis, pt1, pt2, (200, 200, 200), thickness)

    # Draw keypoints
    for j in range(keypoints.shape[0]):
        if keypoints[j, 2] > conf_threshold:
            x, y = int(keypoints[j, 0]), int(keypoints[j, 1])
            conf = keypoints[j, 2]
            color = JOINT_COLORS[j % len(JOINT_COLORS)]

            # Size proportional to confidence
            r = max(2, int(radius * min(conf, 1.0)))
            cv2.circle(vis, (x, y), r, color, -1)
            cv2.circle(vis, (x, y), r, (255, 255, 255), 1)

    return vis


def create_detection_grid(
    render_dir: str,
    frame_id: str,
    keypoints_2d: np.ndarray,
    num_views: int,
    max_cols: int = 4,
    conf_threshold: float = 0.3,
):
    """Create a grid of rendered views with detection overlays.

    Args:
        render_dir: Directory with rendered frames
        frame_id: Frame identifier
        keypoints_2d: (num_views, 22, 3) detected keypoints
        num_views: Number of views
        max_cols: Maximum columns in grid
    Returns:
        grid: Concatenated image grid
    """
    frame_dir = os.path.join(render_dir, frame_id)
    panels = []

    for view_idx in range(num_views):
        img_path = os.path.join(frame_dir, f"cam_{view_idx:03d}.png")
        img = cv2.imread(img_path)
        if img is None:
            continue

        kps = keypoints_2d[view_idx]
        vis = draw_keypoints(img, kps, conf_threshold)

        # Add view label
        n_det = int((kps[:, 2] > conf_threshold).sum())
        label = f"View {view_idx} ({n_det}/22)"
        cv2.putText(vis, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        panels.append(vis)

    if not panels:
        return None

    # Arrange into grid
    n_cols = min(max_cols, len(panels))
    n_rows = (len(panels) + n_cols - 1) // n_cols

    h, w = panels[0].shape[:2]

    # Pad to fill grid
    while len(panels) < n_rows * n_cols:
        panels.append(np.zeros_like(panels[0]))

    rows = []
    for r in range(n_rows):
        row_panels = panels[r * n_cols:(r + 1) * n_cols]
        rows.append(np.hstack(row_panels))

    grid = np.vstack(rows)
    return grid


def visualize_frames(
    render_dir: str,
    detection_dir: str,
    output_dir: str,
    n_samples: int = 10,
    conf_threshold: float = 0.3,
):
    """Create visualization grids for sample frames."""
    os.makedirs(output_dir, exist_ok=True)

    # Find detection files
    det_files = sorted([
        f for f in os.listdir(detection_dir)
        if f.startswith("detections_") and f.endswith(".json")
    ])

    if not det_files:
        print("No detection files found!")
        return

    # Sample frames (deterministic)
    rng = np.random.RandomState(42)
    indices = rng.choice(len(det_files), min(n_samples, len(det_files)), replace=False)
    indices.sort()

    for idx in indices:
        det_path = os.path.join(detection_dir, det_files[idx])
        with open(det_path) as f:
            det_data = json.load(f)

        frame_id = det_data["frame_id"]
        kps_2d = np.array(det_data["keypoints_2d"])
        num_views = det_data["num_views"]

        grid = create_detection_grid(
            render_dir, frame_id, kps_2d, num_views,
            conf_threshold=conf_threshold,
        )

        if grid is not None:
            save_path = os.path.join(output_dir, f"detection_{frame_id}.png")
            cv2.imwrite(save_path, grid)

    print(f"Saved {len(indices)} detection grids to {output_dir}")


def create_success_failure_comparison(
    detection_dir: str,
    render_dir: str,
    output_dir: str,
    n_each: int = 3,
    conf_threshold: float = 0.3,
):
    """Create a comparison showing best and worst detection examples."""
    # Load all-frames results
    results_path = os.path.join(detection_dir, "neural_results_all_frames.json")
    if not os.path.exists(results_path):
        print("neural_results_all_frames.json not found, skipping success/failure comparison")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    # Sort by MPJPE
    sorted_results = sorted(all_results, key=lambda r: r["mpjpe"])
    best = sorted_results[:n_each]
    worst = sorted_results[-n_each:]

    # Create comparison
    panels = []
    labels = []

    for label_prefix, examples in [("BEST", best), ("WORST", worst)]:
        for r in examples:
            frame_id = r["frame_id"]
            det_path = os.path.join(detection_dir, f"detections_{frame_id}.json")
            if not os.path.exists(det_path):
                continue

            with open(det_path) as f:
                det_data = json.load(f)

            kps_2d = np.array(det_data["keypoints_2d"])
            num_views = min(4, det_data["num_views"])

            grid = create_detection_grid(
                render_dir, frame_id, kps_2d, num_views,
                max_cols=4, conf_threshold=conf_threshold,
            )

            if grid is not None:
                # Add MPJPE label
                mpjpe = r["mpjpe"]
                cv2.putText(grid, f"{label_prefix}: MPJPE={mpjpe:.1f}mm (frame {frame_id})",
                            (10, grid.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                panels.append(grid)

    if panels:
        # Stack vertically with separators
        max_w = max(p.shape[1] for p in panels)
        padded = []
        for p in panels:
            if p.shape[1] < max_w:
                pad = np.zeros((p.shape[0], max_w - p.shape[1], 3), dtype=np.uint8)
                p = np.hstack([p, pad])
            padded.append(p)
            # Add thin separator
            padded.append(np.ones((3, max_w, 3), dtype=np.uint8) * 128)

        comparison = np.vstack(padded[:-1])  # Remove last separator
        save_path = os.path.join(output_dir, "detection_examples.png")
        cv2.imwrite(save_path, comparison)
        print(f"Saved success/failure comparison to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize keypoint detections on rendered images"
    )
    parser.add_argument("--render_dir", type=str, required=True)
    parser.add_argument("--detection_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.expanduser(
                            "~/outputs/neural_triangulation/comparison/plots"))
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--conf_threshold", type=float, default=0.3)
    args = parser.parse_args()

    print("Creating detection visualizations...")
    visualize_frames(
        args.render_dir, args.detection_dir, args.output_dir,
        n_samples=args.n_samples, conf_threshold=args.conf_threshold,
    )

    print("\nCreating success/failure comparison...")
    create_success_failure_comparison(
        args.detection_dir, args.render_dir, args.output_dir,
        conf_threshold=args.conf_threshold,
    )


if __name__ == "__main__":
    main()
