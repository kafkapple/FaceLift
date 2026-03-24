#!/usr/bin/env python3
"""HLAC-stratified frame selection for RAT2 SAM2 annotation.

Selects frames from s-DANNCE data using HLAC behavioral class labels
for proportional stratification with minimum per-class guarantees.
Within each class, uses farthest-point sampling (FPS) on keypoint poses
to maximize pose diversity.

Usage:
    python -m mouse_extensions.scripts.select_hlac_frames \
        --hlac_mat /home/joon/data/sdannce/rat/dataverse/SCN2A_M1_20220916_0077_L.mat \
        --target_frames 3000 \
        --min_per_class 50 \
        --output rat2_frame_indices.txt \
        --seed 42

Output: one frame index per line, sorted ascending.
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio


def farthest_point_sampling(points: np.ndarray, n_select: int) -> np.ndarray:
    """Select n_select points from points using farthest-point sampling.

    Args:
        points: (N, D) array of feature vectors.
        n_select: number of points to select.

    Returns:
        indices: (n_select,) array of selected indices.
    """
    n_total = len(points)
    if n_select >= n_total:
        return np.arange(n_total)

    selected = [np.random.randint(n_total)]
    min_dists = np.full(n_total, np.inf)

    for _ in range(n_select - 1):
        last = points[selected[-1]]
        dists = np.linalg.norm(points - last, axis=1)
        min_dists = np.minimum(min_dists, dists)
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)

    return np.array(selected)


def select_hlac_stratified(
    hlac: np.ndarray,
    keypoints: np.ndarray,
    target_total: int = 3000,
    min_per_class: int = 50,
    max_frame: int = 89000,
    seed: int = 42,
) -> np.ndarray:
    """Select frames using HLAC stratification + within-class FPS.

    Args:
        hlac: (T,) array of HLAC class labels (1-indexed).
        keypoints: (T, 23, 3) array of 3D keypoints.
        target_total: target number of frames.
        min_per_class: minimum frames per HLAC class.
        max_frame: maximum frame index (DANNCE has 89K, HLAC has 90K).
        seed: random seed.

    Returns:
        sorted array of selected frame indices.
    """
    np.random.seed(seed)
    classes = sorted(np.unique(hlac[:max_frame]))
    n_valid = min(len(hlac), max_frame)

    print(f"HLAC classes: {classes} ({len(classes)} classes)")
    print(f"Valid frames: {n_valid} (capped at {max_frame})")

    # Phase 1: Proportional allocation with minimum guarantee
    allocations = {}
    for cls in classes:
        cls_count = int(np.sum(hlac[:n_valid] == cls))
        proportion = cls_count / n_valid
        alloc = max(min_per_class, int(target_total * proportion))
        alloc = min(alloc, cls_count)  # can't select more than available
        allocations[cls] = alloc

    # Adjust to hit target_total
    total_alloc = sum(allocations.values())
    if total_alloc > target_total:
        # Scale down proportional classes (keep min_per_class intact)
        excess = total_alloc - target_total
        scalable = {c: a for c, a in allocations.items() if a > min_per_class}
        scale_total = sum(scalable.values())
        for cls in scalable:
            reduction = int(excess * scalable[cls] / scale_total)
            allocations[cls] = max(min_per_class, allocations[cls] - reduction)

    # Phase 2: Within-class FPS for pose diversity
    selected_indices = []
    for cls in classes:
        cls_mask = (hlac[:n_valid] == cls)
        cls_frames = np.where(cls_mask)[0]
        n_select = allocations[cls]

        if len(cls_frames) <= n_select:
            chosen = cls_frames
        else:
            # Flatten keypoints for FPS: (N, 23*3) = (N, 69)
            kp_flat = keypoints[cls_frames].reshape(len(cls_frames), -1)
            fps_idx = farthest_point_sampling(kp_flat, n_select)
            chosen = cls_frames[fps_idx]

        selected_indices.extend(chosen.tolist())
        print(f"  Class {cls}: {len(cls_frames):>6} available → {len(chosen):>4} selected "
              f"({len(chosen)/len(cls_frames)*100:.1f}%)")

    selected = np.array(sorted(set(selected_indices)))
    print(f"\nTotal selected: {len(selected)} frames")
    return selected


def main():
    parser = argparse.ArgumentParser(description="HLAC-stratified frame selection")
    parser.add_argument("--hlac_mat", required=True, help="SocialMapper .mat with HLAC labels")
    parser.add_argument("--target_frames", type=int, default=3000)
    parser.add_argument("--min_per_class", type=int, default=50)
    parser.add_argument("--max_frame", type=int, default=89000,
                        help="Cap at DANNCE frame count (default: 89000)")
    parser.add_argument("--output", default="rat2_frame_indices.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.hlac_mat}...")
    d = sio.loadmat(args.hlac_mat, simplify_cells=True)
    sd = d["sdannce"]
    hlac = sd["hlac"]
    kp = sd["m1"].transpose(0, 2, 1)  # (T, 23, 3)

    print(f"HLAC shape: {hlac.shape}, KP shape: {kp.shape}")

    selected = select_hlac_stratified(
        hlac, kp,
        target_total=args.target_frames,
        min_per_class=args.min_per_class,
        max_frame=args.max_frame,
        seed=args.seed,
    )

    out_path = Path(args.output)
    out_path.write_text("\n".join(str(i) for i in selected) + "\n")
    print(f"Saved {len(selected)} frame indices to {out_path}")


if __name__ == "__main__":
    main()
