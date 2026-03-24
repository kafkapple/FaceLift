#!/usr/bin/env python3
"""HLAC-stratified train/val/test split for RAT2 dataset.

Two-phase design:
  Phase 1 (SAM2): Uniform step=30 → ~3K masks (run kp_sam2_lone.py separately)
  Phase 2 (this script): From the step=30 frames, generate HLAC-stratified
    train/val/test splits with per-class temporal ordering.

Within each HLAC class, frames are kept in temporal order and split:
  first 80% → train, next 10% → val, last 10% → test.
This prevents temporal leakage while ensuring behavioral diversity.

Usage:
    # Mode 1: Generate split files for GS-LRM format directory
    python -m mouse_extensions.scripts.select_hlac_frames \
        --hlac_mat /home/joon/data/sdannce/rat/dataverse/SCN2A_M1_20220916_0077_L.mat \
        --gslrm_dir outputs/datasets/fine_tune/rat/gslrm_format_rat2 \
        --step 30 --max_frame 89000 \
        --split_ratios 0.8 0.1 0.1 \
        --min_per_class 50

    # Mode 2: Just print HLAC distribution for step=N frames (dry run)
    python -m mouse_extensions.scripts.select_hlac_frames \
        --hlac_mat /home/joon/data/sdannce/rat/dataverse/SCN2A_M1_20220916_0077_L.mat \
        --step 30 --max_frame 89000 --dry_run

Output: data_rat2_{train,val,test}.txt in gslrm_dir.
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio


HLAC_NAMES = {
    1: "idle", 2: "sniff_head", 3: "groom", 4: "scrunched",
    5: "active_crouched", 6: "reared", 7: "explore", 8: "locomotion",
}


def analyze_hlac_distribution(
    hlac: np.ndarray,
    frame_indices: np.ndarray,
) -> dict[int, np.ndarray]:
    """Get per-class frame indices from the sampled frames.

    Args:
        hlac: (T,) full HLAC array (90K).
        frame_indices: (N,) array of sampled frame indices (e.g., step=30).

    Returns:
        dict mapping class → sorted array of frame indices in that class.
    """
    labels = hlac[frame_indices]
    class_frames = {}
    for cls in sorted(np.unique(labels)):
        mask = labels == cls
        class_frames[int(cls)] = frame_indices[mask]
    return class_frames


def generate_hlac_split(
    class_frames: dict[int, np.ndarray],
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    min_per_class: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate train/val/test split with per-class temporal ordering.

    Within each class, frames are already in temporal order.
    Split: first R[0] → train, next R[1] → val, last R[2] → test.

    Args:
        class_frames: dict from analyze_hlac_distribution.
        split_ratios: (train, val, test) ratios.
        min_per_class: minimum frames per class in train set.

    Returns:
        (train_indices, val_indices, test_indices) — each sorted.
    """
    train_all, val_all, test_all = [], [], []

    print(f"\n{'Cls':>3} | {'Name':<16} | {'Total':>5} | {'Train':>5} | {'Val':>4} | {'Test':>4}")
    print("-" * 62)

    for cls in sorted(class_frames.keys()):
        frames = np.sort(class_frames[cls])  # temporal order
        n = len(frames)
        name = HLAC_NAMES.get(cls, f"cls_{cls}")

        # Temporal split within class
        n_train = max(min_per_class, int(n * split_ratios[0]))
        n_train = min(n_train, n)  # can't exceed total
        n_remaining = n - n_train
        n_val = max(1, int(n_remaining * split_ratios[1] / (split_ratios[1] + split_ratios[2])))
        n_test = n_remaining - n_val

        # Ensure at least 1 in val and test if class has enough frames
        if n >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n_test)
            n_train = n - n_val - n_test

        train_all.extend(frames[:n_train])
        val_all.extend(frames[n_train:n_train + n_val])
        test_all.extend(frames[n_train + n_val:])

        print(f"  {cls:>1} | {name:<16} | {n:>5} | {n_train:>5} | {n_val:>4} | {n_test:>4}")

    train = np.array(sorted(train_all))
    val = np.array(sorted(val_all))
    test = np.array(sorted(test_all))

    print("-" * 62)
    print(f"{'':>3} | {'TOTAL':<16} | {len(train)+len(val)+len(test):>5} | "
          f"{len(train):>5} | {len(val):>4} | {len(test):>4}")

    return train, val, test


def write_split_files(
    gslrm_dir: Path,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    step: int,
):
    """Write train/val/test split files with GS-LRM format paths.

    Each line: absolute path to UID directory (e.g., /path/to/gslrm_format/000030).
    UID directory name = frame index zero-padded to 6 digits.
    """
    gslrm_dir = Path(gslrm_dir)

    for split_name, indices in [("train", train), ("val", val), ("test", test)]:
        lines = []
        missing = 0
        for idx in indices:
            uid_name = f"{idx:06d}"
            uid_path = gslrm_dir / uid_name
            if uid_path.exists():
                lines.append(str(uid_path.resolve()))
            else:
                missing += 1

        out_file = gslrm_dir / f"data_rat2_{split_name}.txt"
        out_file.write_text("\n".join(lines) + "\n")
        print(f"  {split_name}: {len(lines)} UIDs written to {out_file.name}"
              + (f" ({missing} missing)" if missing else ""))


def main():
    parser = argparse.ArgumentParser(
        description="HLAC-stratified train/val/test split for RAT2")
    parser.add_argument("--hlac_mat", required=True,
                        help="SocialMapper .mat with HLAC labels")
    parser.add_argument("--gslrm_dir", default=None,
                        help="GS-LRM format directory (Phase 2 output)")
    parser.add_argument("--step", type=int, default=30,
                        help="Frame sampling step (must match SAM2 step)")
    parser.add_argument("--max_frame", type=int, default=89000,
                        help="Max frame index (DANNCE limit)")
    parser.add_argument("--split_ratios", type=float, nargs=3,
                        default=[0.8, 0.1, 0.1],
                        help="Train/val/test ratios")
    parser.add_argument("--min_per_class", type=int, default=50,
                        help="Min frames per HLAC class in train set")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print distribution, don't write files")
    args = parser.parse_args()

    # Load HLAC labels
    print(f"Loading {args.hlac_mat}...")
    d = sio.loadmat(args.hlac_mat, simplify_cells=True)
    hlac = d["sdannce"]["hlac"]
    print(f"HLAC: {hlac.shape}, classes: {sorted(np.unique(hlac))}")

    # Generate uniform frame indices (matching SAM2 step)
    frame_indices = np.arange(0, args.max_frame, args.step)
    print(f"Step={args.step}: {len(frame_indices)} frames "
          f"(interval={args.step/50:.2f}s @50fps)")

    # Analyze HLAC distribution on sampled frames
    class_frames = analyze_hlac_distribution(hlac, frame_indices)

    if args.dry_run:
        print("\n=== HLAC Distribution (dry run) ===")
        for cls, frames in sorted(class_frames.items()):
            name = HLAC_NAMES.get(cls, f"cls_{cls}")
            print(f"  Class {cls} ({name}): {len(frames)} frames "
                  f"({len(frames)/len(frame_indices)*100:.1f}%)")
        print(f"  Total: {len(frame_indices)} frames")
        return

    # Generate HLAC-stratified split
    print("\n=== HLAC-Stratified Split ===")
    train, val, test = generate_hlac_split(
        class_frames,
        split_ratios=tuple(args.split_ratios),
        min_per_class=args.min_per_class,
    )

    # Write split files if gslrm_dir provided
    if args.gslrm_dir:
        print(f"\n=== Writing split files to {args.gslrm_dir} ===")
        write_split_files(Path(args.gslrm_dir), train, val, test, args.step)
    else:
        # Just output frame indices
        for split_name, indices in [("train", train), ("val", val), ("test", test)]:
            print(f"\n{split_name}: {len(indices)} frames, "
                  f"range [{indices[0]}-{indices[-1]}]")


if __name__ == "__main__":
    main()
