"""Find high-motion segments from MAMMAL keypoint trajectories.

Computes per-frame motion magnitude from 22 keypoints, identifies
segments with the most movement for visualization.

Usage on gpu03:
    python -m mouse_extensions.behavior.find_active_segments \
        --top-k 5 --segment-len 60 \
        --output-dir outputs/analysis/mouse/behavior_clustering/motion_analysis

    # Just print top segments without saving
    python -m mouse_extensions.behavior.find_active_segments --top-k 10
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_keypoints(kp_path: str) -> np.ndarray:
    """Load all keypoints. Returns (N_frames, 22, 3) in mm."""
    d = np.load(kp_path)
    return d["keypoints"]  # (N, 22, 3)


def compute_frame_motion(keypoints: np.ndarray) -> np.ndarray:
    """Compute per-frame motion magnitude.

    Motion = mean Euclidean displacement of all 22 keypoints between
    consecutive frames (in mm).

    Args:
        keypoints: (N, 22, 3) array

    Returns:
        motion: (N-1,) array of motion magnitudes
    """
    diff = keypoints[1:] - keypoints[:-1]  # (N-1, 22, 3)
    per_joint_disp = np.linalg.norm(diff, axis=2)  # (N-1, 22)
    mean_disp = per_joint_disp.mean(axis=1)  # (N-1,)
    return mean_disp


def compute_segment_motion(
    motion: np.ndarray, segment_len: int, stride: int = 1,
) -> np.ndarray:
    """Compute total motion per sliding-window segment.

    Args:
        motion: (N-1,) per-frame motion
        segment_len: window size in frames
        stride: step between windows

    Returns:
        segment_scores: array of (start_frame, total_motion)
    """
    n = len(motion)
    if n < segment_len:
        return np.array([[0, motion.sum()]])

    segments = []
    for start in range(0, n - segment_len + 1, stride):
        total = motion[start:start + segment_len].sum()
        segments.append([start, total])
    return np.array(segments)


def find_top_segments(
    keypoints: np.ndarray,
    segment_len: int = 60,
    top_k: int = 5,
    min_gap: int = 30,
    jump_frames: set = None,
) -> List[Dict]:
    """Find top-K non-overlapping high-motion segments.

    Args:
        keypoints: (N, 22, 3)
        segment_len: window size
        top_k: number of segments to return
        min_gap: minimum gap between segment starts (avoid overlaps)
        jump_frames: frames to exclude (camera jumps)

    Returns:
        list of dicts with start, end, motion_total, motion_mean, motion_peak
    """
    motion = compute_frame_motion(keypoints)

    # Mask jump frames
    if jump_frames:
        for fi in jump_frames:
            if 0 <= fi < len(motion):
                motion[fi] = 0
            if 0 <= fi - 1 < len(motion):
                motion[fi - 1] = 0

    segments = compute_segment_motion(motion, segment_len, stride=5)

    # Sort by total motion (descending)
    sorted_idx = np.argsort(-segments[:, 1])

    # Greedy non-overlapping selection
    selected = []
    used_starts = []
    for idx in sorted_idx:
        start = int(segments[idx, 0])
        # Check min_gap from all selected
        if any(abs(start - s) < min_gap for s in used_starts):
            continue
        # Check no jump frames in segment
        if jump_frames:
            seg_frames = set(range(start, start + segment_len))
            if seg_frames & jump_frames:
                continue

        end = start + segment_len
        seg_motion = motion[start:end]
        selected.append({
            "start": start,
            "end": end,
            "motion_total": float(segments[idx, 1]),
            "motion_mean": float(seg_motion.mean()),
            "motion_peak": float(seg_motion.max()),
            "peak_frame": start + int(seg_motion.argmax()),
        })
        used_starts.append(start)
        if len(selected) >= top_k:
            break

    return selected


def plot_motion_profile(
    motion: np.ndarray,
    segments: List[Dict],
    output_path: Path,
    segment_len: int = 60,
):
    """Plot motion profile with highlighted segments."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Top: full motion profile
    ax = axes[0]
    frames = np.arange(len(motion))
    ax.plot(frames, motion, color="steelblue", linewidth=0.5, alpha=0.7)

    # Smoothed (rolling mean)
    window = 50
    if len(motion) > window:
        smooth = np.convolve(motion, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window // 2, window // 2 + len(smooth)), smooth,
                color="darkblue", linewidth=2, label=f"Rolling mean ({window}f)")

    # Highlight selected segments
    colors = ["#FF4444", "#44FF44", "#4444FF", "#FF8800", "#FF44FF"]
    for i, seg in enumerate(segments):
        c = colors[i % len(colors)]
        ax.axvspan(seg["start"], seg["end"], alpha=0.2, color=c,
                   label=f"Seg {i+1}: f{seg['start']}-{seg['end']} (motion={seg['motion_mean']:.1f}mm/f)")
        ax.axvline(seg["peak_frame"], color=c, linestyle="--", linewidth=1, alpha=0.5)

    ax.set_ylabel("Mean keypoint displacement (mm/frame)", fontsize=12)
    ax.set_title("Keypoint Motion Profile — Full Timeline", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Bottom: zoomed segments
    ax2 = axes[1]
    for i, seg in enumerate(segments):
        c = colors[i % len(colors)]
        seg_motion = motion[seg["start"]:seg["end"]]
        x = np.arange(len(seg_motion))
        ax2.plot(x, seg_motion, color=c, linewidth=1.5, label=f"Seg {i+1} (f{seg['start']})")
    ax2.set_xlabel("Frame within segment", fontsize=12)
    ax2.set_ylabel("Displacement (mm)", fontsize=12)
    ax2.set_title("Selected High-Motion Segments (overlaid)", fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Motion profile: {output_path}")


def main():
    from mouse_extensions.behavior.paths import GPU03_KEYPOINTS

    JUMP_FRAMES = {1178, 1179, 1180, 1181, 1182, 2358, 2359, 2360, 2361, 2362,
                   3538, 3539, 3540, 3541, 3542}

    parser = argparse.ArgumentParser(description="Find High-Motion Segments")
    parser.add_argument("--kp-path", default=GPU03_KEYPOINTS)
    parser.add_argument("--segment-len", type=int, default=60, help="Segment length in frames")
    parser.add_argument("--top-k", type=int, default=5, help="Number of segments")
    parser.add_argument("--min-gap", type=int, default=60, help="Minimum gap between segments")
    parser.add_argument("--output-dir", default="outputs/analysis/mouse/behavior_clustering/motion_analysis")
    args = parser.parse_args()

    print("Loading keypoints...")
    kp = load_keypoints(args.kp_path)
    print(f"  Shape: {kp.shape} ({kp.shape[0]} frames, {kp.shape[1]} joints)")

    print("Computing motion...")
    motion = compute_frame_motion(kp)
    print(f"  Motion stats: mean={motion.mean():.2f}, std={motion.std():.2f}, "
          f"max={motion.max():.2f} mm/frame")

    print(f"\nFinding top-{args.top_k} high-motion segments (len={args.segment_len})...")
    segments = find_top_segments(
        kp, args.segment_len, args.top_k, args.min_gap, JUMP_FRAMES,
    )

    print(f"\n{'='*70}")
    print(f"TOP {len(segments)} HIGH-MOTION SEGMENTS")
    print(f"{'='*70}")
    for i, seg in enumerate(segments):
        print(f"\n  Segment {i+1}: frames {seg['start']}-{seg['end']}")
        print(f"    Total motion: {seg['motion_total']:.1f} mm")
        print(f"    Mean/frame:   {seg['motion_mean']:.2f} mm")
        print(f"    Peak frame:   {seg['peak_frame']} ({seg['motion_peak']:.2f} mm)")

    # Print as frame-range for render_bodypart_gaussians
    print(f"\n--- Copy-paste for rendering ---")
    for i, seg in enumerate(segments):
        print(f"  --frame-range {seg['start']}:{seg['end']}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    result = {
        "kp_path": args.kp_path,
        "n_frames": int(kp.shape[0]),
        "segment_len": args.segment_len,
        "motion_stats": {
            "mean": float(motion.mean()),
            "std": float(motion.std()),
            "max": float(motion.max()),
            "median": float(np.median(motion)),
        },
        "segments": segments,
    }
    json_path = output_dir / "high_motion_segments.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  JSON: {json_path}")

    # Plot
    plot_motion_profile(motion, segments, output_dir / "motion_profile.png", args.segment_len)


if __name__ == "__main__":
    main()
