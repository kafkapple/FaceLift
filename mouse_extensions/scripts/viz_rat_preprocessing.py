"""RAT preprocessing version comparison visualization.

Usage:
    python -m mouse_extensions.scripts.viz_rat_preprocessing
    python -m mouse_extensions.scripts.viz_rat_preprocessing --frames 0 150 300
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

VERSIONS = [
    ("gslrm_format_rat2", "Original", "fx=604.9, fy=606.8, cx=250.7, cy=260.4"),
    ("gslrm_format_rat2_despilled", "S1: Despill", "fx=604.9, fy=606.8, cx=250.7, cy=260.4"),
    ("rat2_s1despill_s2fxnorm", "S1+S2: Despill+FXNorm", "fx=549, fy=549, cx=256, cy=256"),
]

NUM_CAMS = 6
DEFAULT_DATA_ROOT = Path("/node_data/joon/data/preprocessed/FaceLift_rat")


def _get_frame_dirs(version_path: Path) -> list[Path]:
    """Return sorted list of frame directories (e.g. 000000/, 000030/)."""
    dirs = sorted(
        d for d in version_path.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    return dirs


def _pick_frames(frame_dirs: list[Path], n: int = 3) -> list[Path]:
    """Pick n evenly-spaced frames (early / mid / late)."""
    if len(frame_dirs) <= n:
        return frame_dirs
    indices = np.linspace(0, len(frame_dirs) - 1, n, dtype=int)
    return [frame_dirs[i] for i in indices]


def _load_frame_images(frame_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all camera views for a frame. Returns (rgb, alpha) arrays.

    Returns:
        rgb: (N, H, W, 3) uint8
        alpha: (N, H, W) uint8
    """
    rgbs, alphas = [], []
    for cam_idx in range(NUM_CAMS):
        img_path = frame_dir / "images" / f"cam_{cam_idx:03d}.png"
        img = np.array(Image.open(img_path))  # RGBA
        rgbs.append(img[:, :, :3])
        alphas.append(img[:, :, 3])
    return np.stack(rgbs), np.stack(alphas)


def _read_intrinsics_summary(version_path: Path, frame_dir: Path) -> str:
    """Read actual fx/fy/cx/cy from opencv_cameras.json."""
    cam_file = frame_dir / "opencv_cameras.json"
    if not cam_file.exists():
        return "no camera file"
    with open(cam_file) as f:
        data = json.load(f)
    # Format: {"frames": [{fx, fy, cx, cy, w2c, ...}, ...]}
    cam = data["frames"][0]
    return f"fx={cam['fx']:.1f}, fy={cam['fy']:.1f}, cx={cam['cx']:.1f}, cy={cam['cy']:.1f}"


def _count_frames_and_size(version_path: Path) -> tuple[int, str]:
    """Count frames and estimate total data size."""
    frame_dirs = _get_frame_dirs(version_path)
    n_frames = len(frame_dirs)
    # Estimate size from first frame
    if frame_dirs:
        frame_size = sum(
            f.stat().st_size for f in frame_dirs[0].rglob("*") if f.is_file()
        )
        total_mb = (frame_size * n_frames) / (1024 ** 2)
        size_str = f"{total_mb:.0f}MB" if total_mb < 1024 else f"{total_mb/1024:.1f}GB"
    else:
        size_str = "0MB"
    return n_frames, size_str


def render_per_frame(
    data_root: Path, frame_name: str, output_dir: Path
) -> Path:
    """Create a comparison PNG for one frame: 3 versions × (6 RGB + 6 alpha)."""
    fig, axes = plt.subplots(
        len(VERSIONS) * 2, NUM_CAMS,
        figsize=(NUM_CAMS * 3, len(VERSIONS) * 2 * 2.5),
    )

    for v_idx, (vname, vlabel, _) in enumerate(VERSIONS):
        ver_path = data_root / vname
        frame_dir = ver_path / frame_name

        rgb, alpha = _load_frame_images(frame_dir)
        actual_intr = _read_intrinsics_summary(ver_path, frame_dir)
        n_frames, size_str = _count_frames_and_size(ver_path)

        row_rgb = v_idx * 2
        row_alpha = v_idx * 2 + 1

        for cam in range(NUM_CAMS):
            axes[row_rgb, cam].imshow(rgb[cam])
            axes[row_rgb, cam].axis("off")
            if cam == 0:
                axes[row_rgb, cam].set_title(
                    f"{vlabel} — RGB\n{actual_intr}\n"
                    f"{n_frames} frames, {size_str}",
                    fontsize=8, loc="left", fontweight="bold",
                )
            else:
                axes[row_rgb, cam].set_title(f"cam_{cam:03d}", fontsize=7)

            axes[row_alpha, cam].imshow(alpha[cam], cmap="gray", vmin=0, vmax=255)
            axes[row_alpha, cam].axis("off")
            if cam == 0:
                axes[row_alpha, cam].set_title(
                    f"{vlabel} — Alpha", fontsize=8, loc="left",
                )

    fig.suptitle(
        f"RAT Preprocessing Comparison — Frame {frame_name}",
        fontsize=14, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = output_dir / f"rat_preproc_{frame_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_summary_grid(
    data_root: Path, frame_names: list[str], output_dir: Path
) -> Path:
    """Create a summary grid: rows = versions, cols = frames (RGB only, cam_000)."""
    n_ver = len(VERSIONS)
    n_fr = len(frame_names)

    fig, axes = plt.subplots(n_ver, n_fr, figsize=(n_fr * 4, n_ver * 3.5))
    if n_ver == 1:
        axes = axes[np.newaxis, :]
    if n_fr == 1:
        axes = axes[:, np.newaxis]

    for v_idx, (vname, vlabel, vintr) in enumerate(VERSIONS):
        ver_path = data_root / vname
        for f_idx, fname in enumerate(frame_names):
            img_path = ver_path / fname / "images" / "cam_000.png"
            img = np.array(Image.open(img_path))[:, :, :3]

            axes[v_idx, f_idx].imshow(img)
            axes[v_idx, f_idx].axis("off")

            if f_idx == 0:
                axes[v_idx, f_idx].set_ylabel(
                    f"{vlabel}\n{vintr}",
                    fontsize=8, rotation=0, labelpad=120, va="center",
                )
            if v_idx == 0:
                axes[v_idx, f_idx].set_title(f"Frame {fname}", fontsize=10)

    fig.suptitle(
        "RAT Preprocessing Summary (cam_000)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0.12, 0, 1, 0.96])

    out_path = output_dir / "rat_preproc_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="RAT preprocessing version comparison visualization"
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="Root dir containing preprocessing version folders",
    )
    parser.add_argument(
        "--frames", nargs="*", default=None,
        help="Frame directory names (e.g. 000000 000150 000300). "
             "Default: auto-pick 3 evenly spaced.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/viz/preprocessing_comparison/rat"),
        help="Output directory for comparison PNGs",
    )
    args = parser.parse_args()

    # Resolve output dir relative to project root if not absolute
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        from mouse_extensions.paths import FACELIFT_ROOT
        output_dir = FACELIFT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine frames to visualize
    first_ver_path = args.data_root / VERSIONS[0][0]
    all_frames = _get_frame_dirs(first_ver_path)
    print(f"Found {len(all_frames)} frames in {first_ver_path.name}")

    if args.frames:
        frame_names = [f"{int(f):06d}" for f in args.frames]
    else:
        selected = _pick_frames(all_frames, 3)
        frame_names = [d.name for d in selected]

    print(f"Visualizing frames: {frame_names}")

    # Per-frame detailed comparison
    for fname in frame_names:
        out = render_per_frame(args.data_root, fname, output_dir)
        print(f"  Saved: {out}")

    # Summary grid
    out = render_summary_grid(args.data_root, frame_names, output_dir)
    print(f"  Saved: {out}")

    print("Done.")


if __name__ == "__main__":
    main()
