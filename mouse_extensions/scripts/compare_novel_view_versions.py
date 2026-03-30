"""Compare old (384px) vs new (512px) novel view dataset side-by-side.

Generates comparison grids for visual verification after regeneration.

Usage:
    python -m mouse_extensions.scripts.compare_novel_view_versions \
        --old outputs/datasets/novel_view/mouse_m5t2/_archive_384px \
        --new outputs/datasets/novel_view/mouse_m5t2 \
        --output_dir outputs/viz/comparison/mouse/novel_view_384vs512 \
        --n_samples 5
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_and_resize(path: str, target_size: int = 512) -> np.ndarray:
    """Load image and resize to target, preserving aspect ratio."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return img


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    """Add text label to top-left."""
    out = img.copy()
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare 384px vs 512px novel views")
    parser.add_argument("--old", required=True, help="Old dataset root (with tier0_raw/, pseudo_gt/)")
    parser.add_argument("--new", required=True, help="New dataset root")
    parser.add_argument("--output_dir", default="outputs/viz/comparison/mouse/novel_view_384vs512")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--views", nargs="+", default=["bottom", "top", "front_low", "side_low"])
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = 512

    # Sample frame indices (evenly spaced)
    sample_frames = [i * 720 for i in range(args.n_samples)]  # 0, 720, 1440, 2160, 2880

    for view in args.views:
        rows = []
        for fi in sample_frames:
            fname = f"{fi:05d}.png"

            # Old tier0_raw (GS-LRM input)
            old_input = load_and_resize(str(Path(args.old) / "tier0_raw" / view / fname), target)
            old_input = add_label(old_input, f"OLD input 384px f{fi}")

            # New tier0_raw
            new_input_path = Path(args.new) / "mouse_m5t2" / "tier0_raw" / view / fname
            if new_input_path.exists():
                new_input = load_and_resize(str(new_input_path), target)
                new_input = add_label(new_input, f"NEW input 512px f{fi}")
            else:
                new_input = np.ones((target, target, 3), dtype=np.uint8) * 128
                new_input = add_label(new_input, "NOT YET GENERATED")

            # Old pseudo_gt (MAMMAL target)
            old_target = load_and_resize(str(Path(args.old) / "pseudo_gt" / view / fname), target)
            old_target = add_label(old_target, f"OLD target 384px f{fi}")

            # New pseudo_gt
            new_target_path = Path(args.new) / "mouse_m5t2" / "pseudo_gt" / view / fname
            if new_target_path.exists():
                new_target = load_and_resize(str(new_target_path), target)
                new_target = add_label(new_target, f"NEW target 512px f{fi}")
            else:
                new_target = np.ones((target, target, 3), dtype=np.uint8) * 128
                new_target = add_label(new_target, "NOT YET GENERATED")

            # Row: [old_input | new_input | old_target | new_target]
            row = np.concatenate([old_input, new_input, old_target, new_target], axis=1)
            rows.append(row)

        # Stack all rows vertically
        grid = np.concatenate(rows, axis=0)

        # Add column headers
        header_h = 30
        header = np.ones((header_h, grid.shape[1], 3), dtype=np.uint8) * 40
        col_w = target
        labels = ["OLD GS-LRM (384px)", "NEW GS-LRM (512px)",
                  "OLD MAMMAL (384px,fast)", "NEW MAMMAL (512px,accurate)"]
        for i, lbl in enumerate(labels):
            x = i * col_w + 10
            cv2.putText(header, lbl, (x, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (220, 220, 220), 1, cv2.LINE_AA)
        grid = np.concatenate([header, grid], axis=0)

        save_path = str(out / f"compare_{view}.png")
        cv2.imwrite(save_path, grid)
        print(f"Saved: {save_path} ({grid.shape[1]}x{grid.shape[0]})")

    print(f"\n✅ Comparison grids saved to {out}/")


if __name__ == "__main__":
    main()
