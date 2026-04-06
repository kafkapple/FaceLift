"""Apply fx normalization to existing GS-LRM format dataset.

Transforms images via affine scaling to match pretrained fx=549.
Designed to be applied AFTER despill (color correction before geometry).

Pipeline: raw → sdannce_to_gslrm.py → despill.py → normalize_fx.py → final

Usage:
    python -m mouse_extensions.preprocessing.normalize_fx \
        --input-dir /path/to/gslrm_format_rat2_despilled \
        --output-dir /path/to/rat2_despilled_fxnorm

    # Validate only (no output)
    python -m mouse_extensions.preprocessing.normalize_fx \
        --input-dir /path/to/dataset --validate --num-samples 5
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# GS-LRM pretrained targets
TARGET_FX = 549.0
TARGET_CX = 256.0
TARGET_CY = 256.0
IMG_SIZE = 512


def normalize_frame(
    img_rgba: np.ndarray,
    intrinsics: dict,
    target_fx: float = TARGET_FX,
) -> tuple[np.ndarray, dict]:
    """Apply affine scaling to normalize fx to target.

    Args:
        img_rgba: uint8 [H, W, 4] RGBA image
        intrinsics: dict with fx, fy, cx, cy
        target_fx: target focal length (default 549)

    Returns:
        (normalized_rgba, updated_intrinsics)
    """
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]

    if fx <= target_fx * 1.02:
        # Already normalized or close enough
        return img_rgba.copy(), intrinsics.copy()

    fx_scale = target_fx / fx
    # Use same scale for both axes to maintain square pixels (pretrained assumes fx=fy)
    fy_scale = fx_scale

    # Affine: uniform scale around image center (256, 256)
    # This centers cx/cy regardless of original value (rat cx=250.7 → 256)
    # Matches pretrained model's assumption of centered principal point
    M = np.float32([
        [fx_scale, 0, TARGET_CX * (1 - fx_scale)],
        [0, fy_scale, TARGET_CY * (1 - fy_scale)],
    ])

    # Split channels
    rgb = img_rgba[:, :, :3]
    alpha = img_rgba[:, :, 3]

    # Transform RGB (white border for background)
    rgb_norm = cv2.warpAffine(
        rgb, M, (IMG_SIZE, IMG_SIZE),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )

    # Transform mask with INTER_NEAREST (matches sdannce_to_gslrm.py reference)
    # INTER_LINEAR + threshold causes 1px erosion bias at boundaries
    alpha_warped = cv2.warpAffine(
        alpha, M, (IMG_SIZE, IMG_SIZE),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    alpha_norm = alpha_warped

    # Composite: ensure BG is white where mask is 0
    bg_mask = alpha_norm == 0
    rgb_norm[bg_mask] = 255

    normalized = np.dstack([rgb_norm, alpha_norm])

    new_intrinsics = {
        "fx": float(target_fx),
        "fy": float(target_fx),  # pretrained assumes fx=fy=549 (square pixels)
        "cx": float(TARGET_CX),
        "cy": float(TARGET_CY),
    }
    return normalized, new_intrinsics


def process_frame_dir(
    input_dir: Path, output_dir: Path, validate_only: bool = False,
) -> dict:
    """Process a single frame directory (6 camera views)."""
    cam_file = input_dir / "opencv_cameras.json"
    if not cam_file.exists():
        return {"error": f"No camera file in {input_dir}"}

    with open(cam_file) as f:
        cam_data = json.load(f)

    if not validate_only:
        out_img_dir = output_dir / "images"
        out_img_dir.mkdir(parents=True, exist_ok=True)

    metrics = {"frame": input_dir.name, "views": []}
    new_frames = []

    for i, frame in enumerate(cam_data["frames"]):
        img_path = input_dir / "images" / f"cam_{i:03d}.png"
        img = np.array(Image.open(img_path).convert("RGBA"))

        intrinsics = {
            "fx": frame["fx"], "fy": frame["fy"],
            "cx": frame["cx"], "cy": frame["cy"],
        }

        normalized, new_intr = normalize_frame(img, intrinsics)

        # Validate green ratio preserved
        alpha = normalized[:, :, 3]
        fg = alpha > 128
        if fg.sum() > 0:
            fg_rgb = normalized[fg, :3].astype(float)
            gr_ratio = fg_rgb[:, 1].mean() / (fg_rgb[:, 0].mean() + 1e-8)
        else:
            gr_ratio = 0.0

        metrics["views"].append({
            "cam": i, "fx_old": intrinsics["fx"], "fx_new": new_intr["fx"],
            "gr_ratio": round(gr_ratio, 3), "fg_pix": int(fg.sum()),
        })

        if not validate_only:
            Image.fromarray(normalized).save(out_img_dir / f"cam_{i:03d}.png")

        # Update frame entry with new intrinsics (keep w2c unchanged)
        new_frame = frame.copy()
        new_frame.update({
            "fx": new_intr["fx"], "fy": new_intr["fy"],
            "cx": new_intr["cx"], "cy": new_intr["cy"],
        })
        new_frames.append(new_frame)

    if not validate_only:
        new_cam_data = {"frames": new_frames}
        with open(output_dir / "opencv_cameras.json", "w") as f:
            json.dump(new_cam_data, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Apply fx normalization to GS-LRM format dataset"
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Source dataset (e.g., gslrm_format_rat2_despilled)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (required unless --validate)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate only, no output")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Process only N frames (0=all)")
    args = parser.parse_args()

    if not args.validate and args.output_dir is None:
        parser.error("--output-dir required unless --validate")

    # Find frame directories
    frame_dirs = sorted(
        d for d in args.input_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    print(f"Found {len(frame_dirs)} frames in {args.input_dir.name}")

    if args.num_samples > 0:
        frame_dirs = frame_dirs[:args.num_samples]
        print(f"Processing {len(frame_dirs)} samples (--num-samples)")

    all_metrics = []
    for i, fd in enumerate(frame_dirs):
        m = process_frame_dir(fd, args.output_dir / fd.name if args.output_dir else fd,
                              validate_only=args.validate)
        all_metrics.append(m)
        if (i + 1) % 100 == 0 or i == 0:
            v0 = m["views"][0] if "views" in m else {}
            print(f"  [{i+1}/{len(frame_dirs)}] {fd.name}: "
                  f"fx {v0.get('fx_old', '?'):.0f}→{v0.get('fx_new', '?'):.0f}, "
                  f"G/R={v0.get('gr_ratio', '?')}")

    # Summary
    all_gr = [v["gr_ratio"] for m in all_metrics for v in m.get("views", [])]
    if all_gr:
        print(f"\nSummary: {len(frame_dirs)} frames, "
              f"G/R ratio: mean={np.mean(all_gr):.3f}, "
              f"min={np.min(all_gr):.3f}, max={np.max(all_gr):.3f}")

    # Copy split files with resolved absolute paths
    if not args.validate and args.output_dir:
        input_resolved = str(args.input_dir.resolve())
        output_resolved = str(args.output_dir.resolve())
        for split_file in args.input_dir.glob("data_rat2_*.txt"):
            new_lines = []
            with open(split_file) as f:
                for line in f:
                    old_path = str(Path(line.strip()).resolve())
                    new_path = old_path.replace(input_resolved, output_resolved)
                    new_lines.append(new_path)
            out_split = args.output_dir / split_file.name
            with open(out_split, "w") as f:
                f.write("\n".join(new_lines) + "\n")
            print(f"Split file: {out_split.name} ({len(new_lines)} entries)")

    print("Done.")


if __name__ == "__main__":
    main()
