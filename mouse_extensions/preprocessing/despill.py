"""Green spill removal for multi-view animal data.

Removes environmental green light contamination from foreground pixels.
Designed for lab setups where animals are filmed in green cages.

Algorithm: Two-stage VFX-inspired despill
  Stage 1: Hard cap — G cannot exceed max(R, B)
  Stage 2: Soft correction — proportional spill removal

Usage:
    # Single image
    python -m mouse_extensions.preprocessing.despill \
        --input /path/to/image.png --output /path/to/output.png

    # Batch process entire dataset
    python -m mouse_extensions.preprocessing.despill \
        --input-dir /path/to/gslrm_format_rat2 \
        --output-dir /path/to/gslrm_format_rat2_despilled \
        --spill-ratio 0.7

    # Validate on subset with visualization
    python -m mouse_extensions.preprocessing.despill \
        --input-dir /path/to/gslrm_format_rat2 \
        --validate --num-samples 20 \
        --viz-output /path/to/despill_validation.png
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def despill_green(
    img_rgba: np.ndarray,
    spill_ratio: float = 0.7,
    fg_threshold: int = 128,
) -> Tuple[np.ndarray, dict]:
    """Remove green spill from foreground pixels.

    Args:
        img_rgba: uint8 array [H, W, 4] (RGBA)
        spill_ratio: correction strength (0.5=gentle, 0.9=aggressive)
        fg_threshold: alpha value to classify foreground (0-255)

    Returns:
        (corrected_rgba, metrics_dict)
    """
    rgb = img_rgba[:, :, :3].astype(np.float32) / 255.0
    alpha = img_rgba[:, :, 3]
    fg_mask = alpha > fg_threshold

    if fg_mask.sum() == 0:
        return img_rgba.copy(), {"n_fg": 0, "green_before": 0, "green_after": 0}

    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Measure green contamination BEFORE
    fg_green_before = fg_mask & (G * 255 > R * 255 + 10) & (G * 255 > B * 255 + 10)
    green_ratio_before = fg_green_before.sum() / fg_mask.sum()

    # Stage 1: Hard cap — G cannot exceed max(R, B)
    G_capped = np.where(fg_mask, np.minimum(G, np.maximum(R, B)), G)

    # Stage 2: Soft correction — proportional spill removal
    avg_RB = (R + B) / 2.0
    spill_amount = np.maximum(0.0, G_capped - avg_RB)
    G_corrected = G_capped - spill_amount * spill_ratio

    # Stage 3: Luminance floor — prevent over-darkening
    G_final = np.where(fg_mask, np.maximum(G_corrected, avg_RB * 0.85), G)
    G_final = np.clip(G_final, 0.0, 1.0)

    # Measure green contamination AFTER
    fg_green_after = fg_mask & (G_final * 255 > R * 255 + 10) & (G_final * 255 > B * 255 + 10)
    green_ratio_after = fg_green_after.sum() / fg_mask.sum()

    # Build output
    out = img_rgba.copy()
    out[:, :, 1] = (G_final * 255).astype(np.uint8)

    metrics = {
        "n_fg": int(fg_mask.sum()),
        "green_before": float(green_ratio_before),
        "green_after": float(green_ratio_after),
        "reduction_pct": float((green_ratio_before - green_ratio_after) / max(green_ratio_before, 1e-8) * 100),
        "mean_g_shift": float((G[fg_mask] - G_final[fg_mask]).mean() * 255),
    }
    return out, metrics


def create_validation_grid(
    input_dir: str,
    num_samples: int = 20,
    spill_ratio: float = 0.7,
) -> Tuple[np.ndarray, list]:
    """Create before/after visualization grid for validation.

    Returns:
        (grid_image, list_of_metrics)
    """
    data_path = Path(input_dir)
    frame_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.isdigit()])

    # Collect samples: mix of worst-case and typical
    samples = []
    for fd in frame_dirs[:100]:
        for cam_id in range(6):
            img_path = fd / "images" / f"cam_{cam_id:03d}.png"
            if not img_path.exists():
                continue
            arr = np.array(Image.open(img_path))
            alpha = arr[:, :, 3]
            fg = alpha > 128
            if fg.sum() < 100:
                continue
            rgb = arr[:, :, :3].astype(float)
            g_dom = fg & (rgb[:, :, 1] > rgb[:, :, 0] + 10) & (rgb[:, :, 1] > rgb[:, :, 2] + 10)
            ratio = g_dom.sum() / fg.sum()
            samples.append((str(img_path), ratio, fd.name, cam_id))

    # Sort by green ratio, pick worst + spread
    samples.sort(key=lambda x: -x[1])
    worst = samples[:num_samples // 2]
    typical = samples[len(samples) // 3 : len(samples) // 3 + num_samples // 2]
    selected = worst + typical

    # Build grid: 4 columns × N rows
    # [Original | Despilled | Green Mask Before | Green Mask After]
    cell = 192
    cols = 4
    rows = len(selected)
    grid = np.ones((rows * cell, cols * cell, 3), dtype=np.uint8) * 40

    all_metrics = []
    for i, (path, ratio, frame, cam) in enumerate(selected):
        arr = np.array(Image.open(path))
        corrected, metrics = despill_green(arr, spill_ratio=spill_ratio)
        metrics["frame"] = frame
        metrics["cam"] = cam
        metrics["path"] = path
        all_metrics.append(metrics)

        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]
        fg = alpha > 128

        # Col 0: Original (white BG composite)
        orig_comp = np.ones_like(rgb) * 255
        m = (alpha / 255.0)[:, :, np.newaxis]
        orig_comp = (rgb * m + orig_comp * (1 - m)).astype(np.uint8)

        # Col 1: Despilled
        corr_rgb = corrected[:, :, :3]
        desp_comp = (corr_rgb * m + np.ones_like(rgb) * 255 * (1 - m)).astype(np.uint8)

        # Col 2: Green mask BEFORE (red = green-dominant)
        mask_before = np.ones((*rgb.shape[:2], 3), dtype=np.uint8) * 220
        mask_before[fg] = [80, 80, 80]
        g_dom = fg & (rgb[:, :, 1].astype(int) > rgb[:, :, 0].astype(int) + 10) & \
                (rgb[:, :, 1].astype(int) > rgb[:, :, 2].astype(int) + 10)
        mask_before[g_dom] = [255, 50, 50]

        # Col 3: Green mask AFTER
        corr_f = corrected[:, :, :3].astype(float)
        mask_after = np.ones((*rgb.shape[:2], 3), dtype=np.uint8) * 220
        mask_after[fg] = [80, 80, 80]
        g_dom_after = fg & (corr_f[:, :, 1] > corr_f[:, :, 0] + 10) & (corr_f[:, :, 1] > corr_f[:, :, 2] + 10)
        mask_after[g_dom_after] = [255, 50, 50]

        for j, img_arr in enumerate([orig_comp, desp_comp, mask_before, mask_after]):
            resized = np.array(Image.fromarray(img_arr).resize((cell, cell), Image.LANCZOS))
            grid[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = resized

    # Add header
    from PIL import ImageDraw, ImageFont
    grid_img = Image.fromarray(grid)
    draw = ImageDraw.Draw(grid_img)
    headers = ["Original", "Despilled", "Green Before", "Green After"]
    for j, h in enumerate(headers):
        draw.text((j * cell + 5, 2), h, fill=(255, 255, 0))
    for i, m in enumerate(all_metrics):
        label = f"{m['frame']}/c{m['cam']} {m['green_before']:.0%}→{m['green_after']:.0%}"
        draw.text((5, i * cell + cell - 14), label, fill=(255, 255, 0))

    return np.array(grid_img), all_metrics


def batch_despill(
    input_dir: str,
    output_dir: str,
    spill_ratio: float = 0.7,
) -> dict:
    """Process entire dataset directory."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    frame_dirs = sorted([d for d in in_path.iterdir() if d.is_dir() and d.name.isdigit()])
    total_metrics = []

    for fi, fd in enumerate(frame_dirs):
        # Copy non-image files (cameras, etc.)
        dst_frame = out_path / fd.name
        dst_frame.mkdir(parents=True, exist_ok=True)

        # Copy opencv_cameras.json
        cam_json = fd / "opencv_cameras.json"
        if cam_json.exists():
            import shutil
            shutil.copy2(cam_json, dst_frame / "opencv_cameras.json")

        # Process images
        img_dir = fd / "images"
        dst_img_dir = dst_frame / "images"
        dst_img_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists():
            continue

        for img_file in sorted(img_dir.glob("cam_*.png")):
            arr = np.array(Image.open(img_file))
            corrected, metrics = despill_green(arr, spill_ratio=spill_ratio)
            Image.fromarray(corrected).save(dst_img_dir / img_file.name)
            total_metrics.append(metrics)

        if (fi + 1) % 500 == 0:
            print(f"  Processed {fi + 1}/{len(frame_dirs)} frames")

    # Summary
    if total_metrics:
        avg_before = np.mean([m["green_before"] for m in total_metrics])
        avg_after = np.mean([m["green_after"] for m in total_metrics])
        avg_shift = np.mean([m["mean_g_shift"] for m in total_metrics])
        summary = {
            "total_images": len(total_metrics),
            "avg_green_before": avg_before,
            "avg_green_after": avg_after,
            "avg_g_channel_shift": avg_shift,
            "spill_ratio": spill_ratio,
        }
        print(f"\nDespill complete: {len(total_metrics)} images")
        print(f"  Green ratio: {avg_before:.1%} → {avg_after:.1%}")
        print(f"  Avg G shift: {avg_shift:.1f} (0-255)")
        return summary
    return {}


def main():
    parser = argparse.ArgumentParser(description="Green spill removal")
    parser.add_argument("--input", type=str, help="Single input image")
    parser.add_argument("--output", type=str, help="Single output image")
    parser.add_argument("--input-dir", type=str, help="Batch input directory")
    parser.add_argument("--output-dir", type=str, help="Batch output directory")
    parser.add_argument("--spill-ratio", type=float, default=0.7)
    parser.add_argument("--validate", action="store_true", help="Run validation grid only")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--viz-output", type=str, default="despill_validation.png")
    args = parser.parse_args()

    if args.validate and args.input_dir:
        print(f"Creating validation grid from {args.input_dir}...")
        grid, metrics = create_validation_grid(
            args.input_dir, args.num_samples, args.spill_ratio
        )
        Image.fromarray(grid).save(args.viz_output)
        print(f"Saved: {args.viz_output}")
        avg_b = np.mean([m["green_before"] for m in metrics])
        avg_a = np.mean([m["green_after"] for m in metrics])
        print(f"Green ratio: {avg_b:.1%} → {avg_a:.1%} (reduction: {(avg_b-avg_a)/avg_b*100:.0f}%)")

    elif args.input and args.output:
        arr = np.array(Image.open(args.input))
        corrected, metrics = despill_green(arr, spill_ratio=args.spill_ratio)
        Image.fromarray(corrected).save(args.output)
        print(f"Despilled: {metrics}")

    elif args.input_dir and args.output_dir:
        print(f"Batch despill: {args.input_dir} → {args.output_dir}")
        batch_despill(args.input_dir, args.output_dir, args.spill_ratio)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
