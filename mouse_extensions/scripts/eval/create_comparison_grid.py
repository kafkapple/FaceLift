#!/usr/bin/env python
"""
FL vs PS Comparison Grid Generator.

Creates side-by-side comparison grids:
  Rows: GT, FaceLift (6v), Pose-Splatter (M5)
  Cols: View 0, View 1, View 2, View 3, View 4, View 5

Usage:
    python create_comparison_grid.py \
        --fl-dir outputs/tier_comparison/gslrm_6view_test/samples \
        --ps-dir /tmp/ps_renders \
        --gt-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --frames 3240 3300 3360 3420 3480 3540 3599 \
        --output-dir outputs/comparison/fl_vs_ps_comparison
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_image(path: str) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = Image.open(path)
    if img.mode == 'RGBA':
        # Composite on white background
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return np.array(bg)
    return np.array(img.convert('RGB'))


def create_grid(gt_images: list, fl_images: list, ps_images: list,
                frame_id: int, output_path: Path, img_size: int = 256):
    """
    Create a 3-row × 6-col comparison grid with labels.

    Args:
        gt_images: 6 GT view images (numpy arrays)
        fl_images: 6 FL render images (numpy arrays)
        ps_images: 6 PS render images (numpy arrays)
        frame_id: frame number for title
        output_path: save path
        img_size: resize each image to this size
    """
    num_views = 6
    num_rows = 3
    label_width = 120
    header_height = 40
    padding = 2

    grid_w = label_width + num_views * (img_size + padding) + padding
    grid_h = header_height + num_rows * (img_size + padding) + padding

    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()
            font_small = font

    # Column headers (view numbers)
    for v in range(num_views):
        x = label_width + padding + v * (img_size + padding) + img_size // 2
        draw.text((x, 12), f"View {v}", fill=(0, 0, 0), font=font_small,
                  anchor="mm")

    # Row labels and images
    row_labels = ["GT", "FL 6v", "PS M5"]
    row_colors = [(50, 50, 50), (0, 100, 200), (200, 50, 0)]
    all_images = [gt_images, fl_images, ps_images]

    for row_idx, (label, color, images) in enumerate(
            zip(row_labels, row_colors, all_images)):
        y_start = header_height + padding + row_idx * (img_size + padding)

        # Row label
        draw.text((label_width // 2, y_start + img_size // 2),
                  label, fill=color, font=font, anchor="mm")

        for v in range(num_views):
            x_start = label_width + padding + v * (img_size + padding)

            if images[v] is not None:
                img = Image.fromarray(images[v]).resize(
                    (img_size, img_size), Image.LANCZOS)
                grid.paste(img, (x_start, y_start))
            else:
                # Missing image placeholder
                draw.rectangle(
                    [x_start, y_start, x_start + img_size, y_start + img_size],
                    fill=(200, 200, 200), outline=(150, 150, 150))
                draw.text((x_start + img_size // 2, y_start + img_size // 2),
                          "N/A", fill=(100, 100, 100), font=font, anchor="mm")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path), quality=95)
    return grid_w, grid_h


def create_multi_frame_grid(gt_images_all: dict, fl_images_all: dict,
                            ps_images_all: dict, frame_ids: list,
                            output_path: Path, img_size: int = 180):
    """
    Create a large grid with multiple frames.

    Layout:
        For each frame: 3 rows (GT/FL/PS) × 6 cols (views)
        Frames stacked vertically with dividers.
    """
    num_views = 6
    num_rows_per_frame = 3
    label_width = 100
    header_height = 35
    frame_header_height = 28
    padding = 2
    divider_height = 6

    grid_w = label_width + num_views * (img_size + padding) + padding
    total_frame_h = frame_header_height + num_rows_per_frame * (img_size + padding)
    grid_h = (header_height + len(frame_ids) * total_frame_h
              + (len(frame_ids) - 1) * divider_height + padding)

    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 14)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 12)
            font_title = font
        except (OSError, IOError):
            font = ImageFont.load_default()
            font_small = font
            font_title = font

    # Column headers
    for v in range(num_views):
        x = label_width + padding + v * (img_size + padding) + img_size // 2
        draw.text((x, header_height // 2), f"View {v}", fill=(0, 0, 0),
                  font=font_small, anchor="mm")

    row_labels = ["GT", "FL 6v", "PS M5"]
    row_colors = [(50, 50, 50), (0, 100, 200), (200, 50, 0)]

    for fi, frame_id in enumerate(frame_ids):
        y_base = header_height + fi * (total_frame_h + divider_height)

        # Frame divider
        if fi > 0:
            div_y = y_base - divider_height
            draw.rectangle([0, div_y, grid_w, div_y + divider_height],
                          fill=(220, 220, 220))

        # Frame header
        draw.text((grid_w // 2, y_base + frame_header_height // 2),
                  f"Frame {frame_id}", fill=(80, 80, 80),
                  font=font_title, anchor="mm")

        gt_imgs = gt_images_all.get(frame_id, [None] * 6)
        fl_imgs = fl_images_all.get(frame_id, [None] * 6)
        ps_imgs = ps_images_all.get(frame_id, [None] * 6)
        all_imgs = [gt_imgs, fl_imgs, ps_imgs]

        for row_idx, (label, color, images) in enumerate(
                zip(row_labels, row_colors, all_imgs)):
            y_start = (y_base + frame_header_height + padding
                      + row_idx * (img_size + padding))

            # Row label
            draw.text((label_width // 2, y_start + img_size // 2),
                      label, fill=color, font=font, anchor="mm")

            for v in range(num_views):
                x_start = label_width + padding + v * (img_size + padding)
                if images[v] is not None:
                    img = Image.fromarray(images[v]).resize(
                        (img_size, img_size), Image.LANCZOS)
                    grid.paste(img, (x_start, y_start))
                else:
                    draw.rectangle(
                        [x_start, y_start,
                         x_start + img_size, y_start + img_size],
                        fill=(200, 200, 200), outline=(150, 150, 150))
                    draw.text((x_start + img_size // 2,
                              y_start + img_size // 2),
                              "N/A", fill=(100, 100, 100),
                              font=font_small, anchor="mm")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path), quality=95)
    print(f"Multi-frame grid saved: {output_path} ({grid_w}x{grid_h})")


def main():
    parser = argparse.ArgumentParser(
        description="FL vs PS comparison grid generator")
    parser.add_argument('--fl-dir', type=str, required=True,
                       help="FL renders dir (gslrm_6view_test/samples/)")
    parser.add_argument('--ps-dir', type=str, required=True,
                       help="PS renders dir (rendered_views/)")
    parser.add_argument('--gt-dir', type=str, required=True,
                       help="GT images dir (M5 preprocessed)")
    parser.add_argument('--frames', type=int, nargs='+',
                       default=[3240, 3300, 3360, 3420, 3480, 3540, 3599])
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--img-size', type=int, default=256,
                       help="Per-image size in grid")
    parser.add_argument('--multi', action='store_true',
                       help="Create single multi-frame grid instead of per-frame")
    args = parser.parse_args()

    fl_dir = Path(args.fl_dir)
    ps_dir = Path(args.ps_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_all, fl_all, ps_all = {}, {}, {}

    for frame_id in args.frames:
        fid = f'{frame_id:06d}'
        print(f"Loading frame {fid}...")

        # GT images: M5/{frame_id}/images/cam_00{v}.png (RGBA)
        gt_imgs = []
        for v in range(6):
            gt_path = gt_dir / fid / 'images' / f'cam_{v:03d}.png'
            if gt_path.exists():
                gt_imgs.append(load_image(str(gt_path)))
            else:
                print(f"  WARNING: GT missing: {gt_path}")
                gt_imgs.append(None)

        # FL renders: {frame_id}/render_view_0{v}.png
        fl_imgs = []
        for v in range(6):
            fl_path = fl_dir / fid / f'render_view_{v:02d}.png'
            if fl_path.exists():
                fl_imgs.append(load_image(str(fl_path)))
            else:
                print(f"  WARNING: FL missing: {fl_path}")
                fl_imgs.append(None)

        # PS renders: {frame_id}/render_view_0{v}.png
        ps_imgs = []
        for v in range(6):
            ps_path = ps_dir / fid / f'render_view_{v:02d}.png'
            if ps_path.exists():
                ps_imgs.append(load_image(str(ps_path)))
            else:
                print(f"  WARNING: PS missing: {ps_path}")
                ps_imgs.append(None)

        gt_all[frame_id] = gt_imgs
        fl_all[frame_id] = fl_imgs
        ps_all[frame_id] = ps_imgs

        # Per-frame grid
        if not args.multi:
            w, h = create_grid(
                gt_imgs, fl_imgs, ps_imgs, frame_id,
                output_dir / f'comparison_frame_{fid}.png',
                img_size=args.img_size)
            print(f"  Saved: comparison_frame_{fid}.png ({w}x{h})")

    # Multi-frame grid
    if args.multi:
        create_multi_frame_grid(
            gt_all, fl_all, ps_all, args.frames,
            output_dir / 'fl_vs_ps_multi_frame.png',
            img_size=args.img_size)
    else:
        # Also create a combined multi-frame grid
        create_multi_frame_grid(
            gt_all, fl_all, ps_all, args.frames,
            output_dir / 'fl_vs_ps_multi_frame.png',
            img_size=180)

    print(f"\nAll grids saved to: {output_dir}")


if __name__ == '__main__':
    main()
