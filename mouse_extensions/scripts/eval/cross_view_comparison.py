"""Cross-view comparison grid: same frame rendered with 1v-6v GS-LRM models.

Creates publication-quality comparison grids showing quality progression
as input view count increases from 1 to 6.

Output: outputs/visualizations/cross_view_comparison/
"""
import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Paths
TIER_DIR = Path("/home/joon/dev/FaceLift/outputs/tier_comparison")
GT_DIR = Path("/home/joon/data/preprocessed/FaceLift_mouse/M5")
FAIR_DIR = Path("/home/joon/dev/FaceLift/experiments/comparison/tier")
OUT_DIR = Path("/home/joon/dev/FaceLift/outputs/visualizations/cross_view_comparison")

# Representative test frames (spread across test set 3240-3599)
FRAMES = ["003240", "003320", "003400", "003500"]
# Novel views to show (not view 0 = input)
VIEWS = [1, 3]  # adjacent + opposite

# Fair eval PSNR for labels
FAIR_PSNR = {1: 10.47, 2: 15.95, 3: 18.56, 4: 20.66, 5: 22.16, 6: 23.84}
FAIR_IOU = {1: 0.028, 2: 0.858, 3: 0.899, 4: 0.926, 5: 0.942, 6: 0.954}

# Layout
IMG_SIZE = 256  # downscale from 512 for grid
LABEL_H = 40
PAD = 4
NCOLS = 8  # GT + 1v-6v + alpha(6v)


def load_and_resize(path, size=IMG_SIZE):
    """Load image and resize."""
    img = Image.open(path).convert("RGB")
    return img.resize((size, size), Image.LANCZOS)


def load_alpha_composite(path, size=IMG_SIZE):
    """Load RGBA and composite on white background."""
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(bg, img)
    return composite.convert("RGB").resize((size, size), Image.LANCZOS)


def get_font(size=16):
    """Get a font, fallback to default."""
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def draw_label(draw, x, y, w, text, font, color=(255, 255, 255), bg=(40, 40, 40)):
    """Draw centered label with background."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x + (w - tw) // 2
    ty = y + (LABEL_H - th) // 2
    # Background
    draw.rectangle([x, y, x + w, y + LABEL_H], fill=bg)
    draw.text((tx, ty), text, fill=color, font=font)


def create_comparison_grid(frame_id, view_idx, out_path=None):
    """Create a single comparison row for one frame+view."""
    font = get_font(14)
    font_small = get_font(11)

    cell_w = IMG_SIZE + PAD
    total_w = NCOLS * cell_w - PAD
    total_h = LABEL_H + IMG_SIZE + PAD + 20  # extra for metrics

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    col = 0

    # GT
    gt_path = GT_DIR / frame_id / "images" / f"cam_{view_idx:03d}.png"
    if gt_path.exists():
        gt_img = load_alpha_composite(str(gt_path))
        draw_label(draw, col * cell_w, 0, IMG_SIZE, "GT", font, bg=(0, 100, 0))
        canvas.paste(gt_img, (col * cell_w, LABEL_H))
    col += 1

    # 1v-6v renders
    for nv in range(1, 7):
        render_path = (
            TIER_DIR
            / f"gslrm_{nv}view_test"
            / "samples"
            / frame_id
            / f"render_view_{view_idx:02d}.png"
        )
        if render_path.exists():
            render_img = load_and_resize(str(render_path))
            psnr = FAIR_PSNR[nv]
            iou = FAIR_IOU[nv]
            label = f"{nv}v ({psnr:.1f}dB)"
            # Color code: red→yellow→green based on PSNR
            ratio = min(1.0, max(0.0, (psnr - 10) / 14))
            r = int(200 * (1 - ratio))
            g = int(160 * ratio + 40)
            draw_label(
                draw, col * cell_w, 0, IMG_SIZE, label, font, bg=(r, g, 40)
            )
            canvas.paste(render_img, (col * cell_w, LABEL_H))
        col += 1

    # 6v alpha
    alpha_path = (
        TIER_DIR
        / "gslrm_6view_test"
        / "samples"
        / frame_id
        / f"render_alpha_{view_idx:02d}.png"
    )
    if alpha_path.exists():
        alpha_img = load_and_resize(str(alpha_path))
        draw_label(draw, col * cell_w, 0, IMG_SIZE, "6v Alpha", font, bg=(80, 80, 80))
        canvas.paste(alpha_img, (col * cell_w, LABEL_H))

    # Frame info at bottom
    info = f"Frame {frame_id} | View {view_idx}"
    draw.text((4, total_h - 18), info, fill=(100, 100, 100), font=font_small)

    if out_path:
        canvas.save(out_path, quality=95)
    return canvas


def create_multi_frame_grid(frames, view_idx, out_path):
    """Stack multiple frames vertically for one view."""
    font = get_font(14)
    rows = []
    for frame_id in frames:
        row = create_comparison_grid(frame_id, view_idx)
        rows.append(row)

    if not rows:
        return

    total_w = rows[0].width
    row_h = rows[0].height
    total_h = len(rows) * row_h

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    for i, row in enumerate(rows):
        canvas.paste(row, (0, i * row_h))

    canvas.save(out_path, quality=95)
    print(f"  Saved: {out_path} ({total_w}x{total_h})")


def create_full_grid(frames, views, out_path):
    """Create the full grid: rows = frames x views, columns = GT + 1v-6v + alpha."""
    font = get_font(14)
    font_small = get_font(11)

    cell_w = IMG_SIZE + PAD
    total_w = NCOLS * cell_w - PAD
    row_h = LABEL_H + IMG_SIZE + PAD
    # Header row + data rows
    nrows = len(frames) * len(views)
    total_h = LABEL_H + nrows * row_h + 10

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Column headers
    headers = ["GT", "1-view", "2-view", "3-view", "4-view", "5-view", "6-view", "6v Alpha"]
    for col, hdr in enumerate(headers):
        bg = (0, 100, 0) if col == 0 else (80, 80, 80) if col == 7 else (60, 60, 120)
        draw_label(draw, col * cell_w, 0, IMG_SIZE, hdr, font, bg=bg)

    # Data rows
    row = 0
    for frame_id in frames:
        for view_idx in views:
            y_off = LABEL_H + row * row_h

            # Row label on left side (overlay on GT)
            col = 0

            # GT
            gt_path = GT_DIR / frame_id / "images" / f"cam_{view_idx:03d}.png"
            if gt_path.exists():
                gt_img = load_alpha_composite(str(gt_path))
                canvas.paste(gt_img, (col * cell_w, y_off))
                # Overlay frame/view info
                overlay = ImageDraw.Draw(canvas)
                overlay.rectangle(
                    [col * cell_w, y_off, col * cell_w + IMG_SIZE, y_off + 18],
                    fill=(0, 0, 0, 128),
                )
                overlay.text(
                    (col * cell_w + 4, y_off + 2),
                    f"F{frame_id} V{view_idx}",
                    fill=(255, 255, 255),
                    font=font_small,
                )
            col += 1

            # 1v-6v
            for nv in range(1, 7):
                render_path = (
                    TIER_DIR
                    / f"gslrm_{nv}view_test"
                    / "samples"
                    / frame_id
                    / f"render_view_{view_idx:02d}.png"
                )
                if render_path.exists():
                    render_img = load_and_resize(str(render_path))
                    canvas.paste(render_img, (col * cell_w, y_off))
                    # PSNR overlay
                    psnr = FAIR_PSNR[nv]
                    overlay = ImageDraw.Draw(canvas)
                    ratio = min(1.0, max(0.0, (psnr - 10) / 14))
                    r = int(220 * (1 - ratio))
                    g = int(180 * ratio + 40)
                    overlay.rectangle(
                        [
                            col * cell_w,
                            y_off + IMG_SIZE - 18,
                            col * cell_w + IMG_SIZE,
                            y_off + IMG_SIZE,
                        ],
                        fill=(r, g, 40),
                    )
                    overlay.text(
                        (col * cell_w + 4, y_off + IMG_SIZE - 16),
                        f"{psnr:.1f}dB IoU={FAIR_IOU[nv]:.2f}",
                        fill=(255, 255, 255),
                        font=font_small,
                    )
                col += 1

            # 6v alpha
            alpha_path = (
                TIER_DIR
                / "gslrm_6view_test"
                / "samples"
                / frame_id
                / f"render_alpha_{view_idx:02d}.png"
            )
            if alpha_path.exists():
                alpha_img = load_and_resize(str(alpha_path))
                canvas.paste(alpha_img, (col * cell_w, y_off))

            row += 1

    canvas.save(out_path, quality=95)
    print(f"  Full grid: {out_path} ({total_w}x{total_h})")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Cross-View Comparison Grid Generator")
    print(f"Frames: {FRAMES}")
    print(f"Views: {VIEWS}")
    print(f"Output: {OUT_DIR}")
    print()

    # Individual per-view grids
    for view_idx in VIEWS:
        out_path = OUT_DIR / f"cross_view_v{view_idx}_4frames.png"
        print(f"Creating view {view_idx} multi-frame grid...")
        create_multi_frame_grid(FRAMES, view_idx, str(out_path))

    # Full combined grid
    full_path = OUT_DIR / "cross_view_full_grid.png"
    print(f"\nCreating full combined grid...")
    create_full_grid(FRAMES, VIEWS, str(full_path))

    # Also create individual rows for flexibility
    print(f"\nCreating individual rows...")
    for frame_id in FRAMES:
        for view_idx in VIEWS:
            out_path = OUT_DIR / f"row_{frame_id}_v{view_idx}.png"
            create_comparison_grid(frame_id, view_idx, str(out_path))
            print(f"  {out_path.name}")

    print(f"\nDone! All outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
