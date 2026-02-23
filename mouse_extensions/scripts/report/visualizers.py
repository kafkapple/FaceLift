"""Visualization generators: comparison grids and metric bar charts."""

import io
import base64

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _get_fonts(size_bold: int = 14, size_regular: int = 12):
    """Load system fonts with fallback."""
    font_paths = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ]
    for bold_path, regular_path in font_paths:
        try:
            bold = ImageFont.truetype(bold_path, size_bold)
            regular = ImageFont.truetype(regular_path, size_regular)
            return bold, regular
        except (OSError, IOError):
            continue
    default = ImageFont.load_default()
    return default, default


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL Image to base64 data URI string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


def create_comparison_grid(
    rows: list[dict],  # [{name, color, images: [ndarray|None, ...]}]
    frame_id: int,
    num_views: int = 6,
    img_size: int = 200,
) -> Image.Image:
    """Create a comparison grid image.

    Args:
        rows: List of row dicts with name, color, images.
              First row is typically GT.
        frame_id: Frame number for header.
        num_views: Number of view columns.
        img_size: Size of each thumbnail.

    Returns:
        PIL Image of the grid.
    """
    label_w = 110
    header_h = 30
    pad = 2

    grid_w = label_w + num_views * (img_size + pad) + pad
    grid_h = header_h + len(rows) * (img_size + pad) + pad

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font_bold, font_reg = _get_fonts(13, 11)

    # Column headers
    for v in range(num_views):
        x = label_w + pad + v * (img_size + pad) + img_size // 2
        draw.text((x, header_h // 2), f"View {v}", fill=(80, 80, 80),
                  font=font_reg, anchor="mm")

    # Rows
    for ri, row in enumerate(rows):
        y_start = header_h + pad + ri * (img_size + pad)

        # Parse hex color
        c = row.get("color", "#333333")
        color = tuple(int(c.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

        # Row label
        draw.text((label_w // 2, y_start + img_size // 2),
                  row["name"], fill=color, font=font_bold, anchor="mm")

        # Images
        images = row.get("images", [])
        for v in range(num_views):
            x_start = label_w + pad + v * (img_size + pad)
            if v < len(images) and images[v] is not None:
                img = Image.fromarray(images[v]).resize(
                    (img_size, img_size), Image.LANCZOS)
                grid.paste(img, (x_start, y_start))
            else:
                draw.rectangle(
                    [x_start, y_start,
                     x_start + img_size, y_start + img_size],
                    fill=(230, 230, 230), outline=(200, 200, 200))
                draw.text((x_start + img_size // 2,
                           y_start + img_size // 2),
                          "N/A", fill=(150, 150, 150),
                          font=font_reg, anchor="mm")

    return grid


def create_multi_frame_grid(
    rows_per_frame: dict[int, list[dict]],
    frame_ids: list[int],
    num_views: int = 6,
    img_size: int = 180,
) -> Image.Image:
    """Create a stacked multi-frame comparison grid.

    Args:
        rows_per_frame: {frame_id: [row_dicts]} for each frame.
        frame_ids: Ordered list of frame IDs to show.
        num_views: Views per frame.
        img_size: Thumbnail size.

    Returns:
        PIL Image of the multi-frame grid.
    """
    if not frame_ids:
        return Image.new("RGB", (400, 100), (255, 255, 255))

    sample_rows = rows_per_frame[frame_ids[0]]
    n_rows = len(sample_rows)

    label_w = 100
    header_h = 28
    frame_header_h = 24
    pad = 2
    divider_h = 4

    grid_w = label_w + num_views * (img_size + pad) + pad
    frame_block_h = frame_header_h + n_rows * (img_size + pad)
    grid_h = (header_h + len(frame_ids) * frame_block_h
              + max(0, len(frame_ids) - 1) * divider_h + pad)

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font_bold, font_reg = _get_fonts(12, 11)

    # Column headers
    for v in range(num_views):
        x = label_w + pad + v * (img_size + pad) + img_size // 2
        draw.text((x, header_h // 2), f"View {v}", fill=(80, 80, 80),
                  font=font_reg, anchor="mm")

    for fi, fid in enumerate(frame_ids):
        y_base = header_h + fi * (frame_block_h + divider_h)

        # Divider
        if fi > 0:
            draw.rectangle(
                [0, y_base - divider_h, grid_w, y_base],
                fill=(220, 220, 220))

        # Frame label
        draw.text((grid_w // 2, y_base + frame_header_h // 2),
                  f"Frame {fid}", fill=(100, 100, 100),
                  font=font_reg, anchor="mm")

        rows = rows_per_frame.get(fid, [])
        for ri, row in enumerate(rows):
            y_start = y_base + frame_header_h + pad + ri * (img_size + pad)

            c = row.get("color", "#333333")
            color = tuple(int(c.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

            draw.text((label_w // 2, y_start + img_size // 2),
                      row["name"], fill=color, font=font_bold, anchor="mm")

            images = row.get("images", [])
            for v in range(num_views):
                x_start = label_w + pad + v * (img_size + pad)
                if v < len(images) and images[v] is not None:
                    img = Image.fromarray(images[v]).resize(
                        (img_size, img_size), Image.LANCZOS)
                    grid.paste(img, (x_start, y_start))
                else:
                    draw.rectangle(
                        [x_start, y_start,
                         x_start + img_size, y_start + img_size],
                        fill=(230, 230, 230), outline=(200, 200, 200))

    return grid


def create_metric_bars(
    experiments: list[dict],  # [{name, color, value, std}]
    metric_name: str,
    bar_height: int = 28,
    chart_width: int = 500,
    label_width: int = 130,
) -> Image.Image:
    """Create horizontal bar chart for a single metric.

    Args:
        experiments: List of {name, color, value, std} dicts.
        metric_name: Display name of the metric.
        bar_height: Height of each bar.
        chart_width: Width of the bar area.
        label_width: Width of the label area.

    Returns:
        PIL Image of the bar chart.
    """
    n = len(experiments)
    title_h = 26
    pad = 4
    total_h = title_h + n * (bar_height + pad) + pad
    total_w = label_width + chart_width + 80  # extra for value text

    img = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_bold, font_reg = _get_fonts(12, 11)

    # Title
    draw.text((total_w // 2, title_h // 2), metric_name,
              fill=(50, 50, 50), font=font_bold, anchor="mm")

    # Find max for scale
    values = [e.get("value", 0) or 0 for e in experiments]
    max_val = max(values) if values else 1.0
    if max_val <= 0:
        max_val = 1.0

    for i, exp in enumerate(experiments):
        y = title_h + pad + i * (bar_height + pad)
        val = exp.get("value", 0) or 0
        std = exp.get("std")
        c = exp.get("color", "#333333")
        color = tuple(int(c.lstrip("#")[j:j+2], 16) for j in (0, 2, 4))

        # Label
        draw.text((label_width - 4, y + bar_height // 2),
                  exp["name"], fill=color, font=font_reg, anchor="rm")

        # Bar
        bar_w = int((val / max_val) * chart_width) if max_val > 0 else 0
        bar_w = max(bar_w, 2)
        draw.rectangle(
            [label_width, y + 2, label_width + bar_w, y + bar_height - 2],
            fill=color)

        # Value text
        val_str = f"{val:.2f}"
        if std is not None:
            val_str += f" +/-{std:.2f}"
        draw.text((label_width + bar_w + 6, y + bar_height // 2),
                  val_str, fill=(80, 80, 80), font=font_reg, anchor="lm")

    return img
