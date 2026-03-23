"""Grid and annotation utilities for turntable visualization.

Provides grid creation, row labeling, angle overlays, frame subsampling,
and labeled input strip composition for multi-view visualizations.

Consolidated from turntable_config.py and gaussians_renderer.py (2026-03-23).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from einops import rearrange


# ---------------------------------------------------------------------------
# Frame subsampling (from turntable_config.py, 2026-03-23)
# ---------------------------------------------------------------------------

def subsample_frames_for_grid(
    all_frames: np.ndarray,
    video_views: int,
    grid_views: int = 36,
    segments: Optional[list] = None,
) -> np.ndarray:
    """Subsample video frames for grid display.

    When segments are provided, only transition (non-hold) frames are used
    to avoid duplicates from hold periods in the grid.

    Args:
        all_frames: [H, V, W, C] array of all video frames.
        video_views: Number of video frames (e.g., 144).
        grid_views: Target number of grid frames (e.g., 36).
        segments: Optional list of (start, end, from_cam, to_cam, is_hold).

    Returns:
        [H, grid_views, W, C] subsampled frames.
    """
    if video_views <= grid_views:
        return all_frames

    if segments is not None:
        transition_indices = []
        for seg in segments:
            start, end, _, _, is_hold = seg
            if not is_hold:
                for idx in range(start, end):
                    if idx < video_views:
                        transition_indices.append(idx)

        if len(transition_indices) >= grid_views:
            sub_idx = np.linspace(0, len(transition_indices) - 1, grid_views, dtype=int)
            indices = [transition_indices[i] for i in sub_idx]
        else:
            indices = np.linspace(0, video_views - 1, grid_views, dtype=int).tolist()
    else:
        indices = np.linspace(0, video_views - 1, grid_views, dtype=int).tolist()

    return all_frames[:, indices, :, :]


def create_grid_from_video(
    turntable_image: np.ndarray,
    video_views: int,
    grid_rows: int = 6,
    grid_cols: int = 6,
    segments: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Create grid image from turntable video frames.

    Args:
        turntable_image: [H, V*W, C] concatenated video frames.
        video_views: Number of video frames.
        grid_rows: Number of grid rows (default 6).
        grid_cols: Number of grid columns (default 6).
        segments: Optional segment info to exclude hold frames.

    Returns:
        (grid_image, all_frames, h_img) tuple.
    """
    h_img = turntable_image.shape[0]
    w_per_view = turntable_image.shape[1] // video_views
    all_frames = turntable_image.reshape(h_img, video_views, w_per_view, 3)

    grid_views = grid_rows * grid_cols
    grid_frames = subsample_frames_for_grid(all_frames, video_views, grid_views, segments)

    grid_image = rearrange(
        grid_frames, "h (rows cols) w c -> (rows h) (cols w) c",
        rows=grid_rows, cols=grid_cols,
    )
    return grid_image, all_frames, h_img


# ---------------------------------------------------------------------------
# Row labels — top position (from gaussians_renderer.py, 2026-03-23)
# ---------------------------------------------------------------------------

def _compute_row_segment_labels(
    camera_order: list,
    grid_rows: int,
    grid_cols: int,
    loop: bool = True,
) -> List[Tuple[int, int]]:
    """Compute (from_cam, to_cam) for each grid row based on camera order.

    Shared logic for both top and left label functions.
    """
    full_order = list(camera_order)
    if loop and full_order[0] != full_order[-1]:
        full_order = full_order + [full_order[0]]

    total_frames = grid_rows * grid_cols
    num_segments = len(full_order) - 1
    frames_per_segment = total_frames / num_segments if num_segments > 0 else total_frames

    labels = []
    for row_idx in range(grid_rows):
        row_start = row_idx * grid_cols
        row_end = row_start + grid_cols - 1

        start_seg = int(row_start / frames_per_segment) if frames_per_segment > 0 else 0
        end_seg = int(row_end / frames_per_segment) if frames_per_segment > 0 else 0
        start_seg = min(start_seg, num_segments - 1)
        end_seg = min(end_seg, num_segments - 1)

        if start_seg == end_seg:
            from_cam = full_order[start_seg]
            to_cam = full_order[start_seg + 1]
        else:
            from_cam = full_order[start_seg]
            to_cam = full_order[end_seg + 1]

        labels.append((from_cam, to_cam))
    return labels


def add_row_labels_to_grid(
    grid_image: np.ndarray,
    camera_order: list,
    grid_rows: int,
    grid_cols: int,
    row_height: int,
    label_height: int = 55,
    loop: bool = True,
) -> np.ndarray:
    """Add camera transition labels to the TOP of each row in the grid.

    Camera transitions (e.g., 'Cam 1->3') are shown only when they change.
    Angle ranges are always shown for every row.

    Args:
        grid_image: [H, W, 3] uint8 image.
        camera_order: Camera traversal order.
        grid_rows: Number of rows in the grid.
        grid_cols: Number of columns in the grid.
        row_height: Height of each cell in pixels.
        label_height: Height of label bar in pixels.
        loop: Whether camera path loops back to start.

    Returns:
        Image with label bars (height increases by label_height * grid_rows).
    """
    h, w = grid_image.shape[:2]
    seg_labels = _compute_row_segment_labels(camera_order, grid_rows, grid_cols, loop)
    total_frames = grid_rows * grid_cols

    # Build display text per row
    row_display = []
    prev_cam_text = None
    for row_idx, (from_cam, to_cam) in enumerate(seg_labels):
        row_start = row_idx * grid_cols
        row_end = row_start + grid_cols - 1
        start_angle = int((row_start / total_frames) * 360)
        end_angle = int((row_end / total_frames) * 360)

        cam_text_full = f"Cam {from_cam} -> {to_cam}"
        cam_text = cam_text_full if cam_text_full != prev_cam_text else ""
        prev_cam_text = cam_text_full
        angle_text = f"({start_angle} - {end_angle} deg)"
        row_display.append((cam_text, angle_text))

    # Create new image with label bars
    new_height = h + label_height * grid_rows
    result = np.zeros((new_height, w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for row_idx in range(grid_rows):
        label_y = row_idx * (row_height + label_height)
        img_y = label_y + label_height

        # Copy image row
        src_start = row_idx * row_height
        result[img_y:img_y + row_height, :] = grid_image[src_start:src_start + row_height, :]

        # Draw label bar
        result[label_y:label_y + label_height, :] = (30, 30, 30)
        cam_text, angle_text = row_display[row_idx]

        if cam_text:
            display_text = f"{cam_text}  {angle_text}"
            font_scale, color, font_thick = 1.3, (255, 255, 255), 2
        else:
            display_text = angle_text
            font_scale, color, font_thick = 1.1, (200, 200, 100), 2

        (tw, th), _ = cv2.getTextSize(display_text, font, font_scale, font_thick)
        text_y = label_y + (label_height + th) // 2
        cv2.putText(result, display_text, (10, text_y), font, font_scale, color, font_thick)

    return result


# ---------------------------------------------------------------------------
# Row labels — left position (from turntable_config.py, 2026-03-23)
# ---------------------------------------------------------------------------

def add_left_row_labels(
    grid_image: np.ndarray,
    camera_order: list,
    grid_rows: int,
    grid_cols: int,
    row_height: int,
    label_width: int = 80,
    font_scale: float = 1.0,
    loop: bool = True,
) -> np.ndarray:
    """Add camera transition labels to the LEFT side of each row in the grid.

    Args:
        grid_image: [H, W, 3] uint8 image.
        camera_order: Camera traversal order.
        grid_rows: Number of rows in the grid.
        grid_cols: Number of columns in the grid.
        row_height: Height of each cell in pixels.
        label_width: Width of label bar in pixels.
        font_scale: Font scale for labels.
        loop: Whether camera order loops back to start.

    Returns:
        [H, W + label_width, 3] image with labels on left.
    """
    seg_labels = _compute_row_segment_labels(camera_order, grid_rows, grid_cols, loop)

    h, w = grid_image.shape[:2]
    result = np.zeros((h, w + label_width, 3), dtype=np.uint8)
    result[:, :label_width] = (40, 40, 40)
    result[:, label_width:] = grid_image

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thick = 1

    for row_idx, (from_cam, to_cam) in enumerate(seg_labels):
        label_y = row_idx * row_height + row_height // 2
        text = f"{from_cam}->{to_cam}"
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thick)
        text_x = (label_width - tw) // 2
        text_y = label_y + th // 2
        cv2.putText(result, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thick)

    return result


# ---------------------------------------------------------------------------
# Angle overlay (from turntable_config.py, 2026-03-23)
# ---------------------------------------------------------------------------

def add_angle_overlay_to_grid(
    grid_image: np.ndarray,
    grid_rows: int = 6,
    grid_cols: int = 6,
    font_scale: float = 0.5,
    start_angle: float = 0.0,
) -> np.ndarray:
    """Add azimuth angle overlay to bottom-left of each cell in the grid.

    Args:
        grid_image: [H, W, 3] uint8 grid image.
        grid_rows: Number of rows.
        grid_cols: Number of columns.
        font_scale: Font scale for overlay text.
        start_angle: Starting angle (default 0).

    Returns:
        Grid image with angle overlays.
    """
    result = grid_image.copy()
    h, w = grid_image.shape[:2]
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    total_frames = grid_rows * grid_cols
    font = cv2.FONT_HERSHEY_SIMPLEX

    for row in range(grid_rows):
        for col in range(grid_cols):
            frame_idx = row * grid_cols + col
            angle = (start_angle + (frame_idx / total_frames) * 360) % 360
            x = col * cell_w + 5
            y = (row + 1) * cell_h - 5
            text = str(int(angle))
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
            cv2.rectangle(result, (x - 2, y - th - 2), (x + tw + 2, y + 2), (0, 0, 0), -1)
            cv2.putText(result, text, (x, y), font, font_scale, (255, 255, 255), 1)

    return result


# ---------------------------------------------------------------------------
# Labeled input strip (from gaussians_renderer.py, 2026-03-23)
# ---------------------------------------------------------------------------

def create_labeled_input_strip(
    all_images,  # torch.Tensor [V, C, H, W]
    camera_order: list,
    target_h: int,
    target_w: int,
    border: int = 2,
    input_indices: Optional[list] = None,
    view_indices: Optional[list] = None,
) -> Optional[np.ndarray]:
    """Create image strip with camera labels, ordered by camera_order.

    Shows all views (input + predicted) with color-coded borders:
    green for input views, blue for predicted views.

    Args:
        all_images: [V, C, H, W] tensor of all images.
        camera_order: Camera indices in traversal order.
        target_h: Target height for the strip (including labels).
        target_w: Target width for the entire strip.
        border: Border width between images.
        input_indices: Indices that were input views (green border).
        view_indices: Tensor position -> camera ID mapping.

    Returns:
        [H, W, 3] uint8 labeled image strip, or None if input is invalid.
    """
    num_views = all_images.shape[0]
    if input_indices is None:
        input_indices = list(range(num_views))

    # Build camera_id → tensor_index mapping
    if view_indices is not None:
        cam_to_tensor_idx = {cam_id: tensor_idx for tensor_idx, cam_id in enumerate(view_indices)}
    else:
        cam_to_tensor_idx = {i: i for i in range(num_views)}

    # Reorder images according to camera_order
    reordered_images = []
    for cam_idx in camera_order:
        tensor_idx = cam_to_tensor_idx.get(cam_idx)
        if tensor_idx is not None and tensor_idx < num_views:
            img = all_images[tensor_idx, :3, ...]  # [C, H, W]
            img = rearrange(img, "c h w -> h w c")
            img = (img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            is_input = cam_idx in input_indices
            reordered_images.append((cam_idx, img, is_input))

    if not reordered_images:
        return None

    # Calculate per-image dimensions
    num_cams = len(reordered_images)
    per_img_w = (target_w - border * (num_cams + 1)) // num_cams
    label_space = 25
    per_img_h = target_h - border * 2 - label_space

    # Create strip with light gray background
    strip = np.ones((target_h, target_w, 3), dtype=np.uint8) * 200
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (cam_idx, img, is_input) in enumerate(reordered_images):
        x_start = border + i * (per_img_w + border)

        # Resize maintaining aspect ratio
        orig_h, orig_w = img.shape[:2]
        scale = min(per_img_w / orig_w, per_img_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Color-coded border: green=input, blue=predicted
        border_color = (0, 200, 0) if is_input else (200, 100, 0)
        cv2.rectangle(resized, (0, 0), (new_w - 1, new_h - 1), border_color, 2)

        # Center in slot
        y_offset = border + (per_img_h - new_h) // 2
        x_offset = x_start + (per_img_w - new_w) // 2
        strip[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # Camera label
        label = f"{cam_idx} (In)" if is_input else f"{cam_idx} (Pred)"
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.putText(strip, label, (x_start + (per_img_w - tw) // 2, border + per_img_h + 16),
                    font, 0.5, (0, 0, 0), 1)

    return strip
