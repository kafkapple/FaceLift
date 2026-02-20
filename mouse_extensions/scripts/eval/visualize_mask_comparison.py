"""Visualize GT alpha mask vs PS white-BG mask differences.

Compares foreground definitions to explain PS IoU drop (0.82 -> 0.32).
Saves comparison images to experiments/comparison/tier/vis_mask_comparison/
"""
import numpy as np
from pathlib import Path
from PIL import Image
import json


def load_gt_alpha_mask(m5_dir, frame_id, cam_id):
    """Load GT alpha mask from M5 RGBA images."""
    img_path = Path(m5_dir) / frame_id / 'images' / f'{cam_id}.png'
    if not img_path.exists():
        return None
    img = np.array(Image.open(img_path))
    if img.shape[-1] == 4:
        alpha = img[:, :, 3]
        return (alpha > 0).astype(np.uint8)
    return None


def load_ps_whitebg_mask(ps_render, threshold=250):
    """Extract foreground mask from PS white-BG rendering."""
    if ps_render is None:
        return None
    # White background = all channels > threshold
    is_white = np.all(ps_render > threshold, axis=-1)
    return (~is_white).astype(np.uint8)


def load_ps_rendering(zarr_path, frame_idx, view_idx):
    """Load PS rendering from zarr store."""
    try:
        import zarr
        z = zarr.open(zarr_path, mode='r')
        img = np.array(z[frame_idx, view_idx])
        return img
    except Exception as e:
        print(f"  Warning: Could not load zarr: {e}")
        return None


def load_fl_rendering(fl_render_dir, frame_id, view_idx):
    """Load FL E2E rendering."""
    # Try with cam_000 subdir first, then without
    for pattern in [
        Path(fl_render_dir) / frame_id / 'cam_000' / f'render_view_{view_idx:02d}.png',
        Path(fl_render_dir) / frame_id / f'render_view_{view_idx:02d}.png',
    ]:
        if pattern.exists():
            return np.array(Image.open(pattern))[:, :, :3]
    return None


def load_fl_mask(fl_render_dir, frame_id, view_idx):
    """Extract FL foreground mask from RGBA rendering."""
    for pattern in [
        Path(fl_render_dir) / frame_id / 'cam_000' / f'render_view_{view_idx:02d}.png',
        Path(fl_render_dir) / frame_id / f'render_view_{view_idx:02d}.png',
    ]:
        if pattern.exists():
            img = np.array(Image.open(pattern))
            if img.shape[-1] == 4:
                return (img[:, :, 3] > 0).astype(np.uint8)
            # If RGB only, use non-black as foreground
            return (np.any(img[:, :, :3] > 10, axis=-1)).astype(np.uint8)
    return None


def create_mask_overlay(gt_img, mask_a, mask_b, label_a="Mask A", label_b="Mask B"):
    """Create RGB overlay: green=both, red=A only, blue=B only."""
    h, w = mask_a.shape[:2]
    overlay = gt_img[:h, :w, :3].copy() if gt_img is not None else np.ones((h, w, 3), dtype=np.uint8) * 128

    both = (mask_a > 0) & (mask_b > 0)
    a_only = (mask_a > 0) & (mask_b == 0)
    b_only = (mask_a == 0) & (mask_b > 0)

    # Semi-transparent overlay
    alpha = 0.5
    overlay[both] = (overlay[both] * (1-alpha) + np.array([0, 255, 0]) * alpha).astype(np.uint8)
    overlay[a_only] = (overlay[a_only] * (1-alpha) + np.array([255, 0, 0]) * alpha).astype(np.uint8)
    overlay[b_only] = (overlay[b_only] * (1-alpha) + np.array([0, 0, 255]) * alpha).astype(np.uint8)

    return overlay, both.sum(), a_only.sum(), b_only.sum()


def compute_stats(mask_gt, mask_pred):
    """Compute IoU and coverage."""
    intersection = (mask_gt > 0) & (mask_pred > 0)
    union = (mask_gt > 0) | (mask_pred > 0)
    iou = intersection.sum() / max(union.sum(), 1)
    coverage = intersection.sum() / max((mask_gt > 0).sum(), 1)
    fg_ratio = (mask_gt > 0).sum() / mask_gt.size
    return iou, coverage, fg_ratio


def save_comparison_grid(output_path, gt_img, gt_alpha, ps_whitebg, ps_render,
                         fl_render, fl_mask, frame_id, view_idx, cam_id):
    """Save a 2x3 comparison grid."""
    from PIL import ImageDraw, ImageFont

    h, w = 512, 512

    # Prepare images
    panels = []
    labels = []

    # Row 1: GT image, GT alpha mask, PS white-BG mask
    if gt_img is not None:
        panels.append(Image.fromarray(gt_img[:h, :w, :3]))
        labels.append(f"GT ({frame_id}/{cam_id})")
    else:
        panels.append(Image.new('RGB', (w, h), (128, 128, 128)))
        labels.append("GT (missing)")

    if gt_alpha is not None:
        mask_vis = np.stack([gt_alpha*255]*3, axis=-1).astype(np.uint8)
        fg_ratio_alpha = (gt_alpha > 0).sum() / gt_alpha.size * 100
        panels.append(Image.fromarray(mask_vis[:h, :w]))
        labels.append(f"GT Alpha (fg={fg_ratio_alpha:.1f}%)")
    else:
        panels.append(Image.new('RGB', (w, h), (128, 128, 128)))
        labels.append("GT Alpha (missing)")

    if ps_whitebg is not None:
        mask_vis = np.stack([ps_whitebg*255]*3, axis=-1).astype(np.uint8)
        fg_ratio_wb = (ps_whitebg > 0).sum() / ps_whitebg.size * 100
        panels.append(Image.fromarray(mask_vis[:h, :w]))
        labels.append(f"PS White-BG (fg={fg_ratio_wb:.1f}%)")
    else:
        panels.append(Image.new('RGB', (w, h), (128, 128, 128)))
        labels.append("PS White-BG (missing)")

    # Row 2: Mask overlay, FL rendering, PS rendering
    if gt_alpha is not None and ps_whitebg is not None:
        overlay, both, a_only, b_only = create_mask_overlay(
            gt_img, gt_alpha, ps_whitebg, "GT Alpha", "PS White-BG"
        )
        iou_alpha_wb = compute_stats(gt_alpha, ps_whitebg)[0]
        panels.append(Image.fromarray(overlay[:h, :w]))
        labels.append(f"Overlay G=both R=GTonly B=PSonly IoU={iou_alpha_wb:.3f}")
    else:
        panels.append(Image.new('RGB', (w, h), (128, 128, 128)))
        labels.append("Overlay (missing)")

    if fl_render is not None:
        fl_img = fl_render[:h, :w, :3]
        panels.append(Image.fromarray(fl_img))
        if fl_mask is not None and gt_alpha is not None:
            iou_fl = compute_stats(gt_alpha, fl_mask)[0]
            labels.append(f"FL E2E (IoU={iou_fl:.3f})")
        else:
            labels.append("FL E2E")
    else:
        panels.append(Image.new('RGB', (w, h), (128, 128, 128)))
        labels.append("FL E2E (missing)")

    if ps_render is not None:
        ps_img = ps_render[:h, :w, :3]
        panels.append(Image.fromarray(ps_img))
        if ps_whitebg is not None and gt_alpha is not None:
            iou_ps_gt = compute_stats(gt_alpha, ps_whitebg)[0]
            labels.append(f"PS Render (IoU_gtmask={iou_ps_gt:.3f})")
        else:
            labels.append("PS Render")
    else:
        panels.append(Image.new('RGB', (w, h), (128, 128, 128)))
        labels.append("PS Render (missing)")

    # Assemble grid: 2 rows x 3 cols
    label_h = 25
    grid_w = w * 3 + 4
    grid_h = (h + label_h) * 2 + 4
    grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))

    for i, (panel, label) in enumerate(zip(panels, labels)):
        row, col = i // 3, i % 3
        x = col * (w + 2)
        y = row * (h + label_h + 2)

        # Draw label
        draw = ImageDraw.Draw(grid)
        draw.text((x + 5, y + 2), label, fill=(0, 0, 0))

        # Paste panel
        panel_resized = panel.resize((w, h))
        grid.paste(panel_resized, (x, y + label_h))

    grid.save(output_path)
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--m5_dir', default='/home/joon/data/preprocessed/FaceLift_mouse/M5')
    parser.add_argument('--ps_zarr', default='/home/joon/dev/pose-splatter/data/preprocessed/markerless_mouse_1_nerf/fj5_ds2/images/images.zarr')
    parser.add_argument('--fl_render_dir', default='/home/joon/dev/FaceLift/outputs/phase3_e2e/h5_e2e/samples')
    parser.add_argument('--output_dir', default='/home/joon/dev/FaceLift/experiments/comparison/tier/vis_mask_comparison')
    parser.add_argument('--frames', nargs='+', default=['003240', '003300', '003400', '003500', '003550'])
    parser.add_argument('--views', nargs='+', type=int, default=[1, 3, 5])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Camera mapping: view index -> M5 camera name
    cam_map = {0: 'cam_01', 1: 'cam_02', 2: 'cam_03', 3: 'cam_04', 4: 'cam_05', 5: 'cam_06'}

    # Try to load PS zarr
    ps_zarr = None
    try:
        import zarr
        ps_zarr = zarr.open(args.ps_zarr, mode='r')
        print(f"[Mask Comparison] PS zarr loaded: {ps_zarr.shape}")
    except Exception as e:
        print(f"[Mask Comparison] Warning: Could not load PS zarr: {e}")

    # Summary stats
    summary = []

    for frame_id in args.frames:
        frame_idx = int(frame_id)

        for view_idx in args.views:
            cam_id = cam_map.get(view_idx, f'cam_{view_idx+1:02d}')
            print(f"\n[{frame_id}/view_{view_idx}] Processing...")

            # Load GT image + alpha
            gt_path = Path(args.m5_dir) / frame_id / 'images' / f'{cam_id}.png'
            gt_img = None
            gt_alpha = None
            if gt_path.exists():
                gt_full = np.array(Image.open(gt_path))
                gt_img = gt_full[:, :, :3]
                if gt_full.shape[-1] == 4:
                    gt_alpha = (gt_full[:, :, 3] > 0).astype(np.uint8)
                    print(f"  GT alpha: fg_ratio={gt_alpha.mean()*100:.2f}%")

            # Load PS rendering from zarr
            ps_render = None
            ps_whitebg = None
            if ps_zarr is not None:
                try:
                    ps_render = np.array(ps_zarr[frame_idx, view_idx])
                    # Center crop 576->512
                    if ps_render.shape[1] > 512:
                        margin = (ps_render.shape[1] - 512) // 2
                        ps_render = ps_render[:, margin:margin+512]
                    ps_whitebg = load_ps_whitebg_mask(ps_render, threshold=250)
                    if ps_whitebg is not None:
                        print(f"  PS white-BG: fg_ratio={ps_whitebg.mean()*100:.2f}%")
                except Exception as e:
                    print(f"  Warning: PS zarr load failed: {e}")

            # Load FL E2E rendering
            fl_render = load_fl_rendering(args.fl_render_dir, frame_id, view_idx + 1)
            fl_mask = load_fl_mask(args.fl_render_dir, frame_id, view_idx + 1)
            if fl_render is not None:
                print(f"  FL render loaded")

            # Compute comparison stats
            if gt_alpha is not None and ps_whitebg is not None:
                iou_wb, cov_wb, _ = compute_stats(gt_alpha, ps_whitebg)
                print(f"  PS vs GT_alpha: IoU={iou_wb:.3f}, Coverage={cov_wb:.3f}")

                if fl_mask is not None:
                    iou_fl, cov_fl, _ = compute_stats(gt_alpha, fl_mask)
                    print(f"  FL vs GT_alpha: IoU={iou_fl:.3f}, Coverage={cov_fl:.3f}")
                    summary.append({
                        'frame': frame_id, 'view': view_idx,
                        'gt_fg%': f"{gt_alpha.mean()*100:.2f}",
                        'ps_wb_fg%': f"{ps_whitebg.mean()*100:.2f}",
                        'ps_iou_gt': f"{iou_wb:.3f}",
                        'fl_iou_gt': f"{iou_fl:.3f}" if fl_mask is not None else 'N/A',
                    })

            # Save grid
            out_path = output_dir / f'{frame_id}_view{view_idx}.png'
            try:
                save_comparison_grid(
                    str(out_path), gt_img, gt_alpha, ps_whitebg, ps_render,
                    fl_render, fl_mask, frame_id, view_idx, cam_id
                )
                print(f"  Saved: {out_path}")
            except Exception as e:
                print(f"  Warning: Grid save failed: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("MASK COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Frame':>8} {'View':>5} {'GT_fg%':>8} {'PS_wb%':>8} {'PS_IoU':>8} {'FL_IoU':>8}")
    print("-" * 50)
    for s in summary:
        print(f"{s['frame']:>8} {s['view']:>5} {s['gt_fg%']:>8} {s['ps_wb_fg%']:>8} {s['ps_iou_gt']:>8} {s['fl_iou_gt']:>8}")

    # Save summary JSON
    summary_path = output_dir / 'mask_comparison_summary.json'
    with open(str(summary_path), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == '__main__':
    main()
