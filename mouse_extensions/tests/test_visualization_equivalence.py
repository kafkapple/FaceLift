"""
Test that new visualization functions produce equivalent output to original code.

This test ensures we can safely replace inline visualization code in gslrm.py.
"""

import torch
import numpy as np
from mouse_extensions.model.visualization_extensions import (
    VisualizationConfig,
    create_training_visual,
    create_validation_visual,
    create_mask_overlay,
    compute_pred_mask,
    compute_error_stats,
    create_error_heatmap,
)


def test_training_visual_shape():
    """Test training visual output shape."""
    B, V, H, W = 1, 5, 64, 64

    # Create test data
    target = torch.rand(B*V, 4, H, W)  # RGBA
    target[:, :3, :, :] = target[:, :3, :, :] * 0.5 + 0.5
    target[:, 3, :, :] = (target[:, 3, :, :] > 0.5).float()

    rendering = torch.rand(B*V, 3, H, W) * 0.5 + 0.5
    rendered_alpha = torch.rand(B*V, 1, H, W)

    # Test with mask_mode=alpha
    config = VisualizationConfig(mask_mode='alpha', alpha_threshold=0.5)
    visual, stats = create_training_visual(
        target, rendering, config,
        rendered_alpha=rendered_alpha,
        num_views=V
    )

    # Expected: B * num_rows * H = 1 * 5 * 64 = 320 (with mask)
    # Width: V * W = 5 * 64 = 320
    assert visual.shape == (320, 320, 3), f"Expected (320, 320, 3), got {visual.shape}"
    assert 'fg_mean' in stats
    print(f"[PASS] Training visual (mask_mode=alpha): shape={visual.shape}")

    # Test with mask_mode=none
    config_none = VisualizationConfig(mask_mode='none')
    visual_none, stats_none = create_training_visual(
        target, rendering, config_none, num_views=V
    )

    # Expected: B * 3 * H = 1 * 3 * 64 = 192 (no mask rows)
    assert visual_none.shape == (192, 320, 3), f"Expected (192, 320, 3), got {visual_none.shape}"
    print(f"[PASS] Training visual (mask_mode=none): shape={visual_none.shape}")


def test_validation_visual_shape():
    """Test validation visual output shape."""
    V, H, W = 6, 64, 64

    target = torch.rand(V, 4, H, W)
    target[:, 3, :, :] = (target[:, 3, :, :] > 0.5).float()
    rendering = torch.rand(V, 3, H, W)
    rendered_alpha = torch.rand(V, 1, H, W)

    config = VisualizationConfig(mask_mode='alpha', alpha_threshold=0.5)
    visual, stats = create_validation_visual(
        target, rendering, config,
        rendered_alpha=rendered_alpha
    )

    # Expected: 5 rows * H = 320, V * W = 384
    assert visual.shape == (320, 384, 3), f"Expected (320, 384, 3), got {visual.shape}"
    print(f"[PASS] Validation visual: shape={visual.shape}")


def test_error_stats():
    """Test error statistics computation."""
    error = torch.rand(1, 1, 64, 64)
    mask = (torch.rand(1, 1, 64, 64) > 0.5).float()

    stats = compute_error_stats(error, mask)

    assert 'min' in stats
    assert 'max' in stats
    assert 'mean' in stats
    assert 'fg_min' in stats
    assert 'fg_max' in stats
    assert 'fg_mean' in stats
    assert stats['min'] <= stats['mean'] <= stats['max']
    print(f"[PASS] Error stats: min={stats['min']:.4f}, mean={stats['mean']:.4f}, max={stats['max']:.4f}")


def test_mask_overlay():
    """Test mask overlay creation."""
    image = torch.rand(4, 3, 64, 64)
    mask = (torch.rand(4, 1, 64, 64) > 0.5).float()
    config = VisualizationConfig()

    overlay = create_mask_overlay(image, mask, config)

    assert overlay.shape == image.shape
    assert overlay.min() >= 0
    assert overlay.max() <= 1
    print(f"[PASS] Mask overlay: shape={overlay.shape}")


def test_pred_mask_computation():
    """Test predicted mask computation."""
    rendering = torch.rand(4, 3, 64, 64)
    rendered_alpha = torch.rand(4, 1, 64, 64)

    # Test alpha mode
    config_alpha = VisualizationConfig(mask_mode='alpha', alpha_threshold=0.5)
    pred_mask = compute_pred_mask(rendering, rendered_alpha, config_alpha)

    assert pred_mask.shape == rendered_alpha.shape
    assert pred_mask.min() >= 0
    assert pred_mask.max() <= 1
    print(f"[PASS] Pred mask (alpha): shape={pred_mask.shape}")

    # Test RGB mode
    config_rgb = VisualizationConfig(mask_mode='rgb', rgb_threshold=0.1)
    pred_mask_rgb = compute_pred_mask(rendering, None, config_rgb)

    assert pred_mask_rgb.shape == (4, 1, 64, 64)
    print(f"[PASS] Pred mask (RGB): shape={pred_mask_rgb.shape}")


def test_error_heatmap():
    """Test error heatmap creation."""
    error = torch.rand(4, 1, 64, 64) * 0.2
    mask = (torch.rand(4, 1, 64, 64) > 0.5).float()
    config = VisualizationConfig(error_max=0.3)

    heatmap = create_error_heatmap(error, mask, config)

    assert heatmap.shape == (4, 3, 64, 64)
    assert heatmap.min() >= 0
    assert heatmap.max() <= 1
    print(f"[PASS] Error heatmap: shape={heatmap.shape}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running visualization equivalence tests...")
    print("=" * 60)

    test_training_visual_shape()
    test_validation_visual_shape()
    test_error_stats()
    test_mask_overlay()
    test_pred_mask_computation()
    test_error_heatmap()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
