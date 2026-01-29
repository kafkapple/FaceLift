"""
Evaluation module for FaceLift mouse experiments.

Provides centralized metrics computation to prevent the "3-place modification" bug:
- Before: Adding new metric required changes in import + function signature + call site
- After: Just add method to MetricsComputer class

Usage:
    from mouse_extensions.evaluation import MetricsComputer, get_metrics_computer

    # Option 1: Create instance
    metrics = MetricsComputer()
    results = metrics.compute_per_view_metrics(gt, pred, mask)

    # Option 2: Use singleton
    metrics = get_metrics_computer()
    psnr = metrics.compute_psnr(gt, pred)
"""

from .metrics import MetricsComputer, get_metrics_computer

__all__ = ["MetricsComputer", "get_metrics_computer"]
