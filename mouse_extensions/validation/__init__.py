"""
Validation module for FaceLift mouse experiments.

Usage in gslrm.py:
    from mouse_extensions.validation import run_validation
    return run_validation(self, output_dir, model_results, batch_data, dataset, save_vis)
"""

from .validator import ValidationRunner, run_validation

__all__ = ["ValidationRunner", "run_validation"]
