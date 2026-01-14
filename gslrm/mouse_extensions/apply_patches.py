"""
Apply Patches to Original FaceLift Code

This script applies minimal patches to integrate mouse_extensions
with the original FaceLift repository.

Usage:
    cd /path/to/FaceLift
    python -m gslrm.mouse_extensions.apply_patches [--dry-run]
"""

import os
import sys
import shutil
from pathlib import Path

# Patches to apply
PATCHES = {
    # ==========================================================================
    # 1. gaussians_renderer.py - Change rasterizer import
    # ==========================================================================
    "gslrm/model/gaussians_renderer.py": [
        {
            "description": "Change diff_gaussian_rasterization to diff_gauss import",
            "find": '''from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)''',
            "replace": '''# Original: from diff_gaussian_rasterization import ...
# Modified for alpha support
try:
    from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
    ALPHA_RENDERING_ENABLED = True
except ImportError:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    ALPHA_RENDERING_ENABLED = False
    print("[Warning] diff_gauss not available, alpha rendering disabled")'''
        },
    ],

    # ==========================================================================
    # 2. gslrm.py - Add mouse_extensions import at top
    # ==========================================================================
    "gslrm/model/gslrm.py": [
        {
            "description": "Add mouse_extensions import",
            "find": "from einops import rearrange",
            "replace": '''from einops import rearrange

# Mouse extensions (optional)
try:
    from gslrm.mouse_extensions import (
        compute_mask_from_config,
        compute_ghost_metrics,
        create_threshold_comparison,
        MaskType,
    )
    MOUSE_EXTENSIONS_ENABLED = True
except ImportError:
    MOUSE_EXTENSIONS_ENABLED = False'''
        },
    ],

    # ==========================================================================
    # 3. train_gslrm.py - Add logging extensions import
    # ==========================================================================
    "train_gslrm.py": [
        {
            "description": "Add mouse_extensions logging import",
            "find": "import wandb",
            "replace": '''import wandb

# Mouse extensions (optional)
try:
    from gslrm.mouse_extensions.logging_utils import (
        get_experiment_info,
        get_wandb_log_dict,
        get_validation_log_dict,
    )
    MOUSE_LOGGING_ENABLED = True
except ImportError:
    MOUSE_LOGGING_ENABLED = False'''
        },
    ],
}


def apply_patch(file_path: str, patch: dict, dry_run: bool = False) -> bool:
    """Apply a single patch to a file."""
    if not os.path.exists(file_path):
        print(f"  [SKIP] File not found: {file_path}")
        return False

    with open(file_path, "r") as f:
        content = f.read()

    if patch["find"] not in content:
        if patch["replace"] in content:
            print(f"  [SKIP] Already applied: {patch['description']}")
            return True
        print(f"  [WARN] Pattern not found: {patch['description']}")
        return False

    if dry_run:
        print(f"  [DRY] Would apply: {patch['description']}")
        return True

    # Backup original
    backup_path = file_path + ".orig"
    if not os.path.exists(backup_path):
        shutil.copy(file_path, backup_path)
        print(f"  [BACKUP] Created: {backup_path}")

    # Apply patch
    new_content = content.replace(patch["find"], patch["replace"])
    with open(file_path, "w") as f:
        f.write(new_content)

    print(f"  [OK] Applied: {patch['description']}")
    return True


def main():
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print("Mouse Extensions Patch Applicator")
    print("=" * 60)

    if dry_run:
        print("[DRY RUN MODE - No changes will be made]\n")

    # Check we're in FaceLift directory
    if not os.path.exists("gslrm"):
        print("Error: Must run from FaceLift root directory")
        print("Usage: cd /path/to/FaceLift && python -m gslrm.mouse_extensions.apply_patches")
        sys.exit(1)

    success_count = 0
    total_count = 0

    for file_path, patches in PATCHES.items():
        print(f"\n{file_path}:")
        for patch in patches:
            total_count += 1
            if apply_patch(file_path, patch, dry_run):
                success_count += 1

    print("\n" + "=" * 60)
    print(f"Results: {success_count}/{total_count} patches applied")

    if not dry_run:
        print("\nNext steps:")
        print("1. Install diff_gauss: pip install git+https://github.com/slothfulxtx/diff-gaussian-rasterization.git")
        print("2. Copy mouse_extensions/ to gslrm/mouse_extensions/")
        print("3. Run training with mouse config")


if __name__ == "__main__":
    main()
