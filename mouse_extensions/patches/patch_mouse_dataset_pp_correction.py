#!/usr/bin/env python3
"""
Patch script to add PP correction to mouse_dataset.py

This script modifies gslrm/data/mouse_dataset.py to support principal point
correction, fixing the cx,cy bug in v12/v13 datasets.

Usage:
    python patch_mouse_dataset_pp_correction.py

Run from the FaceLift project root directory.
"""

import os
import sys
import re
import shutil
from datetime import datetime


def backup_file(filepath: str) -> str:
    """Create a backup of the file"""
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    return backup_path


def apply_patch():
    # Target file
    target_file = "gslrm/data/mouse_dataset.py"

    if not os.path.exists(target_file):
        print(f"Error: {target_file} not found. Run from FaceLift project root.")
        sys.exit(1)

    # Read original file
    with open(target_file, 'r') as f:
        content = f.read()

    # Check if already patched
    if 'pp_correction_method' in content:
        print("File already patched. Skipping.")
        return

    # Create backup
    backup = backup_file(target_file)
    print(f"Backup created: {backup}")

    # Patch 1: Add import after existing imports
    import_patch = '''
# PP correction for v12/v13 dataset bug (2026-01-17)
try:
    from mouse_extensions.scripts.solutions.pp_correction_integration import (
        apply_pp_correction, get_actual_principal_point
    )
    PP_CORRECTION_AVAILABLE = True
except ImportError:
    PP_CORRECTION_AVAILABLE = False
    print("[Warning] PP correction module not found. Using original behavior.")
'''

    # Find location after existing imports
    import_section_end = content.find("class MouseViewDataset")
    if import_section_end == -1:
        print("Error: Could not find MouseViewDataset class")
        sys.exit(1)

    # Insert import patch before class definition
    content = content[:import_section_end] + import_patch + "\n\n" + content[import_section_end:]

    # Patch 2: Add pp_correction_method to __init__
    init_patch_marker = 'self.mask_threshold = mouse_config.get("mask_threshold", 250)'
    init_patch_addition = '''

        # Principal Point correction (2026-01-17)
        # Fixes cx,cy bug in v12/v13 datasets that causes ghosting artifacts
        # Options: "none" (original), "actual_pp" (varying cx,cy), "crop" (recommended)
        self.pp_correction_method = mouse_config.get("pp_correction", "none")
        if self.pp_correction_method != "none" and not PP_CORRECTION_AVAILABLE:
            print(f"[Warning] PP correction '{self.pp_correction_method}' requested but module not available")
            self.pp_correction_method = "none"'''

    content = content.replace(
        init_patch_marker,
        init_patch_marker + init_patch_addition
    )

    # Patch 3: Add print statement for PP correction
    print_marker = 'print(f"[MouseViewDataset] Auto mask generation: {self.auto_generate_mask}, threshold={self.mask_threshold}")'
    print_addition = '''
        if self.pp_correction_method != "none":
            print(f"[MouseViewDataset] PP correction: {self.pp_correction_method} (fixing cx,cy bug)")'''

    content = content.replace(print_marker, print_marker + print_addition)

    # Patch 4: Replace intrinsics extraction with PP-corrected version
    old_intrinsics_code = '''                # Extract and adjust camera intrinsics
                intrinsics = np.array([
                    camera["fx"], camera["fy"], camera["cx"], camera["cy"]
                ])
                intrinsics *= resize_ratio'''

    new_intrinsics_code = '''                # Extract and adjust camera intrinsics
                # PP correction handles the cx,cy bug in v12/v13 datasets
                if PP_CORRECTION_AVAILABLE and self.pp_correction_method != "none":
                    image, intrinsics = apply_pp_correction(
                        image,
                        camera,
                        method=self.pp_correction_method,
                        resize_ratio=resize_ratio,
                        target_size=target_size,
                        bg_color=bg_color_255
                    )
                else:
                    intrinsics = np.array([
                        camera["fx"], camera["fy"], camera["cx"], camera["cy"]
                    ])
                    intrinsics *= resize_ratio'''

    if old_intrinsics_code in content:
        content = content.replace(old_intrinsics_code, new_intrinsics_code)
        print("Intrinsics extraction patched successfully.")
    else:
        print("Warning: Could not find intrinsics extraction code to patch.")
        print("You may need to manually integrate the PP correction.")

    # Write patched file
    with open(target_file, 'w') as f:
        f.write(content)

    print(f"\n✅ Patch applied to {target_file}")
    print("\nTo enable PP correction, add to your config YAML:")
    print("""
mouse:
  pp_correction: "crop"  # Options: "none", "actual_pp", "crop"
""")


def show_diff():
    """Show what the patch will do"""
    print("""
PP Correction Patch Summary:
============================

1. IMPORT: Add pp_correction_integration module import

2. __init__: Add pp_correction_method config option
   - "none": Original behavior (buggy cx,cy=256)
   - "actual_pp": Use actual PP from _transform metadata
   - "crop": Crop image so actual PP becomes centered (recommended)

3. __getitem__: Replace intrinsics extraction with PP-corrected version

Config Example:
---------------
mouse:
  pp_correction: "crop"

Expected Result:
----------------
- Eliminates 11-13° ray direction errors
- Fixes ghosting artifacts in novel view synthesis
- Makes cx=cy=256 mathematically correct (not just forced)
""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--show":
        show_diff()
    else:
        apply_patch()
