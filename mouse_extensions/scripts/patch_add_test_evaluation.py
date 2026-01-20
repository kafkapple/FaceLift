#!/usr/bin/env python3
"""
Patch to add test evaluation to train_gslrm.py
==============================================

This script patches train_gslrm.py to add:
1. Test dataloader initialization
2. run_test() function (similar to run_validation)
3. Test evaluation call at end of training
4. Test metrics logging to wandb

Usage:
    python patch_add_test_evaluation.py /path/to/train_gslrm.py

Created: 2026-01-19
"""

import sys
import re
from pathlib import Path


# Code to add for test evaluation
TEST_IMPORT_CODE = '''
# Test evaluation extension
try:
    from mouse_extensions.scripts.test_evaluation_extension import (
        create_test_dataloader,
        run_test_evaluation
    )
    TEST_EVALUATION_AVAILABLE = True
except ImportError:
    TEST_EVALUATION_AVAILABLE = False
'''

TEST_DATALOADER_INIT_CODE = '''
        # Initialize test dataloader (for temporal split datasets)
        self.test_dataloader = None
'''

TEST_SETUP_CODE = '''
    def setup_test_dataloader(self):
        """Setup test dataloader if test set exists."""
        if not TEST_EVALUATION_AVAILABLE:
            return

        self.test_dataloader = create_test_dataloader(
            self.config,
            type(self.dataset),
            batch_size=1,
            num_workers=self.config.training.dataloader.num_workers
        )
'''

RUN_TEST_CODE = '''
    def run_test(self):
        """Run final test evaluation (once at end of training)."""
        if not TEST_EVALUATION_AVAILABLE or self.test_dataloader is None:
            print("[Test] No test dataloader available, skipping test evaluation")
            return

        test_output_dir = self.config.get("validation", {}).get(
            "output_dir", "experiments/validation"
        ).replace("validation", "test")

        return run_test_evaluation(
            self,
            self.test_dataloader,
            output_dir=test_output_dir,
            log_to_wandb=True
        )
'''

TRAIN_END_CODE = '''
        # Run final test evaluation (once at end of training)
        if self.ddp_rank == 0:
            self.run_test()
'''


def patch_train_gslrm(filepath: str) -> str:
    """Patch train_gslrm.py to add test evaluation."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already patched
    if 'TEST_EVALUATION_AVAILABLE' in content:
        print("File already patched!")
        return content

    # 1. Add import after mouse_extensions import
    mouse_import_pattern = r'(try:\s+from mouse_extensions\.utils import.*?MOUSE_LOGGING_AVAILABLE = False)'
    match = re.search(mouse_import_pattern, content, re.DOTALL)
    if match:
        insert_pos = match.end()
        content = content[:insert_pos] + '\n' + TEST_IMPORT_CODE + content[insert_pos:]
        print("Added test evaluation import")
    else:
        print("Warning: Could not find mouse_extensions import")

    # 2. Add test_dataloader initialization in __init__
    init_pattern = r'(self\.val_dataloader = None)'
    content = re.sub(init_pattern, r'\1' + TEST_DATALOADER_INIT_CODE, content)
    print("Added test_dataloader initialization")

    # 3. Add setup_test_dataloader method after setup_dataloaders
    setup_pattern = r'(def setup_dataloaders\(self\):.*?"""Setup data loaders.*?""".*?(?=\n    def \w))'
    match = re.search(setup_pattern, content, re.DOTALL)
    if match:
        insert_pos = match.end()
        content = content[:insert_pos] + '\n' + TEST_SETUP_CODE + content[insert_pos:]
        print("Added setup_test_dataloader method")

    # 4. Add run_test method after run_validation
    val_pattern = r'(def run_validation\(self\):.*?self\.model\.train\(\))'
    match = re.search(val_pattern, content, re.DOTALL)
    if match:
        insert_pos = match.end()
        content = content[:insert_pos] + '\n' + RUN_TEST_CODE + content[insert_pos:]
        print("Added run_test method")

    # 5. Add test evaluation call at end of train()
    train_end_pattern = r'(# Save final checkpoint if needed.*?self\.save_checkpoint_if_needed\(\))'
    match = re.search(train_end_pattern, content, re.DOTALL)
    if match:
        insert_pos = match.end()
        content = content[:insert_pos] + '\n' + TRAIN_END_CODE + content[insert_pos:]
        print("Added test evaluation call at end of training")

    # 6. Add setup_test_dataloader call after setup_dataloaders call
    setup_call_pattern = r'(self\.setup_dataloaders\(\))'
    content = re.sub(setup_call_pattern, r'\1\n        self.setup_test_dataloader()', content)
    print("Added setup_test_dataloader call")

    return content


def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_add_test_evaluation.py /path/to/train_gslrm.py")
        print("\nThis will create a backup and patch the file to add test evaluation.")
        sys.exit(1)

    filepath = Path(sys.argv[1])

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    # Create backup
    backup_path = filepath.with_suffix('.py.bak')
    if not backup_path.exists():
        import shutil
        shutil.copy(filepath, backup_path)
        print(f"Created backup: {backup_path}")

    # Apply patch
    patched_content = patch_train_gslrm(str(filepath))

    # Write patched content
    with open(filepath, 'w') as f:
        f.write(patched_content)

    print(f"\nPatch applied successfully to {filepath}")
    print("\nTo use test evaluation:")
    print("  1. Create temporal split datasets: D7_t, D7_5_t, D7_5b_t")
    print("  2. Train with temporal split dataset (has data_mouse_test.txt)")
    print("  3. Test evaluation runs automatically at end of training")
    print("  4. Check wandb for 'test/' and 'final/' metrics")


if __name__ == "__main__":
    main()
