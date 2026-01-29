"""Checkpoint discovery utilities for inference scripts.

Provides flexible checkpoint finding that supports:
- Exact file paths
- Directory paths (finds best.pt or latest ckpt_*.pt)
- Experiment names (e.g., "M5t_E0_1_facelift")
- Special keywords ("pretrained", "base")
"""

import re
from pathlib import Path
from typing import Optional


def find_checkpoint(
    checkpoint_input: str,
    base_dir: str = "checkpoints/gslrm",
    verbose: bool = True,
) -> str:
    """
    Flexibly find checkpoint file.

    Args:
        checkpoint_input: Can be:
            - Full path to .pt file
            - Directory containing .pt files
            - Dataset/experiment name (e.g., "M5t_E0_1_facelift")
            - "pretrained" or "base" for original checkpoint
        base_dir: Base directory for checkpoint search
        verbose: Print status messages

    Returns:
        Path to the checkpoint file
    """
    base_path = Path(base_dir)
    input_path = Path(checkpoint_input)

    def _print(msg):
        if verbose:
            print(msg)

    # Case 1: Exact file path exists
    if input_path.exists() and input_path.is_file():
        _print(f"Using checkpoint: {input_path}")
        return str(input_path)

    # Case 2: "pretrained" or "base" -> original checkpoint
    if checkpoint_input.lower() in ["pretrained", "base", "original"]:
        pretrained = base_path / "ckpt_0000000000021125.pt"
        if pretrained.exists():
            _print(f"Using pretrained checkpoint: {pretrained}")
            return str(pretrained)
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained}")

    # Case 3: Directory path or name
    search_dir = None
    if input_path.exists() and input_path.is_dir():
        search_dir = input_path
    elif (base_path / checkpoint_input).exists():
        search_dir = base_path / checkpoint_input

    if search_dir:
        # Find best checkpoint first (multiple possible names)
        for best_name in ["best.pt", "best_psnr.pt", "best_checkpoint.pt"]:
            best_pt = search_dir / best_name
            if best_pt.exists():
                _print(f"Using best checkpoint: {best_pt}")
                return str(best_pt)

        # Find all ckpt_*.pt files and get the latest
        pt_files = list(search_dir.glob("ckpt_*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No checkpoint files found in {search_dir}")

        def extract_step(p):
            match = re.search(r"ckpt_(\d+)\.pt", p.name)
            return int(match.group(1)) if match else 0

        pt_files.sort(key=extract_step, reverse=True)
        latest = pt_files[0]
        _print(f"Using latest checkpoint: {latest} (step {extract_step(latest)})")
        return str(latest)

    # Case 4: Try as experiment name pattern
    matching_dirs = list(base_path.glob(f"{checkpoint_input}*"))
    if matching_dirs:
        matching_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return find_checkpoint(str(matching_dirs[0]), base_dir, verbose)

    # Case 5: Try with common suffixes
    for suffix in ["", "_facelift", "_mouse"]:
        candidate = base_path / f"{checkpoint_input}{suffix}"
        if candidate.exists():
            return find_checkpoint(str(candidate), base_dir, verbose)

    # List available options
    available = [d.name for d in base_path.iterdir() if d.is_dir()]
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_input}. "
        f"Available: {', '.join(available[:10])}..."
    )
