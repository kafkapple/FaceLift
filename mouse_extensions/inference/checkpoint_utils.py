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


def find_mvdiffusion_checkpoint(
    checkpoint_input: str,
    base_dir: str = "checkpoints/mvdiffusion",
    verbose: bool = True,
) -> str:
    """
    Flexibly find MVDiffusion checkpoint directory.

    Args:
        checkpoint_input: Can be:
            - Full path to checkpoint directory (with unet/ subfolder)
            - Experiment name (e.g., "mouse_M5")
            - "mouse_M5/checkpoint-3500" format
            - "latest" to find most recent checkpoint
        base_dir: Base directory for checkpoint search
        verbose: Print status messages

    Returns:
        Path to the checkpoint directory (containing unet/)
    
    Examples:
        find_mvdiffusion_checkpoint("mouse_M5")
        # → checkpoints/mvdiffusion/mouse_M5/checkpoint-3500 (latest)
        
        find_mvdiffusion_checkpoint("mouse_M5/checkpoint-2000")
        # → checkpoints/mvdiffusion/mouse_M5/checkpoint-2000
    """
    base_path = Path(base_dir)
    input_path = Path(checkpoint_input)

    def _print(msg):
        if verbose:
            print(msg)

    def _is_valid_checkpoint(path: Path) -> bool:
        """Check if directory contains unet/ subfolder."""
        return (path / "unet").exists()

    def _find_latest_checkpoint(exp_dir: Path) -> Optional[Path]:
        """Find the latest checkpoint-* directory in experiment folder."""
        checkpoints = list(exp_dir.glob("checkpoint-*"))
        if not checkpoints:
            return None
        
        def extract_step(p):
            match = re.search(r"checkpoint-(\d+)", p.name)
            return int(match.group(1)) if match else 0
        
        checkpoints.sort(key=extract_step, reverse=True)
        return checkpoints[0]

    # Case 1: Exact directory path with unet/
    if input_path.exists() and input_path.is_dir():
        if _is_valid_checkpoint(input_path):
            _print(f"Using checkpoint: {input_path}")
            return str(input_path)
        # Maybe it's an experiment directory, find latest checkpoint
        latest = _find_latest_checkpoint(input_path)
        if latest and _is_valid_checkpoint(latest):
            _print(f"Using latest checkpoint: {latest}")
            return str(latest)

    # Case 2: Relative path like "mouse_M5" or "mouse_M5/checkpoint-3500"
    candidate = base_path / checkpoint_input
    if candidate.exists() and candidate.is_dir():
        if _is_valid_checkpoint(candidate):
            _print(f"Using checkpoint: {candidate}")
            return str(candidate)
        # Find latest checkpoint in this experiment
        latest = _find_latest_checkpoint(candidate)
        if latest and _is_valid_checkpoint(latest):
            _print(f"Using latest checkpoint: {latest}")
            return str(latest)

    # Case 3: Just experiment name, find latest checkpoint
    exp_dir = base_path / checkpoint_input.split("/")[0]
    if exp_dir.exists():
        latest = _find_latest_checkpoint(exp_dir)
        if latest and _is_valid_checkpoint(latest):
            _print(f"Using latest checkpoint: {latest}")
            return str(latest)

    # Case 4: Pattern matching on experiment names
    matching_dirs = list(base_path.glob(f"{checkpoint_input}*"))
    matching_dirs = [d for d in matching_dirs if d.is_dir() and d.name != "pipeckpts"]
    if matching_dirs:
        matching_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest = _find_latest_checkpoint(matching_dirs[0])
        if latest and _is_valid_checkpoint(latest):
            _print(f"Using latest checkpoint: {latest}")
            return str(latest)

    # List available experiments
    available = [d.name for d in base_path.iterdir() 
                 if d.is_dir() and d.name != "pipeckpts"]
    raise FileNotFoundError(
        f"MVDiffusion checkpoint not found: {checkpoint_input}. "
        f"Available experiments: {', '.join(available)}"
    )
