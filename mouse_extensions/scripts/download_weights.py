#!/usr/bin/env python3
"""
FaceLift Model Weights Downloader

Downloads pretrained weights from HuggingFace if not present locally.
Supports both FaceLift (human faces) and future mouse-specific weights.

Usage:
    python scripts/download_weights.py
    python scripts/download_weights.py --check-only
    python scripts/download_weights.py --model mvdiffusion
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Model configurations
MODEL_CONFIGS = {
    "facelift": {
        "repo_id": "wlyu/OpenFaceLift",
        "description": "FaceLift pretrained weights (human faces)",
        "files": {
            "mvdiffusion": "mvdiffusion/pipeckpts",
            "gslrm": "gslrm/ckpt_0000000000021125.pt",
            "prompt_embeds": "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt",
        }
    },
    # Future: mouse-specific fine-tuned weights
    # "mouse_facelift": {
    #     "repo_id": "your-repo/mouse-facelift",
    #     "description": "Mouse-FaceLift fine-tuned weights",
    #     "files": {...}
    # }
}

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def check_weights_exist(checkpoint_dir: Path, model_name: str = "facelift") -> Dict[str, bool]:
    """
    Check which model weights exist locally.

    Args:
        checkpoint_dir: Directory containing checkpoints
        model_name: Model configuration name

    Returns:
        Dictionary mapping component name to existence status
    """
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}")

    status = {}
    for component, rel_path in config["files"].items():
        # Handle both checkpoints/ and mvdiffusion/data/ paths
        if rel_path.startswith("mvdiffusion/data/"):
            full_path = PROJECT_ROOT / rel_path
        else:
            full_path = checkpoint_dir / rel_path

        if full_path.is_dir():
            # For directories, check if they have contents
            status[component] = full_path.exists() and any(full_path.iterdir())
        else:
            status[component] = full_path.exists()

    return status


def download_weights(
    checkpoint_dir: Path,
    model_name: str = "facelift",
    components: Optional[List[str]] = None,
    force: bool = False
) -> bool:
    """
    Download model weights from HuggingFace.

    Args:
        checkpoint_dir: Directory to save checkpoints
        model_name: Model configuration name
        components: Specific components to download (None = all)
        force: Force re-download even if exists

    Returns:
        True if successful
    """
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}")

    # Check what needs to be downloaded
    if not force:
        status = check_weights_exist(checkpoint_dir, model_name)
        if all(status.values()):
            print(f"All {model_name} weights already exist locally.")
            return True

        missing = [k for k, v in status.items() if not v]
        print(f"Missing components: {missing}")

    repo_id = config["repo_id"]
    print(f"\nDownloading {model_name} weights from HuggingFace: {repo_id}")
    print("This may take several minutes on first run...\n")

    try:
        # Download entire repository to checkpoints directory
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(checkpoint_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        print(f"\n✓ Successfully downloaded {model_name} weights!")
        return True

    except Exception as e:
        print(f"\nERROR downloading weights: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Try: huggingface-cli login")
        print("3. Visit https://huggingface.co/wlyu/OpenFaceLift to check access")
        return False


def print_status(checkpoint_dir: Path, model_name: str = "facelift"):
    """Print detailed status of model weights."""
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        print(f"Unknown model: {model_name}")
        return

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Description: {config['description']}")
    print(f"HuggingFace: https://huggingface.co/{config['repo_id']}")
    print(f"{'='*60}")

    status = check_weights_exist(checkpoint_dir, model_name)

    for component, exists in status.items():
        icon = "✓" if exists else "✗"
        rel_path = config["files"][component]
        print(f"  [{icon}] {component}: {rel_path}")

    all_exist = all(status.values())
    print(f"\nStatus: {'Ready' if all_exist else 'Missing components'}")

    return all_exist


def ensure_weights(
    checkpoint_dir: Optional[Path] = None,
    model_name: str = "facelift",
    auto_download: bool = True
) -> bool:
    """
    Ensure model weights are available, downloading if necessary.

    This is the main function to call from other scripts.

    Args:
        checkpoint_dir: Checkpoint directory (default: PROJECT_ROOT/checkpoints)
        model_name: Model to check/download
        auto_download: Automatically download if missing

    Returns:
        True if weights are available
    """
    if checkpoint_dir is None:
        checkpoint_dir = DEFAULT_CHECKPOINT_DIR

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    status = check_weights_exist(checkpoint_dir, model_name)

    if all(status.values()):
        return True

    if auto_download:
        print(f"Some {model_name} weights missing. Downloading...")
        return download_weights(checkpoint_dir, model_name)
    else:
        missing = [k for k, v in status.items() if not v]
        print(f"Missing {model_name} components: {missing}")
        print(f"Run: python scripts/download_weights.py")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download FaceLift model weights from HuggingFace"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facelift",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to download"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check status, don't download"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists"
    )

    args = parser.parse_args()

    # Print current status
    all_exist = print_status(args.checkpoint_dir, args.model)

    if args.check_only:
        sys.exit(0 if all_exist else 1)

    if not all_exist or args.force:
        success = download_weights(
            args.checkpoint_dir,
            args.model,
            force=args.force
        )
        sys.exit(0 if success else 1)

    print("\nAll weights ready!")


if __name__ == "__main__":
    main()
