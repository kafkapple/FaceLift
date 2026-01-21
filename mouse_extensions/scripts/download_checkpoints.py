#!/usr/bin/env python3
"""
Download FaceLift pretrained checkpoints from HuggingFace.

Usage:
    python scripts/download_checkpoints.py

Required:
    - HF_TOKEN in .env file or environment variable
    - pip install huggingface_hub python-dotenv
"""

import os
from pathlib import Path

def main():
    # Load environment variables from .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("python-dotenv not installed, using environment variables only")

    # Get HuggingFace token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not found. Some models may require authentication.")

    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return

    # Repository info
    repo_id = "wlyu/OpenFaceLift"
    local_dir = Path(__file__).parent.parent / "checkpoints"
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading checkpoints from {repo_id}")
    print(f"Saving to: {local_dir}")

    try:
        # Download all checkpoints
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=hf_token,
            # Download specific patterns
            allow_patterns=[
                "gslrm/*",           # GS-LRM checkpoints
                "mvdiffusion/*",     # Multi-view diffusion (optional)
            ],
            ignore_patterns=[
                "*.md",
                "*.txt",
            ]
        )
        print("Download complete!")

        # List downloaded files
        print("\nDownloaded files:")
        for f in local_dir.rglob("*.pt"):
            print(f"  {f.relative_to(local_dir)}")
        for f in local_dir.rglob("*.safetensors"):
            print(f"  {f.relative_to(local_dir)}")

    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nTrying alternative download method...")

        # Try downloading just the GS-LRM checkpoint
        try:
            gslrm_file = hf_hub_download(
                repo_id=repo_id,
                filename="gslrm/ckpt_0000000000021125.pt",
                local_dir=str(local_dir),
                token=hf_token,
            )
            print(f"Downloaded GS-LRM checkpoint: {gslrm_file}")
        except Exception as e2:
            print(f"Failed to download GS-LRM: {e2}")


if __name__ == "__main__":
    main()
