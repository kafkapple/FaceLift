"""
Create temporal train/val/test split (Pose-Splatter style).

Pose-Splatter paper (arXiv:2505.18342):
"Consecutive thirds of the video are used for training, validation, and testing."

Split method:
- First 1/3 of frames  -> Training
- Second 1/3 of frames -> Validation
- Last 1/3 of frames   -> Testing

Usage:
    python -m mouse_extensions.preprocessing.create_temporal_split \
        --input M3_3 --output M3_3t --symlink
"""

import argparse
import os
import json
import shutil
from pathlib import Path
from datetime import datetime


def create_temporal_split(
    input_name: str,
    output_name: str,
    base_dir: str = "/home/joon/data/preprocessed/FaceLift_mouse",
    use_symlink: bool = True,
):
    """
    Create temporal 1/3 split from existing dataset.
    
    Pose-Splatter style: consecutive thirds for train/val/test.
    
    Args:
        input_name: Source dataset name (e.g., 'M3_3')
        output_name: Output dataset name (e.g., 'M3_3t')
        base_dir: Base directory for datasets
        use_symlink: Use symlink for samples directory
    """
    input_dir = Path(base_dir) / input_name
    output_dir = Path(base_dir) / output_name
    
    # Check input exists
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_dir}")
    
    # Read all samples from combined list
    all_samples_file = input_dir / "data_mouse_all.txt"
    if not all_samples_file.exists():
        # Try to combine train + val
        train_file = input_dir / "data_mouse_train.txt"
        val_file = input_dir / "data_mouse_val.txt"
        
        if train_file.exists() and val_file.exists():
            with open(train_file) as f:
                train_samples = [l.strip() for l in f if l.strip()]
            with open(val_file) as f:
                val_samples = [l.strip() for l in f if l.strip()]
            all_samples = train_samples + val_samples
            print(f"Combined train ({len(train_samples)}) + val ({len(val_samples)}) = {len(all_samples)}")
        else:
            raise FileNotFoundError(f"No data_mouse_all.txt or train/val files found in {input_dir}")
    else:
        with open(all_samples_file) as f:
            all_samples = [l.strip() for l in f if l.strip()]
    
    # Sort by frame index for temporal ordering
    def get_frame_idx(path: str) -> int:
        """Extract frame index from path like .../sample_0001/..."""
        parts = Path(path).parts
        for p in parts:
            if p.startswith('sample_'):
                return int(p.split('_')[1])
        return 0
    
    all_samples.sort(key=get_frame_idx)
    total = len(all_samples)
    
    # Consecutive thirds split (Pose-Splatter style)
    train_end = total // 3
    val_end = 2 * (total // 3)
    
    train_samples = all_samples[:train_end]
    val_samples = all_samples[train_end:val_end]
    test_samples = all_samples[val_end:]
    
    print(f"\nPose-Splatter Temporal Split (Consecutive 1/3):")
    print(f"  Total samples: {total}")
    print(f"  Train (first 1/3):  {len(train_samples)} (frames 0-{train_end-1})")
    print(f"  Val (second 1/3):   {len(val_samples)} (frames {train_end}-{val_end-1})")
    print(f"  Test (last 1/3):    {len(test_samples)} (frames {val_end}-{total-1})")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write split files
    with open(output_dir / "data_mouse_train.txt", "w") as f:
        f.write("\n".join(train_samples) + "\n")
    
    with open(output_dir / "data_mouse_val.txt", "w") as f:
        f.write("\n".join(val_samples) + "\n")
    
    with open(output_dir / "data_mouse_test.txt", "w") as f:
        f.write("\n".join(test_samples) + "\n")
    
    with open(output_dir / "data_mouse_all.txt", "w") as f:
        f.write("\n".join(all_samples) + "\n")
    
    # Link or copy samples directory
    samples_link = output_dir / "samples"
    if samples_link.exists() or samples_link.is_symlink():
        samples_link.unlink()
    
    if use_symlink:
        samples_src = input_dir / "samples"
        if samples_src.exists():
            os.symlink(samples_src, samples_link)
            print(f"  Symlinked samples -> {samples_src}")
        else:
            print(f"  Warning: samples dir not found at {samples_src}")
    
    # Load source metadata
    src_meta_file = input_dir / "metadata.json"
    if src_meta_file.exists():
        with open(src_meta_file) as f:
            src_meta = json.load(f)
    else:
        src_meta = {}
    
    # Write metadata
    metadata = {
        "version": output_name,
        "source_dataset": input_name,
        "paradigm": src_meta.get("paradigm", "unknown"),
        "total_samples": total,
        "split_method": "temporal_consecutive_thirds",
        "split_reference": "Pose-Splatter (arXiv:2505.18342)",
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_test": len(test_samples),
        "train_ratio": round(len(train_samples) / total, 3),
        "val_ratio": round(len(val_samples) / total, 3),
        "test_ratio": round(len(test_samples) / total, 3),
        "train_frames": f"0-{train_end-1}",
        "val_frames": f"{train_end}-{val_end-1}",
        "test_frames": f"{val_end}-{total-1}",
        "created": datetime.now().isoformat(),
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nCreated: {output_dir}")
    print(f"Metadata: {output_dir / 'metadata.json'}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Create temporal train/val/test split (Pose-Splatter style)"
    )
    parser.add_argument("--input", required=True, help="Source dataset name")
    parser.add_argument("--output", required=True, help="Output dataset name")
    parser.add_argument(
        "--base-dir",
        default="/home/joon/data/preprocessed/FaceLift_mouse",
        help="Base directory",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        default=True,
        help="Use symlink for samples (default: True)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy samples instead of symlink",
    )
    args = parser.parse_args()
    
    use_symlink = not args.copy
    
    create_temporal_split(
        args.input,
        args.output,
        args.base_dir,
        use_symlink,
    )


if __name__ == "__main__":
    main()
