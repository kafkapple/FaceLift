#!/usr/bin/env python3
"""
Apply camera normalization to existing preprocessed dataset.

Usage:
    python apply_normalization.py --input_dir D3_postcorrect --output_dir D3_normalized
"""

import argparse
import json
import shutil
from pathlib import Path
from tqdm import tqdm

from camera_normalizer import normalize_cameras


def process_sample(src_dir: Path, dst_dir: Path, target_fx: float, target_distance: float):
    """Copy sample and apply normalization to cameras."""
    # Copy directory (images stay the same)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    
    # Load and normalize camera file
    cam_file = dst_dir / "opencv_cameras.json"
    with open(cam_file) as f:
        data = json.load(f)
    
    frames = data["frames"]
    normalized_frames, transform_info = normalize_cameras(
        frames, target_fx=target_fx, target_distance=target_distance
    )
    
    # Save normalized data
    output_data = {
        "frames": normalized_frames,
        "_transform": transform_info
    }
    with open(cam_file, "w") as f:
        json.dump(output_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_fx", type=float, default=549.0)
    parser.add_argument("--target_distance", type=float, default=2.7)
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train and val
    for split in ["train", "val"]:
        src_split = args.input_dir / split
        dst_split = args.output_dir / split
        
        if not src_split.exists():
            continue
            
        dst_split.mkdir(exist_ok=True)
        samples = sorted([d for d in src_split.iterdir() if d.is_dir()])
        
        print(f"Processing {split}: {len(samples)} samples")
        for sample_dir in tqdm(samples):
            process_sample(
                sample_dir, 
                dst_split / sample_dir.name,
                args.target_fx,
                args.target_distance
            )
    
    # Copy and update data list files
    for f in args.input_dir.glob("*.txt"):
        content = f.read_text()
        content = content.replace(str(args.input_dir), str(args.output_dir))
        (args.output_dir / f.name).write_text(content)
    
    # Copy metadata
    meta_file = args.input_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        meta["normalized"] = True
        meta["target_fx"] = args.target_fx
        meta["target_distance"] = args.target_distance
        with open(args.output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    print(f"Done! Output: {args.output_dir}")


if __name__ == "__main__":
    main()
