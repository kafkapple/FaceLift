#!/usr/bin/env python3
"""
D3 Post-correction: Apply PP shift to existing D3 dataset.

기존 D3 (triangulation + CORRECT PP)에서 cx, cy를 256으로 보정.
이전에 효과가 있었던 방식.

Usage:
    python postcorrect_d3.py --input_dir D3 --output_dir D3_postcorrect
"""

import argparse
import json
import shutil
from pathlib import Path
from tqdm import tqdm


def postcorrect_sample(src_dir: Path, dst_dir: Path, shift_cx: float, shift_cy: float):
    """Copy sample and apply PP correction."""
    # Copy directory
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    
    # Modify camera file
    cam_file = dst_dir / 'opencv_cameras.json'
    with open(cam_file) as f:
        data = json.load(f)
    
    for frame in data['frames']:
        frame['cx'] = 256.0  # Force to center
        frame['cy'] = 256.0
    
    with open(cam_file, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train and val
    for split in ['train', 'val']:
        src_split = args.input_dir / split
        dst_split = args.output_dir / split
        dst_split.mkdir(exist_ok=True)
        
        samples = sorted(list(src_split.iterdir()))
        print(f'Processing {split}: {len(samples)} samples')
        
        for sample_dir in tqdm(samples):
            if sample_dir.is_dir():
                postcorrect_sample(sample_dir, dst_split / sample_dir.name, 0, 0)
    
    # Copy data lists
    for f in args.input_dir.glob('*.txt'):
        # Update paths in txt file
        content = f.read_text()
        content = content.replace(str(args.input_dir), str(args.output_dir))
        (args.output_dir / f.name).write_text(content)
    
    print(f'Done! Output: {args.output_dir}')


if __name__ == '__main__':
    main()
