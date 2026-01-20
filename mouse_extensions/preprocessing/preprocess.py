#!/usr/bin/env python3
"""
Unified Preprocessing Entry Point

All mouse preprocessing through single command with preset selection.

Usage:
    python -m mouse_extensions.preprocessing.preprocess --preset D7.1 \
        --input-dir /path/to/raw --output-dir /path/to/D7_1

    python -m mouse_extensions.preprocessing.preprocess --list
"""

import argparse
import subprocess
import sys
from pathlib import Path

from .presets import PRESETS, RECOMMENDED, DEPRECATED


def main():
    parser = argparse.ArgumentParser(
        description="Unified Mouse Preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  D7.1  - Individual scale (geometrically correct) [RECOMMENDED]
  D7.2  - Average scale (isotropic)
  D7    - Current production (fy forced to 549)

Examples:
    python -m mouse_extensions.preprocessing.preprocess --preset D7.1 \
        --input-dir /home/joon/data/markerless_mouse_1_nerf \
        --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1
"""
    )
    parser.add_argument('--preset', '-p', help='Preprocessing preset (D7, D7.1, D7.2)')
    parser.add_argument('--input-dir', '-i', type=Path, help='Input data directory')
    parser.add_argument('--output-dir', '-o', type=Path, help='Output directory')
    parser.add_argument('--camera-pkl', type=Path, help='Camera pickle file')
    parser.add_argument('--frame-interval', type=int, default=5, help='Frame sampling interval')
    parser.add_argument('--max-samples', type=int, help='Max samples to process')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--list', '-l', action='store_true', help='List available presets')

    args = parser.parse_args()

    if args.list or not args.preset:
        print("\n=== Available Presets ===\n")
        print("RECOMMENDED:")
        for name in RECOMMENDED:
            preset = PRESETS.get(name, {})
            mark = " [BEST]" if preset.get('recommended') else ""
            ray = preset.get('ray_error', 'N/A')
            print(f"  {name:6} - {preset.get('description', '')}{mark}")
            print(f"          Ray error: {ray}")
        print("\nDEPRECATED:")
        for name in DEPRECATED:
            if name in PRESETS:
                print(f"  {name:6} - {PRESETS[name].get('reason', 'Deprecated')}")
        print()
        return

    if not args.input_dir or not args.output_dir:
        parser.error("--input-dir and --output-dir are required")

    # Normalize preset name
    preset_key = args.preset.replace('.', '_').replace('-', '_')
    preset_lookup = {k.replace('.', '_'): k for k in PRESETS.keys()}
    preset_name = preset_lookup.get(preset_key, args.preset)
    preset = PRESETS.get(preset_name)

    if not preset:
        print(f"Error: Unknown preset '{args.preset}'")
        print(f"Available: {', '.join(PRESETS.keys())}")
        sys.exit(1)

    if preset.get('deprecated'):
        print(f"WARNING: Preset '{preset_name}' is deprecated!")
        print(f"Reason: {preset.get('reason', 'N/A')}")
        print("Consider using D7.1 instead.\n")

    paradigm = preset.get('paradigm', 'object_centered')

    if paradigm == 'pp_centered_shift':
        run_d7_preprocessing(args, preset)
    else:
        run_legacy_preprocessing(args, preset)


def run_d7_preprocessing(args, preset):
    """Run D7+ PP-centered shift preprocessing."""
    from mouse_extensions.scripts.preprocess_D7_pp_centered import D7Config, D7Preprocessor, main as d7_main

    scale_mode = preset.get('scale_mode', 'fx_only')
    camera_pkl = args.camera_pkl or args.input_dir / 'new_cam.pkl'

    print(f"=== D7 Preprocessing (scale_mode={scale_mode}) ===")
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Camera: {camera_pkl}")
    print()

    # Build command args
    cmd_args = [
        '--data-dir', str(args.input_dir),
        '--output-dir', str(args.output_dir),
        '--camera-pkl', str(camera_pkl),
        '--frame-interval', str(args.frame_interval),
        '--scale-mode', scale_mode,
        '--val-ratio', str(args.val_ratio),
    ]
    if args.max_samples:
        cmd_args.extend(['--max-samples', str(args.max_samples)])

    # Inject args and run
    sys.argv = ['preprocess_D7_pp_centered'] + cmd_args
    d7_main()


def run_legacy_preprocessing(args, preset):
    """Run legacy object-centered preprocessing (D1-D4)."""
    print(f"Legacy preprocessing not fully integrated.")
    print(f"Use unified_preprocessor.py directly for {preset} preset.")
    sys.exit(1)


if __name__ == "__main__":
    main()
