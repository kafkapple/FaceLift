#!/usr/bin/env python3
"""
UV Transplant: fitting OBJ에 textured OBJ의 UV 좌표를 이식.

MAMMAL parametric model은 모든 프레임이 동일한 topology (V=14522, F=28800)를 가지므로,
한 프레임의 UV mapping을 모든 프레임에 적용 가능.

Usage:
    # Single file
    python transplant_uv.py \
        --fitting /path/to/step_2_frame_000000.obj \
        --reference /path/to/mouse_frame0_textured.obj \
        --output /path/to/output.obj

    # Batch (all frames)
    python transplant_uv.py \
        --fitting-dir /path/to/fitting/obj/ \
        --reference /path/to/mouse_frame0_textured.obj \
        --output-dir /path/to/textured_obj/ \
        --max-frames 100
"""

import argparse
from pathlib import Path


def parse_textured_reference(ref_path):
    """Extract UV coords and face lines (with v/vt format) from textured OBJ."""
    vt_lines = []
    face_lines = []
    with open(ref_path) as f:
        for line in f:
            if line.startswith('vt '):
                vt_lines.append(line.rstrip('\n'))
            elif line.startswith('f '):
                face_lines.append(line.rstrip('\n'))
    return vt_lines, face_lines


def transplant_single(fitting_path, vt_lines, face_lines, output_path):
    """Transplant UV to a single fitting OBJ."""
    # Extract vertex positions from fitting OBJ
    v_lines = []
    with open(fitting_path) as f:
        for line in f:
            if line.startswith('v '):
                v_lines.append(line.rstrip('\n'))

    # Write combined OBJ
    with open(output_path, 'w') as f:
        f.write(f"# UV-transplanted from {Path(fitting_path).name}\n")
        f.write(f"# Vertices: {len(v_lines)}, UV: {len(vt_lines)}, Faces: {len(face_lines)}\n")
        for v in v_lines:
            f.write(v + '\n')
        for vt in vt_lines:
            f.write(vt + '\n')
        for face in face_lines:
            f.write(face + '\n')

    return len(v_lines)


def main():
    parser = argparse.ArgumentParser(description="UV Transplant for MAMMAL meshes")
    parser.add_argument("--fitting", type=str, help="Single fitting OBJ path")
    parser.add_argument("--fitting-dir", type=str, help="Directory of fitting OBJs")
    parser.add_argument("--reference", type=str, required=True,
                        help="Textured reference OBJ (with UV)")
    parser.add_argument("--output", type=str, help="Single output path")
    parser.add_argument("--output-dir", type=str, help="Output directory for batch")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--frame-step", type=int, default=1, help="Frame step for batch")
    args = parser.parse_args()

    # Parse reference once
    print(f"Loading UV reference: {args.reference}")
    vt_lines, face_lines = parse_textured_reference(args.reference)
    print(f"  UV coords: {len(vt_lines)}, Faces: {len(face_lines)}")

    if args.fitting:
        # Single file mode
        output = args.output or args.fitting.replace('.obj', '_textured.obj')
        n = transplant_single(args.fitting, vt_lines, face_lines, output)
        print(f"Done: {output} (V={n})")

    elif args.fitting_dir:
        # Batch mode
        fitting_dir = Path(args.fitting_dir)
        output_dir = Path(args.output_dir or str(fitting_dir) + '_textured')
        output_dir.mkdir(parents=True, exist_ok=True)

        all_objs = sorted(fitting_dir.glob("step_2_frame_*.obj"))
        selected = all_objs[::args.frame_step]
        if args.max_frames:
            selected = selected[:args.max_frames]

        print(f"Processing {len(selected)}/{len(all_objs)} frames -> {output_dir}")
        for i, obj_path in enumerate(selected):
            out_path = output_dir / obj_path.name
            transplant_single(str(obj_path), vt_lines, face_lines, str(out_path))
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(selected)}] {obj_path.name}")

        print(f"Complete: {len(selected)} files in {output_dir}")
    else:
        parser.error("Provide --fitting or --fitting-dir")


if __name__ == "__main__":
    main()
