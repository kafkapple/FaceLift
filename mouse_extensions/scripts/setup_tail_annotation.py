"""
Setup script for tail annotation refinement.

Copies kp-guided masks (body only) into the annotation directory
so that mask_annotator.py can load and refine them (add tail).

Usage:
    conda activate sdannce
    cd /home/joon/dev/sdannce-poc

    # Test run (50 frames, cameras 1-3)
    python /home/joon/dev/FaceLift/mouse_extensions/scripts/setup_tail_annotation.py \
        --session /home/joon/data/sdannce/rat/dataverse/SCN2A_WK1_2022_09_16_M1 \
        --kp-masks sam2_masks_test \
        --cameras 1,2,3 \
        --keyframes 0,500,1000,1500,2000

    # Full RAT2 (after running kp_sam2_lone.py with step=30)
    python /home/joon/dev/FaceLift/mouse_extensions/scripts/setup_tail_annotation.py \
        --session /home/joon/data/sdannce/rat/dataverse/SCN2A_WK1_2022_09_16_M1 \
        --kp-masks sam2_masks_rat2 \
        --cameras 1,2,3,4,5,6 \
        --keyframes 0,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000,75000,80000,85000

    # Then launch viewer:
    CUDA_VISIBLE_DEVICES=5 python viewers/mask_annotator.py \
        --session /home/joon/data/sdannce/rat/dataverse/SCN2A_WK1_2022_09_16_M1 \
        --port 8770

    # After annotation, propagate:
    CUDA_VISIBLE_DEVICES=5 python segmentation/sam2_propagate.py \
        --session /home/joon/data/sdannce/rat/dataverse/SCN2A_WK1_2022_09_16_M1 \
        --cameras 1,2,3 \
        --mode sparse --step 30 \
        --model base_plus
"""

import argparse
import os
import shutil
import numpy as np


def setup_annotations(session_dir, kp_masks_subdir, cameras, keyframes):
    """Copy kp-guided masks for selected keyframes into annotations dir."""
    kp_dir = os.path.join(session_dir, kp_masks_subdir)
    if not os.path.isdir(kp_dir):
        raise FileNotFoundError(f"KP mask dir not found: {kp_dir}")

    total_copied = 0
    total_missing = 0

    for cam in cameras:
        src_cam = os.path.join(kp_dir, f"Camera{cam}")
        dst_cam = os.path.join(session_dir, "masks", f"Camera{cam}", "annotations")
        os.makedirs(dst_cam, exist_ok=True)

        for fi in keyframes:
            # kp_sam2_lone saves as mask_{frame:06d}.npz
            src_name = f"mask_{fi:06d}.npz"
            # mask_annotator saves as mask_{frame:06d}.npz — use same prefix
            # so that save overwrites the same file (ann_frame_ has higher load priority)
            dst_name = f"mask_{fi:06d}.npz"

            src_path = os.path.join(src_cam, src_name)
            dst_path = os.path.join(dst_cam, dst_name)

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                total_copied += 1
            else:
                print(f"  SKIP (not found): Camera{cam} frame {fi}")
                total_missing += 1

        # List what's in the annotation dir now
        ann_files = sorted(f for f in os.listdir(dst_cam) if f.endswith('.npz'))
        print(f"Camera{cam}: {len(ann_files)} annotations ready ({dst_cam})")

    print(f"\nCopied: {total_copied}, Missing: {total_missing}")
    print(f"\nNext steps:")
    print(f"  1. Launch viewer:")
    print(f"     CUDA_VISIBLE_DEVICES=5 python viewers/mask_annotator.py \\")
    print(f"         --session {session_dir} --port 8770")
    print(f"  2. SSH tunnel:  ssh -L 8770:localhost:8770 gpu03")
    print(f"  3. Browser:     http://localhost:8770")
    print(f"  4. Navigate to each keyframe, add tail with brush, save")
    print(f"  5. After all done, propagate:")
    cams_str = ",".join(str(c) for c in cameras)
    print(f"     CUDA_VISIBLE_DEVICES=5 python segmentation/sam2_propagate.py \\")
    print(f"         --session {session_dir} --cameras {cams_str} \\")
    print(f"         --mode sparse --step 30 --model base_plus")


def main():
    parser = argparse.ArgumentParser(description="Setup tail annotation keyframes")
    parser.add_argument("--session", required=True, help="Session directory")
    parser.add_argument("--kp-masks", required=True, help="KP mask subdirectory name")
    parser.add_argument("--cameras", required=True, help="Comma-separated camera indices")
    parser.add_argument("--keyframes", required=True, help="Comma-separated frame indices")
    args = parser.parse_args()

    cameras = [int(c) for c in args.cameras.split(",")]
    keyframes = [int(f) for f in args.keyframes.split(",")]

    print(f"Session: {args.session}")
    print(f"KP masks: {args.kp_masks}")
    print(f"Cameras: {cameras}")
    print(f"Keyframes: {keyframes} ({len(keyframes)} frames)")
    print()

    setup_annotations(args.session, args.kp_masks, cameras, keyframes)


if __name__ == "__main__":
    main()
