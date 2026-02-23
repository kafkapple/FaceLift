#!/usr/bin/env python3
"""Generate MVDiffusion training data for GS-LRM domain adaptation.

For each frame in the train/val set, run MVDiff inference (view 0 input)
to generate 6 views, then save in GS-LRM dataset format.

Output structure:
    M5_mvdiff/{frame_id}/
    ├── opencv_cameras.json    (copied from GT)
    └── images/
        ├── cam_000.png        (MVDiff view 0)
        ├── cam_001.png        (MVDiff view 1)
        └── ...

Usage:
    CUDA_VISIBLE_DEVICES=6 python generate_mvdiff_train_data.py \
        --mvdiff-checkpoint /node_data/joon/.../mouse_M5t2_randref_sparse/checkpoint-20000 \
        --data-txt ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_train.txt \
        --gt-dir ~/data/preprocessed/FaceLift_mouse/M5 \
        --output-dir ~/data/preprocessed/FaceLift_mouse/M5_mvdiff \
        --guidance-scale 3.0
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as TF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvdiff-checkpoint", type=str, required=True,
                        help="MVDiff checkpoint dir (e.g. .../checkpoint-20000)")
    parser.add_argument("--data-txt", type=str, required=True,
                        help="Train/val txt file listing frame directories")
    parser.add_argument("--gt-dir", type=str, required=True,
                        help="GT data root (for copying opencv_cameras.json)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for MVDiff-generated dataset")
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-view", type=int, default=0,
                        help="Which GT view to use as MVDiff input")
    parser.add_argument("--batch-start", type=int, default=0,
                        help="Start index (for resuming)")
    parser.add_argument("--batch-end", type=int, default=-1,
                        help="End index (-1 for all)")
    args = parser.parse_args()

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from mouse_extensions.inference.mvdiffusion_pipeline import MVDiffusionInference

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = Path(args.gt_dir)

    # Read frame list
    with open(args.data_txt) as f:
        frame_dirs = [line.strip() for line in f if line.strip()]

    total = len(frame_dirs)
    start = args.batch_start
    end = args.batch_end if args.batch_end > 0 else total
    frame_dirs = frame_dirs[start:end]
    print(f"Processing {len(frame_dirs)} frames (index {start}-{end-1} of {total})")

    # Load MVDiff model
    print(f"Loading MVDiff from {args.mvdiff_checkpoint}...")
    mvdiff = MVDiffusionInference(
        checkpoint_path=args.mvdiff_checkpoint,
        device="cuda",
        dtype=torch.float16,
    )
    mvdiff.load_prompt_embeds(
        "mvdiffusion/data/mouse_prompt_embeds_6view_1024/clr_embeds.pt"
    )
    print("MVDiff loaded.")

    # Generate views
    t0 = time.time()
    txt_lines = []

    for i, frame_path in enumerate(frame_dirs):
        frame_path = Path(frame_path).expanduser()
        frame_id = frame_path.name  # e.g., "000000"

        out_frame = output_dir / frame_id
        out_images = out_frame / "images"

        # Skip if already generated
        if (out_images / "cam_005.png").exists():
            txt_lines.append(str(out_frame))
            if i % 100 == 0:
                print(f"[{i+start}/{total}] {frame_id} already exists, skipping")
            continue

        out_images.mkdir(parents=True, exist_ok=True)

        # Copy camera json from GT
        gt_cam_json = frame_path / "opencv_cameras.json"
        if not gt_cam_json.exists():
            # Try gt_dir fallback
            gt_cam_json = gt_dir / frame_id / "opencv_cameras.json"
        if gt_cam_json.exists():
            shutil.copy2(str(gt_cam_json), str(out_frame / "opencv_cameras.json"))

        # Load input image (view 0)
        input_path = frame_path / "images" / f"cam_{args.input_view:03d}.png"
        if not input_path.exists():
            print(f"  WARNING: {input_path} not found, skipping")
            continue

        # Generate 6 views
        with torch.no_grad():
            views = mvdiff.generate_views(
                input_image=str(input_path),
                image_size=512,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
            )

        # Save views as cam_XXX.png (RGB, no alpha)
        for v in range(views.shape[0]):
            view_img = TF.ToPILImage()(views[v].cpu().clamp(0, 1))
            view_img.save(str(out_images / f"cam_{v:03d}.png"))

        txt_lines.append(str(out_frame))

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(frame_dirs) - i - 1) / rate if rate > 0 else 0
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[{i+start+1}/{total}] {frame_id} done. "
                  f"{rate:.1f} frames/s, ETA {eta/60:.0f}min")

    # Write txt file for GS-LRM dataset
    split_name = Path(args.data_txt).stem  # e.g., "data_mouse_t2_train"
    txt_path = output_dir / f"{split_name}_mvdiff.txt"
    with open(txt_path, "w") as f:
        for line in txt_lines:
            f.write(line + "\n")

    elapsed = time.time() - t0
    print(f"\nDone! {len(txt_lines)} frames in {elapsed/60:.1f} min")
    print(f"Dataset txt: {txt_path}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
