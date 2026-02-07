#!/usr/bin/env python3
# DEPRECATED: Use unified pipeline instead:#   python -m mouse_extensions.inference.run --config configs/inference/default.yaml# This script is kept for backward compatibility.
"""CLI entry point for end-to-end Mouse-FaceLift inference.

Two inference paths:
1. GS-LRM only (6-view input):
   - Uses all 6 camera views from a preprocessed sample
   - No MVDiffusion needed
   
2. MVDiffusion + GS-LRM (1-view input):
   - Single image -> MVDiffusion -> 6 views -> GS-LRM -> 3D
   - Can use existing sample view with --input_view_idx

Preprocessing (for raw images not in M5 format):
   - SAM-based mouse detection
   - Background removal (white)
   - Center alignment + coverage normalization
   - Auto-detected or controlled with --skip_preprocess

Usage:
    # === Path 1: GS-LRM only (6-view sample) ===
    python -m mouse_extensions.scripts.inference.run_e2e_inference \
        --sample_dir /path/to/sample/000531 \
        --gslrm_checkpoint M5t_E0_1_facelift \
        --gslrm_config configs/base/gslrm_mouse.yaml \
        --output_dir outputs/gslrm_test

    # === Path 2a: MVDiffusion + GS-LRM (external image with preprocessing) ===
    python -m mouse_extensions.scripts.inference.run_e2e_inference \
        --input_image raw_mouse_photo.jpg \
        --sam_checkpoint checkpoints/sam/sam_vit_h.pth \
        --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse_M5/checkpoint-3500 \
        --gslrm_checkpoint M5t_E0_1_facelift \
        --gslrm_config configs/base/gslrm_mouse.yaml \
        --output_dir outputs/e2e_test

    # === Path 2b: MVDiffusion + GS-LRM (sample view as input, no preprocess needed) ===
    python -m mouse_extensions.scripts.inference.run_e2e_inference \
        --sample_dir /path/to/sample/000531 \
        --input_view_idx 0 \
        --skip_preprocess \
        --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse_M5/checkpoint-3500 \
        --gslrm_checkpoint M5t_E0_1_facelift \
        --gslrm_config configs/base/gslrm_mouse.yaml \
        --output_dir outputs/e2e_from_view

    # === Batch: all samples in a directory ===
    python -m mouse_extensions.scripts.inference.run_e2e_inference \
        --data_dir /path/to/preprocessed/M5 \
        --gslrm_checkpoint M5t_E0_1_facelift \
        --gslrm_config configs/base/gslrm_mouse.yaml \
        --output_dir outputs/batch_test
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch



# Import defaults for simplified CLI
try:
    from .defaults import (
        DEFAULT_CHECKPOINTS, DEFAULT_SPLITS, DEFAULT_DATA_DIR,
        PROMPT_EMBED_PATH, get_checkpoint, get_split
    )
    HAS_DEFAULTS = True
except ImportError:
    HAS_DEFAULTS = False



def interpolate_frames_for_speed(frames, speed_factor):
    """Adjust rotation speed by sampling frames."""
    import numpy as np
    if speed_factor == 1.0:
        return frames
    num_original = frames.shape[0]
    num_target = int(num_original / speed_factor)
    if num_target <= 1:
        return frames[:1]
    indices = np.linspace(0, num_original - 1, num_target)
    return np.stack([frames[int(np.round(idx))] for idx in indices])


def generate_temporal_videos(output_dir: str, fps: int = 10, fixed_angles: list = None, rotation_speed: float = 1.0, grid_views: int = 36):
    """Generate combined temporal videos from per-sample turntables.

    Args:
        output_dir: Directory containing samples/ subfolder with turntable.mp4 files
        fps: Output video FPS
        fixed_angles: List of angles for time_fixed videos (default: [0])
    """
    import cv2
    import numpy as np
    from pathlib import Path

    if fixed_angles is None:
        fixed_angles = [0]

    output_path = Path(output_dir)

    # Look for samples in samples/ subfolder first, then root
    samples_dir = output_path / "samples"
    if samples_dir.exists():
        sample_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    else:
        sample_dirs = sorted([d for d in output_path.iterdir() if d.is_dir() and d.name != "samples"])
    turntable_paths = []
    for sample_dir in sample_dirs:
        # Check for turntable.mp4 directly or in cam_* subdirectory (E2E mode)
        turntable = sample_dir / "turntable.mp4"
        if not turntable.exists():
            cam_dirs = list(sample_dir.glob("cam_*/turntable.mp4"))
            if cam_dirs:
                turntable = cam_dirs[0]
        if turntable.exists():
            turntable_paths.append(turntable)
    
    if len(turntable_paths) < 2:
        print(f"Need at least 2 turntables for temporal videos, found {len(turntable_paths)}")
        return
    
    print(f"\n=== Generating temporal videos from {len(turntable_paths)} samples ===")
    
    # Load all turntables
    all_turntables = []
    for tp in turntable_paths:
        cap = cv2.VideoCapture(str(tp))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if frames:
            all_turntables.append(np.stack(frames))
    
    if not all_turntables:
        print("No turntable frames loaded")
        return
    
    T = len(all_turntables)  # Number of time steps
    V = all_turntables[0].shape[0]  # Number of views per turntable
    H, W = all_turntables[0].shape[1:3]
    
    print(f"Loaded {T} turntables, {V} views each, {H}x{W}")
    
    # Apply rotation speed
    if rotation_speed != 1.0:
        all_turntables = [interpolate_frames_for_speed(t, rotation_speed) for t in all_turntables]
        V = all_turntables[0].shape[0]
        print(f"Applied rotation_speed={rotation_speed}: {V} views after speed adjustment")
    
    from mouse_extensions.utils.video_utils import encode_video_imageio as imageseq2video
    
    # === Output 1: First frame 360° turntable (DISABLED - not useful) ===
    # imageseq2video(all_turntables[0], str(output_path / "turntable_first.mp4"), fps=fps)
    # print("Saved: turntable_first.mp4")  # DISABLED
    
    # === Output 2: Fixed angle, time variation ===
    for angle_idx in fixed_angles:
        angle_idx = angle_idx % V
        fixed_frames = np.stack([t[angle_idx] for t in all_turntables])
        suffix = f"_angle{angle_idx}" if len(fixed_angles) > 1 else ""
        imageseq2video(fixed_frames, str(output_path / f"time_fixed{suffix}.mp4"), fps=fps)
        print(f"Saved: time_fixed{suffix}.mp4 (angle={angle_idx}/{V})")
    
    # === Output 3: Rotating with time ===
    rotating_frames = []
    for t in range(T):
        angle = (t * V // T) % V
        rotating_frames.append(all_turntables[t][angle])
    rotating_frames = np.stack(rotating_frames)
    imageseq2video(rotating_frames, str(output_path / "time_rotating.mp4"), fps=fps)
    print("Saved: time_rotating.mp4")

    # === Output 4: Turntable grid image (first frame) ===
    try:
        first = all_turntables[0]
        grid_v = min(grid_views, V)
        cols = 6
        rows = (grid_v + cols - 1) // cols
        
        # Sample evenly spaced views
        if V > grid_v:
            indices = np.linspace(0, V - 1, grid_v, dtype=int)
            grid_arr = first[indices]
        else:
            grid_arr = first
            grid_v = V
        
        # Pad to fill grid
        total_cells = rows * cols
        if grid_v < total_cells:
            pad_count = total_cells - grid_v
            pad_frame = np.zeros_like(grid_arr[0])
            grid_arr = np.concatenate([grid_arr, np.stack([pad_frame] * pad_count)])
        
        # Create grid image
        grid_rows = []
        for r in range(rows):
            row_frames = [grid_arr[r * cols + c] for c in range(cols)]
            grid_rows.append(np.concatenate(row_frames, axis=1))
        grid_image = np.concatenate(grid_rows, axis=0)
        
        # Save as image
        from PIL import Image
        Image.fromarray(grid_image).save(str(output_path / "turntable_grid.jpg"), quality=95)
        print(f"Saved: turntable_grid.jpg ({rows}x{cols} grid, {grid_v} views)")
    except Exception as e:
        print(f"Grid image skipped: {e}")
    
    print("Temporal videos complete!")

def main():
    parser = argparse.ArgumentParser(
        description="Mouse-FaceLift E2E Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Inference Paths:
  Path 1: GS-LRM only (6-view input)
    --sample_dir OR --data_dir (without --input_view_idx)
    
  Path 2: MVDiffusion + GS-LRM (1-view input)
    --input_image OR (--sample_dir + --input_view_idx)
    Requires: --mvdiffusion_checkpoint

Preprocessing (for raw images):
  By default, preprocessing auto-detects if image is already in M5 format.
  Use --sam_checkpoint to enable SAM-based detection for raw images.
  Use --skip_preprocess to skip preprocessing entirely.

Examples:
  # GS-LRM only
  python -m mouse_extensions.scripts.inference.run_e2e_inference \\
      --sample_dir ~/data/preprocessed/FaceLift_mouse/M5t/000000 \\
      --gslrm_checkpoint M5t_E0_1_facelift \\
      --gslrm_config configs/base/gslrm_mouse.yaml

  # MVDiffusion + GS-LRM with preprocessing (raw image)
  python -m mouse_extensions.scripts.inference.run_e2e_inference \\
      --input_image raw_mouse.jpg \\
      --sam_checkpoint checkpoints/sam/sam_vit_h.pth \\
      --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse_M5/checkpoint-3500 \\
      --gslrm_checkpoint M5t_E0_1_facelift \\
      --gslrm_config configs/base/gslrm_mouse.yaml

  # MVDiffusion + GS-LRM (use view 0 from sample, no preprocess)
  python -m mouse_extensions.scripts.inference.run_e2e_inference \\
      --sample_dir ~/data/preprocessed/FaceLift_mouse/M5t/000000 \\
      --input_view_idx 0 \\
      --skip_preprocess \\
      --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse_M5/checkpoint-3500 \\
      --gslrm_checkpoint M5t_E0_1_facelift \\
      --gslrm_config configs/base/gslrm_mouse.yaml
        """,
    )

    # Input options
    input_group = parser.add_argument_group("Input")
    input_group.add_argument("--input_image", type=str, 
                             help="Single image path (requires MVDiffusion)")
    input_group.add_argument("--sample_dir", type=str, 
                             help="6-view sample directory")
    input_group.add_argument("--data_dir", type=str, 
                             help="Directory containing multiple samples")
    input_group.add_argument("--start_frame", type=int, default=None,
                             help="Start frame index for batch processing")
    input_group.add_argument("--end_frame", type=int, default=None,
                             help="End frame index for batch processing (exclusive)")
    input_group.add_argument("--num_frames", type=int, default=None,
                             help="Number of frames from start (list slicing, overrides start/end_frame)")
    input_group.add_argument("--frame_step", type=int, default=1,
                             help="Frame step for batch processing (default: 1)")
    input_group.add_argument("--split", type=str, default=None,
                             help="Split file path (overrides start/end/step)")
    input_group.add_argument("--input_view_idx", type=int, default=None,
                             help="View index (0-5) to use as input for MVDiffusion. "
                                  "When set with --sample_dir, uses that view for MVDiffusion instead of all 6 views.")

    # Preprocessing options
    preprocess_group = parser.add_argument_group("Preprocessing (for raw images)")
    preprocess_group.add_argument("--sam_checkpoint", type=str, default=None,
                                  help="SAM checkpoint for mouse detection. "
                                       "If None, uses simple resize fallback.")
    preprocess_group.add_argument("--skip_preprocess", action="store_true", default=True,
                                  help="Skip preprocessing entirely (for M5-format images)")
    preprocess_group.add_argument("--save_preprocess_steps", action="store_true",
                                  help="Save visualization of preprocessing steps")

    # Model options
    model_group = parser.add_argument_group("Models")
    model_group.add_argument("--model", type=str, choices=["M5t", "M5t2"], default="M5t",
                             help="Model preset: M5t or M5t2 (auto-sets checkpoints)")
    model_group.add_argument("--gslrm_config", type=str, default="configs/base/gslrm_mouse.yaml",
                             help="GS-LRM YAML config (default: configs/base/gslrm_mouse.yaml)")
    model_group.add_argument("--gslrm_checkpoint", type=str, default=None,
                             help="GS-LRM checkpoint (default: auto from --model)")
    model_group.add_argument("--mvdiffusion_checkpoint", type=str, default=None,
                             help="MVDiffusion checkpoint dir (required for 1-view input)")
    model_group.add_argument("--mvdiffusion_base", type=str, 
                             default="checkpoints/mvdiffusion/pipeckpts",
                             help="Base pipeline for MVDiffusion")
    model_group.add_argument("--prompt_embed_path", type=str, default=None,
                             help="Pre-computed prompt embeddings (default: mouse_prompt_embeds_6view_1024)")
    model_group.add_argument("--prefer_ema", action="store_true", default=True,
                             help="Use EMA UNet weights if available")
    model_group.add_argument("--num_input_views", type=int, default=None,
                             help="Number of input views for GS-LRM (2-6, default: all available)")

    # Camera options
    camera_group = parser.add_argument_group("Camera")
    camera_group.add_argument("--camera_json", type=str, default=None,
                              help="Camera params JSON for E2E mode (default: M5 fixed cameras)")

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output_dir", type=str, default="outputs/e2e_inference",
                              help="Output directory")
    output_group.add_argument("--no_turntable", action="store_true",
                              help="Skip turntable video generation")
    output_group.add_argument("--no_mesh", action="store_true",
                              help="Skip PLY mesh saving")
    output_group.add_argument("--turntable_views", type=int, default=60,
                              help="Number of turntable frames (default: 60)")
    output_group.add_argument("--fps", type=int, default=10,
                              help="Video FPS for temporal outputs (default: 10)")
    output_group.add_argument("--rotation_speed", type=float, default=0.3,
                              help="Rotation speed factor: 0.3=slow, 0.5=half, 1.0=normal (default: 0.3)")
    output_group.add_argument("--grid_views", type=int, default=36,
                              help="Number of views in turntable grid image (default: 36 = 6x6)")

    # Generation options (MVDiffusion)
    gen_group = parser.add_argument_group("Generation (MVDiffusion)")
    gen_group.add_argument("--num_steps", type=int, default=50,
                           help="Diffusion steps (default: 50)")
    gen_group.add_argument("--guidance_scale", type=float, default=3.0,
                           help="Guidance scale (default: 3.0)")
    gen_group.add_argument("--seed", type=int, default=42,
                           help="Random seed (default: 42)")
    gen_group.add_argument("--image_size", type=int, default=512,
                           help="Image resolution (default: 512)")
    gen_group.add_argument("--device", type=str, default="cuda",
                           help="Device (default: cuda)")

    args = parser.parse_args()

    # Apply defaults from --model
    if HAS_DEFAULTS:
        if args.gslrm_checkpoint is None:
            args.gslrm_checkpoint = str(get_checkpoint(args.model, "gslrm"))
            print(f"Using default GS-LRM checkpoint for {args.model}")
        if args.mvdiffusion_checkpoint is None:
            args.mvdiffusion_checkpoint = str(get_checkpoint(args.model, "mvdiffusion"))
            print(f"Using default MVDiffusion checkpoint for {args.model}")
        if args.data_dir and args.data_dir == "auto":
            args.data_dir = str(DEFAULT_DATA_DIR)
    else:
        if args.gslrm_checkpoint is None:
            print("ERROR: --gslrm_checkpoint required (defaults.py not found)")
            return

    # Apply default prompt embed path
    if args.prompt_embed_path is None and HAS_DEFAULTS:
        args.prompt_embed_path = str(PROMPT_EMBED_PATH)
        print(f"Using default prompt embeds: {args.prompt_embed_path}")
    elif args.prompt_embed_path and not Path(args.prompt_embed_path).exists():
        # Try relative to FaceLift root
        facelift_root = Path("/home/joon/dev/FaceLift")
        full_path = facelift_root / "mvdiffusion/data" / args.prompt_embed_path
        if full_path.exists():
            args.prompt_embed_path = str(full_path)

    # Auto-generate output_dir if using default
    if args.output_dir == "outputs/e2e_inference":
        date_suffix = datetime.now().strftime("%y%m%d")
        mode = "e2e" if args.input_view_idx is not None else "gslrm"
        split_name = "test"
        if args.split:
            split_path = Path(args.split)
            fname = split_path.name if split_path.exists() else args.split
            if "_train" in fname:
                split_name = "train"
            elif "_val" in fname:
                split_name = "val"
            elif "_test" in fname:
                split_name = "test"
        name_parts = [mode, args.model, split_name]
        if args.input_view_idx is not None:
            name_parts.append(f"view{args.input_view_idx}")
        if args.num_frames:
            name_parts.append(f"n{args.num_frames}")
        name_parts.append(date_suffix)
        args.output_dir = f"outputs/{'_'.join(name_parts)}"
        print(f"Auto output_dir: {args.output_dir}")

    # Validate input
    if not any([args.input_image, args.sample_dir, args.data_dir]):
        parser.error("Must specify one of: --input_image, --sample_dir, or --data_dir")

    # Check if MVDiffusion is needed
    use_mvdiffusion = (
        args.input_image is not None or 
        (args.sample_dir is not None and args.input_view_idx is not None) or
        (args.data_dir is not None and args.input_view_idx is not None)
    )
    
    if use_mvdiffusion and not args.mvdiffusion_checkpoint:
        parser.error("--mvdiffusion_checkpoint required when using --input_image or --input_view_idx")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    from mouse_extensions.inference.end_to_end import EndToEndPipeline
    from mouse_extensions.inference.gslrm_pipeline import find_sample_dirs

    save_turntable = not args.no_turntable
    save_mesh = not args.no_mesh

    # Save config to output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_dict = {
        "command": "python -m mouse_extensions.scripts.inference.run_e2e_inference",
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
    }
    config_file = output_path / "run_config.json"
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Config saved: {config_file}")

    # Initialize pipeline
    pipeline = EndToEndPipeline(
        gslrm_config=args.gslrm_config,
        gslrm_checkpoint=args.gslrm_checkpoint,
        mvdiffusion_checkpoint=args.mvdiffusion_checkpoint if use_mvdiffusion else None,
        mvdiffusion_base=args.mvdiffusion_base,
        prompt_embed_path=args.prompt_embed_path,
        device=args.device,
        image_size=args.image_size,
        prefer_ema=args.prefer_ema,
        camera_json=args.camera_json,
        sam_checkpoint=args.sam_checkpoint,  # NEW: preprocessing
    )

    # Execute based on input type
    if args.input_image:
        # Path 2a: External image -> MVDiffusion -> GS-LRM
        print(f"\n=== Path 2a: MVDiffusion + GS-LRM (external image) ===")
        print(f"Input: {args.input_image}")
        if args.sam_checkpoint:
            print(f"SAM checkpoint: {args.sam_checkpoint}")
        if args.skip_preprocess:
            print("Preprocessing: SKIPPED")
        out = pipeline.run(
            args.input_image,
            args.output_dir,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            save_turntable=save_turntable,
            save_mesh=save_mesh,
            turntable_views=args.turntable_views,
            skip_preprocess=args.skip_preprocess,  # NEW
            save_preprocess_steps=args.save_preprocess_steps,  # NEW
        )
        print(f"\nDone! Output: {out}")

    elif args.sample_dir and args.input_view_idx is not None:
        # Path 2b: Sample view -> MVDiffusion -> GS-LRM
        print(f"\n=== Path 2b: MVDiffusion + GS-LRM (sample view {args.input_view_idx}) ===")
        print(f"Sample: {args.sample_dir}")
        print(f"Using view {args.input_view_idx} as input")
        
        # Get the specific view image path
        sample_path = Path(args.sample_dir)
        view_image = sample_path / "images" / f"cam_{args.input_view_idx:03d}.png"
        
        if not view_image.exists():
            raise FileNotFoundError(f"View image not found: {view_image}")
        
        out = pipeline.run(
            str(view_image),
            args.output_dir,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            save_turntable=save_turntable,
            save_mesh=save_mesh,
            turntable_views=args.turntable_views,
            skip_preprocess=args.skip_preprocess,  # NEW
            save_preprocess_steps=args.save_preprocess_steps,  # NEW
        )
        print(f"\nDone! Output: {out}")

    elif args.sample_dir:
        # Path 1: 6-view sample -> GS-LRM only
        print(f"\n=== Path 1: GS-LRM only (6-view sample) ===")
        print(f"Sample: {args.sample_dir}")
        out = pipeline.run_from_views(
            args.sample_dir,
            args.output_dir,
            save_turntable=save_turntable,
            save_mesh=save_mesh,
            turntable_views=args.turntable_views,
            num_input_views=args.num_input_views,
        )
        print(f"\nDone! Output: {out}")

    elif args.data_dir:
        # Batch processing - load samples from split file or find all
        if args.split:
            split_path = Path(args.split)
            if not split_path.exists():
                split_path = Path(args.data_dir) / args.split
            if not split_path.exists():
                print(f"ERROR: Split file not found: {args.split}")
                return
            with open(split_path) as f:
                lines = [l.strip().rstrip("/") for l in f if l.strip()]
            samples = []
            for line in lines:
                sample_name = Path(line).name
                sample_path = Path(args.data_dir) / sample_name
                if sample_path.exists():
                    samples.append(str(sample_path))
            samples = sorted(samples, key=lambda x: int(Path(x).name))
        else:
            samples = find_sample_dirs(args.data_dir)
        
        # Apply num_frames (list slicing) - takes priority over start/end_frame
        if args.num_frames is not None:
            samples = samples[:args.num_frames]
        # Apply frame range filtering
        elif args.start_frame is not None or args.end_frame is not None:
            start = args.start_frame if args.start_frame is not None else 0
            end = args.end_frame if args.end_frame is not None else len(samples)
            samples = samples[start:end]
        
        # Apply frame step
        if args.frame_step > 1:
            samples = samples[::args.frame_step]
        
        total = len(samples)
        print(f"\nFound {total} samples to process")
        if args.split:
            print(f"  Split: {args.split}")
        if args.num_frames is not None:
            print(f"  Num frames: {args.num_frames}")
        elif args.start_frame or args.end_frame:
            print(f"  Frame range: [{args.start_frame}:{args.end_frame}]")
        if args.frame_step > 1:
            print(f"  Frame step: {args.frame_step}")
        
        if args.input_view_idx is not None:
            # Batch: MVDiffusion + GS-LRM (using specified view from each sample)
            print(f"=== Batch Path 2b: MVDiffusion (view {args.input_view_idx}) + GS-LRM ===")
            
            from tqdm import tqdm
            for i, sample_dir in enumerate(tqdm(samples, desc="Processing")):
                try:
                    sample_path = Path(sample_dir)
                    view_image = sample_path / "images" / f"cam_{args.input_view_idx:03d}.png"
                    
                    if not view_image.exists():
                        print(f"Skip {sample_dir}: view {args.input_view_idx} not found")
                        continue
                    
                    # Create per-sample output directory (inside samples/ subfolder)
                    sample_name = sample_path.name
                    samples_subdir = Path(args.output_dir) / "samples"
                    samples_subdir.mkdir(parents=True, exist_ok=True)
                    sample_output = samples_subdir / sample_name
                    
                    pipeline.run(
                        str(view_image),
                        str(sample_output),
                        num_steps=args.num_steps,
                        guidance_scale=args.guidance_scale,
                        seed=args.seed,
                        save_turntable=save_turntable,
                        save_mesh=save_mesh,
                        turntable_views=args.turntable_views,
                        skip_preprocess=args.skip_preprocess,  # NEW
                        save_preprocess_steps=args.save_preprocess_steps,  # NEW
                    )
                except Exception as e:
                    print(f"Error processing {sample_dir}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            # Batch: GS-LRM only (using all 6 views)
            print(f"=== Batch Path 1: GS-LRM only (6-view) ===")

            # Create samples/ subfolder for consistency
            samples_subdir = Path(args.output_dir) / "samples"
            samples_subdir.mkdir(parents=True, exist_ok=True)

            from tqdm import tqdm
            for sample_dir in tqdm(samples, desc="Processing"):
                try:
                    pipeline.run_from_views(
                        sample_dir,
                        str(samples_subdir),
                        save_turntable=save_turntable,
                        save_mesh=save_mesh,
                        turntable_views=args.turntable_views,
                        num_input_views=args.num_input_views,
                    )
                except Exception as e:
                    print(f"Error processing {sample_dir}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"\nAll outputs saved to: {args.output_dir}")
        
        # Generate combined temporal videos
        if len(samples) >= 2 and save_turntable:
            generate_temporal_videos(
                args.output_dir, 
                fps=getattr(args, "fps", 10),
                fixed_angles=[0],
                rotation_speed=getattr(args, "rotation_speed", 1.0),
                grid_views=getattr(args, "grid_views", 36),
            )


if __name__ == "__main__":
    main()


