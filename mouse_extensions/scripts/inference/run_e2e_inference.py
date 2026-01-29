#!/usr/bin/env python3
"""CLI entry point for end-to-end Mouse-FaceLift inference.

Two inference paths:
1. GS-LRM only (6-view input):
   - Uses all 6 camera views from a preprocessed sample
   - No MVDiffusion needed
   
2. MVDiffusion + GS-LRM (1-view input):
   - Single image -> MVDiffusion -> 6 views -> GS-LRM -> 3D
   - Can use existing sample view with --input_view_idx

Usage:
    # === Path 1: GS-LRM only (6-view sample) ===
    python -m mouse_extensions.scripts.inference.run_e2e_inference \
        --sample_dir /path/to/sample/000531 \
        --gslrm_checkpoint M5t_E0_1_facelift \
        --gslrm_config configs/base/gslrm_mouse.yaml \
        --output_dir outputs/gslrm_test

    # === Path 2a: MVDiffusion + GS-LRM (external image) ===
    python -m mouse_extensions.scripts.inference.run_e2e_inference \
        --input_image input.png \
        --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse_M5/checkpoint-3500 \
        --gslrm_checkpoint M5t_E0_1_facelift \
        --gslrm_config configs/base/gslrm_mouse.yaml \
        --output_dir outputs/e2e_test

    # === Path 2b: MVDiffusion + GS-LRM (sample view as input) ===
    python -m mouse_extensions.scripts.inference.run_e2e_inference \
        --sample_dir /path/to/sample/000531 \
        --input_view_idx 0 \
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
from pathlib import Path

import torch


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

Examples:
  # GS-LRM only
  python -m mouse_extensions.scripts.inference.run_e2e_inference \\
      --sample_dir /home/joon/data/preprocessed/FaceLift_mouse/M5t/000000 \\
      --gslrm_checkpoint M5t_E0_1_facelift \\
      --gslrm_config configs/base/gslrm_mouse.yaml

  # MVDiffusion + GS-LRM (use view 0 from sample)
  python -m mouse_extensions.scripts.inference.run_e2e_inference \\
      --sample_dir /home/joon/data/preprocessed/FaceLift_mouse/M5t/000000 \\
      --input_view_idx 0 \\
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
    input_group.add_argument("--frame_step", type=int, default=1,
                             help="Frame step for batch processing (default: 1)")
    input_group.add_argument("--input_view_idx", type=int, default=None,
                             help="View index (0-5) to use as input for MVDiffusion. "
                                  "When set with --sample_dir, uses that view for MVDiffusion instead of all 6 views.")

    # Model options
    model_group = parser.add_argument_group("Models")
    model_group.add_argument("--gslrm_config", type=str, default="configs/base/gslrm_mouse.yaml",
                             help="GS-LRM YAML config (default: configs/base/gslrm_mouse.yaml)")
    model_group.add_argument("--gslrm_checkpoint", type=str, required=True,
                             help="GS-LRM checkpoint (path, dir, or experiment name like M5t_E0_1_facelift)")
    model_group.add_argument("--mvdiffusion_checkpoint", type=str, default=None,
                             help="MVDiffusion checkpoint dir (required for 1-view input)")
    model_group.add_argument("--mvdiffusion_base", type=str, 
                             default="checkpoints/mvdiffusion/pipeckpts",
                             help="Base pipeline for MVDiffusion")
    model_group.add_argument("--prompt_embed_path", type=str, default=None,
                             help="Pre-computed prompt embeddings")
    model_group.add_argument("--prefer_ema", action="store_true", default=True,
                             help="Use EMA UNet weights if available")

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
    output_group.add_argument("--turntable_views", type=int, default=120,
                              help="Number of turntable frames (default: 120)")

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
    )

    # Execute based on input type
    if args.input_image:
        # Path 2a: External image -> MVDiffusion -> GS-LRM
        print(f"\n=== Path 2a: MVDiffusion + GS-LRM (external image) ===")
        print(f"Input: {args.input_image}")
        out = pipeline.run(
            args.input_image,
            args.output_dir,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            save_turntable=save_turntable,
            save_mesh=save_mesh,
            turntable_views=args.turntable_views,
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
        )
        print(f"\nDone! Output: {out}")

    elif args.data_dir:
        # Batch processing
        samples = find_sample_dirs(args.data_dir)
        
        # Apply frame range filtering
        if args.start_frame is not None or args.end_frame is not None:
            start = args.start_frame if args.start_frame is not None else 0
            end = args.end_frame if args.end_frame is not None else len(samples)
            samples = samples[start:end]
        
        # Apply frame step
        if args.frame_step > 1:
            samples = samples[::args.frame_step]
        
        total = len(samples)
        print(f"\nFound {total} samples to process")
        if args.start_frame or args.end_frame:
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
                    
                    # Create per-sample output directory
                    sample_name = sample_path.name
                    sample_output = Path(args.output_dir) / sample_name
                    
                    pipeline.run(
                        str(view_image),
                        str(sample_output),
                        num_steps=args.num_steps,
                        guidance_scale=args.guidance_scale,
                        seed=args.seed,
                        save_turntable=save_turntable,
                        save_mesh=save_mesh,
                        turntable_views=args.turntable_views,
                    )
                except Exception as e:
                    print(f"Error processing {sample_dir}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            # Batch: GS-LRM only (using all 6 views)
            print(f"=== Batch Path 1: GS-LRM only (6-view) ===")
            
            from tqdm import tqdm
            for sample_dir in tqdm(samples, desc="Processing"):
                try:
                    pipeline.run_from_views(
                        sample_dir,
                        args.output_dir,
                        save_turntable=save_turntable,
                        save_mesh=save_mesh,
                        turntable_views=args.turntable_views,
                    )
                except Exception as e:
                    print(f"Error processing {sample_dir}: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
