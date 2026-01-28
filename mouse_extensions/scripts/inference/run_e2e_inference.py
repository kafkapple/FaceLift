#!/usr/bin/env python3
"""CLI entry point for end-to-end Mouse-FaceLift inference.

Usage:
    # Single image → 6 views → 3D (full pipeline)
    python -m mouse_extensions.scripts.inference.run_e2e_inference \
        --input_image input.png \
        --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse_M5/checkpoint-3500 \
        --gslrm_checkpoint checkpoints/gslrm/M5_E0_1_facelift \
        --gslrm_config configs/base/gslrm_mouse.yaml \
        --output_dir outputs/e2e_test

    # 6-view folder → 3D (GS-LRM only)
    python -m mouse_extensions.scripts.inference.run_e2e_inference \
        --sample_dir /path/to/sample/000531 \
        --gslrm_checkpoint checkpoints/gslrm/M5_E0_1_facelift \
        --gslrm_config configs/base/gslrm_mouse.yaml \
        --output_dir outputs/gslrm_test

    # Batch: all samples in a directory
    python -m mouse_extensions.scripts.inference.run_e2e_inference \
        --data_dir /path/to/preprocessed/M5 \
        --gslrm_checkpoint checkpoints/gslrm/M5_E0_1_facelift \
        --gslrm_config configs/base/gslrm_mouse.yaml \
        --output_dir outputs/batch_test
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description="Mouse-FaceLift E2E Inference")

    # Input (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_image", type=str, help="Single image (needs MVDiffusion)")
    group.add_argument("--sample_dir", type=str, help="6-view sample directory")
    group.add_argument("--data_dir", type=str, help="Directory of sample directories")

    # Models
    parser.add_argument("--gslrm_config", type=str, required=True, help="GS-LRM YAML config")
    parser.add_argument("--gslrm_checkpoint", type=str, required=True, help="GS-LRM checkpoint")
    parser.add_argument("--mvdiffusion_checkpoint", type=str, default=None, help="MVDiffusion checkpoint")
    parser.add_argument("--mvdiffusion_base", type=str, default="checkpoints/mvdiffusion/pipeckpts")
    parser.add_argument("--prompt_embed_path", type=str, default=None)
    parser.add_argument("--prefer_ema", action="store_true", default=True)

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/e2e_inference")
    parser.add_argument("--no_turntable", action="store_true")
    parser.add_argument("--no_mesh", action="store_true")
    parser.add_argument("--turntable_views", type=int, default=120)

    # Generation
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    from mouse_extensions.inference.end_to_end import EndToEndPipeline
    from mouse_extensions.inference.gslrm_pipeline import find_sample_dirs

    save_turntable = not args.no_turntable
    save_mesh = not args.no_mesh

    pipeline = EndToEndPipeline(
        gslrm_config=args.gslrm_config,
        gslrm_checkpoint=args.gslrm_checkpoint,
        mvdiffusion_checkpoint=args.mvdiffusion_checkpoint,
        mvdiffusion_base=args.mvdiffusion_base,
        prompt_embed_path=args.prompt_embed_path,
        device=args.device,
        image_size=args.image_size,
        prefer_ema=args.prefer_ema,
    )

    if args.input_image:
        # Full E2E: single image → 6 views → 3D
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

    elif args.sample_dir:
        # GS-LRM only from 6-view folder
        out = pipeline.run_from_views(
            args.sample_dir,
            args.output_dir,
            save_turntable=save_turntable,
            save_mesh=save_mesh,
            turntable_views=args.turntable_views,
        )
        print(f"\nDone! Output: {out}")

    elif args.data_dir:
        # Batch mode
        samples = find_sample_dirs(args.data_dir)
        print(f"Found {len(samples)} samples in {args.data_dir}")
        for sample_dir in samples:
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
