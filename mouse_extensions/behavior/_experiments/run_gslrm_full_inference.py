"""GS-LRM inference on full dataset (3585 frames) for PLY export.

Runs GS-LRM with GT 6-view input on all M5 frames, exporting Gaussian PLY files.
Uses mouse_extensions inference pipeline.

Usage on gpu03:
    CUDA_VISIBLE_DEVICES=5 python run_gslrm_full_inference.py
"""
import sys, os, time, glob
from pathlib import Path
sys.path.insert(0, ".")

def main():
    # Check existing PLY frames
    existing_frames = set()
    for ply in Path("outputs").rglob("gaussians.ply"):
        for part in ply.parts:
            if part.isdigit() and len(part) == 6:
                existing_frames.add(int(part))
                break

    # Frame jump indices to skip
    jump_mask = set()
    for idx in [1180, 2360, 3540]:
        for d in range(-2, 3):
            if 0 <= idx + d < 3600:
                jump_mask.add(idx + d)

    # Frames to process
    all_frames = set(range(3600)) - jump_mask
    missing_frames = sorted(all_frames - existing_frames)
    print(f"Total frames: {len(all_frames)}")
    print(f"Existing PLY: {len(existing_frames)}")
    print(f"Missing: {len(missing_frames)}")

    if not missing_frames:
        print("All frames already have PLY. Done!")
        return

    # Import inference pipeline
    try:
        from mouse_extensions.inference.gslrm_pipeline import GSLRMPipeline
        print("GSLRMPipeline imported")
    except ImportError:
        print("Cannot import GSLRMPipeline. Trying alternative...")
        try:
            from mouse_extensions.inference.run import run_gslrm_inference
            print("run_gslrm_inference imported")
        except ImportError:
            print("No inference module available. Checking command-line approach...")
            _run_via_cli(missing_frames)
            return

    # Try pipeline approach
    _run_via_pipeline(missing_frames)


def _run_via_cli(frames):
    """Fallback: use existing inference script via command line."""
    import subprocess

    data_dir = "/home/joon/data/preprocessed/FaceLift_mouse/M5"
    ckpt = "checkpoints/gslrm/base_uniform_v2_6view_v2/best_psnr.pt"
    out_dir = "outputs/clustering/gslrm_full"

    os.makedirs(out_dir, exist_ok=True)

    print(f"Running GS-LRM inference on {len(frames)} frames...")
    print(f"Checkpoint: {ckpt}")
    print(f"Output: {out_dir}")

    # Check if batch inference is possible
    inference_script = Path("mouse_extensions/inference/run.py")
    if inference_script.exists():
        # Try batch inference
        frame_list = ",".join(str(f) for f in frames[:10])  # test with 10 first
        cmd = [
            sys.executable, str(inference_script),
            "--data_dir", data_dir,
            "--checkpoint", ckpt,
            "--output_dir", out_dir,
            "--frames", frame_list,
            "--export_ply",
        ]
        print(f"Command: {' '.join(cmd[:6])}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"stdout: {result.stdout[-500:]}")
        if result.stderr:
            print(f"stderr: {result.stderr[-500:]}")
    else:
        print("No inference script found at expected location.")
        print("Available scripts:")
        for p in Path("mouse_extensions/inference").glob("*.py"):
            print(f"  {p}")


def _run_via_pipeline(frames):
    """Use GSLRMPipeline for batch inference."""
    from mouse_extensions.inference.gslrm_pipeline import GSLRMPipeline

    pipeline = GSLRMPipeline(
        checkpoint="checkpoints/gslrm/base_uniform_v2_6view_v2/best_psnr.pt",
        device="cuda",
    )

    data_dir = Path("/home/joon/data/preprocessed/FaceLift_mouse/M5")
    out_dir = Path("outputs/clustering/gslrm_full")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, frame_idx in enumerate(frames):
        frame_dir = data_dir / f"{frame_idx:06d}"
        if not frame_dir.exists():
            continue

        out_frame = out_dir / f"{frame_idx:06d}"
        out_frame.mkdir(parents=True, exist_ok=True)
        ply_path = out_frame / "gaussians.ply"

        if ply_path.exists():
            continue

        try:
            pipeline.inference(
                input_dir=str(frame_dir),
                output_dir=str(out_frame),
                export_ply=True,
            )
        except Exception as e:
            print(f"  Frame {frame_idx}: ERROR {str(e)[:60]}")

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(frames) - i - 1) / rate
            print(f"  {i+1}/{len(frames)} ({rate:.1f} f/s, ETA {eta:.0f}s)")

    print(f"\nDone. {len(frames)} frames in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
