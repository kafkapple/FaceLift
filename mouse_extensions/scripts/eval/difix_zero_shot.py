"""DiFix Zero-Shot Evaluation (Stage 0).

Run pretrained nvidia/difix on GS-LRM renders to evaluate
artifact removal quality without fine-tuning.

Usage:
    CUDA_VISIBLE_DEVICES=6 python -m mouse_extensions.scripts.eval.difix_zero_shot \
        --input_dir outputs/datasets/novel_view_512/baseline_6v/mouse_m5t2/ablation_1view/cam_000 \
        --target_dir outputs/datasets/novel_view_512/baseline_6v/mouse_m5t2/gt_views/cam_000 \
        --output_dir outputs/difix_zero_shot/1view_cam000 \
        --num_samples 20
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Metrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute PSNR and SSIM between pred and target tensors (B,C,H,W)."""
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(pred.device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    return {
        "psnr": psnr_fn(pred, target).item(),
        "ssim": ssim_fn(pred, target).item(),
    }


def compute_alpha_entropy(img: np.ndarray) -> float:
    """Compute alpha channel entropy as artifact proxy (lower = cleaner)."""
    if img.shape[-1] == 4:
        alpha = img[:, :, 3].astype(float) / 255.0
    else:
        # Convert RGB to grayscale for opacity proxy
        gray = np.mean(img[:, :, :3].astype(float) / 255.0, axis=2)
        alpha = gray
    # Entropy of alpha distribution
    hist, _ = np.histogram(alpha.flatten(), bins=50, range=(0, 1), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist + 1e-10)) * (1.0 / 50)
    return float(entropy)


def load_difix_pipeline(device: str = "cuda"):
    """Load pretrained DiFix pipeline from HuggingFace."""
    print("Loading nvidia/difix pipeline...")
    try:
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(
            "nvidia/difix",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(device)
        print(f"DiFix pipeline loaded on {device}")
        return pipe
    except Exception as e:
        print(f"Failed to load DiFix: {e}")
        print("Trying img2img-turbo fallback...")
        return None


def run_difix_inference(pipe, input_image: Image.Image) -> Image.Image:
    """Run DiFix inference on a single image."""
    # DiFix expects specific resolution, resize if needed
    orig_size = input_image.size
    # Resize to DiFix native (576x1024) or keep 512x512
    result = pipe(
        image=input_image,
        num_inference_steps=1,
        timestep=199,
    ).images[0]
    # Resize back if needed
    if result.size != orig_size:
        result = result.resize(orig_size, Image.LANCZOS)
    return result


def main():
    parser = argparse.ArgumentParser(description="DiFix Zero-Shot Evaluation")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with degraded renders (e.g., ablation_1view/cam_000)")
    parser.add_argument("--target_dir", type=str, required=True,
                        help="Directory with 6-view baseline renders (gt_views/cam_000)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for DiFix-enhanced images + metrics")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of frames to process")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_comparison", action="store_true", default=True,
                        help="Save side-by-side comparison grids")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load pipeline
    pipe = load_difix_pipeline(args.device)
    if pipe is None:
        print("ERROR: Could not load DiFix pipeline. Exiting.")
        return

    # Find input images
    input_dir = Path(args.input_dir)
    target_dir = Path(args.target_dir)
    input_files = sorted(input_dir.glob("*.png"))[:args.num_samples]

    if not input_files:
        print(f"No PNG files found in {input_dir}")
        return

    print(f"Processing {len(input_files)} samples...")

    to_tensor = transforms.ToTensor()
    results = []

    for i, inp_path in enumerate(input_files):
        frame_name = inp_path.name
        target_path = target_dir / frame_name

        # Load images
        inp_img = Image.open(inp_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB") if target_path.exists() else None

        # Run DiFix
        enhanced_img = run_difix_inference(pipe, inp_img)

        # Save enhanced image
        enhanced_path = os.path.join(args.output_dir, f"difix_{frame_name}")
        enhanced_img.save(enhanced_path)

        # Compute metrics
        inp_t = to_tensor(inp_img).unsqueeze(0).to(args.device)
        enh_t = to_tensor(enhanced_img).unsqueeze(0).to(args.device)

        frame_result = {"frame": frame_name}

        if target_img is not None:
            tgt_t = to_tensor(target_img).unsqueeze(0).to(args.device)

            # Input vs target
            inp_metrics = compute_metrics(inp_t, tgt_t)
            # Enhanced vs target
            enh_metrics = compute_metrics(enh_t, tgt_t)

            frame_result["input_psnr"] = inp_metrics["psnr"]
            frame_result["input_ssim"] = inp_metrics["ssim"]
            frame_result["enhanced_psnr"] = enh_metrics["psnr"]
            frame_result["enhanced_ssim"] = enh_metrics["ssim"]
            frame_result["psnr_delta"] = enh_metrics["psnr"] - inp_metrics["psnr"]

            print(f"  [{i+1}/{len(input_files)}] {frame_name}: "
                  f"PSNR {inp_metrics['psnr']:.2f} → {enh_metrics['psnr']:.2f} "
                  f"(Δ{frame_result['psnr_delta']:+.2f})")
        else:
            print(f"  [{i+1}/{len(input_files)}] {frame_name}: no target, skipping metrics")

        # Alpha entropy (artifact proxy)
        inp_arr = np.array(inp_img)
        enh_arr = np.array(enhanced_img)
        frame_result["input_entropy"] = compute_alpha_entropy(inp_arr)
        frame_result["enhanced_entropy"] = compute_alpha_entropy(enh_arr)

        results.append(frame_result)

        # Save comparison grid
        if args.save_comparison and target_img is not None and i < 10:
            grid_w = inp_img.width * 3 + 20
            grid_h = inp_img.height
            grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
            grid.paste(inp_img, (0, 0))
            grid.paste(enhanced_img, (inp_img.width + 10, 0))
            grid.paste(target_img, (inp_img.width * 2 + 20, 0))
            grid.save(os.path.join(args.output_dir, f"compare_{frame_name}"))

    # Aggregate metrics
    if results and "input_psnr" in results[0]:
        avg_inp_psnr = np.mean([r["input_psnr"] for r in results])
        avg_enh_psnr = np.mean([r["enhanced_psnr"] for r in results])
        avg_delta = np.mean([r["psnr_delta"] for r in results])
        avg_inp_ssim = np.mean([r["input_ssim"] for r in results])
        avg_enh_ssim = np.mean([r["enhanced_ssim"] for r in results])

        summary = {
            "num_samples": len(results),
            "avg_input_psnr": float(avg_inp_psnr),
            "avg_enhanced_psnr": float(avg_enh_psnr),
            "avg_psnr_delta": float(avg_delta),
            "avg_input_ssim": float(avg_inp_ssim),
            "avg_enhanced_ssim": float(avg_enh_ssim),
            "per_frame": results,
        }

        print(f"\n=== Summary ===")
        print(f"Input  PSNR: {avg_inp_psnr:.2f}, SSIM: {avg_inp_ssim:.4f}")
        print(f"DiFix  PSNR: {avg_enh_psnr:.2f}, SSIM: {avg_enh_ssim:.4f}")
        print(f"Delta  PSNR: {avg_delta:+.2f}")
    else:
        summary = {"num_samples": len(results), "per_frame": results}

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "difix_zero_shot_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
