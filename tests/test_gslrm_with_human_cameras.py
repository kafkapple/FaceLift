#!/usr/bin/env python3
"""
PoC Test: GS-LRM with Human vs Mouse Camera Parameters

This script tests the hypothesis that camera parameter differences
(not normalization) are causing blurry outputs in mouse fine-tuning.

Test 1: Mouse images + Original Mouse cameras (baseline)
Test 2: Mouse images + Normalized Mouse cameras (current approach)
Test 3: Mouse images + Human cameras (to test camera pattern hypothesis)

Usage:
    python scripts/test_gslrm_with_human_cameras.py \
        --checkpoint checkpoints/gslrm/ckpt_0000000000021125.pt \
        --mouse_sample data_mouse/sample_000000 \
        --output_dir experiments/camera_poc_test
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from einops import rearrange
from easydict import EasyDict as edict
import importlib
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gslrm.data.mouse_dataset import (
    normalize_cameras_to_y_up,
    normalize_cameras_to_z_up,
    normalize_camera_distance,
)


def estimate_up_direction(c2w_matrices):
    """Estimate up direction using PCA of camera positions."""
    positions = np.array([c2w[:3, 3] for c2w in c2w_matrices])
    center = np.mean(positions, axis=0)
    centered = positions - center

    # PCA to find orbit plane normal
    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    min_idx = np.argmin(eigenvalues.real)
    up_direction = eigenvectors[:, min_idx].real

    # Ensure consistent direction with camera ups
    avg_cam_up = np.mean([-c2w[:3, 1] for c2w in c2w_matrices], axis=0)
    if np.dot(up_direction, avg_cam_up) < 0:
        up_direction = -up_direction

    return up_direction / np.linalg.norm(up_direction)


def load_model(checkpoint_path: str, config_path: str, device: torch.device):
    """Load GS-LRM model from checkpoint."""
    config = edict(yaml.safe_load(open(config_path, "r")))
    module_name, class_name = config.model.class_name.rsplit(".", 1)
    print(f"Loading model: {module_name} -> {class_name}")

    ModelClass = importlib.import_module(module_name).__dict__[class_name]
    model = ModelClass(config)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    return model


def load_human_cameras(camera_file: str):
    """Load Human camera parameters from opencv_cameras.json"""
    with open(camera_file, 'r') as f:
        data = json.load(f)

    # Human uses camera order [2, 1, 0, 5, 4, 3]
    camera_indices = [2, 1, 0, 5, 4, 3]

    c2w_list = []
    intrinsics_list = []

    for idx in camera_indices:
        frame = data["frames"][idx]
        w2c = np.array(frame["w2c"])
        c2w = np.linalg.inv(w2c)
        c2w_list.append(c2w)
        intrinsics_list.append([frame["fx"], frame["fy"], frame["cx"], frame["cy"]])

    return np.stack(c2w_list).astype(np.float32), np.stack(intrinsics_list).astype(np.float32)


def load_mouse_cameras(sample_dir: str):
    """Load Mouse camera parameters from sample directory."""
    camera_file = os.path.join(sample_dir, "opencv_cameras.json")
    with open(camera_file, 'r') as f:
        data = json.load(f)

    c2w_list = []
    intrinsics_list = []

    for frame in data["frames"]:
        w2c = np.array(frame["w2c"])
        c2w = np.linalg.inv(w2c)
        c2w_list.append(c2w)
        intrinsics_list.append([frame["fx"], frame["fy"], frame["cx"], frame["cy"]])

    return np.stack(c2w_list).astype(np.float32), np.stack(intrinsics_list).astype(np.float32)


def load_mouse_images(sample_dir: str, num_views: int = 6):
    """Load mouse images from sample directory."""
    images = []
    for i in range(num_views):
        # Try different naming conventions
        possible_paths = [
            os.path.join(sample_dir, f"images/cam_{i:03d}.png"),
            os.path.join(sample_dir, f"images/{i:03d}.png"),
        ]
        img_path = None
        for p in possible_paths:
            if os.path.exists(p):
                img_path = p
                break

        if img_path is None:
            raise FileNotFoundError(f"Image not found for view {i}. Tried: {possible_paths}")

        img = Image.open(img_path).convert("RGB")
        img = img.resize((512, 512), Image.LANCZOS)
        images.append(np.array(img))

    return np.stack(images)


def run_inference(model, images, c2w, intrinsics, device):
    """Run GS-LRM inference."""
    # Prepare input
    images_tensor = torch.from_numpy(images).float().to(device) / 255.0
    images_tensor = rearrange(images_tensor, "v h w c -> 1 v c h w")

    c2w_tensor = torch.from_numpy(c2w).float().to(device).unsqueeze(0)
    intrinsics_tensor = torch.from_numpy(intrinsics).float().to(device).unsqueeze(0)

    # Create index tensor
    num_views = images_tensor.size(1)
    index = torch.stack([
        torch.zeros(num_views).long(),
        torch.arange(num_views).long(),
    ], dim=-1).unsqueeze(0).to(device)

    batch = edict({
        "image": images_tensor,
        "c2w": c2w_tensor,
        "fxfycxcy": intrinsics_tensor,
        "index": index,
    })

    with torch.no_grad():
        with torch.autocast(enabled=True, device_type="cuda", dtype=torch.float16):
            result = model.forward(batch, create_visual=False, split_data=True)

    # Get rendered output
    render = result.render[0]  # [V, C, H, W]
    render = rearrange(render, "v c h w -> v h w c")
    render = (render.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    return render


def analyze_cameras(c2w_matrices, name: str):
    """Analyze camera parameters."""
    positions = c2w_matrices[:, :3, 3]
    distances = np.linalg.norm(positions, axis=1)

    # Extract up vectors (Y-axis in camera space = -Y column of rotation)
    up_vectors = -c2w_matrices[:, :3, 1]

    print(f"\n=== {name} Camera Analysis ===")
    print(f"  Distances: {distances}")
    print(f"  Distance range: {distances.min():.3f} - {distances.max():.3f}")
    print(f"  Up vectors Z: {up_vectors[:, 2]}")
    print(f"  Up vectors X: {up_vectors[:, 0]}")

    return {
        "distances": distances,
        "up_vectors": up_vectors,
        "positions": positions
    }


def create_comparison_figure(results: dict, output_dir: str):
    """Create comparison visualization."""
    n_tests = len(results)
    n_views = results[list(results.keys())[0]]["render"].shape[0]

    fig, axes = plt.subplots(n_tests + 1, n_views, figsize=(3 * n_views, 3 * (n_tests + 1)))

    # First row: input images
    input_images = results[list(results.keys())[0]]["input"]
    for v in range(n_views):
        axes[0, v].imshow(input_images[v])
        axes[0, v].set_title(f"Input View {v}", fontsize=10)
        axes[0, v].axis("off")

    # Subsequent rows: rendered outputs
    for i, (test_name, data) in enumerate(results.items()):
        render = data["render"]
        for v in range(n_views):
            axes[i + 1, v].imshow(render[v])
            if v == 0:
                axes[i + 1, v].set_ylabel(test_name, fontsize=10)
            axes[i + 1, v].axis("off")

    plt.suptitle("GS-LRM Camera Parameter PoC Test\nTop: Input | Below: Rendered outputs with different camera configs",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "camera_poc_comparison.png"), dpi=150)
    plt.close()

    print(f"\nComparison saved to: {output_dir}/camera_poc_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="PoC: GS-LRM with different camera configs")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/gslrm/ckpt_0000000000021125.pt",
                        help="GS-LRM checkpoint path")
    parser.add_argument("--config", type=str,
                        default="configs/gslrm.yaml",
                        help="GS-LRM config path")
    parser.add_argument("--mouse_sample", type=str,
                        default="data_mouse/sample_000000",
                        help="Mouse sample directory")
    parser.add_argument("--human_cameras", type=str,
                        default="utils_folder/opencv_cameras.json",
                        help="Human camera file")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/camera_poc_test",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("GS-LRM Camera Parameter PoC Test")
    print("=" * 60)

    # Load model
    print("\n[1/5] Loading GS-LRM model...")
    model = load_model(args.checkpoint, args.config, device)

    # Load data
    print("\n[2/5] Loading mouse images and cameras...")
    images = load_mouse_images(args.mouse_sample)
    mouse_c2w, mouse_intrinsics = load_mouse_cameras(args.mouse_sample)

    print("\n[3/5] Loading human cameras...")
    human_c2w, human_intrinsics = load_human_cameras(args.human_cameras)

    # Analyze cameras
    mouse_analysis = analyze_cameras(mouse_c2w, "Mouse (Original)")
    human_analysis = analyze_cameras(human_c2w, "Human")

    # Apply normalizations to mouse cameras
    up_direction = estimate_up_direction(mouse_c2w)

    mouse_c2w_zup = normalize_cameras_to_z_up(mouse_c2w.copy(), up_direction)
    mouse_c2w_zup = normalize_camera_distance(mouse_c2w_zup, target_distance=2.7)
    analyze_cameras(mouse_c2w_zup, "Mouse (Z-up normalized)")

    mouse_c2w_yup = normalize_cameras_to_y_up(mouse_c2w.copy(), up_direction)
    mouse_c2w_yup = normalize_camera_distance(mouse_c2w_yup, target_distance=2.7)
    analyze_cameras(mouse_c2w_yup, "Mouse (Y-up normalized)")

    # Run tests
    results = {}

    print("\n[4/5] Running inference tests...")

    # Test 1: Original mouse cameras
    print("  Test 1: Mouse images + Original cameras")
    render1 = run_inference(model, images, mouse_c2w, mouse_intrinsics, device)
    results["1. Original Mouse"] = {"render": render1, "input": images}

    # Test 2: Z-up normalized mouse cameras
    print("  Test 2: Mouse images + Z-up normalized cameras")
    render2 = run_inference(model, images, mouse_c2w_zup, mouse_intrinsics, device)
    results["2. Z-up Normalized"] = {"render": render2, "input": images}

    # Test 3: Y-up normalized mouse cameras
    print("  Test 3: Mouse images + Y-up normalized cameras")
    render3 = run_inference(model, images, mouse_c2w_yup, mouse_intrinsics, device)
    results["3. Y-up Normalized"] = {"render": render3, "input": images}

    # Test 4: Human cameras (camera pattern transfer)
    print("  Test 4: Mouse images + Human cameras")
    render4 = run_inference(model, images, human_c2w, human_intrinsics, device)
    results["4. Human Cameras"] = {"render": render4, "input": images}

    # Create comparison
    print("\n[5/5] Creating comparison visualization...")
    create_comparison_figure(results, args.output_dir)

    # Save individual results
    for test_name, data in results.items():
        test_dir = os.path.join(args.output_dir, test_name.replace(" ", "_").replace(".", ""))
        os.makedirs(test_dir, exist_ok=True)

        for v in range(data["render"].shape[0]):
            Image.fromarray(data["render"][v]).save(os.path.join(test_dir, f"render_{v:03d}.png"))

    print("\n" + "=" * 60)
    print("PoC Test Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nKey question: Does using Human cameras produce sharper output?")
    print("If yes -> Camera pattern is the issue")
    print("If no -> Issue is elsewhere (model architecture, training, etc.)")


if __name__ == "__main__":
    main()
