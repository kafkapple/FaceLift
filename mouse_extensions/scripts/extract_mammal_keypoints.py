#!/usr/bin/env python3
"""Extract MAMMAL 22-keypoint 3D positions from optimized body model params.

Loads ArticulationTorch model and per-frame params to compute 3D keypoint
positions for all frames. Outputs a single NPZ file.

Usage:
    CUDA_VISIBLE_DEVICES=4 /home/joon/anaconda3/envs/mammal_stable/bin/python3 \
        mouse_extensions/scripts/extract_mammal_keypoints.py \
        --mammal_dir /home/joon/dev/MAMMAL_mouse \
        --params_dir <KP_22.parent from mouse_extensions.paths>/params \
        --output <KP_22 from mouse_extensions.paths>
"""

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


KEYPOINT_NAMES = [
    "L_ear", "R_ear", "nose", "neck", "body_middle",
    "tail_root", "tail_middle", "tail_end",
    "L_paw", "L_paw_end", "L_elbow", "L_shoulder",
    "R_paw", "R_paw_end", "R_elbow", "R_shoulder",
    "L_foot", "L_knee", "L_hip",
    "R_foot", "R_knee", "R_hip",
]


def load_params(pkl_path: str, device: str = "cuda") -> dict:
    """Load MAMMAL optimization params from pickle file."""
    with open(pkl_path, "rb") as f:
        params = pickle.load(f)
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params[k] = v.to(device)
    return params


def extract_keypoints(
    mammal_dir: str,
    params_dir: str,
    step: int = 2,
    device: str = "cuda",
) -> tuple:
    """Extract 22 keypoints from MAMMAL body model for all frames.

    Returns:
        keypoints: (N, 22, 3) numpy array in MAMMAL world coordinates
        frame_indices: (N,) array of original video frame indices
    """
    # Add MAMMAL to path so articulation_th can resolve its internal paths
    sys.path.insert(0, mammal_dir)
    orig_cwd = os.getcwd()
    os.chdir(mammal_dir)

    from articulation_th import ArticulationTorch

    model = ArticulationTorch()
    model.eval()

    # Find param files for the requested optimization step
    pattern = os.path.join(params_dir, f"step_{step}_frame_*.pkl")
    param_files = sorted(glob.glob(pattern))

    if not param_files:
        print(f"No step_{step} params found, trying step_1...")
        pattern = os.path.join(params_dir, "step_1_frame_*.pkl")
        param_files = sorted(glob.glob(pattern))

    if not param_files:
        raise FileNotFoundError(f"No param files found in {params_dir}")

    print(f"Found {len(param_files)} param files (step={step})")

    all_keypoints = []
    frame_indices = []

    for i, pf in enumerate(param_files):
        fname = os.path.basename(pf)
        frame_idx = int(fname.split("_frame_")[1].replace(".pkl", ""))

        params = load_params(pf, device)

        with torch.no_grad():
            model.forward(
                thetas=params["thetas"],
                bone_lengths_core=params["bone_lengths"],
                R=params["rotation"],
                T=params["trans"],
                s=params["scale"],
                chest_deformer=params["chest_deformer"],
            )
            kp = model.forward_keypoints22()  # (1, 22, 3)

        all_keypoints.append(kp[0].cpu().numpy())
        frame_indices.append(frame_idx)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(param_files)} frames")

    os.chdir(orig_cwd)

    keypoints = np.stack(all_keypoints, axis=0)  # (N, 22, 3)
    frame_indices = np.array(frame_indices)

    # Diagnostic output
    print(f"\nExtracted keypoints: {keypoints.shape}")
    print(f"Frame index range: {frame_indices.min()} - {frame_indices.max()}")
    print(f"Coordinate range: [{keypoints.min():.4f}, {keypoints.max():.4f}]")
    print(f"Nose (idx=2) sample stats:")
    nose = keypoints[:, 2, :]
    print(f"  mean={nose.mean(axis=0)}, std={nose.std(axis=0)}")

    return keypoints, frame_indices


def main():
    parser = argparse.ArgumentParser(description="Extract MAMMAL 22 keypoints")
    parser.add_argument(
        "--mammal_dir", required=True,
        help="Path to MAMMAL_mouse repo",
    )
    parser.add_argument(
        "--params_dir", required=True,
        help="Path to optimization params directory",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output NPZ path",
    )
    parser.add_argument(
        "--step", type=int, default=2,
        help="Optimization step to use (1 or 2, default: 2)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    keypoints, frame_indices = extract_keypoints(
        mammal_dir=args.mammal_dir,
        params_dir=args.params_dir,
        step=args.step,
        device=args.device,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(
        args.output,
        keypoints=keypoints,
        frame_indices=frame_indices,
        keypoint_names=KEYPOINT_NAMES,
    )
    print(f"\nSaved to {args.output}")
    print(f"  keypoints: {keypoints.shape}")
    print(f"  frame_indices: {frame_indices.shape}")
    print(f"  keypoint_names: {len(KEYPOINT_NAMES)} names")


if __name__ == "__main__":
    main()
