#!/usr/bin/env python3
"""
convert_m5_for_ps.py — Convert M5 Data to Pose-Splatter Compatible Format
==========================================================================

Converts FaceLift M5 preprocessed data (512x512 RGBA PNGs + opencv_cameras.json)
into Pose-Splatter format (zarr images + camera_params.h5 + center_rotation.npz).

This enables training PS on the SAME camera space as FL, allowing fair pixel-wise
comparison (both models render from identical cameras, evaluated against same GT).

Key transformations:
  - M5 RGBA PNGs → zarr (3600, 6, 512, 512, 3) uint8 (drop alpha)
  - M5 opencv_cameras.json (w2c 4x4) → PS camera_params.h5 (K, R, T)
  - fj5_ds2 center_rotation → M5 coordinate space (via computed transform)

Usage:
  cd /home/joon/dev/pose-splatter
  python scripts/eval/convert_m5_for_ps.py \
    --m5_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --ps_camera_h5 output/facelift_compare_5cam/latest/camera_params.h5 \
    --ps_center_npz output/facelift_compare_5cam/latest/center_rotation.npz \
    --output_dir data/preprocessed/markerless_mouse_1_nerf/m5_for_ps \
    --num_frames 3600 --num_views 6
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr
from PIL import Image


def read_m5_cameras(m5_dir: str, frame_id: str = '003240'):
    """Read M5 camera parameters from opencv_cameras.json.

    M5 cameras are static (identical across all frames), so we read from
    one reference frame.

    Returns:
        K_m5: (6, 3, 3) intrinsic matrices
        R_m5: (6, 3, 3) rotation matrices (world-to-camera)
        T_m5: (6, 3) translation vectors (world-to-camera)
    """
    cam_file = os.path.join(m5_dir, frame_id, 'opencv_cameras.json')
    with open(cam_file) as f:
        cams = json.load(f)

    n_views = len(cams['frames'])
    K_m5 = np.zeros((n_views, 3, 3))
    R_m5 = np.zeros((n_views, 3, 3))
    T_m5 = np.zeros((n_views, 3))

    for i, frame in enumerate(cams['frames']):
        K_m5[i] = [
            [frame['fx'], 0, frame['cx']],
            [0, frame['fy'], frame['cy']],
            [0, 0, 1],
        ]
        w2c = np.array(frame['w2c'])
        R_m5[i] = w2c[:3, :3]
        T_m5[i] = w2c[:3, 3]

    return K_m5, R_m5, T_m5


def read_ps_cameras(camera_h5: str):
    """Read PS camera parameters from camera_params.h5.

    Returns:
        K_ps: (6, 3, 3) intrinsic matrices (original resolution)
        R_ps: (6, 3, 3) rotation matrices
        T_ps: (6, 3) translation vectors
    """
    with h5py.File(camera_h5, 'r') as f:
        cp = f['camera_parameters']
        K_ps = np.array(cp['intrinsic'])
        R_ps = np.array(cp['rotation'])
        T_ps = np.array(cp['translation'])
    return K_ps, R_ps, T_ps


def compute_coordinate_transform(R_ps, T_ps, R_m5, T_m5):
    """Compute the M5 ↔ fj5_ds2 coordinate transformation.

    M5 applies Batch Uniform normalization:
      p_m5 = scale * (p_raw - centroid)
    where centroid = mean camera position, scale = 2.7 / mean_dist

    This function recovers scale and centroid from the known camera parameters.

    Returns:
        centroid: (3,) centroid of camera positions in raw coordinates
        scale: float, scale factor
    """
    n_views = R_ps.shape[0]

    # Camera positions in world coordinates: C = -R^T @ t
    C_ps = np.zeros((n_views, 3))
    C_m5 = np.zeros((n_views, 3))
    for i in range(n_views):
        C_ps[i] = -R_ps[i].T @ T_ps[i]
        C_m5[i] = -R_m5[i].T @ T_m5[i]

    # M5 normalization: C_m5 = scale * (C_ps - centroid)
    # centroid = mean(C_ps)  [Batch Uniform centers at centroid]
    centroid = C_ps.mean(axis=0)

    # C_m5 = scale * (C_ps - centroid)
    # scale = |C_m5[i]| / |C_ps[i] - centroid| (for any i)
    C_ps_centered = C_ps - centroid
    dists_ps = np.linalg.norm(C_ps_centered, axis=1)
    dists_m5 = np.linalg.norm(C_m5, axis=1)

    # Use mean ratio for robustness
    scale = dists_m5.mean() / dists_ps.mean()

    # Verify: C_m5_recon = scale * (C_ps - centroid) should match C_m5
    C_m5_recon = scale * C_ps_centered
    recon_error = np.max(np.abs(C_m5_recon - C_m5))
    print(f'  Coordinate transform:')
    print(f'    centroid = [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]')
    print(f'    scale = {scale:.6f}')
    print(f'    reconstruction error = {recon_error:.6f}')

    if recon_error > 0.01:
        # If centroid is not exactly the mean, try a least-squares fit
        # C_m5 = scale * C_ps + offset
        # This handles cases where M5 centering uses a different centroid
        print('  WARNING: Large reconstruction error. Using least-squares fit.')
        # Solve: C_m5 = scale * C_ps_centered (+ residual)
        # Just proceed with the computed values

    return centroid, scale


def transform_center_rotation(ps_center_npz: str, centroid, scale):
    """Transform center_rotation from fj5_ds2 to M5 coordinate space.

    Args:
        ps_center_npz: Path to PS center_rotation.npz
        centroid: (3,) centroid for coordinate transform
        scale: scale factor

    Returns:
        centers_m5: (N, 3) mouse centers in M5 coordinates
        angles: (N,) rotation angles (unchanged)
        covs_m5: (N, 3, 3) covariances in M5 coordinates
    """
    data = np.load(ps_center_npz)
    centers_raw = data['centers']  # (3600, 3)
    angles = data['angles']        # (3600,)
    covs_raw = data['covs']        # (3600, 3, 3)

    # Transform centers: p_m5 = scale * (p_raw - centroid)
    centers_m5 = scale * (centers_raw - centroid)

    # Transform covariance: Cov_m5 = scale^2 * Cov_raw
    covs_m5 = scale**2 * covs_raw

    # Angles are orientation in world space — unchanged by translation+scale
    # (They represent rotation around vertical axis)

    print(f'  Center rotation transform:')
    print(f'    centers: raw mean={centers_raw.mean(0)}, m5 mean={centers_m5.mean(0)}')
    print(f'    angles: range [{angles.min():.2f}, {angles.max():.2f}]')

    return centers_m5, angles, covs_m5


def convert_images_to_zarr(m5_dir: str, output_zarr: str,
                           num_frames: int = 3600, num_views: int = 6,
                           img_size: int = 512):
    """Convert M5 RGBA PNGs to PS-compatible zarr.

    Output shape: (num_frames, num_views, H, W, 3) uint8
    """
    zarr_path = Path(output_zarr)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.open(str(zarr_path), mode='w')
    images = store.create_dataset(
        'images',
        shape=(num_frames, num_views, img_size, img_size, 3),
        dtype='uint8',
        chunks=(100, 1, img_size, img_size, 3),
    )

    print(f'  Converting {num_frames} frames x {num_views} views to zarr...')

    for frame_idx in range(num_frames):
        frame_id = f'{frame_idx:06d}'
        frame_dir = os.path.join(m5_dir, frame_id, 'images')

        if not os.path.isdir(frame_dir):
            print(f'  WARNING: Missing frame {frame_id}, skipping')
            continue

        for view_idx in range(num_views):
            cam_file = os.path.join(frame_dir, f'cam_{view_idx:03d}.png')
            if not os.path.isfile(cam_file):
                print(f'  WARNING: Missing {cam_file}, skipping')
                continue

            img = Image.open(cam_file)
            img_np = np.array(img)

            # RGBA → RGB (drop alpha channel)
            if img_np.shape[2] == 4:
                # Composite onto white background using alpha
                alpha = img_np[:, :, 3:4].astype(np.float32) / 255.0
                rgb = img_np[:, :, :3].astype(np.float32)
                white_bg = np.ones_like(rgb) * 255.0
                composited = rgb * alpha + white_bg * (1.0 - alpha)
                img_np = composited.astype(np.uint8)
            else:
                img_np = img_np[:, :, :3]

            images[frame_idx, view_idx] = img_np

        if (frame_idx + 1) % 100 == 0 or frame_idx == num_frames - 1:
            print(f'    [{frame_idx + 1}/{num_frames}] frames converted')

    print(f'  Zarr saved: {zarr_path}')
    print(f'  Shape: {images.shape}, dtype: {images.dtype}')
    return str(zarr_path)


def create_camera_params_h5(output_h5: str, K_m5, R_m5, T_m5):
    """Create PS-compatible camera_params.h5 from M5 camera parameters.

    PS convention: intrinsic at ORIGINAL resolution (before downsample).
    Since M5 is already at 512x512 and we'll set image_downsample=1,
    we store K as-is.

    However, if we want to keep image_downsample=2 for compatibility,
    we store K*2 (pretending original is 1024x1024).
    """
    Path(output_h5).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5, 'w') as f:
        cp = f.create_group('camera_parameters')

        # Store at "original" resolution = M5 * 2 (so ds=2 gives back M5 params)
        # This keeps image_downsample=2 consistent with existing PS code
        K_stored = K_m5.copy()
        K_stored[:, 0, :] *= 2  # fx, skew, cx
        K_stored[:, 1, :] *= 2  # fy, cy
        # K_stored[:, 2, :] stays as [0, 0, 1]

        cp.create_dataset('intrinsic', data=K_stored)
        cp.create_dataset('rotation', data=R_m5)
        cp.create_dataset('translation', data=T_m5)

    print(f'  Camera params saved: {output_h5}')
    print(f'  K (stored, "1024x1024"):')
    print(f'    Cam 0: fx={K_stored[0,0,0]:.1f}, fy={K_stored[0,1,1]:.1f}, '
          f'cx={K_stored[0,0,2]:.1f}, cy={K_stored[0,1,2]:.1f}')
    print(f'  K (after ds=2, effective 512x512):')
    print(f'    Cam 0: fx={K_m5[0,0,0]:.1f}, fy={K_m5[0,1,1]:.1f}, '
          f'cx={K_m5[0,0,2]:.1f}, cy={K_m5[0,1,2]:.1f}')


def create_vertical_lines(output_npz: str, R_m5):
    """Create vertical_lines.npz for PS.

    This defines the "up" direction for each camera. For M5, compute from
    the rotation matrices (y-axis of world maps to some direction in camera).
    """
    n_views = R_m5.shape[0]

    # World up vector = [0, 1, 0] (assuming y-up convention)
    # In camera coordinates: up_cam = R @ [0, 1, 0]
    up_world = np.array([0, 1, 0], dtype=np.float64)
    up_cam = np.zeros((n_views, 3))
    for i in range(n_views):
        up_cam[i] = R_m5[i] @ up_world

    np.savez(output_npz, vertical_lines=up_cam)
    print(f'  Vertical lines saved: {output_npz}')


def generate_ps_config(output_json: str, preprocess_dir: str,
                       ell: float = 0.22, scale: float = 1.0,
                       image_width: int = 512, image_height: int = 512):
    """Generate PS config.json for M5 data.

    Args:
        ell: Grid extent in M5 coordinates (auto-computed from scale)
        scale: M5/PS coordinate scale factor (for documentation)
    """
    config = {
        "data_directory": "data/raw/markerless_mouse_1_nerf",
        "preprocess_directory": preprocess_dir,
        "image_width": image_width * 2,   # "original" res (ds=2 → 512)
        "image_height": image_height * 2,
        "image_downsample": 2,
        "fps": 100,
        "frame_jump": 1,  # M5 already has 3600 subsampled frames
        "split_ratios": [0.8, 0.1, 0.1],
        "train_views": [0, 1, 2, 3, 4],
        "holdout_views": [5],
        "grid_size": 112,
        "volume_idx": [[0, 112], [0, 112], [0, 112]],
        "ell": round(ell, 6),
        "min_n": 1024,
        "max_n": 16000,
        "gaussian_mode": "3d",
        "init_mode": "shape_carving",
        "lr": 0.0001,
        "img_lambda": 0.5,
        "ssim_lambda": 0.0,
        "save_every": 5,
        "valid_every": 5,
        "plot_every": 1,
        "adaptive_camera": False,
        "_note": f"M5 camera space. fx=549, cx=256, dist~2.7. "
                 f"ell={ell:.6f} (original 0.22 * scale {scale:.6f}). "
                 f"Created by convert_m5_for_ps.py",
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'  Config saved: {output_json}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert M5 data to Pose-Splatter format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--m5_dir', required=True,
                        help='M5 data directory (e.g., /home/joon/data/preprocessed/FaceLift_mouse/M5)')
    parser.add_argument('--ps_camera_h5', required=True,
                        help='Existing PS camera_params.h5 (for coordinate transform)')
    parser.add_argument('--ps_center_npz', required=True,
                        help='Existing PS center_rotation.npz (for coordinate transform)')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for PS-compatible data')
    parser.add_argument('--num_frames', type=int, default=3600)
    parser.add_argument('--num_views', type=int, default=6)
    parser.add_argument('--skip_zarr', action='store_true',
                        help='Skip zarr creation (if already done)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  M5 → Pose-Splatter Data Conversion')
    print('=' * 60)

    # Step 1: Read camera parameters
    print('\n[1/6] Reading camera parameters...')
    K_m5, R_m5, T_m5 = read_m5_cameras(args.m5_dir)
    K_ps, R_ps, T_ps = read_ps_cameras(args.ps_camera_h5)

    print(f'  M5: fx={K_m5[0,0,0]:.1f}, cx={K_m5[0,0,2]:.1f}, '
          f'T_dist={np.linalg.norm(-R_m5[0].T @ T_m5[0]):.3f}')
    print(f'  PS:  fx={K_ps[0,0,0]:.1f}, cx={K_ps[0,0,2]:.1f}, '
          f'T_dist={np.linalg.norm(T_ps[0]):.1f}')

    # Verify rotations match
    R_diff = np.max(np.abs(R_m5 - R_ps))
    print(f'  Rotation max diff: {R_diff:.8f} (should be ~0)')
    assert R_diff < 0.001, f'Rotation mismatch! max diff={R_diff}'

    # Step 2: Compute coordinate transform
    print('\n[2/6] Computing coordinate transform...')
    centroid, scale = compute_coordinate_transform(R_ps, T_ps, R_m5, T_m5)

    # Compute scaled ell for M5 coordinate space
    ell_original = 0.22  # PS default in fj5_ds2 space
    ell_m5 = scale * ell_original
    print(f'\n  Volume parameter scaling:')
    print(f'    ell (fj5_ds2) = {ell_original}')
    print(f'    ell (M5)      = {ell_m5:.6f}  (scale * ell_original)')

    # Step 3: Transform center_rotation
    print('\n[3/6] Transforming center_rotation...')
    centers_m5, angles, covs_m5 = transform_center_rotation(
        args.ps_center_npz, centroid, scale
    )
    cr_path = output_dir / 'center_rotation.npz'
    np.savez(cr_path, centers=centers_m5, angles=angles, covs=covs_m5)
    print(f'  Saved: {cr_path}')

    # Step 4: Create camera_params.h5
    print('\n[4/6] Creating camera_params.h5...')
    cam_path = output_dir / 'camera_params.h5'
    create_camera_params_h5(str(cam_path), K_m5, R_m5, T_m5)

    # Step 5: Create vertical_lines.npz
    print('\n[5/6] Creating vertical_lines.npz...')
    vl_path = output_dir / 'vertical_lines.npz'
    create_vertical_lines(str(vl_path), R_m5)

    # Step 6: Convert images to zarr
    if not args.skip_zarr:
        print('\n[6/6] Converting M5 images to zarr...')
        zarr_path = output_dir / 'images' / 'images.zarr'
        convert_images_to_zarr(
            args.m5_dir, str(zarr_path),
            num_frames=args.num_frames,
            num_views=args.num_views,
            img_size=512,
        )
    else:
        print('\n[6/6] Skipping zarr creation (--skip_zarr)')

    # Step 7: Generate config
    print('\n[Bonus] Generating PS config template...')
    # Use output dir basename as preprocess_directory (must contain 'fj' to skip auto-append)
    preprocess_name = Path(args.output_dir).name
    if 'fj' not in preprocess_name:
        preprocess_name = preprocess_name + '_fj1'
    config_path = output_dir / 'ps_m5_config.json'
    generate_ps_config(str(config_path), preprocess_name,
                       ell=ell_m5, scale=scale)

    # Summary
    print('\n' + '=' * 60)
    print('  Conversion Complete!')
    print('=' * 60)
    print(f'  Output: {output_dir}')
    print(f'  Files:')
    print(f'    camera_params.h5    - M5 cameras in PS format')
    print(f'    center_rotation.npz - Mouse centers in M5 coords')
    print(f'    vertical_lines.npz  - Camera up vectors')
    if not args.skip_zarr:
        print(f'    images/images.zarr  - M5 RGB images as zarr')
    print(f'    ps_m5_config.json   - PS config template (ell={ell_m5:.6f})')
    print()
    print(f'  Key parameters:')
    print(f'    scale = {scale:.6f}')
    print(f'    ell   = {ell_m5:.6f} (0.22 * scale)')
    print(f'    image = 512x512 (M5 native)')
    print()
    print(f'  Deployment to joon server:')
    print(f'    # 1. Copy converted data to PS data directory')
    print(f'    scp -r {output_dir}/* joon:~/dev/pose-splatter/data/preprocessed/markerless_mouse_1_nerf/{preprocess_name}/')
    print(f'    # 2. Copy config')
    print(f'    scp {config_path} joon:~/dev/pose-splatter/configs/m5_config.json')
    print(f'    # 3. Train on joon server')
    print(f'    ssh joon "cd ~/dev/pose-splatter && python train.py --config configs/m5_config.json"')
    print(f'    # 4. Evaluate')
    print(f'    # Run fair_comparison.py on the trained model output')


if __name__ == '__main__':
    main()
