"""GS-LRM inference pipeline for 3D Gaussian reconstruction.

Loads a trained GS-LRM model and reconstructs 3D Gaussians from multi-view images.
"""

import importlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from einops import rearrange
from PIL import Image

from mouse_extensions.inference.checkpoint_utils import find_checkpoint


class GSLRMInference:
    """GS-LRM pipeline: multi-view images → 3D Gaussian splats."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        image_size: Optional[int] = None,
    ):
        """Load GS-LRM model from config and checkpoint.

        Args:
            config_path: YAML config file path.
            checkpoint_path: Checkpoint directory or .pt file.
            device: Torch device.
            image_size: Override image size in config (optional).
        """
        self.device = device

        # Load config
        with open(config_path, "r") as f:
            self.config = edict(yaml.safe_load(f))

        if image_size is not None:
            self.config.model.image_tokenizer.image_size = image_size

        # Load model
        module, class_name = self.config.model.class_name.rsplit(".", 1)
        GSLRM = importlib.import_module(module).__dict__[class_name]
        self.model = GSLRM(self.config).to(device)

        # Find and load checkpoint (supports experiment names, directories, etc.)
        ckpt_path = find_checkpoint(checkpoint_path)
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
            step = checkpoint.get("fwdbwd_pass_step", "?")
            print(f"  Training step: {step}")
        else:
            state_dict = checkpoint

        # Handle DDP prefix
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Filter out loss calculator weights
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("loss_calculator.")
        }

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        print("GS-LRM model loaded")

    def predict(
        self,
        images: torch.Tensor,
        c2ws: torch.Tensor,
        fxfycxcys: torch.Tensor,
        index: torch.Tensor,
    ) -> edict:
        """Run GS-LRM inference.

        Args:
            images: [B, V, C, H, W]
            c2ws: [B, V, 4, 4]
            fxfycxcys: [B, V, 4]
            index: [B, V, 2] — (view_idx, scene_idx)

        Returns:
            Model result edict with gaussians, render, etc.
        """
        batch = edict(
            image=images,
            c2w=c2ws,
            fxfycxcy=fxfycxcys,
            index=index,
        )

        with torch.no_grad(), torch.autocast(
            enabled=True,
            device_type="cuda" if "cuda" in self.device else "cpu",
            dtype=torch.float16,
        ):
            result = self.model.forward(batch, create_visual=True, split_data=True)

        return result

    def save_outputs(
        self,
        result: edict,
        output_dir: str,
        name: str,
        save_turntable: bool = True,
        save_mesh: bool = True,
        save_comparison: bool = True,
        save_turntable_grid: bool = True,
        save_rrd: bool = True,
        gt_images: Optional[torch.Tensor] = None,
        image_size: int = 512,
        camera_indices: list = None,
    ) -> Path:
        """Save inference outputs: PLY, rendered views, turntable, grids, RRD.

        All save_* options are True by default for comprehensive output.
        Turntable parameters (views, fps, elevation, radius) are sourced from
        TurntableVideoConfig via self.config for consistency with train/val paths.

        Args:
            result: Model output from predict().
            output_dir: Base output directory.
            name: Sample name (subfolder).
            save_turntable: Generate turntable video.
            save_mesh: Save PLY file.
            save_comparison: Save GT vs Pred comparison grid.
            save_turntable_grid: Save multi-elevation turntable grid.
            save_rrd: Save Rerun .rrd sequence (3D Gaussians + rendered views).
            gt_images: Input images [B,V,C,H,W] for comparison (auto-extracted if None).
            image_size: Rendering resolution.

        Returns:
            Output path.
        """

        out = Path(output_dir) / name
        out.mkdir(parents=True, exist_ok=True)

        gaussians = result.gaussians[0]

        # Filter Gaussians
        filtered = gaussians.apply_all_filters(
            opacity_thres=0.04,
            scaling_thres=0.1,
            floater_thres=0.6,
            crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
            cam_origins=None,
            nearfar_percent=(0.0001, 1.0),
        )

        # Resolve GT images (for comparison/grid)
        input_images = gt_images
        if input_images is None and hasattr(result, "input") and result.input is not None:
            input_images = result.input.get("image", None)

        # Save PLY
        if save_mesh:
            ply_path = out / "gaussians.ply"
            filtered.save_ply(str(ply_path))
            print(f"  Saved PLY: {ply_path}")

        # Save rendered views
        comp = None
        if result.render is not None:
            comp = result.render[0].detach()
            for i in range(comp.size(0)):
                # Use camera index for filename when camera_indices provided
                cam_idx = camera_indices[i] if camera_indices is not None else i
                view_np = (
                    comp[i].permute(1, 2, 0).cpu().numpy() * 255.0
                ).clip(0, 255).astype(np.uint8)
                Image.fromarray(view_np).save(out / f"render_view_{cam_idx:02d}.png")

            if comp.size(0) > 1:
                grid = rearrange(comp, "v c h w -> h (v w) c")
                grid_np = (grid.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(grid_np).save(out / "render_grid.png")

        # Save rendered alpha if available
        if hasattr(result, 'rendered_alpha') and result.rendered_alpha is not None:
            alpha = result.rendered_alpha[0].detach()  # [V, 1, H, W]
            for i in range(alpha.size(0)):
                cam_idx = camera_indices[i] if camera_indices is not None else i
                alpha_np = (
                    alpha[i, 0].cpu().numpy() * 255.0
                ).clip(0, 255).astype(np.uint8)
                Image.fromarray(alpha_np, mode='L').save(out / f'render_alpha_{cam_idx:02d}.png')

        # GT vs Pred comparison grid
        if save_comparison and comp is not None and input_images is not None:
            try:
                from mouse_extensions.visualization.inference_viz import (
                    save_comparison_grid,
                )
                save_comparison_grid(
                    input_images, comp, str(out / "comparison_grid.png")
                )
                print(f"  Saved comparison grid")
            except Exception as e:
                print(f"  Warning: comparison grid failed: {e}")

        # Turntable video + grid (unified via TurntableRenderer)
        if save_turntable or save_turntable_grid:
            try:
                from mouse_extensions.visualization.turntable_renderer import TurntableRenderer, TurntableVideoConfig
                tt_cfg = TurntableVideoConfig.from_config(self.config)
                # Inference-specific overrides
                tt_cfg.save_view_with_input = False
                tt_cfg.save_orbit = save_turntable
                tt_cfg.save_orbit_with_input = False
                tt_cfg.save_grid = save_turntable_grid
                tt_renderer = TurntableRenderer(tt_cfg)
                tt_renderer.render_all(
                    gaussians=filtered,
                    output_dir=str(out),
                    uid=name,
                    rendering_resolution=image_size,
                    target_images=input_images,
                )
                print(f"  Saved turntable outputs")
            except Exception as e:
                print(f"  Warning: turntable failed: {e}")

        # Multi-elevation turntable grid (separate from TurntableRenderer — uses multi-elevation)
        if save_turntable_grid:
            try:
                from mouse_extensions.visualization.inference_viz import (
                    save_multiview_turntable_grid,
                )
                from mouse_extensions.visualization.turntable_renderer import TurntableVideoConfig as _TVC
                _tt = _TVC.from_config(self.config)
                save_multiview_turntable_grid(
                    filtered,
                    str(out / "turntable_grid.png"),
                    elevations=[0, 15, 30],
                    num_azimuth=6,
                    radius=_tt.orbit_radius,
                    render_res=image_size,
                    gt_images=input_images,
                )
                print(f"  Saved multi-elevation turntable grid")
            except Exception as e:
                print(f"  Warning: multi-elevation grid failed: {e}")

        # Rerun .rrd sequence
        if save_rrd:
            try:
                rrd_path = out / "gaussians.rrd"
                _save_rerun_rrd(
                    filtered, comp, input_images, str(rrd_path), name
                )
                print(f"  Saved RRD: {rrd_path}")
            except Exception as e:
                print(f"  Warning: RRD export failed: {e}")

        return out


def load_sample_data(
    sample_dir: str,
    image_size: int = 512,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load 6-view images and cameras from a sample directory.

    Args:
        sample_dir: Directory with images/ and opencv_cameras.json.
        image_size: Target image size.
        device: Torch device.

    Returns:
        (images [1,V,C,H,W], c2ws [1,V,4,4], fxfycxcys [1,V,4], index [1,V,2])
    """
    sample_path = Path(sample_dir)

    camera_json = sample_path / "opencv_cameras.json"
    if not camera_json.exists():
        raise FileNotFoundError(f"Camera file not found: {camera_json}")

    with open(camera_json, "r") as f:
        camera_data = json.load(f)

    frames = camera_data["frames"]
    num_views = len(frames)

    images_dir = sample_path / "images"
    if not images_dir.exists():
        images_dir = sample_path

    images = []
    c2ws = []
    fxfycxcys = []

    for i, frame in enumerate(frames):
        # Load image
        img_path = images_dir / f"cam_{i:03d}.png"
        if not img_path.exists():
            img_path = images_dir / frame.get("file_path", f"cam_{i:03d}.png").split("/")[-1]

        img = Image.open(img_path)
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img).convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        if img.size[0] != image_size:
            img = img.resize((image_size, image_size), Image.LANCZOS)

        img_np = np.array(img).astype(np.float32) / 255.0
        images.append(torch.from_numpy(img_np).permute(2, 0, 1))

        # Camera extrinsics
        w2c = np.array(frame["w2c"])
        c2ws.append(np.linalg.inv(w2c))

        # Camera intrinsics (scale if resized)
        scale = image_size / frame.get("w", image_size)
        fxfycxcys.append([
            frame["fx"] * scale,
            frame["fy"] * scale,
            frame["cx"] * scale,
            frame["cy"] * scale,
        ])

    images_t = torch.stack(images).unsqueeze(0).to(device)  # [1, V, C, H, W]
    c2ws_t = torch.from_numpy(np.array(c2ws)).float().unsqueeze(0).to(device)
    fxfycxcys_t = torch.from_numpy(np.array(fxfycxcys)).float().unsqueeze(0).to(device)

    # Index: (view_idx, scene_idx) — matches training data loader order
    index = torch.stack([
        torch.arange(num_views).long(),
        torch.zeros(num_views).long(),
    ], dim=-1).unsqueeze(0).to(device)

    return images_t, c2ws_t, fxfycxcys_t, index


def find_sample_dirs(data_dir: str) -> List[str]:
    """Find all sample directories containing opencv_cameras.json."""
    data_path = Path(data_dir)
    samples = []
    for item in sorted(data_path.iterdir()):
        if item.is_dir():
            if (item / "opencv_cameras.json").exists():
                samples.append(str(item))
            elif (item / "images" / "cam_000.png").exists():
                samples.append(str(item))
    return samples


def _save_rerun_rrd(
    gaussians,
    pred_images: Optional[torch.Tensor],
    gt_images: Optional[torch.Tensor],
    output_path: str,
    recording_name: str = "gslrm_inference",
) -> None:
    """Save 3D Gaussians and rendered views as a Rerun .rrd file.

    Logs:
        - 3D Gaussian point cloud (positions + colors)
        - Predicted rendered views (if available)
        - GT input views (if available)

    Args:
        gaussians: Filtered GaussianModel with .xyz, .get_opacity, .get_features.
        pred_images: Predicted views [V, C, H, W] (optional).
        gt_images: GT input views [B, V, C, H, W] or [V, C, H, W] (optional).
        output_path: Path to save .rrd file.
        recording_name: Rerun recording name.
    """
    import rerun as rr

    rr.init(recording_name, spawn=False)
    rr.save(output_path)

    # Log 3D Gaussian point cloud
    means = gaussians.get_xyz.detach().cpu().numpy()  # [N, 3]
    opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze(-1)  # [N]

    # Extract colors from SH features (DC component)
    sh_features = gaussians.get_features  # [N, num_sh, 3]
    if sh_features is not None:
        # DC component (index 0), convert SH to RGB
        dc = sh_features[:, 0, :].detach().cpu().numpy()  # [N, 3]
        colors_rgb = (0.5 + dc * 0.28209479177387814).clip(0, 1)  # SH C0 = 0.2821
        colors_u8 = (colors_rgb * 255).clip(0, 255).astype(np.uint8)
    else:
        colors_u8 = None

    # Filter by opacity for cleaner visualization
    vis_mask = opacity > 0.1
    vis_means = means[vis_mask]

    rr.log(
        "gaussians/points",
        rr.Points3D(
            vis_means,
            colors=colors_u8[vis_mask] if colors_u8 is not None else None,
            radii=np.full(vis_means.shape[0], 0.003, dtype=np.float32),
        ),
    )

    # Log predicted rendered views
    if pred_images is not None:
        for i in range(pred_images.size(0)):
            img_np = (
                pred_images[i].permute(1, 2, 0).cpu().numpy() * 255
            ).clip(0, 255).astype(np.uint8)
            rr.log(f"renders/pred/view_{i:02d}", rr.Image(img_np))

    # Log GT input views
    if gt_images is not None:
        gt = gt_images[0] if gt_images.dim() == 5 else gt_images
        for i in range(gt.size(0)):
            img_np = (
                gt[i].permute(1, 2, 0).cpu().numpy() * 255
            ).clip(0, 255).astype(np.uint8)
            rr.log(f"renders/gt/view_{i:02d}", rr.Image(img_np))
