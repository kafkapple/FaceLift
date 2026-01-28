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

        # Load checkpoint
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.is_dir():
            ckpt_files = sorted(
                ckpt_path.glob("ckpt_*.pt"),
                key=lambda x: int(x.stem.split("_")[-1]),
            )
            if not ckpt_files:
                raise FileNotFoundError(f"No ckpt_*.pt in {ckpt_path}")
            ckpt_path = ckpt_files[-1]

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
        turntable_views: int = 120,
        turntable_fps: int = 30,
        image_size: int = 512,
    ) -> Path:
        """Save inference outputs: PLY, rendered views, turntable video.

        Args:
            result: Model output from predict().
            output_dir: Base output directory.
            name: Sample name (subfolder).
            save_turntable: Generate turntable video.
            save_mesh: Save PLY file.
            turntable_views: Number of turntable frames.
            turntable_fps: Video FPS.
            image_size: Rendering resolution.

        Returns:
            Output path.
        """
        from gslrm.model.gaussians_renderer import render_turntable, imageseq2video

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

        # Save PLY
        if save_mesh:
            ply_path = out / "gaussians.ply"
            filtered.save_ply(str(ply_path))
            print(f"  Saved PLY: {ply_path}")

        # Save rendered views
        if result.render is not None:
            comp = result.render[0].detach()
            for i in range(comp.size(0)):
                view_np = (
                    comp[i].permute(1, 2, 0).cpu().numpy() * 255.0
                ).clip(0, 255).astype(np.uint8)
                Image.fromarray(view_np).save(out / f"render_view_{i:02d}.png")

            if comp.size(0) > 1:
                grid = rearrange(comp, "v c h w -> h (v w) c")
                grid_np = (grid.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(grid_np).save(out / "render_grid.png")

        # Turntable video
        if save_turntable:
            print(f"  Generating turntable ({turntable_views} views)...")
            try:
                vis = render_turntable(
                    filtered,
                    rendering_resolution=image_size,
                    num_views=turntable_views,
                )
                vis = rearrange(vis, "h (v w) c -> v h w c", v=turntable_views)
                vis = np.ascontiguousarray(vis)
                video_path = out / "turntable.mp4"
                imageseq2video(vis, str(video_path), fps=turntable_fps)
                print(f"  Saved turntable: {video_path}")
            except Exception as e:
                print(f"  Warning: turntable failed: {e}")

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
