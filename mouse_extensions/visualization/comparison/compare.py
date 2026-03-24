"""Main entry point for config-driven multi-experiment comparison.

Usage (CLI):
    python -m mouse_extensions.visualization.comparison.compare \
        --config configs/comparison/example_alpha_sweep.yaml

Usage (programmatic):
    from mouse_extensions.visualization.comparison import run_comparison
    run_comparison("configs/comparison/example_alpha_sweep.yaml")

Workflow:
    1. Load YAML config (experiments, camera presets, output settings)
    2. For each experiment: load checkpoint -> build GS-LRM pipeline
    3. For each camera preset: generate camera extrinsics
    4. Render all experiments with each preset's cameras
    5. Compose grids/videos via GridComposer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Single experiment entry in comparison config."""
    name: str
    checkpoint: str
    config_path: Optional[str] = None  # YAML config for the model
    color: Optional[str] = None        # label color hint


@dataclass
class PresetConfig:
    """Camera preset entry in comparison config."""
    type: str
    n_frames: int = 120
    # Preset-specific kwargs (elevation, height, keypoint_idx, etc.)
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    """Output settings."""
    grid_layout: Tuple[int, int] = (1, 2)
    resolution: int = 512
    fps: int = 30
    output_dir: str = "experiments/comparison/"
    save_video: bool = True
    save_frames: bool = False
    save_strip: bool = True


@dataclass
class ComparisonConfig:
    """Full comparison config parsed from YAML."""
    name: str
    experiments: List[ExperimentConfig]
    camera_presets: List[PresetConfig]
    output: OutputConfig

    # Optional: path to sample data for rendering context
    sample_dir: Optional[str] = None
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_comparison_config(config_path: str) -> ComparisonConfig:
    """Load and validate comparison config from YAML.

    Args:
        config_path: path to YAML config file

    Returns:
        ComparisonConfig ready for run_comparison()
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    # Support nested "comparison:" key or flat
    cfg = raw.get("comparison", raw)

    # Parse experiments
    experiments = []
    for exp in cfg["experiments"]:
        experiments.append(ExperimentConfig(
            name=exp["name"],
            checkpoint=exp["checkpoint"],
            config_path=exp.get("config_path"),
            color=exp.get("color"),
        ))

    # Parse camera presets
    presets = []
    for p in cfg.get("camera_presets", [{"type": "orbit_360"}]):
        kwargs = {k: v for k, v in p.items() if k not in ("type", "n_frames")}
        presets.append(PresetConfig(
            type=p["type"],
            n_frames=p.get("n_frames", 120),
            kwargs=kwargs,
        ))

    # Parse output
    out_cfg = cfg.get("output", {})
    layout = out_cfg.get("grid_layout", [1, 2])
    if isinstance(layout, list):
        layout = tuple(layout)

    output = OutputConfig(
        grid_layout=layout,
        resolution=out_cfg.get("resolution", 512),
        fps=out_cfg.get("fps", 30),
        output_dir=out_cfg.get("output_dir", "experiments/comparison/"),
        save_video=out_cfg.get("save_video", True),
        save_frames=out_cfg.get("save_frames", False),
        save_strip=out_cfg.get("save_strip", True),
    )

    return ComparisonConfig(
        name=cfg.get("name", "comparison"),
        experiments=experiments,
        camera_presets=presets,
        output=output,
        sample_dir=cfg.get("sample_dir"),
        device=cfg.get("device", "cuda"),
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _load_pipeline(exp: ExperimentConfig, device: str = "cuda"):
    """Load a GS-LRM inference pipeline for one experiment.

    Auto-discovers config YAML from checkpoint directory structure.
    """
    from mouse_extensions.inference.gslrm_pipeline import GSLRMInference

    ckpt_path = Path(exp.checkpoint).expanduser()

    # Find config: explicit > checkpoint dir > default base config
    if exp.config_path:
        config_path = exp.config_path
    elif ckpt_path.is_dir() and (ckpt_path / "config.yaml").exists():
        config_path = str(ckpt_path / "config.yaml")
    elif ckpt_path.is_file() and (ckpt_path.parent / "config.yaml").exists():
        config_path = str(ckpt_path.parent / "config.yaml")
    else:
        # Look in checkpoints/gslrm/{name}/config.yaml
        base = Path("checkpoints/gslrm")
        candidates = [
            base / ckpt_path.stem / "config.yaml",
            base / ckpt_path.name / "config.yaml",
        ]
        config_path = None
        for c in candidates:
            if c.exists():
                config_path = str(c)
                break
        if config_path is None:
            raise FileNotFoundError(
                f"Cannot find config for experiment '{exp.name}'. "
                f"Provide config_path explicitly."
            )

    return GSLRMInference(
        config_path=config_path,
        checkpoint_path=str(ckpt_path),
        device=device,
    )


def _render_with_cameras(
    pipeline,
    sample_dir: str,
    c2ws: np.ndarray,
    fxfycxcy: np.ndarray,
    resolution: int,
    device: str = "cuda",
) -> np.ndarray:
    """Run GS-LRM inference and render from custom cameras.

    Args:
        pipeline: GSLRMInference instance
        sample_dir: path to sample with images + opencv_cameras.json
        c2ws: (N, 4, 4) camera-to-world matrices to render from
        fxfycxcy: (N, 4) intrinsics
        resolution: render resolution

    Returns:
        (N, H, W, 3) uint8 rendered frames
    """
    import torch
    from mouse_extensions.inference.gslrm_pipeline import load_sample_data
    from mouse_extensions.visualization import render_opencv_cam

    # Run inference once to get Gaussians
    images, input_c2ws, input_fxfycxcys, index = load_sample_data(
        sample_dir, image_size=resolution, device=device,
    )
    result = pipeline.predict(images, input_c2ws, input_fxfycxcys, index)

    gaussians_raw = result.get("gaussians", None)
    if gaussians_raw is None:
        raise RuntimeError("GS-LRM inference returned no Gaussians")
    gaussians = gaussians_raw[0] if isinstance(gaussians_raw, list) else gaussians_raw

    # Filter Gaussians
    gaussians = gaussians.apply_all_filters(
        opacity_thres=0.04,
        scaling_thres=0.1,
        floater_thres=0.6,
        crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
    )

    # Render from each camera
    rendered_frames = []
    for i in range(len(c2ws)):
        c2w_t = torch.from_numpy(c2ws[i].astype(np.float32)).to(device)
        intr_t = torch.from_numpy(fxfycxcy[i].astype(np.float32)).to(device)

        out = render_opencv_cam(
            gaussians,
            height=resolution,
            width=resolution,
            C2W=c2w_t,
            fxfycxcy=intr_t,
        )

        img = out["render"]  # (C, H, W)
        img_np = (img.detach().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        rendered_frames.append(img_np)

    return np.stack(rendered_frames)


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------

def run_comparison(
    config: ComparisonConfig | str,
    sample_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Run a full multi-experiment comparison.

    Args:
        config: ComparisonConfig or path to YAML config file
        sample_dir: override sample directory from config

    Returns:
        dict of {output_name: output_path} for all generated files
    """
    if isinstance(config, str):
        config = load_comparison_config(config)

    sample = sample_dir or config.sample_dir
    if sample is None:
        raise ValueError(
            "sample_dir is required: provide in config or as argument. "
            "This is the directory containing images/ + opencv_cameras.json."
        )

    from .camera_presets import make_preset
    from .grid_composer import GridComposer

    out_base = Path(config.output.output_dir) / config.name
    out_base.mkdir(parents=True, exist_ok=True)
    all_outputs: Dict[str, str] = {}

    # Load all pipelines
    print(f"Loading {len(config.experiments)} experiment pipelines...")
    pipelines = {}
    for exp in config.experiments:
        print(f"  Loading: {exp.name} <- {exp.checkpoint}")
        pipelines[exp.name] = _load_pipeline(exp, config.device)

    # Create composer
    composer = GridComposer(
        layout=config.output.grid_layout,
        labels=[e.name for e in config.experiments],
        resolution=config.output.resolution,
    )

    # For each camera preset
    for preset_cfg in config.camera_presets:
        print(f"\nPreset: {preset_cfg.type} ({preset_cfg.n_frames} frames)")

        preset = make_preset(
            preset_cfg.type,
            n_frames=preset_cfg.n_frames,
            resolution=config.output.resolution,
            **preset_cfg.kwargs,
        )
        c2ws, fxfycxcy = preset.generate()

        # Render each experiment
        all_frames: Dict[str, np.ndarray] = {}
        for exp in config.experiments:
            print(f"  Rendering: {exp.name}...")
            rendered = _render_with_cameras(
                pipelines[exp.name],
                sample,
                c2ws, fxfycxcy,
                config.output.resolution,
                config.device,
            )
            all_frames[exp.name] = rendered
            print(f"    -> {rendered.shape}")

        # Compose outputs
        preset_dir = out_base / preset_cfg.type
        preset_dir.mkdir(parents=True, exist_ok=True)

        if config.output.save_video:
            video_path = str(preset_dir / "comparison.mp4")
            composer.compose_video(all_frames, video_path, fps=config.output.fps)
            print(f"  Video: {video_path}")
            all_outputs[f"{preset_cfg.type}_video"] = video_path

        if config.output.save_strip:
            strip_path = str(preset_dir / "strip.png")
            composer.compose_strip(all_frames, output_path=strip_path)
            print(f"  Strip: {strip_path}")
            all_outputs[f"{preset_cfg.type}_strip"] = strip_path

        if config.output.save_frames:
            frames_dir = preset_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for i in range(min(c2ws.shape[0], 10)):  # save up to 10 sample frames
                img_path = str(frames_dir / f"frame_{i:04d}.png")
                composer.compose_image(all_frames, frame_idx=i, output_path=img_path)
            all_outputs[f"{preset_cfg.type}_frames"] = str(frames_dir)

    # Save run metadata
    meta = {
        "name": config.name,
        "experiments": [{"name": e.name, "checkpoint": e.checkpoint} for e in config.experiments],
        "presets": [{"type": p.type, "n_frames": p.n_frames} for p in config.camera_presets],
        "outputs": all_outputs,
    }
    meta_path = str(out_base / "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    all_outputs["metadata"] = meta_path

    print(f"\nComparison complete: {out_base}")
    return all_outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Config-driven multi-experiment comparison renderer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run comparison from config
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.visualization.comparison.compare \\
        --config configs/comparison/example_alpha_sweep.yaml \\
        --sample-dir /home/joon/data/preprocessed/FaceLift_mouse/M5t2/003240

    # Override output directory
    python -m mouse_extensions.visualization.comparison.compare \\
        --config configs/comparison/example_alpha_sweep.yaml \\
        --sample-dir /path/to/sample \\
        --output-dir /tmp/comparison_output
        """,
    )
    parser.add_argument("--config", required=True, help="YAML comparison config")
    parser.add_argument("--sample-dir", default=None, help="Override sample directory")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    args = parser.parse_args()

    config = load_comparison_config(args.config)

    if args.output_dir:
        config.output.output_dir = args.output_dir
    if args.device:
        config.device = args.device

    outputs = run_comparison(config, sample_dir=args.sample_dir)
    print(f"\nGenerated {len(outputs)} outputs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
