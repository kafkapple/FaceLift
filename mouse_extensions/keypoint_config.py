"""Keypoint configuration loader for multi-species systems.

Loads species-specific keypoint definitions from YAML configs and provides
typed dataclass access. YAML files in configs/keypoints/ are the SSOT;
constants.py re-exports for backward compatibility.

Usage:
    from mouse_extensions.keypoint_config import load_keypoint_config

    cfg = load_keypoint_config("mouse")
    print(cfg.keypoint_names)   # ['L_ear', 'R_ear', ...]
    print(cfg.skeleton_bones)   # [(2, 0), (2, 1), ...]
    print(cfg.kp_colors)        # {0: (255, 255, 0), ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "keypoints"

# Cache loaded configs to avoid repeated file I/O (populated after class def)
_cache: dict[str, "KeypointConfig"] = {}


@dataclass(frozen=True)
class KeypointConfig:
    """Immutable keypoint configuration for a single species."""

    species: str
    num_keypoints: int
    keypoint_names: tuple[str, ...]
    skeleton_bones: tuple[tuple[int, int], ...]
    kp_colors: dict[int, tuple[int, int, int]]
    body_parts: dict[str, list[int]]
    body_part_colors: dict[str, str]
    ablation_tiers: dict[str, list[int]]
    color_groups: dict[str, list[int]] = field(default_factory=dict)

    @property
    def kp_colors_bgr(self) -> dict[int, tuple[int, int, int]]:
        """Keypoint colors in BGR (OpenCV convention)."""
        return {k: (b, g, r) for k, (r, g, b) in self.kp_colors.items()}

    @property
    def joint_groups_bgr(self) -> dict[str, dict]:
        """JOINT_GROUPS format for keypoint_overlay (BGR colors)."""
        return {
            name: {
                "indices": indices,
                "color": self.kp_colors_bgr[indices[0]],
            }
            for name, indices in self.color_groups.items()
        }


def _parse_yaml(data: dict) -> KeypointConfig:
    """Parse raw YAML dict into KeypointConfig."""
    num_kp = data["num_keypoints"]

    # Build index → RGB color map from color_groups + keypoint_colors
    kp_colors: dict[int, tuple[int, int, int]] = {}
    for group_name, indices in data["color_groups"].items():
        rgb = tuple(data["keypoint_colors"][group_name])
        for idx in indices:
            kp_colors[idx] = rgb

    # Parse skeleton bones
    bones = [tuple(b) for b in data["skeleton_bones"]]

    # Parse ablation tiers — "all" means list(range(num_kp))
    ablation_tiers = {}
    for tier_name, indices in data["ablation_tiers"].items():
        if indices == "all":
            ablation_tiers[tier_name] = list(range(num_kp))
        else:
            ablation_tiers[tier_name] = list(indices)

    return KeypointConfig(
        species=data["species"],
        num_keypoints=num_kp,
        keypoint_names=tuple(data["keypoint_names"]),
        skeleton_bones=tuple(bones),
        kp_colors=kp_colors,
        body_parts=data["body_parts"],
        body_part_colors=data["body_part_colors"],
        ablation_tiers=ablation_tiers,
        color_groups=data.get("color_groups", {}),
    )


def load_keypoint_config(
    species: str,
    config_dir: Optional[Path] = None,
) -> KeypointConfig:
    """Load keypoint config for a species.

    Args:
        species: Species name ('mouse' or 'rat').
        config_dir: Override config directory (default: configs/keypoints/).

    Returns:
        Frozen KeypointConfig dataclass.

    Raises:
        FileNotFoundError: If no YAML file found for the species.
    """
    if species in _cache:
        return _cache[species]

    search_dir = config_dir or _CONFIGS_DIR
    # Find YAML file matching species name
    candidates = list(search_dir.glob(f"{species}*.yaml"))
    if not candidates:
        raise FileNotFoundError(
            f"No keypoint config for '{species}' in {search_dir}. "
            f"Available: {[f.stem for f in search_dir.glob('*.yaml')]}"
        )

    config_path = candidates[0]
    with open(config_path) as f:
        data = yaml.safe_load(f)

    cfg = _parse_yaml(data)
    _cache[species] = cfg
    return cfg


def get_available_species(config_dir: Optional[Path] = None) -> list[str]:
    """List available species configs."""
    search_dir = config_dir or _CONFIGS_DIR
    return [f.stem.split("_")[0] for f in search_dir.glob("*.yaml")]
