"""Shared anatomical constants for multi-species keypoint systems.

Thin re-export layer over YAML configs (configs/keypoints/).
All data is loaded from YAML via keypoint_config.py; this module
re-exports for backward compatibility.

Species: Mouse (22kp MAMMAL), Rat (23kp s-DANNCE).
Skeleton anatomy: Dunn 2021 (DANNCE), Nath 2022 (rodent kinematics).
Hind limbs connect to tail_root (sacral/pelvic region), not body_middle.

Consolidated 2026-03-23. YAML migration 2026-03-23.
"""

from __future__ import annotations

from mouse_extensions.keypoint_config import load_keypoint_config


# ===========================================================================
# Mouse 22-keypoint system (MAMMAL format)
# ===========================================================================

_mouse = load_keypoint_config("mouse")

MOUSE_KP_NAMES: list[str] = list(_mouse.keypoint_names)
MAMMAL_KP_COLORS: dict[int, tuple[int, int, int]] = dict(_mouse.kp_colors)
SKELETON_BONES: list[tuple[int, int]] = [tuple(b) for b in _mouse.skeleton_bones]
BODY_PARTS: dict[str, list[int]] = {k: list(v) for k, v in _mouse.body_parts.items()}
BODY_PART_COLORS: dict[str, str] = dict(_mouse.body_part_colors)
MOUSE_ABLATION_TIERS: dict[str, list[int]] = {k: list(v) for k, v in _mouse.ablation_tiers.items()}


# ===========================================================================
# Rat 23-keypoint system (s-DANNCE format)
# ===========================================================================

_rat = load_keypoint_config("rat")

RAT_KP_NAMES: list[str] = list(_rat.keypoint_names)
RAT_KP_COLORS: dict[int, tuple[int, int, int]] = dict(_rat.kp_colors)
RAT_SKELETON_BONES: list[tuple[int, int]] = [tuple(b) for b in _rat.skeleton_bones]
RAT_BODY_PARTS: dict[str, list[int]] = {k: list(v) for k, v in _rat.body_parts.items()}
RAT_BODY_PART_COLORS: dict[str, str] = dict(_rat.body_part_colors)
RAT_ABLATION_TIERS: dict[str, list[int]] = {k: list(v) for k, v in _rat.ablation_tiers.items()}
