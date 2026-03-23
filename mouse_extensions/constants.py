"""Shared anatomical constants for MAMMAL keypoint system.

Single Source of Truth for keypoint colors, skeleton topology, and body part
definitions used across visualization and behavior modules.

Consolidated from camera_system.py, cinematic_sequence.py, keypoint_overlay.py,
render_bodypart_gaussians.py (2026-03-23).
"""

from __future__ import annotations

# MAMMAL 22-keypoint colors (RGB 0-255) — grouped by body region
MAMMAL_KP_COLORS: dict[int, tuple[int, int, int]] = {
    # Head: yellow
    0: (255, 255, 0), 1: (255, 255, 0), 2: (255, 255, 0),
    # Body: magenta
    3: (255, 0, 255), 4: (255, 0, 255),
    # Tail: orange
    5: (255, 165, 0), 6: (255, 165, 0), 7: (255, 165, 0),
    # Left front leg: blue
    8: (0, 0, 255), 9: (0, 0, 255), 10: (0, 0, 255), 11: (0, 0, 255),
    # Right front leg: green
    12: (0, 255, 0), 13: (0, 255, 0), 14: (0, 255, 0), 15: (0, 255, 0),
    # Left hind leg: cyan
    16: (0, 255, 255), 17: (0, 255, 255), 18: (0, 255, 255),
    # Right hind leg: red
    19: (255, 0, 0), 20: (255, 0, 0), 21: (255, 0, 0),
}

# Skeleton bone connectivity (pairs of keypoint indices)
SKELETON_BONES: list[tuple[int, int]] = [
    (2, 0), (2, 1), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
    (3, 11), (11, 10), (10, 8), (8, 9),
    (3, 15), (15, 14), (14, 12), (12, 13),
    (4, 18), (18, 17), (17, 16),
    (4, 21), (21, 20), (20, 19),
]

# Body part definitions (part name → keypoint indices)
BODY_PARTS: dict[str, list[int]] = {
    "face": [0, 1, 2, 3],
    "left_paw": [8, 9, 10],
    "right_paw": [12, 13, 14],
    "tail": [5, 6, 7],
    "torso": [4, 11, 15, 18, 21],
}

# Body part display colors (hex)
BODY_PART_COLORS: dict[str, str] = {
    "face": "#FFD700",
    "left_paw": "#4169E1",
    "right_paw": "#32CD32",
    "tail": "#FF6347",
    "torso": "#DA70D6",
}
