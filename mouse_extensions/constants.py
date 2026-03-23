"""Shared anatomical constants for multi-species keypoint systems.

Single Source of Truth for keypoint colors, skeleton topology, body part
definitions, and ablation tiers used across visualization and behavior modules.

Species: Mouse (22kp MAMMAL), Rat (23kp s-DANNCE).
Skeleton anatomy: Dunn 2021 (DANNCE), Nath 2022 (rodent kinematics).
Hind limbs connect to tail_root (sacral/pelvic region), not body_middle.

Consolidated 2026-03-23. Hip fix + multi-species tiers added 2026-03-23.
"""

from __future__ import annotations


# ===========================================================================
# Mouse 22-keypoint system (MAMMAL format)
# ===========================================================================

# Index → name mapping (2026-03-23)
MOUSE_KP_NAMES: list[str] = [
    "L_ear", "R_ear", "nose", "neck", "body_middle", "tail_root",
    "tail_middle", "tail_end", "L_paw", "L_paw_end", "L_elbow", "L_shoulder",
    "R_paw", "R_paw_end", "R_elbow", "R_shoulder", "L_foot", "L_knee",
    "L_hip", "R_foot", "R_knee", "R_hip",
]

# Keypoint colors (RGB 0-255) — grouped by body region
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

# Skeleton bone connectivity — anatomically correct (2026-03-23)
# Hind limbs → tail_root(5), NOT body_middle(4).
# Ref: Dunn 2021, Nath 2022, DeepLabCut, SLEAP, convert_dannce_to_coco.py
SKELETON_BONES: list[tuple[int, int]] = [
    # Head
    (2, 0), (2, 1),       # nose → ears
    # Spine
    (2, 3), (3, 4), (4, 5),  # nose → neck → body_middle → tail_root
    # Tail
    (5, 6), (6, 7),       # tail_root → tail_middle → tail_end
    # Left front limb (neck → shoulder → elbow → paw → paw_end)
    (3, 11), (11, 10), (10, 8), (8, 9),
    # Right front limb
    (3, 15), (15, 14), (14, 12), (12, 13),
    # Left hind limb (tail_root → hip → knee → foot)
    (5, 18), (18, 17), (17, 16),
    # Right hind limb
    (5, 21), (21, 20), (20, 19),
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

# Ablation tiers — functionally named, literature-based (2026-03-23)
# Ref: Dunn 2021, Nath 2022, DeepLabCut, SLEAP, Keypoint-MoSeq
MOUSE_ABLATION_TIERS: dict[str, list[int]] = {
    # T1: Primary axis — position, heading, speed
    "axis": [2, 4, 5],           # nose, body_middle, tail_root
    # T2: + Head orientation — social interaction, attention
    "head": [0, 1, 2, 3, 4, 5],  # ears, nose, neck, body_middle, tail_root
    # T3: + Proximal limbs — posture classification (rearing, crouching)
    "posture": [0, 1, 2, 3, 4, 5, 11, 15, 18, 21],  # + shoulders, hips
    # T4: + Distal limbs — gait analysis, paw interaction
    "locomotion": [0, 1, 2, 3, 4, 5, 8, 11, 12, 15, 16, 18, 19, 21],  # + paws, feet
    # T5: Full skeleton — joint angles, fine motor, 3D reconstruction
    "skeleton": list(range(22)),
}


# ===========================================================================
# Rat 23-keypoint system (s-DANNCE format)
# ===========================================================================

RAT_KP_NAMES: list[str] = [
    "SpineF", "SpineM", "SpineL", "Offset1", "Offset2",
    "HipL", "HipR", "KneeL", "KneeR", "ShinL", "ShinR",
    "ElbowL", "ElbowR", "ShoulderL", "ShoulderR",
    "EarL", "EarR", "Snout",
    "Tail1", "Tail2", "Tail3",
    "PawL", "PawR",
]

# Rat keypoint colors — consistent body regions with mouse (2026-03-23)
RAT_KP_COLORS: dict[int, tuple[int, int, int]] = {
    # Spine: magenta (matches mouse body)
    0: (255, 0, 255), 1: (255, 0, 255), 2: (255, 0, 255),
    # Offsets: magenta
    3: (255, 0, 255), 4: (255, 0, 255),
    # Left hind: cyan (matches mouse)
    5: (0, 255, 255), 7: (0, 255, 255), 9: (0, 255, 255),
    # Right hind: red (matches mouse)
    6: (255, 0, 0), 8: (255, 0, 0), 10: (255, 0, 0),
    # Left front: blue (matches mouse)
    11: (0, 0, 255), 13: (0, 0, 255), 21: (0, 0, 255),
    # Right front: green (matches mouse)
    12: (0, 255, 0), 14: (0, 255, 0), 22: (0, 255, 0),
    # Head: yellow (matches mouse)
    15: (255, 255, 0), 16: (255, 255, 0), 17: (255, 255, 0),
    # Tail: orange (matches mouse)
    18: (255, 165, 0), 19: (255, 165, 0), 20: (255, 165, 0),
}

# Rat skeleton bones — anatomically correct (2026-03-23)
RAT_SKELETON_BONES: list[tuple[int, int]] = [
    # Spine chain
    (17, 0), (0, 1), (1, 2),    # Snout → SpineF → SpineM → SpineL
    # Tail
    (2, 18), (18, 19), (19, 20),  # SpineL → Tail1 → Tail2 → Tail3
    # Head
    (17, 15), (17, 16),          # Snout → ears
    # Left front limb
    (0, 13), (13, 11), (11, 21),  # SpineF → ShoulderL → ElbowL → PawL
    # Right front limb
    (0, 14), (14, 12), (12, 22),  # SpineF → ShoulderR → ElbowR → PawR
    # Left hind limb (SpineL → hip, anatomically correct)
    (2, 5), (5, 7), (7, 9),     # SpineL → HipL → KneeL → ShinL
    # Right hind limb
    (2, 6), (6, 8), (8, 10),    # SpineL → HipR → KneeR → ShinR
]

RAT_BODY_PARTS: dict[str, list[int]] = {
    "face": [15, 16, 17],             # ears, snout
    "left_paw": [11, 13, 21],         # elbow, shoulder, paw
    "right_paw": [12, 14, 22],
    "tail": [18, 19, 20],
    "torso": [0, 1, 2, 3, 4, 5, 6],   # spine + offsets + hips
}

RAT_BODY_PART_COLORS: dict[str, str] = {
    "face": "#FFD700",       # same as mouse
    "left_paw": "#4169E1",
    "right_paw": "#32CD32",
    "tail": "#FF6347",
    "torso": "#DA70D6",
}

# Rat ablation tiers — parallel structure to mouse (2026-03-23)
RAT_ABLATION_TIERS: dict[str, list[int]] = {
    # T1: Primary axis
    "axis": [17, 1, 2],              # Snout, SpineM, SpineL
    # T2: + Head orientation
    "head": [15, 16, 17, 0, 1, 2],   # ears, Snout, SpineF, SpineM, SpineL
    # T3: + Proximal limbs
    "posture": [15, 16, 17, 0, 1, 2, 13, 14, 5, 6],  # + shoulders, hips
    # T4: + Distal limbs
    "locomotion": [15, 16, 17, 0, 1, 2, 9, 10, 13, 14, 5, 6, 21, 22],  # + shins, paws
    # T5: Full skeleton
    "skeleton": list(range(23)),
}
