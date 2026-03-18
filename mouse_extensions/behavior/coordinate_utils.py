"""Coordinate system utilities: MAMMAL mm <-> GS-LRM normalized space.

SSOT for all coordinate transformations in the behavior module.
Prevents recurring coordinate confusion (MAMMAL mm vs GS-LRM normalized).

Coordinate Systems:
    MAMMAL mm:       Raw skeleton output, range ~[0, 130] mm
    GS-LRM normalized: Scene-centered, scaled by 2.7/distance, range ~[-1, 1]

Variable Naming Convention:
    *_mm     → MAMMAL millimeter space (raw keypoints)
    *_gslrm  → GS-LRM normalized space (Gaussian positions, camera w2c)

Usage:
    from mouse_extensions.behavior.coordinate_utils import (
        mammal_to_gslrm, gslrm_to_mammal,
        assert_gslrm_space, assert_mammal_space,
    )

    kp_gslrm = mammal_to_gslrm(kp_mm)
    assert_gslrm_space(kp_gslrm, "keypoints")
"""

import numpy as np

# M5 dataset scene parameters (from preprocessing)
M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])
M5_DISTANCE_SCALE = 2.7 / 307.785  # ≈ 0.008781

# Coordinate range bounds for assertions
GSLRM_MAX_ABS = 3.0   # GS-LRM coords should be within [-3, 3]
MAMMAL_MIN_EXTENT = 5.0  # MAMMAL mm coords should have extent > 5mm


def mammal_to_gslrm(xyz_mm: np.ndarray) -> np.ndarray:
    """Convert MAMMAL mm coordinates to GS-LRM normalized space.

    Args:
        xyz_mm: (..., 3) coordinates in MAMMAL millimeter space

    Returns:
        (..., 3) coordinates in GS-LRM normalized space
    """
    return (xyz_mm - M5_SCENE_CENTER) * M5_DISTANCE_SCALE


def gslrm_to_mammal(xyz_gslrm: np.ndarray) -> np.ndarray:
    """Convert GS-LRM normalized coordinates back to MAMMAL mm space.

    Args:
        xyz_gslrm: (..., 3) coordinates in GS-LRM normalized space

    Returns:
        (..., 3) coordinates in MAMMAL millimeter space
    """
    return xyz_gslrm / M5_DISTANCE_SCALE + M5_SCENE_CENTER


def assert_gslrm_space(xyz: np.ndarray, name: str = "") -> None:
    """Guard: assert data is in GS-LRM normalized space.

    Raises AssertionError if coordinates exceed expected range.
    """
    label = f" ({name})" if name else ""
    vmin, vmax = float(xyz.min()), float(xyz.max())
    assert abs(vmax) < GSLRM_MAX_ABS and abs(vmin) < GSLRM_MAX_ABS, (
        f"Coordinate guard{label}: expected GS-LRM range "
        f"[-{GSLRM_MAX_ABS}, {GSLRM_MAX_ABS}], "
        f"got [{vmin:.1f}, {vmax:.1f}]. "
        f"Did you forget mammal_to_gslrm()?"
    )


def assert_mammal_space(xyz: np.ndarray, name: str = "") -> None:
    """Guard: assert data is in MAMMAL mm space.

    Raises AssertionError if coordinate extent is too small (likely GS-LRM).
    """
    label = f" ({name})" if name else ""
    extent = float(xyz.max() - xyz.min())
    assert extent > MAMMAL_MIN_EXTENT, (
        f"Coordinate guard{label}: expected MAMMAL mm extent > {MAMMAL_MIN_EXTENT}, "
        f"got {extent:.2f}. "
        f"Data may already be in GS-LRM normalized space. "
        f"Use gslrm_to_mammal() if conversion needed."
    )


def assert_matching_space(xyz_a: np.ndarray, xyz_b: np.ndarray,
                          name_a: str = "a", name_b: str = "b") -> None:
    """Guard: assert two arrays are in the same coordinate space.

    Checks that both are either in MAMMAL mm or GS-LRM normalized range.
    """
    a_is_gslrm = abs(float(xyz_a.max())) < GSLRM_MAX_ABS
    b_is_gslrm = abs(float(xyz_b.max())) < GSLRM_MAX_ABS
    assert a_is_gslrm == b_is_gslrm, (
        f"Coordinate space mismatch: {name_a} appears "
        f"{'GS-LRM' if a_is_gslrm else 'MAMMAL mm'}, "
        f"but {name_b} appears "
        f"{'GS-LRM' if b_is_gslrm else 'MAMMAL mm'}. "
        f"Convert one before combining."
    )
