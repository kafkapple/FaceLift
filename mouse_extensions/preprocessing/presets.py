"""
Dataset Preprocessing Presets for Systematic Testing

Each preset defines:
- center_method: How to determine crop center
- pp_method: How to record principal point after crop
- scale_mode: How to compute scale factor (for D7+)
- description: What this preset tests

Paradigms:
1. Object-Centered (D1-D4): Crop around object, then normalize
2. PP-Centered Shift (D7+): Shift image to center PP at 256
3. Precision Homography (D8+): Homography with skew correction

Recommended: D8 (precision preprocessing)
"""

PRESETS = {
    # ==========================================================================
    # DEPRECATED: Object-Centered Paradigm (D1-D4)
    # ==========================================================================
    "D1": {
        "paradigm": "object_centered",
        "center_method": "per_view_2d",
        "pp_method": "force_256",
        "description": "[DEPRECATED] Per-view 2D centroid + forced PP",
        "deprecated": True,
        "reason": "Per-view center causes cross-view inconsistency",
    },

    "D4": {
        "paradigm": "object_centered",
        "center_method": "triangulation",
        "pp_method": "force_256",
        "description": "[DEPRECATED] 3D triangulation + forced PP=256",
        "deprecated": True,
        "reason": "PP=256 forced causes 5-13 deg ray error",
    },

    # ==========================================================================
    # LEGACY: PP-Centered Shift Paradigm (D7)
    # ==========================================================================
    "D7": {
        "paradigm": "pp_centered_shift",
        "scale_mode": "fx_only",
        "pp_method": "shift_to_256",
        "description": "PP-centered shift, fy forced to 549",
        "ray_error": "~0.4 deg",
        "active": True,
        "note": "Legacy. fy=549 forced despite actual ~551.5",
    },

    "D7.1": {
        "paradigm": "pp_centered_shift",
        "scale_mode": "individual",
        "pp_method": "shift_to_256",
        "description": "Individual scale_x/scale_y for exact fx=fy=549",
        "ray_error": "~0 deg",
        "active": True,
        "note": "Geometrically correct. ~0.6% anisotropic scaling. Ignores skew.",
    },

    "D7.2": {
        "paradigm": "pp_centered_shift",
        "scale_mode": "average",
        "pp_method": "shift_to_256",
        "description": "Average scale for isotropic transform, fx~548, fy~550",
        "ray_error": "~0.2 deg",
        "active": True,
        "note": "Isotropic scaling, slight fx/fy deviation from 549.",
    },

    # ==========================================================================
    # PRECISION: Homography-Based (D8+) - NEW
    # ==========================================================================
    "D8": {
        "paradigm": "precision_homography",
        "transform": "homography",
        "scale_mode": "individual",
        "pp_method": "shift_to_256",
        "skew_correction": True,
        "target_fx": 548.9937744140625,  # Exact GS-LRM pretrained value
        "description": "[RECOMMENDED] Precision homography with skew correction",
        "ray_error": "~0 deg",
        "active": True,
        "recommended": True,
        "note": "D7.1 + skew correction + exact fx (548.9937744140625)",
        "improvements": [
            "Exact fx (548.9937744140625 vs 549.0)",
            "Skew correction via H = K_target @ K_orig^-1",
            "LANCZOS4 interpolation"
        ],
    },

    "D8.1": {
        "paradigm": "precision_homography",
        "transform": "homography",
        "scale_mode": "individual",
        "pp_method": "shift_to_256",
        "skew_correction": True,
        "target_fx": 548.9937744140625,
        "zoom": 1.3,
        "description": "D8 + virtual zoom (1.3x) for larger mouse",
        "ray_error": "~0 deg",
        "active": True,
        "note": "D8 with 1.3x zoom via crop+resize. Mouse 40% of image.",
        "zoom_method": "crop_resize",
        "mouse_size_percent": 40,
        "error_amplification": 2.5,
    },
}


# Version hierarchy
VERSION_HIERARCHY = {
    "deprecated": ["D1", "D4"],
    "legacy": ["D7"],
    "active": ["D7.1", "D7.2"],
    "precision": ["D8", "D8.1"],
    "recommended": "D8",
}


def get_preset(name: str) -> dict:
    """Get preset configuration by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]


def list_presets(include_deprecated: bool = False) -> list:
    """List available presets."""
    if include_deprecated:
        return list(PRESETS.keys())
    return [k for k, v in PRESETS.items() if not v.get("deprecated", False)]


def get_recommended() -> str:
    """Get recommended preset name."""
    return VERSION_HIERARCHY["recommended"]
