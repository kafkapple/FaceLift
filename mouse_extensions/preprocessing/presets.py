"""
Dataset Preprocessing Presets for Systematic Testing

Unified preset definitions for all preprocessing versions (D1-D9).

Usage:
    from mouse_extensions.preprocessing import get_preset, list_presets
    
    config = get_preset("D7.1")
    print(list_presets())

Paradigms:
    1. object_centered (D1-D4): Crop around object [DEPRECATED]
    2. pp_centered_shift (D7+): Shift image to center PP at 256
    3. precision_homography (D8+): Homography with skew correction
    4. geometry_preserving (D6): Accurate PP, various crop strategies
    5. native (D9): Original resolution, no transformation

Recommended: D7.1
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
    # GEOMETRY-PRESERVING: D6 Methods (Accurate PP)
    # ==========================================================================
    "D6-1": {
        "paradigm": "geometry_preserving",
        "method": "resize_only",
        "crop": False,
        "pp_method": "accurate",
        "normalize_fx": True,
        "normalize_translation": True,
        "output_size": 512,
        "description": "Resize only, no crop, letterbox padding",
        "ray_error": "0 deg",
        "active": True,
    },
    "D6-2": {
        "paradigm": "geometry_preserving",
        "method": "virtual_shift",
        "crop": False,
        "pp_method": "virtual_center",
        "normalize_fx": True,
        "normalize_translation": True,
        "output_size": 512,
        "description": "Virtual camera relocation (PP shift, no image change)",
        "ray_error": "0 deg",
        "active": True,
    },
    "D6-3": {
        "paradigm": "geometry_preserving",
        "method": "pp_correct_crop",
        "crop": True,
        "pp_method": "accurate",
        "normalize_fx": True,
        "normalize_translation": True,
        "output_size": 512,
        "description": "Triangulation crop with ACCURATE cx,cy (not forced 256)",
        "ray_error": "0 deg",
        "active": True,
    },

    # ==========================================================================
    # PP-CENTERED SHIFT: D7 Methods
    # ==========================================================================
    "D7": {
        "paradigm": "pp_centered_shift",
        "transform": "affine",
        "scale_mode": "fx_only",
        "pp_method": "shift_to_256",
        "skew_correction": False,
        "target_fx": 549.0,
        "output_size": 512,
        "description": "PP-centered shift, fy forced to 549",
        "ray_error": "~0.4 deg",
        "active": True,
        "note": "Legacy. fy=549 forced despite actual ~551.5",
    },
    "D7.1": {
        "paradigm": "pp_centered_shift",
        "transform": "affine",
        "scale_mode": "individual",
        "pp_method": "shift_to_256",
        "skew_correction": False,
        "target_fx": 549.0,
        "output_size": 512,
        "description": "Individual scale_x/scale_y for exact fx=fy=549",
        "ray_error": "~0 deg",
        "active": True,
        "recommended": True,
        "note": "Geometrically correct. ~0.6% anisotropic scaling. Ignores skew.",
    },
    "D7.2": {
        "paradigm": "pp_centered_shift",
        "transform": "affine",
        "scale_mode": "average",
        "pp_method": "shift_to_256",
        "skew_correction": False,
        "target_fx": 549.0,
        "output_size": 512,
        "description": "Average scale for isotropic transform, fx~548, fy~550",
        "ray_error": "~0.2 deg",
        "active": True,
        "note": "Isotropic scaling, slight fx/fy deviation from 549.",
    },

    # ==========================================================================
    # PRECISION HOMOGRAPHY: D8 Methods
    # ==========================================================================
    "D8": {
        "paradigm": "precision_homography",
        "transform": "homography",
        "scale_mode": "individual",
        "pp_method": "shift_to_256",
        "skew_correction": True,
        "target_fx": 548.9937744140625,
        "output_size": 512,
        "description": "Precision homography with skew correction",
        "ray_error": "~0 deg",
        "active": True,
        "note": "D7.1 + skew correction + exact fx (548.9937744140625)",
    },
    "D8.1": {
        "paradigm": "precision_homography",
        "transform": "homography",
        "scale_mode": "individual",
        "pp_method": "shift_to_256",
        "skew_correction": True,
        "target_fx": 548.9937744140625,
        "output_size": 512,
        "zoom": 1.3,
        "description": "D8 + virtual zoom (1.3x) for larger mouse",
        "ray_error": "~0 deg",
        "active": True,
    },

    # ==========================================================================
    # NATIVE: D9 (Original Resolution)
    # ==========================================================================
    "D9": {
        "paradigm": "native",
        "transform": "none",
        "crop": False,
        "pp_method": "original",
        "normalize_fx": False,
        "normalize_translation": False,
        "output_size": None,  # Keep original (1152x1024)
        "description": "Original resolution, no transformation, 100% geometry accuracy",
        "ray_error": "0 deg",
        "active": True,
        "note": "Requires ~4.5x more memory. For A6000+ GPUs.",
        "memory_factor": 4.5,
    },
}


# Version hierarchy
VERSION_HIERARCHY = {
    "deprecated": ["D1", "D4"],
    "geometry_preserving": ["D6-1", "D6-2", "D6-3"],
    "pp_centered": ["D7", "D7.1", "D7.2"],
    "precision": ["D8", "D8.1"],
    "native": ["D9"],
    "recommended": "D7.1",
}


def get_preset(name: str) -> dict:
    """Get preset configuration by name."""
    if name not in PRESETS:
        available = list(PRESETS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")
    return PRESETS[name]


def list_presets(include_deprecated: bool = False) -> list:
    """List available presets."""
    if include_deprecated:
        return list(PRESETS.keys())
    return [k for k, v in PRESETS.items() if not v.get("deprecated", False)]


def get_recommended() -> str:
    """Get recommended preset name."""
    return VERSION_HIERARCHY["recommended"]


def get_presets_by_paradigm(paradigm: str) -> list:
    """Get all presets for a given paradigm."""
    return [k for k, v in PRESETS.items() if v.get("paradigm") == paradigm]
