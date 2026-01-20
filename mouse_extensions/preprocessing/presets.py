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

Recommended: D7.1 (geometrically correct)
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
        "reason": "PP=256 forced causes 5-13° ray error",
    },

    # ==========================================================================
    # ACTIVE: PP-Centered Shift Paradigm (D7+)
    # ==========================================================================
    "D7": {
        "paradigm": "pp_centered_shift",
        "scale_mode": "fx_only",
        "pp_method": "shift_to_256",
        "description": "PP-centered shift, fy forced to 549",
        "ray_error": "~0.4°",
        "active": True,
        "note": "Current production. fy=549 forced despite actual ~551.5",
    },
    
    "D7.1": {
        "paradigm": "pp_centered_shift",
        "scale_mode": "individual",
        "pp_method": "shift_to_256",
        "description": "[RECOMMENDED] Individual scale_x/scale_y for exact fx=fy=549",
        "ray_error": "~0°",
        "active": True,
        "recommended": True,
        "note": "Geometrically correct. ~0.6% anisotropic scaling.",
    },
    
    "D7.2": {
        "paradigm": "pp_centered_shift",
        "scale_mode": "average",
        "pp_method": "shift_to_256",
        "description": "Average scale for isotropic transform, fx~548, fy~550",
        "ray_error": "~0.2°",
        "active": True,
        "note": "Isotropic scaling, slight fx/fy deviation from 549.",
