"""
Dataset Preprocessing Presets for Systematic Testing

Unified preset definitions for all preprocessing versions (D1-D10).

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
    6. up_aligned_zoom (D10): Up-alignment + adaptive zoom [PROPOSED]

Recommended: D7.1 (stable), D8 (precision), D10 (experimental)
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
    "D8.2": {
        "paradigm": "precision_homography",
        "transform": "homography",
        "scale_mode": "individual",
        "pp_method": "shift_to_256",
        "skew_correction": True,
        "target_fx": 548.9937744140625,
        "output_size": 512,
        "adaptive_zoom": True,
        "zoom_range": [1.0, 1.5],
        "zoom_fill_ratio": 0.85,
        "description": "D8 + Adaptive zoom (bbox-based) - NO up-alignment",
        "ray_error": "~0 deg",
        "active": True,
        "note": "Best of D8 (geometry) + D10.1 (zoom). Mouse fills 85% of frame.",
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
        "output_size": None,
        "description": "Original resolution, no transformation, 100% geometry accuracy",
        "ray_error": "0 deg",
        "active": True,
        "note": "Requires ~4.5x more memory. For A6000+ GPUs.",
        "memory_factor": 4.5,
    },
    "D9_norm": {
        "paradigm": "native",
        "transform": "none",
        "crop": False,
        "pp_method": "original",
        "normalize_fx": False,
        "normalize_translation": True,
        "target_distance": 2.7,
        "output_size": None,
        "description": "Original resolution + translation normalization",
        "ray_error": "0 deg",
        "active": True,
        "experimental": True,
        "note": "Experimental: fx unchanged (1632), translation normalized to 2.7",
        "memory_factor": 4.5,
    },
    "D9_resized": {
        "paradigm": "pp_centered_shift",
        "transform": "affine",
        "scale_mode": "individual",
        "pp_method": "shift_to_256",
        "skew_correction": False,
        "target_fx": 549.0,
        "output_size": 512,
        "description": "D9 resized to 512x512 with full normalization",
        "ray_error": "~0 deg",
        "active": True,
        "recommended_for_d9": True,
        "note": "Safe: matches pretrained distribution exactly",
    },

    # ==========================================================================
    # UP-ALIGNED + ADAPTIVE ZOOM: D10 Methods (PROPOSED)
    # ==========================================================================
    "D10": {
        "paradigm": "up_aligned_zoom",
        "transform": "homography",
        "scale_mode": "individual",
        "pp_method": "shift_to_256",
        "skew_correction": True,
        "up_alignment": True,
        "up_source": "vertical_lines",  # vertical_lines.npz
        "adaptive_zoom": False,
        "zoom": 1.0,
        "target_fx": 548.9937744140625,
        "output_size": 512,
        "description": "D8 + Up-direction alignment from vertical_lines.npz",
        "ray_error": "~0 deg",
        "active": True,
        "experimental": True,
        "note": "Aligns world Z-axis to up direction. Good for turntable consistency.",
    },
    "D10.1": {
        "paradigm": "up_aligned_zoom",
        "transform": "homography",
        "scale_mode": "individual",
        "pp_method": "shift_to_256",
        "skew_correction": True,
        "up_alignment": True,
        "up_source": "vertical_lines",
        "adaptive_zoom": True,
        "zoom_range": [1.2, 1.5],
        "zoom_fill_ratio": 0.8,
        "target_fx": 548.9937744140625,
        "output_size": 512,
        "description": "D10 + Adaptive zoom for larger mouse (bbox-based)",
        "ray_error": "~0 deg",
        "active": True,
        "experimental": True,
        "note": "Auto-adjusts zoom per frame. Mouse fills 80% of frame.",
    },
    "D10.2": {
        "paradigm": "up_aligned_zoom",
        "transform": "homography",
        "scale_mode": "individual",
        "pp_method": "shift_to_256",
        "skew_correction": True,
        "up_alignment": True,
        "up_source": "camera_y_mean",  # No vertical_lines.npz dependency
        "adaptive_zoom": False,
        "zoom": 1.3,
        "target_fx": 548.9937744140625,
        "output_size": 512,
        "description": "D8.1 zoom + Up-alignment from camera Y-axis mean",
        "ray_error": "~0 deg",
        "active": True,
        "note": "Fallback: uses camera Y-axis mean as up. No vertical_lines.npz needed.",
    },
    "D10.3": {
        "paradigm": "precision_homography",      # Based on D7_1, NOT up_aligned
        "transform": "homography",         # Skew correction (D8 style)
        "scale_mode": "individual",
        "pp_method": "shift_to_256",
        "skew_correction": True,
        "up_alignment": False,              # ★ Disabled (not verified)
        "adaptive_zoom": True,
        "zoom_method": "coverage_based",
        "target_fg_coverage": 0.05,         # Target 5% after transform
        "min_fg_coverage": 0.03,
        "zoom_range": [1.0, 2.5],
        "zoom_after_transform": True,       # ★ NEW: Calculate coverage AFTER transform
        "target_fx": 548.9937744140625,
        "output_size": 512,
        "single_folder": True,
        "description": "M3: D7_1 + Homography + Adaptive Zoom (5% FG target)",
        "ray_error": "~0 deg",
        "active": True,
        "experimental": True,
        "note": "Based on verified D7_1. Adds skew correction and coverage-based zoom.",
    },
    # ============================================================
    # P0-P2 Experiment Presets (2026-01-25)
    # ============================================================
    
    # P0: M3 with post-zoom normalization (fx=549, PP=256)
    "M3_norm": {
        "paradigm": "precision_homography",
        "transform": "homography",
        "scale_mode": "individual",
        "pp_method": "shift_to_256",
        "skew_correction": True,
        "up_alignment": False,
        "adaptive_zoom": True,
        "zoom_method": "coverage_based",
        "target_fg_coverage": 0.05,
        "min_fg_coverage": 0.03,
        "zoom_range": [1.0, 2.5],
        "zoom_after_transform": True,
        "target_fx": 548.9937744140625,
        "output_size": 512,
        "single_folder": True,
        # ★ P0 FIX: Enable post-zoom normalization
        "normalize_after_zoom": True,
        "force_pp_to_target": False,  # Changed: keep accurate PP
        "description": "M3 fixed: coverage zoom + fx/PP normalization",
        "active": True,
    },
    
    # P1-1: D7.1 preserving original aspect ratio
    "D7_1_aspect": {
        "paradigm": "pp_centered_shift",
        "transform": "affine",
        "scale_mode": "individual",  # Keep fx/fy ratio
        "pp_method": "shift_to_256",
        "target_fx": 548.9937744140625,
        "output_size": 512,
        "single_folder": True,
        # ★ P1 EXPERIMENT: Preserve aspect ratio (fx != fy allowed)
        "preserve_aspect_ratio": True,
        "description": "D7.1 with original aspect ratio preserved",
        "active": True,
    },
    
    # P1-2: Per-sample adaptive zoom (variable fx experiment)
    "M3_persample": {  # ⛔ DEPRECATED: Identical to M3_2, use M3_2 instead
        "deprecated": True,  # 260126: Merged into M3_2
        "paradigm": "precision_homography",
        "transform": "homography",
        "scale_mode": "individual",
        "pp_method": "shift_to_256",
        "skew_correction": True,
        "up_alignment": False,
        "adaptive_zoom": True,
        "zoom_method": "coverage_based",
        "zoom_scope": "per_sample",  # ★ Per-sample zoom (vs global)
        "zoom_center_mode": "image",  # ★ MVG-correct: PP=256 guaranteed
        "target_fg_coverage": 0.05,
        "min_fg_coverage": 0.03,
        "zoom_range": [1.0, 1.8],  # ★ Reduced to prevent clipping
        "zoom_after_transform": True,
        "target_fx": 548.9937744140625,
        "output_size": 512,
        "single_folder": True,
        "normalize_after_zoom": True,
        "force_pp_to_target": False,
        "description": "Per-sample adaptive zoom + fx normalization",
        "active": True,
    },
    
    # P0 FIX: M3 with MVG-correct center-aligned zoom (PP=256 guaranteed)
    # MVG-correct version of M3_norm (global zoom, center-aligned)
    "M3_1": {
        "paradigm": "precision_homography",
        "transform": "homography",
        "scale_mode": "individual",
        "pp_method": "shift_to_256",
        "skew_correction": True,
        "up_alignment": False,
        "adaptive_zoom": True,
        "zoom_method": "coverage_based",
        "zoom_center_mode": "image",  # MVG-correct: center-aligned zoom
        "target_fg_coverage": 0.05,
        "min_fg_coverage": 0.03,
        "zoom_range": [1.0, 1.8],  # ★ Reduced to prevent clipping
        "zoom_after_transform": True,
        "target_fx": 548.9937744140625,
        "output_size": 512,
        "single_folder": True,
        "normalize_after_zoom": True,
        "force_pp_to_target": False,
        "description": "MVG-correct global zoom (PP=256). Docs: M3_SERIES_SPEC.md",
        "active": True,
    },

    "M3_2": {
        "paradigm": "precision_homography",
        "transform": "homography",
        "scale_mode": "individual",
        "pp_method": "shift_to_256",
        "skew_correction": True,
        "up_alignment": False,
        "adaptive_zoom": True,
        "zoom_method": "coverage_based",
        "zoom_scope": "per_sample",
        "zoom_center_mode": "image",  # ★ MVG-correct: center-aligned zoom
        "target_fg_coverage": 0.05,
        "min_fg_coverage": 0.03,
        "zoom_range": [1.0, 1.8],  # ★ Reduced to prevent clipping
        "zoom_after_transform": True,
        "target_fx": 548.9937744140625,
        "output_size": 512,
        "single_folder": True,
        "normalize_after_zoom": True,
        "force_pp_to_target": False,  # Not needed with center-aligned zoom
        "description": "MVG-correct per-sample zoom (PP=256). Docs: M3_SERIES_SPEC.md",
        "active": True,
    },
}


# Version hierarchy for organization
VERSION_HIERARCHY = {
    "recommended": "M3_2",  # MVG-correct preset

    # === 기하학적 변환 기준 분류 ===
    
    # Affine 변환: 회전, 스케일, 이동 (M1 계열)
    "affine": ["D7", "D7.1", "D7.2"],
    
    # Homography 변환: affine + skew 보정 (M2 계열)
    "homography": ["D8", "D8.1", "D8.2"],
    
    # Homography + Adaptive Coverage Zoom (M3 계열)
    "homography_zoom": ["D10.3"],
    
    # === 특수/실험적 ===
    
    # Up-alignment 실험 (93도 회전 문제 있음)
    "experimental": ["D10", "D10.1", "D10.2"],
    
    # Native: 변환 없음, 원본 유지
    "native": ["D9", "D9_norm", "D9_resized"],
    
    # === 사용 금지 ===
    
    # Geometry Broken: centering/scaling만, PP 미보정 (ray error 심각)
    "geometry_broken": ["D1", "D4", "D6-1", "D6-2", "D6-3"],
}

# M-Series 권장 매핑
M_SERIES = {
    "M1": "D7.1",       # affine, 안정적 기준선
    "M2": "D8",         # homography, 정밀 기하학
    "M3": "D10.3",      # homography_zoom (⚠️ fx=739 버그)
    "M4": "M3_norm",    # ★ M3 fixed: fx=549, PP=256
    "M1a": "D7_1_aspect",  # M1 + aspect ratio preserved
}

# 권장 설정
RECOMMENDED = {
    "stable": "D7.1",       # M1: 검증된 안정적 설정
    "precision": "D8",      # M2: 정밀 기하학
    "production": "M3_norm", # ★ M4: coverage zoom + proper normalization
}





# Alias mapping for M-series to D-series
ALIASES = {
    "M1": "D7.1",
    "M2": "D8",
    "M3": "D10.3",      # Legacy (has fx bug)
    "M4": "M3_norm",    # ★ Recommended
    "M1a": "D7_1_aspect",
}


def resolve_alias(name: str) -> str:
    """Resolve preset alias (e.g., M3 -> D10.3)."""
    return ALIASES.get(name, name)

def get_preset(name: str) -> dict:
    """Get preset configuration by name."""
    name = resolve_alias(name)
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

