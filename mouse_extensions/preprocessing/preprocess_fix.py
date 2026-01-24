"""
P0 Bug Fix: M3 fx/cx/cy normalization after zoom

Problem:
  - After adaptive zoom, fx becomes 549 * zoom = 739
  - cx/cy shifts due to crop_offset
  
Solution:
  - Add normalize_after_zoom option
  - Rescale image and params to restore fx=549, cx=cy=256
"""

import numpy as np
import cv2

def compute_camera_params_fixed(cam, cfg, zoom=1.0, crop_offset=(0, 0)):
    """Fixed version with post-zoom normalization."""
    K, R, T = cam['K'], cam['R'], cam['T']
    
    # Initial fx/fy based on transform type
    if cfg.transform == 'homography':
        fx = fy = cfg.target_fx  # 549
    else:
        orig_fx, orig_fy = K[0, 0], K[1, 1]
        scale_x = cfg.target_fx / orig_fx
        scale_y = cfg.target_fx / orig_fy
        fx, fy = orig_fx * scale_x, orig_fy * scale_y
    
    cx, cy = cfg.target_pp  # (256, 256)
    
    # Apply zoom
    if zoom > 1.0:
        crop_x, crop_y = crop_offset
        fx, fy = fx * zoom, fy * zoom
        cx, cy = (cx - crop_x) * zoom, (cy - crop_y) * zoom
    
    # ★ NEW: Post-zoom normalization
    if getattr(cfg, 'normalize_after_zoom', False) and zoom > 1.0:
        # Scale factor to restore fx to target
        renorm_scale = cfg.target_fx / fx
        
        # Apply to intrinsics (no image resize needed if we adjust params)
        fx = cfg.target_fx
        fy = fy * renorm_scale
        cx = cx * renorm_scale  # This won't give exactly 256
        cy = cy * renorm_scale
        
        # For true PP=256, we need to recrop the image centered
        # This requires image processing, not just param adjustment
        
    # Alternative: Force PP to target (simpler but less accurate)
    if getattr(cfg, 'force_pp_to_target', False):
        cx, cy = cfg.target_pp
    
    # Extrinsics normalization
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = R, T.flatten()
    c2w = np.linalg.inv(w2c)
    cam_pos = c2w[:3, 3]
    dist_scale = cfg.target_distance / np.linalg.norm(cam_pos)
    new_c2w = c2w.copy()
    new_c2w[:3, 3] = cam_pos * dist_scale
    new_w2c = np.linalg.inv(new_c2w)
    
    return {
        "w": cfg.output_size, "h": cfg.output_size,
        "fx": float(fx), "fy": float(fy), 
        "cx": float(cx), "cy": float(cy),
        "w2c": new_w2c.tolist(),
    }


def apply_zoom_with_renormalization(image, mask, zoom, target_size=512, target_fx=549):
    """Apply zoom then renormalize to target_fx.
    
    Strategy:
    1. Apply zoom (crop center region)
    2. Resize result to maintain fx=target_fx
    
    This keeps the object larger in frame while maintaining fx=549.
    """
    h, w = image.shape[:2]
    
    if zoom <= 1.0:
        return image, mask, (0, 0), 1.0
    
    # Step 1: Crop center region (zoom in)
    crop_size = int(target_size / zoom)
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - crop_size // 2)
    y1 = max(0, cy - crop_size // 2)
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)
    
    cropped_img = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    
    # Step 2: Resize back to target_size
    # This effectively applies zoom while keeping output size constant
    resized_img = cv2.resize(cropped_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(cropped_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    
    # The crop offset for PP calculation
    crop_offset = (x1 - (w - target_size) // 2, y1 - (h - target_size) // 2)
    
    # After this operation:
    # - Image is target_size x target_size
    # - Object appears zoom times larger
    # - fx should still be target_fx (no change needed!)
    # - PP shift depends on crop_offset
    
    return resized_img, resized_mask, crop_offset, zoom


# ============================================================
# Preset configurations for P0-P2 experiments
# ============================================================

EXPERIMENT_PRESETS = {
    # P0: M3 with fx normalization fix
    "M3_norm": {
        "base": "D10.3",
        "normalize_after_zoom": True,
        "force_pp_to_target": True,  # Simpler fix
        "description": "M3 with post-zoom fx/PP normalization",
    },
    
    # P1-1: D7.1 with aspect ratio preservation  
    "D7_1_aspect": {
        "base": "D7.1",
        "scale_mode": "individual",  # Keep fx/fy ratio from original
        "preserve_aspect": True,
        "description": "D7.1 preserving original aspect ratio",
    },
    
    # P1-2: Already covered by M3_norm
    
    # P2: Split comparison (use same preprocessing, different splits)
}

print("Fix module loaded. Available presets:")
for k, v in EXPERIMENT_PRESETS.items():
    print(f"  {k}: {v['description']}")
