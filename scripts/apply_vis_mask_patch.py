#!/usr/bin/env python3
"""
Apply visualization mask consistency patch to gslrm.py
"""

import sys
from pathlib import Path

def main():
    gslrm_path = Path("/home/joon/dev/FaceLift/gslrm/model/gslrm.py")
    
    content = gslrm_path.read_text()
    
    # Check if already patched
    if 'vis_mask_type = "none"' in content:
        print("Already patched!")
        return
    
    # Patch 1: save_evaluation_results (around line 2022-2028)
    old_code_1 = '''                # Compute pred mask from rendered (removebg style) - use config threshold
                pred_threshold = self.config.training.losses.get("pred_mask_threshold", 0.1)
                color_distance = (rendered_images - 1.0).abs().mean(dim=1, keepdim=True)
                pred_mask = (color_distance > pred_threshold).float()
                pred_mask_rgb = pred_mask.expand(-1, 3, -1, -1)
                pred_mask_overlay = pred_mask_rgb * fg_color + (1 - pred_mask_rgb) * bg_color
                rendered_with_mask = rendered_images * 0.7 + pred_mask_overlay * 0.3'''
    
    new_code_1 = '''                # Compute pred mask using mask_mode for consistency with loss
                mask_mode = self.config.training.losses.get("mask_mode", None)
                alpha_threshold = self.config.training.losses.get("alpha_mask_threshold", 0.5)
                pred_threshold = self.config.training.losses.get("pred_mask_threshold", 0.1)
                
                # Get rendered_alpha if available
                rendered_alpha_batch = model_results.get("rendered_alpha", None)
                if rendered_alpha_batch is not None:
                    rendered_alpha_batch = rendered_alpha_batch[batch_idx]  # [V, 1, H, W]
                
                if mask_mode == "none":
                    # No mask: show rendered without overlay
                    rendered_with_mask = rendered_images
                    vis_mask_type = "none"
                elif mask_mode == "gt":
                    # GT mask mode: use GT mask for rendered too
                    vis_mask = gt_mask
                    vis_mask_rgb = vis_mask.expand(-1, 3, -1, -1)
                    vis_mask_overlay = vis_mask_rgb * fg_color + (1 - vis_mask_rgb) * bg_color
                    rendered_with_mask = rendered_images * 0.7 + vis_mask_overlay * 0.3
                    vis_mask_type = "gt"
                elif mask_mode == "alpha" and rendered_alpha_batch is not None:
                    # Alpha mask mode: use rendered alpha
                    vis_mask = (rendered_alpha_batch > alpha_threshold).float()
                    vis_mask_rgb = vis_mask.expand(-1, 3, -1, -1)
                    vis_mask_overlay = vis_mask_rgb * fg_color + (1 - vis_mask_rgb) * bg_color
                    rendered_with_mask = rendered_images * 0.7 + vis_mask_overlay * 0.3
                    vis_mask_type = f"alpha>{alpha_threshold}"
                else:
                    # Fallback: RGB pred mask (legacy)
                    color_distance = (rendered_images - 1.0).abs().mean(dim=1, keepdim=True)
                    vis_mask = (color_distance > pred_threshold).float()
                    vis_mask_rgb = vis_mask.expand(-1, 3, -1, -1)
                    vis_mask_overlay = vis_mask_rgb * fg_color + (1 - vis_mask_rgb) * bg_color
                    rendered_with_mask = rendered_images * 0.7 + vis_mask_overlay * 0.3
                    vis_mask_type = f"rgb>{pred_threshold}"'''
    
    if old_code_1 in content:
        content = content.replace(old_code_1, new_code_1)
        print("Patch 1 applied: save_evaluation_results")
    else:
        print("Warning: Patch 1 pattern not found")
    
    # Patch 2: save_validations (around line 2266-2271) - uses 'rendered' instead of 'rendered_images'
    old_code_2 = '''                    # Compute pred mask from rendered image (removebg style) - use config threshold
                    # Pixels far from white (1.0) are foreground
                    pred_threshold = self.config.training.losses.get("pred_mask_threshold", 0.1)
                    color_distance = (rendered - 1.0).abs().mean(dim=1, keepdim=True)
                    pred_mask = (color_distance > pred_threshold).float()
                    pred_mask_rgb = pred_mask.expand(-1, 3, -1, -1)
                    pred_mask_overlay = pred_mask_rgb * fg_color + (1 - pred_mask_rgb) * bg_color
                    rendered_with_mask = rendered * 0.7 + pred_mask_overlay * 0.3'''
    
    new_code_2 = '''                    # Compute pred mask using mask_mode for consistency with loss
                    mask_mode = self.config.training.losses.get("mask_mode", None)
                    alpha_threshold = self.config.training.losses.get("alpha_mask_threshold", 0.5)
                    pred_threshold = self.config.training.losses.get("pred_mask_threshold", 0.1)
                    
                    # Get rendered_alpha if available
                    rendered_alpha_batch = model_results.get("rendered_alpha", None)
                    if rendered_alpha_batch is not None:
                        rendered_alpha_batch = rendered_alpha_batch[batch_idx]  # [V, 1, H, W]
                    
                    if mask_mode == "none":
                        # No mask: show rendered without overlay
                        rendered_with_mask = rendered
                        vis_mask_type = "none"
                    elif mask_mode == "gt":
                        # GT mask mode: use GT mask for rendered too
                        vis_mask = gt_mask
                        vis_mask_rgb = vis_mask.expand(-1, 3, -1, -1)
                        vis_mask_overlay = vis_mask_rgb * fg_color + (1 - vis_mask_rgb) * bg_color
                        rendered_with_mask = rendered * 0.7 + vis_mask_overlay * 0.3
                        vis_mask_type = "gt"
                    elif mask_mode == "alpha" and rendered_alpha_batch is not None:
                        # Alpha mask mode: use rendered alpha
                        vis_mask = (rendered_alpha_batch > alpha_threshold).float()
                        vis_mask_rgb = vis_mask.expand(-1, 3, -1, -1)
                        vis_mask_overlay = vis_mask_rgb * fg_color + (1 - vis_mask_rgb) * bg_color
                        rendered_with_mask = rendered * 0.7 + vis_mask_overlay * 0.3
                        vis_mask_type = f"alpha>{alpha_threshold}"
                    else:
                        # Fallback: RGB pred mask (legacy)
                        color_distance = (rendered - 1.0).abs().mean(dim=1, keepdim=True)
                        vis_mask = (color_distance > pred_threshold).float()
                        vis_mask_rgb = vis_mask.expand(-1, 3, -1, -1)
                        vis_mask_overlay = vis_mask_rgb * fg_color + (1 - vis_mask_rgb) * bg_color
                        rendered_with_mask = rendered * 0.7 + vis_mask_overlay * 0.3
                        vis_mask_type = f"rgb>{pred_threshold}"'''
    
    if old_code_2 in content:
        content = content.replace(old_code_2, new_code_2)
        print("Patch 2 applied: save_validations")
    else:
        print("Warning: Patch 2 pattern not found")
    
    # Write back
    gslrm_path.write_text(content)
    print("\nPatches completed!")
    print("\nChanges:")
    print("  - save_evaluation_results: uses mask_mode (none/gt/alpha) for pred overlay")
    print("  - save_validations: uses mask_mode (none/gt/alpha) for pred overlay")

if __name__ == "__main__":
    main()
