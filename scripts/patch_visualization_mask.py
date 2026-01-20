#!/usr/bin/env python3
"""
Patch gslrm.py visualization to use consistent mask_mode.

Fixes:
1. save_evaluation_results() - uses hardcoded RGB threshold
2. save_validations() - uses hardcoded RGB threshold  

Both should use mask_mode from config for consistency with loss.
"""

import re
import sys
from pathlib import Path

def create_vis_mask_helper():
    """Helper function code to add."""
    return '''
    def _compute_vis_mask_for_pred(self, rendered_images, gt_mask, rendered_alpha=None):
        """
        Compute visualization mask for predictions based on mask_mode config.
        
        Returns mask tensor matching loss computation mask_mode.
        Also returns mask_type string for labeling.
        """
        losses = self.config.training.losses
        mask_mode = losses.get("mask_mode", None)
        alpha_threshold = losses.get("alpha_mask_threshold", 0.5)
        pred_threshold = losses.get("pred_mask_threshold", 0.1)
        
        device = rendered_images.device
        fg_color = torch.tensor([0.2, 0.8, 0.2], device=device).view(1, 3, 1, 1)
        bg_color = torch.tensor([0.8, 0.2, 0.2], device=device).view(1, 3, 1, 1)
        
        if mask_mode == "none":
            # No mask mode: no overlay
            return rendered_images, "none"
        
        elif mask_mode == "gt":
            # GT mask mode: use same GT mask for rendered
            if gt_mask is not None:
                vis_mask = gt_mask
                vis_mask_rgb = vis_mask.expand(-1, 3, -1, -1)
                vis_mask_overlay = vis_mask_rgb * fg_color + (1 - vis_mask_rgb) * bg_color
                rendered_with_mask = rendered_images * 0.7 + vis_mask_overlay * 0.3
                return rendered_with_mask, "gt"
            else:
                return rendered_images, "none"
        
        elif mask_mode == "alpha":
            # Alpha mask mode: use rendered alpha
            if rendered_alpha is not None:
                vis_mask = (rendered_alpha > alpha_threshold).float()
                vis_mask_rgb = vis_mask.expand(-1, 3, -1, -1)
                vis_mask_overlay = vis_mask_rgb * fg_color + (1 - vis_mask_rgb) * bg_color
                rendered_with_mask = rendered_images * 0.7 + vis_mask_overlay * 0.3
                return rendered_with_mask, f"alpha>{alpha_threshold}"
            else:
                # Fallback to RGB pred
                color_distance = (rendered_images - 1.0).abs().mean(dim=1, keepdim=True)
                vis_mask = (color_distance > pred_threshold).float()
                vis_mask_rgb = vis_mask.expand(-1, 3, -1, -1)
                vis_mask_overlay = vis_mask_rgb * fg_color + (1 - vis_mask_rgb) * bg_color
                rendered_with_mask = rendered_images * 0.7 + vis_mask_overlay * 0.3
                return rendered_with_mask, f"rgb>{pred_threshold}"
        
        else:
            # Fallback: RGB pred mask (legacy)
            color_distance = (rendered_images - 1.0).abs().mean(dim=1, keepdim=True)
            vis_mask = (color_distance > pred_threshold).float()
            vis_mask_rgb = vis_mask.expand(-1, 3, -1, -1)
            vis_mask_overlay = vis_mask_rgb * fg_color + (1 - vis_mask_rgb) * bg_color
            rendered_with_mask = rendered_images * 0.7 + vis_mask_overlay * 0.3
            return rendered_with_mask, f"rgb>{pred_threshold}"
'''

def patch_save_evaluation_results(content):
    """Patch save_evaluation_results to use mask_mode."""
    
    # Find and replace the hardcoded pred mask computation (around line 2023-2028)
    old_pattern = r'''(# Create GT mask overlay.*?gt_with_mask = gt_rgb \* 0\.7 \+ gt_mask_overlay \* 0\.3)
                
                # Compute pred mask from rendered \(removebg style\) - use config threshold
                pred_threshold = self\.config\.training\.losses\.get\("pred_mask_threshold", 0\.1\)
                color_distance = \(rendered_images - 1\.0\)\.abs\(\)\.mean\(dim=1, keepdim=True\)
                pred_mask = \(color_distance > pred_threshold\)\.float\(\)
                pred_mask_rgb = pred_mask\.expand\(-1, 3, -1, -1\)
                pred_mask_overlay = pred_mask_rgb \* fg_color \+ \(1 - pred_mask_rgb\) \* bg_color
                rendered_with_mask = rendered_images \* 0\.7 \+ pred_mask_overlay \* 0\.3'''
    
    new_replacement = r'''\1
                
                # Compute pred mask using mask_mode for consistency with loss
                # Get rendered_alpha if available
                rendered_alpha_for_vis = model_results.get("rendered_alpha", None)
                if rendered_alpha_for_vis is not None:
                    rendered_alpha_for_vis = rendered_alpha_for_vis[batch_idx]  # [V, 1, H, W]
                
                rendered_with_mask, mask_type = self._compute_vis_mask_for_pred(
                    rendered_images, gt_mask, rendered_alpha_for_vis
                )'''
    
    content = re.sub(old_pattern, new_replacement, content, flags=re.DOTALL)
    return content

def patch_save_validations(content):
    """Patch save_validations to use mask_mode."""
    
    # Similar pattern around line 2266-2271
    old_pattern = r'''(# Create GT mask overlay.*?gt_with_mask = gt_rgb \* 0\.7 \+ gt_mask_overlay \* 0\.3)
                    
                    # Compute pred mask from rendered image \(removebg style\) - use config threshold
                    # Pixels far from white \(1\.0\) are foreground
                    pred_threshold = self\.config\.training\.losses\.get\("pred_mask_threshold", 0\.1\)
                    color_distance = \(rendered - 1\.0\)\.abs\(\)\.mean\(dim=1, keepdim=True\)
                    pred_mask = \(color_distance > pred_threshold\)\.float\(\)
                    pred_mask_rgb = pred_mask\.expand\(-1, 3, -1, -1\)
                    pred_mask_overlay = pred_mask_rgb \* fg_color \+ \(1 - pred_mask_rgb\) \* bg_color
                    rendered_with_mask = rendered \* 0\.7 \+ pred_mask_overlay \* 0\.3'''
    
    new_replacement = r'''\1
                    
                    # Compute pred mask using mask_mode for consistency with loss
                    rendered_alpha_for_vis = model_results.get("rendered_alpha", None)
                    if rendered_alpha_for_vis is not None:
                        rendered_alpha_for_vis = rendered_alpha_for_vis[batch_idx]  # [V, 1, H, W]
                    
                    rendered_with_mask, mask_type = self._compute_vis_mask_for_pred(
                        rendered, gt_mask, rendered_alpha_for_vis
                    )'''
    
    content = re.sub(old_pattern, new_replacement, content, flags=re.DOTALL)
    return content

def add_helper_method(content):
    """Add helper method after __init__ or before first method."""
    helper_code = create_vis_mask_helper()
    
    # Find a good insertion point - after _create_visual method
    # Look for the end of _create_visual
    pattern = r'(def _create_visual\(self.*?return visual, error_stats\n)'
    
    match = re.search(pattern, content, re.DOTALL)
    if match:
        insert_pos = match.end()
        content = content[:insert_pos] + helper_code + content[insert_pos:]
    else:
        print("Warning: Could not find insertion point for helper method")
    
    return content

def main():
    gslrm_path = Path("/home/joon/dev/FaceLift/gslrm/model/gslrm.py")
    
    if not gslrm_path.exists():
        print(f"Error: {gslrm_path} not found")
        sys.exit(1)
    
    content = gslrm_path.read_text()
    
    # Check if already patched
    if "_compute_vis_mask_for_pred" in content:
        print("Already patched!")
        sys.exit(0)
    
    # Apply patches
    content = add_helper_method(content)
    content = patch_save_evaluation_results(content)
    content = patch_save_validations(content)
    
    # Write back
    gslrm_path.write_text(content)
    print("Patch applied successfully!")
    print("Changes:")
    print("  1. Added _compute_vis_mask_for_pred() helper method")
    print("  2. Updated save_evaluation_results() to use mask_mode")
    print("  3. Updated save_validations() to use mask_mode")

if __name__ == "__main__":
    main()
