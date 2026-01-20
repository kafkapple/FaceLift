#!/usr/bin/env python3
"""
Add mask type label to visualization images.
"""

from pathlib import Path

def main():
    gslrm_path = Path("/home/joon/dev/FaceLift/gslrm/model/gslrm.py")
    
    content = gslrm_path.read_text()
    
    # Check if already patched
    if 'add_mask_label_to_image' in content:
        print("Label patch already applied!")
        return
    
    # Add helper function import at the top (after existing imports)
    import_marker = 'from PIL import Image'
    if import_marker in content:
        new_import = '''from PIL import Image, ImageDraw, ImageFont'''
        content = content.replace(import_marker, new_import)
        print("Updated PIL import")
    
    # Patch 1: Add mask label to save_evaluation_results
    old_save_1 = '''            comparison_image = (comparison_image.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
            Image.fromarray(comparison_image).save(os.path.join(item_output_dir, "gt_vs_pred.png"))'''
    
    new_save_1 = '''            comparison_image = (comparison_image.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
            
            # Add mask type label if available
            if gt_images.size(1) == 4:  # Has mask info
                try:
                    pil_img = Image.fromarray(comparison_image)
                    draw = ImageDraw.Draw(pil_img)
                    # Add label at top-left
                    label_text = f"Mask: {vis_mask_type}"
                    draw.rectangle([0, 0, 150, 20], fill=(0, 0, 0))
                    draw.text((5, 2), label_text, fill=(255, 255, 0))
                    pil_img.save(os.path.join(item_output_dir, "gt_vs_pred.png"))
                except:
                    Image.fromarray(comparison_image).save(os.path.join(item_output_dir, "gt_vs_pred.png"))
            else:
                Image.fromarray(comparison_image).save(os.path.join(item_output_dir, "gt_vs_pred.png"))'''
    
    if old_save_1 in content:
        content = content.replace(old_save_1, new_save_1)
        print("Patch 1 applied: mask label in save_evaluation_results")
    else:
        print("Warning: Patch 1 pattern not found - may already be patched differently")
    
    # Patch 2: Add mask label to save_validations - find the second occurrence
    # This needs a different marker since there are multiple similar patterns
    old_save_2 = '''                comparison_image = (comparison_image.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)

                # Add error scale annotation to the last row
                comparison_image = self.loss_calculator._add_error_scale_annotation(
                    comparison_image, error_stats, h, num_rows
                )

                Image.fromarray(comparison_image).save(os.path.join(item_output_dir, "gt_vs_pred.png"))'''
    
    new_save_2 = '''                comparison_image = (comparison_image.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)

                # Add error scale annotation to the last row
                comparison_image = self.loss_calculator._add_error_scale_annotation(
                    comparison_image, error_stats, h, num_rows
                )

                # Add mask type label if available
                if full_target.size(1) == 4:  # Has mask info
                    try:
                        pil_img = Image.fromarray(comparison_image)
                        draw = ImageDraw.Draw(pil_img)
                        # Add label at top-left
                        label_text = f"Mask: {vis_mask_type}"
                        draw.rectangle([0, 0, 150, 20], fill=(0, 0, 0))
                        draw.text((5, 2), label_text, fill=(255, 255, 0))
                        pil_img.save(os.path.join(item_output_dir, "gt_vs_pred.png"))
                    except:
                        Image.fromarray(comparison_image).save(os.path.join(item_output_dir, "gt_vs_pred.png"))
                else:
                    Image.fromarray(comparison_image).save(os.path.join(item_output_dir, "gt_vs_pred.png"))'''
    
    if old_save_2 in content:
        content = content.replace(old_save_2, new_save_2)
        print("Patch 2 applied: mask label in save_validations")
    else:
        print("Warning: Patch 2 pattern not found")
    
    # Write back
    gslrm_path.write_text(content)
    print("\nLabel patches completed!")

if __name__ == "__main__":
    main()
