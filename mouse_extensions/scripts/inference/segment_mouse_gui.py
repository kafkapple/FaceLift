#!/usr/bin/env python3
"""Simple click-based mouse segmentation GUI using SAM.

Usage:
    python -m mouse_extensions.scripts.inference.segment_mouse_gui \
        --input_dir /path/to/images \
        --output_dir /path/to/output \
        --sam_checkpoint checkpoints/sam/sam_vit_b.pth

Controls:
    - Left click: Add foreground point (green)
    - Right click: Add background point (red)
    - 'm': Generate mask from points
    - 'r': Reset points
    - 's': Save current mask and move to next image
    - 'q': Quit
    - 'n': Next image (skip current)
    - 'p': Previous image
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


class MouseSegmentationGUI:
    """Interactive GUI for mouse segmentation using SAM."""
    
    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_b",
        device: str = "cuda",
    ):
        """Initialize SAM model."""
        self.device = device
        self.points: List[Tuple[int, int]] = []
        self.labels: List[int] = []  # 1=foreground, 0=background
        self.current_mask: Optional[np.ndarray] = None
        self.image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None
        
        # Load SAM
        print("Loading SAM (" + model_type + ") from " + sam_checkpoint + "...")
        from segment_anything import sam_model_registry, SamPredictor
        
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device)
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)
        print("SAM loaded.")
        
    def load_image(self, image_path: str):
        """Load and set image for segmentation."""
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.display_image = self.image.copy()
        self.points = []
        self.labels = []
        self.current_mask = None
        
        # Set image in predictor
        self.predictor.set_image(self.image)
        
    def add_point(self, x: int, y: int, is_foreground: bool):
        """Add a point (foreground or background)."""
        self.points.append((x, y))
        self.labels.append(1 if is_foreground else 0)
        self._update_display()
        
    def reset_points(self):
        """Clear all points."""
        self.points = []
        self.labels = []
        self.current_mask = None
        self._update_display()
        
    def generate_mask(self):
        """Generate mask from current points."""
        if not self.points:
            print("No points added. Left-click to add foreground, right-click for background.")
            return
            
        points_np = np.array(self.points)
        labels_np = np.array(self.labels)
        
        masks, scores, _ = self.predictor.predict(
            point_coords=points_np,
            point_labels=labels_np,
            multimask_output=True,
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        self.current_mask = masks[best_idx]
        print("Generated mask (score: " + str(round(scores[best_idx], 3)) + ")")
        self._update_display()
        
    def _update_display(self):
        """Update display image with points and mask overlay."""
        self.display_image = self.image.copy()
        
        # Draw mask overlay
        if self.current_mask is not None:
            mask_overlay = np.zeros_like(self.display_image)
            mask_overlay[self.current_mask] = [0, 255, 0]  # Green mask
            self.display_image = cv2.addWeighted(
                self.display_image, 0.7, mask_overlay, 0.3, 0
            )
            
        # Draw points
        for (x, y), label in zip(self.points, self.labels):
            color = (0, 255, 0) if label == 1 else (255, 0, 0)  # Green=fg, Red=bg
            cv2.circle(self.display_image, (x, y), 8, color, -1)
            cv2.circle(self.display_image, (x, y), 8, (255, 255, 255), 2)
            
    def get_result(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get processed image (white bg) and mask."""
        if self.current_mask is None:
            raise ValueError("No mask generated. Press 'm' to generate mask.")
            
        # Create white background composite
        mask_3ch = self.current_mask[:, :, np.newaxis].astype(np.float32)
        white_bg = np.ones_like(self.image) * 255
        result = (self.image * mask_3ch + white_bg * (1 - mask_3ch)).astype(np.uint8)
        
        mask_uint8 = (self.current_mask * 255).astype(np.uint8)
        return result, mask_uint8


def mouse_callback(event, x, y, flags, param):
    """Mouse callback for OpenCV window."""
    gui: MouseSegmentationGUI = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        gui.add_point(x, y, is_foreground=True)
    elif event == cv2.EVENT_RBUTTONDOWN:
        gui.add_point(x, y, is_foreground=False)


def run_gui(
    input_dir: str,
    output_dir: str,
    sam_checkpoint: str,
    model_type: str = "vit_b",
    device: str = "cuda",
    target_size: int = 512,
):
    """Run interactive segmentation GUI."""
    # Find images
    input_path = Path(input_dir)
    image_files = sorted(
        [f for f in input_path.iterdir() 
         if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    )
    
    if not image_files:
        print("No images found in " + input_dir)
        return
        
    print("Found " + str(len(image_files)) + " images")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize GUI
    gui = MouseSegmentationGUI(sam_checkpoint, model_type, device)
    
    # Create window
    window_name = "Mouse Segmentation (L=fg, R=bg, M=mask, S=save, R=reset, Q=quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback, gui)
    
    current_idx = 0
    
    while current_idx < len(image_files):
        image_file = image_files[current_idx]
        print("\n[" + str(current_idx+1) + "/" + str(len(image_files)) + "] " + image_file.name)
        
        gui.load_image(str(image_file))
        
        while True:
            # Display
            display = cv2.cvtColor(gui.display_image, cv2.COLOR_RGB2BGR)
            
            # Add status text
            status = "[" + str(current_idx+1) + "/" + str(len(image_files)) + "] " + image_file.name
            if gui.current_mask is not None:
                coverage = gui.current_mask.sum() / gui.current_mask.size * 100
                status += " | Mask: " + str(round(coverage, 1)) + "%"
            cv2.putText(display, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "L=fg R=bg M=mask S=save R=reset N=next Q=quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord('m'):
                gui.generate_mask()
            elif key == ord('r'):
                gui.reset_points()
            elif key == ord('s'):
                if gui.current_mask is not None:
                    result, mask = gui.get_result()
                    
                    # Resize to target size
                    result_resized = cv2.resize(result, (target_size, target_size))
                    mask_resized = cv2.resize(mask, (target_size, target_size))
                    
                    # Save
                    stem = image_file.stem
                    cv2.imwrite(str(output_path / (stem + "_white.png")), 
                               cv2.cvtColor(result_resized, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(output_path / (stem + "_mask.png")), mask_resized)
                    print("  Saved: " + stem + "_white.png, " + stem + "_mask.png")
                    
                    current_idx += 1
                    break
                else:
                    print("Generate mask first (press 'm')")
            elif key == ord('n'):
                current_idx += 1
                break
            elif key == ord('p'):
                current_idx = max(0, current_idx - 1)
                break
                
    cv2.destroyAllWindows()
    print("\nDone! Outputs saved to: " + str(output_path))


def main():
    parser = argparse.ArgumentParser(description="Interactive mouse segmentation with SAM")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed images")
    parser.add_argument("--sam_checkpoint", type=str, 
                        default="checkpoints/sam/sam_vit_b.pth",
                        help="SAM checkpoint path")
    parser.add_argument("--model_type", type=str, default="vit_b",
                        choices=["vit_h", "vit_l", "vit_b"],
                        help="SAM model type")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_size", type=int, default=512,
                        help="Output image size")
    
    args = parser.parse_args()
    
    run_gui(
        args.input_dir,
        args.output_dir,
        args.sam_checkpoint,
        args.model_type,
        args.device,
        args.target_size,
    )


if __name__ == "__main__":
    main()
