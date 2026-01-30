#!/usr/bin/env python3
"""Web-based mouse segmentation GUI using SAM + Gradio."""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import gradio as gr
from PIL import Image


class SAMSegmenter:
    def __init__(self, checkpoint: str, model_type: str = "vit_b", device: str = "cuda"):
        print(f"Loading SAM ({model_type})...")
        from segment_anything import sam_model_registry, SamPredictor
        
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device)
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)
        self.last_masks = None
        self.last_scores = None
        print("SAM loaded.")
        
    def set_image(self, image: np.ndarray):
        self.predictor.set_image(image)
        self.last_masks = None
        self.last_scores = None
        
    def predict(self, points: List[Tuple[int, int]], labels: List[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (masks[3], scores[3])"""
        if not points:
            return None, None
            
        points_np = np.array(points, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.int32)
        
        masks, scores, _ = self.predictor.predict(
            point_coords=points_np,
            point_labels=labels_np,
            multimask_output=True,
        )
        
        self.last_masks = masks
        self.last_scores = scores
        return masks, scores


# Global
segmenter: Optional[SAMSegmenter] = None
image_files: List[Path] = []
output_dir: Optional[Path] = None


def draw_overlay(image: np.ndarray, fg_pts: List, bg_pts: List, mask: Optional[np.ndarray]) -> np.ndarray:
    display = image.copy()
    
    if mask is not None:
        overlay = np.zeros_like(display)
        overlay[mask] = [0, 255, 0]
        display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)
    
    for x, y in fg_pts:
        cv2.circle(display, (int(x), int(y)), 8, (0, 255, 0), -1)
        cv2.circle(display, (int(x), int(y)), 8, (255, 255, 255), 2)
    
    for x, y in bg_pts:
        cv2.circle(display, (int(x), int(y)), 8, (255, 0, 0), -1)
        cv2.circle(display, (int(x), int(y)), 8, (255, 255, 255), 2)
    
    return display


def load_image(idx: int, state: Dict) -> Tuple[np.ndarray, str, Dict]:
    if not image_files or idx < 0 or idx >= len(image_files):
        return None, "No images", state
    
    img_path = image_files[idx]
    img = np.array(Image.open(img_path).convert("RGB"))
    segmenter.set_image(img)
    
    state = {
        "idx": idx,
        "image": img,
        "fg_points": [],
        "bg_points": [],
        "mask": None,
        "all_masks": None,
        "all_scores": None,
        "selected_idx": 0,
    }
    
    return img.copy(), f"[{idx+1}/{len(image_files)}] {img_path.name}", state


def on_click(evt: gr.SelectData, mode: str, state: Dict) -> Tuple[np.ndarray, str, Dict]:
    if state is None or state.get("image") is None:
        return None, "Load image first", state or {}
    
    x, y = evt.index
    
    if mode == "Foreground (Green)":
        state["fg_points"].append((x, y))
    else:
        state["bg_points"].append((x, y))
    
    display = draw_overlay(state["image"], state["fg_points"], state["bg_points"], state.get("mask"))
    n_fg, n_bg = len(state["fg_points"]), len(state["bg_points"])
    return display, f"FG:{n_fg} BG:{n_bg}", state


def gen_mask(mask_size: str, state: Dict) -> Tuple[np.ndarray, str, Dict]:
    """Generate mask. mask_size: 'Large', 'Medium', 'Small', 'Auto (Best Score)'"""
    if state is None or state.get("image") is None:
        return None, "Load image first", state or {}
    
    fg = state.get("fg_points", [])
    bg = state.get("bg_points", [])
    
    if not fg and not bg:
        return draw_overlay(state["image"], fg, bg, None), "Add points first", state
    
    all_pts = fg + bg
    all_labels = [1] * len(fg) + [0] * len(bg)
    
    masks, scores = segmenter.predict(all_pts, all_labels)
    
    if masks is None:
        return draw_overlay(state["image"], fg, bg, None), "Prediction failed", state
    
    state["all_masks"] = masks
    state["all_scores"] = scores
    
    # Select mask based on size option
    # SAM returns: [0]=largest, [1]=medium, [2]=smallest (roughly)
    if mask_size == "Auto (Best Score)":
        idx = int(np.argmax(scores))
    elif mask_size == "Large":
        # Find mask with most pixels
        areas = [m.sum() for m in masks]
        idx = int(np.argmax(areas))
    elif mask_size == "Small":
        areas = [m.sum() for m in masks]
        idx = int(np.argmin(areas))
    else:  # Medium
        areas = [m.sum() for m in masks]
        sorted_idx = np.argsort(areas)
        idx = int(sorted_idx[1])  # middle one
    
    state["selected_idx"] = idx
    state["mask"] = masks[idx]
    
    cov = masks[idx].sum() / masks[idx].size * 100
    score = scores[idx]
    
    display = draw_overlay(state["image"], fg, bg, masks[idx])
    
    # Show all options
    areas = [f"{m.sum()/m.size*100:.1f}%" for m in masks]
    return display, f"Mask {idx+1}/3 | Coverage: {cov:.1f}% | Score: {score:.3f}\nAll: {areas}", state


def cycle_mask(state: Dict) -> Tuple[np.ndarray, str, Dict]:
    """Cycle through 3 masks."""
    if state is None or state.get("all_masks") is None:
        return state.get("image") if state else None, "Generate mask first", state or {}
    
    masks = state["all_masks"]
    scores = state["all_scores"]
    idx = (state.get("selected_idx", 0) + 1) % 3
    
    state["selected_idx"] = idx
    state["mask"] = masks[idx]
    
    cov = masks[idx].sum() / masks[idx].size * 100
    display = draw_overlay(state["image"], state["fg_points"], state["bg_points"], masks[idx])
    
    return display, f"Mask {idx+1}/3 | Coverage: {cov:.1f}% | Score: {scores[idx]:.3f}", state


def reset_pts(state: Dict) -> Tuple[np.ndarray, str, Dict]:
    if state is None or state.get("image") is None:
        return None, "No image", state or {}
    
    state["fg_points"] = []
    state["bg_points"] = []
    state["mask"] = None
    state["all_masks"] = None
    
    return state["image"].copy(), "Cleared", state


def undo_pt(state: Dict) -> Tuple[np.ndarray, str, Dict]:
    if state is None or state.get("image") is None:
        return None, "No image", state or {}
    
    if state.get("bg_points"):
        state["bg_points"].pop()
    elif state.get("fg_points"):
        state["fg_points"].pop()
    
    display = draw_overlay(state["image"], state["fg_points"], state["bg_points"], state.get("mask"))
    return display, f"FG:{len(state['fg_points'])} BG:{len(state['bg_points'])}", state


def save_next(state: Dict) -> Tuple[np.ndarray, str, Dict]:
    if state is None or state.get("mask") is None:
        return state.get("image") if state else None, "Generate mask first!", state or {}
    
    idx = state["idx"]
    mask = state["mask"]
    img = state["image"]
    
    m3 = mask[:, :, np.newaxis].astype(np.float32)
    white = np.ones_like(img) * 255
    result = (img * m3 + white * (1 - m3)).astype(np.uint8)
    
    result_512 = cv2.resize(result, (512, 512))
    mask_512 = cv2.resize((mask * 255).astype(np.uint8), (512, 512))
    
    stem = image_files[idx].stem
    Image.fromarray(result_512).save(output_dir / f"{stem}.png")
    Image.fromarray(mask_512).save(output_dir / f"{stem}_mask.png")
    
    if idx < len(image_files) - 1:
        return load_image(idx + 1, state)
    return state["image"], f"Saved {stem} | All done!", state


def go_prev(state: Dict) -> Tuple[np.ndarray, str, Dict]:
    idx = state.get("idx", 0) if state else 0
    return load_image(max(0, idx - 1), state)


def go_next(state: Dict) -> Tuple[np.ndarray, str, Dict]:
    idx = state.get("idx", 0) if state else 0
    return load_image(min(len(image_files) - 1, idx + 1), state)


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="Mouse Segmentation") as app:
        state = gr.State({})
        
        gr.Markdown("# 🐭 Mouse Segmentation Tool")
        gr.Markdown("1. Click **FG** on mouse, **BG** on background\n2. **Generate Mask** (try different sizes)\n3. **Save & Next**")
        
        with gr.Row():
            with gr.Column(scale=3):
                img_display = gr.Image(label="Click to add points", type="numpy", height=550)
            
            with gr.Column(scale=1):
                status = gr.Textbox(label="Status", lines=3)
                
                mode = gr.Radio(
                    ["Foreground (Green)", "Background (Red)"],
                    value="Foreground (Green)",
                    label="Click Mode"
                )
                
                gr.Markdown("---")
                
                mask_size = gr.Radio(
                    ["Auto (Best Score)", "Large", "Medium", "Small"],
                    value="Auto (Best Score)",
                    label="Mask Size"
                )
                
                with gr.Row():
                    mask_btn = gr.Button("🎭 Generate Mask", variant="primary")
                    cycle_btn = gr.Button("🔄 Cycle 1/2/3", size="sm")
                
                gr.Markdown("---")
                
                with gr.Row():
                    gr.Button("⬅️", size="sm").click(go_prev, [state], [img_display, status, state])
                    gr.Button("➡️ Skip", size="sm").click(go_next, [state], [img_display, status, state])
                
                with gr.Row():
                    gr.Button("↩️ Undo", size="sm").click(undo_pt, [state], [img_display, status, state])
                    gr.Button("🗑️ Reset", size="sm").click(reset_pts, [state], [img_display, status, state])
                
                gr.Markdown("---")
                gr.Button("💾 Save & Next", variant="primary").click(save_next, [state], [img_display, status, state])
                
                gr.Markdown("🟢=FG 🔴=BG 💚=Mask")
        
        img_display.select(on_click, [mode, state], [img_display, status, state])
        mask_btn.click(gen_mask, [mask_size, state], [img_display, status, state])
        cycle_btn.click(cycle_mask, [state], [img_display, status, state])
        app.load(lambda s: load_image(0, s or {}), [state], [img_display, status, state])
    
    return app


def main():
    global segmenter, image_files, output_dir
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam_checkpoint", default="checkpoints/sam/sam_vit_b.pth")
    parser.add_argument("--model_type", default="vit_b")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    image_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    
    if not image_files:
        print(f"No images in {args.input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    output_dir = Path(args.output_dir) if args.output_dir else input_path / "segmented"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    
    segmenter = SAMSegmenter(args.sam_checkpoint, args.model_type, args.device)
    
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
