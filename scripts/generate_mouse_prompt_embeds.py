#!/usr/bin/env python3
"""
Mouse 카메라용 prompt_embeds 생성 스크립트.

FaceLift의 prompt_embeds는 수평 6방향 (elevation 0°):
- front, front_right, right, back, left, front_left

Mouse 카메라는 경사 6방향 (elevation ~20°):
- 위에서 경사진 각도로 본 6방향

Usage:
    python scripts/generate_mouse_prompt_embeds.py \
        --output_dir mvdiffusion/data/mouse_prompt_embeds_6view \
        --elevation 20
"""

import argparse
import json
import os
from typing import List, Optional

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer


class MousePromptEmbedGenerator:
    """Generator for Mouse camera prompt embeddings."""

    def __init__(
        self,
        model_name: str = 'stabilityai/stable-diffusion-2-1-unclip',
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16
    ):
        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype

        print(f"Loading CLIP models from {model_name}...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder')
        self.text_encoder = self.text_encoder.to(self.device, dtype=self.dtype)

    def _encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts to embeddings."""
        print(f"Encoding {len(prompts)} prompts...")
        for i, p in enumerate(prompts):
            print(f"  [{i}] {p}")

        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            prompt_embeds = self.text_encoder(text_inputs.input_ids)
            prompt_embeds = prompt_embeds[0].detach().cpu()

        print(f"Embeddings shape: {prompt_embeds.shape}")
        return prompt_embeds

    def generate_mouse_embeds(
        self,
        output_dir: str,
        elevation: float = 20.0,
        camera_json: Optional[str] = None
    ):
        """Generate prompt embeddings for Mouse camera setup."""
        os.makedirs(output_dir, exist_ok=True)

        # Mouse 카메라 뷰 정의 (경사 뷰)
        # elevation ~20° 에서 바라보는 6방향
        if elevation > 10:
            # 경사 뷰 (위에서 비스듬히 내려다봄)
            views = [
                "top-front",      # 0: 위에서 앞쪽
                "top-front-right", # 1: 위에서 앞오른쪽
                "top-right",      # 2: 위에서 오른쪽
                "top-back",       # 3: 위에서 뒤쪽
                "top-left",       # 4: 위에서 왼쪽
                "top-front-left"  # 5: 위에서 앞왼쪽
            ]
            view_description = "from above at an angle"
        else:
            # 수평 뷰 (FaceLift 호환)
            views = ["front", "front_right", "right", "back", "left", "front_left"]
            view_description = "from the side"

        # Color prompts
        color_prompts = [
            f"a rendering image of a 3D model, {view} view, {view_description}, color map."
            for view in views
        ]

        # Normal prompts
        normal_prompts = [
            f"a rendering image of a 3D model, {view} view, {view_description}, normal map."
            for view in views
        ]

        # Generate and save
        print("\n=== Generating Color Embeddings ===")
        color_embeds = self._encode_prompts(color_prompts)
        color_path = os.path.join(output_dir, "clr_embeds.pt")
        torch.save(color_embeds.half(), color_path)
        print(f"Saved: {color_path}")

        print("\n=== Generating Normal Embeddings ===")
        normal_embeds = self._encode_prompts(normal_prompts)
        normal_path = os.path.join(output_dir, "normal_embeds.pt")
        torch.save(normal_embeds.half(), normal_path)
        print(f"Saved: {normal_path}")

        # Save metadata
        metadata = {
            "elevation": elevation,
            "views": views,
            "view_description": view_description,
            "color_prompts": color_prompts,
            "normal_prompts": normal_prompts,
            "shape": list(color_embeds.shape),
            "dtype": "float16"
        }

        meta_path = os.path.join(output_dir, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {meta_path}")

        print("\n✅ Mouse prompt embeddings generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Generate Mouse camera prompt embeddings")

    parser.add_argument(
        "--output_dir", type=str,
        default="mvdiffusion/data/mouse_prompt_embeds_6view",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--elevation", type=float, default=20.0,
        help="Camera elevation angle in degrees (0=horizontal, 90=top-down)"
    )
    parser.add_argument(
        "--camera_json", type=str, default=None,
        help="Optional: Camera JSON to extract actual angles"
    )
    parser.add_argument(
        "--model_name", type=str,
        default="checkpoints/mvdiffusion/pipeckpts",
        help="CLIP model to use (local path or HuggingFace model)"
    )

    args = parser.parse_args()

    print("="*60)
    print("Mouse Camera Prompt Embeddings Generator")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Elevation: {args.elevation}°")

    generator = MousePromptEmbedGenerator(model_name=args.model_name)
    generator.generate_mouse_embeds(
        output_dir=args.output_dir,
        elevation=args.elevation,
        camera_json=args.camera_json
    )


if __name__ == "__main__":
    main()
