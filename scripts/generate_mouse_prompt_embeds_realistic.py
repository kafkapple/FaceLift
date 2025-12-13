#!/usr/bin/env python3
"""
Realistic Mouse Prompt Embeddings 생성 스크립트

실제 촬영 영상에 맞는 프롬프트 사용:
- "rendering image of 3D model" → "photograph of a mouse"
- 경사 카메라 각도 반영

Usage:
    python scripts/generate_mouse_prompt_embeds_realistic.py
    python scripts/generate_mouse_prompt_embeds_realistic.py --style facelift  # 원본 스타일
    python scripts/generate_mouse_prompt_embeds_realistic.py --style realistic  # 현실적 스타일
"""

import os
import json
import argparse
import torch
from transformers import CLIPTextModel, CLIPTokenizer


PROMPT_STYLES = {
    # 원본 FaceLift 스타일 (빠른 수렴)
    "facelift": {
        "views": ["front", "front_right", "right", "back", "left", "front_left"],
        "template": "a rendering image of 3D models, {view} view, color map.",
        "normal_template": "a rendering image of 3D models, {view} view, normal map.",
        "description": "Original FaceLift prompts for fast convergence",
    },

    # 현재 Mouse 스타일 (경사 카메라)
    "mouse_elevated": {
        "views": ["top-front", "top-front-right", "top-right", "top-back", "top-left", "top-front-left"],
        "template": "a rendering image of a 3D model, {view} view, from above at an angle, color map.",
        "normal_template": "a rendering image of a 3D model, {view} view, from above at an angle, normal map.",
        "description": "Mouse prompts with elevated camera angle",
    },

    # 현실적 스타일 (실제 촬영 영상)
    "realistic": {
        "views": ["front", "front-right", "right", "back", "left", "front-left"],
        "template": "a photograph of a mouse, {view} view, from above at an angle.",
        "normal_template": "a depth image of a mouse, {view} view, from above at an angle.",
        "description": "Realistic prompts for real captured video",
    },

    # 하이브리드 스타일
    "hybrid": {
        "views": ["front", "front-right", "right", "back", "left", "front-left"],
        "template": "a multi-view image of a mouse, {view} view, elevated camera.",
        "normal_template": "a normal map of a mouse, {view} view, elevated camera.",
        "description": "Hybrid prompts balancing realism and 3D terminology",
    },

    # 단순화 스타일 (도메인 중립)
    "simple": {
        "views": ["front", "front-right", "right", "back", "left", "front-left"],
        "template": "a mouse, {view} view, top-down angle.",
        "normal_template": "a mouse depth map, {view} view, top-down angle.",
        "description": "Simplified domain-neutral prompts",
    },
}


def generate_embeddings(style: str, output_dir: str):
    """Generate prompt embeddings for the specified style."""

    if style not in PROMPT_STYLES:
        raise ValueError(f"Unknown style: {style}. Available: {list(PROMPT_STYLES.keys())}")

    config = PROMPT_STYLES[style]
    os.makedirs(output_dir, exist_ok=True)

    # Load CLIP model (SD 2.1 text encoder for 1024-dim output)
    print(f"Loading CLIP model from stabilityai/stable-diffusion-2-1...")
    model_id = "stabilityai/stable-diffusion-2-1"

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_encoder = text_encoder.to(device).half()

    # Generate prompts
    views = config["views"]
    color_prompts = [config["template"].format(view=view) for view in views]
    normal_prompts = [config["normal_template"].format(view=view) for view in views]

    print(f"\n=== Style: {style} ===")
    print(f"Description: {config['description']}")
    print(f"\n=== Color Prompts ===")
    for i, p in enumerate(color_prompts):
        print(f"  [{i}] {p}")

    print(f"\n=== Normal Prompts ===")
    for i, p in enumerate(normal_prompts):
        print(f"  [{i}] {p}")

    # Encode function
    def encode_prompts(prompts):
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            prompt_embeds = text_encoder(text_inputs.input_ids)[0]
            prompt_embeds = prompt_embeds.detach().cpu()

        return prompt_embeds

    # Generate embeddings
    print("\nEncoding color prompts...")
    color_embeds = encode_prompts(color_prompts)
    print(f"  Shape: {color_embeds.shape}")
    torch.save(color_embeds.half(), os.path.join(output_dir, "clr_embeds.pt"))

    print("Encoding normal prompts...")
    normal_embeds = encode_prompts(normal_prompts)
    print(f"  Shape: {normal_embeds.shape}")
    torch.save(normal_embeds.half(), os.path.join(output_dir, "normal_embeds.pt"))

    # Save metadata
    metadata = {
        "style": style,
        "description": config["description"],
        "views": views,
        "color_prompts": color_prompts,
        "normal_prompts": normal_prompts,
        "shape": list(color_embeds.shape),
        "model": model_id,
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Done! Saved to {output_dir}/")
    print(f"   - clr_embeds.pt")
    print(f"   - normal_embeds.pt")
    print(f"   - metadata.json")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate prompt embeddings for mouse MVDiffusion")
    parser.add_argument(
        "--style",
        type=str,
        default="realistic",
        choices=list(PROMPT_STYLES.keys()),
        help="Prompt style to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: mvdiffusion/data/mouse_prompt_embeds_{style})"
    )
    parser.add_argument(
        "--list-styles",
        action="store_true",
        help="List available styles and exit"
    )

    args = parser.parse_args()

    if args.list_styles:
        print("Available prompt styles:\n")
        for name, config in PROMPT_STYLES.items():
            print(f"  {name}:")
            print(f"    {config['description']}")
            print(f"    Example: {config['template'].format(view=config['views'][0])}")
            print()
        return

    output_dir = args.output_dir or f"mvdiffusion/data/mouse_prompt_embeds_{args.style}"
    generate_embeddings(args.style, output_dir)


if __name__ == "__main__":
    main()
