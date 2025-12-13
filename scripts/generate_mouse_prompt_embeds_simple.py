#!/usr/bin/env python3
"""
간단한 Mouse Prompt Embeddings 생성 스크립트

HuggingFace에서 직접 CLIP 모델을 로드하여 사용.
로컬 체크포인트가 없어도 동작.

Usage:
    python scripts/generate_mouse_prompt_embeds_simple.py
"""

import os
import json
import torch
from transformers import CLIPTextModel, CLIPTokenizer


def main():
    output_dir = "mvdiffusion/data/mouse_prompt_embeds_6view"
    os.makedirs(output_dir, exist_ok=True)

    # CLIP 모델 로드 (Stable Diffusion 2.1의 text encoder 사용)
    # 이 모델은 1024 차원 출력을 생성하여 FaceLift와 호환됨
    print("Loading CLIP model from stabilityai/stable-diffusion-2-1...")
    model_id = "stabilityai/stable-diffusion-2-1"

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    text_encoder = text_encoder.to("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = text_encoder.half()

    # Mouse preset views (경사 카메라, elevation ~20°)
    views = ["top-front", "top-front-right", "top-right", "top-back", "top-left", "top-front-left"]
    description = "from above at an angle"

    # Color prompts
    color_prompts = [
        f"a rendering image of a 3D model, {view} view, {description}, color map."
        for view in views
    ]

    # Normal prompts
    normal_prompts = [
        f"a rendering image of a 3D model, {view} view, {description}, normal map."
        for view in views
    ]

    print("\n=== Color Prompts ===")
    for i, p in enumerate(color_prompts):
        print(f"  [{i}] {p}")

    print("\n=== Normal Prompts ===")
    for i, p in enumerate(normal_prompts):
        print(f"  [{i}] {p}")

    # Encode prompts
    def encode_prompts(prompts):
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(text_encoder.device)

        with torch.no_grad():
            prompt_embeds = text_encoder(text_inputs.input_ids)[0]
            prompt_embeds = prompt_embeds.detach().cpu()

        return prompt_embeds

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
        "mode": "text_only",
        "preset": "mouse",
        "views": views,
        "description": description,
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


if __name__ == "__main__":
    main()
