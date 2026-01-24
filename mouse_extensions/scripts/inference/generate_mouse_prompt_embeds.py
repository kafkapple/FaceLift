#!/usr/bin/env python3
"""Generate CLIP text embeddings for mouse multi-view prompts."""

import os
import json
import torch
from transformers import CLIPTokenizer, CLIPTextModel


def generate_embeddings(prompts, tokenizer, text_encoder, device):
    text_inputs = tokenizer(
        prompts, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        embeds = text_encoder(text_inputs.input_ids)[0]
    return embeds.cpu()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "openai/clip-vit-large-patch14"
    print(f"Loading {model_name}...")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)
    
    configs = {
        "mouse_prompt_embeds_6view": {
            "desc": "3D rendering style (original)",
            "prompts": [
                "a rendering image of a 3D model, top-front view, from above at an angle, color map.",
                "a rendering image of a 3D model, top-front-right view, from above at an angle, color map.",
                "a rendering image of a 3D model, top-right view, from above at an angle, color map.",
                "a rendering image of a 3D model, top-back view, from above at an angle, color map.",
                "a rendering image of a 3D model, top-left view, from above at an angle, color map.",
                "a rendering image of a 3D model, top-front-left view, from above at an angle, color map."
            ]
        },
        "mouse_prompt_embeds_6view_real": {
            "desc": "Real photograph style",
            "prompts": [
                "a photograph of a real mouse, top-front view, from above at an angle.",
                "a photograph of a real mouse, top-front-right view, from above at an angle.",
                "a photograph of a real mouse, top-right view, from above at an angle.",
                "a photograph of a real mouse, top-back view, from above at an angle.",
                "a photograph of a real mouse, top-left view, from above at an angle.",
                "a photograph of a real mouse, top-front-left view, from above at an angle."
            ]
        },
        "mouse_prompt_embeds_6view_minimal": {
            "desc": "Minimal direction-only",
            "prompts": ["front view", "front-right view", "right view", "back view", "left view", "front-left view"]
        }
    }
    
    base_dir = "mvdiffusion/data"
    
    for dir_name, cfg in configs.items():
        output_dir = os.path.join(base_dir, dir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=== {dir_name}: {cfg[desc]} ===")
        for i, p in enumerate(cfg["prompts"]):
            print(f"  View {i}: {p[:50]}...")
        
        embeds = generate_embeddings(cfg["prompts"], tokenizer, text_encoder, device)
        
        torch.save(embeds, os.path.join(output_dir, "clr_embeds.pt"))
        print(f"Saved: clr_embeds.pt (shape: {embeds.shape})")
        
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump({"description": cfg["desc"], "prompts": cfg["prompts"], "shape": list(embeds.shape)}, f, indent=2)
    
    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
