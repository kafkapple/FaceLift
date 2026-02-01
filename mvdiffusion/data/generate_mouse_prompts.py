"""
Generate mouse-specific text embeddings for MVDiffusion.
Two variants: rendering-style and real-image-style.
"""

import os
import json
import torch
from transformers import CLIPTokenizer, CLIPTextModel

# Mouse camera arrangement (not uniform, top-angled views)
MOUSE_VIEWS = [
    "top-front",
    "top-front-right", 
    "top-right",
    "top-back",
    "top-left",
    "top-front-left"
]

# Variant A: Rendering-style (기존 방식)
PROMPTS_RENDERING = {
    "description": "3D rendering style, consistent with Objaverse training",
    "color": [
        f"a rendering image of a 3D model, {view} view, from above at an angle, color map."
        for view in MOUSE_VIEWS
    ],
    "normal": [
        f"a rendering image of a 3D model, {view} view, from above at an angle, normal map."
        for view in MOUSE_VIEWS
    ]
}

# Variant B: Real-image-style (실제 이미지 명시)
PROMPTS_REAL = {
    "description": "Real laboratory mouse capture, emphasizes actual image domain",
    "color": [
        f"a laboratory mouse photographed from {view} angle, multi-camera capture, white background."
        for view in MOUSE_VIEWS
    ],
    "normal": [
        f"surface normals of a laboratory mouse from {view} angle, multi-camera setup."
        for view in MOUSE_VIEWS
    ]
}

# Variant C: Relative-view-style (상대적 뷰 강조)
PROMPTS_RELATIVE = {
    "description": "Emphasizes relative camera positions, not absolute directions",
    "color": [
        f"a mouse viewed from camera {i+1} of 6, multi-view synchronized capture, {view} relative position."
        for i, view in enumerate(MOUSE_VIEWS)
    ],
    "normal": [
        f"surface normals from camera {i+1} of 6, {view} relative position."
        for i, view in enumerate(MOUSE_VIEWS)
    ]
}


def generate_embeddings(prompts, output_dir, model_name="stabilityai/stable-diffusion-2-1-unclip"):
    """Generate CLIP embeddings for prompts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    text_encoder = text_encoder.cuda().half()
    
    for prompt_type in ["color", "normal"]:
        prompt_list = prompts[prompt_type]
        print(f"\nEncoding {prompt_type} prompts:")
        for p in prompt_list:
            print(f"  {p}")
        
        # Tokenize
        inputs = tokenizer(
            prompt_list,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")
        
        # Encode
        with torch.no_grad():
            embeds = text_encoder(inputs.input_ids)[0].detach().cpu()
        
        # Save
        filename = "clr_embeds.pt" if prompt_type == "color" else "normal_embeds.pt"
        torch.save(embeds, os.path.join(output_dir, filename))
        print(f"Saved: {output_dir}/{filename}, shape: {embeds.shape}")
    
    # Save metadata
    meta = {
        "views": MOUSE_VIEWS,
        "description": prompts["description"],
        "color_prompts": prompts["color"],
        "normal_prompts": prompts["normal"],
        "model": model_name
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {output_dir}/metadata.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["rendering", "real", "relative"], default="rendering")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    variants = {
        "rendering": (PROMPTS_RENDERING, "mouse_prompt_embeds_rendering"),
        "real": (PROMPTS_REAL, "mouse_prompt_embeds_real"),
        "relative": (PROMPTS_RELATIVE, "mouse_prompt_embeds_relative")
    }
    
    prompts, default_dir = variants[args.variant]
    output_dir = args.output_dir or f"mvdiffusion/data/{default_dir}"
    
    print(f"Generating {args.variant} variant embeddings...")
    generate_embeddings(prompts, output_dir)
    print("\nDone!")
