#!/usr/bin/env python3
"""
Prompt Embeddings 생성 스크립트 (모듈화 버전)

지원 모드:
1. text_only: 텍스트 프롬프트만 사용 (기본)
2. with_angles: 텍스트 + 정량적 각도 정보
3. from_camera: 카메라 JSON에서 자동 추출

Usage:
    # 기본 (텍스트만)
    python scripts/generate_prompt_embeds.py \
        --mode text_only \
        --preset mouse \
        --output_dir mvdiffusion/data/mouse_prompt_embeds_6view

    # 각도 포함
    python scripts/generate_prompt_embeds.py \
        --mode with_angles \
        --elevation 20 \
        --output_dir mvdiffusion/data/mouse_embeds_with_angles

    # 카메라 JSON에서 추출
    python scripts/generate_prompt_embeds.py \
        --mode from_camera \
        --camera_json data_mouse/sample_000000/opencv_cameras.json \
        --output_dir mvdiffusion/data/mouse_embeds_from_camera
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer


# =============================================================================
# Preset Configurations
# =============================================================================

PRESETS = {
    "facelift": {
        "views": ["front", "front_right", "right", "back", "left", "front_left"],
        "elevation": 0.0,
        "description": "from the side",
        "azimuths": [0, 60, 90, 180, 270, 300],  # degrees
    },
    "mouse": {
        "views": ["top-front", "top-front-right", "top-right", "top-back", "top-left", "top-front-left"],
        "elevation": 20.0,
        "description": "from above at an angle",
        "azimuths": [0, 60, 90, 180, 270, 300],
    },
    "topdown": {
        "views": ["top-front", "top-right", "top-back", "top-left"],
        "elevation": 45.0,
        "description": "from above looking down",
        "azimuths": [0, 90, 180, 270],
    },
}


@dataclass
class ViewConfig:
    """Single view configuration."""
    index: int
    name: str
    azimuth: float  # degrees
    elevation: float  # degrees
    distance: float = 2.7


class PromptEmbedGenerator:
    """Modular prompt embedding generator."""

    def __init__(
        self,
        model_path: str = "checkpoints/mvdiffusion/pipeckpts",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16
    ):
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype

        print(f"Loading CLIP from {model_path}...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
        self.text_encoder = self.text_encoder.to(self.device, dtype=self.dtype)

    def encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts to embeddings."""
        print(f"\nEncoding {len(prompts)} prompts:")
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
            prompt_embeds = self.text_encoder(text_inputs.input_ids)[0]
            prompt_embeds = prompt_embeds.detach().cpu()

        print(f"  Shape: {prompt_embeds.shape}")
        return prompt_embeds

    # =========================================================================
    # Mode 1: Text Only (기존 방식)
    # =========================================================================
    def generate_text_only(
        self,
        views: List[str],
        description: str,
        output_dir: str
    ) -> Dict:
        """Generate embeddings from text prompts only."""
        os.makedirs(output_dir, exist_ok=True)

        color_prompts = [
            f"a rendering image of a 3D model, {view} view, {description}, color map."
            for view in views
        ]
        normal_prompts = [
            f"a rendering image of a 3D model, {view} view, {description}, normal map."
            for view in views
        ]

        print("\n=== Color Embeddings ===")
        color_embeds = self.encode_prompts(color_prompts)
        torch.save(color_embeds.half(), os.path.join(output_dir, "clr_embeds.pt"))

        print("\n=== Normal Embeddings ===")
        normal_embeds = self.encode_prompts(normal_prompts)
        torch.save(normal_embeds.half(), os.path.join(output_dir, "normal_embeds.pt"))

        metadata = {
            "mode": "text_only",
            "views": views,
            "description": description,
            "color_prompts": color_prompts,
            "normal_prompts": normal_prompts,
            "shape": list(color_embeds.shape),
        }

        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    # =========================================================================
    # Mode 2: With Quantitative Angles (정량적 각도 포함)
    # =========================================================================
    def generate_with_angles(
        self,
        views: List[str],
        azimuths: List[float],
        elevation: float,
        distance: float,
        output_dir: str
    ) -> Dict:
        """Generate embeddings with quantitative angle information."""
        os.makedirs(output_dir, exist_ok=True)

        color_prompts = []
        normal_prompts = []

        for i, (view, azimuth) in enumerate(zip(views, azimuths)):
            # 정량적 각도 포함 프롬프트
            angle_desc = f"elevation {elevation:.0f} degrees, azimuth {azimuth:.0f} degrees"
            color_prompts.append(
                f"a rendering image of a 3D model, {view} view, {angle_desc}, color map."
            )
            normal_prompts.append(
                f"a rendering image of a 3D model, {view} view, {angle_desc}, normal map."
            )

        print("\n=== Color Embeddings (with angles) ===")
        color_embeds = self.encode_prompts(color_prompts)
        torch.save(color_embeds.half(), os.path.join(output_dir, "clr_embeds.pt"))

        print("\n=== Normal Embeddings (with angles) ===")
        normal_embeds = self.encode_prompts(normal_prompts)
        torch.save(normal_embeds.half(), os.path.join(output_dir, "normal_embeds.pt"))

        metadata = {
            "mode": "with_angles",
            "views": views,
            "azimuths": azimuths,
            "elevation": elevation,
            "distance": distance,
            "color_prompts": color_prompts,
            "normal_prompts": normal_prompts,
            "shape": list(color_embeds.shape),
        }

        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    # =========================================================================
    # Mode 3: From Camera JSON (카메라에서 자동 추출)
    # =========================================================================
    def generate_from_camera(
        self,
        camera_json: str,
        output_dir: str
    ) -> Dict:
        """Generate embeddings from camera JSON file."""
        os.makedirs(output_dir, exist_ok=True)

        print(f"Loading camera from {camera_json}...")
        with open(camera_json, "r") as f:
            cameras = json.load(f)

        views = []
        azimuths = []
        elevations = []

        for cam_name, cam_data in cameras.items():
            # Extract rotation matrix and compute angles
            if "R" in cam_data:
                R = np.array(cam_data["R"])
                elevation, azimuth = self._rotation_to_angles(R)
            elif "extrinsic" in cam_data:
                ext = np.array(cam_data["extrinsic"])
                R = ext[:3, :3]
                elevation, azimuth = self._rotation_to_angles(R)
            else:
                print(f"  Warning: No rotation found for {cam_name}, using defaults")
                elevation, azimuth = 20.0, 0.0

            # Generate view name from angles
            view_name = self._angles_to_view_name(elevation, azimuth)
            views.append(view_name)
            azimuths.append(azimuth)
            elevations.append(elevation)

            print(f"  {cam_name}: elevation={elevation:.1f}°, azimuth={azimuth:.1f}° → {view_name}")

        avg_elevation = np.mean(elevations)

        # Generate prompts
        color_prompts = []
        normal_prompts = []

        for view, elev, azim in zip(views, elevations, azimuths):
            angle_desc = f"elevation {elev:.0f} degrees, azimuth {azim:.0f} degrees"
            color_prompts.append(
                f"a rendering image of a 3D model, {view} view, {angle_desc}, color map."
            )
            normal_prompts.append(
                f"a rendering image of a 3D model, {view} view, {angle_desc}, normal map."
            )

        print("\n=== Color Embeddings (from camera) ===")
        color_embeds = self.encode_prompts(color_prompts)
        torch.save(color_embeds.half(), os.path.join(output_dir, "clr_embeds.pt"))

        print("\n=== Normal Embeddings (from camera) ===")
        normal_embeds = self.encode_prompts(normal_prompts)
        torch.save(normal_embeds.half(), os.path.join(output_dir, "normal_embeds.pt"))

        metadata = {
            "mode": "from_camera",
            "camera_json": camera_json,
            "views": views,
            "azimuths": azimuths,
            "elevations": elevations,
            "avg_elevation": avg_elevation,
            "color_prompts": color_prompts,
            "normal_prompts": normal_prompts,
            "shape": list(color_embeds.shape),
        }

        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def _rotation_to_angles(self, R: np.ndarray) -> Tuple[float, float]:
        """Convert rotation matrix to elevation and azimuth angles."""
        # Assuming camera looks at origin from position
        # Extract elevation from R[2, 2] and azimuth from R[0, 2], R[1, 2]
        elevation = math.degrees(math.asin(-R[2, 1]))
        azimuth = math.degrees(math.atan2(R[0, 2], R[2, 2]))
        return elevation, azimuth

    def _angles_to_view_name(self, elevation: float, azimuth: float) -> str:
        """Convert angles to human-readable view name."""
        # Elevation prefix
        if elevation > 30:
            elev_prefix = "top-"
        elif elevation > 10:
            elev_prefix = "upper-"
        else:
            elev_prefix = ""

        # Azimuth direction
        azimuth = azimuth % 360
        if azimuth < 30 or azimuth >= 330:
            direction = "front"
        elif 30 <= azimuth < 60:
            direction = "front-right"
        elif 60 <= azimuth < 120:
            direction = "right"
        elif 120 <= azimuth < 150:
            direction = "back-right"
        elif 150 <= azimuth < 210:
            direction = "back"
        elif 210 <= azimuth < 240:
            direction = "back-left"
        elif 240 <= azimuth < 300:
            direction = "left"
        else:
            direction = "front-left"

        return f"{elev_prefix}{direction}"


def main():
    parser = argparse.ArgumentParser(description="Generate Prompt Embeddings (Modular)")

    parser.add_argument("--mode", type=str, default="text_only",
                        choices=["text_only", "with_angles", "from_camera"],
                        help="Generation mode")
    parser.add_argument("--preset", type=str, default=None,
                        choices=list(PRESETS.keys()),
                        help="Use preset configuration")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for embeddings")

    # Mode-specific arguments
    parser.add_argument("--elevation", type=float, default=20.0,
                        help="Elevation angle (for with_angles mode)")
    parser.add_argument("--camera_json", type=str, default=None,
                        help="Camera JSON path (for from_camera mode)")
    parser.add_argument("--model_path", type=str,
                        default="checkpoints/mvdiffusion/pipeckpts",
                        help="CLIP model path")

    args = parser.parse_args()

    print("=" * 60)
    print("Prompt Embeddings Generator (Modular)")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output_dir}")

    generator = PromptEmbedGenerator(model_path=args.model_path)

    if args.mode == "text_only":
        if args.preset:
            preset = PRESETS[args.preset]
            views = preset["views"]
            description = preset["description"]
        else:
            views = ["front", "front_right", "right", "back", "left", "front_left"]
            description = "from the side"

        generator.generate_text_only(views, description, args.output_dir)

    elif args.mode == "with_angles":
        if args.preset:
            preset = PRESETS[args.preset]
            views = preset["views"]
            azimuths = preset["azimuths"]
            elevation = preset["elevation"]
        else:
            views = ["front", "front_right", "right", "back", "left", "front_left"]
            azimuths = [0, 60, 90, 180, 270, 300]
            elevation = args.elevation

        generator.generate_with_angles(views, azimuths, elevation, 2.7, args.output_dir)

    elif args.mode == "from_camera":
        if not args.camera_json:
            raise ValueError("--camera_json required for from_camera mode")
        generator.generate_from_camera(args.camera_json, args.output_dir)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
