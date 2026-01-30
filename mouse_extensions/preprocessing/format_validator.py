"""
FaceLift Data Format Validator

Ensures preprocessed data matches MouseDataset expectations.
Reference: gslrm/data/mouse_dataset.py
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

from ..paths import PREPROCESSED_DIR


class FormatValidator:
    """Validates preprocessed data against FaceLift/MouseDataset requirements."""
    
    # Required fields based on mouse_dataset.py
    REQUIRED_CAMERA_KEYS = ["frames"]
    REQUIRED_FRAME_KEYS = ["w", "h", "fx", "fy", "cx", "cy", "w2c", "file_path", "view_id"]
    OPTIONAL_FRAME_KEYS = ["_transform"]  # Optional metadata
    
    def __init__(self, reference_path: Optional[str] = None):
        """
        Args:
            reference_path: Path to a known-good sample (e.g., D1) for comparison
        """
        self.reference_path = reference_path
        self.errors = []
        self.warnings = []
    
    def validate_sample(self, sample_dir: Path) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a single preprocessed sample directory.
        
        Returns:
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        sample_dir = Path(sample_dir)
        
        # 1. Check directory structure
        self._check_directory_structure(sample_dir)
        
        # 2. Check camera JSON
        self._check_camera_json(sample_dir)
        
        # 3. Check images
        self._check_images(sample_dir)
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _check_directory_structure(self, sample_dir: Path):
        """Check required files and folders exist."""
        # Required: opencv_cameras.json
        cam_file = sample_dir / "opencv_cameras.json"
        if not cam_file.exists():
            self.errors.append(f"Missing opencv_cameras.json in {sample_dir}")
        
        # Required: images/ directory
        images_dir = sample_dir / "images"
        if not images_dir.exists():
            self.errors.append(f"Missing images/ directory in {sample_dir}")
    
    def _check_camera_json(self, sample_dir: Path):
        """Validate camera JSON structure and content."""
        cam_file = sample_dir / "opencv_cameras.json"
        if not cam_file.exists():
            return
        
        try:
            with open(cam_file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in {cam_file}: {e}")
            return
        
        # Check top-level structure
        for key in self.REQUIRED_CAMERA_KEYS:
            if key not in data:
                self.errors.append(f"Missing required key '{key}' in {cam_file}")
                return
        
        # Check frames list
        frames = data.get("frames", [])
        if not isinstance(frames, list):
            self.errors.append(f"'frames' must be a list in {cam_file}")
            return
        
        if len(frames) == 0:
            self.errors.append(f"Empty frames list in {cam_file}")
            return
        
        # Check each frame
        for i, frame in enumerate(frames):
            for key in self.REQUIRED_FRAME_KEYS:
                if key not in frame:
                    self.errors.append(f"Frame {i} missing required key '{key}' in {cam_file}")
            
            # Validate w2c matrix shape
            if "w2c" in frame:
                w2c = frame["w2c"]
                if not (isinstance(w2c, list) and len(w2c) == 4 and all(len(row) == 4 for row in w2c)):
                    self.errors.append(f"Frame {i} w2c must be 4x4 matrix in {cam_file}")
            
            # Validate file_path exists
            if "file_path" in frame:
                img_path = sample_dir / frame["file_path"]
                if not img_path.exists():
                    self.errors.append(f"Frame {i} file_path '{frame['file_path']}' not found")
    
    def _check_images(self, sample_dir: Path):
        """Validate image files."""
        images_dir = sample_dir / "images"
        if not images_dir.exists():
            return
        
        # Check for expected image files
        expected_images = [f"cam_{i:03d}.png" for i in range(6)]
        for img_name in expected_images:
            img_path = images_dir / img_name
            if not img_path.exists():
                self.errors.append(f"Missing image {img_name} in {images_dir}")
                continue
            
            # Validate image format
            try:
                img = Image.open(img_path)
                if img.mode != "RGBA":
                    self.warnings.append(f"{img_name} is {img.mode}, expected RGBA")
                if img.size != (512, 512):
                    self.warnings.append(f"{img_name} is {img.size}, expected (512, 512)")
            except Exception as e:
                self.errors.append(f"Cannot open {img_name}: {e}")
    
    def validate_dataset(self, dataset_dir: Path, max_samples: int = 10) -> Dict:
        """
        Validate multiple samples from a dataset.
        
        Returns:
            Summary dict with pass/fail counts and error details
        """
        dataset_dir = Path(dataset_dir)
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
            "warnings": []
        }
        
        for split in ["train", "val"]:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue
            
            sample_dirs = sorted(split_dir.iterdir())[:max_samples]
            for sample_dir in sample_dirs:
                if not sample_dir.is_dir():
                    continue
                
                results["total"] += 1
                is_valid, errors, warnings = self.validate_sample(sample_dir)
                
                if is_valid:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].extend(errors)
                
                results["warnings"].extend(warnings)
        
        return results
    
    def compare_with_reference(self, sample_dir: Path) -> Dict:
        """Compare sample with reference (D1) format."""
        if not self.reference_path:
            return {"error": "No reference path set"}
        
        ref_dir = Path(self.reference_path)
        sample_dir = Path(sample_dir)
        
        # Load both camera JSONs
        ref_cam = json.load(open(ref_dir / "opencv_cameras.json"))
        sample_cam = json.load(open(sample_dir / "opencv_cameras.json"))
        
        comparison = {
            "ref_keys": set(ref_cam.get("frames", [{}])[0].keys()),
            "sample_keys": set(sample_cam.get("frames", [{}])[0].keys()),
        }
        comparison["missing_keys"] = comparison["ref_keys"] - comparison["sample_keys"]
        comparison["extra_keys"] = comparison["sample_keys"] - comparison["ref_keys"]
        
        return comparison


def validate_d3_dataset():
    """Quick validation of D3 dataset."""
    validator = FormatValidator(
        reference_path=str(PREPROCESSED_DIR / "D1_pp_centered" / "train" / "sample_000000")
    )
    
    print("=" * 60)
    print("D3 Dataset Validation Report")
    print("=" * 60)
    
    # Validate D3
    d3_path = PREPROCESSED_DIR / "D3"
    results = validator.validate_dataset(d3_path, max_samples=20)
    
    print(f"\nSamples checked: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    
    if results["errors"]:
        print(f"\n❌ Errors ({len(results['errors'])}):")
        for err in results["errors"][:10]:
            print(f"  - {err}")
    
    if results["warnings"]:
        print(f"\n⚠️ Warnings ({len(results['warnings'])}):")
        for warn in results["warnings"][:5]:
            print(f"  - {warn}")
    
    # Compare with D1 reference
    print("\n" + "=" * 60)
    print("Comparison with D1 Reference")
    print("=" * 60)
    
    d3_sample = d3_path / "train" / "000000"
    comparison = validator.compare_with_reference(d3_sample)
    
    print(f"\nD1 frame keys: {sorted(comparison['ref_keys'])}")
    print(f"D3 frame keys: {sorted(comparison['sample_keys'])}")
    
    if comparison["missing_keys"]:
        print(f"\n❌ Missing in D3: {comparison['missing_keys']}")
    else:
        print("\n✅ All required keys present")
    
    if comparison["extra_keys"]:
        print(f"ℹ️ Extra in D3 (OK): {comparison['extra_keys']}")
    
    return results["failed"] == 0


if __name__ == "__main__":
    success = validate_d3_dataset()
    exit(0 if success else 1)
