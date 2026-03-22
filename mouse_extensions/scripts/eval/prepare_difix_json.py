"""Prepare DiFix training JSON from our difix_pairs_512 dataset.

Converts our symlink-based pair structure to DiFix's expected JSON format:
{
    "train": {"id": {"image": "...", "target_image": "...", "prompt": "..."}},
    "test": {"id": {"image": "...", "target_image": "...", "prompt": "..."}}
}

Usage:
    python -m mouse_extensions.scripts.eval.prepare_difix_json \
        --pairs_dir outputs/datasets/difix_pairs_512 \
        --output_json outputs/datasets/difix_pairs_512/difix_training.json \
        --num_train 5000 --num_test 500
"""

import argparse
import json
import os
from pathlib import Path


def build_difix_json(
    pairs_dir: str,
    output_json: str,
    num_train: int = 5000,
    num_test: int = 500,
    prompt: str = "a high quality 3D rendering of a mouse, clean, sharp, artifact-free",
):
    """Build DiFix training JSON from our pair structure."""
    pairs_dir = Path(pairs_dir)
    manifest_path = pairs_dir / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Loaded manifest: {len(manifest.get('pairs', []))} pairs")
    else:
        print("No manifest.json, scanning directories...")
        manifest = None

    # Scan for type3 pairs (view ablation: N-view → 6-view)
    type3_dir = pairs_dir / "type3_view_ablation"
    pairs = []

    if type3_dir.exists():
        for pair_dir in sorted(type3_dir.iterdir()):
            if not pair_dir.is_dir():
                continue
            input_path = pair_dir / "input.png"
            target_path = pair_dir / "target.png"
            if input_path.exists() and target_path.exists():
                # Resolve symlinks to absolute paths
                pairs.append({
                    "input": str(input_path.resolve()),
                    "target": str(target_path.resolve()),
                    "id": pair_dir.name,
                })

    print(f"Found {len(pairs)} Type 3 pairs")

    if not pairs:
        print("ERROR: No pairs found!")
        return

    # Split into train/test (by frame index to avoid data leakage)
    # M5t2 splits: train=0-2879, val=2880-3239, test=3240-3599
    train_pairs = []
    test_pairs = []
    for p in pairs:
        # Extract frame from id (e.g., "00000_cam_000_1v" → frame 0)
        frame = int(p["id"].split("_")[0])
        if frame < 2880:
            train_pairs.append(p)
        else:
            test_pairs.append(p)

    # Limit to requested sizes
    if num_train > 0 and len(train_pairs) > num_train:
        # Sample uniformly
        step = len(train_pairs) // num_train
        train_pairs = train_pairs[::step][:num_train]
    if num_test > 0 and len(test_pairs) > num_test:
        step = len(test_pairs) // num_test
        test_pairs = test_pairs[::step][:num_test]

    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    # Build DiFix JSON format
    data = {"train": {}, "test": {}}

    for p in train_pairs:
        data["train"][p["id"]] = {
            "image": p["input"],
            "target_image": p["target"],
            "prompt": prompt,
        }

    for p in test_pairs:
        data["test"][p["id"]] = {
            "image": p["input"],
            "target_image": p["target"],
            "prompt": prompt,
        }

    # Save
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved: {output_json}")
    print(f"  Train: {len(data['train'])} pairs")
    print(f"  Test: {len(data['test'])} pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--num_train", type=int, default=5000,
                        help="Max training pairs (0=all)")
    parser.add_argument("--num_test", type=int, default=500,
                        help="Max test pairs (0=all)")
    parser.add_argument("--prompt", type=str,
                        default="a high quality 3D rendering of a mouse, clean, sharp, artifact-free")
    args = parser.parse_args()
    build_difix_json(args.pairs_dir, args.output_json, args.num_train, args.num_test, args.prompt)
