"""Build DiFix-compatible dataset from poc_mesh_gs_pairs/ outputs.

Converts the raw pipeline output into a structured dataset with:
- Consistent 384x384 images (gt_rgb resized from 512)
- Per-frame cameras.json (extracted from camera_config)
- manifest.json with train/val/test split and all sample metadata

Usage:
    python -m mouse_extensions.scripts.eval.build_difix_dataset \
        --src outputs/poc_mesh_gs_pairs \
        --dst outputs/difix_pairs \
        [--resize-gt]  # resize gt_rgb 512→384
"""
import os
import json
import argparse
import shutil
from pathlib import Path

from PIL import Image

# M5t2 temporal split (canonical)
SPLITS = {
    "train": (0, 2879),
    "val": (2880, 3239),
    "test": (3240, 3599),
}

GT_VIEWS = [f"cam_{v:03d}" for v in range(6)]
NOVEL_VIEWS = ["bottom", "top", "front_low", "side_low"]
RENDER_RES = 384

# Subdirectories to copy (source_name → dest_name)
SUBDIRS = {
    "gt_rgb": "gt_rgb",
    "gslrm_gt": "gslrm_gt",
    "gslrm_novel": "gslrm_novel",
    "mammal_gt": "mammal_gt",
    "mammal_novel": "mammal_novel",
}


def get_split(frame_idx: int) -> str:
    for split_name, (lo, hi) in SPLITS.items():
        if lo <= frame_idx <= hi:
            return split_name
    return "unknown"


def build_frame_cameras(cam_config: dict) -> dict:
    """Extract per-view camera params from camera_config JSON."""
    cameras = {}
    for view_name, cam_data in cam_config.get("gt_cameras", {}).items():
        cameras[view_name] = {
            "c2w": cam_data["c2w"],
            "fxfycxcy": cam_data["fxfycxcy"],
            "resolution": cam_data.get("resolution", RENDER_RES),
        }
    for view_name, cam_data in cam_config.get("novel_cameras", {}).items():
        cameras[view_name] = {
            "c2w": cam_data["c2w"],
            "fxfycxcy": cam_data["fxfycxcy"],
            "resolution": cam_data.get("resolution", RENDER_RES),
        }
    return cameras


def copy_images(src_dir: Path, dst_dir: Path, resize_to: int | None = None):
    """Copy images, optionally resizing."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_file in sorted(src_dir.glob("*.png")):
        dst_path = dst_dir / img_file.name
        if resize_to is not None:
            img = Image.open(img_file)
            if img.size[0] != resize_to:
                img = img.resize((resize_to, resize_to), Image.LANCZOS)
            img.save(dst_path)
        else:
            shutil.copy2(img_file, dst_path)


def build_manifest_entries(frame_idx: int, split: str, dst_frame: str) -> list[dict]:
    """Generate manifest entries for all views in a frame."""
    entries = []
    for view_name in GT_VIEWS:
        entries.append({
            "sample_id": f"{frame_idx:06d}_{view_name}",
            "frame_idx": frame_idx,
            "view_name": view_name,
            "view_type": "gt",
            "split": split,
            "paths": {
                "gt_rgb": f"frames/{dst_frame}/gt_rgb/{view_name}.png",
                "gslrm": f"frames/{dst_frame}/gslrm_gt/{view_name}.png",
                "mammal": f"frames/{dst_frame}/mammal_gt/{view_name}.png",
            },
            "cam_key": view_name,
        })
    for view_name in NOVEL_VIEWS:
        entries.append({
            "sample_id": f"{frame_idx:06d}_{view_name}",
            "frame_idx": frame_idx,
            "view_name": view_name,
            "view_type": "novel",
            "split": split,
            "paths": {
                "gslrm": f"frames/{dst_frame}/gslrm_novel/{view_name}.png",
                "mammal": f"frames/{dst_frame}/mammal_novel/{view_name}.png",
            },
            "cam_key": view_name,
        })
    return entries


def main():
    parser = argparse.ArgumentParser(description="Build DiFix dataset from poc_mesh_gs_pairs")
    parser.add_argument("--src", required=True, help="Source: outputs/poc_mesh_gs_pairs")
    parser.add_argument("--dst", required=True, help="Destination: outputs/difix_pairs")
    parser.add_argument("--resize-gt", action="store_true",
                        help="Resize gt_rgb from 512 to 384")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without copying")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    # Find all frame directories
    frame_dirs = sorted(src.glob("frame_*"))
    print(f"Found {len(frame_dirs)} frame directories in {src}")

    manifest_entries = []
    split_counts = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    skipped = []

    for frame_dir in frame_dirs:
        frame_name = frame_dir.name  # frame_00000
        frame_idx = int(frame_name.replace("frame_", ""))
        split = get_split(frame_idx)
        split_counts[split] += 1

        # Verify completeness
        required = ["gt_rgb", "gslrm_gt", "mammal_gt"]
        missing = [d for d in required if not (frame_dir / d).is_dir()]
        if missing:
            skipped.append((frame_idx, missing))
            continue

        dst_frame = f"{frame_idx:06d}"
        dst_frame_dir = dst / "frames" / dst_frame

        if args.dry_run:
            print(f"  {frame_name} → frames/{dst_frame}/ [{split}]")
            manifest_entries.extend(
                build_manifest_entries(frame_idx, split, dst_frame))
            continue

        # Copy image subdirectories
        for src_subdir, dst_subdir in SUBDIRS.items():
            src_sub = frame_dir / src_subdir
            if src_sub.is_dir():
                resize = RENDER_RES if (src_subdir == "gt_rgb" and args.resize_gt) else None
                copy_images(src_sub, dst_frame_dir / dst_subdir, resize_to=resize)

        # Build and save per-frame cameras.json
        cam_config_path = frame_dir / f"camera_config_{frame_name}.json"
        if cam_config_path.exists():
            with open(cam_config_path) as f:
                cam_config = json.load(f)
            cameras = build_frame_cameras(cam_config)
            cam_dst = dst_frame_dir / "cameras.json"
            cam_dst.parent.mkdir(parents=True, exist_ok=True)
            with open(cam_dst, "w") as f:
                json.dump(cameras, f, indent=2)

        manifest_entries.extend(
            build_manifest_entries(frame_idx, split, dst_frame))

    # Build manifest
    manifest = {
        "version": "1.0",
        "description": "MAMMAL mesh + GS-LRM render pairs for DiFix 3D+ training",
        "source": "FaceLift M5t2 dataset",
        "render_resolution": RENDER_RES,
        "gt_resolution": 512,
        "gt_resized": args.resize_gt,
        "splits": {
            name: {"start": lo, "end": hi, "count": split_counts[name]}
            for name, (lo, hi) in SPLITS.items()
        },
        "gt_views": GT_VIEWS,
        "novel_views": NOVEL_VIEWS,
        "total_frames": len(frame_dirs) - len(skipped),
        "total_samples": len(manifest_entries),
        "samples": manifest_entries,
    }

    if not args.dry_run:
        manifest_path = dst / "manifest.json"
        dst.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest saved: {manifest_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Dataset build {'(DRY RUN) ' if args.dry_run else ''}summary:")
    print(f"  Frames: {len(frame_dirs) - len(skipped)} / {len(frame_dirs)}")
    print(f"  Samples: {len(manifest_entries)}")
    for name in ["train", "val", "test"]:
        count = split_counts[name]
        samples = sum(1 for e in manifest_entries if e["split"] == name)
        print(f"  {name:>5s}: {count} frames, {samples} samples")
    if skipped:
        print(f"\n  Skipped {len(skipped)} incomplete frames:")
        for idx, missing in skipped[:5]:
            print(f"    frame_{idx:05d}: missing {missing}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped)-5} more")


if __name__ == "__main__":
    main()
