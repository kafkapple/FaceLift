"""Build DiFix-compatible training dataset from tier-based novel_view data.

Creates (degraded, clean) image pairs for DiFix 3D+ fine-tuning, organized by
pair type with a unified manifest.

Pair Types:
  Type 1 (gt_view):       N-view GS-LRM render @ GT camera  <->  Original GT RGB
  Type 2 (novel_view):    6-view GS-LRM render @ novel cam   <->  MAMMAL mesh pseudo-GT
  Type 3 (view_ablation): N-view GS-LRM render @ GT camera  <->  6-view GS-LRM render

Usage:
    # Build all pair types with symlinks
    python -m mouse_extensions.scripts.eval.build_difix_dataset \
        --src outputs/datasets/novel_view \
        --dst outputs/datasets/difix_pairs \
        --types 1,2,3 --n_views 1,2,3,4,5 --use-symlinks

    # Build only Type 3 (self-supervised) for Stage 1 training
    python -m mouse_extensions.scripts.eval.build_difix_dataset \
        --src outputs/datasets/novel_view \
        --dst outputs/datasets/difix_pairs \
        --types 3 --n_views 1,2,3,4,5 --use-symlinks

    # Dry run to preview
    python -m mouse_extensions.scripts.eval.build_difix_dataset \
        --src outputs/datasets/novel_view --dst outputs/datasets/difix_pairs \
        --types 1,2,3 --dry-run
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

# M5t2 canonical temporal split
M5T2_SPLITS = {
    "train": (0, 2880),
    "val": (2880, 3240),
    "test": (3240, 3600),
}

GT_VIEWS = [f"cam_{v:03d}" for v in range(6)]
NOVEL_VIEWS = ["bottom", "top", "front_low", "side_low"]
SPECIES_DATASET = "mouse_m5t2"

# View ablation PSNR reference (from H4 experiment, fair eval)
VIEW_ABLATION_PSNR = {
    1: 10.47,
    2: 15.95,
    3: 18.56,
    4: 20.66,
    5: 22.16,
    6: 23.84,
}


def get_split(frame_idx: int) -> str:
    """Return train/val/test for M5t2 frame index."""
    for split_name, (start, end) in M5T2_SPLITS.items():
        if start <= frame_idx < end:
            return split_name
    return "unknown"


def get_degradation_level(n_views: int) -> str:
    """Qualitative degradation level based on view count."""
    if n_views <= 1:
        return "severe"
    elif n_views <= 2:
        return "heavy"
    elif n_views <= 3:
        return "moderate"
    elif n_views <= 4:
        return "mild"
    else:
        return "subtle"


# ============================================================
# Path builders for source data (novel_view structure)
# ============================================================
def src_ablation_path(src: Path, n_views: int, cam: str, frame_idx: int) -> Path:
    """Source: ablation_{N}view/cam_NNN/NNNNN.png"""
    return src / SPECIES_DATASET / f"ablation_{n_views}view" / cam / f"{frame_idx:05d}.png"


def src_gt_rgb_path(src: Path, cam: str, frame_idx: int) -> Path:
    """Source: gt_rgb/cam_NNN/NNNNN.png"""
    return src / SPECIES_DATASET / "gt_rgb" / cam / f"{frame_idx:05d}.png"


def src_gt_views_path(src: Path, cam: str, frame_idx: int) -> Path:
    """Source: gt_views/cam_NNN/NNNNN.png (6-view GS-LRM render at GT camera)"""
    return src / SPECIES_DATASET / "gt_views" / cam / f"{frame_idx:05d}.png"


def src_tier0_path(src: Path, view: str, frame_idx: int) -> Path:
    """Source: tier0_raw/{view}/NNNNN.png"""
    return src / SPECIES_DATASET / "tier0_raw" / view / f"{frame_idx:05d}.png"


def src_pseudo_gt_path(src: Path, view: str, frame_idx: int) -> Path:
    """Source: pseudo_gt/{view}/NNNNN.png"""
    return src / SPECIES_DATASET / "pseudo_gt" / view / f"{frame_idx:05d}.png"


# ============================================================
# Pair builders
# ============================================================
def build_type1_pairs(
    src: Path,
    dst: Path,
    frame_indices: list[int],
    n_views_list: list[int],
    use_symlinks: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Type 1: N-view GS-LRM render @ GT camera <-> GT RGB.

    Real GT supervision — strongest training signal.
    """
    type_dir = dst / "type1_gt_view"
    entries = []
    skipped = 0

    for frame_idx in frame_indices:
        split = get_split(frame_idx)
        for cam in GT_VIEWS:
            gt_path = src_gt_rgb_path(src, cam, frame_idx)
            if not gt_path.exists():
                skipped += 1
                continue

            for nv in n_views_list:
                input_path = src_ablation_path(src, nv, cam, frame_idx)
                if not input_path.exists():
                    skipped += 1
                    continue

                pair_name = f"{frame_idx:05d}_{cam}_{nv}v"
                pair_dir = type_dir / pair_name

                if not dry_run:
                    pair_dir.mkdir(parents=True, exist_ok=True)
                    _link_or_copy(input_path, pair_dir / "input.png", use_symlinks)
                    _link_or_copy(gt_path, pair_dir / "target.png", use_symlinks)

                entries.append({
                    "pair_id": f"type1_{pair_name}",
                    "pair_type": "type1_gt_view",
                    "frame_idx": frame_idx,
                    "split": split,
                    "view_name": cam,
                    "view_type": "gt",
                    "n_views": nv,
                    "degradation_level": get_degradation_level(nv),
                    "psnr_expected": VIEW_ABLATION_PSNR.get(nv),
                    "input_path": f"type1_gt_view/{pair_name}/input.png",
                    "target_path": f"type1_gt_view/{pair_name}/target.png",
                })

    if skipped:
        print(f"  Type 1: skipped {skipped} missing pairs")
    return entries


def build_type2_pairs(
    src: Path,
    dst: Path,
    frame_indices: list[int],
    use_symlinks: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Type 2: 6-view GS-LRM novel render <-> MAMMAL pseudo-GT.

    Novel view pairs — bridges to unseen viewpoints.
    """
    type_dir = dst / "type2_novel_view"
    entries = []
    skipped = 0

    for frame_idx in frame_indices:
        split = get_split(frame_idx)
        for view in NOVEL_VIEWS:
            input_path = src_tier0_path(src, view, frame_idx)
            target_path = src_pseudo_gt_path(src, view, frame_idx)

            if not input_path.exists() or not target_path.exists():
                skipped += 1
                continue

            pair_name = f"{frame_idx:05d}_{view}"
            pair_dir = type_dir / pair_name

            if not dry_run:
                pair_dir.mkdir(parents=True, exist_ok=True)
                _link_or_copy(input_path, pair_dir / "input.png", use_symlinks)
                _link_or_copy(target_path, pair_dir / "target.png", use_symlinks)

            entries.append({
                "pair_id": f"type2_{pair_name}",
                "pair_type": "type2_novel_view",
                "frame_idx": frame_idx,
                "split": split,
                "view_name": view,
                "view_type": "novel",
                "n_views": 6,
                "degradation_level": "novel_view",
                "input_path": f"type2_novel_view/{pair_name}/input.png",
                "target_path": f"type2_novel_view/{pair_name}/target.png",
            })

    if skipped:
        print(f"  Type 2: skipped {skipped} missing pairs")
    return entries


def build_type3_pairs(
    src: Path,
    dst: Path,
    frame_indices: list[int],
    n_views_list: list[int],
    use_symlinks: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Type 3: N-view GS-LRM render <-> 6-view GS-LRM render (self-supervised).

    Zero domain gap — same renderer, different input view counts.
    """
    type_dir = dst / "type3_view_ablation"
    entries = []
    skipped = 0

    for frame_idx in frame_indices:
        split = get_split(frame_idx)
        for cam in GT_VIEWS:
            target_path = src_gt_views_path(src, cam, frame_idx)
            if not target_path.exists():
                skipped += 1
                continue

            for nv in n_views_list:
                input_path = src_ablation_path(src, nv, cam, frame_idx)
                if not input_path.exists():
                    skipped += 1
                    continue

                pair_name = f"{frame_idx:05d}_{cam}_{nv}v"
                pair_dir = type_dir / pair_name

                if not dry_run:
                    pair_dir.mkdir(parents=True, exist_ok=True)
                    _link_or_copy(input_path, pair_dir / "input.png", use_symlinks)
                    _link_or_copy(target_path, pair_dir / "target.png", use_symlinks)

                entries.append({
                    "pair_id": f"type3_{pair_name}",
                    "pair_type": "type3_view_ablation",
                    "frame_idx": frame_idx,
                    "split": split,
                    "view_name": cam,
                    "view_type": "gt",
                    "n_views": nv,
                    "degradation_level": get_degradation_level(nv),
                    "psnr_expected": VIEW_ABLATION_PSNR.get(nv),
                    "input_path": f"type3_view_ablation/{pair_name}/input.png",
                    "target_path": f"type3_view_ablation/{pair_name}/target.png",
                })

    if skipped:
        print(f"  Type 3: skipped {skipped} missing pairs")
    return entries


# ============================================================
# Helpers
# ============================================================
def _link_or_copy(src_path: Path, dst_path: Path, use_symlinks: bool):
    """Create symlink or copy file."""
    if use_symlinks:
        # Use absolute path for symlink target
        abs_src = src_path.resolve()
        if dst_path.exists() or dst_path.is_symlink():
            dst_path.unlink()
        os.symlink(abs_src, dst_path)
    else:
        shutil.copy2(src_path, dst_path)


def discover_frame_indices(src: Path) -> list[int]:
    """Discover available frames from metadata directory."""
    meta_dir = src / SPECIES_DATASET / "metadata"
    if not meta_dir.exists():
        print(f"WARNING: No metadata found at {meta_dir}")
        return []

    indices = []
    for f in sorted(meta_dir.iterdir()):
        if f.suffix == ".json":
            try:
                indices.append(int(f.stem))
            except ValueError:
                continue
    return indices


def build_manifest(
    entries: list[dict],
    types_built: list[int],
    n_views_list: list[int],
    use_symlinks: bool,
) -> dict:
    """Build unified manifest with pair metadata."""
    # Count by type and split
    type_counts = {}
    split_counts = {"train": 0, "val": 0, "test": 0}
    for entry in entries:
        pt = entry["pair_type"]
        type_counts[pt] = type_counts.get(pt, 0) + 1
        split = entry["split"]
        if split in split_counts:
            split_counts[split] += 1

    return {
        "dataset_name": "DiFix 3D+ Training Pairs for GS-LRM Artifact Removal",
        "version": "2.0",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "FaceLift Novel View Dataset (M5t2)",
        "pair_types_built": [f"type{t}" for t in types_built],
        "n_views_used": n_views_list,
        "use_symlinks": use_symlinks,
        "total_pairs": len(entries),
        "pairs_by_type": type_counts,
        "pairs_by_split": split_counts,
        "splits": {
            name: {"start": s, "end": e}
            for name, (s, e) in M5T2_SPLITS.items()
        },
        "view_ablation_psnr_reference": VIEW_ABLATION_PSNR,
        "training_strategy": {
            "stage_0": "Zero-shot DiFix 3D+ (no fine-tuning)",
            "stage_1": "Type 3 (self-supervised, view ablation pairs)",
            "stage_1_5": "Type 3 (80%) + Type 2 (20%, novel view bridge)",
            "stage_2": "Type 1 (50%) + Type 3 (30%) + Type 2 (20%)",
        },
        "samples": entries,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Build DiFix training pairs from tier-based novel_view dataset"
    )
    parser.add_argument(
        "--src", required=True,
        help="Source: outputs/datasets/novel_view",
    )
    parser.add_argument(
        "--dst", required=True,
        help="Destination: outputs/datasets/difix_pairs",
    )
    parser.add_argument(
        "--types", default="1,2,3",
        help="Pair types to build (comma-separated: 1,2,3)",
    )
    parser.add_argument(
        "--n_views", default="1,2,3,4,5",
        help="View counts for Type 1/3 ablation (comma-separated)",
    )
    parser.add_argument(
        "--use-symlinks", action="store_true",
        help="Use symlinks instead of copying (saves disk)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview plan without creating files",
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    types_to_build = [int(t) for t in args.types.split(",")]
    n_views_list = [int(v) for v in args.n_views.split(",")]

    print(f"=== DiFix Dataset Builder ===")
    print(f"  Source: {src}")
    print(f"  Destination: {dst}")
    print(f"  Types: {types_to_build}")
    print(f"  N-views: {n_views_list}")
    print(f"  Symlinks: {args.use_symlinks}")
    print(f"  Dry run: {args.dry_run}")

    # Discover available frames
    frame_indices = discover_frame_indices(src)
    if not frame_indices:
        print("ERROR: No frames found. Run collect_dataset.py first.")
        return

    print(f"  Frames: {len(frame_indices)} ({frame_indices[0]}..{frame_indices[-1]})")
    print()

    all_entries = []

    # Build Type 1 pairs
    if 1 in types_to_build:
        print("Building Type 1 (GT view) pairs...")
        entries = build_type1_pairs(
            src, dst, frame_indices, n_views_list,
            use_symlinks=args.use_symlinks, dry_run=args.dry_run,
        )
        all_entries.extend(entries)
        print(f"  Type 1: {len(entries)} pairs")

    # Build Type 2 pairs
    if 2 in types_to_build:
        print("Building Type 2 (novel view) pairs...")
        entries = build_type2_pairs(
            src, dst, frame_indices,
            use_symlinks=args.use_symlinks, dry_run=args.dry_run,
        )
        all_entries.extend(entries)
        print(f"  Type 2: {len(entries)} pairs")

    # Build Type 3 pairs
    if 3 in types_to_build:
        print("Building Type 3 (view ablation) pairs...")
        entries = build_type3_pairs(
            src, dst, frame_indices, n_views_list,
            use_symlinks=args.use_symlinks, dry_run=args.dry_run,
        )
        all_entries.extend(entries)
        print(f"  Type 3: {len(entries)} pairs")

    # Build and save manifest
    manifest = build_manifest(all_entries, types_to_build, n_views_list, args.use_symlinks)

    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)
        manifest_path = dst / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest saved: {manifest_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"DiFix Dataset Build {'(DRY RUN) ' if args.dry_run else ''}Summary:")
    print(f"  Total pairs: {len(all_entries)}")
    for pt, count in manifest["pairs_by_type"].items():
        print(f"    {pt}: {count}")
    print(f"  By split:")
    for split, count in manifest["pairs_by_split"].items():
        print(f"    {split}: {count}")


if __name__ == "__main__":
    main()
