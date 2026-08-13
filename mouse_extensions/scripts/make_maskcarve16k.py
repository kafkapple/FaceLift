#!/usr/bin/env python3
"""Standalone CLI for the paper count-match filter: ply dir -> *_maskcarve16k dir.

Chains existing modules (no new logic):
  load ply -> compute_visibility_counts (multiview_visibility_filter)
           -> build_count_match_mask (cinematic_sequence, n-of-6 vote + top-K opacity)
           -> masked PLY write (same pattern as filter_ply.py)

Reference: PARADIGM_COMPARISON_SSOT.md §36 / CINEMATIC_V11_SPEC.md
(GS native ~100k -> 16k; defaults n_filter=5, top_k=16000).

Usage:
    python -m mouse_extensions.scripts.make_maskcarve16k \
        --ply-dir  <gaussians>/ply_a0.3 \
        --sample-dir-base <preprocessed>/M5 \
        --output   <gaussians>/my_maskcarve16k

Frame dirs are matched by frame id: ply <stem>.ply -> sample dir <stem>/.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from mouse_extensions.behavior.multiview_visibility_filter import compute_visibility_counts

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("make_maskcarve16k")


def count_match_mask(opacity: np.ndarray, vis_counts: np.ndarray,
                     n_filter: int, top_k: int) -> np.ndarray:
    """Same selection rule as cinematic_sequence.build_count_match_mask."""
    vis = vis_counts >= n_filter
    cands = np.where(vis)[0]
    mask = np.zeros_like(opacity, dtype=bool)
    if len(cands) <= top_k:
        mask[cands] = True
        return mask
    mask[cands[np.argsort(-opacity[cands])[:top_k]]] = True
    return mask


def carve_single(src: Path, dst: Path, frame_dir: Path, n_filter: int,
                 top_k: int, n_views: int) -> dict:
    ply = PlyData.read(str(src))
    vertex = ply["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    opacity = np.asarray(vertex["opacity"], dtype=np.float64)
    # GS-LRM plys store logit opacity; ranking is monotonic either way.
    vc = compute_visibility_counts(xyz, str(frame_dir), n_views=n_views)
    mask = count_match_mask(opacity, vc, n_filter, top_k)
    out = vertex.data[mask]
    dst.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out, "vertex")]).write(str(dst))
    return {"n_original": len(vertex.data), "n_kept": int(mask.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ply-dir", type=Path, required=True)
    ap.add_argument("--sample-dir-base", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-filter", type=int, default=5, help="min views (of n-views) seeing the gaussian")
    ap.add_argument("--top-k", type=int, default=16000)
    ap.add_argument("--n-views", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="process only first N plys (smoke test)")
    args = ap.parse_args()

    plys = sorted(args.ply_dir.rglob("*.ply"))
    if args.limit:
        plys = plys[: args.limit]
    if not plys:
        raise SystemExit(f"no .ply under {args.ply_dir}")
    for src in tqdm(plys, desc="count-match"):
        digits = "".join(c for c in src.stem if c.isdigit())
        frame_dir = args.sample_dir_base / (f"{int(digits):06d}" if digits else src.stem)
        if not frame_dir.is_dir():
            frame_dir = args.sample_dir_base / src.stem
        if not frame_dir.is_dir():
            log.warning("no frame dir for %s — skipped", src.stem)
            continue
        dst = args.output / src.relative_to(args.ply_dir)
        s = carve_single(src, dst, frame_dir, args.n_filter, args.top_k, args.n_views)
        log.info("%s: %d -> %d", src.stem, s["n_original"], s["n_kept"])


if __name__ == "__main__":
    main()
