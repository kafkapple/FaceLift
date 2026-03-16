#!/usr/bin/env bash
# migrate_outputs.sh — Reorganize outputs/ into structured layout
#
# Moves directories using mv (same partition, instant, no extra disk).
# Creates backward-compat symlinks at old locations.
#
# Usage:
#   bash migrate_outputs.sh --dry-run   # preview only
#   bash migrate_outputs.sh             # execute migration

set -euo pipefail

OUTPUTS="/home/joon/dev/FaceLift/outputs"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE (no changes will be made) ==="
fi

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
do_mv() {
    local src="$1" dst="$2"
    if [[ ! -e "$src" ]]; then
        echo "  SKIP (not found): $src"
        return
    fi
    if [[ -e "$dst" ]]; then
        echo "  SKIP (dst exists): $dst"
        return
    fi
    echo "  MV: $src -> $dst"
    if ! $DRY_RUN; then
        mkdir -p "$(dirname "$dst")"
        mv "$src" "$dst"
    fi
}

do_symlink() {
    local target="$1" link="$2"
    if [[ -L "$link" ]]; then
        echo "  SKIP (symlink exists): $link"
        return
    fi
    if [[ -e "$link" ]]; then
        echo "  SKIP (path exists, not symlink): $link"
        return
    fi
    echo "  LINK: $link -> $target"
    if ! $DRY_RUN; then
        ln -s "$target" "$link"
    fi
}

do_mkdir() {
    local dir="$1"
    echo "  MKDIR: $dir"
    if ! $DRY_RUN; then
        mkdir -p "$dir"
    fi
}

# -------------------------------------------------------
# Pre-flight: disk usage
# -------------------------------------------------------
echo ""
echo "=== Before migration: disk usage ==="
du -sh "$OUTPUTS"/ 2>/dev/null || echo "  outputs/ not found"
echo ""

# -------------------------------------------------------
# Step 1: Create target directories
# -------------------------------------------------------
echo "=== Step 1: Create target structure ==="
do_mkdir "$OUTPUTS/datasets"
do_mkdir "$OUTPUTS/experiments"
do_mkdir "$OUTPUTS/visualizations"
do_mkdir "$OUTPUTS/_archive"

# -------------------------------------------------------
# Step 2: Move datasets
# -------------------------------------------------------
echo ""
echo "=== Step 2: Move dataset directories ==="

# novel_view_dataset -> datasets/novel_view
do_mv "$OUTPUTS/novel_view_dataset" "$OUTPUTS/datasets/novel_view"

# tier_comparison -> datasets/view_ablation
do_mv "$OUTPUTS/tier_comparison" "$OUTPUTS/datasets/view_ablation"

# -------------------------------------------------------
# Step 3: Move experiments
# -------------------------------------------------------
echo ""
echo "=== Step 3: Move experiment directories ==="

do_mv "$OUTPUTS/phase3_e2e" "$OUTPUTS/experiments/phase3_e2e"
do_mv "$OUTPUTS/comparison" "$OUTPUTS/experiments/comparison"

# -------------------------------------------------------
# Step 4: Move visualizations
# -------------------------------------------------------
echo ""
echo "=== Step 4: Move visualization directories ==="

do_mv "$OUTPUTS/camera_follow" "$OUTPUTS/visualizations/camera_follow"
do_mv "$OUTPUTS/triangulation" "$OUTPUTS/visualizations/triangulation"

# -------------------------------------------------------
# Step 5: Move archives
# -------------------------------------------------------
echo ""
echo "=== Step 5: Move archive directories ==="

do_mv "$OUTPUTS/_archive_renders" "$OUTPUTS/_archive/renders"
do_mv "$OUTPUTS/_archive_analysis" "$OUTPUTS/_archive/analysis"
do_mv "$OUTPUTS/poc_mesh_gs_pairs_v0_archive" "$OUTPUTS/_archive/poc_v0"

# Old difix_pairs (PoC format) -> archive
do_mv "$OUTPUTS/difix_pairs" "$OUTPUTS/_archive/difix_pairs_v0"

# -------------------------------------------------------
# Step 6: Backward-compat symlinks
# -------------------------------------------------------
echo ""
echo "=== Step 6: Create backward-compat symlinks ==="

do_symlink "$OUTPUTS/datasets/novel_view" "$OUTPUTS/novel_view_dataset"
do_symlink "$OUTPUTS/datasets/difix_pairs" "$OUTPUTS/difix_pairs"

# -------------------------------------------------------
# Post-flight: disk usage
# -------------------------------------------------------
echo ""
echo "=== After migration: disk usage ==="
if ! $DRY_RUN; then
    du -sh "$OUTPUTS"/*/ 2>/dev/null || echo "  (empty)"
    echo ""
    echo "Total:"
    du -sh "$OUTPUTS"/ 2>/dev/null
else
    echo "  (dry run — no changes made)"
fi

echo ""
echo "=== Migration complete ==="
