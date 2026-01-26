# Documentation Sync Guide

## SSOT (Single Source of Truth) Principle

presets.py (authority source)
       |
       v
docs/generated/DATASET_INDEX.md (auto-generated)

---

## Document Types

### 1. Auto-Generated (from presets.py)

| File | Source | Update Method |
|------|--------|---------------|
| docs/generated/DATASET_INDEX.md | presets.py | python generate_preset_docs.py |

### 2. Manual - Qualitative

| Location | Content |
|----------|---------|
| docs/theory/*.md | PP/MVG theory, geometry analysis |
| docs/experiments/*.md | Hypothesis verification plans |
| docs/practical/*.md | Quick reference |

### 3. Manual - Quantitative

| Location | Content |
|----------|---------|
| docs/analysis/*.md | Experiment results (PSNR/SSIM) |

---

## Workflow

### Adding New Preset

1. Edit presets.py - add preset definition
2. Add to VERSION_HIERARCHY
3. Run: python mouse_extensions/scripts/generate_preset_docs.py
4. Create configs/datasets/NEW.yaml if needed
5. Run preprocessing

### Recording Experiment Results

1. Add to docs/analysis/RESULTS_YYMMDD.md
2. Update docs/experiments/HYPOTHESIS_VERIFICATION_PLAN.md

---

## File Structure

docs/
  generated/           # Auto-generated (DO NOT EDIT)
    DATASET_INDEX.md   <- from presets.py
  theory/              # Manual (qualitative)
  experiments/         # Manual (hypothesis)
  analysis/            # Manual (quantitative)
  practical/           # Manual (reference)

---

## Important

1. NEVER edit docs/generated/ directly
2. Theory docs require manual writing
3. Record experiment results after experiments complete

---

*v1.0 | 2026-01-26*
