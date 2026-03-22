# paths.py SSOT Design — Centralized Path Management

> **Status**: DESIGNED (구현 대기) | **Priority**: P1 | **Blast Radius**: 38+ files
> **Prerequisite**: Rat FT 학습 완료 후 구현 (실행 중 paths.py 변경 시 checkpoint 오류 위험)

---

## 1. Why: Problem Statement

### Incident (2026-03-22)

```
S18 disk cleanup → config grep only → "no references" → deleted MAMMAL_mouse/
→ 38 Python files referencing keypoints_22_3d.npz via hardcoded absolute paths
→ Simultaneous failure across behavior analysis pipeline
```

**Root cause**: 46 Python files contain hardcoded absolute paths (`/node_data/joon/...`, `/home/joon/...`). No central registry exists, so dependency analysis is impossible without grepping all source code.

### Current State: 4 Fragmented Path Systems

| File | Role | Problem |
|------|------|---------|
| `mouse_extensions/paths.py` | General infra (FACELIFT_ROOT, DATA_ROOT) | Missing keypoints, checkpoints |
| `mouse_extensions/behavior/paths.py` | Clustering outputs + 2 GPU03 constants | String constants, not Path objects |
| `scripts/inference/defaults.py` | 5 inference path constants | Isolated, not shared |
| `configs/datasets/*.yaml` | Tilde-expansion (`~/data/...`) | Different mechanism, no cross-ref |

### Impact Numbers

- **46 files** with hardcoded absolute paths
- **38 files** referencing the same keypoints path independently
- **3 storage tiers** (NVMe/SSD/NFS) with no programmatic mapping
- **0 validation** — deleted data discovered only when scripts crash

---

## 2. How: Solution Design

### Architecture: Single paths.py with Storage Tier Mapping

```python
# mouse_extensions/paths.py (redesigned)

import os
from pathlib import Path

# Storage tier roots (env var override for portability)
HOT  = Path(os.getenv("FACELIFT_HOT",  "/node_data/joon"))     # NVMe
WARM = Path(os.getenv("FACELIFT_WARM", "/node_data_2/joon"))    # SSD
COLD = Path(os.getenv("FACELIFT_COLD", "/home/joon"))           # NFS

# Project root (auto-detect from file location)
PROJECT = Path(os.getenv("FACELIFT_ROOT", Path(__file__).parent.parent.resolve()))

# === Critical Data Assets (SSOT) ===
# Rule: register here only if 2+ files share the path
KP_22       = COLD / "data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz"
M5_DATA     = COLD / "data/preprocessed/FaceLift_mouse/M5"
M5T2_DATA   = COLD / "data/preprocessed/FaceLift_mouse/M5t2"
RAT_DATA    = PROJECT / "outputs/sdannce_rat_ft/gslrm_format"
CKPT_DIR    = HOT / "checkpoints/FaceLift/gslrm"
WANDB_DIR   = WARM / "wandb_runs"

# === Project-internal Paths ===
OUTPUTS     = PROJECT / "outputs"
CONFIGS     = PROJECT / "configs"
REPORT_BASE = OUTPUTS / "report/clustering"

# === Feature Paths (clustering analysis) ===
FEATURES_DIR    = REPORT_BASE / "features"
GAUSSIAN_RAW    = FEATURES_DIR / "gaussian_raw_features.npz"
COV_N2_STATIC   = FEATURES_DIR / "covariance_n2/covariance_static.npy"
COV_N2_TEMPORAL  = FEATURES_DIR / "covariance_n2/covariance_temporal.npy"
TEMPORAL_FEATURES = FEATURES_DIR / "temporal/temporal_features.npz"

# === Validation ===
def validate(*paths: Path) -> list[Path]:
    """Return list of missing paths. Empty = all OK."""
    return [p for p in paths if not p.exists()]

def check_critical():
    """Call before training/analysis. Warns on missing critical files."""
    missing = validate(KP_22, M5_DATA)
    if missing:
        import warnings
        for p in missing:
            warnings.warn(f"MISSING: {p}")
    return len(missing) == 0
```

### Migration Pattern (per file)

```python
# BEFORE (hardcoded):
KP_PATH = "/node_data/joon/data/results/MAMMAL_mouse/.../keypoints_22_3d.npz"
kp = np.load(KP_PATH)

# AFTER (SSOT import):
from mouse_extensions.paths import KP_22
kp = np.load(KP_22)

# BEFORE (argparse default):
parser.add_argument("--keypoints", default="/node_data/joon/...")

# AFTER:
from mouse_extensions.paths import KP_22
parser.add_argument("--keypoints", default=str(KP_22))
```

### behavior/paths.py Handling

Merge data paths into top-level paths.py. Keep output directory structure:

```python
# mouse_extensions/behavior/paths.py (after migration)
from mouse_extensions.paths import KP_22, M5_DATA, REPORT_BASE  # SSOT imports

# Output subdirectories (behavior-specific, keep here)
DATA_DIR = REPORT_BASE / "data"
FEATURES_DIR = REPORT_BASE / "features"
# ... (existing 16 subdirectory constants unchanged)

# Backward compatibility
GPU03_KEYPOINTS = str(KP_22)   # deprecated alias
GPU03_M5_DATA = str(M5_DATA)   # deprecated alias
```

---

## 3. What: Implementation Plan

### Phase 0: Validation Only (30 min) — Do First

Add `check_critical()` and `validate()` to existing paths.py WITHOUT changing any other file.

```
Files modified: 1 (mouse_extensions/paths.py)
Risk: Zero (additive only)
```

### Phase 1: SSOT Constants (1 hour)

Add all shared path constants (KP_22, M5_DATA, etc.) to paths.py. Update behavior/paths.py to import from top-level.

```
Files modified: 2 (paths.py, behavior/paths.py)
Risk: Low (backward-compatible aliases maintained)
```

### Phase 2a: Core Migration (1-2 hours)

Replace hardcoded paths in behavior/ core files (13 files):

```bash
# Files to migrate (highest impact):
hlac_comprehensive.py, hlac_m5t2.py, capsule_filter.py,
multiview_visibility_filter.py, unified_pipeline.py,
h1_sdannce_probe.py, ...
```

### Phase 2b: Scripts Migration (1 hour)

Replace in scripts/ (8+ files):
```bash
render_camera_follow.py, visualize_keypoint_inference.py,
eval/compute_h1_metrics.py, eval/convert_m5_for_ps.py, ...
```

### Phase 2c: _experiments/ (SKIP or minimal)

13 one-off scripts — add `from mouse_extensions.paths import KP_22` header only. Don't refactor internals.

### Phase 3: Config YAML (DEFER to post-NeurIPS)

OmegaConf custom resolver integration. Current tilde-expansion works fine.

### Safety Checklist

```
Before starting:
- [ ] No training jobs running (check nvidia-smi)
- [ ] Git commit current state
- [ ] Run existing tests to establish baseline

After each phase:
- [ ] grep -r '/node_data/joon' mouse_extensions/ --include='*.py' | wc -l  (should decrease)
- [ ] python -c "from mouse_extensions.paths import KP_22; print(KP_22)"
- [ ] Run one behavior script to verify imports work

After all phases:
- [ ] Full grep audit: zero hardcoded paths in core files
- [ ] check_critical() returns True on gpu03
- [ ] Git commit with detailed message
```

---

## 4. Estimated Effort

| Phase | Time | Files | Risk |
|:-----:|:----:|:-----:|:----:|
| 0 | 30 min | 1 | Zero |
| 1 | 1 hour | 2 | Low |
| 2a | 1-2 hours | 13 | Medium |
| 2b | 1 hour | 8+ | Medium |
| 2c | Skip | — | — |
| 3 | Defer | — | — |
| **Total** | **3-4 hours** | **24** | |

---

## Related

- Handoff: `docs/experiments/SESSION_HANDOFF_260322_S19.md` §7
- Storage: `gpu03 Storage Management.md` (Obsidian)
- Deliberation: 3-model consensus on SSOT approach (260322)

---

*Created: 2026-03-22 | paths.py SSOT Design Spec*
