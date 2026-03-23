# Keypoint Ablation Framework

5-tier keypoint subset system for behavior analysis granularity control.

---

## Overview

Not all behavioral tasks need the full 22/23-keypoint skeleton. This framework
defines **5 progressive tiers** of keypoint subsets, each adding anatomical
detail required by increasingly fine-grained tasks.

**SSOT**: `configs/keypoints/{species}.yaml` → `ablation_tiers` section.
**Loader**: `mouse_extensions.keypoint_config.load_keypoint_config(species)`.

## Tier Definitions

### T1: Axis (3 keypoints)

| Mouse | Rat |
|-------|-----|
| nose(2), body_middle(4), tail_root(5) | Snout(17), SpineM(1), SpineL(2) |

**Purpose**: Primary body axis — position, heading direction, locomotion speed.
**Literature**: Standard centroid-based tracking (Mathis 2018, DeepLabCut).
**Tasks**: Trajectory analysis, speed computation, heading direction, open-field test metrics.

### T2: Head (6 keypoints)

Adds: ears, neck/SpineF to T1.

**Purpose**: Head orientation — social gaze, attention direction, exploratory behavior.
**Literature**: Dunn 2021 (DANNCE social interaction), Pereira 2022 (SLEAP social).
**Tasks**: Social interaction scoring, gaze direction, ear asymmetry (stress indicator).

### T3: Posture (10 keypoints)

Adds: shoulders, hips to T2.

**Purpose**: Proximal limb configuration — posture classification.
**Literature**: Nath 2022 (rearing detection), Wiltschko 2015 (MoSeq syllables).
**Tasks**: Rearing, grooming, crouching classification, body elongation metrics.

### T4: Locomotion (14 keypoints)

Adds: paws/feet, distal joints to T3.

**Purpose**: Gait analysis, paw-object interaction.
**Literature**: Machado 2015 (LocoMouse), Hausmann 2021 (gait quantification).
**Tasks**: Stride length/frequency, paw placement, skilled reaching.

### T5: Skeleton (22/23 keypoints)

Full keypoint set.

**Purpose**: Joint angles, fine motor analysis, 3D reconstruction fidelity.
**Literature**: Dunn 2021 (full DANNCE), Marshall 2021 (3D pose dynamics).
**Tasks**: Joint angle computation, kinematic chain analysis, 3D Gaussian body-part isolation.

## Task-Tier Activation Matrix

| Task | Minimum Tier | Notes |
|------|:------------:|-------|
| Centroid tracking | T1 | 3 points sufficient |
| Speed / heading | T1 | body_middle velocity |
| Social gaze | T2 | Ear + nose orientation |
| Rearing detection | T3 | Shoulder/hip elevation |
| Grooming | T3 | Proximal limb posture |
| Gait analysis | T4 | Paw trajectories |
| Skilled reaching | T4 | Paw + elbow coordination |
| 3D reconstruction | T5 | Full skeleton needed |
| HLAC features | T5 | Covariance over all joints |
| Body-part rendering | T5 | All joints for segmentation |

## Usage

```python
from mouse_extensions.keypoint_config import load_keypoint_config

cfg = load_keypoint_config("mouse")

# Select tier for your task
tier_indices = cfg.ablation_tiers["posture"]  # T3: 10 keypoints
kp_subset = all_keypoints[:, tier_indices, :]  # (T, 10, 3)

# Available tiers
print(list(cfg.ablation_tiers.keys()))
# ['axis', 'head', 'posture', 'locomotion', 'skeleton']
```

## Adding a New Species

1. Create `configs/keypoints/{species}_{num_kp}.yaml`
2. Define all sections (see `mouse_22.yaml` as template)
3. Verify: `python -c "from mouse_extensions.keypoint_config import load_keypoint_config; print(load_keypoint_config('{species}'))"`
4. Add species-specific exports to `constants.py` if needed

## Cross-Species Tier Alignment

Tiers are **functionally aligned** (same behavioral purpose) across species,
not **index-aligned** (different keypoint numbering). The tier names
(`axis`, `head`, `posture`, `locomotion`, `skeleton`) are consistent.

---

*Created: 2026-03-23 | SSOT: configs/keypoints/*.yaml*
