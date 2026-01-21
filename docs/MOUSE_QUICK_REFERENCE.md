# FaceLift Mouse Extension - Quick Reference

> **Last Updated**: 2026-01-21
> **Full Documentation**: Obsidian `30_Projects/_CODES/code_Face_Lift/docs/`

---

## 1. Training

```bash
# Basic training
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/D6_E1.yaml

# Multi-GPU
torchrun --standalone --nproc_per_node=4 train_gslrm.py --config configs/mouse/D6_E1.yaml
```

---

## 2. Inference

```bash
# From 6-view sample
./scripts/run_inference.sh data/D6-3/sample_000100

# Single image with Zero123++
./scripts/run_inference.sh mouse.png --use_zero123pp

# Manual inference
python inference_mouse.py \
    --sample_dir data/D6-3/sample_000100 \
    --checkpoint checkpoints/gslrm/mouse/ \
    --output_dir outputs/inference/
```

---

## 3. Key Configuration

### 3.1 Turntable Visualization

```yaml
visualization:
  turntable:
    num_views: 64           # Grid views (8x8)
    resolution: 384
    elevation: 20
    radius: 2.7
    inference_views: 150    # Video frames
    fps: 30
```

### 3.2 Camera Exclusion (Ablation)

```yaml
training:
  dataset:
    exclude_camera_indices: [3]     # Exclude camera 3
    # include_camera_indices: [0,1,2,4,5]  # Or explicit include
```

### 3.3 Loss Configuration

```yaml
training:
  losses:
    l2_loss_weight: 1.0
    perceptual_loss_weight: 0.5
    mask_mode: "gt"              # none, gt, alpha
    alpha_loss_weight: 0.1       # Optional
```

---

## 4. Key Code Locations

| Feature | File | Line |
|---------|------|------|
| Projection Matrix | `gslrm/model/gaussians_renderer.py` | :287-306 |
| Loss Computation | `mouse_extensions/model/loss_extensions.py` | - |
| Turntable Render | `gslrm/model/gaussians_renderer.py` | :974 |
| Camera Exclusion | `gslrm/data/mouse_dataset.py` | :254-270 |

---

## 5. Preprocessing

```bash
# D6-3 preprocessing (recommended)
python -m mouse_extensions.preprocessing.preprocessor_d6 \
    --method D6-3 \
    --input_dir /path/to/raw \
    --output_dir /path/to/D6-3

# Verify
python -m mouse_extensions.preprocessing.format_validator /path/to/D6-3
```

---

## 6. Documentation Map

| Topic | Location |
|-------|----------|
| Full Theory & Analysis | Obsidian `260121_Comprehensive_Review_Plan.md` |
| Mask/Loss Guide | Obsidian `260121_Mask_Loss_Educational_Guide.md` |
| Paper Comparison | Obsidian `260121_Paper_Settings_Comparison.md` |
| Preprocessing Registry | `docs/PREPROCESSING_REGISTRY.md` |

---

*Quick Reference - See Obsidian for detailed documentation*
