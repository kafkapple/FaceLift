# RTX 3060 (12GB VRAM) Configs

Lightweight configs for running FaceLift experiments on consumer GPUs.

## Quick Start

Recommended training order:

```bash
# 1. GS-LRM (train first)
python train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/rtx3060/gslrm_3060.yaml

# 2. MVDiffusion (after GS-LRM converges)
cd mvdiffusion && accelerate launch --config_file 1gpu.yaml \
  train_diffusion.py --config ../configs/mouse/rtx3060/mvdiffusion_3060.yaml

# 3. E2E (combine trained MVDiff + GS-LRM)
# TODO: add E2E config when ready
```

## Configs

### GS-LRM

| Config | Views | VRAM | Notes |
|--------|:-----:|:----:|-------|
| `gslrm_3060.yaml` | 4 | ~10-11GB | Default, LPIPS disabled |
| `gslrm_3060_3view.yaml` | 3 | ~9-10GB | Fewer views, slightly less VRAM |

Usage:
```bash
# 4-view (default)
python train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/rtx3060/gslrm_3060.yaml

# 3-view
python train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/rtx3060/gslrm_3060_3view.yaml
```

### MVDiffusion

| Config | Views | VRAM | Notes |
|--------|:-----:|:----:|-------|
| `mvdiffusion_3060.yaml` | 6 | ~5-6GB | Default 6-view |
| `mvdiffusion_3060_3view.yaml` | 3 | ~4-5GB | 3-view (cameras [0,2,4]) |
| `mvdiffusion_3060_8bit.yaml` | 6 | ~4-5GB | 8-bit Adam, requires bitsandbytes |

Usage:
```bash
cd mvdiffusion

# 6-view (default)
accelerate launch --config_file 1gpu.yaml \
  train_diffusion.py --config ../configs/mouse/rtx3060/mvdiffusion_3060.yaml

# 3-view
accelerate launch --config_file 1gpu.yaml \
  train_diffusion.py --config ../configs/mouse/rtx3060/mvdiffusion_3060_3view.yaml

# 6-view + 8-bit Adam (install bitsandbytes first)
pip install bitsandbytes
accelerate launch --config_file 1gpu.yaml \
  train_diffusion.py --config ../configs/mouse/rtx3060/mvdiffusion_3060_8bit.yaml
```

## Common Settings (vs A6000 baseline)

| Parameter | A6000 Baseline | RTX 3060 |
|-----------|:--------------:|:--------:|
| `train_batch_size` | 4 (MVDiff) / 2 (GS-LRM) | **1** |
| `gradient_accumulation_steps` | 4 | **16** |
| `validation_batch_size` | 2 | **1** |
| `dataloader_num_workers` | 8 | **4** |
| `amp_dtype` (GS-LRM) | bf16 | **fp16** |
| `lpips_loss_weight` (GS-LRM) | 0.5 | **0.0** |
| Turntable views (GS-LRM) | 144 | **36** |

## Notes

- Batch reduced but effective batch preserved via gradient accumulation
- Training will be slower (~4x wall time) but results should be equivalent
- GS-LRM uses fp16 instead of bf16 (better RTX 3060 performance)
- LPIPS/perceptual loss disabled in GS-LRM to save ~200MB VRAM
- Turntable views reduced (144 -> 36) to avoid validation VRAM spikes
