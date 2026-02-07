# RTX 3060 (12GB VRAM) Configs

Lightweight configs for running FaceLift experiments on consumer GPUs.

## GS-LRM
- `gslrm_3060.yaml`: 4-view, batch=1, fp16
- Expected VRAM: ~10-11GB
- Usage: `python train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml -e configs/mouse/rtx3060/gslrm_3060.yaml`

## MVDiffusion
- `mvdiffusion_3060.yaml`: 6-view, batch=1, grad_accum=16
- Expected VRAM: ~5-6GB
- Usage: `cd mvdiffusion && accelerate launch --config_file 1gpu.yaml train_diffusion.py --config ../configs/mouse/rtx3060/mvdiffusion_3060.yaml`

## Notes
- Batch reduced but effective batch preserved via gradient accumulation
- Training will be slower but results should be equivalent
- GS-LRM uses fp16 instead of bf16 (better RTX 3060 performance)
- Turntable views reduced (144 -> 36) to avoid validation VRAM spikes
