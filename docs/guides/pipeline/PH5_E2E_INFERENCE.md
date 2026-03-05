# Phase 5: E2E Inference Pipeline

> **Navigation**: [← Hub](../PIPELINE_DEEP_DIVE.md) | [Prev: PH4](PH4_POSE_CONDITIONING.md) | [Next: PH6 →](PH6_FAIR_EVALUATION.md)
>
> **핵심 파일**: `mouse_extensions/inference/end_to_end.py`, `mouse_extensions/scripts/inference/run_e2e_inference.py`

---

> **Why**: GS-LRM은 GT 6-view로 PSNR 23.84를 달성하지만,
> 실제 deployment에서는 단일 이미지만 가용. MVDiffusion이 6-view를 생성하고
> GS-LRM이 3D로 재구성하는 End-to-End 파이프라인이 필요.

### 5.1 Pipeline Architecture

**File**: `mouse_extensions/inference/end_to_end.py`

```
┌─────────────────────────────────────────────────────────────────┐
│ EndToEndPipeline (L22-307)                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Input: Single image (or M5 sample)                              │
│                              │                                  │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ Stage 0 (Optional): SAM Preprocessing (L147-171)         │   │
│ │   SAM detection → white BG → center align → normalize    │   │
│ └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ Stage 1: MVDiffusion (L174-189)                          │   │
│ │   Input : [1, 3, H, W] reference image                   │   │
│ │   Model : SD2.1-UnCLIP + Era3D RMA                       │   │
│ │   Output: [6, 3, H, W] multi-view images                 │   │
│ │   ★ Pose conditioning injected here (if configured)      │   │
│ └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ Stage 2: GS-LRM (L191-216)                               │   │
│ │   Input : images [1, V, 3, H, W]                         │   │
│ │           c2ws   [1, V, 4, 4]     (from m5_cameras.json) │   │
│ │           fxfycxcys [1, V, 4]                             │   │
│ │           index  [1, V, 2]        (view_idx, scene_idx)   │   │
│ │   Output: 3D Gaussian parameters → PLY, video, RRD       │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│ run_from_views() (L228-307) — GS-LRM only mode                 │
│   View ablation (L265-290):                                     │
│     1v → [0], 2v → [0,3], 3v → [0,2,4] (evenly distributed)   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 CLI Entry Point

**File**: `mouse_extensions/scripts/inference/run_e2e_inference.py`

#### Path 분기 로직 (L312-317)

```python
use_mvdiffusion = (
    args.input_image is not None or
    (args.sample_dir is not None and args.input_view_idx is not None) or
    (args.data_dir is not None and args.input_view_idx is not None)
)
```

| Path | Trigger | 용도 |
|------|---------|------|
| **Path 1** | `--sample_dir` (no input_view_idx) | GS-LRM only (GT views) |
| **Path 2a** | `--input_image` | E2E from single image |
| **Path 2b** | `--data_dir + --input_view_idx` | E2E batch (test set) |
| **Batch Path 1** | `--data_dir` (no input_view_idx) | GS-LRM only batch |

#### Batch Path 2b 상세 (L482-518) — 가장 일반적인 사용

```
for each sample in data_dir/split.txt:
  1. Load reference image: sample/images/cam_{input_view_idx:03d}.png
  2. MVDiffusion inference → 6 views
  3. GS-LRM inference → Gaussians
  4. Render turntable + eval views
  5. Save to output_dir/samples/{sample_id}/
```

### 5.3 Commands

```bash
# ★ E2E Batch Inference (test set, MOST COMMON)
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt --input_view_idx 0 --skip_preprocess \
    --model M5t --gslrm_config configs/base/gslrm_mouse.yaml \
    --gslrm_checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt \
    --mvdiffusion_checkpoint <CHECKPOINT_PATH> \
    --mvdiffusion_base checkpoints/mvdiffusion/pipeckpts \
    --prompt_embed_path mvdiffusion/data/mouse_prompt_embeds_6view_1024 \
    --prefer_ema --no_turntable --no_mesh \
    --num_steps 50 --guidance_scale 3.0 --seed 42 \
    --output_dir outputs/phase3_e2e/<experiment_name>

# GS-LRM only (GT input, for upper bound)
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt --skip_preprocess \
    --model M5t --gslrm_config configs/base/gslrm_mouse.yaml \
    --gslrm_checkpoint <CHECKPOINT_PATH> \
    --no_turntable --no_mesh \
    --output_dir outputs/tier_comparison/<experiment_name>
```

### 5.4 E2E Verification Protocol

> ⚠️ **CRITICAL**: `--input_view_idx` 누락 시 GS-LRM only (Path 1)로 실행됨.
> PSNR_gt ~20 dB (GT input) vs ~8 dB (E2E) — 혼동 주의.

```bash
# 1. Pipeline Path 확인
grep 'Batch Path' LOG   # → "Path 2b: MVDiffusion" 확인

# 2. MVDiffusion 로딩 확인
grep 'Loading MVDiffusion' LOG   # → 존재해야 함

# 3. run_config.json 확인
python -c "import json; d=json.load(open('run_config.json')); print(d['args']['input_view_idx'])"
# → 0 이어야 함 (null이면 GS-LRM only)

# 4. Sanity check: E2E PSNR_fg 범위 7-10 dB. 12+ dB → GS-LRM only 오인

# 5. generated_views/ 디렉토리 존재 확인 (E2E에서만 생성)
ls outputs/.../samples/sample_003240/generated_views/
```

---

*← [PH4](PH4_POSE_CONDITIONING.md) | [Hub](../PIPELINE_DEEP_DIVE.md) | [Next: PH6 →](PH6_FAIR_EVALUATION.md)*
