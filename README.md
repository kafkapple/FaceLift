# FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads

### 🌺 ICCV 2025 🌺

[Weijie Lyu](https://weijielyu.github.io/), [Yi Zhou](https://zhouyisjtu.github.io/), [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/), [Zhixin Shu](https://zhixinshu.github.io/)  
University of California, Merced - Adobe Research

[![Website](https://img.shields.io/badge/Website-FaceLift?logo=googlechrome&logoColor=hsl(204%2C%2086%25%2C%2053%25)&label=FaceLift&labelColor=%23f5f5dc&color=hsl(204%2C%2086%25%2C%2053%25))](https://weijielyu.github.io/FaceLift)
[![Paper](https://img.shields.io/badge/Paper-arXiv?logo=arxiv&logoColor=%23B31B1B&label=arXiv&labelColor=%23f5f5dc&color=%23B31B1B)](https://arxiv.org/abs/2412.17812)
[![Video](https://img.shields.io/badge/Video-YouTube?logo=youtube&logoColor=%23FF0000&label=YouTube&labelColor=%23f5f5dc&color=%23FF0000)](https://youtu.be/H-EZKmuYvRM)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace?logo=huggingface&logoColor=%23FFD21E&label=HuggingFace&labelColor=%23f5f5dc&color=%23FFD21E)](https://huggingface.co/spaces/wlyu/FaceLift)

<div align='center'>
<img alt="image" src='media/teaser.png'>
</div>

> *FaceLift* transforms a single facial image into a high-fidelity 3D Gaussian head representation, and it generalizes remarkably well to real-world human images.

This is a self-reimplementation of *FaceLift*.

## 🔧 Prerequisites

### Model Checkpoints

Model checkpoints will be automatically downloaded from [HuggingFace](https://huggingface.co/wlyu/OpenFaceLift) on first run.

Alternatively, you can manually place the checkpoints in the `checkpoints/` directory:
- `checkpoints/mvdiffusion/pipeckpts/` - Multi-view diffusion model checkpoints
- `checkpoints/gslrm/ckpt_0000000000021125.pt` - GS-LRM model checkpoints
s
### Environment Setup

```bash
bash setup_env.sh
```

## 🚀 Inference

### Command Line Interface

Process images from a directory:

```bash
python inference.py --input_dir examples/ --output_dir outputs/
```

**Available Arguments:**
| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input_dir` | `-i` | `examples/` | Input directory containing images |
| `--output_dir` | `-o` | `outputs/` | Output directory for results |
| `--auto_crop` | - | `True` | Automatically crop faces |
| `--seed` | - | `4` | Random seed for reproducible results |
| `--guidance_scale_2D` | - | `3.0` | Guidance scale for multi-view generation |
| `--step_2D` | - | `50` | Number of diffusion steps |

### Web Interface

Launch the interactive Gradio web interface:

```bash
python gradio_app.py
```

Open your browser and navigate to `http://localhost:7860` to use the web interface. If running on a server, use the provided public link.

## 🎓 Training

### Data Structure

Training data are currently not available. to train with your own data, follow the structure in `FaceLift/data_sample/`:

**Multi-view Diffusion Data:**
```
data_sample/
├── mvdiffusion/
│   ├── data_mvdiff_train.txt          # Training data list
│   ├── data_mvdiff_val.txt            # Validation data list
│   └── sample_000/
│       ├── cam_000.png                # Front view (RGBA, 512×512)
│       ├── cam_001.png                # Front-right view
│       ├── cam_002.png                # Right view
│       ├── cam_003.png                # Back view
│       ├── cam_004.png                # Left view
│       └── cam_005.png                # Front-left view
```

**GS-LRM Data:**
```
data_sample/
├── gslrm/
│   ├── data_gslrm_train.txt           # Training data list
│   ├── data_gslrm_val.txt             # Validation data list
│   └── sample_000/
│       ├── images/
│       │   ├── cam_000.png            # Multi-view images (RGBA, 512×512)
│       │   ├── cam_001.png
│       │   ├── ...
│       │   └── cam_031.png            # 32 views total
│       └── opencv_cameras.json        # Camera parameters
```

### Multi-view Diffusion Training

```bash
accelerate launch --config_file mvdiffusion/node_config/8gpu.yaml \
    train_diffusion.py --config configs/mvdiffusion.yaml
```

### Gaussian Reconstructor Training

Our Gaussian Reconstructor is based on GS-LRM and uses pre-trained weights from Objaverse data.

- Stage I: 256 resolution on Objaverse - `gslrm_pretrain_256.yaml`
- Stage II: 512 resolution on Objaverse - `gslrm_pretrain_512.yaml`
- Stage III: 512 resolution on synthetic heads data - `gslrm.yaml`

```bash
torchrun --nproc_per_node 8 --nnodes 1 \
    --rdzv_id ${JOB_UUID} --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    train_gslrm.py --config configs/gslrm.yaml
```

## 📝 Citation

If you find our work useful for your research, please consider citing our paper:

```bibtex
@InProceedings{Lyu_2025_ICCV,
    author    = {Lyu, Weijie and Zhou, Yi and Yang, Ming-Hsuan and Shu, Zhixin},
    title     = {FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {12691-12701}
}
```

## 📄 License

Copyright 2025 Adobe Inc.

Codes are licensed under [Apache-2.0 License](http://www.apache.org/licenses/LICENSE-2.0).

Model weights are licensed from Adobe Inc. under the [Adobe Research License](Adobe%20Research%20License%20v1.2.txt).

## 🐭 Mouse Domain Adaptation (Quick Start)

FaceLift can be adapted to other domains like mouse/animal reconstruction. This section covers the mouse dataset pipeline.

### Phase 1: Data Preparation

**1. Convert Raw Data to FaceLift Format**

```bash
# Input: markerless_mouse_1_nerf (6-camera videos + masks)
# Output: FaceLift format (512x512 images per frame)
python scripts/convert_markerless_to_facelift.py \
    --input_dir /path/to/markerless_mouse_1_nerf \
    --output_dir data_mouse \
    --frame_interval 5 \
    --target_size 512 \
    --target_ratio 0.6
```

**2. Center-Align All Views**

```bash
# Apply consistent centering across all views
python scripts/preprocess_center_align_all_views.py \
    --input_dir data_mouse \
    --output_dir data_mouse_centered \
    --target_ratio 0.6
```

### Phase 2: MVDiffusion Training

**Standard Training (24GB+ VRAM)**

```bash
torchrun --nproc_per_node=1 train_diffusion.py \
    --config configs/mouse_mvdiffusion_centered_real.yaml
```

**Low Memory Training (RTX 3060 12GB)**

```bash
torchrun --nproc_per_node=1 train_diffusion.py \
    --config configs/mouse_mvdiffusion_lowmem.yaml
```

### Phase 3: GS-LRM Fine-tuning

**Standard Training (24GB+ VRAM)**

```bash
torchrun --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse_gslrm_synthetic.yaml
```

**Low Memory Training (RTX 3060 12GB)**

```bash
torchrun --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse_gslrm_lowmem.yaml
```

### Memory Requirements

| Config | GPU VRAM | Effective Batch | Notes |
|--------|----------|-----------------|-------|
| `*_centered_real.yaml` | 24GB+ | 16 | Full quality |
| `*_lowmem.yaml` | 12GB | 16 | RTX 3060 compatible |

**Low Memory Optimizations:**
- `batch_size: 4 → 1` with `gradient_accumulation: 4 → 16`
- `use_ema: false` (saves ~1.5GB)
- `use_8bit_adam: true`
- `amp_dtype: fp16` (better on consumer GPUs)

### Data Structure

```
data_mouse_centered/
├── sample_000000/
│   ├── images/
│   │   ├── cam_000.png    # Reference view (512x512)
│   │   ├── cam_001.png    # Multi-view images
│   │   └── ...
│   └── opencv_cameras.json
├── sample_000001/
└── ...
├── data_mouse_train.txt   # Training sample paths
└── data_mouse_val.txt     # Validation sample paths
```

## 🙏 Acknowledgements

This work is built upon [Era3D](https://penghtyx.github.io/Era3D/) and [GS-LRM](https://sai-bi.github.io/project/gs-lrm/). We thank the authors for their excellent work.

The code has been reimplemented and the weights retrained. Results may differ slightly from those reported in the paper.
