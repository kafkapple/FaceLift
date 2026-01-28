# MVDiffusion M5 Finetune Guide

> Single Image → Multi-View Diffusion → 3D Reconstruction 파이프라인 | 2026-01-28

---

## 1. Overview

```
Single Image ──→ MVDiffusion (6 views) ──→ GS-LRM (3D Gaussians) ──→ Novel View Rendering
                  ↑ finetune 대상            ↑ pretrained (frozen)
```

**목적**: Mouse 단일 이미지에서 6개 뷰를 생성하고, GS-LRM으로 3D 재구성하는 파이프라인의 MVDiffusion 단계를 finetune.

**핵심 설정**:
- 데이터: M5 전처리 (Affine, D7.1 preset, 512×512, fx=549, cx=cy=256)
- Train: 3,240 samples / Val: 360 samples
- Loss 범위: ~0.02–0.05

---

## 2. Prerequisites

### Conda 환경

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate facelift
```

### Pretrained Weights

MVDiffusion pretrained checkpoint가 `checkpoints/` 디렉토리에 필요. Config에서 `pretrained_model_name_or_path`로 참조.

---

## 3. Config

**파일**: `configs/mvdiffusion/mouse_mvdiffusion_M5.yaml`

주요 설정:

| 항목 | 값 | 설명 |
|------|-----|------|
| `resolution` | 512 | 입출력 해상도 |
| `num_views` | 6 | 생성 뷰 수 |
| `fx` | 549 | Focal length (pretrained 호환) |
| `cx, cy` | 256 | Principal point |
| `train_samples` | 3,240 | 학습 데이터 수 |
| `val_samples` | 360 | 검증 데이터 수 |

---

## 4. diffusers 0.36.0 Compatibility Patches

diffusers 0.36.0에서 deprecated/제거된 API 4가지에 대한 호환성 패치가 필요.

### Patch 1: `_load_state_dict_into_model` 제거

diffusers 0.36.0에서 내부 함수 `_load_state_dict_into_model`이 제거됨.

```python
# Before (broken)
from diffusers.utils import _load_state_dict_into_model

# After (compat wrapper)
def _load_state_dict_into_model_compat(model, state_dict):
    """Compatibility wrapper for diffusers >= 0.36.0"""
    model.load_state_dict(state_dict, strict=False)
    return model
```

### Patch 2: `load_state_dict(variant=)` 파라미터 제거

```python
# Before (broken)
pipe.load_state_dict(state_dict, variant="fp16")

# After (remove variant param)
pipe.load_state_dict(state_dict)
```

### Patch 3: `_gradient_checkpointing_func` 누락

Gradient checkpointing 활성화 시 내부 함수 참조 누락.

```python
# Manual fix: set the function explicitly
import torch.utils.checkpoint

for module in model.modules():
    if hasattr(module, "gradient_checkpointing"):
        module._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
```

### Patch 4: `_convert_deprecated_attention_blocks` 누락

```python
# Before (broken - AttributeError)
model._convert_deprecated_attention_blocks()

# After (hasattr guard)
if hasattr(model, "_convert_deprecated_attention_blocks"):
    model._convert_deprecated_attention_blocks()
```

---

## 5. GPU & xformers Notes

### GPU 제약 (gpu03)

| GPU | 타입 | 지원 |
|-----|------|------|
| 0–3 | Blackwell | ❌ 미지원 (CUDA 호환 문제) |
| 4–7 | A6000 | ✅ 사용 가능 |

**반드시 `CUDA_VISIBLE_DEVICES=4,5,6,7` 중 선택하여 실행.**

### xformers Fallback

Blackwell GPU에서 xformers validation 실패 가능. try/except로 fallback 처리:

```python
try:
    from diffusers.models.attention_processor import XFormersAttnProcessor
    unet.set_attn_processor(XFormersAttnProcessor())
except Exception:
    # Fallback to default attention
    pass
```

---

## 6. Running

### 실행 스크립트

**파일**: `scripts/run_mvdiff_M5.sh`

```bash
cd /home/joon/dev/FaceLift

# 직접 실행
bash scripts/run_mvdiff_M5.sh

# nohup (백그라운드)
nohup bash scripts/run_mvdiff_M5.sh > logs/mvdiff_M5_finetune.log 2>&1 &
```

### 로그 모니터링

```bash
# 실시간 로그
tail -f logs/mvdiff_M5_finetune.log

# Loss 확인
grep -i "loss" logs/mvdiff_M5_finetune.log | tail -20

# GPU 사용량
nvidia-smi -l 5
```

---

## 7. WandB Monitoring

| 항목 | 값 |
|------|-----|
| Project | `FaceLift-Mouse` |
| Experiment | `mvdiff_M5_finetune` |

주요 모니터링 지표:
- `train/loss`: Diffusion loss (목표: 0.02–0.05)
- `val/loss`: Validation loss
- Learning rate schedule

---

## 8. Troubleshooting

### `AttributeError: _load_state_dict_into_model`

→ Patch 1 적용. diffusers 0.36.0에서 제거된 함수.

### `TypeError: load_state_dict() got unexpected keyword argument variant`

→ Patch 2 적용. `variant` 파라미터 제거.

### `AttributeError: _gradient_checkpointing_func`

→ Patch 3 적용. 수동으로 `torch.utils.checkpoint.checkpoint` 할당.

### `CUDA error on GPU 0-3`

→ Blackwell GPU 사용 불가. `CUDA_VISIBLE_DEVICES=4` 이상으로 설정.

### `xformers` 관련 에러

→ try/except fallback 확인. A6000에서는 정상 동작.

---

*FaceLift MVDiffusion Guide | 2026-01-28*
