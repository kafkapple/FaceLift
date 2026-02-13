# MVDIFFUSION_FINETUNE_GUIDE

> Single Image → Multi-View Diffusion → 3D Reconstruction 파이프라인 | 2026-01-30

---

## 1. Overview

```
Single Image ──→ MV-Diffusion (6 views) ──→ GS-LRM (3D Gaussians) ──→ Novel View Rendering
                  ↑ finetune 대상            ↑ pretrained (frozen)
```

**목적**: Mouse 단일 이미지에서 6개 뷰를 생성하고, GS-LRM으로 3D 재구성하는 파이프라인의 MV-Diffusion 단계를 finetune.

**핵심 설정**:
- 데이터: M5 전처리 (Affine, D7.1 preset, 512×512, fx=549, cx=cy=256)
- **M5t2 (80/10/10 temporal split)**: Train 2,880 / Val 360 / Test 360 ← **권장**
- **M5 (80/10/10 random split)**: Train 3,240 / Val 360
- **M5t (1:1:1 temporal split)**: Train 1,198 / Val 1,198 / Test 1,204 ← Pose Splatter 비교용
- Loss 범위: ~0.02–0.05

---

## 2. Prerequisites

### Conda 환경

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate facelift
```

### Pretrained Weights

MV-Diffusion pretrained checkpoint가 `checkpoints/` 디렉토리에 필요. Config에서 `pretrained_model_name_or_path`로 참조.

---

## 3. Config

### M5 Config (기본)

**파일**: `configs/mvdiffusion/mouse_mvdiffusion_M5.yaml`

| 항목 | 값 | 설명 |
|------|-----|------|
| `resolution` | 512 | 입출력 해상도 |
| `num_views` | 6 | 생성 뷰 수 |
| `fx` | 549 | Focal length (pretrained 호환) |
| `cx, cy` | 256 | Principal point |
| `train_samples` | 3,240 | 학습 데이터 수 (80%) |
| `val_samples` | 360 | 검증 데이터 수 (10%) |

### M5t2 Config (권장)

**파일**: `configs/mvdiffusion/mouse_mvdiffusion_M5t2.yaml`

| 항목 | 값 | 설명 |
|------|-----|------|
| `split` | **8:1:1 temporal** | 최대 학습 데이터 + temporal leakage 방지 |
| `train_samples` | **2,880** | 시간순 80% |
| `val_samples` | 360 | 시간순 10% |
| `test_samples` | 360 | 시간순 10% |

**데이터 파일**: `data_mouse_t2_{train,val,test}.txt`

> M5t2는 현재 모든 MV-Diffusion ablation 실험의 기본 데이터셋.
> M5t (1:1:1) 대비 학습 데이터 2.4배 → MV-Diffusion 병목 해소 (H3 진단 결과).

### M5t Config (Pose Splatter 비교용)

**파일**: `configs/mvdiffusion/mouse_mvdiffusion_M5t.yaml`

| 항목 | 값 | 설명 |
|------|-----|------|
| `split` | 1:1:1 temporal | Pose Splatter 논문과 동일 |
| `train_samples` | 1,198 | 시간순 첫 1/3 |
| `val_samples` | 1,198 | 시간순 중간 1/3 |
| `test_samples` | 1,204 | 시간순 마지막 1/3 |

**데이터 파일**: `data_mouse_1to1_*.txt`

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

*FaceLift MV-Diffusion Guide | 2026-01-28*

## Prompt Embeddings Design (from archive)

# MV-Diffusion Prompt Embeddings 설계 가이드

## 개요

MVDiffusion은 **prompt embeddings**를 사용해서 각 출력 뷰의 방향을 제어합니다.
이 문서는 prompt embeddings의 역할, 구조, 생성 방법을 설명합니다.

---

## 1. Prompt Embeddings 역할

```
┌─────────────────────────────────────────────────────────────────────┐
│  MV-Diffusion 추론 과정                                               │
│                                                                     │
│  입력: [같은 이미지 × 6] + [다른 prompt_embeds × 6]                   │
│         └─ reference view    └─ 각 뷰 방향 정보                      │
│                                                                     │
│  UNet 내부:                                                          │
│  - Cross-attention에서 prompt_embeds 사용                            │
│  - 각 위치(0~5)마다 다른 prompt_embeds → 다른 뷰 생성                 │
│                                                                     │
│  출력: [다른 뷰 × 6]                                                  │
│         └─ 각각 다른 카메라 방향에서 본 이미지                        │
└─────────────────────────────────────────────────────────────────────┘
```

**핵심**: prompt_embeds가 "이 위치에서는 어떤 방향의 뷰를 생성해라"라는 신호를 제공

---

## 2. Prompt Embeddings 구조

### 파일 구조
```
mvdiffusion/data/fixed_prompt_embeds_6view/  # FaceLift 기본
├── clr_embeds.pt      # Color map용 embeddings
└── normal_embeds.pt   # Normal map용 embeddings

mvdiffusion/data/mouse_prompt_embeds_6view/  # Mouse 카메라용
├── clr_embeds.pt
├── normal_embeds.pt
└── metadata.json      # 설정 정보
```

### Tensor Shape
```python
prompt_embeds.shape = [6, 77, 1024]
#                      │   │    │
#                      │   │    └─ CLIP embedding dimension
#                      │   └─ Token sequence length (CLIP max)
#                      └─ Number of views
```

### Dtype
- `torch.float16` (half precision)

---

## 3. 텍스트 프롬프트 설계

### FaceLift 기본 (수평 6방향, elevation 0°)
```python
views = ["front", "front_right", "right", "back", "left", "front_left"]

color_prompts = [
    "a rendering image of 3D models, front view, color map.",
    "a rendering image of 3D models, front_right view, color map.",
    "a rendering image of 3D models, right view, color map.",
    "a rendering image of 3D models, back view, color map.",
    "a rendering image of 3D models, left view, color map.",
    "a rendering image of 3D models, front_left view, color map.",
]
```

### Mouse 카메라용 (경사 6방향, elevation ~20°)
```python
views = ["top-front", "top-front-right", "top-right",
         "top-back", "top-left", "top-front-left"]

color_prompts = [
    "a rendering image of a 3D model, top-front view, from above at an angle, color map.",
    "a rendering image of a 3D model, top-front-right view, from above at an angle, color map.",
    # ...
]
```

### 프롬프트 설계 원칙
1. **방향 명시**: front, back, left, right 등 방향 키워드 포함
2. **elevation 표현**: top-, from above 등으로 경사 표현
3. **일관성**: 모든 뷰에 동일한 구조의 문장 사용
4. **짧고 명확**: CLIP이 잘 이해할 수 있는 간결한 문장

---

## 4. Prompt Embeddings 생성 방법

### 생성 스크립트
```bash
# FaceLift 기본 (수평)
python mvdiffusion/data/generate_fixed_text_embeds.py

# Mouse 카메라용 (경사)
python scripts/generate_mouse_prompt_embeds.py \
    --output_dir mvdiffusion/data/mouse_prompt_embeds_6view \
    --elevation 20
```

### 생성 과정
```python
# 1. CLIP 모델 로드
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")

# 2. 텍스트 토큰화
text_inputs = tokenizer(prompts, padding="max_length", max_length=77, ...)

# 3. CLIP 인코딩
with torch.no_grad():
    prompt_embeds = text_encoder(text_inputs.input_ids)
    prompt_embeds = prompt_embeds[0]  # [6, 77, 1024]

# 4. 저장
torch.save(prompt_embeds.half(), "clr_embeds.pt")
```

---

## 5. Config에서 설정

### mouse_dataset.py에서 로딩
```python
# config에서 경로 지정 가능
prompt_embed_path = config.get(
    "prompt_embed_path",
    "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt"  # 기본값
)
self.color_prompt_embedding = torch.load(prompt_embed_path)
```

### YAML config 예시
```yaml
# FaceLift 카메라용
prompt_embed_path: 'mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt'

# Mouse 카메라용
prompt_embed_path: 'mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt'
```

---

## 6. 실험별 Prompt Embeddings

| 실험 | Prompt Embeds | 카메라 설정 | 설명 |
|------|---------------|-------------|------|
| FaceLift 기본 | fixed_prompt_embeds_6view | 수평 6방향 | Human face용 |
| Option A | fixed_prompt_embeds_6view | 수평 6방향 | Mouse → FaceLift 카메라 변환 |
| Option B | mouse_prompt_embeds_6view | 경사 6방향 | Mouse 원본 카메라 |

---

## 7. 주의사항

1. **학습-추론 일치**: 학습 시 사용한 prompt_embeds와 추론 시 사용하는 것이 같아야 함
2. **카메라 매칭**: prompt_embeds의 뷰 방향과 실제 데이터 카메라가 일치해야 함
3. **GS-LRM 연계**: MV-Diffusion 출력 뷰의 카메라 ≈ GS-LRM에 전달하는 카메라
