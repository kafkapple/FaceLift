# Mouse Quick Reference

> 매일 사용하는 명령어, 데이터셋, 실험 설정 빠른 참조
> Last updated: 2026-01-31

---

## Raw Data: markerless_mouse_1_nerf

### 프레임 정보
- **총 프레임**: 18,000 (6개 카메라 동일)
- **FPS**: 100
- **해상도**: 1152×1024
- **frame_interval=5** → 3,600 샘플

### ⚠️ Frame Discontinuity (불연속 위치)
```
DISCONTINUITY_FRAMES = {5900, 11800, 17700}
```

**의미**: 해당 위치에서 원본 녹화의 프레임 불연속(갭) 발생
- 해당 프레임 자체는 **정상** (제외 불필요)
- temporal smoothness 가정하는 알고리즘 사용 시 주의
- 현재 전처리: **모든 프레임 사용** (제외 없음)

---

## 데이터셋 요약

| Preset | PP | fx | zoom_range | 용도 |
|--------|-----|-----|------------|------|
| D7.1 (M1) | 256 | 549 | - | 기준선 |
| D8 (M2) | 256 | 549 | - | 정밀 기준선 |
| **M3_2** | 256 | 549 | [1.0,1.8] | 이전 권장 |
| M3_2b | 256 | 549 | [1.0,1.5] | H2 baseline |
| M3_3 | 256 | 549 | [1.0,2.5] | H2 test |
| M4 | 가변 | 549 | [1.0,2.5] | H1 test |
| **M5** | 256 | 549 | - | ⭐ **현재 권장** (Affine, 512x512) |

---

## 실험 설정 요약

| 설정 | mask_mode | alpha_loss | 용도 |
|------|-----------|------------|------|
| E0_1_facelift | none | 0.0 | 마스크 없음 |
| **E1_2_alpha** | gt | 0.1 | ⭐ **권장** |
| E1_3_lgm | gt | 1.0 | 강한 alpha |
| E2_1_alpha | none | 0.1 | alpha만 |

---

## 전처리 명령어

```bash
cd /home/joon/dev/FaceLift

# 권장: M5
python -m mouse_extensions.preprocessing.preprocess \
    --preset M5 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M5

# 기타 프리셋
python -m mouse_extensions.preprocessing.preprocess --preset M3_2 ...
python -m mouse_extensions.preprocessing.preprocess --preset M4 ...
```

---

## 학습 명령어

```bash
cd /home/joon/dev/FaceLift

# Modular mode (권장) - M5 + E1_2_alpha
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5 -e E1_2_alpha

# Legacy mode
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/M5_E1_2_alpha.yaml
```

---

## WandB Auto-Resume (GS-LRM & MVDiffusion 공통)

### 동작 방식

**자동 저장/로드**: checkpoint 디렉토리에 `wandb_run_id.txt` 저장

```
첫 학습 시:
  wandb.init() → 새 run 생성 → wandb_run_id.txt 저장

Resume 시:
  wandb_run_id.txt 확인 → run_id 로드 → 이전 run 이어서
```

### 저장 위치

| 모델 | 위치 |
|------|------|
| GS-LRM | `{checkpoint_dir}/wandb_run_id.txt` |
| MVDiffusion | `{checkpoint_prefix}/{output_dir}/wandb_run_id.txt` |

### Resume 명령어 (동일)

**GS-LRM**:
```bash
# 첫 실행이든 resume이든 동일한 명령어
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e E0_1_facelift \
    > logs/M5t2_E0_1.log 2>&1 &
```

**MVDiffusion**:
```bash
# 첫 실행이든 resume이든 동일한 명령어
export CUDA_VISIBLE_DEVICES=7 && nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t2.yaml \
    > logs/mvdiff_M5t2.log 2>&1 &
```

### 수동 run_id 지정 (이전 실험용)

`wandb_run_id.txt`가 없는 이전 실험은 수동 생성:

```bash
# 1. WandB UI에서 run_id 확인
# https://wandb.ai/joon/FaceLift-Mouse → run 클릭 → Overview → Run ID (8자리)

# 2. 수동 저장
echo "yz9fl25q" > /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/wandb_run_id.txt

# 3. 일반 명령어로 resume
CUDA_VISIBLE_DEVICES=5 nohup torchrun ... train_gslrm.py -d M5t2 -e E0_1_facelift ...
```

### 현재 상태

| Model | Dataset | wandb_run_id.txt | 상태 |
|-------|---------|------------------|------|
| GS-LRM | M5t2 | ✅ `yz9fl25q` | resume 가능 |
| MVDiffusion | M5t2 | ❌ 없음 | 새 run으로 시작 (checkpoint는 resume) |

---

## 검증 명령어

```bash
# PP/MVG 일관성 검증
python mouse_extensions/scripts/diagnostics/verify_pp_mvg_consistency.py \
    --datasets M5 --verbose

# 클리핑 분석
python mouse_extensions/scripts/analysis/clipping_analyzer.py \
    /home/joon/data/preprocessed/FaceLift_mouse/M5 -r 50 -m 10
```

---
export CUDA_VISIBLE_DEVICES=7 && nohup accelerate launch \
--config_file configs/accelerate/1gpu.yaml \
train_diffusion.py \
--config configs/mvdiffusion/mouse_mvdiffusion_M5t2.yaml \
> logs/mvdiff_M5t2.log 2>&1 &
## MVDiffusion Finetune (Single Image → 6 Views)

### 목적
단일 이미지 → MVDiffusion (6 views) → GS-LRM (3D) 파이프라인 구축.
GS-LRM과 동일한 전처리 데이터(M5)로 finetune하여 카메라 파라미터 일관성 확보.

### Config 선택

| Config | Split | Train/Val/Test | 용도 |
|--------|-------|---------------|------|
| mouse_mvdiffusion_M5.yaml | 80/10/10 random | 2880/360/360 | 일반 학습 |
| mouse_mvdiffusion_M5t.yaml | 1:1:1 temporal | 1198/1198/1204 | **Pose Splatter 비교** |

### M5 (기본 - Random Split)
- **파일**: configs/mvdiffusion/mouse_mvdiffusion_M5.yaml
- **전처리**: M5 (Affine, D7.1 preset, 512x512, fx=549, cx=cy=256)
- **데이터**: train 2880 / val 360 샘플 (80/10/10)

### M5t (Pose Splatter 비교용 - 1:1:1 Temporal Split)
- **파일**: configs/mvdiffusion/mouse_mvdiffusion_M5t.yaml
- **전처리**: 동일 (M5 데이터 공유)
- **데이터**: train 1198 / val 1198 / test 1204 (*_1to1.txt 파일)
- **특징**: Pose Splatter 논문과 동일한 1:1:1 temporal consecutive split

### 실행 명령어

```bash
cd /home/joon/dev/FaceLift

# M5t (Pose Splatter 비교) - Background with log
CUDA_VISIBLE_DEVICES=4 nohup accelerate launch \
    --mixed_precision=fp16 \
    train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t.yaml \
    > logs/mvdiff_M5t.log 2>&1 &

# 로그 확인
tail -f logs/mvdiff_M5t.log
```

### WandB
- Project: FaceLift-Mouse
- Group: mvdiffusion

---

*Quick Reference | 2026-01-28*


---

## Quick Start Commands (merged)

> Merged from QUICK_START.md (v7.0, 2026-01-25)

### Background 실행

```bash
cd /home/joon/dev/FaceLift
conda activate facelift

CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E0_1_facelift > logs/M3_2_E0_1.log 2>&1 &
```

### GPU 사용 (gpu03)

| GPU | Architecture | Available |
|-----|-------------|-----------|
| 0-3 | Blackwell (sm_120) | X (PyTorch 미지원) |
| 4-7 | A6000 (sm_86) | O |

#### ⚠️ facelift 환경 GPU 선택 주의

**문제**: `conda activate facelift` 시 `env_vars.sh`가 `CUDA_VISIBLE_DEVICES=4` 자동 설정
→ 사용자가 다른 GPU 지정해도 무시됨

**해결**: `export`로 먼저 설정 (conda activate 전에 환경변수 존재해야 함)

```bash
# 올바른 방법 (GPU 6 사용)
export CUDA_VISIBLE_DEVICES=6 && nohup accelerate launch ...

# 잘못된 방법 (GPU 4로 덮어씌워짐)
CUDA_VISIBLE_DEVICES=6 nohup accelerate launch ...
```

#### GPU 상태 확인

```bash
nvidia-smi
ps aux | grep train_gslrm
ps aux | grep train_diffusion
```

### 폐기 데이터셋

| ID | Issue | Ray Error |
|----|-------|-----------|
| ~~M3~~ | fx=739 unnormalized | 6.96 deg |
| ~~M3_norm~~ | Variable PP | 13.62 deg |
| ~~M3_persample~~ | Variable PP | 16.15 deg |


---

## E2E Inference (MVDiffusion + GS-LRM)

### Unified Pipeline

```bash
cd /home/joon/dev/FaceLift

# Single image -> 6 views -> 3D Gaussians
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.inference.run \
    --config configs/inference/m5_1view.yaml \
    --image /path/to/input.png

# Batch processing (1-view mode)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.inference.run \
    --config configs/inference/m5_1view.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --start_frame 0 --end_frame 100 --frame_step 5 \
    --view_idx 0
```

### Camera Parameters Warning

MVDiffusion output -> GS-LRM input requires M5 camera parameters.
Use MVDiffusionInference.compute_cameras(). Never use identity matrices.

---

*Updated: 2026-01-29 | Added E2E inference section*


---

## Inference: Wild Image → 3D (신규 이미지 추론)

### 개요

학습된 모델로 **임의의 생쥐 이미지**를 3D로 복원하는 파이프라인.

```
Wild Image → [전처리] → [MVDiffusion] → 6 Views → [GS-LRM] → 3D Gaussians
```

### 전처리 파이프라인

| 단계 | 처리 | 목적 |
|------|------|------|
| 1. SAM Segmentation | 생쥐 영역 검출 | 배경 제거 |
| 2. Background Removal | 흰색 배경 합성 | M5 형식 일치 |
| 3. Center Alignment | centroid → (256,256) | 위치 정규화 |
| 4. Coverage Normalization | scale to ~6% | M5 학습 통계 일치 |
| 5. Resize | 512×512 | 입력 해상도 |

### E2E Inference 명령어

```bash
cd /home/joon/dev/FaceLift

# SAM 전처리 포함 (자동 segmentation)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --input_image /path/to/wild_mouse.jpg \
    --sam_checkpoint checkpoints/sam/sam_vit_b.pth \
    --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse_M5t/checkpoint-4000 \
    --gslrm_checkpoint checkpoints/gslrm/M5t_E1_2_alpha/best_psnr.pt \
    --output_dir outputs/inference_test

# 전처리 단계별 시각화 저장
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --input_image /path/to/wild_mouse.jpg \
    --sam_checkpoint checkpoints/sam/sam_vit_b.pth \
    --save_preprocess_steps \
    ...

# 전처리 건너뛰기 (이미 M5 형식인 경우)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --input_image /path/to/already_preprocessed.png \
    --skip_preprocess \
    ...
```


### 기존 마스크 활용 (수동 전처리)

RGB 이미지와 마스크가 별도로 있는 경우, 배경 제거 후 추론:

```bash
cd /home/joon/dev/FaceLift

# 배경 제거: 마스크 영역 외 흰색으로
python -c "
import cv2, numpy as np
img = cv2.imread(input.png)
mask = cv2.imread(mask.png, 0) / 255.0
white = np.ones_like(img) * 255
result = (img * mask[...,None] + white * (1-mask[...,None])).astype(np.uint8)
cv2.imwrite(output.png, result)
"

# 전처리된 이미지로 추론
export CUDA_VISIBLE_DEVICES=4 && python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --input_image output.png --skip_preprocess \
    --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse_M5t/checkpoint-4000 \
    --gslrm_checkpoint checkpoints/gslrm/M5t_E1_2_alpha/best_psnr.pt \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/e2e_output
```

### 자동 감지

이미 전처리된 이미지(512×512, 흰색 배경, 중앙 물체)는 자동 감지되어 전처리 건너뜀.

---

## Manual Segmentation GUI (수동 어노테이션)

### 목적

SAM 자동 segmentation이 실패하는 경우(엉뚱한 물체 선택), 
사용자가 직접 **점을 클릭**하여 생쥐 영역을 지정.

### 실행

```bash
cd /home/joon/dev/FaceLift

# 로컬 네트워크만 (gpu03 내부)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.segment_mouse_web \
    --sam_checkpoint checkpoints/sam/sam_vit_b.pth \
    --input_dir /path/to/images \
    --output_dir /path/to/output \
    --port 7860

# 외부 접속 (Public URL 생성) ⭐
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.segment_mouse_web \
    --sam_checkpoint checkpoints/sam/sam_vit_b.pth \
    --input_dir /path/to/images \
    --output_dir /path/to/output \
    --port 7860 \
    --share
```

### 접속 방법

| 방법 | URL | 조건 |
|------|-----|------|
| **Gradio Share** | `https://xxxxx.gradio.live` | `--share` 옵션 필요, 1주 유효 |
| SSH 터널 | `http://localhost:7860` | `ssh -L 7860:localhost:7860 gpu03` 후 접속 |
| 내부 접속 | `http://gpu03:7860` | 같은 네트워크만 |

### Gradio Share 원리

```
로컬 서버 (gpu03:7860) ←→ Gradio Cloud 터널 ←→ Public URL (*.gradio.live)
```

- 별도 포트포워딩/방화벽 설정 불필요
- 데이터는 터널 통과만 (저장 안됨)
- 1주 후 만료 (재시작 시 새 URL)

### GUI 사용법

| 단계 | 동작 |
|------|------|
| 1 | **Mode 선택**: Foreground (Green) / Background (Red) |
| 2 | **이미지 클릭**: 녹색=생쥐, 빨간색=배경 |
| 3 | **Generate Mask**: 클릭 점 기반 SAM 마스크 생성 |
| 4 | **Save & Next**: 512×512 흰배경 이미지 + 마스크 저장 |

### 버튼 설명

| 버튼 | 기능 |
|------|------|
| ⬅️ Prev | 이전 이미지 |
| Skip ➡️ | 현재 이미지 건너뛰기 |
| 🎭 Generate Mask | SAM 마스크 생성 |
| ↩️ Undo | 마지막 점 취소 |
| 🔄 Reset | 모든 점 초기화 |
| 💾 Save & Next | 저장 후 다음 이미지 |

### 출력 파일

```
output_dir/
├── image_001.png          # 512×512 흰배경 이미지
├── image_001_mask.png     # 이진 마스크
├── image_002.png
├── image_002_mask.png
└── ...
```

### Checkpoint 다운로드

```bash
# SAM ViT-B (358MB, 권장)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
    -O checkpoints/sam/sam_vit_b.pth

# SAM ViT-H (2.4GB, 고정밀)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -O checkpoints/sam/sam_vit_h.pth
```

---

*Updated: 2026-01-30 | Added Inference & Manual Segmentation*


---

## MVDiffusion Resume 학습 (이어서 학습)

### Config 체계 (2개)

| Config | 용도 | 핵심 설정 |
|--------|------|-----------|
| mouse_mvdiffusion_M5t.yaml | **새로 시작** | resume: null, steps: 10000 |
| mouse_mvdiffusion_M5t_resume.yaml | **이어서 학습** | resume: latest, wandb_run_id, steps: 20000 |

### 왜 Config 분리?

- resume_from_checkpoint: latest 남아있으면 → 새 학습 시 의도치 않게 resume
- wandb_run_id 남아있으면 → 새 실험 로그가 이전 run에 오염

### Resume 실행

```bash
cd /home/joon/dev/FaceLift

# 이어서 학습 (checkpoint-5000 → 20000)
CUDA_VISIBLE_DEVICES=4 nohup accelerate launch \
    --mixed_precision=fp16 \
    train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t_resume.yaml \
    > logs/mvdiff_M5t_v2.log 2>&1 &
```

### WandB Run ID 확인

```bash
# 로그에서 확인
grep "Run ID" logs/mvdiff_M5t.log

# wandb 폴더에서 확인
ls wandb/ | grep run-
# → run-20260130_115712-f2gmqhy3  (f2gmqhy3가 run_id)
```

### Checkpoint 설정 권장

| 파라미터 | 권장값 | 이유 |
|----------|--------|------|
| checkpointing_steps | 2000 | 디스크 절약 |
| checkpoints_total_limit | 2 | 최근 2개만 유지 |
| validation_steps | 200 | 적절한 모니터링 빈도 |
| use_ema | true | EMA 모델이 더 안정적 |

### Diffusion 학습 특성

**Early Stopping 불필요:**
- Diffusion 모델은 학습이 매우 안정적 (GAN과 달리 mode collapse 없음)
- Validation loss가 noisy해서 best 판단 어려움
- 고정 step + EMA + 마지막 checkpoint 사용이 일반적

**Best Model 저장 불필요:**
- PSNR/LPIPS가 fluctuate → best 판단 부정확
- EMA가 smoothing 역할 (이미 적용됨)
- 마지막 checkpoint 사용 권장

---

*Updated: 2026-02-01 | Added Resume training section*


---

## MVDiffusion 생쥐 전용 프롬프트 (Mouse-Specific Prompts)

### 배경

기존 MVDiffusion은 사람/일반 3D 모델용 프롬프트 사용:
- 뷰: front, front_right, right, back, left, front_left (수평 뷰)
- 프롬프트: "a rendering image of 3D models, {view} view, color map."

생쥐 데이터는 **위에서 비스듬히** 촬영하므로 뷰 방향 불일치 문제.

### 생쥐 전용 프롬프트 (권장)

| 항목 | 값 |
|------|-----|
| 경로 | `mvdiffusion/data/mouse_prompt_embeds_6view_1024/clr_embeds.pt` |
| Shape | [6, 77, 1024] (SD 2.1 unclip 호환) |
| 뷰 | top-front, top-front-right, top-right, top-back, top-left, top-front-left |

**프롬프트 내용:**
```
View 0: "a rendering image of a 3D model, top-front view, from above at an angle, color map."
View 1: "a rendering image of a 3D model, top-front-right view, from above at an angle, color map."
View 2: "a rendering image of a 3D model, top-right view, from above at an angle, color map."
View 3: "a rendering image of a 3D model, top-back view, from above at an angle, color map."
View 4: "a rendering image of a 3D model, top-left view, from above at an angle, color map."
View 5: "a rendering image of a 3D model, top-front-left view, from above at an angle, color map."
```

### Config 체계 (프롬프트별)

| Config | Prompt | Output Dir | WandB | 용도 |
|--------|--------|------------|-------|------|
| mouse_mvdiffusion_M5t.yaml | 일반 (front/back) | mouse_M5t | mvdiff_M5t_finetune | 기존 비교용 |
| **mouse_mvdiffusion_M5t_mouse_prompt.yaml** | **생쥐 (top-*)** | **mouse_M5t_mp** | **mvdiff_M5t_mouse_prompt** | ⭐ 권장 |
| mouse_mvdiffusion_M5t_mouse_prompt_resume.yaml | 생쥐 (top-*) | mouse_M5t_mp | (이어붙이기) | resume용 |

### 실행 명령어 (권장)

```bash
cd /home/joon/dev/FaceLift

# 생쥐 전용 프롬프트로 학습 (권장)
CUDA_VISIBLE_DEVICES=4 nohup accelerate launch \
    --mixed_precision=fp16 \
    train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t_mouse_prompt.yaml \
    > logs/mvdiff_M5t_mouse_prompt.log 2>&1 &

# 이어서 학습
CUDA_VISIBLE_DEVICES=4 nohup accelerate launch \
    --mixed_precision=fp16 \
    train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t_mouse_prompt_resume.yaml \
    > logs/mvdiff_M5t_mouse_prompt.log 2>&1 &
```

### 문헌 근거

MVDiffusion은 text prompt를 view control에 직접 사용하지 않음:
- Text prompt: "무엇을 생성할지" (scene description)
- View control: Correspondence-Aware Attention (depth 기반)

그러나 prompt embedding이 cross-attention을 통해 생성에 영향을 주므로,
실제 카메라 배치와 일치하는 프롬프트가 더 적절함.

### CLIP Embedding Dimension

| SD 버전 | Text Encoder | Dim | 호환 |
|---------|--------------|-----|------|
| SD 1.5 | CLIP ViT-L/14 | 768 | X |
| **SD 2.x** | **CLIP ViT-H/14** | **1024** | **O** |

MVDiffusion은 SD 2.1 unclip 기반 → 1024-dim 필수

---

*Updated: 2026-02-01 | Added mouse-specific prompt section*


---

## MVDiffusion Prompt Ablation 실험 (Rendering vs Real vs Hybrid)

### 실험 목적

생쥐 데이터는 **실제 영상**이지만 현재 프롬프트는 "3D model rendering"으로 설정.
최적의 프롬프트 스타일 탐색을 위한 ablation 실험.

### 3가지 프롬프트 스타일

| Style | 설명 | 예시 |
|-------|------|------|
| **Rendering** (현재) | 3D 렌더링 이미지 표현 | "a rendering image of a 3D model, top-front view..." |
| **Real** | 실제 실험실 촬영 표현 | "a laboratory mouse photographed from top-front angle..." |
| **Hybrid** | 최소 설명 + 기술 용어 | "a mouse, top-front view, multi-view capture, color map." |

### 프롬프트 상세

#### Rendering (rendering) - 현재 권장
```
View 0: "a rendering image of a 3D model, top-front view, from above at an angle, color map."
View 1: "a rendering image of a 3D model, top-front-right view, from above at an angle, color map."
...
```
- 경로: `mvdiffusion/data/mouse_prompt_embeds_6view_1024/clr_embeds.pt`

#### Real Image (real)
```
View 0: "a laboratory mouse photographed from top-front angle, multi-camera capture, white background."
View 1: "a laboratory mouse photographed from top-front-right angle, multi-camera capture, white background."
...
```
- 경로: `mvdiffusion/data/mouse_prompt_embeds_6view_real/clr_embeds.pt`

#### Hybrid (hybrid)
```
View 0: "a mouse, top-front view, multi-view capture, color map."
View 1: "a mouse, top-front-right view, multi-view capture, color map."
...
```
- 경로: `mvdiffusion/data/mouse_prompt_embeds_6view_hybrid/clr_embeds.pt`

### Config 파일

| Style | Config | Output Dir | WandB |
|-------|--------|------------|-------|
| Rendering | mouse_mvdiffusion_M5t_mouse_prompt.yaml | mouse_M5t_mp | mvdiff_M5t_mouse_prompt |
| Real | mouse_mvdiffusion_M5t_real_prompt.yaml | mouse_M5t_real | mvdiff_M5t_real_prompt |
| Hybrid | mouse_mvdiffusion_M5t_hybrid_prompt.yaml | mouse_M5t_hybrid | mvdiff_M5t_hybrid_prompt |

### 실행 명령어

```bash
cd /home/joon/dev/FaceLift

# Rendering (현재 진행중)
export CUDA_VISIBLE_DEVICES=4 && nohup accelerate launch \
    --mixed_precision=fp16 train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t_mouse_prompt.yaml \
    > logs/mvdiff_M5t_mouse_prompt.log 2>&1 &

# Real Image Style
export CUDA_VISIBLE_DEVICES=5 && nohup accelerate launch \
    --mixed_precision=fp16 train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t_real_prompt.yaml \
    > logs/mvdiff_M5t_real_prompt.log 2>&1 &

# Hybrid Style
export CUDA_VISIBLE_DEVICES=6 && nohup accelerate launch \
    --mixed_precision=fp16 train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t_hybrid_prompt.yaml \
    > logs/mvdiff_M5t_hybrid_prompt.log 2>&1 &
```

### 비교 기준

| Metric | 측정 방법 |
|--------|----------|
| PSNR/SSIM | Validation set 정량 평가 |
| LPIPS | Perceptual 유사도 |
| Visual | 생성 품질 육안 확인 (WandB) |
| E2E | GS-LRM 연결 시 3D 품질 |

### 가설

| Style | 장점 | 단점 |
|-------|------|------|
| Rendering | MVDiffusion 원래 학습 분포와 일치 | 실제 데이터와 semantic gap |
| Real | 데이터 실제 특성 반영 | pretrained 분포와 mismatch |
| Hybrid | 최소 bias, 모델 자유도 높음 | 정보 부족할 수 있음 |

---

*Updated: 2026-02-01 | Added Prompt Ablation section*


---

## Temporal Sequence Inference (E2E Pipeline)

### Config 파일

| Config | 모드 | 설명 |
|--------|------|------|
| `m5t_gslrm_temporal.yaml` | GS-LRM only | 6-view → 3D (MVDiffusion 없음) |
| `m5t_e2e_temporal.yaml` | **E2E** | 1-view → MVDiffusion → 6-view → GS-LRM → 3D |

### 주요 설정

```yaml
input:
  data_dir: /home/joon/data/preprocessed/FaceLift_mouse/M5t
  view_idx: 0           # 입력 뷰 (0-5), null이면 6-view 모드
  frame_range:
    start: 0
    end: 200            # 처리할 프레임 범위
    step: 5             # 프레임 간격 (5 = 매 5번째 프레임)

output:
  fps: 6                # 영상 속도 (낮을수록 천천히)
  rerun: true           # RRD 3D 시각화 저장
```

### 실행 명령어

```bash
cd /home/joon/dev/FaceLift

# E2E (MVDiffusion + GS-LRM)
export CUDA_VISIBLE_DEVICES=7 && python -m mouse_extensions.inference.run \
    --config configs/inference/m5t_e2e_temporal.yaml

# GS-LRM only (6-view input)
export CUDA_VISIBLE_DEVICES=7 && python -m mouse_extensions.inference.run \
    --config configs/inference/m5t_gslrm_temporal.yaml
```

### Checkpoint 위치

| Model | Checkpoint | 용도 |
|-------|------------|------|
| MVDiffusion | `checkpoints/mvdiffusion/mouse_M5t/checkpoint-8000` | 일반 프롬프트 (학습 완료) |
| MVDiffusion | `checkpoints/mvdiffusion/mouse_M5t_mp/checkpoint-*` | 생쥐 프롬프트 (학습 중) |
| GS-LRM | `/node_data/joon/checkpoints/FaceLift/gslrm/M5t_E0_1_facelift/best_psnr.pt` | M5t 학습 완료 |

### FPS 가이드

| fps | 체감 속도 | 용도 |
|-----|----------|------|
| 24 | 빠름 | 일반 재생 |
| 12 | 보통 | 디테일 확인 |
| **6** | 천천히 | **권장** (프레임별 분석) |
| 3 | 매우 천천히 | 디버깅 |

### 출력 파일

```
outputs/m5t_e2e_temporal/
├── videos/
│   ├── turntable.mp4       # 360° 회전 영상
│   └── time_fixed.mp4      # 시간축 영상 (고정 각도)
├── rerun/
│   └── sequence.rrd        # 3D Gaussian 시각화 (Rerun)
└── gaussians/              # .ply, .npz 파일
```

### RRD 시각화 (Rerun)

```bash
# 로컬에서 확인
rerun outputs/m5t_e2e_temporal/rerun/sequence.rrd

# SSH 터널 (원격)
ssh -L 9090:localhost:9090 gpu03
rerun --web-viewer --port 9090 outputs/.../sequence.rrd
```

---

*Updated: 2026-02-01 | Added Temporal Sequence Inference*

---

## M5t2 데이터셋 (Temporal 80:10:10)

### 개요
- **목적**: Data leakage 방지 + Train 데이터 최대화
- **Split**: Temporal 80:10:10 (연속 프레임이 같은 split)
- **프롬프트**: Mouse prompt ()

### Split 구성
| Split | 샘플 수 | Frame 범위 |
|-------|--------|-----------|
| Train | 2880 | 0-2879 |
| Val | 360 | 2880-3239 |
| Test | 360 | 3240-3599 |

### 파일 위치
- Split 파일: `M5/data_mouse_t2_{train,val,test}.txt`
- Dataset config: `configs/datasets/M5t2.yaml`
- MVDiffusion config: `configs/mvdiffusion/mouse_mvdiffusion_M5t2.yaml`

### 학습 명령어

```bash
# MVDiffusion M5t2 (GPU 7)
export CUDA_VISIBLE_DEVICES=7 && nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t2.yaml \
    > logs/mvdiff_M5t2.log 2>&1 &

# GS-LRM M5t2 (GPU 4)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e E0_1_facelift > logs/M5t2_E0_1.log 2>&1 &
```

### MVDiffusion 뷰 일관성 개선 (M5t2_consistent)

6개 뷰 생성 시 일부 뷰가 불안정한 문제 해결을 위한 설정.

**변경 사항**:

| 설정 | 기존 (M5t2) | 개선 (M5t2_consistent) | 효과 |
|------|-------------|----------------------|------|
| `condition_drop_rate` | 0.05 | **0.0** | CFG dropout 제거 → 일관된 conditioning |
| `sparse_mv_attention` | true | **false** | 모든 뷰 쌍 attention → 멀리 떨어진 뷰도 일관성 |

**Trade-off**:
- ⬆️ VRAM 사용량, ⬇️ 출력 다양성, ⬆️ **뷰 일관성**

**실행**:
```bash
export CUDA_VISIBLE_DEVICES=7 && nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t2_consistent.yaml \
    > logs/mvdiff_M5t2_consistent.log 2>&1 &
```

---

## Pose-Splatter 비교 실험

### 데이터셋 비교
| 항목 | FaceLift M5t2 | Pose-Splatter fj5_ds2 |
|------|--------------|----------------------|
| 해상도 | 512×512 | 512×576 |
| Split | 80:10:10 temporal | 80:10:10 temporal |
| Train/Val/Test | 2880/360/360 | 2880/360/360 |
| Views | 6 (all) | 5 (holdout 1) |

### Pose-Splatter 실행
```bash
ssh joon
cd /home/joon/dev/pose-splatter
source ~/miniconda3/etc/profile.d/conda.sh && conda activate splatter
python mouse_extensions/scripts/run_comparison.py
```

---

## Slow Playback Temporal Videos (느린 재생 시퀀스)

### 목적
연속 프레임의 turntable 렌더링을 **느리고 부드럽게** 시각화.
- 프레임별 품질 분석
- 시간축 일관성 확인
- 발표/데모용 영상 생성

### 설정 파일

| Config | Checkpoint | Test Split | 학습량 |
|--------|------------|------------|-------|
| **m5t_temporal_slow.yaml** | M5t (11,800 step) | 1:1:1 test (1,204 프레임) | ✅ 충분 |
| m5t2_temporal_slow.yaml | M5t2 (400 step) | 80:10:10 test (360 프레임) | ⚠️ 초기 |

### 주요 파라미터

| 파라미터 | 기본값 | 느린재생 | 설명 |
|----------|--------|---------|------|
| --fps | 24 | **10** | 비디오 FPS (낮을수록 느림) |
| --rotation_speed | 0.5 | **0.3** | 회전 속도 (낮을수록 천천히) |
| --num_views | 36 | **60** | 360도 분할 (36=10도, 60=6도) |
| --end_frame | 전체 | 200 | 처리할 프레임 수 |
| --no_gaussian | - | **사용** | .ply/.npz 저장 끔 (rerun으로 대체) |
| --save_rerun | True | True | .rrd 저장 (프레임별 분석에 최적) |

**기타 기본값**: `--resolution 384`, `--elevation 20.0`, `--radius 2.7`, `--config configs/base/gslrm_mouse.yaml`

### 모델별 차이점

| 항목 | M5t (권장) | M5t2 |
|------|-----------|------|
| Checkpoint | `M5t_E0_1_facelift` | `M5t2_E0_1_facelift` |
| Split | `data_mouse_1to1_test.txt` | `data_mouse_t2_test.txt` |
| Test 프레임 | 1,204개 | 360개 |
| 학습 상태 | ✅ 11,800 step | ⚠️ 400 step (중단됨) |

### 실행 명령어 (통합)

```bash
cd /home/joon/dev/FaceLift

# MODEL: M5t 또는 M5t2
# SPLIT: 1to1 또는 t2
MODEL=M5t
SPLIT=1to1

export CUDA_VISIBLE_DEVICES=4 && \
source ~/anaconda3/etc/profile.d/conda.sh && conda activate facelift && \
nohup python -m mouse_extensions.scripts.inference.simple_temporal \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/${MODEL}_E0_1_facelift/best_psnr.pt \
    --config configs/base/gslrm_mouse.yaml \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_${SPLIT}_test.txt \
    --start_frame 0 --end_frame 200 \
    --fps 10 --rotation_speed 0.3 --num_views 60 \
    --no_gaussian \
    --output_dir outputs/temporal_${MODEL}_slow \
    > logs/temporal_${MODEL}_slow.log 2>&1 &

# 로그 확인
tail -f logs/temporal_${MODEL}_slow.log
```

**M5t2 사용시**: `MODEL=M5t2`, `SPLIT=t2`로 변경

### 옵션 변형

| 목적 | 추가 옵션 |
|------|----------|
| 비디오만 (용량 최소) | `--no_gaussian --no_rerun` |
| Gaussian 파일 필요 | `--save_gaussian` |
| 전체 프레임 | `--end_frame` 제거 또는 1204(M5t)/360(M5t2) |

### 속도 조절 가이드

| 용도 | fps | rotation_speed | num_views | 결과 |
|------|-----|----------------|-----------|------|
| 빠른 미리보기 | 24 | 0.5 | 36 | 기본 (일반 속도) |
| 발표용 | 15 | 0.4 | 48 | 약간 느림 |
| **분석용** | **10** | **0.3** | **60** | **권장 (느림)** |
| 디버깅 | 5 | 0.2 | 72 | 매우 느림 |

### 전체 프레임 처리

```bash
# M5t 전체 (1,204 프레임) - 약 2시간 소요
... --end_frame 1204 ...

# M5t2 전체 (360 프레임) - 약 30분 소요
# --end_frame 제거 또는 --end_frame 360
```

### 출력 구조

```
outputs/temporal_M5t_slow/
├── turntable.mp4           # 360도 회전 (느리게)
├── time_fixed_0.mp4        # 시간축 (각도 0도 고정)
├── time_rotating.mp4       # 시간+회전 동시 변화
├── grid_6view.mp4          # 입력 6뷰 그리드
├── rerun/                  # 기본 활성화 (프레임별 분석)
│   └── sequence.rrd        # Rerun 인터랙티브 뷰어
└── gaussians/              # --save_gaussian 시에만 (기본 비활성화)
    ├── frame_000000.ply    # 3D 뷰어용
    ├── frame_000000.npz    # 분석용
    └── ...
```

### Rerun 뷰어로 확인

```bash
# 로컬에서 직접
rerun outputs/temporal_M5t_slow/rerun/sequence.rrd

# SSH 터널 (원격 서버)
ssh -L 9090:localhost:9090 gpu03
rerun --web-viewer --port 9090 outputs/temporal_M5t_slow/rerun/sequence.rrd
# 브라우저: http://localhost:9090
```

---

*Updated: 2026-02-02*


---

## E2E Inference 체크포인트 현황 (260202)

### 사용 가능한 체크포인트

| Model | Dataset | Checkpoint | Steps | 상태 |
|-------|---------|------------|-------|------|
| **GS-LRM** | M5t | `/node_data/joon/.../M5t_E0_1_facelift/best_psnr.pt` | 11,800 | ✅ 권장 |
| **GS-LRM** | M5t2 | `/node_data/joon/.../M5t2_E0_1_facelift/best_psnr.pt` | 400+ | ✅ 사용가능 |
| **MVDiffusion** | M5t | `.../mvdiffusion/mouse_M5t/checkpoint-8000` | 8,000 | ✅ 권장 |
| **MVDiffusion** | M5t2 | `.../mvdiffusion/mouse_M5t2/checkpoint-5000` | 5,000 | ✅ 사용가능 |
| MVDiffusion | M5t2_consistent | `.../mouse_M5t2_consistent/` | 학습중 | 🔄 대기 |

### E2E 추론 명령어 (M5t 예시)

```bash
cd /home/joon/dev/FaceLift

# M5t E2E (MVDiffusion + GS-LRM)
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --start_frame 0 --end_frame 200 \
    --gslrm_checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t_E0_1_facelift/best_psnr.pt \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --mvdiffusion_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t/checkpoint-8000 \
    --prompt_embed_path mouse_prompt_embeds_6view_1024 \
    --prefer_ema \
    --skip_preprocess \
    --turntable_views 60 \
    --output_dir outputs/e2e_M5t \
    > logs/e2e_M5t.log 2>&1 &
```

**M5t2 사용시**: checkpoint 경로만 변경
- `M5t_E0_1_facelift` → `M5t2_E0_1_facelift`
- `mouse_M5t/checkpoint-8000` → `mouse_M5t2/checkpoint-5000`

---

## 비디오 출력 종류

### simple_temporal 출력

| 파일 | 설명 | 축 |
|------|------|-----|
| `turntable.mp4` | 각 프레임의 360도 회전 연결 | 회전 (시간 고정) |
| `time_fixed_0.mp4` | 고정 각도(0도)에서 시간 변화 | 시간 (회전 고정) |
| `time_rotating.mp4` | **시간+회전 동시 변화** | 시간 ↔ 회전 |
| `full_all.mp4` | 모든 시간 × 모든 각도 | T×V 전체 |
| `grid_first.png` | 첫 프레임 360도 그리드 | 정지 이미지 |

### time_rotating 계산 방식

```python
for t in range(T):           # 시간 프레임
    angle = (t * V // T) % V  # 시간에 비례해서 각도 증가
    frame = turntables[t][angle]
```
- **T**: 시간 프레임 수 (예: 200)
- **V**: 뷰 수 (예: 60)
- 효과: 시간이 지나면서 서서히 회전

---

## Slow Motion 수정 (260202)

### 문제
`--rotation_speed 0.5` 사용 시 **잔상(ghosting)** 발생

### 원인
`interpolate_frames_for_speed()` 함수가 프레임 간 **블렌딩** 수행:
```python
blended = (1-t) * frame[lower] + t * frame[upper]  # 두 프레임 혼합\!
```

### 수정
기본값을 **nearest neighbor**로 변경 (블렌딩 없음):
```python
def interpolate_frames_for_speed(..., interpolate: bool = False):
    if interpolate:
        # 블렌딩 (잔상 발생)
    else:
        # Nearest neighbor (잔상 없음) ← 기본값
        nearest = int(np.round(idx))
        new_frames.append(frames[nearest])
```

### 결과
- `--rotation_speed 0.3`: 프레임 반복으로 느린 재생 (잔상 없음)
- 블렌딩 원할 시: 코드에서 `interpolate=True` 명시 필요

---

*Updated: 2026-02-02 | Added E2E checkpoint status, video types, slow motion fix*
