# Mouse Quick Reference

> 매일 사용하는 명령어, 데이터셋, 실험 설정 빠른 참조
> Last updated: 2026-01-28

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
    --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse_M5t/checkpoint-3500 \
    --gslrm_checkpoint checkpoints/gslrm/M5t_E0_1_facelift \
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
