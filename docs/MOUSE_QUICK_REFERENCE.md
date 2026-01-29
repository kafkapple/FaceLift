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
| 0-3 | Blackwell | X (PyTorch unsupported) |
| 4-7 | A6000 | O |

```bash
nvidia-smi
ps aux | grep train_gslrm
```

### 폐기 데이터셋 (사용 금지)

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
