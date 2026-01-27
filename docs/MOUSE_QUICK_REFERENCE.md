
---

## Frame Context Inspection Tool

프레임 주변 컨텍스트를 슬로우모션으로 추출하여 검사하는 도구.

### 위치
`mouse_extensions/scripts/extract_frame_context.py`

### 사용법
```bash
# 기본: 3초 전후, 0.25x 속도
python extract_frame_context.py --video input.mp4 --frames 5900

# 슬로우모션 (10x 느리게)
python extract_frame_context.py --video input.mp4 --frames 5900,11800 --speed 0.1

# 윈도우 확장 (5초 전후)
python extract_frame_context.py --video input.mp4 --frames 5900 --window 5.0
```

### 파라미터
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--video, -v` | 입력 비디오 | (필수) |
| `--frames, -f` | 타겟 프레임 (콤마 구분) | (필수) |
| `--window, -w` | 전후 윈도우 (초) | 3.0 |
| `--speed, -s` | 재생 속도 배율 | 0.25 |
| `--output, -o` | 출력 디렉토리 | 비디오 위치 |

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

## 실험 명령어 Quick Reference

### 전처리 (Preprocessing)

```bash
cd /home/joon/dev/FaceLift

# 권장: M3_2
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_2 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_2

# H2 실험용
python -m mouse_extensions.preprocessing.preprocess --preset M3_2b ...
python -m mouse_extensions.preprocessing.preprocess --preset M3_3 ...

# H1 실험용
python -m mouse_extensions.preprocessing.preprocess --preset M4 ...
```

### 학습 (Training)

```bash
cd /home/joon/dev/FaceLift

# 기본 (권장)
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E1_2_alpha

# 가설 검증
CUDA_VISIBLE_DEVICES=1 torchrun ... -d M3_2b -e E1_2_alpha  # H2 baseline
CUDA_VISIBLE_DEVICES=2 torchrun ... -d M3_3 -e E1_2_alpha   # H2 test
CUDA_VISIBLE_DEVICES=3 torchrun ... -d M4 -e E1_2_alpha     # H1 test
```

### 검증 (Validation)

```bash
# PP/MVG 일관성 검증
python mouse_extensions/scripts/diagnostics/verify_pp_mvg_consistency.py \
    --datasets M3_2 --verbose

# 클리핑 분석
python mouse_extensions/scripts/analysis/clipping_analyzer.py \
    /home/joon/data/preprocessed/FaceLift_mouse/M3_2 -r 50 -m 10
```

---

## 데이터셋 요약

| Preset | PP | fx | zoom_range | 용도 |
|--------|-----|-----|------------|------|
| D7.1 (M1) | 256 | 549 | - | 기준선 |
| D8 (M2) | 256 | 549 | - | 정밀 기준선 |
| **M3_2** | 256 | 549 | [1.0,1.8] | ⭐ **권장** |
| M3_2b | 256 | 549 | [1.0,1.5] | H2 baseline |
| M3_3 | 256 | 549 | [1.0,2.5] | H2 test |
| M4 | 가변 | 549 | [1.0,2.5] | H1 test |

---

## 실험 설정 요약

| 설정 | mask_mode | alpha_loss | 용도 |
|------|-----------|------------|------|
| E0_1_facelift | none | 0.0 | 마스크 없음 |
| **E1_2_alpha** | gt | 0.1 | ⭐ **권장** |
| E1_3_lgm | gt | 1.0 | 강한 alpha |
| E2_1_alpha | none | 0.1 | alpha만 |


---

## MVDiffusion Finetune (Single Image → 6 Views)

### 목적
단일 이미지 → MVDiffusion (6 views) → GS-LRM (3D) 파이프라인 구축.
GS-LRM과 동일한 전처리 데이터(M5)로 finetune하여 카메라 파라미터 일관성 확보.

### Config
- **파일**: `configs/mvdiffusion/mouse_mvdiffusion_M5.yaml`
- **전처리**: M5 (Affine, D7.1 preset, 512x512, fx=549, cx=cy=256)
- **GS-LRM 기준**: M5_E1_2_alpha (Alpha IoU=0.8398)
- **데이터**: train 3239 / val 359 샘플

### 실행 명령어
```bash
cd /home/joon/dev/FaceLift
CUDA_VISIBLE_DEVICES=7 accelerate launch     --config_file mvdiffusion/node_config/1gpu.yaml     train_diffusion.py     --config configs/mvdiffusion/mouse_mvdiffusion_M5.yaml
```

### WandB
- Project: `mouse_facelift`
- Exp: `mvdiff_M5_finetune`
- Group: `mvdiffusion`
