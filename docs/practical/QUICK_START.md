# FaceLift Mouse Quick Start

> **Version**: v7.0 | **Updated**: 2026-01-25
> **목적**: 실험 실행에 필요한 최소 명령어
> **중요**: M3_norm, M3_persample은 PP 문제로 **폐기**. M3_1, M3_2 사용 필수!

---

## 1. 즉시 실행 명령어

### 권장 실험 (Production)

```bash
cd /home/joon/dev/FaceLift
conda activate facelift

# ⭐ M3_2 + E0_1_facelift (권장 - MVG-correct)
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E0_1_facelift

# M3_1 (Global zoom 버전)
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_1 -e E0_1_facelift

# D3_normalized (현재 최고 PSNR 27.09)
CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D3_normalized -e E0_1_facelift
```

### Background 실행

```bash
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E0_1_facelift > logs/M3_2_E0_1.log 2>&1 &
```

---

## 2. 데이터셋 (중요!)

### MVG-Correct 데이터셋 (사용 권장)

| ID | PP | Ray Error | Val PSNR | 설명 |
|----|-----|-----------|----------|------|
| D3_normalized | 256 | 0° | **27.09** | 최고 성능 |
| D7_1 | 256 | 0° | 20.93 | 안정적 기준선 |
| D8 | 256 | 0° | 20.21 | Homography |
| **M3_1** | 256 | 0° | TBD | Global zoom + MVG |
| **M3_2** | 256 | 0° | TBD | Per-sample zoom + MVG ⭐ |

### 폐기 데이터셋 (사용 금지!)

| ID | 문제점 | Ray Error |
|----|--------|-----------|
| ~~M3~~ | fx=739 미정규화 | 6.96° |
| ~~M3_norm~~ | PP 가변 | **13.62°** |
| ~~M3_persample~~ | PP 가변 | **16.15°** |

---

## 3. 전처리 명령어

```bash
# M3_2 전처리 (권장)
/home/joon/anaconda3/envs/facelift/bin/python \
    -m mouse_extensions.preprocessing.preprocess \
    --preset M3_2 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_2

/home/joon/anaconda3/envs/facelift/bin/python \
    -m mouse_extensions.preprocessing.preprocess \
    --preset M3_persample \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_persample

# PP 검증
/home/joon/anaconda3/envs/facelift/bin/python \
    mouse_extensions/scripts/diagnostics/verify_pp_mvg_consistency.py \
    --datasets M3_1,M3_2 --verbose
```

---

## 4. 실험 설정

| 실험 | mask_mode | alpha_loss | 현재 권장 |
|------|-----------|------------|-----------|
| **E0_1_facelift** | none | 0.0 | ⭐ **권장** |
| E0_2_mouse | none | 0.0 | 안정적 |
| E1_2_alpha | gt | 0.1 | 실험적 |

---

## 5. GPU 사용

| GPU | 아키텍처 | 사용 가능 |
|-----|----------|-----------|
| 0-3 | Blackwell | X PyTorch 미지원 |
| 4-7 | A6000 | O 사용 |

```bash
nvidia-smi
ps aux | grep train_gslrm
```

---

## 6. 핵심 문서

| 문서 | 위치 |
|------|------|
| M3 시리즈 상세 | docs/datasets/M3_SERIES_SPEC.md |
| PP 분석 | docs/analysis/PP_MVG_COMPREHENSIVE_ANALYSIS.md |
| MoC Index | docs/00_MoC_INDEX.md |

---

*Quick Start v7.0 | 2026-01-25*
