# Rat Fine-Tuning Data Preparation Guide

> **Navigation**: [<- INDEX](../INDEX.md) | [SDANNCE_VIDEO_AVAILABILITY](../datasets/SDANNCE_VIDEO_AVAILABILITY.md)
> **Version**: v1.0 | **Created**: 2026-03-22 | **Status**: ACTIVE

---

## 1. Overview

Mouse GS-LRM (α=0.3, PSNR 24.85)을 rat 도메인에 fine-tuning하기 위한 데이터 준비 가이드.

### Pipeline

```
[1] Frame Selection (DANNCE keypoints 기반 다양한 pose 선택)
     ↓
[2] SAM2 Annotation (keypoint-guided segmentation, per-camera)
     ↓
[3] SAM2 Propagation (annotation 간 프레임 보간)
     ↓
[4] GS-LRM Format 변환 (sdannce_to_gslrm.py --sam2_mask_dir)
     ↓
[5] Fine-tuning (α=0.3 checkpoint, low LR)
```

---

## 2. Data Location (gpu03)

### Source Data

| 항목 | 경로 |
|------|------|
| **세션 디렉토리** | `/home/joon/data/sdannce/rat/dataverse/SCN2A_WK1_2022_09_16_M1/` |
| **비디오** | `.../videos/Camera{1-6}/0.mp4` (6개 카메라, 1200×1920, 30fps) |
| **3D Keypoints** | `.../SDANNCE/bsl0.5_FM/save_data_AVG0.mat` (90,000 frames × 3 × 23 joints) |
| **Calibration** | `.../calibration/` (6 cameras, DANNCE MATLAB convention) |
| **기존 SAM2 마스크** | `.../sam2_masks/Camera{1-6}/mask_NNNNNN.npz` (20 frames × 6 cams = 120) |

### Output Paths

| 항목 | 경로 |
|------|------|
| **SAM2 마스크 (확장)** | `.../sam2_masks/Camera{1-6}/mask_NNNNNN.npz` (기존 위치에 추가) |
| **GS-LRM 포맷** | `outputs/sdannce_rat_ft/gslrm_format/` |
| **학습 frame list** | `outputs/sdannce_rat_ft/data_rat_train.txt` |

---

## 3. Step 1: Frame Selection

### 현재 마스크 커버리지

```
기존 20 frames: 0, 25, 50, 75, ..., 475 (step=25, Camera1-6 각각)
목표: 500~1000 frames (다양한 pose 포함)
```

### 추천 전략: DANNCE Keypoint 기반 Diverse Sampling

```python
# 간단 버전: 균등 샘플링 (90K frames에서 1000개)
frame_indices = list(range(0, 90000, 90))  # 1000 frames, step=90 (~3초 간격)

# 고급 버전: Pose diversity (k-means on keypoints)
import scipy.io as sio
import numpy as np
from sklearn.cluster import KMeans

data = sio.loadmat('.../save_data_AVG0.mat')
kp = data['pred']  # (90000, 3, 23)
kp_flat = kp.reshape(90000, -1)  # (90000, 69)

# NaN/invalid 제거
valid = ~np.isnan(kp_flat).any(axis=1)
kmeans = KMeans(n_clusters=500, random_state=42)
kmeans.fit(kp_flat[valid])
# 각 클러스터 중심에 가장 가까운 프레임 선택
```

---

## 4. Step 2: SAM2 Annotation (sdannce-poc)

### 환경 설정

```bash
ssh gpu03
cd /home/joon/dev/sdannce-poc
conda activate sdannce  # Python 3.10, torch 2.4.1+cu121
```

### 방법 A: Keypoint-Guided (추천, 자동)

`kp_sam2_lone.py`가 DANNCE keypoint를 SAM2 positive prompt로 사용하여 자동 마스크 생성.

```bash
# 1000 frames, 6 cameras (GPU 5 사용)
CUDA_VISIBLE_DEVICES=5 python segmentation/kp_sam2_lone.py \
    --session_dir /home/joon/data/sdannce/rat/dataverse/SCN2A_WK1_2022_09_16_M1 \
    --output /home/joon/data/sdannce/rat/dataverse/SCN2A_WK1_2022_09_16_M1/sam2_masks \
    --cameras 1,2,3,4,5,6 \
    --start 0 --end 90000 --step 90
```

**소요 시간**: ~30분 (1000 frames × 6 cams, A6000)

### 방법 B: Video Propagation (대량 처리)

기존 annotation에서 중간 프레임을 SAM2 video predictor로 보간.

```bash
# sam2_propagate.py를 lone mode로 적응 필요
python segmentation/sam2_propagate.py \
    --mode range --start 0 --end 5000 --step 5
```

> ⚠️ `sam2_propagate.py`는 현재 social pair (2마리) 전용. Lone mode 적응 필요.
> 추천: 방법 A (kp_sam2_lone.py)로 충분한 밀도의 annotation 생성.

### 출력 형식

```
sam2_masks/Camera{N}/mask_NNNNNN.npz
  → key "rat1": shape (1200, 1920), dtype bool
```

### QC (Quality Check)

```bash
# 마스크 오버레이 시각화 (kp_sam2_lone.py가 자동 생성)
# overlay 영상: sam2_masks/overlay_6cam_grid.mp4
# 개별 프레임: sam2_masks/Camera{N}/overlay_NNNNNN.png (옵션)
```

---

## 5. Step 3: GS-LRM Format 변환

```bash
cd /home/joon/dev/FaceLift
conda activate facelift

python -m mouse_extensions.scripts.sdannce_to_gslrm \
    --session_dir /home/joon/data/sdannce/rat/dataverse/SCN2A_WK1_2022_09_16_M1 \
    --output_dir outputs/sdannce_rat_ft/gslrm_format \
    --animal_id 1 \
    --frame_indices $(python3 -c "print(' '.join(str(i) for i in range(0, 90000, 90)))") \
    --sam2_mask_dir /home/joon/data/sdannce/rat/dataverse/SCN2A_WK1_2022_09_16_M1/sam2_masks
```

**출력 구조**:
```
outputs/sdannce_rat_ft/gslrm_format/
├── 000000/
│   ├── images/       # 6 RGBA images (512×512, SAM2 마스크 적용)
│   └── opencv_cameras.json
├── 000090/
│   ├── images/
│   └── opencv_cameras.json
├── ...
└── data_sdannce_test.txt  # frame list
```

---

## 6. Step 4: Fine-Tuning

### Config

```yaml
# configs/mouse/experiments/rat_ft_v1.yaml
model:
  num_input_views: 6

training:
  schedule:
    max_fwdbwd_passes: 5000
  optimizer:
    lr: 5e-5  # 1/10 of mouse training
  dataset:
    data_path: outputs/sdannce_rat_ft/gslrm_format
    data_list: outputs/sdannce_rat_ft/gslrm_format/data_sdannce_test.txt

  checkpointing:
    checkpoint_dir: /node_data/joon/checkpoints/FaceLift/gslrm/rat_ft_v1
    resume_from: /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_6view_alpha03_v3/ckpt_0000000000015840.pt

wandb:
  name: rat_ft_v1_alpha03
  project: facelift
```

### 학습 명령어

```bash
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/experiments/rat_ft_v1.yaml
```

### 비교 실험 3-way

| 실험 | Checkpoint | 설명 |
|------|-----------|------|
| Zero-shot | α=0.3 mouse | 학습 없이 바로 추론 |
| Mouse-FT | α=0.3 → rat FT | Mouse에서 전이 학습 |
| Scratch (optional) | Random init | Rat 데이터만으로 처음부터 |

---

## 7. 평가 전략

Rat GT render가 없으므로:

1. **Held-out view**: 6 camera 중 1개를 제외하고 학습 → 제외 카메라로 PSNR 평가
2. **Turntable 정성 평가**: 360° 회전 영상으로 artifact 유무 확인
3. **Keypoint overlay**: DANNCE keypoint를 렌더에 투영하여 해부학적 정확성 확인
4. **Cross-frame consistency**: 인접 프레임 렌더 간 일관성

---

## 8. 주의사항

| 항목 | 주의 |
|------|------|
| **Conda env** | SAM2 = `sdannce`, GS-LRM = `facelift` (혼용 금지!) |
| **DANNCE convention** | MATLAB row vectors, `K[2,0]=cx`. `cv2.projectPoints()` 사용 금지 |
| **FOV** | Zero-pad 1920²→512, fx=605 (1.1× vs trained 549). 약간의 불일치 |
| **마스크 커버리지** | 1-3% (rat이 프레임에서 작음). 정상 |
| **GPU** | SAM2 annotation = GPU 5, GS-LRM training = GPU 4 |

---

*FaceLift | Rat Fine-Tuning Data Guide | 2026-03-22*
