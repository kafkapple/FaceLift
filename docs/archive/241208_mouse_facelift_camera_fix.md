# Mouse-FaceLift 카메라 파라미터 버그 수정 및 파이프라인 가이드

- **날짜**: 2024-12-08
- **주제**: 카메라 파라미터 버그 분석 및 수정
- **목적**: 학습/추론 실패 원인 분석 및 해결

---

## 1. 발견된 문제점

### 1.1 카메라 데이터 로딩 버그

**위치**: `scripts/process_mouse_data.py` - `load_camera_params()` 함수

**문제**: MAMMAL 카메라 데이터는 **LIST** 형식인데, 코드가 **DICT**로 처리

```python
# 버그 코드
cam_dict = pickle.load(f)
for i in range(num_views):
    if i in cam_dict:  # LIST에서 'in' 연산은 값 검사, 인덱스 검사 아님!
        cameras.append(cam_dict[i])
```

**결과**: 실제 카메라 대신 기본 flat 카메라로 대체됨

### 1.2 카메라 거리 불일치

| 파라미터 | MAMMAL 실제 | FaceLift 기대값 | 기존 코드 (버그) |
|---------|-------------|----------------|-----------------|
| 거리 | 246-414 단위 | ~2.7 단위 | 2.7 (기본값) |
| 고도각 | 11-31° (다양) | 20° | 0° (flat) |
| 방위각 | 불규칙 배치 | turntable | 균등 60° 간격 |
| FOV | 40-44° | 50° | 50° |

### 1.3 MAMMAL 실제 카메라 정보

```
Camera 0: dist=246.1, elevation=14.9°, azimuth=-147.0°, FOV=40.4°
Camera 1: dist=414.5, elevation=20.6°, azimuth=34.0°, FOV=43.6°
Camera 2: dist=363.7, elevation=11.3°, azimuth=86.1°, FOV=41.1°
Camera 3: dist=340.0, elevation=10.7°, azimuth=-11.3°, FOV=39.9°
Camera 4: dist=318.3, elevation=26.5°, azimuth=144.0°, FOV=43.2°
Camera 5: dist=305.7, elevation=30.8°, azimuth=-64.1°, FOV=41.1°
```

---

## 2. 적용된 수정사항

### 2.1 `load_camera_params()` 수정

```python
# 수정된 코드
cam_data = pickle.load(f)

# Handle both list and dict formats
if isinstance(cam_data, list):
    # MAMMAL format: list of camera dicts
    for i in range(min(num_views, len(cam_data))):
        cameras.append(cam_data[i])
elif isinstance(cam_data, dict):
    # Dict format with integer or string keys
    for i in range(num_views):
        if i in cam_data:
            cameras.append(cam_data[i])
        elif str(i) in cam_data:
            cameras.append(cam_data[str(i)])
```

### 2.2 `convert_to_facelift_format()` 수정

- 카메라 거리 정규화 추가 (평균 거리 → 2.7)
- FaceLift 표준 intrinsics 사용 (FOV 50°)
- CLI 옵션 추가: `--target_distance`, `--target_fov`

### 2.3 `generate_default_cameras()` 수정

- elevation 파라미터 추가 (기본값: 20°)
- FaceLift `get_turntable_cameras()`와 동일한 카메라 배치

**커밋**: `c14d9c1` - `fix(mouse): fix camera loading and add distance normalization`

---

## 3. 파이프라인 실행 가이드

### 3.0 사전 준비 (gpu05)

```bash
# gpu05 접속
ssh gpu05

# kafka 드라이브 마운트 확인
ls /media/joon/kafka/data/raw/markerless_mouse_1_nerf/
# 없으면: sudo mount /dev/sdb1 /media/joon/kafka

# conda 환경 활성화
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mouse_facelift

# 코드 최신화
cd /home/joon/FaceLift
git pull origin main
```

### 3.1 데이터 전처리

```bash
# 기존 데이터 백업
mv data_mouse data_mouse_old_$(date +%Y%m%d)

# 데이터 재전처리 (카메라 정규화 포함)
python scripts/process_mouse_data.py \
    --video_dir /media/joon/kafka/data/raw/markerless_mouse_1_nerf/videos_undist \
    --meta_dir /media/joon/kafka/data/raw/markerless_mouse_1_nerf \
    --output_dir data_mouse \
    --num_samples 2000 \
    --num_views 6 \
    --image_size 512 \
    --target_distance 2.7 \
    --target_fov 50 \
    --val_ratio 0.1
```

**확인 사항**:
```
Camera normalization: avg_dist=XXX.X, target=2.7, scale=0.XXXXXX
Camera 0 after norm: dist=2.70, elev=XX.X°, fx=549.0, fov=50.0°
```

### 3.2 Pretrained 체크포인트 다운로드

```bash
# HuggingFace 토큰 설정
echo "HF_TOKEN=your_huggingface_token" > .env

# 체크포인트 다운로드
python scripts/download_checkpoints.py

# 확인
ls -la checkpoints/gslrm/
```

### 3.3 학습 실행

```bash
# 디버그 모드 (빠른 테스트, ~30분)
python train_mouse.py --config configs/mouse_config_debug.yaml

# 전체 학습 (Single GPU)
python train_mouse.py --config configs/mouse_config.yaml

# Multi-GPU 학습 (권장)
torchrun --nproc_per_node=4 train_mouse.py --config configs/mouse_config.yaml
```

### 3.4 추론

```bash
# 단일 이미지 추론
python inference_mouse.py \
    --input_image path/to/mouse.png \
    --output_dir outputs/test/ \
    --checkpoint checkpoints/gslrm/mouse/

# 디렉토리 추론
python inference_mouse.py \
    --input_dir data_mouse/sample_000000/images/ \
    --output_dir outputs/sample_000000/ \
    --checkpoint checkpoints/gslrm/mouse/
```

---

## 4. 데이터 구조

### 4.1 원본 데이터 (MAMMAL)

```
/media/joon/kafka/data/raw/markerless_mouse_1_nerf/
├── new_cam.pkl              # 카메라 파라미터 (LIST of 6 cameras)
├── videos_undist/           # 6개 뷰 비디오 (0.mp4 ~ 5.mp4)
└── simpleclick_undist/      # Segmentation 마스크 비디오
```

### 4.2 전처리된 데이터

```
data_mouse/
├── data_mouse_train.txt     # 학습 샘플 경로 (1,799개)
├── data_mouse_val.txt       # 검증 샘플 경로 (199개)
└── sample_XXXXXX/
    ├── opencv_cameras.json  # FaceLift 형식 카메라
    └── images/
        ├── cam_000.png      # RGBA (512x512, 마스크 포함)
        ├── cam_001.png
        └── ...
```

### 4.3 opencv_cameras.json 형식

```json
{
  "frames": [
    {
      "w": 512, "h": 512,
      "fx": 549.0, "fy": 549.0,
      "cx": 256.0, "cy": 256.0,
      "w2c": [[...], [...], [...], [...]],  // 4x4 world-to-camera
      "file_path": "images/cam_000.png",
      "view_id": 0
    },
    ...
  ]
}
```

---

## 5. Novel View Synthesis

### 5.1 기본 개념

FaceLift는 **3D Gaussian Splatting** 기반으로, 입력 이미지로부터 3D 표현을 생성하고 **임의의 카메라 위치**에서 렌더링 가능.

- 학습 데이터: 6개 뷰
- 출력 가능 뷰: **무제한** (임의의 카메라 위치)

### 5.2 다양한 뷰 생성 방법

```python
from gslrm.model.gaussians_renderer import get_turntable_cameras

# 24개 뷰 turntable
w, h, num_views, fxfycxcy, c2ws = get_turntable_cameras(
    hfov=50,
    num_views=24,
    w=512, h=512,
    radius=2.7,
    elevation=20
)

# 다양한 고도에서 촬영
for elev in [0, 15, 30, 45]:
    cameras = get_turntable_cameras(num_views=8, elevation=elev)
```

### 5.3 360° 회전 비디오 생성

```python
# 60 프레임 회전 비디오
cameras = get_turntable_cameras(num_views=60, elevation=20)
# 렌더링 후 videoio로 저장
```

---

## 6. 주요 설정 파일

### 6.1 configs/mouse_config.yaml

| 섹션 | 파라미터 | 값 | 설명 |
|-----|---------|-----|------|
| model.gaussians | n_gaussians | 2 | Gaussian 수 (12288로 증가 가능) |
| training.dataset | num_views | 6 | 뷰 수 |
| training.dataset | num_input_views | 1 | 입력 뷰 수 |
| training.optimizer | lr | 0.00005 | 학습률 |
| training.schedule | max_fwdbwd_passes | 100000 | 최대 스텝 |
| training.checkpointing | resume_ckpt | checkpoints/gslrm | Pretrained 체크포인트 |

### 6.2 configs/mouse_config_debug.yaml

디버그용 빠른 설정:
- `max_fwdbwd_passes: 1000`
- `batch_size_per_gpu: 4`
- `wandb.offline: true`

---

## 7. 문제 해결

### Q: 학습이 너무 빨리 끝남
**A**: `max_fwdbwd_passes` 값 확인 (10000 → 100000)

### Q: 추론 결과가 이상함
**A**:
1. 카메라 파라미터 확인 (`opencv_cameras.json`)
2. Pretrained 체크포인트 로드 확인
3. 데이터 재전처리 필요

### Q: kafka 마운트 안됨
**A**: `sudo mount /dev/sdb1 /media/joon/kafka`

---

## 8. 참고 자료

- FaceLift 논문: [arXiv link]
- GS-LRM: Gaussian Splatting Large Reconstruction Model
- MAMMAL 데이터셋: Multi-view Animal Motion capture

---

*🤖 Generated with Claude Code*
