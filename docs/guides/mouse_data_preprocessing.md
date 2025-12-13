# Mouse Data Preprocessing Pipeline

> FaceLift 학습을 위한 생쥐 멀티뷰 데이터 전처리 가이드

---

## 개요

생쥐 비디오 데이터를 FaceLift (MVDiffusion + GSLRM) 학습에 적합한 형식으로 변환하는 파이프라인.

**스크립트**: `scripts/process_mouse_data.py`

---

## 입력 데이터 구조

```
raw_data/
├── video_dir/
│   └── videos_undist/        # Undistorted 6-view 비디오
│       ├── 0.mp4             # Camera 0
│       ├── 1.mp4             # Camera 1
│       └── ...               # Camera 2-5
│
└── meta_dir/
    ├── new_cam.pkl           # MAMMAL 형식 카메라 캘리브레이션
    └── simpleclick_undist/   # SimpleClick segmentation 마스크 비디오
        ├── 0.mp4
        ├── 1.mp4
        └── ...
```

### 카메라 파라미터 (new_cam.pkl)

MAMMAL 프로젝트에서 생성된 카메라 캘리브레이션 데이터:

```python
cam_data = [
    {
        "R": np.array,      # 3x3 rotation matrix (world-to-camera)
        "T": np.array,      # 3x1 translation vector
        "K": np.array,      # 3x3 intrinsic matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
    },
    ...  # 6 cameras
]
```

### Segmentation 마스크

- **도구**: SimpleClick (interactive segmentation)
- **형식**: 비디오 파일 (프레임별 binary mask)
- **값**: 흰색(255) = foreground, 검정(0) = background

---

## 전처리 단계

### 1. 카메라 파라미터 로드 및 정규화

```python
# scripts/process_mouse_data.py: lines 36-123, 189-278

# MAMMAL 카메라 특성
# - 카메라 거리: 246-414 units (실제 mm 단위)
# - 다양한 FOV

# FaceLift 표준 (Human face 학습 기준)
# - 카메라 거리: 2.7 units
# - FOV: 50°
# - fx = fy ≈ 549.5 (for 512px image)

# 정규화 과정
avg_distance = np.mean([np.linalg.norm(-R.T @ T) for R, T in cameras])
scale_factor = 2.7 / avg_distance  # ~0.008 for mouse data

# Translation만 스케일링 (Rotation 유지)
T_normalized = T * scale_factor
```

**왜 정규화하는가?**
- FaceLift 모델은 특정 카메라 거리/FOV로 학습됨
- 다른 스케일의 데이터는 depth 예측에 영향
- 일관된 좌표계에서 학습해야 일반화 성능 향상

### 2. 프레임 추출

```python
# scripts/process_mouse_data.py: lines 281-349

# 샘플링 전략
frame_indices = compute_sampling_indices(
    total_frames,
    num_samples=2000,
    skip_start_percent=0.05,  # 시작 5% 스킵 (setup 프레임)
    skip_end_percent=0.05     # 끝 5% 스킵
)

# 균일 샘플링으로 다양한 포즈 확보
```

### 3. 마스크 적용 및 RGBA 변환

```python
# scripts/process_mouse_data.py: lines 352-402

def process_frame_with_mask(frame, mask, target_size=512):
    # BGR → RGB 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 마스크 이진화
    if mask is not None:
        alpha = (mask > 127).astype(np.uint8) * 255
    else:
        alpha = np.ones_like(frame[:,:,0]) * 255

    # 정사각형이 아닌 경우 CENTER CROP
    h, w = frame_rgb.shape[:2]
    if h != w:
        min_dim = min(h, w)
        start_h = (h - min_dim) // 2
        start_w = (w - min_dim) // 2
        frame_rgb = frame_rgb[start_h:start_h+min_dim, start_w:start_w+min_dim]
        alpha = alpha[start_h:start_h+min_dim, start_w:start_w+min_dim]

    # 512x512로 리사이즈
    frame_rgb = cv2.resize(frame_rgb, (target_size, target_size))
    alpha = cv2.resize(alpha, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

    # RGBA 이미지 생성
    rgba = np.zeros((target_size, target_size, 4), dtype=np.uint8)
    rgba[:, :, :3] = frame_rgb
    rgba[:, :, 3] = alpha

    return rgba
```

**Center Crop 주의사항**:
- 원본 비디오가 정사각형이 아닌 경우 적용
- 생쥐가 프레임 가장자리에 있으면 일부가 잘릴 수 있음
- 대부분의 프레임에서 생쥐가 중앙에 위치하므로 큰 문제 없음

### 4. 출력 저장

```python
# 디렉토리 구조
output_dir/
├── sample_000000/
│   ├── images/
│   │   ├── cam_000.png    # 512x512 RGBA PNG
│   │   ├── cam_001.png
│   │   ├── cam_002.png
│   │   ├── cam_003.png
│   │   ├── cam_004.png
│   │   └── cam_005.png
│   └── opencv_cameras.json
├── sample_000001/
│   └── ...
├── data_mouse_train.txt   # 학습 샘플 경로 목록 (90%)
└── data_mouse_val.txt     # 검증 샘플 경로 목록 (10%)
```

---

## 출력 형식

### opencv_cameras.json

```json
{
  "id": "sample_000000",
  "frames": [
    {
      "w": 512,
      "h": 512,
      "fx": 549.5,           // FaceLift 표준 focal length
      "fy": 549.5,
      "cx": 256.0,           // 이미지 중심
      "cy": 256.0,
      "w2c": [               // 4x4 world-to-camera matrix
        [r00, r01, r02, t0],
        [r10, r11, r12, t1],
        [r20, r21, r22, t2],
        [0, 0, 0, 1]
      ],
      "file_path": "images/cam_000.png",
      "view_id": 0
    },
    // ... 5 more cameras
  ]
}
```

### 이미지 파일

| 속성 | 값 |
|------|-----|
| 형식 | PNG (RGBA) |
| 크기 | 512 x 512 |
| 채널 | R, G, B, Alpha |
| Alpha | 255=foreground, 0=background |
| 배경 | 투명 (학습 시 white로 composite) |

---

## 사용법

```bash
python scripts/process_mouse_data.py \
    --video_dir /path/to/videos \
    --meta_dir /path/to/masks_and_cameras \
    --output_dir /path/to/output \
    --num_samples 2000 \
    --image_size 512 \
    --num_views 6 \
    --val_ratio 0.1 \
    --bg_color white \
    --target_distance 2.7 \
    --target_fov 50.0
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--num_samples` | 2000 | 추출할 프레임 수 |
| `--image_size` | 512 | 출력 이미지 크기 |
| `--num_views` | 6 | 카메라 뷰 수 |
| `--val_ratio` | 0.1 | 검증 세트 비율 |
| `--target_distance` | 2.7 | FaceLift 표준 카메라 거리 |
| `--target_fov` | 50.0 | FaceLift 표준 FOV (degrees) |

---

## 데이터 흐름 요약

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw Videos     │     │  SimpleClick     │     │  MAMMAL         │
│  (6 cameras)    │     │  Masks           │     │  Calibration    │
└────────┬────────┘     └────────┬─────────┘     └────────┬────────┘
         │                       │                        │
         └───────────────────────┼────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  process_mouse_data.py │
                    │                        │
                    │  1. Load cameras       │
                    │  2. Normalize scale    │
                    │  3. Extract frames     │
                    │  4. Apply masks        │
                    │  5. Center crop        │
                    │  6. Resize to 512x512  │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  FaceLift Format       │
                    │                        │
                    │  - RGBA PNG images     │
                    │  - opencv_cameras.json │
                    │  - train/val splits    │
                    └────────────────────────┘
```

---

## 관련 문서

- [Mouse FaceLift Usage Guide](./mouse_facelift_usage.md)
- [GSLRM Inference Domain Gap](../reports/251212_research_gslrm_inference_domain_gap.md)
- [MVDiffusion Training Checkpoint Issue](../reports/251212_research_mvdiffusion_training_checkpoint_issue.md)

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2024-12-04 | 초기 스크립트 작성 |
| 2024-12-08 | 카메라 정규화 로직 추가 |
| 2025-12-12 | 문서화 |
