# FaceLift Inference Guide: 인간 vs 생쥐 데이터 비교

**날짜**: 2024-12-18
**목적**: Pretrained 모델 추론 방법 및 인간/생쥐 데이터 구조 비교

---

## 1. 디렉토리 구조

### 인간 데이터 (FaceLift 원본)

```
FaceLift/
├── examples/                          # 입력 이미지 (단일 뷰)
│   ├── 000.png ~ 007.png
│
├── data_sample/                       # 샘플 데이터
│   ├── gslrm/sample_000/              # GS-LRM용 (32뷰)
│   │   ├── images/cam_000.png ~ cam_031.png
│   │   └── opencv_cameras.json
│   │
│   └── mvdiffusion/sample_000/        # MVDiffusion용 (6뷰)
│       └── cam_000.png ~ cam_005.png
│
├── utils_folder/
│   └── opencv_cameras.json            # 추론용 6뷰 카메라 설정
│
└── checkpoints/
    ├── mvdiffusion/pipeckpts/         # MVDiffusion 체크포인트
    └── gslrm/ckpt_*.pt                # GS-LRM 체크포인트
```

### 생쥐 데이터 (Mouse)

```
FaceLift/  (gpu05 서버)
└── data_mouse/
    ├── data_mouse_train.txt           # 학습 샘플 목록
    ├── data_mouse_val.txt             # 검증 샘플 목록
    ├── sample_000000/
    │   ├── images/
    │   │   ├── cam_000.png ~ cam_005.png   # 6뷰
    │   └── opencv_cameras.json
    └── sample_XXXXXX/...
```

---

## 2. 카메라 파라미터 비교

### 인간 데이터 (utils_folder/opencv_cameras.json)

```json
{
  "frames": [
    {
      "w": 512,
      "h": 512,
      "fx": 548.9937744140625,
      "fy": 548.9937744140625,
      "cx": 256.0,
      "cy": 256.0,
      "w2c": [[4x4 matrix]],
      "blender_camera_name": "lrm_cam.000",
      "blender_camera_location": [-2.7, 0.0, 0.0]
    }
  ]
}
```

**특징:**
- 카메라 거리: **2.7** (고정)
- 좌표계: **Y-up** (Blender 표준)
- 뷰 배치: XZ 평면 균일 orbit
- Elevation: 0° (수평)

### 생쥐 데이터 (data_mouse/sample_XXXXXX/opencv_cameras.json)

```json
{
  "frames": [
    {
      "w": 512,
      "h": 512,
      "fx": 548.993771650447,
      "fy": 548.993771650447,
      "cx": 256.0,
      "cy": 256.0,
      "w2c": [[4x4 matrix]],
      "file_path": "images/cam_000.png",
      "view_id": 0
    }
  ]
}
```

**특징:**
- 카메라 거리: **1.9 ~ 3.4** (불균일) → 2.7로 정규화 필요
- 좌표계: **Z-up** (원본) → Y-up으로 정규화 필요
- 뷰 배치: XY 평면 비균일 orbit
- Elevation: 다양 (~20° 범위)

---

## 3. 핵심 파라미터 비교표

| 항목 | 인간 (FaceLift) | 생쥐 (Mouse) | 변환 필요 |
|------|-----------------|--------------|-----------|
| **이미지 해상도** | 512×512 | 256×288 → 512×512 | 리사이즈 |
| **카메라 거리** | 2.7 (고정) | 1.9~3.4 | `normalize_camera_distance()` |
| **Up direction** | Y-up | Z-up | `normalize_to_y_up()` |
| **카메라 개수** | 6 (추론), 32 (학습) | 6 | - |
| **focal length** | fx=fy=548.99 | fx=fy=548.99 | 동일 |
| **principal point** | cx=cy=256 | cx=cy=256 | 동일 |

---

## 4. 추론 실행 방법

### Step 1: 환경 설정

```bash
# gpu05 서버
cd ~/FaceLift
conda activate mouse_facelift
```

### Step 2: 인간 데이터 추론 (Pretrained)

```bash
# 기본 실행 (examples/ → outputs/)
python inference.py

# 커스텀 입력
python inference.py \
  --input_dir /path/to/face/images \
  --output_dir /path/to/output \
  --auto_crop \
  --seed 4 \
  --guidance_scale_2D 3.0 \
  --step_2D 50
```

**파이프라인 흐름:**
```
입력 이미지 (1장)
    ↓ preprocess_image() - 얼굴 검출 & 크롭
전처리된 이미지 (512×512)
    ↓ MVDiffusion - 멀티뷰 생성
6개 뷰 이미지
    ↓ GS-LRM - 3D 재구성
3D Gaussians
    ↓ Rasterizer - 렌더링
turntable.mp4 + gaussians.ply
```

### Step 3: 생쥐 데이터 추론 (Fine-tuned)

```bash
# 학습된 체크포인트로 추론
python inference_mouse.py \
  --config configs/mouse_gslrm_v2.yaml \
  --input_dir data_mouse/sample_000000 \
  --output_dir outputs/mouse_inference
```

---

## 5. 코드 핵심 함수

### 카메라 설정 (inference.py:139-162)

```python
def setup_camera_parameters(device):
    """6뷰 카메라 설정 (인간 데이터용)"""
    camera_file = "utils_folder/opencv_cameras.json"
    camera_data = json.load(open(camera_file))["frames"]

    # 6뷰 선택: [2, 1, 0, 5, 4, 3] 순서
    camera_indices = [2, 1, 0, 5, 4, 3]
    selected_cameras = [camera_data[i] for i in camera_indices]

    for cam in selected_cameras:
        intrinsics = [cam["fx"], cam["fy"], cam["cx"], cam["cy"]]
        extrinsics = np.linalg.inv(cam["w2c"])  # w2c → c2w 변환

    return intrinsics_tensor, extrinsics_tensor
```

### 이미지 전처리 (inference.py:191-203)

```python
# 얼굴 검출 & 크롭 (auto_crop=True)
input_image = preprocess_image(input_image_np)

# 또는 크롭 없이 (auto_crop=False)
input_image = preprocess_image_without_cropping(input_image_np)

# 실패 시 폴백: 배경 제거만
input_image = remove(input_image)  # rembg
input_image = input_image.resize((512, 512))
```

### MVDiffusion 추론 (inference.py:207-225)

```python
mv_imgs = unclip_pipeline(
    input_image,
    None,
    prompt_embeds=color_prompt_embedding,
    guidance_scale=3.0,
    num_images_per_prompt=1,
    num_inference_steps=50,
    generator=generator,
    eta=1.0,
).images

# 6뷰 추출
views = [mv_imgs[i] for i in [1, 2, 3, 4, 5, 6]]  # 또는 [0:6]
```

### GS-LRM 추론 (inference.py:233-264)

```python
# 입력 준비
lrm_input = torch.from_numpy(images).float() / 255  # [B, V, H, W, C]
lrm_input = rearrange(lrm_input, "b v h w c -> b v c h w")

batch = {
    "image": lrm_input,           # [1, 6, 3, 512, 512]
    "c2w": demo_c2w,              # [1, 6, 4, 4]
    "fxfycxcy": demo_fxfycxcy,    # [1, 6, 4]
    "index": demo_index,          # [1, 6, 2]
}

# 추론
with torch.autocast(enabled=True, device_type="cuda", dtype=torch.float16):
    result = gs_lrm_model.forward(batch, create_visual=False, split_data=True)

# 결과 저장
result.gaussians[0].save_ply("gaussians.ply")
```

---

## 6. 생쥐 데이터 정규화 코드

### Y-up 정규화 (mouse_dataset.py)

```python
def normalize_cameras_to_y_up(c2w_matrices):
    """Z-up → Y-up 좌표계 변환"""
    # 1. 카메라 위치들로 orbit plane 추정
    positions = c2w_matrices[:, :3, 3]
    centered = positions - positions.mean(axis=0)

    # 2. PCA로 up direction 추정
    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    up_direction = eigenvectors[:, 0]  # 최소 분산 방향

    # 3. Y-up으로 회전
    target_up = np.array([0, 1, 0])
    R_align = rotation_matrix_from_vectors(up_direction, target_up)

    # 4. 모든 카메라에 적용
    for i in range(len(c2w_matrices)):
        c2w_matrices[i, :3, :3] = R_align @ c2w_matrices[i, :3, :3]
        c2w_matrices[i, :3, 3] = R_align @ c2w_matrices[i, :3, 3]

    return c2w_matrices
```

### 거리 정규화 (mouse_dataset.py)

```python
def normalize_camera_distance(c2w_matrices, target_distance=2.7):
    """카메라 거리를 고정값으로 정규화"""
    for i in range(len(c2w_matrices)):
        cam_pos = c2w_matrices[i, :3, 3]
        current_dist = np.linalg.norm(cam_pos)
        scale = target_distance / current_dist
        c2w_matrices[i, :3, 3] = cam_pos * scale
    return c2w_matrices
```

---

## 7. 출력 파일 구조

```
outputs/{image_name}/
├── input.png         # 전처리된 입력 이미지
├── multiview.png     # 6뷰 생성 결과 (가로 연결)
├── output.png        # 렌더링 결과
├── gaussians.ply     # 3D Gaussian Splatting 모델
└── turntable.mp4     # 360° 회전 비디오 (150프레임, 30fps)
```

---

## 8. 관련 파일 경로

| 파일 | 경로 | 설명 |
|------|------|------|
| 추론 스크립트 | `inference.py` | 인간 데이터용 |
| 생쥐 추론 | `inference_mouse.py` | 생쥐 데이터용 |
| 카메라 설정 | `utils_folder/opencv_cameras.json` | 6뷰 카메라 |
| GS-LRM config | `configs/gslrm.yaml` | 원본 설정 |
| 생쥐 config | `configs/mouse_gslrm_v2.yaml` | Fine-tune 설정 |
| 데이터셋 코드 | `gslrm/data/mouse_dataset.py` | 정규화 포함 |
| MVDiffusion 체크포인트 | `checkpoints/mvdiffusion/pipeckpts/` | |
| GS-LRM 체크포인트 | `checkpoints/gslrm/ckpt_*.pt` | |

---

## 9. 트러블슈팅

### 문제: 흐릿한 출력 (Mode Collapse)
**원인**: L2 loss 우세, 낮은 LR
**해결**:
- `perceptual_loss_weight: 1.0` (0.5에서 증가)
- `lpips_loss_weight: 1.0` (0.5에서 증가)
- `lr: 5e-6` (1e-6에서 증가)

### 문제: Gradient Explosion
**원인**: 도메인 갭으로 인한 불안정
**해결**:
- `grad_clip_norm: 1.0`
- `allowed_gradnorm_factor: 100`
- Skip threshold = 100

### 문제: 잘못된 3D 형상
**원인**: 카메라 정규화 누락
**해결**:
- `normalize_cameras: true`
- `target_camera_distance: 2.7`
