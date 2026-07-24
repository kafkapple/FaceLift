# CH1: Environment & Data

> 서버 환경, 데이터 구조, 전처리 파이프라인, 데이터 로딩 코드를 상세히 설명합니다.
>
> ← [[EXPERIMENT_MASTER_GUIDE]] | [[CH2_GSLRM_CODE_FLOW]] →

---

## 1. Server Environment

### 1.1 gpu03 (Main Training Server)

```bash
# SSH 접속
ssh gpu03    # ProxyJump via storage server

# 환경 활성화
conda activate facelift

# GPU 확인 (4-7만 사용 가능)
nvidia-smi
# GPU 0-3: RTX PRO 6000 Blackwell (97GB) — PyTorch 미지원 (sm_120)
# GPU 4-7: RTX A6000 (48GB) — Ampere, 학습/추론 전용
```

| 항목 | 값 |
|------|-----|
| Python | 3.11 |
| PyTorch | 2.x (CUDA 11.8) |
| Conda env | `facelift` |
| GPU 범위 | `CUDA_VISIBLE_DEVICES=4,5,6,7` |
| Project root | `/home/joon/dev/FaceLift/` |
| Data root | `/home/joon/data/preprocessed/FaceLift_mouse/` |
| Checkpoints | `/node_data/joon/checkpoints/FaceLift/` (symlink: `checkpoints/`) |

### 1.2 joon (Pose-Splatter Server)

```bash
ssh joon
conda activate splatter
# GPU: RTX 3060 (12GB), CUDA 11.x
```

---

## 2. Data Structure

### 2.1 원본 데이터 (DANNCE → MAMMAL)

**데이터 출처**: DANNCE `markerless_mouse_1` (Dunn et al. 2021, Harvard, 1152×1024, 100fps)
→ MAMMAL (An et al. 2023)에서 segment mask 추가 → `markerless_mouse_1_nerf/`로 가공

6대 카메라로 촬영한 실험 마우스 영상:
- **원본**: 18,000 frames × 6 cameras (DANNCE: 1152×1024, 100fps)
- **MAMMAL 가공**: 512×512 비디오 + simpleclick segment mask
- **서브샘플**: `frame_jump=5` → 3,600 frames
- **해상도**: 전처리(M5)로 512×512 RGBA 통일

> **참고**: PoseSplatter 논문은 **별도 Duke 데이터**(1536×2048, 30fps, 324K frames)를 사용.
> 본 프로젝트에서 PS 코드를 M5 데이터에 적용한 것. 상세: [[datasets/RAW_DATA]]

### 2.2 전처리된 데이터셋 (M-series)

```
/home/joon/data/preprocessed/FaceLift_mouse/
├── M5/          # ★ Primary: centering + uniform distance norm
├── M0/          # Raw: 원본 PP, 정규화 없음
├── M5_4/        # Ablation: centering only (norm X)
├── M5_5/        # Ablation: centering + per-view norm
├── M5_mvdiff/   # DA1: Stage 1 (multi-view diffusion) 생성 RGB 이미지
└── (기타 M1~M4, M5h 등은 실험용, 현재 미사용)
```

### 2.3 프레임 디렉토리 구조

각 프레임은 하나의 디렉토리로 구성됩니다:

```
000042/                          # Frame ID (zero-padded 6자리)
├── opencv_cameras.json          # 6개 뷰의 카메라 파라미터
└── images/
    ├── cam_000.png              # View 0 (RGBA 512×512)
    ├── cam_001.png              # View 1
    ├── cam_002.png              # View 2
    ├── cam_003.png              # View 3
    ├── cam_004.png              # View 4
    └── cam_005.png              # View 5
```

### 2.4 opencv_cameras.json 형식

```json
{
  "frames": [
    {
      "file_path": "images/cam_000.png",
      "w": 512, "h": 512,
      "fx": 549.0, "fy": 549.0,      // Focal length (normalized)
      "cx": 256.0, "cy": 256.0,      // Principal point (centered)
      "w2c": [                         // World-to-Camera 4×4 matrix
        [r00, r01, r02, tx],
        [r10, r11, r12, ty],
        [r20, r21, r22, tz],
        [0,   0,   0,   1 ]
      ],
      "view_id": 0
    },
    // ... 5 more views
  ]
}
```

### 2.5 M0 vs M5 vs M5_4 vs M5_5 비교

| 항목 | M0 (Raw) | M5 (Standard) | M5_4 (Center) | M5_5 (Center+Norm) |
|------|----------|---------------|----------------|---------------------|
| **cx, cy** | 뷰마다 다름 (202~220) | **256.0 고정** | **256.0 고정** | **256.0 고정** |
| **PP centering** | ❌ | ✅ shift_to_256 | ✅ shift_to_256 | ✅ shift_to_256 |
| **Camera recenter** | ❌ | ✅ | ❌ | ❌ |
| **Distance norm** | ❌ | ✅ (uniform dist=2.7) | ❌ | ✅ (per-view) |
| **Alpha channel** | ✅ RGBA | ✅ RGBA | ✅ RGBA | ✅ RGBA |
| **GS-LRM 결과** | 발산 | **22.34 dB** | **21.80 dB** | 21.58 dB |

**Why M0 fails**: GS-LRM pretrained model은 cx=cy=256, distance≈2.7을 기대합니다. M0는 PP가 off-center이고 distance가 정규화되지 않아 distribution mismatch로 발산합니다.

---

## 3. Preprocessing Pipeline

### 3.1 Offline Preprocessing (1회 실행)

**스크립트**: `mouse_extensions/preprocessing/preprocess.py`

```bash
# M5 데이터셋 생성 예시
python -m mouse_extensions.preprocessing.preprocess \
    --preset M5 \
    --input /path/to/raw_data \
    --output /home/joon/data/preprocessed/FaceLift_mouse/M5
```

**UnifiedPreprocessor 흐름**:

```python
# mouse_extensions/preprocessing/preprocess.py
class UnifiedPreprocessor:
    def process_frame(self, frame_idx):
        # 1. 원본 카메라 pickle 로드
        cameras = load_cameras(frame_idx)

        # 2. Affine transform 계산 (M5 preset)
        #    - scale: target_fx(549) / original_fx
        #    - shift: PP → (256, 256)
        transform = compute_affine(cameras, preset)

        # 3. 이미지에 affine 적용 → 512×512 RGBA
        for view in range(6):
            img = apply_affine(raw_image, transform)
            img.save(f"images/cam_{view:03d}.png")

        # 4. 카메라 파라미터 업데이트
        #    cx=256, cy=256, fx=549, fy=549
        cameras = update_intrinsics(cameras, transform)

        # 5. (M5 only) Camera re-centering
        #    3D 중심점 추정 → 전체 카메라 translation shift
        if preset.recenter:
            center = estimate_3d_center(cameras, masks)
            cameras = recenter(cameras, center)

        # 6. (M5 only) Distance normalization
        #    모든 카메라의 distance → 2.7 통일
        if preset.normalize_distance:
            cameras = normalize_distance(cameras, target=2.7)

        # 7. opencv_cameras.json 저장
        save_cameras(cameras, output_dir)
```

### 3.2 Preset 정의 비교

**파일**: `mouse_extensions/preprocessing/presets.py`

| Preset | pp_method | recenter | normalize_dist | target_dist |
|--------|-----------|:--------:|:--------------:|:-----------:|
| M0 | original | ❌ | ❌ | — |
| M5 | shift_to_256 | ✅ | ✅ (uniform) | 2.7 |
| M5_4 | shift_to_256 | ❌ | ❌ | — |
| M5_5 | shift_to_256 | ❌ | ✅ (per-view) | 2.7 |

**uniform vs per-view normalization**:
- **Uniform** (M5): 6개 카메라의 평균 distance 계산 → 단일 scale factor → intrinsics도 동일하게 조정
- **Per-view** (M5_5): 각 카메라의 distance 개별 정규화 → intrinsics도 각각 다르게 조정

### 3.3 Center Estimation

**파일**: `mouse_extensions/preprocessing/center_estimation.py`

```python
class CenterEstimator:
    def estimate(self, cameras, masks):
        """3 가지 방법으로 3D 중심점 추정"""

        # Method 1: Triangulation (DLT)
        #   각 뷰의 mask centroid → 2D point
        #   DLT로 3D triangulation
        center_tri = self.triangulate(cameras, mask_centroids)

        # Method 2: Visual Hull
        #   각 뷰의 mask → 3D cone
        #   교집합의 중심점
        center_hull = self.visual_hull(cameras, masks)

        # Method 3: Global Average
        #   모든 카메라 위치의 평균
        center_avg = np.mean(camera_positions, axis=0)

        return center_tri  # 기본값
```

---

## 4. Data Split

### 4.1 M5t2 Split (Canonical)

**파일**: `configs/datasets/M5t2.yaml`

```yaml
data:
  data_root: /home/joon/data/preprocessed/FaceLift_mouse/M5
  train_split: data_mouse_t2_train.txt    # frames 0-2879 (2880)
  val_split: data_mouse_t2_val.txt        # frames 2880-3239 (360)
  test_split: data_mouse_t2_test.txt      # frames 3240-3599 (360)
```

| Split | Frames | 비율 | 용도 |
|-------|--------|:----:|------|
| Train | 0 - 2879 | 80% | 학습 |
| Val | 2880 - 3239 | 10% | 검증 (best_psnr 기준) |
| Test | 3240 - 3599 | 10% | 최종 평가 (fair comparison) |

**Split 파일 형식** (txt, 각 줄에 absolute path):
```
/home/joon/data/preprocessed/FaceLift_mouse/M5/000000/
/home/joon/data/preprocessed/FaceLift_mouse/M5/000001/
...
/home/joon/data/preprocessed/FaceLift_mouse/M5/002879/
```

### 4.2 Why 80:10:10?

이전 M5t split (1:1:1)에서 M5t2 (80:10:10)로 변경한 이유:
- **H1bis 실험**: M5t2가 M5t 대비 +2.9 dB, IoU 2배 향상
- FaceLift GS-LRM은 feed-forward 모델이므로 학습 데이터가 많을수록 일반화 성능 향상
- ~~Pose-Splatter(per-scene optimization)는 split 비율 영향 적음~~ 🔴 **260723 기각** — PS는 dataset-trained(Train/Val/Test split 보유)이므로 split 비율에 **민감**. SSOT §2.1 C5 / D12

---

## 5. Dataset Loading Code

### 5.1 MouseViewDataset

**파일**: `mouse_extensions/data/mouse_dataset.py` (~470줄)

```python
class MouseViewDataset(Dataset):
    """FaceLift GS-LRM용 multi-view 마우스 데이터셋"""

    def __init__(self, config, split='train'):
        # config에서 설정 로드
        self.num_views = config.num_views          # 6 (total views)
        self.num_input_views = config.num_input_views  # 4 (default)
        self.random_view_selection = config.random_view_selection  # True
        self.target_distance = config.target_distance  # 2.7
        self.image_size = config.image_size        # 512

        # Split 파일에서 프레임 경로 로드
        self.samples = self._load_split_file(split_file)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        # ─── Step 1: 카메라 로드 ───
        cameras = load_json(sample_dir / "opencv_cameras.json")

        # ─── Step 2: View 선택 ───
        # random_view_selection=True면 6개 중 랜덤 순서
        # False면 고정 순서 [0,1,2,3,4,5]
        view_indices = self._select_views()

        # ─── Step 3: 이미지 로드 + 전처리 ───
        images = []
        for vi in view_indices:
            img = Image.open(cameras[vi]["file_path"])  # RGBA
            img = img.resize((512, 512), Image.LANCZOS)

            # RGBA → white background composite
            if img.mode == 'RGBA':
                bg = Image.new('RGBA', img.size, (255,255,255,255))
                img = Image.alpha_composite(bg, img)

            images.append(to_tensor(img))  # [C, H, W]

        # ─── Step 4: w2c → c2w 변환 ───
        c2w_list = []
        fxfycxcy_list = []
        for vi in view_indices:
            w2c = np.array(cameras[vi]["w2c"])  # [4,4]
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            c2w = np.eye(4)
            c2w[:3, :3] = R.T           # Rotation transpose
            c2w[:3, 3] = -R.T @ t       # Translation inversion
            c2w_list.append(c2w)
            fxfycxcy_list.append([fx, fy, cx, cy])

        # ─── Step 5: Z-up 정렬 ───
        # PCA로 up direction 추정 → Rodrigues rotation
        c2w_array = np.stack(c2w_list)
        c2w_aligned = normalize_cameras_to_z_up(c2w_array)

        # ─── Step 6: Distance normalization ───
        # 모든 카메라의 distance → 2.7
        c2w_normed, fxfycxcy_normed = normalize_camera_distance_with_intrinsics(
            c2w_aligned, fxfycxcy_array, target_distance=2.7
        )

        # ─── Step 7: Auto-mask 생성 ───
        if self.auto_generate_mask:
            for i, img in enumerate(images):
                # RGB > threshold(250/255) → background
                mask = (img > 250/255).all(dim=0)  # white background
                alpha = (~mask).float()            # foreground = 1
                images[i] = torch.cat([img[:3], alpha.unsqueeze(0)])

        return {
            "image": torch.stack(images),        # [V, 4, 512, 512]
            "c2w": torch.tensor(c2w_normed),     # [V, 4, 4]
            "fxfycxcy": torch.tensor(fxfycxcy_normed),  # [V, 4]
            "all_c2w": all_cameras_c2w,           # [6, 4, 4]
            "all_fxfycxcy": all_cameras_intrinsics,
            "index": view_camera_scene_indices,
            "bg_color": torch.tensor([1., 1., 1.]),
        }
```

### 5.2 Camera Normalization 상세

**파일**: `mouse_extensions/data/preprocessing.py`

```python
def normalize_cameras_to_z_up(c2w_matrices):
    """카메라 Y-up → Z-up 정렬 (GS-LRM pretrained model 기대값)

    Why: 원본 DANNCE 카메라는 임의 orientation.
         GS-LRM은 Z-up 좌표계로 학습됨.
    """
    # 1. 모든 카메라의 up direction (Y축) 수집
    up_vectors = c2w_matrices[:, :3, 1]  # [N, 3]

    # 2. PCA로 평균 up direction 추정
    mean_up = np.mean(up_vectors, axis=0)
    mean_up /= np.linalg.norm(mean_up)

    # 3. mean_up → [0, 0, 1] (Z-up)으로의 rotation 계산
    z_up = np.array([0, 0, 1])
    axis = np.cross(mean_up, z_up)
    angle = np.arccos(np.clip(np.dot(mean_up, z_up), -1, 1))
    R_align = rodrigues_rotation(axis, angle)

    # 4. 모든 카메라에 alignment rotation 적용
    for i in range(len(c2w_matrices)):
        c2w_matrices[i, :3, :3] = R_align @ c2w_matrices[i, :3, :3]
        c2w_matrices[i, :3, 3] = R_align @ c2w_matrices[i, :3, 3]

    return c2w_matrices


def normalize_camera_distance_with_intrinsics(c2w, fxfycxcy, target=2.7):
    """카메라 거리 정규화 + intrinsics 비례 조정

    Why: GS-LRM은 distance=2.7에서 학습됨.
         거리 변경 시 projection 보존을 위해 fx,fy도 조정.
    """
    # 1. 현재 평균 distance 계산
    distances = np.linalg.norm(c2w[:, :3, 3], axis=1)
    mean_dist = np.mean(distances)

    # 2. Scale factor
    scale = target / mean_dist

    # 3. Translation 스케일링
    c2w[:, :3, 3] *= scale

    # 4. Intrinsics도 비례 조정 (perspective projection 보존)
    #    distance가 멀어지면 → 물체가 작아지므로 → fx,fy를 키워서 보상
    fxfycxcy[:, :2] *= scale  # fx, fy만 조정 (cx, cy는 유지)

    return c2w, fxfycxcy
```

### 5.3 View Selection Logic

```python
def _select_views(self):
    """학습용 뷰 선택 로직"""
    all_views = list(range(self.num_views))  # [0,1,2,3,4,5]

    if self.random_view_selection:
        # 매 배치마다 랜덤 순서 → input views도 랜덤
        random.shuffle(all_views)
    # else: 고정 순서 [0,1,2,3,4,5]

    # 처음 num_input_views개가 input, 나머지는 target에 포함
    # Note: target = ALL views (input 포함), input 수만 다름
    return all_views
```

---

## 6. Split Generation

**파일**: `mouse_extensions/preprocessing/split_generator.py`

```bash
# M5t2 split 생성 (80:10:10 temporal)
python -m mouse_extensions.preprocessing.split_generator \
    --data_root /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --strategy temporal \
    --ratios 0.8 0.1 0.1 \
    --output_prefix data_mouse_t2
```

**생성 파일**:
```
configs/splits/
├── data_mouse_t2_train.txt    # 2880 lines
├── data_mouse_t2_val.txt      # 360 lines
└── data_mouse_t2_test.txt     # 360 lines
```

---

## Navigation

| Link | Document |
|------|----------|
| ← Master Guide | [[EXPERIMENT_MASTER_GUIDE]] |
| → Code Flow | [[CH2_GSLRM_CODE_FLOW]] |
| → Experiments | [[CH3_EXPERIMENTS_AND_RESULTS]] |
| Dataset Registry | [[PREPROCESSING_REGISTRY]] |
| Dataset Guide | [[MOUSE_DATASET_GUIDE]] |

---

*CH1 Environment & Data v1.0 | 2026-02-23*
