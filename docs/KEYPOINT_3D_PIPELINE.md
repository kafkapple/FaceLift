# MAMMAL 3D Keypoint Pipeline for FaceLift

MAMMAL body model에서 추출한 22-keypoint 3D 좌표를 FaceLift 시각화에 활용하기 위한
좌표 변환 파이프라인 및 데이터 위치 문서.

---

## 1. 데이터 위치

### 원본 MAMMAL Params (최적화 결과)
```
/node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/params/
├── step_1_frame_000000.pkl   # Step 1 optimization
├── step_1_frame_000005.pkl   # 5-frame interval
├── ...
├── step_2_frame_000000.pkl   # Step 2 (refined, 사용 권장)
├── step_2_frame_000005.pkl
├── ...
└── step_2_frame_017995.pkl   # Total: 7200 files (3600 × 2 steps)
```

각 `.pkl` 파일 구조:
| Key | Shape | Description |
|-----|-------|-------------|
| `thetas` | `[1, 140, 3]` | Joint rotation (axis-angle) |
| `trans` | `[1, 3]` | Global translation (T) |
| `scale` | `[1, 1]` | Global scale (s) |
| `rotation` | `[1, 3]` | Global rotation (R, axis-angle) |
| `bone_lengths` | `[1, 20]` | Bone length parameters |
| `chest_deformer` | `[1, 1]` | Chest deformation |

### 추출된 3D Keypoints (NPZ)
```
/node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz
```

| Key | Shape | Description |
|-----|-------|-------------|
| `keypoints` | `(3600, 22, 3)` | MAMMAL world 좌표 (mm 단위) |
| `frame_indices` | `(3600,)` | 원본 비디오 프레임 인덱스 (0, 5, 10, ..., 17995) |
| `keypoint_names` | `(22,)` | Keypoint 이름 리스트 |

### FaceLift 전처리된 카메라
```
~/data/preprocessed/FaceLift_mouse/M5/{NNNNNN}/opencv_cameras.json
```
- M5 frame 000000 ~ 003599 (3600 프레임)
- 각 프레임당 6개 카메라 뷰 (`cam_000.png` ~ `cam_005.png`)

---

## 2. 좌표계 (Coordinate Systems)

### MAMMAL World Space (원본)
- **단위**: mm (밀리미터)
- **원점**: DANNCE multi-camera 캘리브레이션 기준
- **좌표 범위**: 대략 [-50, 150] mm
- **출처**: `ArticulationTorch.forward()` → `forward_keypoints22()`
- 전역 R/T/s 적용 후의 3D 좌표

### FaceLift Normalized World Space
- **단위**: 무차원 (normalized)
- **원점**: 6개 카메라 position의 centroid (scene center)
- **카메라 거리**: 평균 2.7 (target_distance)
- **좌표 범위**: 대략 [-1.0, 1.0]

---

## 3. 좌표 변환 (MAMMAL → FaceLift)

### 변환 공식
```
p_facelift = (p_mammal - scene_center) × distance_scale
```

### 파라미터 (M5 데이터셋 고정값)

| 파라미터 | 값 | 출처 |
|----------|-----|------|
| `scene_center` | `[59.672, 51.517, 107.099]` mm | 6카메라 position centroid |
| `mean_cam_dist` | `307.785` mm | 카메라-씬센터 평균 거리 |
| `target_distance` | `2.7` | FaceLift convention |
| `distance_scale` | `2.7 / 307.785 = 0.008772` | 스케일 팩터 |

### 유도 과정

1. **Raw 카메라 로드**: `~/data/raw/markerless_mouse_1_nerf/new_cam.pkl`
   - 6개 카메라의 world-space position 추출

2. **Scene center 계산**: 6개 카메라 position의 centroid
   ```python
   positions = [c2w[:3, 3] for c2w in cameras]  # 6개 카메라 위치
   scene_center = np.mean(positions, axis=0)     # [59.672, 51.517, 107.099]
   ```

3. **Distance scale 계산**:
   ```python
   centered = positions - scene_center
   mean_dist = np.linalg.norm(centered, axis=1).mean()  # 307.785 mm
   distance_scale = target_distance / mean_dist          # 2.7 / 307.785
   ```

4. **변환 적용**:
   ```python
   kp_facelift = (kp_mammal - scene_center) * distance_scale
   ```

### 검증
- 변환 후 카메라 위치: origin 기준 평균 거리 = 2.700 (정확히 일치)
- 변환 후 keypoint 좌표 범위: [-1.0, +1.0] (이미지 내 22/22 투영 성공)

---

## 4. 프레임 인덱스 매핑

```
M5 frame index (i) → 원본 비디오 프레임 (i × 5) → MAMMAL param 파일
```

| M5 Frame | Video Frame | Param File | Split |
|----------|-------------|------------|-------|
| 0 | 0 | `step_2_frame_000000.pkl` | Train |
| 2879 | 14395 | `step_2_frame_014395.pkl` | Train |
| 2880 | 14400 | `step_2_frame_014400.pkl` | Val |
| 3239 | 16195 | `step_2_frame_016195.pkl` | Val |
| 3240 | 16200 | `step_2_frame_016200.pkl` | Test |
| 3599 | 17995 | `step_2_frame_017995.pkl` | Test |

**Split**: Train (0-2879, 80%) / Val (2880-3239, 10%) / Test (3240-3599, 10%)

---

## 5. 코드 모듈

### 추출 스크립트
- **파일**: `mouse_extensions/scripts/extract_mammal_keypoints.py`
- **환경**: `mammal_stable` (SMAL/articulation 의존성)
- **기능**: MAMMAL params → ArticulationTorch → forward_keypoints22() → NPZ 저장
- **입력**: `--mammal_dir`, `--params_dir`, `--step` (default: 2)
- **출력**: `keypoints_22_3d.npz`

### 좌표 변환 모듈
- **파일**: `mouse_extensions/scripts/visualize_keypoint_inference.py`
- **함수**: `transform_mammal_to_facelift(keypoints_3d)`
- **상수**: `M5_SCENE_CENTER`, `M5_DISTANCE_SCALE`
- **출처**: `mouse_extensions/preprocessing/preprocess.py` `_normalize_cameras_batch()` (line 1113)

### 시각화 모듈
- **파일**: `mouse_extensions/visualization/keypoint_overlay.py`
- **주요 API**:
  - `project_3d_to_2d(kp_3d, w2c, intrinsics)` → 3D→2D 투영
  - `draw_keypoint_overlay(image, kp_2d, valid, ...)` → 오버레이 그리기
  - `KeypointVisualizer` → 배치 처리 + 멀티뷰 지원
  - `create_legend()` → 색상 범례 생성
  - `compute_face_camera_c2w(kp_3d, distance)` → 얼굴 수직 방향 카메라 c2w 생성
  - `CameraFollowConfig` → 카메라 팔로우 설정 (target, distance, smoothing)
  - `KeypointFollowCamera` → 시퀀스 전체 카메라 궤적 생성 + EMA smoothing

### 카메라 팔로우 렌더링 스크립트
- **파일**: `mouse_extensions/scripts/render_camera_follow.py`
- **기능**: GS-LRM 추론 + 키포인트 기반 카메라 추적 + 비디오 생성
- **출력**:
  - `camera_follow_face_clean.mp4` — 클린 렌더링
  - `camera_follow_face_overlay.mp4` — 키포인트 스켈레톤 오버레이
  - `camera_follow_face_sidebyside.mp4` — 클린 | 오버레이 병렬
  - `multiview_grid_overlay.mp4` — GT 6카메라 뷰 2×3 그리드 + 키포인트 오버레이
- **플래그**: `--grid_only` (추론 없이 그리드만), `--no_grid` (그리드 생략)

### 관련 상위 코드
- **MAMMAL body model**: `/home/joon/dev/MAMMAL_mouse/articulation_th.py`
  - `forward(thetas, bone_lengths_core, R, T, s, chest_deformer)` → V_final, J_final
  - `forward_keypoints22()` → (B, 22, 3) from V_final/J_final via mapper
- **Keypoint mapper**: `/home/joon/dev/MAMMAL_mouse/mouse_model/keypoint22_mapper.json`
  - 47개 항목 중 keypoint_id < 22인 22개만 사용
  - Type "V" = vertex 평균, Type "J" = joint 평균
- **FaceLift 전처리**: `mouse_extensions/preprocessing/preprocess.py`
  - `_normalize_cameras_batch()` — re-center + uniform scale
  - Raw camera: `~/data/raw/markerless_mouse_1_nerf/new_cam.pkl`

---

## 6. 22 Keypoint 정의

| Index | Name | Body Part | Color |
|-------|------|-----------|-------|
| 0 | L_ear | Head | Yellow |
| 1 | R_ear | Head | Yellow |
| 2 | nose | Head | Yellow |
| 3 | neck | Body | Magenta |
| 4 | body_middle | Body | Magenta |
| 5 | tail_root | Tail | Orange |
| 6 | tail_middle | Tail | Orange |
| 7 | tail_end | Tail | Orange |
| 8 | L_paw | Left Front | Blue |
| 9 | L_paw_end | Left Front | Blue |
| 10 | L_elbow | Left Front | Blue |
| 11 | L_shoulder | Left Front | Blue |
| 12 | R_paw | Right Front | Green |
| 13 | R_paw_end | Right Front | Green |
| 14 | R_elbow | Right Front | Green |
| 15 | R_shoulder | Right Front | Green |
| 16 | L_foot | Left Hind | Cyan |
| 17 | L_knee | Left Hind | Cyan |
| 18 | L_hip | Left Hind | Cyan |
| 19 | R_foot | Right Hind | Red |
| 20 | R_knee | Right Hind | Red |
| 21 | R_hip | Right Hind | Red |

### Skeleton Bones
```
Head:       [0,2], [1,2]
Spine:      [2,3], [3,4], [4,5]
Tail:       [5,6], [6,7]
L Front:    [11,3], [10,11], [9,10], [8,9]
R Front:    [15,3], [14,15], [13,14], [12,13]
L Hind:     [18,5], [17,18], [16,17]
R Hind:     [21,5], [20,21], [19,20]
```

---

## 7. Camera-Follow 동작 원리

### 개요

매 프레임 MAMMAL 3D 키포인트에서 얼굴 방향을 계산하고, 얼굴에 수직인 방향에서
바라보는 가상 카메라를 배치하여 생쥐 얼굴을 추적하는 영상을 생성한다.

### 카메라 포즈 계산 (`compute_face_camera_c2w`)

```
입력: kp_3d (22, 3) — FaceLift normalized world 좌표
      distance — 카메라와 얼굴 중심 사이 거리

1. 얼굴 키포인트 추출
   nose = kp_3d[2], neck = kp_3d[3]
   L_ear = kp_3d[0], R_ear = kp_3d[1]
   face_center = mean(L_ear, R_ear, nose)

2. 시선 방향 (nose→neck 방향)
   gaze_dir = normalize(neck - nose)
   → 코에서 목 방향 = 머리 안쪽으로 향하는 벡터

3. 카메라 배치 (정면)
   cam_pos = face_center - distance × gaze_dir
   → nose→neck의 반대 방향, 즉 얼굴 정면에 배치

4. Up hint: 얼굴 법선 (face normal)
   v_ear = R_ear - L_ear
   v_nose = nose - ear_midpoint
   face_normal = normalize(cross(v_ear, v_nose))
   → 두개골 바깥 방향, 이것을 up으로 사용하면 귀가 수평 정렬

5. 카메라 방향 (OpenCV convention: z=forward, y=down)
   forward = normalize(face_center - cam_pos)    → z축 (시선)
   right   = normalize(cross(forward, face_normal)) → x축
   down    = cross(forward, right)                 → y축

   c2w = [right | down | forward | cam_pos]  (4×4)
```

### Temporal Smoothing (EMA)

프레임 간 카메라 떨림 방지를 위해 Exponential Moving Average 적용:

```
alpha = smoothing_alpha (0=최대 스무딩, 1=스무딩 없음)

pos_t    = alpha × pos_raw    + (1-alpha) × pos_{t-1}
target_t = alpha × target_raw + (1-alpha) × target_{t-1}
up_t     = alpha × up_raw     + (1-alpha) × up_{t-1}

→ 스무딩된 값으로 c2w 재구성
```

#### 스무딩이 필요한 이유

1. **MAMMAL 키포인트 노이즈**: per-frame optimization 결과이므로 프레임 간 키포인트
   위치가 미세하게 떨림 (특히 귀, 코 등 작은 부위)
2. **짧은 벡터의 방향 증폭**: `nose→neck` 같은 짧은 벡터 (~0.1 FaceLift units)에서
   1-2mm 노이즈가 큰 각도 변화로 증폭됨 (M5 데이터 평균 5.6°/frame)
3. **렌더링 떨림**: 카메라가 매 프레임 튀면 시청자에게 어지러운 영상이 됨

#### 스무딩 파라미터 가이드

| alpha 값 | 동작 | 권장 용도 |
|-----------|------|-----------|
| `0.3` | 매우 부드러움 (70% 이전 유지) | 프레젠테이션, 긴 영상 |
| `0.6` | 중간 (움직임과 안정성 균형) | 일반 시각화 (기본 권장) |
| `0.8` | 빠른 반응 | 빠른 머리 움직임 추적 |
| `1.0` | 스무딩 없음 (raw) | 디버깅, 노이즈 확인 |

M5 test set (360 frames) 분석 결과:
- 프레임 간 시선 변화: 평균 5.6°, 최대 51°
- 총 각도 범위: 177° (거의 180° 회전)
- 얼굴 중심 이동: 최대 1.41 FaceLift units

### 파라미터 (`CameraFollowConfig`)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `target` | `"face"` | 추적 대상 (`"face"` 또는 `"body"`) |
| `distance` | `0.8` | 카메라-얼굴 거리 (FaceLift normalized units) |
| `smoothing_alpha` | `0.3` | EMA 알파 (낮을수록 부드러운 움직임) |
| `resolution` | `512` | 렌더링 해상도 |
| `fx`, `fy` | `549.0` | 초점 거리 (pixel) |
| `cx`, `cy` | `256.0` | 주점 (image center) |

### 멀티뷰 그리드 검증

`--grid_only` 또는 기본 모드에서 GT 6개 카메라 뷰에 3D 키포인트를 투영하여
2×3 그리드로 시각화. 모든 뷰에서 키포인트가 올바르게 정렬되는지 확인 가능.
GS-LRM 추론 없이 빠르게 동작 (GT 이미지 + opencv_cameras.json 활용).

---

## 8. 3D Keypoint 생성 원리

### MAMMAL Body Model 기반 (NOT triangulation)

3D 키포인트는 multi-view triangulation이 **아닌**, **MAMMAL body model fitting**
결과입니다.

#### 과정

```
6-view 이미지 → DANNCE 2D keypoint detection (per-view)
                        ↓
              MAMMAL body model optimization
              (thetas, trans, scale, rotation, bone_lengths)
                        ↓
              ArticulationTorch.forward()
                        ↓
              forward_keypoints22() → (22, 3) 3D 좌표
```

1. **MAMMAL (An et al. 2023)**: 쥐 전용 articulated body model (SMAL 계열)
   - 140개 joint rotation + 20개 bone length + global R/T/s
   - 학습된 mesh template + skinning weights 기반

2. **Per-frame optimization**: 각 프레임에서 6개 카메라 뷰의 silhouette 및
   2D keypoint와 body model projection을 맞추는 최적화 수행
   - Step 1: coarse fitting
   - Step 2: refined fitting (사용 권장)

3. **3D keypoint 추출**: 최적화된 파라미터로 body model forward pass →
   vertex/joint 위치에서 22개 keypoint 추출
   - `keypoint22_mapper.json`: vertex 평균(Type "V") 또는 joint 평균(Type "J")

#### Triangulation과의 차이

| 항목 | Triangulation | MAMMAL Body Model |
|------|---------------|-------------------|
| 입력 | 2D detections + camera calibration | 2D detections + silhouettes |
| 방법 | DLT / SVD | Body model optimization |
| 장점 | 단순, 빠름 | 해부학적 제약, self-occlusion 처리 |
| 단점 | Occlusion에 취약, outlier 민감 | 모델 정확도에 의존 |
| 결과 | 독립적 3D 점들 | 일관된 skeleton + mesh |

---


---

## 9. Multi-View Triangulation Analysis (Oracle Experiment)

### 개요

MAMMAL 3D GT keypoint를 다양한 수의 novel view 카메라에 투영한 뒤,
가우시안 노이즈를 추가하고 DLT triangulation으로 복원하여 이론적 상한을 측정.

### Pipeline

```
MAMMAL 3D GT (22, 3) mm
    ↓ facelift_to_mammal 역변환
FaceLift normalized (22, 3)
    ↓ project_3d_to_2d() × N cameras
2D projections (N, 22, 2) px
    ↓ + Gaussian noise σ={0,1,2,5} px
Noisy 2D (N, 22, 2) px
    ↓ triangulate_batch() — DLT/SVD
Recovered 3D (22, 3) normalized
    ↓ facelift_to_mammal()
Recovered 3D (22, 3) mm
    ↓ compute_mpjpe()
MPJPE (mm) per joint
```

### 실험 변수

| Variable | Values |
|----------|--------|
| View count (N) | 6, 12, 24 |
| Noise σ | 0, 1, 2, 5 px |
| Camera arrangement | Turntable (hfov=50°, radius=2.7, elev=20°) |
| Test frames | 3240-3599 (360 frames) |

### 핵심 코드

- **스크립트**: `mouse_extensions/scripts/multiview_triangulation_eval.py`
- **함수**:
  - `build_novel_view_cameras(N, render_size)` → (N, 3, 4) projection matrices
  - `project_3d_to_2d(kp_3d, P)` → (N, 22, 2) 2D projections
  - `triangulate_batch(kp_2d, P)` → (22, 3) recovered 3D

### 결과 위치
```
outputs/triangulation/oracle_saturation/saturation_analysis.json
```

---

## 10. Neural 2D Keypoint Detection Pipeline

### 개요

Oracle 실험의 GT 2D projection 대신 **학습된 2D detector**를 사용하여
GS-LRM rendered novel view에서 keypoint를 검출하고 triangulation하는 end-to-end 파이프라인.

### Why: Oracle 실험과의 차이

| 항목 | Oracle | Neural Detector |
|------|--------|-----------------|
| 2D keypoints 출처 | GT 3D → project (완벽) | HRNet-w48 inference (오차 포함) |
| 이미지 출처 | N/A (좌표 연산만) | GS-LRM rendered images |
| 노이즈 모델 | Gaussian σ (정규분포) | 실제 detector 오차 (비정규) |
| 검출률 | 100% | Confidence threshold 의존 |
| Domain gap | 없음 | Real camera ↔ Synthetic render |

### Architecture (2-Env Pipeline)

```
┌─────────────────── facelift env (GPU 6) ───────────────────┐
│                                                              │
│  M5 test frames (3240-3599)                                 │
│       ↓                                                      │
│  load_sample_data() → 4-view images (4, 3, 512, 512)       │
│       ↓                                                      │
│  GS-LRM predict() → GaussianModel (N_pts × {xyz, sh, ...}) │
│       ↓                                                      │
│  apply_all_filters() → Filtered gaussians                   │
│       ↓                                                      │
│  get_turntable_cameras(N) → (N, 3, 4) proj matrices        │
│       ↓                                                      │
│  render_opencv_cam() × N → RGB images (N, 384, 384, 3)     │
│       ↓                                                      │
│  Save: cam_{000-N}.png + cameras.json                       │
│                                                              │
└────────── outputs/triangulation/neural_detection/renders/ ───────────┘
                              ↓ (disk)
┌─────────────────── mmpose env (GPU 4) ────────────────────┐
│                                                              │
│  Load rendered images + cameras.json                        │
│       ↓                                                      │
│  get_bbox_from_alpha() → [x1,y1,x2,y2] per view            │
│       ↓                                                      │
│  inference_topdown(HRNet-w48) → 2D keypoints (N, 22, 3)    │
│       ↓                           with [x, y, confidence]   │
│  triangulate_batch(kp_2d, P, conf_thr=0.3)                 │
│       ↓                                                      │
│  facelift_to_mammal() → 3D predictions (22, 3) mm          │
│       ↓                                                      │
│  compute_mpjpe() vs MAMMAL GT → MPJPE (mm)                 │
│                                                              │
└────────── outputs/triangulation/neural_detection/results/ ───────────┘
```

### Phase A: 환경 설치

```bash
# gpu03에서 별도 conda 환경 생성 (facelift env와 분리)
conda create -n mmpose python=3.10 -y
conda activate mmpose
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim
mim install mmengine mmcv mmdet mmpose
```

### Phase B: DANNCE 2D → COCO Format 변환

**입력**: DANNCE 2D keypoints (6 views × 18000 frames × 22kp)
```
~/data/raw/markerless_mouse_1_nerf/keypoints2d_undist/result_view_{0-5}.pkl
    Shape: (18000, 22, 3) — [x, y, confidence]
```

**출력**: COCO 형식 데이터셋
```
~/data/processed/mmpose_mouse/
├── images/                        # 개별 프레임 PNG (1152×1024)
│   ├── {frame_NNNNNN_view_V}.png  # e.g. frame_000000_view_0.png
│   └── ...                        # Total: 17,280(train) + 2,160(val) + 2,160(test)
├── annotations/
│   ├── train.json                 # COCO keypoint annotation (frames 0-2879 × 6 views)
│   ├── val.json                   # frames 2880-3239 × 6 views
│   └── test.json                  # frames 3240-3599 × 6 views
└── mouse_keypoint_info.json       # 22kp skeleton definition
```

**Split** (M5 frame 기준, DANNCE frame = M5 frame × 5):
| Split | M5 Frames | DANNCE Frames | Images |
|-------|-----------|---------------|--------|
| Train | 0-2879 | 0-14395 (×5) | 17,280 |
| Val | 2880-3239 | 14400-16195 (×5) | 2,160 |
| Test | 3240-3599 | 16200-17995 (×5) | 2,160 |

**COCO annotation 형식**:
```json
{
  "images": [{"id": 1, "file_name": "frame_000000_view_0.png", "width": 1152, "height": 1024}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                    "keypoints": [x0, y0, v0, x1, y1, v1, ...],
                    "bbox": [x, y, w, h], "num_keypoints": 22}],
  "categories": [{"id": 1, "name": "mouse", "keypoints": [...], "skeleton": [...]}]
}
```

**스크립트**: `mouse_extensions/scripts/keypoint_detection/convert_dannce_to_coco.py`

### Phase C: HRNet-w48 Fine-tuning

**Config**: `mouse_extensions/configs/mmpose/hrnet_w48_mouse_22kp.py`

| 항목 | 값 |
|------|------|
| Backbone | HRNet-w48 (AP-10K pretrained) |
| Head | HeatmapHead, out_channels=22 |
| Codec | MSRAHeatmap, input (256,192), heatmap (64,48), σ=2 |
| LR | 5e-4 (head), 5e-5 (backbone, lr_mult=0.1) |
| Schedule | LinearLR warmup (5ep) → CosineAnnealing (5-100ep, η_min=1e-6) |
| Batch | 32/GPU |
| Augmentation | RandomFlip, RandomHalfBody, RandomBBoxTransform (rot±30°, scale 0.75-1.25) |
| EMA | momentum=0.0002 |
| Checkpoint | Best coco/AP, max_keep=3 |

**Data flow (training)**:
```
Image (1152×1024) → LoadImage
    → GetBBoxCenterScale (bbox → center, scale)
    → RandomFlip (horizontal, with flip_indices)
    → RandomHalfBody (prob=0.3, min 6kp)
    → RandomBBoxTransform (rotate, scale, shift)
    → TopdownAffine → crop+resize to (256, 192)
    → GenerateTarget → MSRAHeatmap (64, 48) × 22 channels
    → PackPoseInputs
```

**학습 명령어**:
```bash
conda activate mmpose && cd ~/dev/FaceLift
CUDA_VISIBLE_DEVICES=4 python tools/train.py \
    mouse_extensions/configs/mmpose/hrnet_w48_mouse_22kp.py \
    --work-dir work_dirs/hrnet_w48_mouse_22kp
```

**Checkpoint 위치**: `work_dirs/hrnet_w48_mouse_22kp/best_coco_AP.pth`

### Phase D: Novel View Rendering + Detection + Triangulation

#### D-1: GS-LRM Rendering (facelift env)

**스크립트**: `mouse_extensions/scripts/keypoint_detection/render_novel_views_for_detection.py`

```bash
conda activate facelift && cd ~/dev/FaceLift
CUDA_VISIBLE_DEVICES=6 python mouse_extensions/scripts/keypoint_detection/render_novel_views_for_detection.py \
    --config configs/mouse/uniform/base_uniform_v2.yaml \
    --checkpoint checkpoints/gslrm/M5t2_E0_1_facelift/best_psnr.pt \
    --num_views 6 12 24 --render_size 384
```

**출력**:
```
outputs/triangulation/neural_detection/renders/
├── 6views/{003240-003599}/cam_{000-005}.png + cameras.json
├── 12views/{003240-003599}/cam_{000-011}.png + cameras.json
└── 24views/{003240-003599}/cam_{000-023}.png + cameras.json
```

#### D-2: Detection + Triangulation (mmpose env)

**스크립트**: `mouse_extensions/scripts/keypoint_detection/detect_and_triangulate.py`

```bash
conda activate mmpose && cd ~/dev/FaceLift
for NV in 6 12 24; do
    CUDA_VISIBLE_DEVICES=4 python mouse_extensions/scripts/keypoint_detection/detect_and_triangulate.py \
        --render_dir outputs/triangulation/neural_detection/renders/${NV}views \
        --mmpose_config mouse_extensions/configs/mmpose/hrnet_w48_mouse_22kp.py \
        --mmpose_checkpoint work_dirs/hrnet_w48_mouse_22kp/best_coco_AP.pth \
        --gt_3d_path /node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz \
        --output_dir outputs/triangulation/neural_detection/results/${NV}views
done
```

**Detection data flow**:
```
Rendered image (384×384, RGB/RGBA)
    → get_bbox_from_alpha() → [x1,y1,x2,y2]
    → inference_topdown(HRNet-w48, image, bbox)
    → pred_instances.keypoints (22, 2) + keypoint_scores (22,)
    → keypoints_2d (N_views, 22, 3) — [x, y, confidence]
```

**Triangulation data flow**:
```
keypoints_2d (N, 22, 3) + proj_matrices (N, 3, 4)
    → conf > 0.3 필터링
    → triangulate_batch() — per-joint DLT/SVD
    → pred_3d_fl (22, 3) FaceLift normalized
    → facelift_to_mammal() = points / 0.008772 + [59.672, 51.517, 107.099]
    → pred_3d_mm (22, 3) MAMMAL world (mm)
    → compute_mpjpe(pred, gt) → per-joint error (mm)
```

**결과**: `outputs/triangulation/neural_detection/results/{N}views/neural_results.json`

#### D-2 Results (HRNet-w48, epoch 100, best AP=0.9086)

| Views | MPJPE (mm) | PA-MPJPE (mm) | Det. Rate | Effective σ |
|:-----:|:----------:|:-------------:|:---------:|:-----------:|
| 6 | 41.85 ± 35.07 | 23.72 ± 4.00 | 29.1% | 55.5 px |
| 12 | 79.77 ± 197.27 | 22.92 ± 4.50 | 29.2% | 174.7 px |
| 24 | 55.53 ± 137.05 | 23.09 ± 4.10 | 29.1% | 181.9 px |

**Per-joint detection rate (24v)**:
- Best: tail_end (83.5%), nose (70.2%), tail_root (65.6%), L_knee (63.8%)
- Worst: shoulders (0%), elbows (0%), paw_end (<1%), neck (0.1%)

### Phase E: Oracle vs Neural 비교 분석

**스크립트**: `mouse_extensions/scripts/keypoint_detection/compare_oracle_vs_real.py`

```bash
python mouse_extensions/scripts/keypoint_detection/compare_oracle_vs_real.py \
    --oracle_path outputs/triangulation/oracle_saturation/saturation_analysis.json \
    --neural_path outputs/triangulation/neural_detection/results \
    --output_dir outputs/triangulation/neural_detection/comparison
```

**핵심 분석**:
- `estimate_effective_sigma()`: Oracle MPJPE vs σ curve에서 보간하여 detector effective noise 추정
- Per-joint error 비교 (head/spine/tail/legs)
- Domain gap: detection rate by joint group × view count

**출력**:
```
outputs/triangulation/neural_detection/comparison/
├── plots/
│   ├── oracle_vs_real.png              # MPJPE vs view count (oracle curves + neural)
│   ├── per_joint_error.png             # Bar chart: per-joint MPJPE + detection rate
│   └── domain_gap_analysis.png         # Detection rate by joint group
├── viewcount/
│   ├── per_joint_viewcount_comparison.png  # 6v/12v/24v per-joint bars
│   ├── group_summary_comparison.png        # Joint group level comparison
│   ├── oracle_neural_gap_log.png           # Log-scale gap visualization
│   └── summary_table.png                  # Summary metrics table
└── report.md                           # Summary table + effective σ + key findings
```

**Viewcount visualization script**: `mouse_extensions/scripts/keypoint_detection/viewcount_comparison_viz.py`

#### Phase E Key Findings

- Oracle (σ=5px) vs Neural 격차: **12~37x** — domain gap이 지배적
- PA-MPJPE ~23mm stable across views → alignment 후에도 절대 위치 오차 존재
- Detection rate ~29% 가 주요 병목 (GS-LRM render ↔ real camera domain gap)
- 뷰 수 증가(6→24)가 MPJPE를 낮추지 않음 → 뷰 수보다 detection 품질이 핵심

### Coordinate Transform Constants (M5)

| Constant | Value | Source |
|----------|-------|--------|
| `M5_SCENE_CENTER` | `[59.672, 51.517, 107.099]` mm | 6-camera centroid |
| `M5_DISTANCE_SCALE` | `2.7 / 307.785 = 0.008772` | target_dist / mean_cam_dist |

**Forward**: `p_fl = (p_mammal - center) × scale`
**Inverse**: `p_mammal = p_fl / scale + center`

---

*Updated: 2026-03-05*


## 11. Methodological Rationale & Literature Review

### 11.1 Ground Truth 특성 (MAMMAL Pseudo-GT)

현재 GT 3D keypoints는 **marker-based motion capture가 아닌** MAMMAL (An et al., NeurIPS 2023) pseudo-GT입니다.

| Criteria | Marker-based MoCap | Markerless (DANNCE/MAMMAL) |
|----------|-------------------|---------------------------|
| **정확도** | Sub-mm (gold standard) | ~2-5mm (DANNCE), ~3-8mm (MAMMAL 추정) |
| **침습성** | 마커 부착 필요 (행동 변형) | 비침습적 |
| **마우스 적용** | 극히 어려움 (소형, 모피) | 실용적 |
| **데이터셋** | 마우스용 사실상 없음 | DANNCE, MAMMAL 공개 |

**핵심 사항**:
- 마우스는 체구(~3cm 체폭)와 모피로 인해 reflective marker 부착 비실용적
- Dunn et al. (2021)은 **painted marker**(잉크 도트)를 reference로 사용 — 전통적 MoCap과 다름
- MAMMAL pseudo-GT는 2D detection 품질에 직접 의존하므로 ~3-8mm 오차 가능
- **결과 보고 시**: "MAMMAL pseudo-GT 대비 MPJPE"로 명시 필요

### 11.2 왜 HRNet인가? (vs DANNCE)

| Aspect | DANNCE | HRNet + DLT |
|--------|--------|-------------|
| **입력** | 고정 6-cam multi-view 이미지 (동시) | 단일 이미지 |
| **방식** | 직접 3D: 이미지 → 3D voxel volume → 3D heatmap | 2D detection → 삼각측량 |
| **카메라 요구** | 학습 시와 정확히 동일한 배치 필수 | 임의 가상 카메라 적용 가능 |
| **Novel view 적용** | **불가** (카메라 기하학 종속) | **가능** (단일 이미지로 동작) |

**DANNCE가 GS-LRM 렌더뷰에 적용 불가한 이유**:
1. DANNCE는 학습 시 사용한 **정확히 같은 카메라 배치**에서만 동작 (3D voxel 구성이 extrinsics 종속)
2. 단일 뷰 적용 설계 아님 (multi-view 입력 필수)
3. 적용하려면 DANNCE 재학습 + 원본 카메라 배치 재현 필요 → 비실용적

**DANNCE vs MAMMAL의 2D 처리 차이**:
- **DANNCE**: 2D 중간 단계 없이 **직접 3D 예측** (multi-view → 3D volume → 3D heatmap → soft-argmax)
  - 단, 학습 GT는 사람 annotate 2D → triangulation으로 생성 (아이러니)
- **MAMMAL**: **2D detection을 명시적 중간 단계로 사용** (2D backbone → cross-view attention → 3D lifting)

### 11.3 2D vs 3D 비교 평가 방법론

| Approach | 장점 | 단점 |
|----------|------|------|
| **2D per-view (PCK)** | 단순, 추가 오류 없음, 뷰별 진단 가능 | Depth 정보 없음, GT 2D 필요 |
| **3D via triangulation (MPJPE)** | 최종 목표 직접 대응, 타 논문 비교 가능 | 렌더링+검출+삼각측량 오류 누적 |

**SOTA 관행**: 대부분 **3D MPJPE를 primary metric**으로 사용 (Pose-Splatter, DANNCE, MAMMAL).
**권장 보강**: 2D PCK를 보조 metric으로 추가하여 오류 원인 분리 (렌더링 품질 vs 기하학 문제).

### 11.4 현재 파이프라인 평가

```
GS-LRM renders → HRNet 2D detection → DLT triangulation → 3D MPJPE vs MAMMAL pseudo-GT
```

**판단**: 방법론적으로 sound하며, SOTA 관행에 부합.

**알려진 한계**:
1. HRNet domain gap: 실제 카메라 이미지로 학습 → GS-LRM 렌더링 아티팩트에 취약 (Detection rate ~29%)
2. MAMMAL pseudo-GT 오차 (~3-8mm) → MPJPE < 5mm 차이는 GT noise floor 안에 있을 가능성
3. Triangulation은 detection rate가 낮으면 (2 view 미만 검출 시) 3D 추정 불가

**References**:
- Dunn et al. (2021). *Nature Methods* 18, 564-573. (DANNCE)
- An et al. (2023). *NeurIPS 2023*. (MAMMAL)
- Marshall et al. (2021). *Neuron* 109(3), 420-437. (Rodent MoCap)
- Iskakov et al. (2019). *ICCV 2019*. (Learnable Triangulation)
- Sun et al. (2019). *CVPR 2019*. (HRNet)

---

*Updated: 2026-03-05*
