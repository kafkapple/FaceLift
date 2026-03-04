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

1. 얼굴 3점 추출
   L_ear = kp_3d[0], R_ear = kp_3d[1], nose = kp_3d[2]
   face_center = mean(L_ear, R_ear, nose)

2. 얼굴 평면 벡터
   v_ear = R_ear - L_ear           (귀 사이 방향)
   v_nose = nose - ear_midpoint    (코 방향)

3. 얼굴 법선 (face normal)
   face_normal = normalize(cross(v_ear, v_nose))
   → 두개골 바깥 방향 (얼굴에 수직)

4. 카메라 배치
   cam_pos = face_center + distance × face_normal
   → 얼굴 법선 방향으로 distance 만큼 떨어진 위치

5. 카메라 방향 (OpenCV convention: z=forward, y=down)
   forward = normalize(face_center - cam_pos)   → z축 (시선)
   right   = normalize(cross(forward, v_nose))   → x축
   down    = cross(forward, right)                → y축

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

*Created: 2026-03-04 | Updated: 2026-03-04*
