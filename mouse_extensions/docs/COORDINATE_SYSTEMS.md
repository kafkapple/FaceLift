# Coordinate Systems & Scale Reference

FaceLift 프로젝트의 다중 좌표계 간 변환 규칙. **모든 렌더링 비교 작업의 전제조건**.

---

## 1. 좌표계 정의

### 1.1 MAMMAL World (mm-scale)

| 항목 | 값 |
|------|-----|
| **단위** | mm (밀리미터) |
| **원점** | 케이지 기준 (DANNCE/MAMMAL fitting 원점) |
| **축 방향** | 미확인 (OBJ 로딩 후 검증 필요) |
| **대표 범위** | 수십~수백 mm |
| **파일** | `step_2_frame_XXXXXX.obj` |
| **경로** | `/home/joon/dev/MAMMAL_mouse/results/fitting/.../obj/` |

### 1.2 FaceLift Normalized (GS-LRM operating space)

| 항목 | 값 |
|------|-----|
| **단위** | 무차원 (normalized) |
| **원점** | 씬 중심 (생쥐 대략 원점 근처) |
| **축 방향** | OpenCV convention: X-right, Y-down(?), Z-forward(?) |
| **대표 범위** | 약 -0.5 ~ +0.5 (생쥐 extents) |
| **카메라 거리** | radius ≈ 2.7 (turntable default) |
| **카메라 포맷** | C2W matrix [4,4] + fxfycxcy [4] |

### 1.3 Blender (render_mammal_32view_v2.py)

| 항목 | 값 |
|------|-----|
| **단위** | Blender units (TARGET_OBJECT_SIZE=1.5로 정규화) |
| **축 방향** | Z-up, Y-forward |
| **MAMMAL→Blender** | `(x, y, z) → (x, z, -y)` |
| **별도 스케일링** | mm → Blender units (자체 정규화) |

### 1.4 pyrender / OpenGL

| 항목 | 값 |
|------|-----|
| **축 방향** | X-right, Y-up, Z-backward (out of screen) |
| **OpenCV→OpenGL** | `c2w_gl = c2w_cv @ diag(1, -1, -1, 1)` |

---

## 2. 변환 공식

### 2.1 MAMMAL → FaceLift (핵심 변환)

```python
M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])  # mm
M5_DISTANCE_SCALE = 2.7 / 307.785  # ≈ 0.008781

# Forward
point_fl = (point_mammal_mm - M5_SCENE_CENTER) * M5_DISTANCE_SCALE

# Inverse
point_mammal_mm = point_fl / M5_DISTANCE_SCALE + M5_SCENE_CENTER
```

**출처**: `keypoint_viz.py:164-165`, `render_camera_follow.py`

**의미**:
- `M5_SCENE_CENTER` = 원본 6개 카메라 rig의 centroid (mm 좌표)
- `M5_DISTANCE_SCALE` = 2.7 / (카메라 평균 거리 307.785mm)
- 전처리 (`_normalize_cameras_batch`): centroid→origin + 거리→2.7 정규화
- 변환 후 카메라 위치: 모든 프레임에서 동일 (origin 중심, 거리 ~2.7)
- 변환 후 생쥐: 프레임마다 다른 위치 (max dist ~1.2, 항상 카메라 내부)

**검증 결과 (2026-03-11)**:
- Frame 0: mouse centroid ≈ [0.33, -0.23, -0.63], max dist=1.03 ✓
- Frame 200: mouse centroid ≈ [-0.04, 0.50, -0.62], max dist=1.07 ✓
- Frame 1000: mouse centroid ≈ [-0.51, 0.40, -0.83], max dist=1.22 ✓

### 2.2 OpenCV C2W → OpenGL C2W (pyrender용)

```python
cv_to_gl = np.diag([1, -1, -1, 1])
c2w_gl = c2w_cv @ cv_to_gl
```

**이유**: OpenCV는 Y-down/Z-forward, OpenGL은 Y-up/Z-backward

### 2.3 GS-LRM Camera JSON (opencv_cameras.json)

```json
{
  "frames": [{
    "w2c": [[4x4 matrix]],   // World-to-Camera (need inv for C2W)
    "fx": 548.99, "fy": 548.99,
    "cx": 256.0, "cy": 256.0,
    "w": 512, "h": 512
  }]
}
```

```python
# w2c → c2w
c2w = np.linalg.inv(w2c)
fxfycxcy = np.array([fx, fy, cx, cy])
```

---

## 3. 검증 체크리스트

### 변환 검증 (Phase 0)

- [ ] MAMMAL keypoints를 FaceLift space로 변환 후 원점 근처에 위치하는지 확인
- [ ] FaceLift space에서 max distance from origin < 2.0 (카메라 radius 2.7 내부)
- [ ] MAMMAL mesh vertices를 FaceLift space로 변환 후 GS-LRM Gaussian cloud와 겹치는지 확인

### 렌더링 검증

- [ ] GT 카메라에서 MAMMAL mesh가 GT RGB 이미지의 생쥐 위치와 대략 일치
- [ ] Novel view에서 MAMMAL mesh와 GS-LRM 렌더링의 생쥐 크기/위치 일치
- [ ] 배경색 통일 (white: [1,1,1])

---

## 4. 알려진 이슈 & 트러블슈팅

### Issue Log

| 날짜 | 이슈 | 원인 | 해결 |
|------|------|------|------|
| 2026-03-11 | Phase 0: MAMMAL centroid ≠ origin | 정상. M5_SCENE_CENTER=카메라 rig 중심. 생쥐는 이동하므로 원점 벗어남 | max dist ~1.2 < camera radius 2.7 → OK |
| 2026-03-11 | 카메라 포즈 모든 프레임 동일 | 전처리 시 카메라 rig를 origin에 정규화 | per-sample centering + uniform scale → 2.7 |
| 2026-03-11 | **Mesh pose 완전 불일치** | OBJ frame indexing step=5 누락 (`{N}` → `{N*5}`) | `get_frame_obj_path`: `frame_idx * 5` 적용. [[DATASET_FRAME_INDEXING]] 참조 |
| 2026-03-11 | **Textured mesh "해골" 왜곡** | 수동 UV expansion의 vertex 재인덱싱이 trimesh 내부 재인덱싱과 불일치 → vertex 위치 뒤섞임 | trimesh native OBJ loading + face 대응 기반 `expanded_to_orig` 매핑으로 수정 |

### 잠재적 문제

1. **MAMMAL OBJ 축 방향**: OBJ 파일의 Y/Z 축이 예상과 다를 수 있음
   - 검증: mesh bounds 출력 후 keypoint 위치와 비교
2. **GS-LRM C2W 축 규칙**: OpenCV convention 가정이 맞는지 확인 필요
   - 검증: GT 카메라에서 GS-LRM 렌더링이 GT RGB와 일치하면 OK
3. **스케일 불일치**: M5_DISTANCE_SCALE이 모든 프레임에 동일하게 적용 가능한지
   - 검증: 여러 프레임에서 변환된 keypoint centroid 비교
4. **pyrender intrinsics**: fx/fy가 FaceLift와 pyrender에서 동일하게 해석되는지
   - 검증: GT 카메라에서 렌더링된 MAMMAL silhouette이 GT RGB의 생쥐와 겹치는지

---

## 5. Related Files

| 파일 | 역할 |
|------|------|
| `keypoint_viz.py:164-165` | M5_SCENE_CENTER, M5_DISTANCE_SCALE 정의 |
| `render_camera_follow.py` | transform_mammal_to_facelift() 함수 |
| `render_mammal_32view_v2.py` | Blender 좌표 변환 (별도 체계) |
| `gslrm_pipeline.py:load_sample_data()` | opencv_cameras.json → C2W 변환 |
| `gaussians_renderer.py:render_opencv_cam()` | C2W + fxfycxcy로 렌더링 |
| `poc_mesh_gs_pairs.py` | PoC 스크립트 (이 문서의 변환 공식 사용) |
| `DATASET_FRAME_INDEXING.md` | **프레임 인덱싱 매핑 (step=5 규칙)** |

---

*FaceLift | Coordinate Systems Reference | Created: 2026-03-11*
*이슈 발생 시 Section 4 업데이트 필수*
