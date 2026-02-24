# Visualization Settings (시각화 설정)

Turntable 및 시각화 관련 설정 문서.

---

## Turntable 설정

### Base Config 기본값 (gslrm_mouse.yaml)

| 설정 | 값 | 설명 |
|------|-----|------|
| **smooth_trajectory** | `true` | 카메라 간 부드러운 보간 |
| **camera_order** | `[1,3,5,0,4,2]` | 360° 회전 순서 |
| **num_views** | `144` | 총 프레임 수 (6카메라×24프레임) |
| **video_views** | `144` | 비디오 프레임 수 |
| **fps** | `30` | 비디오 프레임레이트 |
| **loop** | `true` | 시작점으로 복귀 |
| **grid_rows** | `6` | 그리드 행 수 |
| **grid_cols** | `6` | 그리드 열 수 |

### 카메라 순서 (MOUSE_CAMERA_ORDER)

```
[1, 3, 5, 0, 4, 2] → 360° 회전
      5
    /   \
   3     0    (위에서 본 배치)
    \   /
      1
     ...
      4
      2
```

- 인접 카메라로 이동하며 360° 회전
- 각 구간 24프레임 보간 (SLERP for rotation, linear for translation)

---

## 알려진 Override Configs

| Config | smooth | fps | num_views | 용도 |
|--------|--------|-----|-----------|------|
| **E_debug.yaml** | true | 2 | 36 | 빠른 디버깅 |
| **E_vis_quality.yaml** | true | 5 | 180 | 고품질 시각화 |

일반 실험 configs (E0, E1, E2 등)는 base 설정을 그대로 사용.

---

## 시각화 색상 체계

### 공통 색상 (FG/BG)

| 용도 | 색상 | RGB |
|------|------|-----|
| **Foreground (전경)** | Green | (0.2, 0.8, 0.2) |
| **Background (배경)** | Red | (0.8, 0.2, 0.2) |

### 시각화별 차이

| 시각화 | 방식 | 결과 |
|--------|------|------|
| **gt_vs_pred** | overlay_blend=0.3 | 30% 투명도 오버레이 |
| **alpha_comparison** | 직접 색상 | 100% 불투명 마스크 |

→ 같은 RGB 값이지만 블렌딩 방식 차이로 다르게 보임 (의도된 설계)

---

## 문제 해결 이력

### 2026-01-26: Turntable 수정

**문제:** smooth trajectory, fps, camera_order 설정이 적용되지 않음

**원인:** Python 코드 기본값을 수정했으나, config 파일이 덮어씀
- `smooth_trajectory: false` (config) vs `True` (Python default)
- `fps: 10` (config) vs `30` (Python default)

**해결:**
1. `configs/base/gslrm_mouse.yaml` 수정
   - smooth_trajectory: false → true
   - fps: 10 → 30
   - num_views: 36 → 144
   - camera_order, video_views 추가

2. Python 백업 기본값 수정 (config 누락 시 대비)
   - gaussians_renderer.py: font_scale 0.5 → 1.0
   - validator.py: smooth_trajectory False → True, fps 15 → 30

3. dataset_views wandb 로깅 제거 (train_gslrm.py)

**교훈:** Config 파일이 항상 Python 기본값보다 우선함

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `configs/base/gslrm_mouse.yaml` | 기본 시각화 설정 |
| `gslrm/model/gaussians_renderer.py` | 렌더링 함수, font_scale |
| `mouse_extensions/validation/validator.py` | 검증 시각화 |
| `mouse_extensions/visualization/alpha_visualization.py` | Alpha 비교 시각화 |
| `mouse_extensions/model/visualization_extensions.py` | GT vs Pred 시각화 |

---

---

## Output File System

> *Source: TURNTABLE_VIS_GUIDE.md (merged 2026-02-11)*

### Single-Frame (Train/Val/Inference)

| Filename | Content |
|----------|---------|
| turntable_orbit_{uid}.mp4 | 360 synthetic orbit (physical CCW) |
| turntable_orbit_with_input_{uid}.mp4 | orbit + labeled input strip |
| turntable_view_with_input_{uid}.mp4 | 6 cam trajectory + hold(15f) + input strip |
| turntable_{uid}.jpg | 6x6 grid image |
| turntable_6view_{uid}.mp4 | 2x3 multiview grid (GT top + Pred bottom) |

### Temporal (Batch Inference)

| Filename | Content |
|----------|---------|
| time_fixed.mp4 | Fixed view, temporal variation |
| time_rotating.mp4 | Temporal + rotation combined |

---

## Rotation Direction

| Function | Coordinate System | Default |
|----------|-------------------|---------|
| get_turntable_cameras | cos->x, sin->y (from +X) | clockwise=True (physical CCW) |
| compute_camera_order | atan2(x,y) (from +Y) | Physical CCW order |

`rotation_direction: "ccw"` (default) = all videos physical CCW (counterclockwise from above).

---

## Turntable Commands

> Consolidated from `COMMANDS.md` (2026-02-24)

### Quick Test (verify_turntable.sh)

```bash
cd /home/joon/dev/FaceLift
nohup bash scripts/verify_turntable.sh 6 > ./logs/verify_turntable.log 2>&1 &
tail -f ./logs/verify_turntable.log

# Custom checkpoint
bash scripts/verify_turntable.sh 6 /path/to/checkpoint.pt
```

| Step | Script | Verifies |
|------|--------|----------|
| 1/2 | render_from_checkpoint.py | Inference path (orbit only) |
| 2/2 | TurntableRenderer.render_all() | Train/Val path (orbit + view_traj + grid) |

### Standalone Inference

```bash
CUDA_VISIBLE_DEVICES=6 python mouse_extensions/scripts/inference/render_from_checkpoint.py \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/ckpt_0000000000009200.pt \
    --config configs/base/gslrm_mouse.yaml \
    --data_path /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_val.txt \
    --output_dir outputs/verify_turntable/inference \
    --mode turntable --num_samples 1
```

### Result Check

```bash
ls outputs/verify_turntable/inference/
ls outputs/verify_turntable/renderer/
scp -r gpu03:~/dev/FaceLift/outputs/verify_turntable/ .
```

---

*Visualization Settings v1.1 | 2026-02-24*
