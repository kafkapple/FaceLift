# Phase 7: 3D Keypoint Pipeline

> **Navigation**: [← Hub](../PIPELINE_DEEP_DIVE.md) | [Prev: PH6](PH6_FAIR_EVALUATION.md)
>
> **핵심 파일**: `mouse_extensions/scripts/keypoint_detection/render_novel_views_for_detection.py`, `mouse_extensions/scripts/keypoint_detection/detect_and_triangulate.py`

---

> **Why**: 3D Gaussian에서 행동 분석용 3D keypoint를 추출하려면
> novel view 렌더링 → 2D 검출 → 다시점 삼각측량 파이프라인이 필요.
> Oracle (perfect 2D) vs Neural (HRNet) 비교로 domain gap 정량화.

### 7.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase A-C: Oracle Saturation (상한선)                            │
│   Perfect 2D keypoints (GT 3D → project) + noise σ              │
│   → multi-view triangulation → MPJPE                            │
│   → "views × noise" saturation curve                            │
│                                                                 │
│ Result: σ=5px, 24views → MPJPE 1.52mm (upper bound)            │
├─────────────────────────────────────────────────────────────────┤
│ Phase D: Neural Detection                                       │
│                                                                 │
│   [render_novel_views_for_detection.py]  (facelift env)         │
│     GS-LRM + M5 test frames                                    │
│     → Gaussian predict → turntable N views                      │
│     → outputs/{N}views/{frame_id}/cam_*.png + cameras.json      │
│                              │                                  │
│   [detect_and_triangulate.py]  (mmpose env)                     │
│     HRNet-w48 2D keypoint detection (22 joints)                 │
│     → multi-view triangulation (DLT)                            │
│     → FaceLift coords → MAMMAL mm coords                        │
│     → MPJPE / PA-MPJPE vs GT 3D                                 │
│                                                                 │
│ Result: 24views → MPJPE 55.5mm, Det.Rate 29.1%                 │
├─────────────────────────────────────────────────────────────────┤
│ Phase E: Oracle vs Neural Comparison                            │
│   Gap = Neural / Oracle = 12~37× (domain gap 지배적)            │
│   Detection rate ~29% = primary bottleneck                      │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Novel View Rendering

**File**: `mouse_extensions/scripts/keypoint_detection/render_novel_views_for_detection.py`

#### render_frame (L134-228) — 단일 프레임 처리

```python
# 1. GS-LRM predict (L150-156)
#    load_sample_data() → 4 input views → gslrm_model.predict()
#    → Gaussian splat parameters

# 2. Gaussian filtering (L160-166)
#    opacity_thres=0.04, scaling_thres=0.1, floater_thres=0.6
#    crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0]

# 3. Turntable cameras (L168-172)
#    get_turntable_cameras_safe(num_views, render_size)
#    → c2ws [N, 4, 4], fxfycxcys [N, 4]

# 4. Render each view (L191-217)
#    render_single_view(gaussians, c2w, fxfycxcy, ...)
#    → RGBA PNG (cam_000.png, cam_001.png, ...)

# 5. Save cameras.json (L220-226)
#    cameras_to_serializable() → {c2w, w2c, K, fxfycxcy, P}
```

#### cameras_to_serializable (L95-131) — Projection matrix 생성

```python
# c2w (4×4) → w2c = inv(c2w)
# K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
# Rt = w2c[:3, :]  (3×4)
# P = K @ Rt        (3×4 projection matrix for triangulation)
```

### 7.3 Detection & Triangulation

**File**: `mouse_extensions/scripts/keypoint_detection/detect_and_triangulate.py`

> ⚠️ 이 스크립트는 **mmpose conda env**에서 실행해야 합니다 (facelift env 아님).

#### 좌표계 변환 상수 (L34-40)

```python
M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])  # mm
M5_DISTANCE_SCALE = 2.7 / 307.785  # = 0.008772

def facelift_to_mammal(points_3d):
    """FaceLift normalized coords → MAMMAL world mm."""
    return points_3d / M5_DISTANCE_SCALE + M5_SCENE_CENTER
```

#### detect_keypoints_single (L91-129)

```python
# Input: MMPose model, image path, bbox [x1,y1,x2,y2]
# Process: inference_topdown() → pred_instances.keypoints[0]
# Output: (22, 3) ndarray — last channel = confidence score
```

#### triangulate_and_eval (L173-226) — 핵심 알고리즘

```
Input: keypoints_2d (num_views, 22, 3), proj_matrices (num_views, 3, 4),
       gt_3d_mm (22, 3)

1. triangulate_batch()       → pred_3d_fl (FaceLift normalized space)
2. facelift_to_mammal()      → pred_3d_mm (millimeters)
3. compute_mpjpe()           → MPJPE (mm)
4. compute_pa_mpjpe()        → Procrustes-aligned MPJPE
5. per-joint detection rate  → conf > threshold인 뷰 비율

Output: {pred_3d_mm, mpjpe, mpjpe_std, per_joint_error,
         pa_mpjpe, detection_rates, mean_detection_rate}
```

#### 22-Joint Definition (L315-322)

```
L_ear, R_ear, nose, neck, body_middle, tail_root, tail_middle, tail_end,
L_paw, L_paw_end, L_elbow, L_shoulder,
R_paw, R_paw_end, R_elbow, R_shoulder,
L_foot, L_knee, L_hip, R_foot, R_knee, R_hip
```

### 7.4 Commands

```bash
# Step 1: Render novel views (facelift env)
conda activate facelift
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.keypoint_detection.render_novel_views_for_detection \
    --config configs/base/gslrm_mouse.yaml \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt \
    --data_root ~/data/preprocessed/FaceLift_mouse/M5 \
    --output_dir outputs/triangulation/neural_detection/renders \
    --num_views 6 12 24 --render_size 384

# Step 2: Detect + Triangulate (mmpose env)
conda activate mmpose
python -m mouse_extensions.scripts.keypoint_detection.detect_and_triangulate \
    --render_dir outputs/triangulation/neural_detection/renders/24views \
    --mmpose_config mouse_extensions/configs/mmpose/hrnet_w48_mouse_22kp.py \
    --mmpose_checkpoint /node_data/joon/checkpoints/mmpose/hrnet_w48_mouse_best.pth \
    --output_dir outputs/triangulation/neural_detection/results/24views \
    --conf_threshold 0.3

# Step 3: Oracle vs Neural comparison
python -m mouse_extensions.scripts.keypoint_detection.compare_oracle_vs_real \
    --oracle_path outputs/triangulation/oracle_saturation/saturation_analysis.json \
    --neural_dir outputs/triangulation/neural_detection/results \
    --output_dir outputs/triangulation/neural_detection/comparison

# Step 4: View-count comparison visualization
python -m mouse_extensions.scripts.keypoint_detection.viewcount_comparison_viz \
    --results_dir outputs/triangulation/neural_detection/results \
    --oracle_path outputs/triangulation/oracle_saturation/saturation_analysis.json \
    --output_dir outputs/triangulation/neural_detection/comparison/viewcount
```

---

*← [PH6](PH6_FAIR_EVALUATION.md) | [Hub](../PIPELINE_DEEP_DIVE.md)*
