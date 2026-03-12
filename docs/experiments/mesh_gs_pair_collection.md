# Mesh–GS Image Pair Collection Strategy

> **Navigation**: [← INDEX](../INDEX.md) | [EXPERIMENT_REGISTRY](EXPERIMENT_REGISTRY.md)
> **Purpose**: MAMMAL mesh rendering과 GS-LRM 3D Gaussian prediction을 동일 카메라에서 비교하는 image pair dataset 수집 전략

---

## 1. 목표 (What)

동일한 카메라 pose에서 촬영한 **MAMMAL mesh rendering ↔ GS-LRM prediction** image pair를 대량 수집하여, 향후 mesh → 3D Gaussian quality prediction 모델 학습에 활용.

## 2. 왜 필요한가 (Why)

GS-LRM은 feed-forward 방식으로 multi-view 이미지에서 3D Gaussian을 예측하지만, 출력의 기하학적 정확도를 정량화하기 어렵다. MAMMAL mesh fitting은 동일 데이터에 대한 template-based 3D reconstruction을 제공하므로, 두 결과를 동일 viewpoint에서 비교하면:

1. **기하학적 정확도 평가**: mesh silhouette vs GS-LRM silhouette (IoU)
2. **질감 재현 평가**: textured mesh vs GS rendering (PSNR, SSIM)
3. **학습 데이터 생성**: (mesh_render, gs_render, gt_rgb) triplet으로 quality estimator 학습

## 3. 파이프라인 (How)

```
Phase 1: GS-LRM Inference (facelift env, GPU)
│  M5 sample → 6-view input → GS-LRM → 3D Gaussians
│  → Render at GT cameras (6 views) + novel cameras (4 views)
│  → Save camera config JSON
│
Phase 2: MAMMAL Mesh Rendering (mammal_stable env, EGL)
│  Per-frame OBJ + UV texture template
│  → Transform MAMMAL→FaceLift space
│  → Render at identical cameras via pyrender
│
Phase 3: Comparison & Dataset Assembly
│  (gt_rgb, mammal_render, gslrm_render) × N frames × M views
│  → Metrics: PSNR, SSIM, IoU, L1
│  → Metadata JSON per sample
```

### 데이터 규모

| 구성 | 수량 |
|------|------|
| Total frames | 3,600 (M5 전체) |
| GT camera views | 6 per frame |
| Novel views | 4 per frame (bottom, top, front_low, side_low) |
| **GT pairs** | 3,600 × 6 = **21,600** |
| **Novel pairs** | 3,600 × 4 = **14,400** |
| **Total pairs** | **36,000** |

### UV 텍스처 렌더링

MAMMAL template mesh (14,522 vertices) + UV texture map (512×512)을 pyrender로 렌더링.
Per-frame OBJ의 vertex positions를 template의 UV-expanded topology (15,399 vertices)에 매핑하여 텍스처 적용.

> **⚠️ 주의**: trimesh의 OBJ 로딩 시 내부 vertex 재인덱싱이 발생.
> 수동 UV expansion 대신 trimesh native loading + face 대응 매핑을 사용해야 함.
> **✅ 해결 (2026-03-12)**: Before/After 검증 완료 — IoU(flat) 0.653→1.000, PSNR +1.67 dB
> 상세: [[../../mouse_extensions/docs/UV_TEXTURE_RENDERING_BUG]]

### 좌표 변환

```python
# MAMMAL world (mm) → FaceLift normalized space
point_fl = (point_mm - M5_SCENE_CENTER) * M5_DISTANCE_SCALE
# M5_SCENE_CENTER = [59.672, 51.517, 107.099]  (camera rig centroid)
# M5_DISTANCE_SCALE = 2.7 / 307.785 ≈ 0.008781
```

## 4. 출력 구조

```
outputs/poc_mesh_gs_pairs/
├── frame_{idx:05d}/         # Per-frame directory
│   ├── gt_rgb/              # GT images (6 views, 512×512 RGBA)
│   │   └── cam_{v:03d}.png
│   ├── gslrm_gt/            # GS-LRM at GT cameras
│   │   └── cam_{v:03d}.png
│   ├── gslrm_novel/         # GS-LRM at novel cameras
│   │   └── {view_name}.png
│   ├── mammal_gt/           # MAMMAL mesh at GT cameras (textured)
│   │   └── cam_{v:03d}.png
│   ├── mammal_novel/        # MAMMAL mesh at novel cameras
│   │   └── {view_name}.png
│   └── camera_config_frame_{idx:05d}.json
├── videos/{train,val,test}/ # Comparison videos
│   ├── comparison_cam{000,004}.mp4
│   └── fg_only/comparison_cam{000,004}.mp4
└── diagnostics/             # Debug images
```

## 5. 실행 명령어

```bash
# Phase 1: GS-LRM (facelift env, GPU 필요)
conda activate facelift
CUDA_VISIBLE_DEVICES=5 python poc_mesh_gs_pairs.py --phase gslrm \
    --frames $(seq 0 3599 | tr '\n' ' ')

# Phase 2: MAMMAL mesh (mammal_stable env, EGL)
conda activate mammal_stable
PYOPENGL_PLATFORM=egl python poc_mesh_gs_pairs.py --phase mammal \
    --frames $(seq 0 3599 | tr '\n' ' ')

# Phase 3: Comparison grids + videos
python poc_mesh_gs_pairs.py --phase compare --frames $(seq 0 3599 | tr '\n' ' ')
python poc_mesh_gs_pairs.py --phase video --split
```

## 6. M5t2 Split

| Split | Frame Range | Count | Ratio |
|-------|:-----------:|:-----:|:-----:|
| Train | 0 ~ 2,879 | 2,880 | 80% |
| Val | 2,880 ~ 3,239 | 360 | 10% |
| Test | 3,240 ~ 3,599 | 360 | 10% |

## 7. Related Documents

- ↑ [[../INDEX]] — Document hub
- ↔ [[FL_vs_PS_comparison]] — FaceLift vs Pose-Splatter comparison
- ↔ [[../../mouse_extensions/docs/COORDINATE_SYSTEMS]] — Coordinate transforms
- ↔ [[../../mouse_extensions/docs/DATASET_FRAME_INDEXING]] — Frame indexing & data specs

---

*FaceLift | Mesh–GS Pair Collection Strategy | 2026-03-11*
