# Mesh-GS Pair Collection Pipeline v2.0

> **Navigation**: [← INDEX](../INDEX.md) | [EXPERIMENT_REGISTRY](EXPERIMENT_REGISTRY.md)
> **Purpose**: Novel view dataset collection — GS-LRM + MAMMAL mesh pairs, tier-based structure
> **Updated**: 2026-03-12 | **Version**: v2.0

---

## 1. Overview

3-tier dataset pipeline for NeurIPS 2026 Evaluations & Datasets Track:

| Tier | Source | Content | Benchmark Use |
|:----:|--------|---------|:-------------:|
| **Tier 0** | GS-LRM (6v GT input) | Raw novel view renders | Baseline |
| **Tier 1** | OpenCV cleanup | Artifact-removed renders | Primary GT |
| **Tier 2** | AI enhancement (Nano Banana/Gemini) | Qualitative improvement | Qualitative only |
| **Pseudo-GT** | MAMMAL mesh fitting | Template-based renders | Reference (not absolute GT) |

## 2. Why

GS-LRM은 6-view GT 입력 시 PSNR 23.84 dB로 우수하지만, novel view에서는 Gaussian artifact가 발생.
MAMMAL mesh는 동일 카메라에서 pseudo-GT를 제공하여 artifact 정량화 가능.
3-tier 구조로 raw → cleaned → enhanced 진행, 각 단계의 기여를 독립 측정.

## 3. Directory Structure (v2.0)

```
outputs/novel_view_dataset/
├── manifest.json                    # Dataset-level metadata
├── cameras/
│   └── novel_views.json             # Global novel view camera params (same all frames)
├── mouse_m5t2/                      # Species + dataset
│   ├── tier0_raw/                   # Raw GS-LRM novel view renders
│   │   ├── bottom/                  # -70° elevation
│   │   │   ├── 00000.png
│   │   │   └── ...
│   │   ├── top/                     # +70° elevation
│   │   ├── front_low/               # -30° elevation, 0° azimuth
│   │   └── side_low/                # -30° elevation, 90° azimuth
│   ├── pseudo_gt/                   # MAMMAL mesh renders at novel cameras
│   │   └── (same view structure)
│   ├── tier1_cleaned/               # Stage 1: OpenCV artifact removal (TODO)
│   ├── tier2_enhanced/              # Stage 2: AI enhancement (TODO)
│   ├── artifact_masks/              # Binary artifact masks (TODO)
│   ├── gt_views/                    # GS-LRM renders at GT camera poses
│   │   ├── cam_000/ ... cam_005/
│   ├── gt_rgb/                      # Original GT camera images
│   │   └── (same cam structure)
│   └── metadata/                    # Per-frame metadata JSON
│       ├── 00000.json ... NNNNN.json
├── splits/
│   ├── train.json                   # Frame 0-2879
│   ├── val.json                     # Frame 2880-3239
│   └── test.json                    # Frame 3240-3599
└── visualizations/                  # Comparison grids, videos
```

**Key design**: view-first organization (`tier0_raw/bottom/00000.png`) — easy to glob all frames for one view type.

## 4. Novel View Cameras

All frames share identical 4 novel view cameras:

| View | Elevation | Azimuth | Purpose |
|------|:---------:|:-------:|---------|
| `bottom` | -70° | 0° | Ventral view (belly, paws) |
| `top` | +70° | 0° | Dorsal view (back, ears) |
| `front_low` | -30° | 0° | Low frontal (face, whiskers) |
| `side_low` | -30° | 90° | Low lateral (profile, gait) |

- **Intrinsics**: fx=fy=411.75 @ 384×384 (scaled from GT 548.99@512)
- **Convention**: OpenCV (X-right, Y-down, Z-forward)
- **Radius**: 2.7 (turntable distance)

## 5. Coordinate Transform

```python
point_fl = (point_mm - M5_SCENE_CENTER) * M5_DISTANCE_SCALE
# M5_SCENE_CENTER = [59.672, 51.517, 107.099]  (mm)
# M5_DISTANCE_SCALE = 2.7 / 307.785 ≈ 0.008781
```

Frame mapping: M5 frame idx N → MAMMAL video frame N×5 (100fps→20fps)

> **UV texture**: Template mesh vertex expansion (14,522→~15,399) handled via face-level mapping.
> **✅ Resolved**: See [[../../mouse_extensions/docs/UV_TEXTURE_RENDERING_BUG]]

## 6. Usage

### Migration from PoC v0

```bash
# On gpu03, facelift env
python -m mouse_extensions.scripts.novel_view.collect_dataset \
    --mode migrate \
    --poc_dir outputs/poc_mesh_gs_pairs_v0_archive
```

### Fresh Generation

```bash
# Phase 1: GS-LRM renders (facelift env, needs GPU)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.novel_view.collect_dataset \
    --mode generate --phase gslrm \
    --frame_range 0 150

# Phase 2: MAMMAL pseudo-GT (mammal_stable env, EGL)
conda activate mammal_stable
PYOPENGL_PLATFORM=egl python -m mouse_extensions.scripts.novel_view.collect_dataset \
    --mode generate --phase mammal \
    --frame_range 0 150

# Rebuild manifest
python -m mouse_extensions.scripts.novel_view.collect_dataset --mode manifest

# Visualize comparison grids
python -m mouse_extensions.scripts.novel_view.collect_dataset \
    --mode visualize --frames 0 100 137
```

## 7. Metadata Schema

### Per-Frame (`metadata/NNNNN.json`)

```json
{
  "frame_idx": 137,
  "mammal_video_frame": 685,
  "split": "train",
  "species": "mouse",
  "dataset": "M5t2",
  "resolution": 384,
  "data_available": {
    "tier0_raw": true, "pseudo_gt": true,
    "gt_views": true, "gt_rgb": true,
    "tier1_cleaned": false, "tier2_enhanced": false,
    "artifact_masks": false
  },
  "gt_cameras": { "cam_000": { "c2w": [...], "fxfycxcy": [...] }, ... }
}
```

### Dataset Manifest (`manifest.json`)

Total frames, availability counts, pipeline versions, split boundaries, coordinate transform constants.

## 8. Data Scale

| Component | Count |
|-----------|:-----:|
| Total frames (M5t2) | 3,600 |
| GT camera views | 6 per frame = 21,600 |
| Novel views | 4 per frame = 14,400 |
| **Total pairs** | **36,000** |

## 9. M5t2 Splits

| Split | Frame Range | Count | Ratio |
|-------|:-----------:|:-----:|:-----:|
| Train | 0 ~ 2,879 | 2,880 | 80% |
| Val | 2,880 ~ 3,239 | 360 | 10% |
| Test | 3,240 ~ 3,599 | 360 | 10% |

## 10. History

| Version | Date | Change |
|:-------:|------|--------|
| v2.0 | 2026-03-12 | Tier-based structure, per-frame metadata, migration pipeline, view-first org |
| v1.0 | 2026-03-11 | PoC flat structure (poc_mesh_gs_pairs.py) |

## 11. Related Documents

- ↑ [[../INDEX]] — Document hub
- ↔ [[PHASE2_NOVEL_VIEW_ROADMAP]] — Phase 2 roadmap
- ↔ [[fl_vs_ps_comparison]] — FaceLift vs Pose-Splatter comparison
- ↔ [[../../mouse_extensions/docs/COORDINATE_SYSTEMS]] — Coordinate transforms
- ↔ [[../../mouse_extensions/docs/DATASET_FRAME_INDEXING]] — Frame indexing & data specs
- ↔ [[../../mouse_extensions/docs/UV_TEXTURE_RENDERING_BUG]] — UV texture bug fix
- ↔ [[DIFIX_STRATEGY]] — DiFix artifact removal strategy (uses this dataset as source)

---

## Dataset QA Viewer

> Consolidated from DATASET_QA_VIEWER.md (260331 audit)

### 1. Overview

Novel view dataset (3,600 frames)의 품질을 시각적으로 검수하는 **zero-dependency HTTP viewer**.

- **듀얼 모드**: Novel-view (4 synthetic views) / GT-view (6 physical cameras)
- **View Ablation**: GT-view 모드에서 1~6 input views 선택 가능
- **자동 감지**: 디렉토리 스캔으로 가용 tier 자동 표시
- **Export**: 프레임별 비교 이미지 PNG 다운로드
- **Exclude 관리**: 이상 프레임 태깅 + exclude_list.json 자동 저장
- **의존성**: Python 3 stdlib만 사용

### 2. Quick Start

```bash
# gpu03에서 서버 시작
ssh gpu03
cd /home/joon/dev/FaceLift
python -m mouse_extensions.scripts.novel_view.qa_viewer

# Mac에서 SSH 터널 + 브라우저
ssh -L 8899:localhost:8899 gpu03
open http://localhost:8899
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset_dir` | `outputs/datasets/novel_view` | Dataset root |
| `--port` | `8899` | HTTP server port |
| `--host` | `0.0.0.0` | Bind address |

### 3. Dual Mode

#### 3.1 Novel-view Mode (기본)

4개 synthetic viewpoint에서의 비교:

| Column | Source Dir | Description |
|--------|-----------|-------------|
| GS-LRM 6v | `tier0_raw/` | GS-LRM 6-view → novel view render |
| MAMMAL | `pseudo_gt/` | Mesh-based pseudo-GT (untextured) |
| MAMMAL (tex) | `pseudo_gt_textured/` | Mesh-based pseudo-GT (textured) |

- Views: `bottom` (-70°), `top` (+70°), `front_low` (-30°), `side_low` (-30°/90°)
- GT column 없음 (novel view에는 ground truth 없음)

#### 3.2 GT-view Mode

6개 physical camera 위치에서의 비교 + view ablation:

| Column | Source Dir | Description |
|--------|-----------|-------------|
| GT RGB | `gt_rgb/cam_000..005/` | Ground truth images |
| GS-LRM Nv | `ablation_Nview/` or `gt_views/` | N-view input ablation |

- Views: `cam_000` ~ `cam_005` (6 physical cameras)
- **View Ablation**: 버튼으로 1~6 input views 선택
  - N=6: `gt_views/` 사용
  - N=1~5: `ablation_Nview/` 사용

#### 3.3 자동 감지

서버 시작 시 데이터 디렉토리를 스캔하여 가용 tier만 표시. 존재하지 않는 디렉토리의 컬럼은 자동 숨김.

### 4. Features

#### 4.1 Navigation

| 동작 | 방법 |
|------|------|
| 페이지 이동 | `←`/`→` 키, Prev/Next 버튼 |
| 첫/끝 페이지 | `«`/`»` 버튼 |
| 페이지 직접 입력 | Page input |
| 프레임 직접 이동 | Frame input (binary search) |
| 페이지 크기 조절 | Per page input (1~24) |
| 이미지 확대 | 이미지 클릭 → zoom |
| 확대 닫기 | 클릭 또는 `Esc` |

#### 4.2 Exclude

1. 드롭다운에서 reason 선택 (6종: mesh_fitting_failure, gs_lrm_artifact 등)
2. Exclude 버튼 클릭 → 빨간 테두리 + 반투명
3. 다시 클릭하면 제외 해제
4. `exclude_list.json`에 자동 저장

#### 4.3 Export

- **Export 버튼**: 해당 프레임의 모든 이미지를 Canvas API로 합성 → PNG 다운로드
- 파일명: `qa_NNNNN_{mode}_{N}v.png`
- 외부 라이브러리 불필요 (순수 Canvas API)

### 5. Data Structure

```
outputs/datasets/novel_view/
├── exclude_list.json           # exclude 상태 (자동 저장)
└── mouse_m5t2/
    ├── metadata/               # 3,600 frame JSON files
    ├── gt_rgb/cam_000..005/    # GT RGB (6 cameras)
    ├── gt_views/cam_000..005/  # GS-LRM 6v at GT positions
    ├── tier0_raw/              # GS-LRM 6v at novel views
    │   ├── bottom/top/front_low/side_low/
    ├── pseudo_gt/              # MAMMAL untextured
    │   ├── bottom/top/front_low/side_low/
    ├── pseudo_gt_textured/     # MAMMAL textured (optional)
    │   ├── bottom/top/front_low/side_low/
    └── ablation_{1..5}view/    # View ablation at GT positions
        └── cam_000..005/
```

### 6. Textured MAMMAL 생성

`pseudo_gt_textured/`가 없는 경우, collect_dataset.py로 생성:

```bash
# gpu03에서 실행 (GPU 4~7만 사용)
cd /home/joon/dev/FaceLift
PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=7 \
python -m mouse_extensions.scripts.novel_view.collect_dataset \
    --mode generate --phase mammal --use-texture \
    --frame_range 0 3600
```

> **텍스처 파일 위치**: `MAMMAL_TEXTURED_OBJ` → `/home/joon/dev/MAMMAL_mouse/exports/mouse_frame0_textured.obj`, `MAMMAL_TEXTURE_PNG` → `/home/joon/dev/MAMMAL_mouse/exports/texture_final.png`.
> Resume 지원: 기존 프레임은 자동 건너뜀. 강제 재생성은 `--force` 추가.
> 현재 상태: **3,600 프레임 × 4 views 생성 완료** (2026-03-13).

### 7. 1-View GS-LRM 가능 여부

GS-LRM은 코드상 **N=1부터 지원**합니다. Ablation pipeline에서 first-N slicing:

```python
images_n = images[:, :n_views]  # n_views=1 가능
```

단, 1-view 입력의 reconstruction 품질은 매우 낮을 수 있음 (depth ambiguity).

### 8. Troubleshooting

| 증상 | 원인 | 해결 |
|------|------|------|
| 이미지 투명 | 해당 tier/view에 이미지 미생성 | `ls`로 파일 확인 |
| 빈 페이지 | metadata/ 없음 | dataset 생성 완료 확인 |
| 컬럼 안 보임 | 해당 tier 디렉토리 미존재 | 데이터 생성 후 서버 재시작 |
| Export 실패 | CORS 문제 | 같은 서버에서 접속 확인 |

### 9. Code Reference

| Component | File |
|-----------|------|
| QA Viewer Server | `mouse_extensions/scripts/novel_view/qa_viewer.py` |
| Dataset Collection | `mouse_extensions/scripts/novel_view/collect_dataset.py` |
| MAMMAL Rendering | [MAMMAL_MESH_RENDERING_PIPELINE](../theory/MAMMAL_MESH_RENDERING_PIPELINE.md) |

---

*FaceLift | Mesh-GS Pair Collection Pipeline | 2026-03-12*
