# Dataset QA Viewer

> **Navigation**: [← INDEX](../INDEX.md) | [mesh_gs_pair_collection](mesh_gs_pair_collection.md)
> **Purpose**: Novel view dataset 품질 검수 (QA) 도구 사용 매뉴얼
> **Updated**: 2026-03-13 | **Version**: v2.0

---

## 1. Overview

Novel view dataset (3,600 frames)의 품질을 시각적으로 검수하는 **zero-dependency HTTP viewer**.

- **듀얼 모드**: Novel-view (4 synthetic views) / GT-view (6 physical cameras)
- **View Ablation**: GT-view 모드에서 1~6 input views 선택 가능
- **자동 감지**: 디렉토리 스캔으로 가용 tier 자동 표시
- **Export**: 프레임별 비교 이미지 PNG 다운로드
- **Exclude 관리**: 이상 프레임 태깅 + exclude_list.json 자동 저장
- **의존성**: Python 3 stdlib만 사용

---

## 2. Quick Start

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

---

## 3. Dual Mode

### 3.1 Novel-view Mode (기본)

4개 synthetic viewpoint에서의 비교:

| Column | Source Dir | Description |
|--------|-----------|-------------|
| GS-LRM 6v | `tier0_raw/` | GS-LRM 6-view → novel view render |
| MAMMAL | `pseudo_gt/` | Mesh-based pseudo-GT (untextured) |
| MAMMAL (tex) | `pseudo_gt_textured/` | Mesh-based pseudo-GT (textured) |

- Views: `bottom` (-70°), `top` (+70°), `front_low` (-30°), `side_low` (-30°/90°)
- GT column 없음 (novel view에는 ground truth 없음)

### 3.2 GT-view Mode

6개 physical camera 위치에서의 비교 + view ablation:

| Column | Source Dir | Description |
|--------|-----------|-------------|
| GT RGB | `gt_rgb/cam_000..005/` | Ground truth images |
| GS-LRM Nv | `ablation_Nview/` or `gt_views/` | N-view input ablation |

- Views: `cam_000` ~ `cam_005` (6 physical cameras)
- **View Ablation**: 버튼으로 1~6 input views 선택
  - N=6: `gt_views/` 사용
  - N=1~5: `ablation_Nview/` 사용

### 3.3 자동 감지

서버 시작 시 데이터 디렉토리를 스캔하여 가용 tier만 표시. 존재하지 않는 디렉토리의 컬럼은 자동 숨김.

---

## 4. Features

### 4.1 Navigation

| 동작 | 방법 |
|------|------|
| 페이지 이동 | `←`/`→` 키, Prev/Next 버튼 |
| 첫/끝 페이지 | `«`/`»` 버튼 |
| 페이지 직접 입력 | Page input |
| 프레임 직접 이동 | Frame input (binary search) |
| 페이지 크기 조절 | Per page input (1~24) |
| 이미지 확대 | 이미지 클릭 → zoom |
| 확대 닫기 | 클릭 또는 `Esc` |

### 4.2 Exclude

1. 드롭다운에서 reason 선택 (6종: mesh_fitting_failure, gs_lrm_artifact 등)
2. Exclude 버튼 클릭 → 빨간 테두리 + 반투명
3. 다시 클릭하면 제외 해제
4. `exclude_list.json`에 자동 저장

### 4.3 Export

- **Export 버튼**: 해당 프레임의 모든 이미지를 Canvas API로 합성 → PNG 다운로드
- 파일명: `qa_NNNNN_{mode}_{N}v.png`
- 외부 라이브러리 불필요 (순수 Canvas API)

---

## 5. Data Structure

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

---

## 6. Textured MAMMAL 생성

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

---

## 7. 1-View GS-LRM 가능 여부

GS-LRM은 코드상 **N=1부터 지원**합니다. Ablation pipeline에서 first-N slicing:

```python
images_n = images[:, :n_views]  # n_views=1 가능
```

단, 1-view 입력의 reconstruction 품질은 매우 낮을 수 있음 (depth ambiguity).

---

## 8. Troubleshooting

| 증상 | 원인 | 해결 |
|------|------|------|
| 이미지 투명 | 해당 tier/view에 이미지 미생성 | `ls`로 파일 확인 |
| 빈 페이지 | metadata/ 없음 | dataset 생성 완료 확인 |
| 컬럼 안 보임 | 해당 tier 디렉토리 미존재 | 데이터 생성 후 서버 재시작 |
| Export 실패 | CORS 문제 | 같은 서버에서 접속 확인 |

---

## 9. Code Reference

| Component | File |
|-----------|------|
| QA Viewer Server | `mouse_extensions/scripts/novel_view/qa_viewer.py` |
| Dataset Collection | `mouse_extensions/scripts/novel_view/collect_dataset.py` |
| MAMMAL Rendering | [MAMMAL_MESH_RENDERING_PIPELINE](../theory/MAMMAL_MESH_RENDERING_PIPELINE.md) |

---

*FaceLift | Dataset QA Viewer Manual v2.0 | 2026-03-13*
