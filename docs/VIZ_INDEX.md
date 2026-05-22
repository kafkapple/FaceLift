# Visualization Index — FaceLift Cinematic Pipeline

> **Top-level navigation**. 코드/설정/출력/문서 한눈에. (2026-05-22 cleanup 후)

## 🎬 Cinematic (production demo videos)

| 항목 | 위치 |
|---|---|
| **Code (entry)** | `mouse_extensions/behavior/cinematic_sequence.py` (1960L+, no-split) |
| **Configs** | `configs/mouse/cinematic/` (7 yamls — `_base + v11 + 3 FINAL + paper + default`) |
| **Spec** | `docs/specs/CINEMATIC_V11_SPEC.md` (v13 current) |
| **Local outputs** | `~/results/FaceLift/cinematic/{preset}/` |
| **gpu03 outputs** | `/node_data/joon/cinematic_repro/{preset}/` |

### Run
```bash
# v11 production (768px/20fps/~100s, live ckpt)
CUDA_VISIBLE_DEVICES=N python -m mouse_extensions.behavior.cinematic_sequence \
    --config configs/mouse/cinematic/cinematic_v11.yaml \
    --output-dir /node_data/joon/cinematic_repro/v11 \
    --use-cache --save-segments

# Paper PLY (α=0.3, ckpt-independent)  →  cinematic_v11_FINAL_alpha03_16k.yaml
# Smoke/dev (512px/15fps)               →  cinematic_default.yaml
```

## 🔬 Sub-modules (specialized viz)

| 모듈 | 용도 |
|---|---|
| `mouse_extensions/visualization/` | core (turntable_renderer, keypoint_overlay, alpha_visualization, camera_utils) |
| `mouse_extensions/behavior/render_bodypart_gaussians.py` | body part Gaussian filtering viz |
| `mouse_extensions/behavior/generate_6view_grid.py` | 6-cam grid (GT vs render) |
| `mouse_extensions/behavior/visualize_clusters.py` | behavior cluster viz |
| `mouse_extensions/scripts/render_camera_follow.py` | face-follow standalone runner |
| `mouse_extensions/analysis/{saturation,triangulation}_visualizer.py` | analysis-side viz |
| `mouse_extensions/evaluation/temporal_visualizer.py` | temporal evaluation viz |
| `mouse_extensions/reports/coordinate_report_generator.py` | coordinate report w/ figures |
| `gslrm/model/gaussians_renderer.py` | low-level Gaussian render core |

## 🎨 Other configs

| 위치 | 내용 |
|---|---|
| `configs/visualization/` | turntable_6x6.yaml, turntable_video.yaml (standalone turntable) |

## 📚 Reference docs

| 문서 | 내용 |
|---|---|
| `docs/specs/CINEMATIC_V11_SPEC.md` | cinematic v11 production spec (current) |
| `docs/guides/VISUALIZATION_GUIDE.md` | 시각화 일반 가이드 (segment types, yaml options) |
| `mouse_extensions/docs/COORDINATE_SYSTEMS.md` | coord SSOT (MAMMAL ↔ FL ↔ PS) |
| `mouse_extensions/coordinate_utils.py` | M5_SCENE_CENTER, M5_DISTANCE_SCALE, transform helpers |

## 🧹 Archive locations

- `~/results/FaceLift/cinematic/_archive/` — local iteration mp4 (v3-v10, quick_test 등)
- `~/results/FaceLift/cinematic/_originals/` — Mar 30/31 원본 영상
- `/node_data/joon/cinematic_repro/_archive/` — gpu03 segment caches + iteration runs
- `~/dev/BehaviorSplatter/outputs/inference/_archive/` — BS cinematic_fix1-5 iterations

## 🔑 Cinematic Naming Convention

자동 생성: `cinematic_{config_stem}_{res}px_{fps}fps_f{start}-{end}.mp4`
- 예: `cinematic_v11_768px_20fps_f0-3500.mp4`
- 코드 line 1931 (`_seg_*` index), `--output-dir` CLI로만 위치 결정

---

*Engram | VIZ_INDEX.md | 2026-05-15 | TopLevel viz navigation*
