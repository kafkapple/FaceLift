# Pipeline Deep Dive: mouse_extensions Code Walkthrough

> **Navigation**: [← INDEX](../INDEX.md) | [EXPERIMENT_MASTER_GUIDE](EXPERIMENT_MASTER_GUIDE.md) | [CH1](chapters/CH1_ENVIRONMENT_AND_DATA.md) | [CH2](chapters/CH2_GSLRM_CODE_FLOW.md)
>
> **Related**: [KEYPOINT_3D_PIPELINE](../KEYPOINT_3D_PIPELINE.md) | [MVDIFFUSION_FINETUNE_GUIDE](MVDIFFUSION_FINETUNE_GUIDE.md)

**Created**: 2026-01-22
**Updated**: 2026-03-05
**Project**: FaceLift Mouse
**Server**: ssh gpu03 (`/home/joon/dev/FaceLift`)
**Status**: Reference Document (SSOT)

> **v3.0 Changes (2026-03-05)**: MoC 분리 — 7개 Phase를 `pipeline/` 서브디렉토리로 분리. Hub에는 Overview + Index + Cross-phase 참조만 유지.

---

## Overview

이 문서는 FaceLift Mouse 프로젝트의 전체 파이프라인을 **실제 코드 파일과 라인 번호**와 함께 상세히 설명합니다.
모든 경로는 `/home/joon/dev/FaceLift/` 기준입니다.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         FACELIFT MOUSE PIPELINE                              │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬───────────┤
│ PHASE 1  │ PHASE 2  │ PHASE 3  │ PHASE 4  │ PHASE 5  │ PHASE 6  │ PHASE 7  │
│ Preproc  │ Training │ Forward  │ Pose     │ E2E      │ Fair     │ 3D KP    │
│          │ Data     │ & Loss   │ Cond.    │ Inference│ Eval     │ Pipeline │
│          │          │          │          │          │          │          │
│ Raw →    │ Dataset  │ GSLRM    │ Plucker  │ 1 img →  │ FL vs PS │ Render → │
│ M5 fmt   │ → Batch  │ Forward  │ Spatial  │ 6 views  │ metrics  │ Detect → │
│          │          │ → Loss   │ Token    │ → 3DGS   │          │ Triang.  │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴───────────┘
```

---

## Phase Index

| Phase | Title | 핵심 파일 | 요약 |
|:-----:|-------|-----------|------|
| [**PH1**](pipeline/PH1_PREPROCESSING.md) | Data Preprocessing | `preprocess.py`, `presets.py` | Raw 1152×1024 → M5 512×512 RGBA, 카메라 정규화 |
| [**PH2**](pipeline/PH2_DATA_LOADING.md) | Training Data Loading | `mouse_dataset.py` | Dataset → view selection → batch dict |
| [**PH3**](pipeline/PH3_MODEL_FORWARD_LOSS.md) | Model Forward & Loss | `gslrm.py`, `loss_extensions.py` | Transformer → Gaussian → Render → L2+SSIM+Alpha |
| [**PH4**](pipeline/PH4_POSE_CONDITIONING.md) | Pose Conditioning | `pose_conditioning.py`, `pose_conditioning_integration.py` | Plucker ray → UNet injection (add/spatial_token) |
| [**PH5**](pipeline/PH5_E2E_INFERENCE.md) | E2E Inference | `end_to_end.py`, `run_e2e_inference.py` | 1 image → MVDiff 6-view → GS-LRM 3DGS |
| [**PH6**](pipeline/PH6_FAIR_EVALUATION.md) | Fair Evaluation | `fair_comparison.py` | FL vs PS: test-only, GT mask, unified metrics |
| [**PH7**](pipeline/PH7_3D_KEYPOINT.md) | 3D Keypoint Pipeline | `render_novel_views_for_detection.py`, `detect_and_triangulate.py` | Render → HRNet 2D → DLT triangulation → MPJPE |

---

## Potential Issues & Error Points

### ⚠️ Issue 1: Dataset Path Format

**Location**: `mouse_dataset.py:60-70`

```
# Bad:  relative paths in data_mouse_train.txt
# Good: absolute paths (/home/joon/data/.../sample_000000)
```

### ⚠️ Issue 2: Camera Normalization Mismatch

**Location**: `preprocess.py` vs `mouse_dataset.py`

```
Preprocessing (M5):     fx=548.99, distance→2.7
Runtime normalization:  distance→2.7 (fx도 스케일링)

★ Double normalization risk:
  - 전처리에서 이미 2.7로 정규화
  - 런타임에서 다시 정규화 시도
  - 해결: target_camera_distance=0 (비활성화) 또는 2.7 (idempotent)
```

### ⚠️ Issue 3: E2E Camera-Image Mismatch

**Location**: `mouse_extensions/inference/mvdiffusion_pipeline.py`

```
문제: MVDiffusion의 compute_cameras()가 합성 orbit 카메라 생성 (elevation=0°)
실제: M5 6대 카메라 elevation -75°~+81°, 비정규 azimuth
결과: GS-LRM render 100% 흰색 (카메라-이미지 불일치)

해결: cameras/m5_cameras.json에서 실제 카메라 로드 + --camera_json CLI
★ MVDiffusion은 카메라를 명시적으로 입력받지 않음 (prompt embedding 뷰 구분)
★ GS-LRM에 넘기는 카메라는 학습 시 실제 카메라와 일치 필수
```

### ⚠️ Issue 4: View 5 Elevation Outlier

```
View 0: +14.9°, View 1: +20.6°, View 2: +11.3°
View 3: +10.7°, View 4: +26.5°, View 5: +30.8° ⚠️ OUTLIER (z-score=1.54)

해결: E4_6 config로 View 5 제외 실험
  exclude_camera_indices: [5]
```

### ⚠️ Issue 5: bf16 + Log Operation NaN

**Location**: `loss_extensions.py:388`

```python
# Problem: entropy = -opacity * torch.log(opacity)  # bf16 NaN
# Solution:
opacity_f = opacity.float()
entropy = -opacity_f * torch.log(opacity_f.clamp(1e-4, 1-1e-4))
```

### ⚠️ Issue 6: E2E Path Confusion

**Location**: `run_e2e_inference.py:312-317`

```
★ --input_view_idx 누락 시 Path 1 (GS-LRM only)로 분기.
  → PSNR_gt ~20+ dB (GT input) → E2E로 오인하기 쉬움.
  → 반드시 run_config.json의 args.input_view_idx 확인.
```

### ⚠️ Issue 7: Pose Injector Weights 미저장 버그 (수정 완료)

```
2026-02-27 이전: pose_injector weights가 checkpoint에 저장되지 않음
  → 추론 시 random weights 사용 → E2E 성능 저하
수정: train_diffusion.py, mvdiffusion_pipeline.py에 save/load 추가
영향 파일: end_to_end.py, run_e2e_inference.py
```

### ⚠️ Issue 8: mmpose vs facelift Conda Env

```
render_novel_views_for_detection.py → facelift env (PyTorch + diff_gauss)
detect_and_triangulate.py           → mmpose env (mmpose + mmdet)

★ 두 env 혼용 시 import 에러. Step별로 env 전환 필수.
```

---

## Quick Reference Commands

### Preprocessing
```bash
python -m mouse_extensions.preprocessing.preprocess --preset M5 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M5
```

### GS-LRM Training
```bash
CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5 -e E0_1
```

### MVDiffusion Training
```bash
CUDA_VISIBLE_DEVICES=5 accelerate launch train_diffusion.py \
    --config configs/mvdiffusion/mouse_M5t2.yaml \
    --pose_config configs/experiments/H7v2_plucker_spatial.yaml
```

### E2E Inference (batch)
```bash
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt --input_view_idx 0 --skip_preprocess \
    --model M5t --gslrm_config configs/base/gslrm_mouse.yaml \
    --gslrm_checkpoint <GS-LRM_CKPT> --mvdiffusion_checkpoint <MVDIFF_CKPT> \
    --mvdiffusion_base checkpoints/mvdiffusion/pipeckpts \
    --prompt_embed_path mvdiffusion/data/mouse_prompt_embeds_6view_1024 \
    --prefer_ema --no_turntable --no_mesh \
    --num_steps 50 --guidance_scale 3.0 --seed 42 \
    --output_dir outputs/phase3_e2e/<name>
```

### Fair Evaluation
```bash
python mouse_extensions/scripts/eval/fair_comparison.py evaluate_fl \
    --render_dir outputs/phase3_e2e/<exp>/samples \
    --gt_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --output outputs/phase3_e2e/<exp>/fair_eval.json
```

### 3D Keypoint Pipeline
```bash
# Step 1 (facelift env): Render novel views
python -m mouse_extensions.scripts.keypoint_detection.render_novel_views_for_detection \
    --config configs/base/gslrm_mouse.yaml --checkpoint <CKPT> \
    --num_views 6 12 24 --output_dir outputs/triangulation/.../renders

# Step 2 (mmpose env): Detect + Triangulate
python -m mouse_extensions.scripts.keypoint_detection.detect_and_triangulate \
    --render_dir outputs/triangulation/.../renders/24views \
    --mmpose_config mouse_extensions/configs/mmpose/hrnet_w48_mouse_22kp.py \
    --mmpose_checkpoint <MMPOSE_CKPT> \
    --output_dir outputs/triangulation/.../results/24views
```

---

## File Summary

| File | Location | Lines | Purpose |
|------|----------|:-----:|---------|
| **Phase 1: Preprocessing** | | | |
| `preprocess.py` | `mouse_extensions/preprocessing/` | ~400 | Main preprocessing |
| `presets.py` | `mouse_extensions/preprocessing/` | ~150 | Preset definitions (M5, M5h) |
| `camera_normalizer.py` | `mouse_extensions/preprocessing/` | ~200 | Camera normalization |
| **Phase 2: Data Loading** | | | |
| `mouse_dataset.py` | `gslrm/data/` | ~560 | Training dataset class |
| `preprocessing.py` | `mouse_extensions/data/` | ~250 | Runtime camera utilities |
| **Phase 3: Model & Loss** | | | |
| `gslrm.py` | `gslrm/model/` | ~2300 | Main model (forward + loss) |
| `loss_extensions.py` | `mouse_extensions/model/` | ~830 | Mask, alpha, opacity, depth loss |
| `train_gslrm.py` | `/` | ~1100 | Training entry point |
| **Phase 4: Pose Conditioning** | | | |
| `pose_conditioning.py` | `mouse_extensions/model/` | ~360 | 3 encoder architectures |
| `pose_conditioning_integration.py` | `mouse_extensions/model/` | ~540 | UNet injector + M5 cameras |
| **Phase 5: E2E Inference** | | | |
| `end_to_end.py` | `mouse_extensions/inference/` | ~310 | E2E pipeline coordinator |
| `mvdiffusion_pipeline.py` | `mouse_extensions/inference/` | ~210 | MVDiffusion stage |
| `gslrm_pipeline.py` | `mouse_extensions/inference/` | ~430 | GS-LRM stage + outputs |
| `run_e2e_inference.py` | `mouse_extensions/scripts/inference/` | ~580 | CLI entry point |
| `m5_cameras.json` | `mouse_extensions/inference/cameras/` | - | M5 6-camera config |
| **Phase 6: Fair Evaluation** | | | |
| `fair_comparison.py` | `mouse_extensions/scripts/eval/` | ~760 | Fair comparison metrics |
| **Phase 7: 3D Keypoint** | | | |
| `render_novel_views_for_detection.py` | `mouse_extensions/scripts/keypoint_detection/` | ~300 | GS-LRM → turntable render |
| `detect_and_triangulate.py` | `mouse_extensions/scripts/keypoint_detection/` | ~400 | HRNet detect → triangulate |
| `compare_oracle_vs_real.py` | `mouse_extensions/scripts/keypoint_detection/` | ~350 | Oracle vs neural comparison |
| `viewcount_comparison_viz.py` | `mouse_extensions/scripts/keypoint_detection/` | ~380 | View-count comparison viz |

---

*FaceLift Mouse Project | Pipeline Deep Dive v3.0 | 2026-03-05*
