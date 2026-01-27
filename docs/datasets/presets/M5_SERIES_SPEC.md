# M5 Series Specification

> **Created**: 2026-01-28
> **Status**: Active
> **Core Change**: Re-centered cameras + Uniform distance normalization

---

## 1. 핵심 변경사항

### 1.1 문제 (M1~M4의 공통 결함)

모든 이전 M-series는 **per-view distance normalization** 사용:
```python
# preprocess.py (기존 방식)
dist_scale = target_distance / np.linalg.norm(cam_pos)  # Per-view!
```

이로 인해:
- 모든 카메라가 정확히 dist=2.7에 배치 (원본 거리 차이 무시)
- 카메라 리그 형상 **최대 34.7% 왜곡** (baseline 비율 파괴)
- Camera centroid가 원점에서 0.96 offset (pretrained 분포와 불일치)

### 1.2 해결 (M5)

$$\bar{t} = \frac{1}{N}\sum_{i} C_{w,i} \quad \text{(centroid)}$$
$$C'_{w,i} = C_{w,i} - \bar{t} \quad \text{(re-center)}$$
$$s = \frac{d_{target}}{\frac{1}{N}\sum_i ||C'_{w,i}||} \quad \text{(uniform scale)}$$
$$C''_{w,i} = s \cdot C'_{w,i} \quad \text{(final position)}$$

결과:
- Centroid offset: 0.96 → **0.00**
- Mean distance: **2.70** (pretrained 호환)
- Individual distances: **자연 변동 보존** (2.31~3.21)
- Baseline 비율: **원본과 동일**

### 1.3 안전성

| 항목 | 변경? | 이유 |
|------|-------|------|
| 이미지 픽셀 | No | 이미지 변환은 M1-M4와 동일 |
| fx, fy | No | 549 유지 |
| cx, cy | No | 256 유지 (PP ray 정확) |
| Camera rotation | No | 방향 불변 |
| Camera position | **Yes** | centroid→origin + uniform scale |
| Ray 방향 | No | K^{-1}[u,v,1] 불변 |

---

## 2. M5 Variants

| Preset | Image Transform | Zoom | 용도 |
|--------|----------------|------|------|
| **M5** | Affine (D7.1) | 없음 | Affine baseline + re-centering |
| **M5h** | Homography (D8) | 없음 | Homography baseline + re-centering |
| **M5h_1** | Homography | Global [1.0, 1.8] | M3_1 + re-centering |
| **M5h_2** ★ | Homography | Per-sample [1.0, 1.5] | **권장**: M3_2b + re-centering |

### 공통 설정

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| recenter_cameras | True | M5 핵심 |
| target_distance | 2.7 | Uniform (mean) |
| target_fx | 549 | Pretrained 호환 |
| image_size | 512 | |
| pp_method | shift_to_256 | PP=256 보장 |
| zoom_center_mode | image | PP=256 유지 |

---

## 3. Evolution History

```
M1 (Affine)
 └→ M2 (Homography)
     └→ M3 (+ Adaptive Zoom)
         ├→ M3_1 (Global zoom)
         ├→ M3_2 (Per-sample zoom) ← 이전 권장
         ├→ M3_2b (Conservative zoom)
         └→ M3_3 (Safe zoom, 0% clip)
     └→ M4 (+ Safety Margin)
         ├→ M4_1 (Clipping-safe)
         ├→ M4_2 (Coverage test)
         └→ M4_3 (Global baseline)
     └→ M5 (+ Re-centering + Uniform Norm) ★ NEW
         ├→ M5 (Affine, no zoom)
         ├→ M5h (Homography, no zoom)
         ├→ M5h_1 (Homography + Global zoom)
         └→ M5h_2 (Homography + Per-sample zoom) ← 현재 권장
```

**M5 핵심 차별점**: 이미지 변환은 M3/M4와 동일하되, **카메라 정규화만 교체**
- Per-view dist norm → Re-center + Uniform dist norm
- centroid offset 0.96 → 0.00

---

## 4. 실험 계획

| 실험 | Dataset | 비교 대상 | 검증 목표 |
|------|---------|-----------|-----------|
| E1 | M5h_2 전처리 | M3_2b | Re-centering 효과 (ghosting 감소?) |
| E2 | M5h 전처리 | M2 (D8) | Zoom 없이 순수 re-centering 효과 |
| E3 | Synthetic 렌더링 | - | 완벽한 카메라에서 GS-LRM 동작 확인 |

### E1 실행 (실제 데이터 전처리)

```bash
cd /home/joon/dev/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh && conda activate facelift

python -m mouse_extensions.preprocessing.preprocess \
    --preset M5h_2 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M5h_2
```

### E3 실행 (Synthetic 렌더링)

**Step 1: UV Transplant** (fitting OBJ에 텍스처 UV 이식)

```bash
cd /home/joon/dev/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh && conda activate facelift

python mouse_extensions/scripts/blender/transplant_uv.py \
    --fitting-dir /home/joon/dev/MAMMAL_mouse/results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260126_025249/obj/ \
    --reference /home/joon/dev/MAMMAL_mouse/exports/mouse_frame0_textured.obj \
    --output-dir /home/joon/data/synthetic/textured_obj/ \
    --max-frames 100 --frame-step 24
```

**Step 2: Blender 렌더링** (32-view orbit, textured)

```bash
CUDA_VISIBLE_DEVICES=7 /home/joon/blender-4.0.2-linux-x64/blender --background \
    --python mouse_extensions/scripts/blender/render_mammal_32view_v2.py -- \
    --experiment MAMMAL_CENTER \
    --output_dir /home/joon/data/synthetic/MAMMAL_CENTER_TEXTURED \
    --mesh /home/joon/data/synthetic/textured_obj/step_2_frame_000000.obj \
    --texture /home/joon/dev/MAMMAL_mouse/results/sweep/run_wild-sweep-9/texture_final.png \
    --num_views 32 \
    --num_samples 1
```

> **Note**: `--mesh`는 단일 mesh를 N회 반복 렌더링. 다중 프레임을 렌더하려면
> transplanted OBJ 폴더를 `--mammal_results`의 `obj/` 하위에 배치.

**Step 3: GS-LRM Inference** (렌더링 결과로 추론)

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate facelift

python inference_mouse.py \
    --sample_dir /home/joon/data/synthetic/MAMMAL_CENTER_TEXTURED/sample_00000 \
    --checkpoint checkpoints/gslrm/D8_E0_paper_original/ \
    --config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/e3_synthetic/ \
    --save_turntable --save_mesh
```

---

## 5. 코드 참조

| 파일 | 내용 |
|------|------|
| `mouse_extensions/preprocessing/presets.py` | M5 프리셋 정의 |
| `mouse_extensions/preprocessing/preprocess.py` | `_normalize_cameras_batch()` 구현 |
| `docs/datasets/presets/M5_SERIES_SPEC.md` | 이 문서 |

---

## 6. Execution Commands

### Basic Format

```bash
cd /home/joon/dev/FaceLift

python -m mouse_extensions.preprocessing.preprocess \
    --preset <PRESET> \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/<PRESET>
```

### M5 Series

```bash
# M5: Affine + re-centering (baseline)
python -m mouse_extensions.preprocessing.preprocess --preset M5 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M5

# M5h: Homography + re-centering (no zoom)
python -m mouse_extensions.preprocessing.preprocess --preset M5h \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M5h

# M5h_1: Homography + re-centering + global zoom
python -m mouse_extensions.preprocessing.preprocess --preset M5h_1 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M5h_1

# M5h_2: Homography + re-centering + per-sample zoom (RECOMMENDED)
python -m mouse_extensions.preprocessing.preprocess --preset M5h_2 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M5h_2
```

### Split Settings

| Parameter | Default | Note |
|-----------|---------|------|
| `--val-ratio` | 0.1 | 90% train / 10% val (same as M3) |
| `--frame-interval` | 1 | Frame interval |
| `--max-samples` | None | Use all |

Split uses same logic as M3 series (random 10% validation).
Outputs: `data_mouse_train.txt`, `data_mouse_val.txt` auto-generated.

---

## 7. Naming Convention

| Pattern | Meaning |
|---------|---------|
| **M5** | Affine (simpler, consistent with M1=Affine) |
| **M5h** | Homography (suffix h = more complex) |
| **M5_N** | Affine sub-variant |
| **M5h_N** | Homography sub-variant |

*M5 Series | Updated: 2026-01-28 | Per-view norm bug fix + re-centering*
