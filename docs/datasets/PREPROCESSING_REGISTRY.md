# Preprocessing Registry (전처리 레지스트리)

> **Navigation**: [← Index](../INDEX.md) | [Commands](../experiments/COMMANDS.md)
> **SSOT**: 데이터셋 전처리 설정 중앙 관리
> **최종 업데이트**: 2026-01-29

---

## 1. 분류 체계

### 1.1 기하학적 변환 기준

| 카테고리 | 변환 | 특징 | 권장 |
|----------|------|------|------|
| **affine** | Affine | 회전, 스케일, 이동 | M5 ⭐ |
| **homography** | Homography | affine + skew 보정 | M5h |
| **homography_zoom** | Homography + Zoom | + coverage 최적화 | M5h_2 |
| **geometry_broken** | centering만 | PP 미보정 | ⛔ 사용 금지 |
| **experimental** | up_alignment | 93° 회전 문제 | ⚠️ |

### 1.2 VERSION_HIERARCHY

| 분류 | 프리셋 | 설명 |
|------|--------|------|
| **recentered_affine** | M5, M5_4, M5_5, M0, M0_n | 카메라 재정렬 + affine |
| **affine** | D7, D7.1, D7.2 | 기본 변환 |
| **homography** | D8, D8.1, D8.2, M5h | skew 보정 |
| **homography_zoom** | M3_1, M3_2, M5h_1, M5h_2 | + adaptive zoom |
| **geometry_broken** | D1, D4, D6-* | ⛔ PP 미보정 |

---

## 2. M-Series (권장)

### 2.1 M5 Series (Ablation Study)

> **상세**: [M5_SERIES_SPEC.md](./M5_SERIES_SPEC.md) - 정규화 방식 상세 비교

| Preset | PP Centering | Translation Norm | 설명 | 상태 |
|--------|--------------|------------------|------|------|
| **M5** | ✅ PP=256 | Batch Uniform | 기준선: centroid→origin + 동일 scale | ✅ **권장** |
| **M5_4** | ✅ PP=256 | None | Ablation: centering만, norm 없음 | 🔬 실험 |
| **M5_5** | ✅ PP=256 | Per-view | Ablation: centering + 개별 norm | 🔬 실험 |
| **M5h** | ✅ PP=256 | Batch Uniform | M5 + homography | ✅ |
| **M5h_1** | ✅ PP=256 | Batch Uniform | M5h + global zoom | ✅ |
| **M5h_2** | ✅ PP=256 | Batch Uniform | M5h + per-sample zoom | ✅ **권장** |

### 2.2 M0 Series (Baseline - No Centering)

| Preset | PP Centering | Translation Norm | 설명 | 상태 |
|--------|--------------|------------------|------|------|
| **M0** | ❌ Original | None | Raw baseline: 완전 원본 | 🔬 실험 |
| **M0_n** | ❌ Original | Per-view | Baseline + per-view norm | 🔬 실험 |

### 2.3 Legacy M-Series

| Alias | Preset | 카테고리 | PP | fx | 상태 |
|-------|--------|----------|-----|-----|------|
| **M1** | D7.1 | affine | 256 | 549 | ✅ 기준선 |
| **M2** | D8 | homography | 256 | 549 | ✅ 정밀 |
| **M3_1** | M3_1 | homography_zoom | 256 | 549 | ✅ MVG-correct |
| **M3_2** | M3_2 | homography_zoom | 256 | 549 | ✅ |

### 2.4 Zoom 전처리 (Adaptive Zoom)

#### Zoom Variants

| Preset | Transform | Zoom Scope | 설명 | 상태 |
|--------|-----------|------------|------|------|
| **M3_1** | Homography | Global | 전체 데이터 동일 zoom factor | ✅ |
| **M5h_1** | Homography | Global | M5 + global zoom | ✅ |
| **M5h_2** | Homography | Per-sample | **M5 + 샘플별 adaptive zoom** | ⭐ **권장** |

#### M5 (Affine) + Zoom이 없는 이유

**기술적 제약 아님, 설계 선택:**

| 측면 | 설명 |
|------|------|
| **코드 구조** | Zoom은 transform과 독립적 → 기술적으로 affine+zoom 가능 |
| **Ablation 목적** | M5는 최소 baseline (affine, no zoom) |
| **Zoom 시 정밀도** | Zoom은 기하학 오차를 증폭시킴 |
| **Skew 보정** | Affine은 skew 무시 (fx/fy 0.46% 차이) |

**이론적 근거:**


**권장 경로:**
- Zoom 불필요 시: **M5** (affine baseline)
- Zoom 필요 시: **M5h_2** (homography + per-sample zoom)

### 2.5 데이터셋 분류 요약

| 분류 | 데이터셋 | 용도 |
|------|----------|------|
| **기준선** | M5 | Ablation 비교 기준 |
| **Ablation** | M5_4, M5_5, M0, M0_n | 가설 검증 실험 |
| **비교용** | **M5t** | Pose Splatter 공정 비교 (1:1:1) |
| **확장 (선택)** | M5h, M5h_1, M5h_2 | Ablation 후 최적 조합에 적용 |

---

## 3. 카메라 정규화 모드 (중요!)

### 3.1 세 가지 모드

| 모드 | 설정 | 동작 | 기하학 |
|------|------|------|--------|
| **Batch Uniform** | `recenter_cameras=True` | centroid→origin + 동일 scale 적용 (avg=2.7) | ✅ 보존 |
| **Per-view** | `normalize_translation=True` | 각 카메라 개별 2.7로 스케일 | ⚠️ 왜곡 |
| **None** | 둘 다 False | raw translation 유지 | ✅ 보존 |

### 3.2 코드 동작 (preprocess.py)

```python
# recenter_cameras=True 일 때:
# 1. per-view normalize_translation 스킵됨 (skip_distance_norm=True)
# 2. 대신 _normalize_cameras_batch() 호출

def _normalize_cameras_batch(cam_params_list, target_distance=2.7):
    # Step 1: Centroid → Origin
    centroid = positions.mean(axis=0)
    centered_positions = positions - centroid

    # Step 2: Uniform Scale (SAME scale for ALL cameras)
    mean_dist = np.linalg.norm(centered_positions, axis=1).mean()
    scale = target_distance / mean_dist

    # Result: average distance = 2.7, but individual distances vary
    # Camera rig geometry is PRESERVED
```

### 3.3 프리셋별 정규화

| Preset | recenter_cameras | normalize_translation | **실제 동작** |
|--------|------------------|----------------------|--------------|
| **M5** | ✅ True | (무시됨) | Batch Uniform |
| **M5_4** | ❌ False | ❌ False | None |
| **M5_5** | ❌ False | ✅ True | Per-view |
| **M0** | ❌ False | ❌ False | None |
| **M0_n** | ❌ False | ✅ True | Per-view |

---

## 4. 2×3 Ablation Factorial Design

### 4.1 실험 매트릭스

|  | **No Norm** | **Per-view** | **Batch Uniform** |
|--|-------------|--------------|-------------------|
| **Center (PP=256)** | M5_4 | M5_5 | **M5** ✅ |
| **No Center** | M0 | M0_n | (N/A) |

### 4.2 각 프리셋의 목적

| Preset | 검증 내용 | 비교 대상 |
|--------|----------|----------|
| **M5** | 기준선 (Center + Batch Uniform) | - |
| **M5_4** | "Norm 없이 centering만으로 충분한가?" | M5 vs M5_4 |
| **M5_5** | "Batch 대신 Per-view norm도 가능한가?" | M5 vs M5_5 |
| **M0** | "완전 raw는 어디서 실패하는가?" | 실패 분석용 |
| **M0_n** | "Centering 없이 norm만으로 가능한가?" | M5 vs M0_n |

### 4.3 M5_4 vs M5_5 상세 비교

| 항목 | M5_4 | M5_5 |
|------|------|------|
| **PP Centering** | ✅ PP=256 | ✅ PP=256 |
| **Translation Norm** | ❌ None | ⚠️ Per-view (각 카메라 개별 2.7) |
| **기하학 보존** | ✅ 원본 거리 비율 유지 | ⚠️ 왜곡 (모든 카메라 동일 거리) |
| **검증 가설** | Norm 필요성 | Norm 방식 비교 |

### 4.4 가설 및 우선순위

| 순위 | 가설 | 비교 | 핵심 질문 |
|------|------|------|----------|
| **P1** | Centering 필수? | M5 vs M0_n | PP=256이 핵심인가? |
| **P2** | Batch vs Per-view? | M5 vs M5_5 | 기하학 보존이 중요한가? |
| **P3** | Norm 없이 center만? | M5 vs M5_4 | Pretrained 스케일 매칭 필수? |
| **P4** | Raw baseline | M0 | 어디서 실패하는가? |

### 4.5 예상 결과 시나리오

| 결과 패턴 | 의미 | 결론 |
|----------|------|------|
| M5 ≈ M5_5 >> M5_4 | Per-view도 충분, Norm이 핵심 | Norm 필수, 방식은 유연 |
| M5 >> M5_5 ≈ M5_4 | Batch Uniform이 핵심 | 기하학 보존 중요 |
| M5 ≈ M5_4 ≈ M5_5 | Centering만으로 충분 | Norm 불필요 |
| M5 >> M5_4, M5 >> M5_5 | 둘 다 필요 | 현재 M5 설정이 최적 |

### 4.6 기존 결과 (참고)

| 실험 | 결과 | 의미 |
|------|------|------|
| D3n (No Center + Uniform) | PSNR ~3 ⛔ | Centering 없이 실패 |
| M5 (Center + Batch Uniform) | PSNR ~23 ✅ | 현재 최고 |

**D3n 실패 원인 분석**: Uniform norm은 모든 카메라를 동일 비율로 스케일하지만, PP가 가변이면 광선 방향이 불일치하여 3D 재구성 실패.

---

## 5. Train/Val/Test Split

### 5.1 기본 Split (90/10)

전처리 시 자동 생성:
- `data_mouse_train.txt` (90%)
- `data_mouse_val.txt` (10%)

### 5.2 Temporal Split (Pose Splatter 호환)

**1:1:1 temporal consecutive split** - Pose Splatter 논문과 공정 비교용

```bash
# 방법 1: 전처리 시 자동 생성
python -m mouse_extensions.preprocessing.preprocess \
    --preset M0 --input-dir ... --output-dir .../M0 --temporal-variant
# 결과: M0/ (90/10) + M0t/ (1:1:1 temporal, symlinks)

# 방법 2: 기존 데이터에 추가
python -m mouse_extensions.preprocessing.split_generator \
    --dataset_dir /path/to/M0 --create-variant M0t
```

### 5.3 Temporal Variant 구조

```
M0/           ← 실제 데이터 + 90/10 split
├── 000000/
├── ...
├── data_mouse_train.txt (90%)
└── data_mouse_val.txt (10%)

M0t/          ← symlinks + 1:1:1 temporal
├── 000000 → ../M0/000000
├── ...
├── data_mouse_train.txt (33%)
├── data_mouse_val.txt (33%)
├── data_mouse_test.txt (34%)
└── split.json
```

---

## 6. 프리셋별 상세 설정

### 6.1 affine 계열

| Preset | transform | scale_mode | pp_method | 비고 |
|--------|-----------|------------|-----------|------|
| D7 | affine | fx_only | shift_to_256 | 기본 |
| **D7.1** | affine | individual | shift_to_256 | M1 |
| D7.2 | affine | average | shift_to_256 | 평균 스케일 |

### 6.2 homography 계열

| Preset | transform | skew_correction | pp_method | 비고 |
|--------|-----------|-----------------|-----------|------|
| **D8** | homography | ✅ | shift_to_256 | M2 |
| D8.1 | homography | ✅ | shift_to_256 | + zoom |

### 6.3 homography_zoom 계열

| Preset | zoom_scope | zoom_center_mode | PP | 비고 |
|--------|------------|------------------|-----|------|
| **M3_1** | global | image | 256 | MVG-correct |
| **M3_2** | per_sample | image | 256 | ✅ |
| **M5h_1** | global | image | 256 | M5 + homography |
| **M5h_2** | per_sample | image | 256 | ⭐ 권장 |

### 6.4 geometry_broken (⛔ 사용 금지)

| Preset | 문제점 |
|--------|--------|
| D1 | centering만, PP 미보정 → ray error |
| D4 | PP=256 강제 → 37px 오차 |
| D6-1~3 | 다양한 PP 문제 |

---

## 7. 버그 수정 이력

### 7.1 normalize_after_zoom PP 버그 (2026-01-25)

**문제**: center-aligned zoom에서 PP가 스케일링됨 (256 → 190)
**수정**: `preprocess.py` - zoom_center_mode 조건 추가

### 7.2 compute_l1 NameError (2026-01-29)

**문제**: validator.py에서 compute_l1 파라미터 누락 → validation 메트릭 0으로 기록
**수정**: `MetricsComputer` 클래스로 모듈화 (`mouse_extensions/evaluation/metrics.py`)

---

## 8. 전처리 명령어

### 8.1 M5 (권장 기준선)

```bash
python -m mouse_extensions.preprocessing.preprocess \
    --preset M5 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M5
```

### 8.2 M5 + Temporal Variant

```bash
python -m mouse_extensions.preprocessing.preprocess \
    --preset M5 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --temporal-variant
# 결과: M5/ + M5t/
```

### 8.3 M0 (Baseline)

```bash
python -m mouse_extensions.preprocessing.preprocess \
    --preset M0 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M0 \
    --temporal-variant
```

### 8.4 기존 데이터에 Temporal Split 추가

```bash
python -m mouse_extensions.preprocessing.split_generator \
    --dataset_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --create-variant M5t
```

---

## 9. 데이터 위치

### 9.1 Raw Data

```
/home/joon/data/raw/markerless_mouse_1_nerf/
├── raw_videos/           # 6개 MP4 비디오
├── simpleclick_undist/   # 마스크 MP4 비디오
└── new_cam.pkl           # 카메라 파라미터
```

### 9.2 Preprocessed Data

```
/home/joon/data/preprocessed/FaceLift_mouse/
├── M5/        # ⭐ 권장 기준선 (Batch Uniform norm)
├── M5t/       # M5 + temporal 1:1:1 split (symlinks)
├── M5_4/      # Ablation: center + no norm
├── M5_5/      # Ablation: center + per-view norm
├── M5h_2/     # M5 + homography + per-sample zoom
├── M0/        # Baseline: raw (no center, no norm)
├── M0_n/      # Baseline: per-view norm only
├── D7_1/      # Legacy M1
└── D8/        # Legacy M2
```

---

## 10. 관련 문서

| 문서 | 위치 | 내용 |
|------|------|------|
| Commands | `../experiments/COMMANDS.md` | 명령어 SSOT |
| M5 Series Spec | `./M5_SERIES_SPEC.md` | M5 시리즈 상세 |
| Raw Data | `./RAW_DATA.md` | 원본 데이터 정보 |

---

*Preprocessing Registry v7.0 | 2026-01-29 | Added M5 ablation, M0 baseline, normalization modes, temporal split*
