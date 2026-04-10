# Raw Data Sources

> **Navigation**: [← Index](../INDEX.md) | [Commands](../experiments/COMMANDS.md)
> **SSOT**: 원본 데이터 출처 및 샘플링 전략

---

## 1. 원본 데이터셋

### 1.1 Markerless Mouse (DANNCE → MAMMAL → 본 프로젝트)

**데이터 출처 체인**:
```
DANNCE (Harvard, Dunn et al. 2021)
  │  markerless_mouse_1: 6cam, 1152×1024, 100fps, 18,000 frames
  v
MAMMAL (An et al. 2023)
  │  markerless_mouse_1_nerf/: segment mask 추가, NeRF용 재가공
  v
본 프로젝트 (FaceLift Mouse)
  │  M5 전처리: center crop → 512×512 RGBA, 카메라 정규화
  v
M5t2 데이터 (3,600 samples × 6 views × 512×512 RGBA)
```

---

### 1.2 Rat (s-DANNCE → 본 프로젝트)

**데이터 출처 체인**:
```
s-DANNCE (Marshall et al., Harvard Dataverse)
  │  6cam rat, 1280×1024, 50fps, .mat keypoints + labels
  v
본 프로젝트 (FaceLift Rat)
  │  RAT1: SAM2 mask + gslrm_format 변환 (1001 frames, despilled)
  │  RAT2: 2967 frames, v1→v8 진화 (현재 v8 = scene-centered recentering)
  v
RAT2_v8_recentered (현재 active ⭐)
```

---

## 2. 버전별 상세 명세

### 2.1 원본 (DANNCE)

| 항목 | 값 |
|------|-----|
| **출처** | Harvard, Dunn et al. 2021 (Nature Methods) |
| **데이터명** | markerless_mouse_1 |
| **해상도** | 1152 × 1024 px |
| **FPS** | 100 fps |
| **총 프레임** | 18,000 |
| **영상 길이** | 180초 (3분) |
| **카메라** | 6 views (동기화, 고정 rig) |
| **마스크** | 없음 |
| **저장소** | [spoonsso/dannce](https://github.com/spoonsso/dannce) |

> **불연속 위치**: frame 5900, 11800, 17700 — 정보 제공용, 해당 프레임 정상 (제외 불필요)

---

### 2.2 MAMMAL 가공 버전 (markerless_mouse_1_nerf)

> An, L., et al. (2023). *Nature Communications*, 14, 7727.
> GitHub: [anl13/MAMMAL_mouse](https://github.com/anl13/MAMMAL_mouse)

| 항목 | 값 |
|------|-----|
| **출처** | An et al. 2023 — DANNCE에 segment mask 추가 |
| **데이터명** | markerless_mouse_1_nerf |
| **해상도** | 1152 × 1024 px (undistorted, 원본 해상도 유지) |
| **FPS** | 100 fps (원본 동일) |
| **총 프레임** | 18,000 (원본 동일) |
| **카메라** | 6 views (동일) |
| **추가된 것** | segment mask (simpleclick_undist/) |
| **변경된 것** | undistortion 적용, 카메라 파라미터 재가공 |
| **마스크 방식** | SimpleClick 반자동 어노테이션 |
| **서버 경로** | `/home/joon/data/raw/markerless_mouse_1_nerf/` |

**파일 구조**:
```
/home/joon/data/raw/markerless_mouse_1_nerf/
├── videos_undist/         # 6개 undistorted MP4 (cam 0-5, 1152×1024, 100fps)
│   ├── 0.mp4 ~ 5.mp4
├── simpleclick_undist/    # segment mask MP4 (동일 해상도/fps)
│   ├── 0.mp4 ~ 5.mp4
├── keypoints2d_undist/    # DANNCE 2D keypoints (6, 18000, 22, 3)
├── new_cam.pkl            # 카메라 파라미터 (fx=549, cx/cy 실측값)
├── camera_params.h5       # 캘리브레이션 원본
└── center_rotation.npz    # 센터/회전 정보
```

**카메라 파라미터 (new_cam.pkl)**:
| 항목 | 값 |
|------|-----|
| **fx** | 549.0 (exact) |
| **fy** | ~549.0 (fx ≈ fy, 0.46% 차이) |
| **cx, cy** | 카메라별 실측값 (≠ 256 고정) |
| **평균 거리** | ~2.6m (범위: 2.4–2.8m) |

---

### 2.3 본 프로젝트 — M5t2 (현재 사용 버전 ⭐)

| 항목 | 값 |
|------|-----|
| **전처리 preset** | M5 (recentered_affine, Batch Uniform norm) |
| **Split 전략** | temporal_80_10_10 (t2 suffix) |
| **해상도** | 512 × 512 px (RGBA 4채널) |
| **Target fx** | 548.9938 (원본 549.0과 거의 동일 — recentered affine crop 방식으로 원본 focal length 유지, uniform downscale이 아님) |
| **총 프레임 (샘플)** | 3,600 (= 18,000 / frame_interval=5) |
| **유효 FPS** | 20 fps (100 / 5) |
| **시간 간격 / 샘플** | 0.05초 |
| **카메라** | 6 views → 6-채널 입력 |
| **출력 형식** | RGBA PNG (RGB 이미지 + alpha mask) |
| **카메라 정규화** | Batch Uniform: centroid→origin, avg dist=2.7m |
| **서버 경로 (데이터)** | `/home/joon/data/preprocessed/FaceLift_mouse/M5/` |
| **Split 파일** | `data_mouse_t2_{train/val/test}.txt` |

**Train/Val/Test Split (M5t2)**:

| Split | 프레임 수 | 비율 | 샘플 인덱스 범위 | 원본 프레임 범위 |
|-------|:--------:|:----:|:---------------:|:--------------:|
| **Train** | 2,880 | 80% | 000000–002879 | 0–14,395 |
| **Val** | 360 | 10% | 002880–003239 | 14,400–16,195 |
| **Test** | 360 | 10% | 003240–003599 | 16,200–17,995 |
| **Total** | **3,600** | 100% | — | 0–17,995 |

---

### 2.4 Fitting Intervals (샘플링 전략 비교)

> 모든 interval은 **원본 18,000 프레임**에 적용됨 (subsampled 3,600이 아님).

| 설정 | frame_interval | samples (=18000/interval) | 시간 간격 | 용도 |
|------|:--------------:|:-------------------------:|:---------:|------|
| **Original (paper_fast)** | 5 | 3,600 | 0.05s | 기본 전처리 ⭐ |
| **PoC subset** | 120 | 150 (→100 used) | 1.20s | 빠른 feasibility 검증 |
| **Production keyframes** | 20 | 900 | 0.20s | 행동 keyframe 추출 |

---

## 3. 버전 비교 요약

| 항목 | DANNCE 원본 | MAMMAL 가공 | 본 프로젝트 (M5t2) |
|------|:-----------:|:-----------:|:-----------------:|
| **출처** | Dunn et al. 2021 | An et al. 2023 | 본 연구 |
| **해상도** | 1152 × 1024 | 1152 × 1024 (undist) | **512 × 512 RGBA** |
| **FPS** | 100 | 100 | 20 (effective, step=5) |
| **총 프레임** | 18,000 | 18,000 | **3,600 samples** |
| **카메라** | 6대 | 6대 | 6대 |
| **마스크** | 없음 | simpleclick_undist | RGBA alpha |
| **fx** | 549.0 | 549.0 | 548.9938 |
| **카메라 정규화** | 없음 | 없음 | Batch Uniform (avg=2.7m) |
| **Split** | — | — | 2880 / 360 / 360 |

---

### 2.5 Rat — RAT1 / RAT2 시리즈

**RAT1** (260323 baseline):
| 항목 | 값 |
|------|-----|
| **출처** | s-DANNCE (Harvard Dataverse) |
| **카메라** | 6 views |
| **프레임** | 1,001 (SAM2 mask 적용 subset) |
| **Val PSNR** | 17.49 dB (100-UID), zero-shot 2.77 dB (FT gain +14.72) |
| **상태** | ✅ Baseline (overfitting 17.5dB gap) |
| **데이터 경로** | `/home/joon/dev/FaceLift/outputs/sdannce_rat_ft/gslrm_format/` (NFS, 소규모) |
| **Split 파일** | `data_sdannce_{train,val,test}.txt` |
| **Config** | `configs/datasets/RAT1.yaml` |

**RAT2 v2~v6** — ❌ 모두 retracted (v3/v6 audit 260407 참조). `feedback_fx_normalization_checklist.md` / `feedback_bg_dominated_loss.md` 참조.

**RAT2 v8_recentered** (현재 active ⭐, 260407):
| 항목 | 값 |
|------|-----|
| **Pipeline** | `rat2_s1despill_s2fxnorm` → `sdannce_recenter_adapter` (Plan v2 Phase A.2) |
| **총 프레임** | 2,967 |
| **Split** | Train 2385 / Val 290 / Test 292 |
| **Scene center** | com3d 기반 recentering → `[-0.501, +0.330, +0.112]` |
| **Cam distance** | 2.7 (mouse-equivalent normalization) |
| **Pairwise angle** | 13.9° (v6의 5.8°는 camera convergence point 기준 오측) |
| **Parallax (50mm)** | ~5.8 px (v6 ~1.1 px) |
| **fx** | 549.0 (normalized) |
| **데이터 경로** | `/node_data/joon/data/preprocessed/FaceLift_rat/rat2_v8_recentered/` ⚠️ NVMe |
| **Split 파일** | `data_rat2_{train,val,test}.txt` |
| **Config** | `configs/datasets/RAT2_v8_recentered.yaml` |
| **Audit 근거** | `~/results/FaceLift/rat/AUDIT_NARROW_BASELINE.md` |

---

### 2.6 s-DANNCE 행동 데이터 (별도 구조)

> ⚠️ `preprocessed/`가 아닌 `sdannce/` 독립 경로. 이유: 행동 .mat ≠ 이미지 전처리.

```
/home/joon/data/sdannce/
├── mouse/
│   ├── dataverse/    # Harvard Dataverse .mat 원본 (keypoints + labels)
│   └── features/     # 추출 .npz (covariance, S1, S3 등)
├── rat/
│   ├── dataverse/    # Harvard Dataverse .mat 원본
│   └── features/     # 추출 .npz
└── metadata/         # cohort 메타데이터, HLAC 매핑
```

**관련 문서**: [[SDANNCE_VIDEO_AVAILABILITY]]

---

## 3. 절대 경로 SSOT (⭐ Quick Reference)

> **⚠️ Storage Tier 규칙** (CLAUDE.md §2.4): 학습 데이터는 **반드시 `/node_data/` (local NVMe)**.
> NFS (`/home/joon/dev/`) 금지 — cgroup v2 page cache → oomd kill 위험.
> `/home/joon/data → /node_data/joon/data` symlink.

| 용도 | 절대 경로 | Storage |
|------|----------|:-------:|
| **Mouse 원본** | `/home/joon/data/raw/markerless_mouse_1_nerf/` | NVMe (symlink) |
| **Mouse M5t2 (현재)** | `/home/joon/data/preprocessed/FaceLift_mouse/M5/` + `data_mouse_t2_*.txt` | NVMe |
| **Rat RAT1** | `/home/joon/dev/FaceLift/outputs/sdannce_rat_ft/gslrm_format/` | NFS (소규모 OK) |
| **Rat RAT2_v8 (현재)** | `/node_data/joon/data/preprocessed/FaceLift_rat/rat2_v8_recentered/` | NVMe ✅ |
| **s-DANNCE 원본 (rat)** | `/home/joon/data/sdannce/rat/dataverse/` | NVMe (symlink) |
| **s-DANNCE features (rat)** | `/home/joon/data/sdannce/rat/features/` | NVMe (symlink) |
| **체크포인트** | `/node_data/joon/checkpoints/` | NVMe ✅ |
| **코드/config** | `/home/joon/dev/FaceLift/` | NFS (소량 OK) |
| **outputs/ (viz, reports)** | `/home/joon/dev/FaceLift/outputs/` | NFS (비학습) |

---

## 4. PoseSplatter 데이터와의 관계

> ⚠️ PoseSplatter (Goffinet et al. 2025)는 DANNCE/MAMMAL 데이터를 사용하지 **않으며**, **자체 녹화한 별도 데이터**를 사용합니다
> (Duke, 324K frames, 1536×2048, 30fps, 28cm 플라스틱 실린더, DOI: 10.7924/r4z323k2c).
> 본 프로젝트에서 PS 코드를 M5 데이터에 적용한 것. 자세한 비교: [[experiments/fl_vs_ps_comparison]] §2

---

## 5. 전처리 명령어

### 5.1 Mouse (M5t2)

```bash
# M5t2 생성 (M5 기반 temporal split)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M5 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --temporal-variant

# 검증
python mouse_extensions/scripts/diagnostics/verify_pp_mvg_consistency.py \
    --datasets M5t2 --verbose
```

### 5.2 Rat (RAT2_v8_recentered)

```bash
# Phase A.2: scene-centered recentering (Plan v2)
# Pipeline: rat2_s1despill_s2fxnorm → sdannce_recenter_adapter
python -m mouse_extensions.preprocessing.adapters.sdannce_recenter_adapter \
    --input-dir /node_data/joon/data/preprocessed/FaceLift_rat/rat2_s1despill_s2fxnorm \
    --output-dir /node_data/joon/data/preprocessed/FaceLift_rat/rat2_v8_recentered
```

### 5.3 학습 실행 (config 지정)

```bash
# Mouse
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e E0_1_facelift

# Rat (v8)
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d RAT2_v8_recentered -e <rat_experiment>
```

---

## 6. FPS 혼동 주의

> ⚠️ **DANNCE/MAMMAL mouse 데이터는 항상 100fps**. 아래 "30fps" 출처는 전부 **다른 맥락**:
> - PoseSplatter의 자체 rat 데이터 (30fps) — 다른 데이터셋
> - MAMMAL fitting 스크립트 `--fps 30` — 출력 영상 FPS (데이터 FPS 아님)
> - s-DANNCE rat 데이터 (50fps) — 다른 동물
>
> **검증**: `ffprobe videos_undist/0.mp4` → `r_frame_rate=100/1, nb_frames=18000, duration=180.0s`

---

## 7. Frame Discontinuity

```python
DISCONTINUITY_FRAMES = {5900, 11800, 17700}
```

해당 위치에서 원본 비디오 녹화 갭 존재. **프레임 자체는 정상** — 학습에서 제외하지 않음.

---

## 관련 문서

- [[PREPROCESSING_REGISTRY]] — 전처리 preset 정의 (M5, M5_4, M5_5 등)
- [[M5_SERIES_SPEC]] — 카메라 정규화 방식 상세 비교
- Obsidian `theory/CAMERA_NORMALIZATION.md` — Batch Uniform 설계 근거 + MVG 이론
- [[MULTI_ANIMAL_PREPROCESSING]] — Plucker ray, 다중 동물 전처리
- [[SDANNCE_VIDEO_AVAILABILITY]] — Rat (s-DANNCE) 데이터셋 목록

---

*Raw Data Sources v3.0 | Updated: 2026-04-08 | Rat (RAT1/RAT2_v8) + s-DANNCE + 절대 경로 SSOT 섹션 추가*
