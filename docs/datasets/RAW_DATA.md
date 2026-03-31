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

## 4. PoseSplatter 데이터와의 관계

> ⚠️ PoseSplatter (Goffinet et al. 2025)는 DANNCE/MAMMAL 데이터를 사용하지 **않으며**, **자체 녹화한 별도 데이터**를 사용합니다
> (Duke, 324K frames, 1536×2048, 30fps, 28cm 플라스틱 실린더, DOI: 10.7924/r4z323k2c).
> 본 프로젝트에서 PS 코드를 M5 데이터에 적용한 것. 자세한 비교: [[experiments/fl_vs_ps_comparison]] §2

---

## 5. 전처리 명령어

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

*Raw Data Sources v2.1 | Updated: 2026-03-30 | v13 섹션 삭제, CAMERA_NORMALIZATION 역링크 추가*
