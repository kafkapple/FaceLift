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
M5 데이터 (3,600 frames × 6 views × 512×512)
```

| 항목 | DANNCE 원본 | MAMMAL 가공 (markerless_mouse_1_nerf) | 본 프로젝트 (M5) |
|------|:-----------:|:-------------------------------------:|:----------------:|
| **출처** | Dunn et al. 2021 | An et al. 2023 | 본 연구 |
| **해상도** | 1152 × 1024 | 512 × 512 (비디오) | 512 × 512 (RGBA) |
| **FPS** | 100 | ~30 (비디오 변환) | 6 (frame_interval=5) |
| **프레임** | 18,000 | ~18,000 | 3,600 |
| **카메라** | 6대 | 6대 | 6대 |
| **마스크** | 없음 | simpleclick_undist | RGBA alpha |
| **저장소** | [spoonsso/dannce](https://github.com/spoonsso/dannce) | [anl13/MAMMAL_mouse](https://github.com/anl13/MAMMAL_mouse) | 로컬 |

> **⚠️ PoseSplatter 데이터와의 관계**:
> PoseSplatter (Goffinet et al. 2025)는 DANNCE/MAMMAL 데이터를 사용하지 **않으며**, **자체 녹화한 별도 데이터**를 사용합니다
> (Duke, 324K frames, 1536×2048, 30fps, 28cm 플라스틱 실린더, DOI: 10.7924/r4z323k2c).
> 본 프로젝트에서 PS 코드를 M5 데이터에 적용한 것이지, PS가 원래 DANNCE 데이터를 쓰는 것이 아닙니다.
> 자세한 비교: [[experiments/FL_vs_PS_comparison]] §2

### 1.2 데이터 위치

**서버 경로**:
```
/home/joon/data/raw/markerless_mouse_1_nerf/
├── videos_undist/        # 6개 undistorted MP4 비디오 (카메라별)
│   ├── 0.mp4
│   ├── 1.mp4
│   ├── 2.mp4
│   ├── 3.mp4
│   ├── 4.mp4
│   └── 5.mp4
├── simpleclick_undist/   # 마스크 MP4 비디오
│   ├── 0.mp4
│   ├── 1.mp4
│   └── ...
├── keypoints2d_undist/   # DANNCE 2D detections (6, 18000, 22, 3)
├── new_cam.pkl           # 카메라 파라미터
├── camera_params.h5      # 카메라 캘리브레이션 원본
└── center_rotation.npz   # 센터/회전 정보
```

---

## 2. v13 vs markerless_mouse_1_nerf

### 2.1 비교표

| 항목 | v13 | markerless_mouse_1_nerf |
|------|-----|-------------------------|
| **형태** | 사전 전처리된 샘플 | Raw 비디오 |
| **프레임 수** | ~1,800 샘플 | ~18,000 프레임 |
| **샘플링** | 이미 적용됨 | Raw (frame_jump 필요) |
| **마스크** | PNG 이미지 | MP4 비디오 |
| **카메라** | 정규화 안됨 | 정규화 안됨 |
| **권장** | ⚠️ Legacy | ✅ 권장 |

### 2.2 v13 문제점

1. **샘플 수 제한**: 1,800개 (전체의 10%)
2. **PP 버그**: cx=cy=256 고정 (실제 값 아님)
3. **정규화 누락**: fx, translation 정규화 안됨
4. **유지보수 중단**: 더 이상 업데이트 안됨

---

## 3. 샘플링 전략

### 3.1 Frame Interval

| 설정 | 값 | 설명 |
|------|-----|------|
| **frame_interval** | 5 | 5프레임마다 1개 샘플링 |
| **원본 FPS** | 100 fps | DANNCE 녹화 원본 (1152×1024) |
| **유효 FPS** | 20 fps | 100 / 5 = 20 |

### 3.2 샘플 수 계산

```
Raw frames: 18,000
frame_interval: 5
Total samples: 18,000 / 5 = 3,600
```

### 3.3 Train/Val Split

| Split | 비율 | 샘플 수 | 프레임 범위 |
|-------|------|---------|-------------|
| **Train** | 90% | 3,240 | 전체 무작위 |
| **Val** | 10% | 360 | 전체 무작위 |

**Temporal Split (_t 접미사)**:
| Split | 비율 | 프레임 범위 |
|-------|------|-------------|
| Train | 60% | 0 - 10,800 |
| Val | 20% | 10,800 - 14,400 |
| Test | 20% | 14,400 - 18,000 |

---

## 4. Frame Discontinuity

### 4.1 불연속 위치

원본 비디오에서 녹화 갭이 있는 위치:

```python
DISCONTINUITY_FRAMES = {5900, 11800, 17700}
```

### 4.2 중요 사항

| 오해 | 실제 |
|------|------|
| 해당 프레임 제외 필요? | ❌ 아님 |
| 해당 프레임 불량? | ❌ 정상 |
| 샘플 수 감소? | ❌ 전체 3,600 사용 |

**결론**: DISCONTINUITY_FRAMES는 **정보 제공용**. 해당 프레임 자체는 정상이며 학습에서 제외하지 않음.

### 4.3 검증 도구

```bash
# 특정 프레임 주변 슬로우모션 추출
python /tmp/extract_frame_context.py \
    --video /home/joon/data/raw/markerless_mouse_1_nerf/raw_videos/0.mp4 \
    --frames 5900,11800,17700 \
    --speed 0.1
```

---

## 5. 전처리 명령어

### 5.1 M3_2 (권장)

```bash
cd /home/joon/dev/FaceLift

python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_2 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_2
```

### 5.2 전처리 검증

```bash
python mouse_extensions/scripts/diagnostics/verify_pp_mvg_consistency.py \
    --datasets M3_2 --verbose
```

---

## 6. References

### Primary Citation (DANNCE)

> Dunn, T. W., et al. (2021). **Geometric deep learning enables 3D kinematic profiling across species and environments**. *Nature Methods*, 18(5), 564–573.

### Data Processing (MAMMAL)

> An, L., et al. (2023). **Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL**. *Nature Communications*, 14, 7727.
> - DANNCE `markerless_mouse_1` 데이터에 segment mask 추가 및 NeRF용 가공
> - GitHub: [anl13/MAMMAL_mouse](https://github.com/anl13/MAMMAL_mouse)

### Comparison Method (PoseSplatter) — 별도 데이터

> Goffinet, J., et al. (2025). **PoseSplatter: Pose Conditioned Gaussian Splatting from a Single Image**. arXiv:2505.18342.
> - **자체 녹화 데이터** 사용 (Duke, 1536×2048, 30fps, 324K frames, DOI: 10.7924/r4z323k2c)
> - DANNCE/MAMMAL 데이터와 **무관**. 같은 Duke 연구 생태계이나 별도 녹화.
> - David Carlson (PS 교신저자)은 DANNCE 공저자이기도 하나, 데이터는 새로 촬영.

---

## 7. 관련 문서

- [[PREPROCESSING_REGISTRY]] - 프리셋 정의
- [[../../mouse_extensions/docs/DATASET_FRAME_INDEXING]] - **⚠️ 프레임 인덱싱 매핑 (step=5 규칙)**

---

*Raw Data Sources v1.1 | Updated: 2026-03-11 (FPS 수정: 30→100, 유효 6→20)*

> **Note**: Camera Configuration은 별도 문서로 분리됨 → [[M5_SERIES_SPEC]] 참조
> _(이전 병합되어 있던 Camera Configuration 내용은 [[M5_SERIES_SPEC]]에 포함, 2026-03-22 정리)_
