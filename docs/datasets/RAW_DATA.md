# Raw Data Sources

> **Navigation**: [← Index](./00_INDEX.md) | [MoC](../00_MoC_INDEX.md)
> **SSOT**: 원본 데이터 출처 및 샘플링 전략

---

## 1. 원본 데이터셋

### 1.1 Markerless Mouse (DANNCE)

| 항목 | 값 |
|------|-----|
| **출처** | DANNCE (3-Dimensional Aligned Neural Network for Computational Ethology) |
| **원본 저장소** | [github.com/spoonsso/dannce](https://github.com/spoonsso/dannce/tree/master/demo) |
| **피사체** | 마우스 (단일) |
| **뷰 수** | 6개 카메라 |
| **총 프레임** | ~18,000 프레임 |
| **원본 FPS** | 30 fps |
| **해상도** | 512 × 512 |

### 1.2 데이터 위치

**서버 경로**:
```
/home/joon/data/raw/markerless_mouse_1_nerf/
├── raw_videos/           # 6개 MP4 비디오 (카메라별)
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
└── new_cam.pkl           # 카메라 파라미터
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
| **원본 FPS** | 30 fps | |
| **유효 FPS** | 6 fps | 30 / 5 = 6 |

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

### Related Works

- **MAMMAL** (2023): Multi-animal 3D reconstruction
- **Pose Splatter** (2025): 3D Gaussian Splatting for animal pose

---

## 7. 관련 문서

- [[CAMERA_CONFIG]] - 6카메라 배치
- [[PREPROCESSING_REGISTRY]] - 프리셋 정의
- [[VERSION_SCHEMA]] - 데이터셋 버전

---

*Raw Data Sources v1.0 | 2026-01-26*
