# Mouse Dataset Source

> **Navigation**: [← MoC](../../00_MoC_INDEX.md) | [Practical](../) | [Preprocessing Registry](./PREPROCESSING_REGISTRY.md)

FaceLift Mouse 프로젝트에서 사용하는 데이터셋 출처 및 구조.

---

## 1. Markerless Mouse Dataset (DANNCE)

### 1.1 Overview

| 항목 | 값 |
|------|-----|
| **출처** | DANNCE (3-Dimensional Aligned Neural Network for Computational Ethology) |
| **원본 저장소** | [github.com/spoonsso/dannce](https://github.com/spoonsso/dannce/tree/master/demo) |
| **피사체** | 2마리 마우스 |
| **뷰 수** | 6개 카메라 |
| **시간 구간** | 6개 세션 |
| **총 비디오** | 72개 (2 × 6 × 6) |
| **프레임/비디오** | 3000 프레임 |
| **샘플링** | 100 프레임/비디오 |
| **총 샘플 수** | **7,200개** |
| **원본 FPS** | 300 fps |
| **샘플링 간격** | 5 frames (frame_jump=5) |
| **유효 FPS** | 60 fps (300/5) |

### 1.2 Data URLs

**비디오 데이터**:
- demo/markerless_mouse_1/videos/link_to_videos.txt
- demo/markerless_mouse_2/videos/link_to_videos.txt

**사전학습 가중치**:
- demo/markerless_mouse_1/DANNCE/train_results/link_to_weights.txt
- demo/markerless_mouse_1/DANNCE/train_results/AVG/link_to_weights.txt

### 1.3 Labels

| 유형 | 설명 |
|------|------|
| **Human Annotation** | 172개 샘플 (1,032 이미지), 22개 랜드마크 수작업 라벨링 |
| **Model Prediction** | DANNCE 모델로 전체 비디오 키네마틱/행동 예측 |

### 1.4 Sampling Strategy

| 항목 | 값 | 설명 |
|------|-----|------|
| **원본 FPS** | 300 fps | 고속 카메라 촬영 |
| **frame_jump** | 5 | 5프레임마다 1개 샘플링 |
| **유효 FPS** | 60 fps | 실제 학습 데이터 시간 해상도 |
| **프레임/비디오** | 3000 → 600 | 샘플링 후 프레임 수 |

**⚠️ MAMMAL과 정렬 주의**:
- MAMMAL: interval=1 (매 프레임)
- Pose-Splatter: frame_jump=5
- **불일치 시 temporal mismatch 발생**

---


### 1.4 Landmark Definition (22 keypoints)

마우스 신체 22개 관절점:
- Head: nose, left_ear, right_ear
- Spine: neck, mid_back, lower_back, tail_base
- Limbs: 4 legs × (shoulder/hip, elbow/knee, wrist/ankle, paw)

---

## 2. Related Projects

이 데이터셋을 사용하는 주요 프로젝트:

| 프로젝트 | 용도 | 참조 |
|----------|------|------|
| **DANNCE** | 3D pose estimation | Dunn et al. (2021) |
| **MAMMAL** | Multi-animal 3D reconstruction | An et al. (2023) |
| **Pose Splatter** | 3D Gaussian Splatting | Goffinet et al. (2025) |
| **FaceLift Mouse** | Single-image 3D reconstruction | This project |

---

## 3. FaceLift 전처리 버전

| 버전 | 해상도 | 정규화 | 샘플 수 | 상태 |
|------|--------|--------|---------|------|
| D7.1 | 512×512 | ✅ | 3,238 train / 359 val | ⭐ 권장 |
| D8 | 512×512 | ✅ | 3,238 train / 359 val | 🧪 실험적 |
| D9 | 1152×1024 | ❌ | 3,238 train / 359 val | ⚠️ 고해상도 |

→ 상세: [PREPROCESSING_REGISTRY.md](./PREPROCESSING_REGISTRY.md)

---

## 4. Data Locations

### 4.1 Raw Data

```
/home/joon/data/raw/markerless_mouse_1_nerf/
├── raw_videos/           # 6개 MP4 비디오
├── simpleclick_undist/   # 마스크 MP4 비디오
└── new_cam.pkl           # 카메라 파라미터
```

### 4.2 Preprocessed Data

```
/home/joon/data/preprocessed/FaceLift_mouse/
├── D7_1/      # 512×512, 정규화 ✅ (권장)
├── D8/        # 512×512, skew 보정
└── D9/        # 1152×1024, 원본 해상도
```

---

## 5. References

### Primary Citation (DANNCE)

> Dunn, T. W., Marshall, J. D., Severson, K. S., Aldarondo, D. E., Hildebrand, D. G. C., Chettih, S. N., Wang, W. L., Gellis, A. J., Carlson, D. E., Aronov, D., Freiwald, W. A., Wang, F., & Ölveczky, B. P. (2021). **Geometric deep learning enables 3D kinematic profiling across species and environments**. *Nature Methods*, 18(5), 564–573. https://doi.org/10.1038/s41592-021-01106-6

### Related Works

> Goffinet, J., Min, Y., Tomasi, C., & Carlson, D. E. (2025). **Pose Splatter: A 3D Gaussian Splatting Model for Quantifying Animal Pose and Appearance**. arXiv:2505.18342. https://doi.org/10.48550/arXiv.2505.18342

> An, L., Ren, J., Yu, T., Hai, T., Jia, Y., & Liu, Y. (2023). **Three-dimensional surface motion capture of multiple freely moving pigs using MAMMAL**. *Nature Communications*, 14(1), 7727. https://doi.org/10.1038/s41467-023-43483-w

---

## 6. License & Attribution

데이터 사용 시 DANNCE 원본 논문 인용 필수.

---

*FaceLift Mouse | Dataset Source v1.0 | 2026-01-23*
