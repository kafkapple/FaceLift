# Synthetic Data Generation Overview

**Version**: 1.0.0
**Created**: 2026-01-28
**Purpose**: PP 가설 검증을 위한 합성 데이터 생성

---

## 1. 핵심 가설

> **PP (Principal Point) 불일치가 Ghosting의 주요 원인**

```
물체가 이미지 중앙에 없음 → PP offset 발생 → 광선 방향 오류 → Ghosting
```

---

## 2. FaceLift 원본 학습 설정

### 데이터 형식
```
sample_XXX/
├── images/cam_000.png ~ cam_031.png  (32개 뷰)
└── opencv_cameras.json
```

### 카메라 파라미터
| 항목 | 값 |
|------|-----|
| 뷰 수 | 32 |
| azimuth | 0°~360° (11.25° 간격) |
| elevation | 20° |
| distance | 2.7 |
| fx, fy | 549 |
| cx, cy | 256 |
| resolution | 512×512 |

### 학습 샘플링
```
32개 뷰 → 8개 랜덤 샘플링 → 4개 입력 + 4개 타겟
```

---

## 3. 합성 데이터셋 계획

### 3.1 Position Experiments (단순 모델)

**목적**: 빠른 PP 가설 검증

| 실험 | 위치 | PP | 상태 |
|------|------|-----|------|
| POS_C | 중앙 | 정확 | ✅ 완료 |
| POS_R | 오른쪽 | 불일치 | ✅ 완료 |
| POS_R_PP | 오른쪽 | 보정 | ✅ 완료 |
| POS_RANDOM | 랜덤 | 불일치 | 🔄 진행 |
| POS_RANDOM_PP | 랜덤 | 보정 | ⏳ 대기 |

**위치**: `/home/joon/data/synthetic/position_experiments/`

### 3.2 MAMMAL 32-View (실제 마우스 메시)

**목적**: FaceLift 동일 형식으로 정식 학습

| 실험 | 마우스 위치 | PP | 설명 |
|------|------------|-----|------|
| MAMMAL_CENTER | 원점 (중앙) | cx=cy=256 ✅ | 기준선 |
| MAMMAL_OFFSET | +0.3, +0.2 이동 | cx=cy=256 ❌ | PP 불일치 |
| MAMMAL_OFFSET_PP | +0.3, +0.2 이동 | 보정됨 ✅ | PP 가설 검증 |

**위치**: `/home/joon/data/synthetic/mammal_32view/`

---

## 4. 데이터 소스

### MAMMAL Fitting Results
```
/home/joon/dev/MAMMAL_mouse/results/fitting/
  markerless_mouse_1_nerf_v012345_kp22_20260126_025249/
  ├── obj/step_2_frame_*.obj    (2139개 메시)
  └── params/step_2_frame_*.pkl (포즈 파라미터)
```

### 메시 사양
- 정점: ~14,400
- 스케일: mm 단위 (body ~70mm)
- 프레임 간격: 5

---

## 5. 검증 계획

### Phase 1: Position Experiments
1. GS-LRM으로 각 실험 학습
2. Ghosting 정도 비교
3. PP 보정 효과 정량화

### Phase 2: MAMMAL 32-View
1. FaceLift 동일 설정으로 학습
2. 실제 마우스 데이터와 비교
3. 최종 PP 가설 검증

---

## 6. 파일 위치

| 파일 | 경로 |
|------|------|
| Position 렌더링 | `mouse_extensions/scripts/blender/render_position_experiments.py` |
| MAMMAL 렌더링 | `mouse_extensions/scripts/blender/render_mammal_32view.py` |
| 계획 문서 | `docs/analysis/MAMMAL_MESH_RENDERING_PLAN.md` |
| 매뉴얼 | `docs/analysis/SYNTHETIC_DATA_MANUAL.md` |

---

*FaceLift Synthetic Data | PP Hypothesis Validation*
