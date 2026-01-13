# FaceLift Mouse Adaptation: 구현 가이드

> 원본 FaceLift 를 fork 한 후, 생쥐(Mouse) 데이터에 적용하기 위한 단계별 구현 가이드

## 목차

1. [00_Index](./00_Index.md) - 이 문서
2. **이론 (Theory)**
   - [Camera_Projection](./theory/Camera_Projection.md) - 카메라 투영 모델과 intrinsics
   - [Preprocessing_Pipeline](./theory/Preprocessing_Pipeline.md) - 전처리 파이프라인 이론
3. **튜토리얼 (Tutorials)**
   - [Step1_Fork_and_Setup](./tutorials/Step1_Fork_and_Setup.md) - Fork 및 환경 설정
   - [Step2_Mouse_Dataset](./tutorials/Step2_Mouse_Dataset.md) - MouseViewDataset 구현
   - [Step3_Preprocessing](./tutorials/Step3_Preprocessing.md) - 카메라 정규화 전처리
   - [Step4_Config_Setup](./tutorials/Step4_Config_Setup.md) - Mouse config 설정
   - [Step5_Training](./tutorials/Step5_Training.md) - 학습 실행
4. **참조 (Reference)**
   - [Config_Options](./reference/Config_Options.md) - 설정 옵션 상세
   - [Camera_Parameters](./reference/Camera_Parameters.md) - 카메라 파라미터 설명

---

## Quick Overview: 원본 vs Mouse 적용

### 원본 FaceLift 구조

```
FaceLift/                    
├── train_gslrm.py           # GS-LRM 학습 스크립트
├── gslrm/
│   ├── data/
│   │   └── dataset.py       # RandomViewDataset (32 views, human)
│   └── model/
│       └── gslrm.py         # GS-LRM 모델
└── configs/
    └── gslrm.yaml           # Human face config
```

### Mouse 적용 추가 모듈

```diff
FaceLift/                    
├── train_gslrm.py           # 수정: use_mouse_dataset 플래그 추가
├── gslrm/
│   ├── data/
│   │   ├── dataset.py       # 원본 유지
+│   │   └── mouse_dataset.py # [NEW] MouseViewDataset (6 views)
│   └── model/
│       └── gslrm.py         # 원본 유지
+├── scripts/
+│   └── preprocess_mouse.py  # [NEW] 카메라 정규화 전처리
└── configs/
    ├── gslrm.yaml           # 원본 유지
+   └── mouse/
+       └── gslrm.yaml        # [NEW] Mouse config
```

---

## 핵심 개념: Why 전처리가 필요한가?

### FaceLift Pretrained Model 가정

| 파라미터 | FaceLift 기대값 | 설명 |
|----------|-----------------|------|
| `fx, fy` | 549 | 초점 거리 (pixels) |
| `cx, cy` | 256 | 주점 (이미지 중앙) |
| `distance` | 2.7 | 카메라-원점 거리 |
| 좌표계 | Z-up | 위쪽 방향 |

### 실제 Mouse 데이터

| 파라미터 | Mouse 원본 | 문제 |
|----------|------------|------|
| `fx, fy` | ~1600 | 너무 큼 |
| `cx, cy` | 다양 | 뷰마다 다름 |
| `distance` | 246-414mm | 뷰마다 다름 |
| 좌표계 | 다양 | 불확실 |

### 해결책: 전처리

```python
# 핵심 수식
image_scale = (target_fx / orig_fx) * (orig_dist / target_dist)
```

→ 모든 뷰에서 **동일한 투영 크기**로 정규화

---

## 구현 순서 (Step-by-Step)

### Step 1: Fork & 환경 설정
→ [Step1_Fork_and_Setup.md](./tutorials/Step1_Fork_and_Setup.md)

### Step 2: MouseViewDataset 구현
→ [Step2_Mouse_Dataset.md](./tutorials/Step2_Mouse_Dataset.md)

### Step 3: 전처리 스크립트 구현
→ [Step3_Preprocessing.md](./tutorials/Step3_Preprocessing.md)

### Step 4: Config 설정
→ [Step4_Config_Setup.md](./tutorials/Step4_Config_Setup.md)

### Step 5: 학습 실행
→ [Step5_Training.md](./tutorials/Step5_Training.md)

---

## 핵심 파일 요약

| 파일 | 역할 | 수정/신규 |
|------|------|-----------|
| `gslrm/data/mouse_dataset.py` | 6-view dataset with camera normalization | 신규 |
| `scripts/preprocess_mouse.py` | 원본 데이터 → FaceLift 형식 변환 | 신규 |
| `configs/mouse/gslrm.yaml` | Mouse 학습 설정 | 신규 |
| `train_gslrm.py` | `use_mouse_dataset` 플래그 추가 | 수정 |

---

*Created: 2026-01-13*
*Version: v1.0*
