---
date: 2025-12-20
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, critical, pipeline, camera-alignment]
project: mouse-facelift
status: active
generator: ai-assisted
generator_tool: claude-code
---

# CRITICAL: 합성 데이터 생성 파이프라인 핵심 사항

> **⚠️ 필독**: 이 문서는 GS-LRM 학습 실패를 방지하기 위한 핵심 사항입니다.

---

## 1. 핵심 원칙

### 🔴 절대 규칙: 이미지-카메라 일관성

```
┌─────────────────────────────────────────────────────────────────────┐
│  MVDiffusion 생성 이미지 = FaceLift 표준 카메라 가정                  │
│                                                                      │
│  따라서 합성 데이터의 카메라 정보도 FaceLift 표준이어야 함!           │
│                                                                      │
│  ❌ 원본 마우스 카메라 복사 금지                                      │
│  ✅ FaceLift 표준 카메라 생성 사용                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. FaceLift 표준 카메라 사양

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| **거리** | **2.7** | GS-LRM pretrained 기준, 절대 변경 금지 |
| 뷰 수 | 6 | 0°, 60°, 120°, 180°, 240°, 300° azimuth |
| Elevation | 20° | 고정 |
| FOV | ~50° | fx=fy=548.99 for 512x512 |
| 좌표계 | Z-up | Human 데이터와 동일 |

---

## 3. 잘못된 방식 vs 올바른 방식

### ❌ 잘못된 방식 (이전 버그)

```python
# scripts/generate_synthetic_data.py (OLD - 버그!)
# Copy opencv_cameras.json if exists
cameras_src = sample_dir / "opencv_cameras.json"  # 원본 마우스: 거리 2.0~3.4
cameras_dst = output_dir / "opencv_cameras.json"
if cameras_src.exists():
    shutil.copy(cameras_src, cameras_dst)  # ❌ 불일치 발생!
```

**결과**:
- MVDiffusion 이미지는 거리 2.7 가정
- 카메라 정보는 거리 2.0~3.4
- GS-LRM: 잘못된 Plucker ray → white prediction / mode collapse

### ✅ 올바른 방식 (수정됨)

```python
# scripts/generate_synthetic_data.py (NEW - 수정됨)
# CRITICAL: FaceLift 표준 카메라 사용
standard_cameras = generate_facelift_standard_cameras(
    sample_id=output_dir.name,
    num_views=6,
    camera_distance=2.7,  # FaceLift 표준
    elevation_deg=20.0,
    image_size=512
)
with open(cameras_dst, 'w') as f:
    json.dump(standard_cameras, f, indent=4)  # ✅ 일관성 유지!
```

---

## 4. 파이프라인 실행 순서

```bash
# 1. Pixel-based 전처리 (Grade A 품질)
python scripts/preprocess_pixel_based.py \
    --input_dir data_mouse \
    --output_dir data_mouse_pixel_based

# 2. MVDiffusion 학습 (checkpoint-2000 사용)
# (이미 완료됨)

# 3. 합성 데이터 생성 (FaceLift 표준 카메라 적용)
python scripts/generate_synthetic_data.py \
    --input_dir data_mouse_pixel_based \
    --output_dir data_mouse_synthetic_standard \
    --mvdiff_checkpoint checkpoints/mvdiffusion/mouse_pixel_based/checkpoint-2000

# 4. GS-LRM 학습
torchrun --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse_gslrm_synthetic_standard.yaml
```

---

## 5. GS-LRM 학습 필수 설정

```yaml
# configs/mouse_gslrm_*.yaml
training:
  dataset:
    num_input_views: 5     # ⚠️ NOT 1! Pretrained와 유사하게
    normalize_distance_to: 0.0  # 이미 2.7로 정규화됨

  losses:
    l2_loss_weight: 1.0
    lpips_loss_weight: 0.0      # ⚠️ Mouse 도메인에서 비활성화
    perceptual_loss_weight: 0.0 # ⚠️ Mouse 도메인에서 비활성화
    ssim_loss_weight: 0.5
```

---

## 6. 검증 체크리스트

### 합성 데이터 생성 후

```bash
# 카메라 거리 확인 (모두 2.7이어야 함)
python -c "
import json
from pathlib import Path
sample = Path('data_mouse_synthetic_standard/sample_000000')
with open(sample / 'opencv_cameras.json') as f:
    cam = json.load(f)
for i, frame in enumerate(cam['frames']):
    dist = cam.get('camera_distance', 'N/A')
    print(f'View {i}: distance = {dist}')
print(f'Camera type: {cam.get(\"camera_type\", \"unknown\")}')
"
```

예상 출력:
```
View 0: distance = 2.7
View 1: distance = 2.7
...
Camera type: facelift_standard_6view
```

---

## 7. 문제 발생 시

| 증상 | 원인 | 해결책 |
|------|------|--------|
| White prediction | 카메라 불일치 | 합성 데이터 재생성 (FaceLift 표준 카메라) |
| Mode collapse | num_input_views=1 | num_input_views=5로 변경 |
| Gradient explosion | LPIPS 활성화 | lpips=0, perceptual=0 설정 |

---

## 8. 관련 파일

- `scripts/generate_synthetic_data.py`: 합성 데이터 생성 (수정됨)
  - `generate_facelift_standard_cameras()`: FaceLift 표준 카메라 생성 함수
- `docs/reports/251219_known_issues_and_solutions.md`: 이슈 종합
- `gslrm/data/mouse_dataset.py`: 카메라 정규화 함수

---

*🤖 Generated with Claude Code*
