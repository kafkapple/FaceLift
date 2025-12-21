# Mouse-FaceLift: 카메라 설정 및 중앙 정렬 핵심 사항

**날짜**: 2025-12-19
**주제**: 마우스 데이터 중앙 정렬 및 카메라 파라미터 처리 방식
**상태**: 분석 완료

---

## 1. 핵심 발견사항

### 문제 원인
- GS-LRM 합성 데이터 학습 시 **gradient explosion** 발생
- 근본 원인: MVDiffusion 출력 이미지에서 **마우스 위치가 뷰마다 다름**
- GS-LRM은 객체가 **원점(중앙)**에 있다고 가정 → 위치 불일치로 3D 복원 실패

### FaceLift 파이프라인 설계 핵심
```
[Single View Input] → [MVDiffusion] → [6 Multi-View Images] → [GS-LRM] → [3D Gaussians]
                           ↓                    ↓
                    중앙 정렬 입력 가정      고정 카메라 사용
```

**MVDiffusion과 GS-LRM 모두 객체가 이미지 중앙에 있다고 가정**

---

## 2. 카메라 설정 비교

### Human FaceLift (원본)
| 항목 | 값 |
|------|-----|
| 카메라 배치 | Turntable (균등 60° 간격) |
| 거리 | 2.7 |
| FOV | 50° |
| Elevation | 0° (또는 20° for rendering) |
| 객체 정렬 | **MTCNN 얼굴 감지 → 중앙 정렬** |

```python
# face_utils.py
TRAINING_SET_FACE_SIZE = 194.27
TRAINING_SET_FACE_CENTER = [251.83, 280.01]
# → 모든 얼굴이 동일한 위치/크기로 정규화
```

### Mouse FaceLift (현재)
| 항목 | 값 |
|------|-----|
| 카메라 배치 | 실제 MAMMAL 카메라 (불균등) |
| 거리 | **정규화됨** (원본 246~414 → 2.7) |
| FOV | **정규화됨** (→ 50°) |
| 객체 정렬 | **없음** ← 문제! |

```python
# process_mouse_data.py
scale_factor = target_distance / avg_distance  # 거리만 정규화
# 객체 위치는 정규화되지 않음!
```

---

## 3. 해결 방안

### 필요한 전처리
```
[원본 이미지] → [중앙 정렬] → [MVDiffusion 학습] → [합성 데이터 생성] → [GS-LRM 학습]
     ↓
SimpleClick 마스크로 배경 제거됨
객체가 각 뷰마다 다른 위치
     ↓
[중앙 정렬 후]
모든 뷰에서 객체가 중앙에 위치
```

### 카메라 파라미터 왜곡 문제
**Q: 중앙 정렬하면 카메라 파라미터가 왜곡되지 않나?**

**A: 문제 없음!**

이유:
1. MVDiffusion은 **고정 카메라 배치**를 가정하고 학습
2. 출력 이미지도 **동일한 고정 카메라** 가정
3. GS-LRM도 **고정 카메라 파라미터** 사용
4. 실제 카메라 파라미터는 MVDiffusion→GS-LRM 파이프라인에서 사용되지 않음

```python
# inference_mouse.py:891-893
# Prepare data for GSLRM (use fixed camera poses)
# MVDiffusion uses similar camera setup to Zero123++
c2ws, fxfycxcys = compute_mvdiffusion_cameras(...)
```

---

## 4. 실행 계획

### Phase 1: 데이터 전처리 (중앙 정렬)
```bash
# data_mouse 전체에 중앙 정렬 적용
python scripts/preprocess_center_align.py \
    --input_dir data_mouse \
    --output_dir data_mouse_centered \
    --num_samples 2000
```

전처리 내용:
- 각 샘플의 View 0 (cam_000.png)를 기준으로 bbox 계산
- 동일한 scale/offset을 모든 뷰에 적용
- 결과: 모든 뷰에서 객체가 중앙에 위치

### Phase 2: MVDiffusion 재학습
```bash
# 중앙 정렬된 데이터로 MVDiffusion 재학습
python train_diffusion.py \
    --config configs/mouse_mvdiffusion_centered.yaml
```

예상 학습 시간: ~24시간 (A100 기준)

### Phase 3: 합성 데이터 생성
```bash
# 새 MVDiffusion으로 합성 데이터 생성
python scripts/generate_gslrm_training_data.py \
    --mvdiff_checkpoint checkpoints/mvdiffusion/mouse_centered/... \
    --input_dir data_mouse_centered \
    --output_dir data_mouse_synthetic_centered
```

### Phase 4: GS-LRM 학습
```bash
# 합성 데이터로 GS-LRM 학습
python train_gslrm.py \
    --config configs/mouse_gslrm_synthetic_centered.yaml
```

---

## 5. 파일 구조

```
data_mouse/                    # 원본 (중앙 정렬 안됨)
├── sample_000000/
│   ├── images/
│   │   ├── cam_000.png       # 마우스가 왼쪽에 있을 수 있음
│   │   ├── cam_001.png       # 마우스가 오른쪽에 있을 수 있음
│   │   └── ...
│   └── opencv_cameras.json   # 실제 카메라 파라미터

data_mouse_centered/           # 중앙 정렬됨
├── sample_000000/
│   ├── images/
│   │   ├── cam_000.png       # 마우스가 중앙
│   │   ├── cam_001.png       # 마우스가 중앙
│   │   └── ...
│   └── opencv_cameras.json   # 동일 (변경 불필요)
```

---

## 6. 핵심 코드 참조

### 중앙 정렬 함수
```python
# scripts/preprocess_mouse_for_mvdiffusion.py
def center_align_with_background_removal(
    image: np.ndarray,
    output_size: int = 512,
    target_object_ratio: float = 0.6,  # 객체가 이미지의 60% 차지
    background_color: Tuple = (255, 255, 255),
) -> Tuple[np.ndarray, dict]:
    # 1. 배경 제거 (이미 SimpleClick으로 처리됨)
    # 2. Bounding box 계산
    # 3. Scale + Center 적용
    # 4. 변환 파라미터 반환 (다른 뷰에 동일하게 적용)
```

### 고정 카메라 생성
```python
# inference_mouse.py
def compute_mvdiffusion_cameras(image_size=512, camera_distance=2.7):
    MVDIFFUSION_AZIMUTHS = [0, 60, 120, 180, 240, 300]  # 균등 배치
    fov = 50.0
    # 6개 뷰 모두 동일한 intrinsic
```

---

## 7. 검증 방법

### 중앙 정렬 검증
```python
# 모든 뷰에서 마우스 중심이 (256, 256) 근처인지 확인
for view in views:
    bbox = find_mouse_bbox(view)
    center = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
    assert abs(center[0] - 256) < 50
    assert abs(center[1] - 256) < 50
```

### MVDiffusion 출력 검증
- 입력: 중앙 정렬된 View 0
- 출력: 6개 뷰 모두 마우스가 중앙에 위치
- Grid 이미지로 시각화하여 확인

### GS-LRM 복원 검증
- Turntable 비디오에서 ghosting 없이 단일 마우스만 보이는지 확인
- PSNR > 20 이상

---

## 8. 실행 현황 (2025-12-19)

### 완료
- [x] 중앙 정렬 전처리 스크립트 작성 (`scripts/preprocess_center_align_all_views.py`)
- [x] 알파 채널 기반 bbox 감지 구현
- [x] 테스트 데이터 (10개 샘플) 전처리 성공

### 진행 중
- [ ] 전체 데이터셋 (2000개 샘플) 중앙 정렬 전처리
  - 진행률: ~12% (246/2000)
  - 예상 완료: ~25분

### 대기
- [ ] MVDiffusion 재학습 (`configs/mouse_mvdiffusion_centered.yaml`)
- [ ] 합성 데이터 생성
- [ ] GS-LRM 학습

---

## 9. 참고 사항

### Segmentation 방식 비교
| 방식 | 사용처 | 정확도 |
|------|--------|--------|
| SimpleClick | data_mouse 원본 | 매우 높음 (human-in-the-loop) |
| rembg | FaceLift Human | 높음 (사람 최적화) |
| rembg (마우스) | 테스트용 | 보통 (장비 artifact 포함) |

### 주의사항
- MVDiffusion 재학습 시 prompt embedding도 동일하게 사용
- GS-LRM은 pretrained 모델에서 fine-tuning
- 학습 데이터 수: 최소 1000개 이상 권장
