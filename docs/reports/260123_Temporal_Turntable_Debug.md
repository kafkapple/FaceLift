# Temporal Turntable Video Debug Report

> **Date**: 2026-01-23
> **Status**: 🔴 In Progress
> **Issue**: 생쥐가 보이지 않고 기하학적 색상만 표시

---

## 1. 현상

| 증상 | 설명 |
|------|------|
| **No Mouse** | 비디오에 생쥐가 전혀 보이지 않음 |
| **Strange Colors** | 기하학적 곡선, 색상들만 표시 (raw Gaussian 같음) |
| **No Temporal Change** | 시간에 따른 변화 없음 |

---

## 2. 생성된 파일

| 파일 | 위치 | 용도 |
|------|------|------|
| video_utils.py | `mouse_extensions/utils/video_utils.py` | CV2 기반 MP4 인코딩 |
| temporal_turntable.py | `mouse_extensions/scripts/inference/temporal_turntable.py` | 추론 스크립트 |

---

## 3. 학습 vs 추론 데이터 로딩 비교

### 학습 시 (Training)

**파일**: `gslrm/data/mouse_dataset.py`

```python
# 학습 시 데이터 로딩 (예상)
- DataLoader가 batch를 구성
- image, c2w, fxfycxcy, index 등 edict 형식
- batch.images: [B, V, 3, H, W]
- batch.c2w: [B, V, 4, 4]
- batch.fxfycxcy: [B, V, 4]
```

### 추론 시 (Inference - temporal_turntable.py)

```python
def load_single_sample(sample_dir, sample_idx, device):
    # opencv_cameras.json에서 카메라 파라미터 로드
    # images/*.png에서 이미지 로드
    
    return edict(
        image=images,      # [1, V, 3, H, W]
        c2w=c2ws,         # [1, V, 4, 4]
        fxfycxcy=fxfycxcy, # [1, V, 4]
        index=index,       # [1, V, 2]
        bg_color=torch.tensor([1.0, 1.0, 1.0]),
    )
```

---

## 4. 가능한 원인 분석

### 4.1 데이터 형식 차이
- 학습 시 batch key 이름: `images` vs 추론 시: `image`
- 학습 시 추가 필드가 있을 수 있음

### 4.2 카메라 파라미터 차이
- intrinsics (fxfycxcy) 값의 범위
- extrinsics (c2w) 행렬 형식

### 4.3 Turntable 카메라 설정
- elevation: 20° (학습 범위: 10.7° ~ 30.8°)
- radius: 2.7 (정규화된 값)

### 4.4 Gaussian 출력 품질
- opacity 값 분포 확인 필요
- xyz 위치 범위 확인 필요

---

## 5. 디버깅 단계 (TODO)

1. [ ] 학습 데이터 로더 분석 (`gslrm/data/mouse_dataset.py`)
2. [ ] batch key 이름 및 형식 비교
3. [ ] 학습 시 turntable 생성 코드와 추론 코드 비교
4. [ ] Gaussian statistics 출력 (xyz, opacity, scaling)
5. [ ] 단일 뷰 렌더링 테스트
6. [ ] 학습 시 저장된 turntable.jpg와 비교

---

## 6. 참고 코드 위치

| 코드 | 파일 | 라인 |
|------|------|------|
| 학습 turntable 생성 | `gslrm/model/gslrm.py` | ~1561 |
| render_turntable | `gslrm/model/gaussians_renderer.py` | ~1175 |
| 학습 데이터 로더 | `gslrm/data/mouse_dataset.py` | TBD |
| 추론 데이터 로더 | `mouse_extensions/scripts/inference/temporal_turntable.py` | ~76 |

---

## 7. 테스트 영상 위치

```
outputs/temporal_turntable/test/D7_1_E1_1_paper_random_0000000000003400_temporal_single_angle_f0-5.mp4
```

---

*Updated: 2026-01-23*

---

## 8. 해결된 문제 (2026-01-23 11:14)

### 8.1 발견된 차이점

| 항목 | 학습 (mouse_dataset.py) | 기존 추론 | 수정 후 |
|------|-------------------------|----------|---------|
| **이미지 채널** | 4채널 (RGBA) | 3채널 (RGB) | ✅ 4채널 |
| **Alpha 생성** | 자동 (threshold 250) | 없음 | ✅ 자동 생성 |
| **카메라 정규화** | Z-up + 거리 2.7 | 없음 | ✅ 추가 |
| **Intrinsics 정규화** | 비례 조정 | 없음 | ✅ 추가 |

### 8.2 수정 내용

`load_single_sample()` 함수에 다음 추가:

```python
# 1. Alpha 채널 자동 생성
if auto_generate_mask and alpha is None:
    is_background = np.all(img_np > threshold, axis=2)
    alpha = (~is_background).astype(np.float32)[:, :, np.newaxis]
img_np = np.concatenate([img_np, alpha], axis=2)

# 2. 카메라 정규화
if normalize_to_z_up:
    c2ws_np = normalize_cameras_to_z_up(c2ws_np, up_direction=None)

# 3. 거리 정규화
if target_camera_distance > 0:
    c2ws_np, fxfycxcy_np = normalize_camera_distance_with_intrinsics(
        c2ws_np, fxfycxcy_np, target_camera_distance
    )
```

### 8.3 결과

- Gaussian 통계 정상: xyz [-2.17, 2.67], opacity [0, 1]
- high opacity (>0.5): ~55k (3.5%)
- 테스트 영상 생성 성공: `test2/*.mp4`

---

## 9. 핵심 교훈

> **학습과 추론의 데이터 전처리는 반드시 동일해야 함**
> - Alpha 채널 (마스크)
> - 카메라 정규화 (좌표계, 거리)
> - Intrinsics 조정

---

*Updated: 2026-01-23 11:14*
