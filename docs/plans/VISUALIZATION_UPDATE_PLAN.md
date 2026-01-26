# 시각화 및 전처리 업데이트 계획

> **작성일**: 2026-01-25
> **목적**: Turntable 시각화 개선, Alpha mask 시각화 추가, M3 시리즈 문서 정합성
> **우선순위**: P0 (즉시)

---

## 1. 요청사항 요약

### 1.1 문서/설정 일관성 (Task A)
- [ ] M3_1, M3_2 프리셋 관련 모든 문서 업데이트
- [ ] Quick Reference 업데이트
- [ ] Dataset config 파일 생성

### 1.2 Alpha Mask 시각화 (Task B)
- [ ] alpha_loss 사용 시 rendered alpha 시각화 추가
- [ ] GT mask와 rendered alpha 비교 이미지
- [ ] Train/Val 모두 적용
- [ ] WandB 로깅

### 1.3 Turntable 영상 개선 (Task C)

#### C-1: Train 기본 방식 복원
- [ ] Val과 동일한 기존 방식으로 별도 저장
- [ ] 현재 train turntable의 뷰 분리 문제 해결

#### C-2: 카메라 순서 변경
- [ ] 실제 물리적 배치 순서 적용: 0→4→2→1→3→5
- [ ] 시계방향 360도 회전 순서

#### C-3: 프레임 속도 조정
- [ ] 2배 이상 느리게
- [ ] 카메라 간 보간 렌더링 (extrinsic 기반)

#### C-4: Grid 이미지 개선
- [ ] Row 기준 카메라 ID
- [ ] 각 카메라 구간 6등분
- [ ] 구간: 0→4, 4→2, 2→1, 1→3, 3→5, 5→0

---

## 2. 카메라 배치 순서

### 2.1 물리적 배치 (시계방향)

```
        cam 0 (정면)
           |
    cam 5  |  cam 4
      \    |    /
       \   |   /
        \  |  /
         \ | /
   cam 3 --●-- cam 2
           |
        cam 1
```

### 2.2 회전 순서
```
cam 0 → cam 4 → cam 2 → cam 1 → cam 3 → cam 5 → (cam 0)
```

### 2.3 Grid Layout (6 rows x 6 cols)

| Row | 구간 | Col 0 | Col 1 | ... | Col 5 |
|-----|------|-------|-------|-----|-------|
| 0 | 0→4 | cam0 | interp | ... | ~cam4 |
| 1 | 4→2 | cam4 | interp | ... | ~cam2 |
| 2 | 2→1 | cam2 | interp | ... | ~cam1 |
| 3 | 1→3 | cam1 | interp | ... | ~cam3 |
| 4 | 3→5 | cam3 | interp | ... | ~cam5 |
| 5 | 5→0 | cam5 | interp | ... | ~cam0 |

---

## 3. 구현 계획

### Phase 1: 문서 정합성 (10분)
1. MOUSE_QUICK_REFERENCE.md 업데이트
2. configs/datasets/M3_1.yaml, M3_2.yaml 생성
3. EXPERIMENT_REGISTRY.md에 M3_1, M3_2 추가

### Phase 2: Alpha Mask 시각화 (20분)
1. 기존 mask 비교 코드 분석
2. alpha_loss_weight > 0 조건부 시각화 추가
3. Train/Val 공통 함수 생성
4. WandB 이미지 로깅

### Phase 3: Turntable 기본 복원 (15분)
1. Val turntable 코드 분석
2. Train에 동일 방식 추가 (별도 파일)

### Phase 4: 카메라 순서 및 보간 (30분)
1. 카메라 순서 상수 정의
2. Extrinsic 보간 함수 구현 (slerp for rotation, lerp for translation)
3. MP4 프레임 속도 조정
4. Grid 레이아웃 변경

---

## 4. 파일 위치

| 파일 | 역할 |
|------|------|
| mouse_extensions/visualization/turntable.py | Turntable 렌더링 |
| mouse_extensions/visualization/alpha_vis.py | Alpha mask 시각화 (NEW) |
| train_gslrm.py | 메인 학습 루프 |
| configs/datasets/M3_1.yaml | M3_1 데이터셋 설정 (NEW) |
| configs/datasets/M3_2.yaml | M3_2 데이터셋 설정 (NEW) |

---

*Implementation Plan v1.0 | 2026-01-25*
