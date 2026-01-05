# 251219: 알려진 이슈 종합 및 카메라 정렬

**날짜:** 2025-12-19
**주제:** 프로젝트 이슈 종합 및 카메라 정렬 핵심 사항
**상태:** ✅ 이슈 정리 완료, Living Document

---

## 이슈 우선순위 요약

| 우선순위 | 이슈 | 상태 | 영향도 |
|---------|------|------|--------|
| 🔴 P0 | 카메라 거리 불일치 (2.0~3.4 vs 2.7) | ✅ 해결됨 | Critical |
| 🔴 P0 | 이미지-카메라 정보 불일치 (합성 데이터) | ✅ 해결됨 | Critical |
| 🟠 P1 | num_input_views 설정 오류 | ✅ 해결됨 | High |
| 🟠 P1 | Perceptual Loss 도메인 불일치 | ✅ 해결됨 | High |
| 🟡 P2 | 이미지 중앙 정렬 누락 | ✅ 해결됨 | Medium |
| 🟡 P2 | Z-up vs Y-up 좌표계 혼동 | ⚠️ 모니터링 | Medium |

---

## 1. P0: Critical Issues

### 카메라 거리 불일치
```
FaceLift Human: 카메라 거리 2.7 고정
Mouse 원본: 카메라 거리 2.0 ~ 3.4 가변
→ GS-LRM: 잘못된 Plucker ray → white prediction
```

**해결**: `preprocess_pixel_based.py`로 거리 2.7 정규화

### 합성 데이터 이미지-카메라 불일치
```
MVDiffusion 출력: FaceLift 표준 뷰 (거리 2.7 가정)
opencv_cameras.json: 마우스 원본 (거리 2.0~3.4) ❌
→ 해결: FaceLift 표준 카메라 생성 사용
```

---

## 2. P1: High Priority Issues

### num_input_views 설정 오류
```yaml
# Before (문제)
num_input_views: 1  # 1개 입력 → 5개 예측 (너무 어려움)

# After (해결) 
num_input_views: 5  # 5개 입력 → 1개 예측 (pretrained와 유사)
```

**효과**: PSNR 13-15 → 20-23 dB (약 7dB 개선)

### Perceptual Loss 도메인 불일치
```yaml
# 문제: VGG 기반 loss가 Mouse 도메인에서 gradient explosion

# 해결
losses:
  lpips_loss_weight: 0.0      # 비활성화
  perceptual_loss_weight: 0.0  # 비활성화
  ssim_loss_weight: 0.5        # 유지 (안정적)
```

---

## 3. P2: Medium Priority Issues

### 이미지 중앙 정렬 누락

**문제**: Human FaceLift는 MTCNN으로 얼굴 중앙 정렬, Mouse는 각 뷰마다 위치 다름

**해결**: `preprocess_center_align_all_views.py`
- View 0 기준 bbox 계산
- 동일 scale/offset을 모든 뷰에 적용

---

## 4. 카메라 정렬 핵심 사항

### FaceLift 파이프라인 설계
```
[Single View Input] → [MVDiffusion] → [6 Views] → [GS-LRM] → [3D]
                           ↓              ↓
                    중앙 정렬 가정    고정 카메라 사용
```

**핵심**: MVDiffusion과 GS-LRM 모두 객체가 이미지 중앙에 있다고 가정

### Human vs Mouse 카메라 비교
| 항목 | Human FaceLift | Mouse |
|------|----------------|-------|
| 카메라 배치 | Turntable (균등 60°) | MAMMAL (불균등) |
| 거리 | 2.7 | ✅ 정규화됨 |
| FOV | 50° | ✅ 정규화됨 |
| 객체 정렬 | MTCNN | ✅ pixel_based |

### 카메라 파라미터 왜곡 걱정?

**걱정 없음!**
1. MVDiffusion은 고정 카메라 배치 가정하고 학습
2. GS-LRM도 고정 카메라 파라미터 사용
3. 실제 카메라 파라미터는 파이프라인에서 사용되지 않음

---

## 5. 데이터 파이프라인 체크리스트

### 전처리 단계
- [x] 원본 비디오 → FaceLift 형식 변환
- [x] 이미지 중앙 정렬
- [x] 카메라 거리 정규화 2.7

### 학습 설정 체크리스트
```yaml
training:
  dataset:
    num_input_views: 5          # NOT 1!
    normalize_distance_to: 0.0  # 이미 정규화된 데이터

  losses:
    lpips_loss_weight: 0.0      # Mouse 도메인에서 비활성화
    perceptual_loss_weight: 0.0 # Mouse 도메인에서 비활성화
    ssim_loss_weight: 0.5       # 안정적
```

---

## 6. 잠재적 이슈 (모니터링 필요)

| 이슈 | 상태 |
|------|------|
| MVDiffusion 품질 | checkpoint-2000 사용 중, 더 긴 학습 필요 가능 |
| 6뷰 제한 | Human 32뷰 vs Mouse 6뷰, 복원 품질 영향 가능 |
| Elevation 분포 | Human (-70°~+70°) vs Mouse (-51°~+78°) 불균등 |

---

## 관련 파일
- `scripts/preprocess_pixel_based.py` - 중앙 정렬 + 카메라 정규화
- `scripts/normalize_cameras_to_facelift.py` - 카메라 거리 정규화
- `docs/reports/251219_known_issues_and_solutions.md` - 상세 이슈 목록

---

*통합 문서: 251219_known_issues_and_solutions.md + 251219_mouse_facelift_camera_alignment.md*
