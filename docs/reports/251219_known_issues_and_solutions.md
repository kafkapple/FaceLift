---
date: 2025-12-19
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, debugging, troubleshooting, camera-normalization]
project: mouse-facelift
status: living-document
generator: ai-assisted
generator_tool: claude-code
---

# Mouse-FaceLift: 알려진 이슈 및 해결책 종합

> **Living Document**: 프로젝트 진행 중 발생한 모든 이슈와 해결책을 기록합니다.
> 새로운 이슈 발생 시 반드시 이 문서에 추가하세요.
>
> 📌 **전체 연구 흐름**: [[000_MoC_Mouse_FaceLift]] 참조

---

## 이슈 우선순위 요약

| 우선순위 | 이슈 | 상태 | 영향도 |
|---------|------|------|--------|
| 🔴 P0 | 카메라 거리 불일치 (2.0~3.4 vs 2.7 고정) | ✅ 해결됨 | Critical |
| 🔴 P0 | 이미지-카메라 정보 불일치 (합성 데이터) | ✅ 해결됨 | Critical |
| 🔴 P0 | 뷰 순서 랜덤화 → 흐릿한 출력 | ✅ 해결됨 | Critical |
| 🟠 P1 | num_input_views 설정 오류 | ✅ 해결됨 | High |
| 🟠 P1 | Perceptual Loss 도메인 불일치 | ✅ 해결됨 | High |
| 🟠 P1 | Masked SSIM 음수 loss | ✅ 해결됨 | High |
| 🟡 P2 | 이미지 중앙 정렬 누락 | ✅ 해결됨 | Medium |
| 🟡 P2 | Z-up vs Y-up 좌표계 혼동 | ⚠️ 모니터링 | Medium |
| 🟡 P2 | 이미지 Clipping (CoM 무시) | ✅ 분석완료 | Medium |
| 🟢 P3 | 데이터 수 부족 (51개 → 2000개) | ✅ 해결 중 | Low |

---

## 🔴 P0: Critical Issues

### Issue 1: 카메라 거리 불일치

**발견일**: 2025-12-19
**증상**: GS-LRM 학습 시 white prediction (mode collapse)
**문서**: `251218_research_camera_normalization_issue.md`

#### 문제
```
FaceLift Human: 카메라 거리 2.7 고정
Mouse 원본: 카메라 거리 2.0 ~ 3.4 가변
```

GS-LRM은 Plucker ray 좌표로 3D 위치를 계산하는데, 카메라 거리가 다르면 잘못된 위치에 Gaussian이 생성됨.

#### 해결책
```bash
# 카메라 정규화 스크립트 실행
python scripts/normalize_cameras_to_facelift.py \
    --input_dir data_mouse_centered \
    --output_dir data_mouse_normalized \
    --target_distance 2.7
```

#### 해결 원리
1. 모든 카메라의 viewing ray가 만나는 점(scene center) 계산
2. 카메라 위치를 scene center 기준으로 재배치
3. 모든 카메라를 정확히 거리 2.7로 정규화

#### 관련 파일
- `scripts/normalize_cameras_to_facelift.py` - 정규화 스크립트
- `configs/mouse_gslrm_synthetic.yaml` - `normalize_distance_to: 0.0` 설정

---

### Issue 2: 이미지-카메라 정보 불일치 (합성 데이터)

**발견일**: 2025-12-19
**증상**: 합성 데이터로 학습 시 white prediction
**연관**: Issue 1과 동일 근본 원인

#### 문제
```
합성 데이터 생성 시:
- MVDiffusion 출력: FaceLift 표준 뷰 (거리 2.7 가정)
- opencv_cameras.json: 마우스 원본 카메라 정보 (거리 2.0~3.4)
→ 이미지와 카메라 정보 불일치!
```

#### 해결책
정규화된 데이터(`data_mouse_normalized`)에서 합성 데이터 생성:
```bash
python scripts/generate_synthetic_data.py \
    --input_dir data_mouse_normalized \
    --output_dir data_mouse_synthetic_normalized
```

이렇게 하면 `opencv_cameras.json`에 거리 2.7로 정규화된 카메라 정보가 복사됨.

---

## 🟠 P1: High Priority Issues

### Issue 3: num_input_views 설정 오류

**발견일**: 2025-12-18
**증상**: 학습 불안정, PSNR 정체 (13-15 dB)
**문서**: `251218_research_gslrm_finetune_debugging.md`

#### 문제
```yaml
# 잘못된 설정
num_views: 6
num_input_views: 1  # 1개 입력 → 5개 예측 (너무 어려움)

# Pretrained 모델 기대
num_views: 32
num_input_views: 6  # 6개 입력 → 2개 예측
```

#### 해결책
```yaml
# 올바른 설정
num_views: 6
num_input_views: 5  # 5개 입력 → 1개 예측 (pretrained와 유사)
```

#### 효과
- PSNR: 13-15 → 20-23 dB (약 7dB 개선)

---

### Issue 4: Perceptual Loss 도메인 불일치

**발견일**: 2025-12-18
**증상**: Gradient explosion (grad_norm > 100)
**문서**: `251218_research_gslrm_finetune_debugging.md`

#### 문제
- LPIPS와 Perceptual Loss는 VGG (ImageNet/Human 학습) 기반
- Mouse body는 VGG 학습 도메인 밖
- Out-of-distribution 입력 → 큰 gradient 발생

#### 해결책
```yaml
losses:
  l2_loss_weight: 1.0
  lpips_loss_weight: 0.0      # 비활성화
  perceptual_loss_weight: 0.0  # 비활성화
  ssim_loss_weight: 0.5        # 유지 (안정적)
```

#### 효과
- Gradient explosion 완전 제거
- 학습 안정성 확보

---

## 🟡 P2: Medium Priority Issues

### Issue 5: 이미지 중앙 정렬 누락

**발견일**: 2025-12-19
**증상**: MVDiffusion/GS-LRM이 객체 위치 불일치로 혼란
**문서**: `251219_mouse_facelift_camera_alignment.md`

#### 문제
```
Human FaceLift: MTCNN으로 얼굴 감지 → 중앙 정렬
Mouse: 각 뷰마다 마우스 위치가 다름
→ MVDiffusion, GS-LRM 모두 객체가 중앙에 있다고 가정
```

#### 해결책
```bash
python scripts/preprocess_center_align_all_views.py \
    --input_dir data_mouse \
    --output_dir data_mouse_centered \
    --target_ratio 0.6
```

#### 원리
1. View 0의 alpha 채널에서 bbox 계산
2. 동일한 scale/offset을 모든 뷰에 적용
3. 결과: 모든 뷰에서 객체가 이미지 중앙에 위치

---

### Issue 6: Z-up vs Y-up 좌표계 혼동

**발견일**: 2025-12-18
**증상**: Blurry output (초기 의심 원인)
**문서**: `251218_research_camera_normalization_issue.md`

#### 분석 결과
```
Human 데이터: Z-up (Up vector = [0, 0, 1])
Mouse 데이터: ~Z-up (Up vector = [0, 0.015, 1])
→ 거의 일치, 근본 원인 아님
```

#### 현재 상태
- 카메라 거리 정규화가 더 중요한 문제로 판명
- Z-up 정규화는 추가 적용하지 않음
- 모니터링 유지

---

## 🟢 P3: Low Priority Issues

### Issue 7: 데이터 수 부족

**발견일**: 2025-12-19
**증상**: 빠른 overfitting

#### 해결책
- 전체 2000개 샘플 사용
- 합성 데이터로 데이터 증강

---

## 데이터 파이프라인 체크리스트

### 전처리 단계
- [ ] 원본 비디오 → FaceLift 형식 변환 (`convert_markerless_to_facelift.py`)
- [ ] 이미지 중앙 정렬 (`preprocess_center_align_all_views.py`)
- [ ] 카메라 거리 정규화 2.7 (`normalize_cameras_to_facelift.py`)

### 학습 단계
- [ ] MVDiffusion 학습 (중앙 정렬된 데이터 사용)
- [ ] 합성 데이터 생성 (정규화된 카메라 사용)
- [ ] GS-LRM 학습 (합성 데이터 사용)

### 설정 체크리스트
```yaml
# GS-LRM 필수 설정
training:
  dataset:
    num_input_views: 5          # NOT 1!
    normalize_distance_to: 0.0  # 이미 정규화된 데이터 사용

  losses:
    lpips_loss_weight: 0.0      # Mouse 도메인에서 비활성화
    perceptual_loss_weight: 0.0 # Mouse 도메인에서 비활성화
    ssim_loss_weight: 0.5       # 안정적
```

---

## 잠재적 이슈 (모니터링 필요)

### 1. MVDiffusion 품질
- 현재 checkpoint-2000 사용 중
- 더 긴 학습이 필요할 수 있음

### 2. 6뷰 제한
- Human FaceLift: 32뷰
- Mouse: 6뷰 (MAMMAL 카메라 수)
- 3D 복원 품질에 영향 가능

### 3. Elevation 분포
- Human: -70° ~ +70° (균등)
- Mouse: -51° ~ +78° (불균등)
- 특정 각도에서 품질 저하 가능

---

## 문서 업데이트 이력

| 날짜 | 변경 내용 |
|------|-----------|
| 2025-12-19 | 초기 문서 작성, P0-P3 이슈 정리 |

---

*🤖 Generated with Claude Code*
