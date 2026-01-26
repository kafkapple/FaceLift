# Experiment Schema (실험 스키마)

> **SSOT**: 실험 명명 규칙, 카테고리 체계, Config 포맷 표준
> **최종 업데이트**: 2026-01-25

---

## 1. 명명 규칙 (Naming Convention)

### 1.1 기본 패턴

```
E{카테고리}_{번호}_{이름}                     - 기본 설정
E{카테고리}_{번호}_{서브번호}_{이름}_{변형}    - 변형 설정
```

### 1.2 구성 요소

| 요소 | 설명 | 예시 |
|------|------|------|
| **카테고리** | 0, 1, 2, ... | E**0**, E**1**, E**2** |
| **번호** | 카테고리 내 순번 | E0_**1**, E0_**2** |
| **이름** | 설정 식별자 | E0_1_**facelift**, E1_2_**alpha** |
| **서브번호** | 변형 순번 | E0_1_**1**, E1_2_**3** |
| **변형** | 변형 키워드 | E0_1_1_facelift_**alpha** |

### 1.3 예시

```
E0_1_facelift                    # 카테고리 0, 1번, FaceLift 기본
E0_1_1_facelift_alpha            # E0_1의 1번 변형: +alpha
E0_1_2_facelift_fixed            # E0_1의 2번 변형: +fixed
E0_1_3_facelift_alpha_fixed      # E0_1의 3번 변형: +alpha+fixed

E1_2_alpha                       # 카테고리 1, 2번, alpha 설정
E1_2_1_alpha_3v                  # E1_2의 1번 변형: 3 view
E1_2_2_alpha_5v                  # E1_2의 2번 변형: 5 view
E1_2_3_alpha_fixed               # E1_2의 3번 변형: fixed view
E1_2_4_alpha_overfit             # E1_2의 4번 변형: overfit test
```

---

## 2. 카테고리 체계 (Category System)

| 카테고리 | mask_mode | alpha | 설명 | 상태 |
|----------|-----------|-------|------|------|
| **E0** | none | 0.0 or 0.1 | FaceLift Baseline | ⭐ 권장 |
| **E1** | gt | 0.0~1.0 | GT Mask 기반 | 활성 |
| **E2** | none | 0.1 | Alpha Only | 실험적 |

### E0: FaceLift Baseline

- **핵심**: mask_mode=none (전체 이미지 학습)
- **출처**: FaceLift/GS-LRM 논문 원본
- **권장**: Val PSNR 가장 높음 (D7_1: 20.9)

### E1: GT Mask

- **핵심**: mask_mode=gt (foreground만 학습)
- **출처**: Pose Splatter + LGM
- **주의**: 과적합 경향 (Gap +10.3)

### E2: Alpha Only

- **핵심**: mask_mode=none + alpha supervision만
- **출처**: LGM 변형
- **상태**: 실험적

---

## 3. Config 파일 포맷 (File Format)

### 3.1 헤더 템플릿

```yaml
# =============================================================================
# {ID}: {짧은 설명}
# =============================================================================
# {상세 설명}
# - 주요 설정1
# - 주요 설정2
```

### 3.2 필수 섹션

```yaml
model:
  num_views: 6
  num_input_views: 4

training:
  dataset:
    random_view_selection: true/false

  losses:
    mask_mode: none/gt
    alpha_loss_weight: 0.0/0.1/1.0
    bg_loss_weight: 0.0
```

### 3.3 선택 섹션

```yaml
training:
  optimizer:           # lr 조정 필요시
  runtime:             # grad_clip 필요시
  schedule:            # max_steps 조정시

validation:
  enabled: true/false  # overfit 테스트시 false
```

---

## 4. 전체 Config 목록

### E0: FaceLift Baseline

| ID | alpha | view | 설명 |
|----|-------|------|------|
| **E0_1_facelift** | 0.0 | random | 논문 원본 ⭐ |
| E0_1_1_facelift_alpha | 0.1 | random | +alpha |
| E0_1_2_facelift_fixed | 0.0 | fixed | +fixed |
| E0_1_3_facelift_alpha_fixed | 0.1 | fixed | +alpha+fixed |
| E0_2_mouse | 0.0 | random | Mouse-adapted (lr 조정) |

### E1: GT Mask

| ID | alpha | 변형 | 설명 |
|----|-------|------|------|
| E1_1_base | 0.0 | - | GT mask만 |
| **E1_2_alpha** | 0.1 | - | GT mask + alpha |
| E1_2_1_alpha_3v | 0.1 | 3v | 3 view input |
| E1_2_2_alpha_5v | 0.1 | 5v | 5 view input |
| E1_2_3_alpha_fixed | 0.1 | fixed | fixed view |
| E1_2_4_alpha_overfit | 0.1 | overfit | 1 sample test |
| E1_3_lgm | 1.0 | - | LGM style (강한 alpha) |

### E2: Alpha Only

| ID | alpha | 설명 |
|----|-------|------|
| E2_1_alpha | 0.1 | mask 없이 alpha만 |

---

## 5. 변형 키워드 (Variant Keywords)

| 키워드 | 의미 | 적용 |
|--------|------|------|
| **alpha** | alpha_loss_weight > 0 | +alpha supervision |
| **fixed** | random_view_selection: false | 고정 뷰 선택 |
| **3v/5v** | num_input_views: 3/5 | view 수 변경 |
| **overfit** | 1 sample, validation off | overfit 테스트 |
| **lgm** | alpha_loss_weight: 1.0 | LGM 스타일 |

---

## 6. 신규 실험 추가 가이드

### 6.1 기본 설정 추가

```bash
# 1. 다음 번호 확인
ls configs/experiments/E{N}_*.yaml

# 2. 파일 생성
# E{N}_{다음번호}_{이름}.yaml
```

### 6.2 변형 설정 추가

```bash
# 1. 기존 변형 번호 확인
ls configs/experiments/E{N}_{M}_*.yaml

# 2. 파일 생성
# E{N}_{M}_{다음서브번호}_{이름}_{변형}.yaml
```

### 6.3 체크리스트

- [ ] 헤더 주석에 ID와 설명 포함
- [ ] model 섹션 포함 (num_views, num_input_views)
- [ ] training.dataset.random_view_selection 명시
- [ ] training.losses 섹션 완전히 명시
- [ ] EXPERIMENT_REGISTRY.md 업데이트

---

## 7. 관련 문서

| 문서 | 내용 |
|------|------|
| [EXPERIMENT_REGISTRY](./EXPERIMENT_REGISTRY.md) | 실험 목록 및 명령어 |
| [HYPOTHESIS_EXPERIMENT_PLAN](./HYPOTHESIS_EXPERIMENT_PLAN.md) | 가설 검증 계획 |
| [TRAINING_LOGGING_GUIDE](./TRAINING_LOGGING_GUIDE.md) | 학습/로깅 가이드 |

---

*Experiment Schema v1.0 | 2026-01-25*
