# Experiment Registry (실험 레지스트리)

> **Navigation**: [← MoC](../00_MoC_INDEX.md) | [HYPOTHESIS_PLAN](./HYPOTHESIS_EXPERIMENT_PLAN.md) | [QUICK_START](./QUICK_START.md)

> **SSOT**: 모든 실험 ID, 명령어, 설정의 중앙 관리 문서
> **최종 업데이트**: 2026-01-25

---

## 명명 규칙

```
E{카테고리}_{번호}_{이름}                    - 기본 설정
E{카테고리}_{번호}_{서브번호}_{이름}_{변형}   - 변형 설정

예시:
E0_1_facelift                   - E0 카테고리, 1번, FaceLift 기본
E0_1_1_facelift_alpha           - E0_1_facelift의 1번 변형 (alpha 추가)
E1_2_3_alpha_fixed              - E1_2_alpha의 3번 변형 (fixed view)
```

---

## 카테고리 체계

| 카테고리 | mask_mode | 설명 | 상태 |
|----------|-----------|------|------|
| **E0** | none | FaceLift Baseline | ⭐ **권장** |
| **E1** | gt | GT Mask 기반 | 활성 |
| **E2** | none | Alpha only | 실험적 |

---

## E0: FaceLift Baseline ⭐

> mask_mode=none, FaceLift 논문 기반

### 기본 설정

| ID | 파일 | alpha | view | 설명 |
|----|------|-------|------|------|
| **E0_1_facelift** | `E0_1_facelift.yaml` | 0.0 | random | FaceLift 논문 원본 |
| E0_2_mouse | `E0_2_mouse.yaml` | 0.0 | random | Mouse-adapted (lr 조정) |

### E0_1 변형

| ID | 파일 | alpha | view | 변형 내용 |
|----|------|-------|------|----------|
| E0_1_1_facelift_alpha | `E0_1_1_facelift_alpha.yaml` | **0.1** | random | +alpha |
| E0_1_2_facelift_fixed | `E0_1_2_facelift_fixed.yaml` | 0.0 | **fixed** | +fixed |
| E0_1_3_facelift_alpha_fixed | `E0_1_3_facelift_alpha_fixed.yaml` | **0.1** | **fixed** | +alpha+fixed |

---

## E1: GT Mask

> mask_mode=gt, RGB loss를 GT mask 영역으로 제한

### 기본 설정

| ID | 파일 | alpha | 설명 |
|----|------|-------|------|
| E1_1_base | `E1_1_base.yaml` | 0.0 | GT mask만 |
| **E1_2_alpha** | `E1_2_alpha.yaml` | 0.1 | GT mask + alpha |
| E1_3_lgm | `E1_3_lgm.yaml` | 1.0 | LGM 스타일 |

### E1_2 변형

| ID | 파일 | 변형 내용 |
|----|------|----------|
| E1_2_1_alpha_3v | `E1_2_1_alpha_3v.yaml` | 3 view input |
| E1_2_2_alpha_5v | `E1_2_2_alpha_5v.yaml` | 5 view input |
| E1_2_3_alpha_fixed | `E1_2_3_alpha_fixed.yaml` | fixed view selection |
| E1_2_4_alpha_overfit | `E1_2_4_alpha_overfit.yaml` | overfit 테스트 |

---

## E2: Alpha Only

> mask_mode=none, alpha supervision만

| ID | 파일 | alpha | 설명 |
|----|------|-------|------|
| E2_1_alpha | `E2_1_alpha.yaml` | 0.1 | 마스크 없이 alpha만 |

---

## 데이터셋

### Production

| Alias | Coverage | fx | PP | Val PSNR | 설명 |
|-------|----------|-----|-----|----------|------|
| **D3_normalized** | **84%** | 549 | 256 | **27.1** | ⭐ SOTA |
| M1 (D7_1) | 50% | 549 | 256 | 20.9 | Baseline |
| M2 (D8) | 50% | 549 | 256 | 20.2 | Homography |

### 실험용

| Alias | Coverage | fx | PP | 가설 |
|-------|----------|-----|-----|------|
| M3 | 78% | **739** ❌ | 가변 | ⛔ 사용금지 |
| ~~M3_norm~~ | 78% | 549 | **가변** | ⛔ 폐기 (ray 13.62°) |
| ~~M3_persample~~ | 50% | 549 | **가변** | ⛔ 폐기 (ray 16.15°) |
| **M3_1** | 80% | 549 | **256** | ✅ MVG-correct (Global) |
| **M3_2** | 80% | 549 | **256** | ✅ MVG-correct (Per-sample) ⭐ |

---

## 실행 명령어

### 기본 형식

```bash
CUDA_VISIBLE_DEVICES={GPU} torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d {DATASET} -e {EXPERIMENT}
```

### 권장 실험 (D3_normalized)

```bash
# SOTA 확인
-d D3_normalized -e E0_1_facelift

# +Alpha
-d D3_normalized -e E0_1_1_facelift_alpha

# +Fixed
-d D3_normalized -e E0_1_2_facelift_fixed

# +Alpha+Fixed
-d D3_normalized -e E0_1_3_facelift_alpha_fixed
```

---

## 변경 이력

| 날짜 | 버전 | 변경 |
|------|------|------|
| 2026-01-25 | v2.4 | 명명규칙 통일 (E{cat}_{num}_{subnum}_{name}_{variant}) |
| 2026-01-25 | v2.3 | E0_paper_* 추가 (폐기) |
| 2026-01-25 | v2.2 | random_view_selection 버그 수정 |

---

*Experiment Registry v2.4 | 2026-01-25*
