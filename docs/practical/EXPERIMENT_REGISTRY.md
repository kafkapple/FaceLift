# Experiment Registry (실험 레지스트리)

> **SSOT (Single Source of Truth)**: 모든 실험 ID, 명령어, 설정의 중앙 관리 문서
> **최종 업데이트**: 2026-01-24

---

## 카테고리 체계

| 카테고리 | RGB Mask | Alpha Loss | 설명 | 상태 |
|----------|----------|------------|------|------|
| **E0** | none | ❌ | Baseline | 활성 |
| **E1** | **gt** | ✅ | GT Mask 기반 | ⭐ **권장** |
| **E2** | none | ✅ | Alpha supervision only | 실험적 |
| **E3** | 기타 | 다양 | Composite, BG 등 | deprecated |

---

## E0: Baseline (no mask)

> RGB loss 전체 이미지, alpha supervision 없음

| ID | 파일 | 설명 |
|----|------|------|
| E0_1_facelift | `E0_1_facelift.yaml` | FaceLift 논문 원본 설정 |
| E0_2_mouse | `E0_2_mouse.yaml` | Mouse 데이터 baseline |

---

## E1: GT Mask ⭐ (권장)

> RGB loss를 GT mask 영역으로 제한 + alpha supervision

| ID | 파일 | alpha_loss | 설명 |
|----|------|------------|------|
| E1_1_base | `E1_1_base.yaml` | 0.0 | GT mask만 |
| **E1_2_alpha** ⭐ | `E1_2_alpha.yaml` | 0.1 | **Production 권장** |
| E1_3_lgm | `E1_3_lgm.yaml` | 1.0 | LGM 스타일 (강한 alpha) |

### E1_2 변형

| ID | 파일 | 설명 |
|----|------|------|
| E1_2_alpha_3v | `E1_2_alpha_3v.yaml` | 3 view |
| E1_2_alpha_5v | `E1_2_alpha_5v.yaml` | 5 view |
| E1_2_alpha_fixed | `E1_2_alpha_fixed.yaml` | 고정 view 선택 |
| E1_2_alpha_overfit | `E1_2_alpha_overfit.yaml` | Overfit 테스트용 |

---

## E2: Alpha Only (실험적)

> RGB loss 전체 이미지, alpha supervision만으로 형태 학습

| ID | 파일 | alpha_loss | 설명 |
|----|------|------------|------|
| E2_1_alpha | `E2_1_alpha.yaml` | 0.1 | 마스크 없이 alpha만 |

---

## E3: Experimental → Deprecated

> **사용 권장하지 않음** - `configs/experiments/_deprecated/` 참조

| ID | 문제점 |
|----|--------|
| E2_2_mask | Alpha mask 피드백 루프 위험 |
| E3_1~E3_6 | Composite/BG 실험, 검증 안됨 |

---

## 데이터셋 (M-Series)

> 분류: affine (M1) → homography (M2) → homography_zoom (M3)

| Alias | Config | 카테고리 | 설명 |
|-------|--------|----------|------|
| **M1** | `-d M1` (D7_1) | affine | 안정적 기준선 |
| **M2** | `-d M2` (D8) | homography | 정밀 기하학 |
| **M3** | `-d M3` (D10_3) | homography_zoom | ⭐ Production |

---

## 실행 명령어

### 기본 형식

```bash
CUDA_VISIBLE_DEVICES={GPU} torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d {DATASET} -e {EXPERIMENT}
```

### 권장 조합

| 우선순위 | 명령어 | 설명 |
|----------|--------|------|
| **P0** | `-d M3 -e E1_2_alpha` | ⭐ Production |
| P1 | `-d M1 -e E1_2_alpha` | 안정적 기준선 |
| P2 | `-d M3 -e E1_3_lgm` | 강한 alpha |

### 예시

```bash
# Production 권장
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3 -e E1_2_alpha

# Background로 실행
CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3 -e E1_2_alpha > logs/M3_E1_2_alpha.log 2>&1 &
```

---

## 파일 위치

| 유형 | 경로 |
|------|------|
| Base | `configs/base/gslrm_mouse.yaml` |
| Dataset | `configs/datasets/{M1,M2,M3}.yaml` |
| Experiment | `configs/experiments/E{0,1,2}_*.yaml` |
| Deprecated | `configs/experiments/_deprecated/` |

---

## 관련 문서

| 문서 | 내용 |
|------|------|
| [MOUSE_QUICK_REFERENCE](./MOUSE_QUICK_REFERENCE.md) | 전체 워크플로우 |
| [PREPROCESSING_REGISTRY](./datasets/PREPROCESSING_REGISTRY.md) | 데이터셋 전처리 |
| [Mask Theory](../theory/mask/) | 마스크 이론/문헌 |

---

## 변경 이력

| 날짜 | 버전 | 변경 |
|------|------|------|
| 2026-01-24 | v2.0 | 체계 재설계: gt_ 제거, deprecated 분리 |
| 2026-01-24 | v1.0 | 초기 생성 |

---

*Experiment Registry v2.0 | 2026-01-24*
