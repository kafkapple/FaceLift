# Experiment Registry (중앙 실험 관리)

> **SSOT (Single Source of Truth)**: 모든 실험 ID, 명령어, 설정의 중앙 관리 문서
> 다른 문서에서는 이 문서를 백링크로 참조할 것
>
> **최종 업데이트**: 2026-01-24

---

## Quick Navigation

- [실험 카테고리](#실험-카테고리)
- [실험 ID 목록](#실험-id-목록)
- [데이터셋 ID 목록](#데이터셋-id-목록)
- [실행 명령어](#실행-명령어)
- [관련 문서](#관련-문서)

---

## 실험 카테고리

| 카테고리 | 설명 | 권장 |
|----------|------|------|
| **E0** | Baseline (mask=none) | E0_1_facelift |
| **E1** | GT Mask + Alpha (권장) | **E1_2_gt_alpha** ⭐ |
| **E2** | Alpha Only | E2_1_alpha |
| **E3** | Advanced (composite, bg) | E3_6_clean |

---

## 실험 ID 목록

### E0: Baseline

| ID | 파일 | mask_mode | alpha_loss | 설명 |
|----|------|-----------|------------|------|
| E0_1_facelift | E0_1_facelift.yaml | none | 0.0 | FaceLift 원본 논문 설정 |
| E0_2_mouse | E0_2_mouse.yaml | none | 0.0 | Mouse 데이터 기본 설정 |

### E1: GT Mask (⭐ 권장)

| ID | 파일 | mask_mode | alpha_loss | 설명 |
|----|------|-----------|------------|------|
| E1_1_gt | E1_1_gt.yaml | gt | 0.0 | GT mask만 사용 |
| **E1_2_gt_alpha** ⭐ | E1_2_gt_alpha.yaml | gt | 0.1 | **Production 권장** |
| E1_2_gt_alpha_3v | E1_2_gt_alpha_3v.yaml | gt | 0.1 | 3 view 변형 |
| E1_2_gt_alpha_5v | E1_2_gt_alpha_5v.yaml | gt | 0.1 | 5 view 변형 |
| E1_2_gt_alpha_fixed | E1_2_gt_alpha_fixed.yaml | gt | 0.1 | 고정 view 선택 |
| E1_2_gt_alpha_overfit | E1_2_gt_alpha_overfit.yaml | gt | 0.1 | Overfit 테스트용 |
| E1_3_gt_alpha_lgm | E1_3_gt_alpha_lgm.yaml | gt | 1.0 | LGM 논문 설정 |

### E2: Alpha Only

| ID | 파일 | mask_mode | alpha_loss | 설명 |
|----|------|-----------|------------|------|
| E2_1_alpha | E2_1_alpha.yaml | none | 0.1 | Alpha loss만 |
| E2_2_alpha_mask | E2_2_alpha_mask.yaml | alpha | 0.1 | ⚠️ 피드백 루프 위험 |

### E3: Advanced

| ID | 파일 | mask_mode | 설명 |
|----|------|-----------|------|
| E3_1_composite | E3_1_composite.yaml | composite | GT×α 혼합 |
| E3_2_composite_strong | E3_2_composite_strong.yaml | composite | 강화 버전 |
| E3_3_bg | E3_3_bg.yaml | gt | Background penalty |
| E3_4_bg_strong | E3_4_bg_strong.yaml | gt | BG penalty 강화 |
| E3_5_combined | E3_5_combined.yaml | gt | 복합 설정 |
| E3_6_clean | E3_6_clean.yaml | gt | 깔끔한 배경 |

---

## 데이터셋 ID 목록

### M-Series (권장)

| Alias | ID | Preset | 특징 | 상태 |
|-------|-----|--------|------|------|
| M1 | D7_1 | D7.1 | Affine, PP-centered | ✅ 검증됨 |
| M2 | D8 | D8 | Homography + skew | ✅ 안정 |
| **M3** | **D10_3** | D10.3 | Coverage zoom, 3597샘플 | ✅ **Ready** |

### Legacy

| ID | Preset | 상태 | 비고 |
|----|--------|------|------|
| D7 | D7 | ⚪ | D7_1 권장 |
| D8_2 | D8.2 | ⚪ | 특수 용도 |
| D9 | D9 | ⚠️ | Native, 미정규화 |

---

## 실행 명령어

### 기본 형식 (Modular Mode)

```bash
CUDA_VISIBLE_DEVICES={GPU} torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d {DATASET_ID} -e {EXPERIMENT_ID}
```

### 권장 조합

| 우선순위 | Dataset | Experiment | 명령어 |
|----------|---------|------------|--------|
| **P0** | D10_3 | E1_2_gt_alpha | train_gslrm.py -d D10_3 -e E1_2_gt_alpha |
| P1 | D7_1 | E1_2_gt_alpha | train_gslrm.py -d D7_1 -e E1_2_gt_alpha |
| P2 | D10_3 | E1_3_gt_alpha_lgm | train_gslrm.py -d D10_3 -e E1_3_gt_alpha_lgm |

### 병렬 실행 예시

```bash
# GPU 0-3: M3 실험
CUDA_VISIBLE_DEVICES=0 nohup torchrun ... -d D10_3 -e E1_2_gt_alpha > logs/D10_3_E1_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup torchrun ... -d D10_3 -e E1_1_gt > logs/D10_3_E1_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup torchrun ... -d D10_3 -e E0_2_mouse > logs/D10_3_E0_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup torchrun ... -d D10_3 -e E1_3_gt_alpha_lgm > logs/D10_3_E1_3.log 2>&1 &
```

---

## 설정 파일 위치

| 유형 | 경로 |
|------|------|
| Base | configs/base/gslrm_mouse.yaml |
| Dataset | configs/datasets/{ID}.yaml |
| Experiment | configs/experiments/{ID}.yaml |

---

## 관련 문서

| 문서 | 위치 | 내용 |
|------|------|------|
| Quick Reference | [MOUSE_QUICK_REFERENCE.md](./MOUSE_QUICK_REFERENCE.md) | 전체 워크플로우 |
| 전처리 레지스트리 | [datasets/PREPROCESSING_REGISTRY.md](./datasets/PREPROCESSING_REGISTRY.md) | 데이터셋별 전처리 |
| 마스크 설정 | [config/MASK_SYSTEM_GUIDE.md](./config/MASK_SYSTEM_GUIDE.md) | 마스크 옵션 상세 |
| 마스크 이론 | [../theory/mask/](../theory/mask/) | 문헌 근거 |

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|-----------|
| 2026-01-24 | v1.0 | 초기 생성, E0-E3 체계 정립 |

---

*FaceLift Experiment Registry v1.0 | 2026-01-24*
