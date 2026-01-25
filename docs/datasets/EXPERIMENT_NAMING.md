# Experiment Naming Convention

> **Navigation**: [← Index](./00_INDEX.md) | [Results](./EXPERIMENT_RESULTS.md)
> **SSOT**: 모든 실험 설정의 명명 규칙 및 체계

---

## 1. 실제 실험 파일 목록

```
configs/experiments/
├── E0_1_facelift.yaml          # FaceLift 원본 설정
├── E0_1_1_facelift_alpha.yaml  # + alpha loss
├── E0_1_2_facelift_fixed.yaml  # + fixed view
├── E0_1_3_facelift_alpha_fixed.yaml
├── E0_2_mouse.yaml             # Mouse 특화
├── E1_1_base.yaml              # 기본 설정
├── E1_2_alpha.yaml             # ⭐ GT mask + alpha (권장)
├── E1_2_1_alpha_3v.yaml        # 3 view
├── E1_2_2_alpha_5v.yaml        # 5 view
├── E1_2_3_alpha_fixed.yaml     # fixed view
├── E1_2_4_alpha_overfit.yaml   # overfit 테스트
├── E1_3_lgm.yaml               # LGM 설정 (alpha=1.0)
├── E2_1_alpha.yaml             # Alpha only (no mask)
├── E_debug.yaml                # 디버깅용
├── E_exclude_view4.yaml        # View 4 제외
└── E_overfit_6v.yaml           # 6 view overfit
```

---

## 2. 명명 규칙

### 형식
```
E{카테고리}_{번호}_{설명}.yaml

예시:
- E1_2_alpha  → E1 카테고리, 2번, alpha 관련
- E0_1_facelift → E0 카테고리, 1번, facelift 원본
```

### 카테고리

| 카테고리 | 의미 | 대표 실험 |
|----------|------|-----------|
| **E0** | Baseline (원본 설정) | E0_1_facelift |
| **E1** | GT Mask 계열 | **E1_2_alpha** ⭐ |
| **E2** | Alpha Only 계열 | E2_1_alpha |
| **E_** | 특수 목적 | E_debug, E_overfit_6v |

---

## 3. 핵심 실험 비교

| Config | mask_mode | alpha_loss | 설명 |
|--------|-----------|------------|------|
| **E1_2_alpha** | **gt** | 0.1 | ⭐ **권장** |
| E2_1_alpha | none | 0.1 | Alpha만 |
| E1_3_lgm | gt | 1.0 | LGM 설정 |
| E0_1_facelift | none | 0.0 | 원본 |

### E1_2_alpha.yaml (권장)
```yaml
training:
  losses:
    mask_mode: gt              # RGB loss → GT mask 영역
    normalize_by_mask: true    # 작은 전경 보정
    alpha_loss_weight: 0.1     # alpha 수렴 유도
```

---

## 4. 우선순위

| Priority | 데이터셋 | 실험 | 목적 |
|----------|---------|------|------|
| **P0** | M3_2 | **E1_2_alpha** | 27+ PSNR 목표 ⭐ |
| **P1** | M3_1 | E1_2_alpha | Global zoom 비교 |
| **P2** | D7_1 | E1_2_alpha | Affine baseline |
| **P3** | D8 | E1_2_alpha | Homography 검증 |

---

## 5. 실행 명령어

### P0: 권장 실험
```bash
cd /home/joon/dev/FaceLift

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E1_2_alpha
```

### P1: 대안 실험
```bash
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_1 -e E1_2_alpha
```

---

## 6. View Count 변형

| Config | Input Views | Target Views |
|--------|-------------|--------------|
| E1_2_alpha | 4 | 2 |
| E1_2_1_alpha_3v | 3 | 3 |
| E1_2_2_alpha_5v | 5 | 1 |
| E_overfit_6v | 6 | 0 |

---

## 7. 관련 문서

- [[00_INDEX]] - Dataset Hub
- [[EXPERIMENT_RESULTS]] - 실험 결과 비교
- [[HYPOTHESIS_VERIFICATION]] - 가설 검증
- [[VIEW_SELECTION_ANALYSIS]] - View 선택 분석

---

*Experiment Naming Convention v1.1 | 2026-01-26 | Fixed to match actual files*
