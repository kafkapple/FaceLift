# 실험 우선순위 (2026-01-25)

## 데이터셋 우선순위

| Priority | Dataset | Samples | 이유 |
|----------|---------|---------|------|
| **P0** | M3_2 | 3,597 | PP 버그 수정, per-sample zoom, 최신 |
| P1 | M3_1 | 3,597 | Global zoom, PP=256, 안정적 |
| P2 | D3_normalized | 3,600 | 기존 실험 비교용 |

## 실험 우선순위

### EXP-A: 6뷰 Overfit (최우선)

```bash
# M3_2 + E_overfit_6v
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E_overfit_6v
```

**목적**: 모든 뷰가 input이면 학습되는지 확인

### EXP-B: View 4 제외 (조건부)

```bash
# EXP-A에서 View 4 여전히 낮으면 실행
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E_exclude_view4
```

**목적**: View 4 데이터 자체가 문제인지 격리

### EXP-C: 기존 설정 비교 (베이스라인)

```bash
# 기존 4-input 방식과 비교
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E1_2_alpha
```

## 판단 기준

| 결과 | 해석 | 다음 단계 |
|------|------|-----------|
| EXP-A 모든 뷰 PSNR 25+ | Novel view 문제 확정 | Val random 적용 |
| EXP-A View 4 여전히 낮음 | 데이터 문제 가능 | EXP-B 실행 |
| EXP-B View 5 개선 | View 4가 원인 | View 4 제외 권장 |

## Config 파일

| Config | 용도 | num_input |
|--------|------|-----------|
| E_overfit_6v | 6뷰 모두 input | 6 |
| E_exclude_view4 | View 4 제외 | 4 (of 5) |
| E_debug | 빠른 테스트 | 4 |

---

*Updated: 2026-01-25*
