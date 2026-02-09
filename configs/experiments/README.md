# Experiment Configurations

> **중앙 레지스트리**: [`docs/experiments/EXPERIMENT_REGISTRY.md`](../../docs/experiments/EXPERIMENT_REGISTRY.md)

---

## 카테고리 체계 (v2.0)

| 카테고리 | RGB Mask | Alpha | 상태 |
|----------|----------|-------|------|
| **E0** | none | ❌ | Baseline |
| **E1** | gt | ✅ | ⭐ 권장 |
| **E2** | none | ✅ | 실험적 |
| **E3** | 기타 | - | deprecated |

---

## 활성 실험

```
E0_1_facelift.yaml    # 논문 원본
E0_2_mouse.yaml       # Mouse baseline
E1_1_base.yaml        # GT mask만
E1_2_alpha.yaml       # ⭐ Production
E1_2_alpha_*.yaml     # 변형 (3v, 5v, fixed, overfit)
E1_3_lgm.yaml         # LGM 스타일
E2_1_alpha.yaml       # Alpha only
```

## Deprecated

`_deprecated/` 폴더 참조

---

## 사용법

```bash
# Modular mode (권장)
train_gslrm.py -d M3 -e E1_2_alpha

# Override
train_gslrm.py -d M3 -e E1_2_alpha -s training.losses.alpha_loss_weight 0.2
```

---

*v2.0 | 2026-01-24*
