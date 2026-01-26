# Experiment Results Registry

> **SSOT**: 실험 결과 비교표
> **최종 업데이트**: 2026-01-26

---

## Validation PSNR 비교

| Dataset | Transform | Zoom | PP | Val PSNR | 비고 |
|---------|-----------|------|-----|----------|------|
| **D3_normalized** | ? | ? | 256 | **27.09** | ⭐ 최고 (전처리 불명) |
| **D7_1** (M1) | Affine | ❌ | 256 | 20.93 | 기준선 |
| **D8** (M2) | Homography | ❌ | 256 | 20.21 | Affine과 유사 |
| M3_norm | Homo+Zoom | ✅ | **가변** | 17.09 | ❌ PP 불일치 |
| **M3_2** | Homo+Zoom | ✅ | 256 | TBD | ⏳ 검증 예정 ⭐ |

---

## 주요 발견

| 요인 | 영향 | 근거 |
|------|------|------|
| **PP=256 정합** | +3~4 PSNR | M3_norm(17) vs D7_1(21) |
| **Coverage** | +6 PSNR | D7_1(21) vs D3(27) |
| **Transform** | ~0.7 PSNR | Affine ≈ Homography |

---

## 실험 명령어

```bash
# 권장: M3_2 + E1_2_alpha
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1     train_gslrm.py -d M3_2 -e E1_2_alpha
```

---

## 관련 문서

- [[HYPOTHESIS_VERIFICATION]] - 가설 검증 (H1-H6)
- [[VERSION_SCHEMA]] - 데이터셋 버전
- [[../practical/MOUSE_QUICK_REFERENCE]] - 전체 명령어

---

*EXPERIMENT_RESULTS v4.0 | 2026-01-26 | 중복 제거*
