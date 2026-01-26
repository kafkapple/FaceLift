# FaceLift Mouse Documentation Hub

> **최종 업데이트**: 2026-01-26
> **원칙**: 각 정보는 한 곳에만 존재, 나머지는 링크

---

## Quick Navigation

| 필요한 것 | 문서 | 설명 |
|-----------|------|------|
| **데이터셋 선택** | [[datasets/VERSION_SCHEMA]] | M-Series, D-Series 전체 |
| **전처리 프리셋** | [[datasets/PREPROCESSING_REGISTRY]] | 프리셋 정의, 명령어 |
| **실험 결과** | [[datasets/EXPERIMENT_RESULTS]] | PSNR 비교표 |
| **가설 검증** | [[datasets/HYPOTHESIS_VERIFICATION]] | H1-H6 상태 |
| **명령어** | [[practical/MOUSE_QUICK_REFERENCE]] | 전처리/학습/검증 |
| **PP/fx 이론** | [[theory/PP_FX_MVG_ANALYSIS]] | Ray Error, 수식 |
| **시각화 설정** | [[VISUALIZATION_SETTINGS]] | Turntable, 색상 |

---

## 현재 권장 설정

| 항목 | 권장 | 이유 |
|------|------|------|
| **데이터셋** | M3_2 | PP=256, fx=549, Coverage ~6% |
| **실험** | E1_2_alpha | GT mask + alpha 0.1 |

```bash
# Quick Start
torchrun --standalone --nproc_per_node=1 train_gslrm.py -d M3_2 -e E1_2_alpha
```

---

## 문서 구조

```
docs/
├── 00_MoC_INDEX.md        ← 이 문서 (네비게이션만)
├── datasets/              ← SSOT (Single Source of Truth)
│   ├── VERSION_SCHEMA.md      # 모든 데이터셋 버전
│   ├── PREPROCESSING_REGISTRY.md  # 프리셋 상세
│   ├── EXPERIMENT_RESULTS.md  # 실험 결과 수치
│   └── HYPOTHESIS_VERIFICATION.md # 가설 검증 매트릭스
├── theory/                ← 이론/분석
│   └── PP_FX_MVG_ANALYSIS.md  # PP/fx 수식, 원리
├── practical/             ← 실무 가이드
│   └── MOUSE_QUICK_REFERENCE.md  # 명령어 모음
└── _archive/              ← 구버전 보관
```

---

*MoC v4.0 | 2026-01-26 | Navigation Only*
