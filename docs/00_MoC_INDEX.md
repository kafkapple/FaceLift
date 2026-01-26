# FaceLift Mouse Documentation Hub

> **최종 업데이트**: 2026-01-27
> **원칙**: 각 정보는 한 곳에만 존재 (SSOT), 나머지는 링크

---

## Quick Navigation

| 필요한 것 | 문서 | 설명 |
|-----------|------|------|
| **데이터셋 권장** | [[datasets/DATASET_MASTER_REFERENCE]] | ⭐ M3_2 권장, 실패 원인 |
| **전처리 프리셋** | [[datasets/PREPROCESSING_REGISTRY]] | 프리셋 상세, 명령어 |
| **실험 빠른 시작** | [[EXPERIMENT_QUICKSTART]] | 실험 우선순위, GPU 배정 |
| **가설 검증** | [[datasets/HYPOTHESIS_VERIFICATION]] | H1-H6 상태 |
| **PP/fx 이론** | [[theory/PP_FX_MVG_ANALYSIS]] | Ray Error, 수식 |
| **시각화 설정** | [[VISUALIZATION_SETTINGS]] | Turntable, 색상 |

---

## 현재 권장 설정 (2026-01-27)

| 항목 | 권장 | 이유 |
|------|------|------|
| **데이터셋** | M3_2 | PP=256, fx=549, Coverage ~5% |
| **실험** | E1_2_alpha | GT mask + alpha 0.1 |
| **가설 검증** | M3_2b, M3_3 | Coverage ↑ → PSNR ↑ 검증 |

```bash
# Quick Start (권장)
torchrun --standalone --nproc_per_node=1 train_gslrm.py -b gslrm_mouse -d M3_2 -e E1_2_alpha
```

---

## 문서 구조

```
docs/
├── 00_MoC_INDEX.md           ← 이 문서 (네비게이션)
├── EXPERIMENT_QUICKSTART.md  ← 실험 빠른 시작
├── datasets/                 ← 데이터셋 SSOT
│   ├── DATASET_MASTER_REFERENCE.md  # ⭐ 메인 가이드
│   ├── PREPROCESSING_REGISTRY.md    # 프리셋 상세
│   ├── M3_SERIES_SPEC.md
│   └── HYPOTHESIS_VERIFICATION.md
├── theory/                   ← 이론/분석
│   ├── PP_FX_MVG_ANALYSIS.md # PP/fx 종합
│   ├── camera/               # 카메라 이론
│   └── mask/                 # 마스크 이론
├── practical/                ← 실무 가이드
│   └── MOUSE_QUICK_REFERENCE.md
├── tutorials/                ← 튜토리얼
└── _archive/                 ← 구버전
```

---

## 핵심 원칙

### PP (Principal Point) 규칙
```
✅ 성공: zoom_center_mode = "image" → PP = 256 고정
❌ 실패: zoom_center_mode = "object" → PP 가변 → Ray Error
```

### 데이터셋 권장 순위
1. **M3_2** - Per-sample zoom, PP=256 ⭐
2. **M3_1** - Global zoom, PP=256 (안전)
3. **D7_1/D8** - 기준선

---

*MoC v4.1 | 2026-01-27 | SSOT Navigation*
