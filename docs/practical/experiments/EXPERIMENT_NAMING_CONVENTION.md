# Experiment Naming Convention v2.0

**Created**: 2026-01-24
**Status**: Active

---

## 1. 기본 형식

```
{Dataset}_E{MaskMode}[_{SubNum}][_fixed]
```

| 요소 | 설명 | 예시 |
|------|------|------|
| `Dataset` | 데이터셋 ID | D7_1, D8, v13 |
| `E{MaskMode}` | 마스크 모드 번호 (E0-E5) | E2 |
| `_{SubNum}` | 서브 변형 (선택) | _1, _2 |
| `_fixed` | Fixed view 선택 시만 (선택) | _fixed |

---

## 2. Mask Mode 번호 체계 (E0-E5)

| 번호 | mask_mode | 문헌 | 설명 |
|------|-----------|------|------|
| **E0** | `none` | GS-LRM | Baseline (논문 기본) |
| **E1** | `gt` | Pose Splatter | GT 마스크 RGB loss |
| **E2** ⭐ | `gt_alpha_sup` | LGM + Pose Splatter | **권장** (GT + α supervision) |
| **E3** | `alpha_sup_only` | LGM | Alpha supervision만 |
| **E4** | `bg_penalty` | Obj-Centric 2DGS | Background penalty |
| **E5** | `composite` | Nerfstudio | Background compositing |

---

## 3. Sub-Numbering 규칙

### 3.1 View Count 변형
| Sub | Views | 설명 |
|-----|-------|------|
| _{없음} | 5v | 기본 (4 input + 1 target) |
| _4v | 4v | 4 views (3 input + 1 target) |
| _6v | 6v | 전체 6 views |

### 3.2 Parameter 변형
| Sub | 변형 | 예시 |
|-----|------|------|
| _a | Conservative | alpha_loss_weight: 0.05 |
| _b | Aggressive | alpha_loss_weight: 0.2 |
| _c | Custom | 기타 파라미터 변형 |

### 3.3 View Selection
| Suffix | 설명 |
|--------|------|
| (없음) | Random view selection (**기본**) |
| _fixed | Fixed view selection |

---

## 4. 이전 vs 현재 매핑

### 이전 (복잡, 불일치)
```
D7_1_E1_1_paper_random    ← random이 기본인데 명시
D7_1_E2_2_gt_mask         ← 넘버링 불일치
D7_mask_E1_gt_alpha_sup   ← 별도 폴더
```

### 현재 (단순, 일관)
```
D7_1_E0              ← Baseline (random)
D7_1_E0_fixed        ← Baseline (fixed views)
D7_1_E2              ← GT + Alpha Sup (random, 5v)
D7_1_E2_4v           ← GT + Alpha Sup (random, 4v)
D7_1_E2_fixed        ← GT + Alpha Sup (fixed, 5v)
```

---

## 5. 권장 실험 세트

### P0: 기본 비교 (Random)
```
D7_1_E0    # Baseline (none)
D7_1_E2    # GT + Alpha Sup ★ RECOMMENDED
D7_1_E3    # Alpha Sup Only
```

### P1: View Count Ablation
```
D7_1_E2       # 5v (default)
D7_1_E2_4v    # 4v
D7_1_E2_6v    # 6v
```

### P2: Mask Mode Ablation
```
D7_1_E0    # none
D7_1_E1    # gt (mask only, no alpha sup)
D7_1_E2    # gt + alpha sup
D7_1_E4    # bg_penalty
D7_1_E5    # composite
```

---

## 6. Config 파일 위치

```
configs/
├── base/                    # 공통 설정
│   ├── gslrm_base.yaml
│   └── mouse_data.yaml
├── datasets/                # 데이터셋별 설정
│   ├── D7_1.yaml
│   └── D8.yaml
└── experiments/             # 실험 설정 (E0-E5)
    ├── E0_baseline.yaml
    ├── E1_gt.yaml
    ├── E2_gt_alpha_sup.yaml   ★
    ├── E3_alpha_sup_only.yaml
    ├── E4_bg_penalty.yaml
    └── E5_composite.yaml
```

### 조합 사용법
```bash
# 직접 조합
python train_gslrm.py -d D7_1 -e E2

# 또는 완전한 config
python train_gslrm.py --config configs/mouse/D7_1_E2.yaml
```

---

## 7. 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|-----------|
| v2.0 | 2026-01-24 | 마스크 모드 기반 재설계, random 기본화 |
| v1.0 | 2026-01-21 | 초기 버전 (E1_1 스타일) |

---

*FaceLift Mouse | Experiment Naming Convention v2.0 | 2026-01-24*
