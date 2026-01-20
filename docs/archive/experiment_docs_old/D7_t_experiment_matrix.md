# D7_t Experiment Matrix

> 체계적 가설 검증을 위한 실험 설계
> Dataset: D7_t (Temporal 1:1:1 split, PoseSplatter 방식)

## 전체 실험 설정 비교표

| ID | Views | Random | Mask | α Threshold | α Loss | 핵심 변수 |
|----|-------|--------|------|-------------|--------|----------|
| **E1.1** | 4 | ✅ | none | - | - | Paper baseline |
| **E1.2** | 4 | ❌ | none | - | - | Fixed control |
| **E2.1** | 4 | ❌ | rgb_pred | - | - | Self-supervised mask |
| **E2.2** | 4 | ❌ | gt | - | - | Oracle mask |
| **E2.3** | 4 | ❌ | alpha | 0.5 | - | Rendered alpha mask |
| **E3.2** | 5 | ❌ | alpha | 0.5 | - | More input views |
| **E4.2** | 5 | ❌ | alpha | 0.5 | 0.1 | Direct α supervision |
| **E5.1** | 5 | ✅ | alpha | 0.5 | - | Combined best |

---

## 가설별 비교 설계

### H1: Random View Selection Effect
> "Random view selection이 fixed보다 일반화 성능을 향상시키는가?"

| 실험 | 비교 | 차이점 | 예상 |
|------|------|--------|------|
| **E1.1** vs **E1.2** | random vs fixed | view selection만 다름 | E1.1 > E1.2 |

**검증 메트릭**: val/psnr, test/psnr (temporal generalization)

---

### H2: Mask Method Comparison
> "어떤 mask 방법이 foreground 품질에 최적인가?"

| 실험 | Mask | 특징 | 예상 순위 | Priority |
|------|------|------|----------|----------|
| **E1.2** | none | No mask (baseline) | 4th | P1 |
| **E2.1** | rgb_pred | Self-supervised (RGB prediction) | 3rd | **P2** |
| **E2.3** | alpha | Self-supervised (rendered α) | 2nd | **P2** |
| **E2.2** | gt | Oracle (GT mask) | 1st | **P2** |

**비교 체인**: E1.2 < E2.1 < E2.3 ≈ E2.2

**핵심 질문**:
- Alpha mask가 GT mask에 얼마나 근접하는가?
- Self-supervised (α) vs Oracle (gt) 성능 gap?
- RGB pred vs Alpha: 어떤 self-supervised 방식이 더 효과적인가?

---

### H3: View Count Effect
> "Input view 수 증가가 novel view 품질을 향상시키는가?"

| 실험 | Views | Input:Novel | 예상 |
|------|-------|-------------|------|
| **E2.3** | 4 | 4:2 (67%) | Baseline |
| **E3.2** | 5 | 5:1 (83%) | E3.2 > E2.3 |

**Trade-off**: 더 많은 input → 더 적은 novel view supervision

---

### H4: Alpha Loss Effect
> "Direct alpha supervision이 mask 품질을 개선하는가?"

| 실험 | α Loss | 기대 효과 |
|------|--------|----------|
| **E3.2** | ❌ | Baseline (mask만 사용) |
| **E4.2** | ✅ 0.1 | 배경 gaussian 억제 |

**검증 메트릭**: mask_iou, background artifact 시각적 검토

---

### H5: Combined Best
> "최적 설정들의 조합이 시너지를 내는가?"

| 실험 | 구성 | 비교 대상 |
|------|------|----------|
| **E5.1** | 5v + random + alpha | E3.2 (fixed), E1.1 (4v) |

**예상**: E5.1 ≥ max(E3.2, E1.1)

---

## 비교 다이어그램

```
Paper Baseline (E1.1: 4v, random, none)
         │
         ├──[H1: random vs fixed]──→ E1.2 (4v, fixed, none)
         │                              │
         │                              ├──[H2: mask method]──→ E2.1 (rgb_pred)
         │                              │                       E2.2 (gt) ← Oracle
         │                              │                       E2.3 (alpha) ★
         │                              │                           │
         │                              │                           └──[H3: views]──→ E3.2 (5v, alpha)
         │                              │                                                  │
         │                              │                                                  └──[H4: α loss]──→ E4.2
         │                              │
         └──────────────────────────────┴─────────────[H5: combined]──→ E5.1 (5v, random, alpha)
```

---

## 실험 우선순위 및 명령어

### Priority 1: Baseline 확립 (E1.1 + E1.2)

```bash
# E1.1: TRUE Paper Baseline (4v, random, no mask)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E1_1_paper_random.yaml > logs/d7t_e1_1_paper_random.log 2>&1 &

# E1.2: Fixed Control
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E1_2_paper_fixed.yaml > logs/d7t_e1_2_paper_fixed.log 2>&1 &
```

### Priority 2: 4-View Alpha (E2.3) - H2/H3 비교 기준

```bash
# E2.3: 4v Alpha Mask (mask method baseline for H2, view count baseline for H3)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E2_3_alpha_mask.yaml > logs/d7t_e2_3_alpha_mask.log 2>&1 &
```

### Priority 3: 5-View Alpha (E3.2) - 기존 Best 재현

```bash
# E3.2: 5v Alpha (기존 best 재현, H3/H4 baseline)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E3_2_5v_alpha.yaml > logs/d7t_e3_2_5v_alpha.log 2>&1 &
```

### Priority 4: Mask Method Oracle (E2.2)

```bash
# E2.2: GT Mask (H2 oracle upper bound)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E2_2_gt_mask.yaml > logs/d7t_e2_2_gt_mask.log 2>&1 &
```

### Priority 5: Alpha Loss (E4.2)

```bash
# E4.2: 5v Alpha + Alpha Loss (H4 검증)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E4_2_5v_alpha_loss.yaml > logs/d7t_e4_2_5v_alpha_loss.log 2>&1 &
```

### Priority 6: Combined Best (E5.1)

```bash
# E5.1: 5v Alpha + Random (H5 final candidate)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E5_1_5v_alpha_random.yaml > logs/d7t_e5_1_5v_alpha_random.log 2>&1 &
```

### Optional: RGB Pred Mask (E2.1)

```bash
# E2.1: RGB Pred Mask (H2 추가 비교)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E2_1_rgb_mask.yaml > logs/d7t_e2_1_rgb_mask.log 2>&1 &
```

---

## 예상 결과 분석

### 성공 시나리오

```
H1 검증: E1.1 > E1.2 (random 효과 확인)
H2 검증: E2.2 > E2.3 > E2.1 > E1.2 (mask hierarchy 확인)
H3 검증: E3.2 > E2.3 (more views 효과 확인)
H4 검증: E4.2 ≥ E3.2 (alpha loss 효과 확인)
H5 검증: E5.1 ≥ E3.2 (combined best 확인)

Final Best: E5.1 or E4.2
```

### 대안 시나리오

- H1 실패 (E1.1 ≈ E1.2): Mouse 데이터에서는 view selection 영향 적음
- H3 실패 (E3.2 ≈ E2.3): 5v의 추가 view가 큰 이득 없음
- H4 실패 (E4.2 ≈ E3.2): Alpha loss 불필요

---

## WandB 필터링

```
Group: D7_t
Experiments: E1_1, E1_2, E2_1, E2_2, E2_3, E3_2, E4_2, E5_1

Key Metrics:
- val/psnr (primary)
- val/lpips (perceptual)
- val/mask_iou (mask quality)
- experiment.split_method: "temporal"
```

---

*Created: 2026-01-19*
*Last Updated: 2026-01-19*
