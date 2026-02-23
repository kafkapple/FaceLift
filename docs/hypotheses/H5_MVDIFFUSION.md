# H5: MV-Diffusion 학습 최적화

> **Terminology**: 'MVDiffusion'/'MVDiff' = FaceLift Stage 1 (SD2.1-UnCLIP + Era3D RMA).
> 코드 폴더 `mvdiffusion/`은 Era3D에서 상속된 명칭. Tang et al. MVDiffusion 논문과 별개.


> **가설**: Multi-view diffusion (Stage 1)을 mouse 데이터에 맞게 fine-tuning하면 E2E 품질이 개선될 것이다.
>
> ← [RESEARCH_HYPOTHESES.md](../RESEARCH_HYPOTHESES.md) | **상태**: 🔄 진행중 | **Updated**: 2026-02-15

---

## 1. 실험 설계 (v2.0 — 단일 변수 원칙)

### 1.1 Baseline

| 항목 | 값 |
|------|-----|
| Config | `mouse_mvdiffusion_M5t2.yaml` |
| sparse_mv_attention | **true** |
| reference_view_idx | **0** |
| max_train_steps | **10,000** |
| Checkpoint | `mouse_M5t2/checkpoint-5000` |
| Best PSNR (MVDiff) | 27.30 (CFG 3.0) |

### 1.2 실험 매트릭스

| Config | 변경 변수 | sparse_mv | ref_view | steps | LR | 상태 |
|--------|----------|:---------:|:--------:|:-----:|:--:|:----:|
| **M5t2** (baseline) | - | true | 0 | 10K | piecewise 5e-5 | ✅ ckpt-5000 |
| **M5t2_cfgr** | attention type | **false** | 0 | 10K | piecewise 5e-5 | ✅ 완료 |
| **M5t2_randref_sparse** (P0) | ref augmentation | true | **random** | 10K | piecewise 5e-5 | ✅ ckpt-10K |
| **M5t2_pose_spherical** (P1) | pose conditioning | true | **random** | 10K | piecewise 5e-5 | ✅ 완료 |
| **M5t2_20k_cosine** (E1) | LR + duration | true | 0 | **20K** | **cosine** 5e-5→0 | 🔄 GPU 4 |
| **M5t2_randref_20k_resume** (E2) | P0 resume + LR decay | true | **random** | **20K** | **piecewise** 1e-5 | 🔄 GPU 7 |
| **M5t2_pose_extrinsic_add** (E3) | pose encoding | true | **random** | 10K | **cosine** | ⏳ R1 후 |
| **M5t2_pose_spherical_add** (E4) | pose integration | true | **random** | 10K | **cosine** | ⏳ R1 후 |

### 1.3 의사결정 흐름

```
Phase 1: cfgr vs baseline E2E 비교
    │
    └─ baseline 승 (sparse=true 유지) ✅ 확인 (260209)
       │
       Phase 2B: P0 (randref), P1 (pose_spherical)
       │  ├─ P0: oscillation at constant LR 5e-5
       │  └─ P1: plateau after 5K, pose concat 효과 미미
       │
       Phase 3: LR 개선 + Pose 변형 (260215~)
       ├─ R1: E1 (cosine 20K) + E2 (P0 resume 1e-5)
       ├─ R2: E3 (extrinsic+add) + E4 (spherical+add)
       └─ 수렴 판단: 최근 2K step PSNR 변동 < 0.3 dB
```

### 1.4 ⚠️ 2변수 Config (실행 비권장)

| Config | 변경 1 | 변경 2 | 비고 |
|--------|--------|--------|------|
| M5t2_randref | sparse=false | ref=random | → randref_sparse 사용 |
| M5t2_symmetric | sparse=false | ref=[0,3] | 보류 |
| M5t2_20k | sparse=false | 20K steps | → 20k_sparse 사용 |

### 1.5 삭제된 실험

| Config | 이유 | 일자 |
|--------|------|------|
| M5t2_cyclic | 2변수 변경 (sparse=false + ref=[0..5]), 성능 저조 | 260207 |
| M5t2_consistent | cfgr과 유사 (condition_drop 차이만), 중복 | 260207 |

---

## 2. Phase 1-2 결과

### 2.1 Phase 1: Sparse vs Full Attention ✅

| Config | sparse_mv | PSNR_wh | vs Baseline |
|--------|-----------|---------|-------------|
| **baseline** | true (sparse) | **21.29** | - |
| cfgr | false (full) | 20.81 | **-0.48 dB** |

**판정**: Sparse attention이 더 우수 → sparse 유지

### 2.2 Phase 2B: P0 (randref) + P1 (pose_spherical) ✅

| 실험 | 관찰 | 문제점 |
|------|------|--------|
| P0 (randref_sparse, 10K) | PSNR ~27 도달했으나 oscillation | constant LR 5e-5, 수렴 불안정 |
| P1 (pose_spherical, 10K) | 5K 이후 plateau | concat integration 효과 미미, constant LR |

**근본 원인**: `step_rules: "1:100000,0.5"` = 10K 범위 내 LR decay가 전혀 없음.

---

## 3. Phase 3: LR 개선 + Pose 변형 (260215~)

### 3.1 설계 근거

P0/P1 실패 분석에서 도출:
1. **LR decay 부재** → E1(cosine), E2(piecewise 감소)
2. **Pose concat 비효율** → E3/E4(add integration)
3. **Pose encoding 대안** → E3(extrinsic 6D rot+trans)

### 3.2 Experiment Matrix

| # | 실험 | 핵심 변경 | LR | GPU | Round |
|:-:|------|----------|:--:|:---:|:-----:|
| **E1** | 20K cosine (새 학습) | cosine 5e-5→0, 20K | cosine | 4 | R1 |
| **E2** | P0 resume (ckpt-10K) | step_rules 1:10000,0.2 → LR=1e-5 | piecewise | 7 | R1 |
| **E3** | pose extrinsic + add | extrinsic 6D, add integration, cosine | cosine | 4 | R2 |
| **E4** | pose spherical + add | spherical (기존) + add (vs P1 concat), cosine | cosine | 7 | R2 |

### 3.3 LR 전략 비교

| 실험 | Scheduler | LR 궤적 | 근거 |
|------|-----------|---------|------|
| Baseline/P0/P1 | piecewise (5e-5 constant) | 5e-5 flat | step_rules 미트리거 |
| **E1** | **cosine** | 5e-5 → warmup → cosine → 0 | 표준 diffusion fine-tune |
| **E2** | **piecewise (수정)** | 5e-5 (1~10K) → 1e-5 (10K~20K) | P0 resume 안전성 |
| **E3/E4** | **cosine** | 5e-5 → warmup → cosine → 0 | E1 검증 후 표준화 |

### 3.4 Pose Encoding 비교

```
P1 (기존):  spherical + concat → extra cross-attn token
E3 (신규):  extrinsic + add    → 직접 hidden state injection
E4 (신규):  spherical + add    → P1과 integration만 다름
```

| 비교 | 격리 변수 | 기대 인사이트 |
|------|-----------|-------------|
| P1 vs E4 | concat vs add | Integration 방식 효과 |
| E3 vs E4 | extrinsic vs spherical | Pose encoding 방식 효과 |

### 3.5 Phase 3 실행 현황 (260215)

| 실험 | Step | LR | Loss | ETA |
|------|:----:|:--:|:----:|:---:|
| E1 | ~31/20K | warmup (1.5e-5) | 0.02~0.08 | ~60h |
| E2 | ~10054/20K | **1e-5** ✅ 전환 확인 | 0.007~0.025 | ~30h |
| E3 | - | - | - | R1 완료 후 |
| E4 | - | - | - | R1 완료 후 |

### 3.6 성공 기준

| 실험 | 기준 |
|------|------|
| E1 | val PSNR > 28.0 (Baseline 27.30 대비 +0.7), 안정 수렴 |
| E2 | 10K→20K 구간 PSNR 단조 증가, oscillation 제거 |
| E3/E4 | E2E fg_PSNR > 8.0, sIoU > 0.55 |

---

## 4. 실행 명령어

### Phase 1: E2E 비교 ✅ 완료

```bash
# → COMMANDS.md §2.5 Phase 1 참조
```

### Phase 2B: P0/P1 ✅ 완료

```bash
# → COMMANDS.md §2.5 Phase 2B 참조
```

### Phase 3: → COMMANDS.md §Phase 3 참조

---

## 5. 체크포인트 위치

Base: `/node_data/joon/checkpoints/FaceLift/mvdiffusion/`

| 실험 | 경로 | 상태 |
|------|------|:----:|
| M5t2 baseline | `mouse_M5t2/checkpoint-5000` | ✅ |
| M5t2_cfgr | `mouse_M5t2_cfgr/checkpoint-10000` | ✅ |
| P0 randref | `mouse_M5t2_randref_sparse/checkpoint-10000` | ✅ |
| P1 pose_spherical | `mouse_M5t2_pose_spherical/` | ✅ |
| **E1 20k_cosine** | `mouse_M5t2_20k_cosine/` | 🔄 학습중 |
| **E2 resume** | `mouse_M5t2_randref_sparse/` (재사용) | 🔄 학습중 |
| pipeckpts | `mvdiffusion/pipeckpts/` | ✅ |

---

## 6. 관련 문서

- [COMMANDS.md](../experiments/COMMANDS.md) — 명령어 SSOT
- [RESEARCH_HYPOTHESES.md](../RESEARCH_HYPOTHESES.md) — 전체 가설 맵
- [260215_EXPERIMENT_ANALYSIS.md](../experiments/260215_EXPERIMENT_ANALYSIS.md) — Phase 3 분석

---

*H5 MV-Diffusion | v3.0 | 2026-02-15*
