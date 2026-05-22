---
version: 1.1
created: 2026-04-17
last_validated: 2026-04-17
next_review: 2026-05-01
status: canonical
role: single_entry_moc
---

# FaceLift SSOT — Canonical Entry

> **Purpose**: FL 수치·설계·caveat 인용 시 **단일 진입점**.
> **Scope**: Mouse M5t2 (C57BL/6, 6 overhead cam, 3,600 frames). Rat = PoC only.
> **Precedence**: 이 파일 ↔ 다른 파일 충돌 시, 여기 `canonical_source` 포인터가 이김. 이 파일은 pointer + caveat, raw data 아님.
> **Owner**: `gpu03:/home/joon/dev/FaceLift/docs/FACELIFT_SSOT.md`.

---

## 1. Role Matrix

| Query | Canonical file | 유형 |
|:---|:---|:---:|
| 🔢 **Numbers** (all ablations, fair eval) | `docs/experiments/MASTER_RESULTS_TABLE.md` | SSOT |
| 📊 **Stat.sig / limitation narrative** | `docs/experiments/UNIFIED_ABLATION_REPORT.md` | SSOT |
| 🎛️ **Alpha loss mechanism** | `docs/experiments/ALPHA_LOSS_NOVEL_VIEW_ANALYSIS.md` (⚠️§2.1 OBSOLETE) | SSOT |
| 📦 **Preprocessing / dataset** | `docs/datasets/PREPROCESSING_REGISTRY.md` | SSOT |
| 🧪 **Config ↔ checkpoint 매핑** | `docs/experiments/EXPERIMENT_REGISTRY.md` v8.0 | SSOT |
| 🗂️ **Full navigation** | `docs/INDEX.md` | MoC |
| 🧭 **Project conventions** | `CLAUDE.md` (root) | Instr |
| 📐 **Theory / paper strategy** | Obsidian `30_Projects/_CODES/FaceLift/docs/INDEX.md` v7.6 | Secondary MoC |
| 🎨 **Visual embedding / feature dim SSOT** | `docs/specs/VISUAL_EMBEDDING_SSOT.md` (🆕 260418) | SSOT |
| 📝 **ICML AI4Science Workshop plan** (D-6) | `docs/experiments/ICML_CLUSTERING_MODULE_PLAN.md` (🆕 260418) | Plan |
| 🗄️ **Results MoC** (per-species artifacts) | `~/results/FaceLift/INDEX.md` v4 | MoC |

---

## 2. Canonical Numbers (verified 2026-04-17, cross-file grep PASS)

### 2.1 Fair eval (M5t2 test, n=1800) — MASTER §1-2

| Config | PSNR_gt | IoU | SSIM |
|:---|:---:|:---:|:---:|
| **GS-LRM 6v (GT input)** ⭐ | **23.84** | 0.954 | 0.963 |
| GS-LRM 6v α=0.3 | 23.29 | **0.956** | 0.961 |
| GS-LRM 4v (GT input) | 20.66 | 0.926 | 0.928 |
| **E2E best** (e2_resume) | **8.20** | 0.521 | 0.761 |
| **Pose-Splatter** (M5) | **13.78** | 0.846 | — |

**View count monotonic**: 10.47 → 15.95 → 18.56 → 20.66 → 22.16 → **23.84** (n=1800 each). 1v degenerate (IoU=0.028, model predicts 전체 FG).

**FL 6v > PS by +10.06 dB**. E2E bottleneck = MVDiff (**86% of quality gap**).

### 2.2 Paper-specific protocol (ICML main.tex:L280)
**24.26** = 6v best-fit fg PSNR (n=2160, view 0 included) — **§2.1과 다른 eval**. Source pointer 부재 (D2 pending).

### 2.3 Hypothesis status
H4 (view) ✅ / H5 (MVDiff) ✅ / H6 (α loss) ✅ / H7 (SSIM) ❌ / H8 (opacity) ✅ / HP (preprocessing) ✅ / H_T1 (dist mismatch) 🔬 testing.
Details: MASTER §1-4 + `experiments/hypothesis_roadmap.md`.

---

## 3. Mandatory Caveats — 수치 인용 시 반드시 동반

### C1. View ablation = CONDITIONAL upper bound
전 view-count 실험이 `random_view_selection=true` → 학습 중 6-view geometry optimizer에 누설. 0/6 SOTA가 이 패턴 미사용. **N-view 값은 pure N-view training의 upper bound**. Phase 1 fixed-view validation 대기. (Source: MASTER §1 L68-83)

**Paper Limitations 템플릿**:
> Training used within-batch random view sampling from 6 cameras; reported N-view performance is an upper bound on pure N-view training. Fixed-view validation is planned.

### C2. Alpha checkpoint asymmetry
Baseline = val-best, α variants = fixed-step 15840 → 체계적 baseline 유리. Publishable 전 baseline 재평가 필요 (eval ~1h). (MASTER §2a L119)

### C3. Split semantics
Val vs test PSNR 다름, frame-level vs camera-level holdout 다름. **인용 시 split type 명시 필수** (rat v6 retraction 260407).

### C4. PSNR_wh inflation
Full-image PSNR은 98% white BG로 +10-13 dB 인플레이션. Cross-model 비교는 **PSNR_gt만**. (MASTER §0 L26-36)

---

## 4. Cross-Project Usage

### 4.1 ICML AI4Science Workshop
Paper thesis: "PoseSplatter vs FaceLift Partial Decoupling". D-7 (Apr 21→24 AoE).

**인용 Rules**:
1. 수치: source protocol 명시 (`n=1800 fair` or `n=2160 paper`). main.tex에 `% Source:` 주석 권고 (**D2**)
2. View ablation 표 (Appendix D): **C1 caveat 본문 삽입 필수** (**D1 pending**)
3. α ablation: C2 미해소 → absolute PSNR 비교 지양. **Gaussian quality metrics** (α entropy 2.97→0.73, opacity suppression)는 drift-free → PS opacity=0.0123 대비 "architecture-specific divergence" 증거로 사용 가능
4. 제출 시: `paper/FROZEN_NUMBERS_YYMMDD.md` 수동 snapshot

### 4.2 BehaviorSplatter
- Data: `~/data/preprocessed/FaceLift_mouse/M5/` 공유
- Pretrained ckpt: `/node_data/joon/checkpoints/FaceLift/gslrm/ckpt_0000000000021125.pt` (α=0 6v baseline 상응)
- BS-side PSNR SSOT: `losses/metrics.py::compute_psnr`

### 4.3 Rat (PoC only — not paper)
- **M1 (sdannce s4_d1)**: v9-v12 closed, billboard 한계 확정 (13.9° baseline)
- **Rat 7M (dannce)**: v13 kill, **v13.1 P1 pivot** (PP-only normalize) 대기. Handoff: `260416_1700_facelift_rat7m_p1_pivot.md`

---

## 5. Drift Ledger (2026-04-17)

| # | Status | File | Issue / Fix |
|:-:|:-:|:---|:---|
| D1 | 🔴 pending | `paper/main.tex` | random_view caveat 부재 (grep 0건) → ICML 세션에서 C1 삽입 |
| D2 | 🔴 pending | `paper/main.tex:L280` | 24.26 source pointer 부재 → `% Source:` 주석 |
| D3-D9 | ✅ fixed 260417 | — | 서버 INDEX 9.04→8.20, CLAUDE v7.0→v7.6, Obsidian INDEX L30 v7.4→v7.6, ALPHA §2.1 OBSOLETE stamp, UNIFIED §2 WARNING mirror, SERVER_KEY_REFERENCES → `_archive/`, FL vs PS JSON merged verified |

**Cross-file consistency**: 23.84 / 20.66 / 8.20 / 13.78 / 0.954 / 0.956 = 6/6 PASS (grep verified, drift 없음).

---

## 6. Entry Mechanisms (fragility 낮은 순)

1. **paper.tex inline `% Source:` + caveat** — 제출 후 submission hash로 immutable (🟢 Low)
2. **FACELIFT_SSOT.md § 1 Role Matrix** — 세션 진입 1-file (🟡 Med) ← this
3. **FL `docs/INDEX.md` ⭐ ENTRY POINT link** (🟡 Med) ✅ Added
4. **FL `CLAUDE.md` ⭐ Canonical Entry link** (🟡 Med) ✅ Added
5. **ICML Workshop `.claude/CLAUDE.md`** (🟡 Med) ⏸️ post-D-0
6. **Obsidian INDEX 서버 SSOT link** (🟢 Low) ✅ Added

**Enforcement 한계**: L1만 immutable. 나머지는 지시 + convention (Devil 수용).

---

## 7. Stale Check

- Last validated: 2026-04-17 (9 후보 파일 grep + 6 수치 cross-check PASS)
- Next review: 2026-05-01 또는 (신규 실험 merge / 제출 / drift ≥ 5 누적)
- Drift detection: `grep` 기반 수치 consistency weekly (상세 script는 `_archive/` 참고)

---

## 8. --explain v2

**Why**: 9 후보 문서에 분산된 수치·caveat·drift를 세션마다 재발견하는 silent cost 제거. 특히 `random_view_selection` slip 같은 methodology trap을 세션 진입 시 명시적으로 가시화.

**Pattern**: "1 file = role matrix + inline caveats + drift ledger + explicit precedence rule"

**Worked example**:
```
9 ambiguous files, precedence implicit, drifts scattered
    ↓ 1 canonical entry + cross-file invariant 검증
FACELIFT_SSOT → (by query) → MASTER/UNIFIED/PREPROC
              + C1-C4 forced visibility + D1-D9 tracked (7 fixed / 2 cross-session)
```

**Delta**:
- 진입: implicit 9-file search → **explicit 1-file entry**
- Methodology trap: silent → **§3 forced + paper template**
- Drift state: scattered → **§5 central ledger with status**
- **Don't confuse with**: MASTER_RESULTS_TABLE = raw numbers SSOT. FACELIFT_SSOT = entry MoC. 진입은 항상 FACELIFT_SSOT.

---

*FaceLift SSOT v1.1 | 2026-04-17 | Simplified from v1.0*
