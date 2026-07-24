---
version: 1.2
created: 2026-04-17
last_validated: 2026-07-23
next_review: 2026-08-15
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
| **Pose-Splatter** (M5) | **13.78** 🔴 | 0.846 🔴 | — |

**View count monotonic**: 10.47 → 15.95 → 18.56 → 20.66 → 22.16 → **23.84** (n=1800 each). 1v degenerate (IoU=0.028, model predicts 전체 FG).

**FL 6v > PS by +10.06 dB** 🔴 (아래 C5). E2E bottleneck = MVDiff (**86% of quality gap**).

> ### 🔴 C5 — PS 비교 인용 동결 (260723 `/fact --int`)
>
> **PS 관련 수치(13.78 / 0.846 / +10.06 dB) 및 방법론 서술을 논문·대외 자료에 인용 금지.** 2건의 독립 결함:
>
> **(a) 방법론 오기재** — PS를 "per-scene optimization"으로 기술해 왔으나 실제는 **per-frame optimization을 제거한 방법**.
>
> **1차 근거 (원 논문 초록 직접 확인, 260723)**
> `arXiv:2505.18342` — *Pose Splatter: A 3D Gaussian Splatting Model for Quantifying Animal Pose and Appearance*, **Goffinet, Min, Tomasi, Carlson**
> - 초록: 기존 기법의 한계로 *"expensive **per-frame optimization**"* 을 지목하고, 본 방법은 *"without prior knowledge of animal geometry, **per-frame optimization**, or manual annotations"* 로 달성한다고 명시
> - *"eliminates annotation and **per-frame optimization** bottlenecks"*
> - 구성: **shape carving + 3D Gaussian splatting** + rotation-invariant visual embedding / 데이터셋: mice, rats, zebra finches
>
> ⚠️ **기존 근거 논증 철회**: 초판은 "PS가 Train/Val/Test split을 가지므로 per-scene이 아니다"를 근거로 삼았으나 **이 추론은 무효**.
> per-scene 방법도 holdout split을 갖는다 — 본 repo `evaluation_protocol_v1.md §1.3` 이 3DGS(Kerbl 2023)를 "50-300 cameras, every 8th image holdout"으로 기록. **결론은 유지되나 근거는 원 논문으로 교체됨.**
>
> 미확인 사항: "End-to-end 3D UNet" 서술과 "~30ms/frame", 게재 venue(NeurIPS 2025)는 **초록에서 확인 불가** — 본문 확인 필요.
> → 올바른 대비축 = **pretrained-generalizable(GS-LRM) vs dataset-trained(PS)**, 양쪽 다 feed-forward.
> → **논문 Contribution 4** ("fair comparison framework: feed-forward vs per-scene") 프레이밍 **재작성 필요**.
> ⚠️ 표기 주의: 본 문서 §3의 `C1~C4` 는 **caveat 번호**이고, 위 "Contribution 4"는 `PAPER_DRAFT` 의 **기여 번호** — 별개 네임스페이스.
>
> **(b) 재현 불가** — 13.78 산출 체크포인트가 **소실**.
> - 13.78 = v11.0(**260223**), PS `m5_baseline_gs` 6-view 재학습본 기준
> - **260619 pose-splatter accidental deletion** → `experiments/`(체크포인트) · `output/`(렌더) 손실 (`baselines/pose_splatter/HANDOFF_260619.md`)
> - 잔존 아티팩트는 **구본뿐**: `paper_standard_evaluation.json` = **24.68**(full-image) · `posesplatter_fair.json` = **16.80**(fair v1, "different camera"로 폐기됨)
> - **어느 로컬 아티팩트에도 13.78 없음** (전수 grep 0건)
> - 복구 핸드오프는 **260619 작성 후 미실행**(git untracked, 34일 경과). 게다가 Step 3-B는 `m5_4view`/`m5_5view`만 재생성 → **13.78을 낸 6-view baseline은 재현 대상에 없음**
>
> **Threat model**: NeurIPS D&B Track은 재현성이 심사 핵심. 헤드라인 우위 주장(+10.06 dB)의 근거 데이터가 부재하고, 동시에 비교 대상의 방법론 분류가 틀림 → 리뷰어가 둘 중 하나만 짚어도 baseline 비교 전체가 무효화.
>
> **해소 조건 (전부 충족 필요)**: ① PS `m5_baseline_gs` 6-view 재학습 → 13.78 재현 또는 신규 값 확정 ② 방법론 재기술 ③ 동일 하드웨어 속도 실측(현 "~minutes/frame" 주장도 미검증). 상세 원장 = Obsidian `30_Projects/FaceLift/_Agent/260723_FaceLift_curation_fact_log.md`

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

### C5. Pose-Splatter 비교 전면 인용 동결 🔴
PS 방법론 오분류 + `13.78/0.846` 재현 불가. **전문 = §2.1 C5 블록**, 재현성 검사 = §5.1, 복구계획 = `outputs/reports/260724_ps_baseline_STATUS.md`.

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
| **D10** | 🔴 **pending** | PS 비교 전반 | **방법론 오기재** — PS를 per-scene으로 분류. ✅ 문서 정정 완료(guides 본문 포함 전수, `test_sph_harm_compat.py` 검증) / ❌ **논문 Contribution 4 프레이밍 재작성 미완** (§3 caveat C4와 별개 네임스페이스) |
| **D11** | 🔴 **폐기 확정 → 신규 산출 필요** | `13.78` / `0.846` | **재현 불가**(체크포인트·전처리 데이터·전처리 코드 전부 소실, §5.1). 단 **신규 baseline 산출은 가능** — PS repo 실행 가능 상태 회복(260723) + `preprocess_generic.py --camera_params` 로 FL 카메라 직접 주입 가능. 상세 = `outputs/reports/260724_ps_baseline_STATUS.md` |
| **D12** | 🔴 **pending** | `fl_vs_ps_comparison` §2.5 H_Split | PS train-independence 전제 붕괴 → **실험 설계 무효**, 재설계 필요 |
| **D13** | ✅ fixed 260723 | Obsidian vault | INDEX·Implementation_Notes E2E `9.04/0.577`→`8.20/0.521` (D3-D9가 서버만 고치고 vault 누락했던 건) |
| **D14** | 🔴 **pending** | FL-PS gap 파생값 | **3종 공존** — `+7.13`(3건, v10.0) / `+9.62`(7건, 중간본) / `+10.06`(27건, v11.0). 정본 미확정. D11 해소 후 일괄 재산출 필요 |
| **D15** | ✅ fixed 260724 | 파이프라인 조사 | `fix_m5_camera_params.py` = 소실된 convert 의 카메라부 완전 대체 확인. 재산출 계획 = `outputs/reports/260724_ps_baseline_STATUS.md`. 잔여 격차 = 이미지→zarr |

**Cross-file consistency**: 23.84 / 20.66 / 8.20 / 0.954 / 0.956 = 5/5 PASS (grep verified).
🔴 **13.78 은 consistency 대상에서 제외** — 값이 일관되게 인용되는 것과 값이 **재현 가능한 것**은 별개. D11 해소 전까지 인용 동결 (§2.1 C5).

### 5.1 D11 재현성 검사 결과 (260723, gpu03 실측)

`13.78` 재현 가능성을 5단계로 추적한 결과 **구조적 재현 불가** 확정.

| 체인 단계 | 상태 | 근거 |
|---|:-:|---|
| Raw 데이터 | ✅ 생존 | `~/data/raw/markerless_mouse_1_nerf` |
| FaceLift M5 원본 | ✅ 생존 | `~/data/preprocessed/FaceLift_mouse/M5` (3,600 frames) |
| **전처리 코드** `convert_m5_for_ps.py` | 🔴 **소실** | 작업트리 부재 + **git에 커밋된 적 없음** (`git log -- "*convert_m5_for_ps*"` 0건) |
| **전처리 산출물** `m5_for_ps_fj1` | 🔴 **소실** | `~/data/preprocessed/markerless_mouse_1_nerf/` **빈 디렉토리**. 구 경로 `~/dev/project_splatter` 도 부재 |
| 체크포인트 `output/m5_baseline_gs/20260220_143834` | 🔴 소실 | 260619 accidental deletion |
| Config `m5_baseline_gs.json` | ✅ 생존 | `holdout_views:[5]`, split 0.8/0.1/0.1, 50 epochs, "FaceLift camera space" |

**핵심**: 체크포인트만 없으면 재학습으로 복구 가능했음. 그러나 **전처리 코드가 버전관리되지 않아** 동일 데이터셋을 만들 수 없음.
잔존 `fix_m5_camera_params.py` / `recompute_m5_centers.py` / `fix_m5_all.py` 는 **이미 변환된 데이터를 사후 패치**하는 스크립트 — 원 변환을 대체 못 함. 게다가 그 존재 자체가 원 변환이 버그(raw fx=1632 저장)였음을 시사하므로, 재작성 시 **패치 순서까지 복원**해야 동일 결과가 나옴.

**보존 기록에도 13.78 없음**: `benchmark_results.json`, `docs/practical/EXPERIMENT_RESULTS.md`, `docs/reports/*` 전수 확인 — `5cam_baseline_gs`=15.86, `5cam_baseline_gs_hires`=13.84, `facelift_compare_5cam`=24.68. **13.78은 0건.**
`posesplatter_fair.json` 전 slice 확인 — overall 16.80 / holdout 16.53 / per-view 15.65~17.55. **13.78·0.846 없음.**

> **결론**: `13.78 / 0.846` 은 재현도 출처 확인도 불가. **폐기하고 신규 baseline을 산출**하는 것이 유일한 정상 경로.
> 신규 산출 시 필요: ① M5→PS 변환기 **재작성**(FaceLift `M5` 기준, 카메라 fx=549/cx=cy=256 공간) ② `m5_baseline_gs.json`로 50 epochs 학습 ③ fair 프로토콜 평가.
> ⚠️ 재작성된 변환기는 원본과 다를 수밖에 없으므로 **13.78 재현이 아니라 신규 baseline**임을 명시할 것.

> **260723 방법론 교훈**: D3-D9의 "6/6 PASS, drift 없음" 판정은 **cross-file grep 일치만** 검증했음. 같은 틀린 값이 모든 파일에 일관되게 퍼져 있으면 이 검사는 통과함. **일관성 검사 ≠ 정확성 검사.** 이후 SSOT 검증은 (a) 파일 간 일치 (b) **원천 아티팩트 존재·재현** 2축으로 수행할 것.

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
