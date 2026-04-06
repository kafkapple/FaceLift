# FaceLift Mouse Project

GS-LRM 기반 Multi-view Mouse 3D Reconstruction

---

## 0. Phase Structure (2026-03-12~)

| Phase | Focus | Status | SSOT |
|:-----:|-------|:------:|------|
| **Phase 1** | MVDiff bottleneck (Sil loss, DA, Spatial Token) | 🔧 유지보수 | `docs/experiments/hypothesis_roadmap.md` |
| **Phase 2** ⭐ | Novel View + Multi-Species + NeurIPS Dataset Track | **🔬 ACTIVE** | `docs/experiments/PHASE2_NOVEL_VIEW_ROADMAP.md` |

> **기본 포커스는 Phase 2.** Phase 1 작업은 명시적 요청 시에만.
> 핵심 인프라(GS-LRM, eval, turntable, coordinate transforms)는 공통 활용.

### Documentation Split Principle (문서 배치 원칙)

**서버 docs/ = "이걸 어떻게 실행하나?"** — 구현/실행에 직접 필요한 문서만.
**Obsidian = "이게 왜 이런 설계인가?"** — 이론/분석/전략/논문/연구 노트.

| 서버 (gpu03) | 역할 | Obsidian | 역할 |
|-------------|------|----------|------|
| `experiments/` | 실험 설정, 결과, 비교 | `neurips/` | 논문 초안, Gap 분석, 전략 |
| `specs/` | 기술 사양 (I/O, 프로토콜) | `theory/` | 수학, 이론, 원리, 모델 계보 |
| `guides/` | 실행 가이드, 코드 워크스루 | `research/` | 분석, 비교, 대안 기술 |
| `datasets/` | 전처리 레지스트리, 데이터 명세 | `_Notes/` | 날짜 기반 연구 일지 |
| `hypotheses/` | 가설 검증 결과 (수치 테이블) | `Presentation/` | 발표 슬라이드 |

**판단 기준**: 문서 작성 시 "이 문서 없이 실험을 실행할 수 있는가?"
- **Yes** → Obsidian (이론/전략)
- **No** → 서버 docs (실행 필수)

**중복 금지**: 동일 내용을 양쪽에 두지 않음. 서버에서 Obsidian 참조 시 포인터만 남김.

### Obsidian 경로

| 경로 | 내용 |
|------|------|
| `~/Documents/Obsidian/30_Projects/_CODES/FaceLift/docs/INDEX.md` | **Obsidian MoC** (v7.0) |
| `docs/neurips/` | 논문 초안, Gap 분석, Executive Summary, 프로젝트 종합 |
| `docs/theory/` | 좌표계, Stage1/2 이론, MVDiff 이론, Loss 수식 |
| `docs/research/` | E2E 분석, PS 비교, MV-Adapter 기술 분석 |
| `_Notes/` | 24개 연구 노트 (260114~260317) |

---

## 1. Research Goal

**목표**: Template-free, monocular input → 3D reconstruction 모델을 **small non-rigid moving object**로 확장

- **첫 대상**: 6개 카메라 뷰 생쥐 데이터
- **Phase 2 확장**: Multi-species (rat, marmoset), novel view rendering, body-part camera follow
- **Target**: NeurIPS 2026 Evaluations & Datasets Track

### Success Criteria

| 유형 | 지표 |
|------|------|
| **정량** | PSNR, loss, IoU |
| **정성** | 3D Gaussian 재구성 품질, Turntable MP4, mask GT/pred 시각화, 전처리 검증 보고서 |

### Related Papers

- [FaceLift](https://arxiv.org/abs/2412.17812) - 원본 논문
- [GS-LRM](https://arxiv.org/abs/2404.19702) - 기반 아키텍처
- [3D Gaussian Splatting](https://arxiv.org/abs/2307.01097) - 렌더링 기법

**Upstream**: https://github.com/weijielyu/FaceLift

---

## 2. Core Principles

### 2.1 Documentation Rule (중요\!)

**모듈 작업 완료 시 필수 수행:**

1. **연구 노트 작성**: `outputs/reports/YYMMDD_*.md`
2. **Git 커밋**: Conventional commits format
3. **교육용 문서 작성**:
   - 초보자도 코드 흐름 이해 가능하도록 상세 설명
   - 왜(Why) → 어떻게(How) → 무엇(What) 구조
   - 디버깅 및 수정 방법 포함

```
📝 문서 작성 체크리스트:
□ 작업 목적/동기 설명
□ 코드 변경 사항 상세 기술
□ 핵심 로직 흐름도/다이어그램
□ 발생 가능한 에러 및 해결법
□ Git commit message와 연계
```

### 2.2 문서 체계화 규칙 (Auto-Update)

**⭐ 최우선 참조: [[INDEX.md]]** - Map of Content (모든 문서의 네비게이션 허브)

**문서 구조:**
```
docs/
├── INDEX.md           ← ★ 문서 허브 (항상 최신 유지)
├── hypotheses/        # 실험 가설 검증 결과 (H4~H7)
├── experiments/       # 실험 운영 가이드, 명령어
├── datasets/          # 데이터셋 명세
├── guides/            # How-to 가이드
├── specs/             # 기술 사양 (구 theory/ — 구현 밀착 문서만)
└── _archive/          # 통합/폐기된 문서
```

**문서 생성/수정 시 자동 업데이트:**

| 트리거 | 업데이트 대상 | 내용 |
|--------|-------------|------|
| 새 문서 생성 | `INDEX.md` | Backlink 추가 |
| 전처리 변경 | `PREPROCESSING_REGISTRY.md` | 버전 이력 |
| 실험 완료 | `experiments/EXPERIMENT_REGISTRY.md` | 결과 반영 |
| 기술 사양 추가 | `specs/*.md` | 해당 문서 업데이트 |
| 이론/연구 추가 | Obsidian `docs/theory/` 또는 `docs/research/` | Obsidian INDEX 업데이트 |

**통합 규칙:**
- 동일 주제 reports 3개 이상 → Obsidian `theory/`로 통합 (이론) 또는 서버 `specs/`로 통합 (구현)
- 통합된 원본 → `_archive/reports/`로 이동
- **MoC.md에 통합 문서 링크 추가** (필수)

### 2.3 Multi-View Geometry Rigor (중요\!)

> **Canonical Reference**: Obsidian `docs/theory/COORDINATE_SYSTEMS.md` → "Camera Conventions" 섹션
> **Code SSOT**: `mouse_extensions/behavior/camera_system.py`, `mouse_extensions/visualization/camera_utils.py`

**카메라/좌표계 작업 시 필수 검토:** OpenCV convention (X-right, Y-down, Z-forward), World Z-up, fx=549, trans=2.7 (pretrained 호환).

**⚠️ 카메라 파라미터 변경 시**: Ray Error 검증 필수 — `θ = arctan(√((Δcx/fx)² + (Δcy/fy)²))`

### 2.4 Storage Tier Rule (⚠️ cgroup v2 safety)

> **상세**: `docs/specs/RAT_PREPROCESSING_STRATEGY.md` §Q3, `docs/datasets/PREPROCESSING_REGISTRY.md` §10

**학습 데이터는 반드시 `/node_data/` (local NVMe)에 저장. NFS (`/home/joon/dev/`) 금지.**
- `/home/joon/data → /node_data/joon/data` (symlink) ✅
- `train_gslrm.py`에 pre-flight 경고 내장 (`_check_data_storage_tier`)
- NFS page cache = cgroup 과금 → oomd kill 위험

| 용도 | 위치 | Storage |
|------|------|:-------:|
| 학습/전처리 데이터 | `/node_data/joon/data/preprocessed/` | Local NVMe ✅ |
| 체크포인트 | `/node_data/joon/checkpoints/` | Local NVMe ✅ |
| 코드/config | `/home/joon/dev/FaceLift/` | NFS (OK, 소량) |
| outputs/ (viz, reports) | `/home/joon/dev/FaceLift/outputs/` | NFS (OK, 비학습) |

### 2.5 DataLoader Resource Management (공용 서버 필수)

> **교훈 (260331)**: OMP_NUM_THREADS로 해결 시도 → 실패. 프로세스 수와 스레드 수는 별개 레이어.

**3-Layer 구분** (혼동 주의):

| 설정 | 제어 대상 | 레이어 | 기본값 |
|------|----------|--------|:------:|
| `OMP_NUM_THREADS` | OpenMP 스레드 (프로세스 내부) | Thread | 1 |
| `num_workers` | DataLoader worker 프로세스 수 | Process | base=8 |
| `nproc_per_node` | GPU당 학습 rank 수 | GPU Rank | 1 |

**프로세스 곱셈 효과**: `persistent_workers=True` + 복수 DataLoader(train/val) → workers가 **합산** 상주
- `num_workers=4` → 1 main + 4 train + 4 val = **9 procs** (× ~640MB RSS each)
- `prefetch_factor=8` → worker당 8 batch 미리 로드 → **page cache 폭증** → cgroup v2 과금 → oomd kill

**실험 config 필수 override**: `num_workers ≤ 4, prefetch_factor ≤ 2`

### 2.6 Rotation Direction Convention

> **상세**: Obsidian `COORDINATE_SYSTEMS.md` → "Turntable vs Camera Order" 참조

**핵심 규칙**: `rotation_direction=ccw` → physical CCW (위에서 반시계) = math CW → `clockwise=True` 전달.
**Smooth Trajectory**: `smooth_trajectory: true` → CubicSpline + RotationSpline, smoothstep easing.

---

## 3. Environment

```bash
ssh gpu03
cd /home/joon/dev/FaceLift
conda activate facelift
```

### GPU Usage

| 규칙 | 상세 |
|------|------|
| **허용 GPU** | `CUDA_VISIBLE_DEVICES=4,5,6,7` (4~7번만 사용) |
| **동시 실행** | VRAM 허용 시 같은 GPU에서 2-3개 작업 가능 |
| **DataLoader** | 실험 config에서 `num_workers≤4, prefetch_factor≤2` 권장 (cgroup v2 page cache 방지) |

### Data Directory Convention

**GS-LRM 이미지 데이터** (기존 — 변경 없음):
```
/home/joon/data/
├── raw/              # 원본 (영상, 이미지)
├── preprocessed/     # 파이프라인 전처리 출력 (M3_2, M5t2 등)
└── processed/        # 후처리
```

**s-DANNCE 행동 데이터** (별도 구조):
```
/home/joon/data/sdannce/
├── mouse/
│   ├── dataverse/    # Harvard Dataverse .mat 다운로드 원본 (keypoints + labels)
│   └── features/     # 우리 추출 .npz (covariance, S1, S3 등)
├── rat/
│   ├── dataverse/    # Harvard Dataverse .mat 다운로드 원본
│   └── features/     # 우리 추출 .npz
└── metadata/         # cohort 메타데이터, HLAC 매핑 등
```

> ⚠️ s-DANNCE는 `preprocessed/`가 아닌 `sdannce/` 독립 경로 사용.
> 이유: 행동 .mat ≠ 이미지 전처리. `dataverse/` = 출처 명시, `features/` = 우리 산출물.

---

## 4. Project Structure

```
FaceLift/
├── mouse_extensions/       # ★ 모든 확장 구현은 여기에
│   ├── preprocessing/      # 전처리 (preset 기반 통합)
│   ├── model/             # loss, visualization
│   ├── visualization/     # 렌더링/시각화 모듈
│   ├── behavior/          # 행동 분석 (HLAC, clustering, BAMS)
│   └── scripts/           # 유틸리티, 진단
├── configs/mouse/          # 실험 설정 (base + dataset + experiment)
├── gslrm/                  # GS-LRM 코어 (최소 수정)
└── docs/                   # 기술 문서
```

### Output Directory Structure (v2 — 2026-03-23)

**SSOT**: `mouse_extensions/paths.py` — 모든 경로는 이 모듈에서 관리.

```
outputs/
├── experiments/{species}/{experiment_id}/   # Training runs (checkpoints, logs, config)
├── eval/{species}/{experiment_id}/          # Quantitative evaluation (metrics JSON)
├── viz/                                     # Visualization media
│   ├── turntable/{species}/{experiment_id}/ # A-type: 360° orbit MP4
│   ├── cinematic/{species}/{experiment_id}/ # B-type: 논문용 데모 MP4
│   ├── bodypart/{species}/{experiment_id}/  # Body-part isolation
│   └── comparison/{species}/                # Alpha, temporal grids
├── features/{species}/                      # Extracted features (large NPZ)
│   ├── gaussian/                            # Raw Gaussian properties
│   ├── covariance/                          # Covariance features
│   ├── temporal/                            # Temporal features
│   └── hlac/                                # HLAC behavior features (통합)
├── analysis/{species}/                      # Exploratory analysis
│   ├── behavior_clustering/                 # Behavior clustering + reports
│   ├── bams/                                # BAMS experiments
│   ├── neural_texture/                      # Neural texture experiments
│   └── filtering/                           # View filtering strategies
├── datasets/                                # Generated/derived datasets
│   ├── novel_view/                          # Novel view renders
│   ├── fine_tune/{species}/                 # Fine-tuning data (e.g., rat gslrm_format)
│   └── generated/                           # Other derived data
├── reports/{report_slug}/                   # Publication reports (HTML, figures)
└── _archive/                                # Deprecated experiments (read-only)
```

**Factory functions** (in `mouse_extensions/paths.py`):
```python
get_experiment_dir(species, experiment_id)  # → experiments/{species}/{id}/
get_eval_dir(species, experiment_id)        # → eval/{species}/{id}/
get_viz_dir(species, experiment_id, type)   # → viz/{type}/{species}/{id}/
get_feature_dir(species, feature_type)      # → features/{species}/{type}/
get_analysis_dir(species, analysis_type)    # → analysis/{species}/{type}/
```

> **Backward compat**: 이전 경로 (`outputs/sdannce_poc/`, `outputs/report/`, `outputs/hlac_*` 등)는
> symlink로 새 위치를 가리킴. 새 코드는 반드시 factory 함수 사용.

### Extension Guidelines

> **원칙**: 모든 추가 구현은 `mouse_extensions/` 하위에 배치
> - 단일 기능 원칙, 간결한 모듈
> - 원본 코드(`gslrm/`) 수정 최소화

---

## 5. Quick Commands

### Training

3가지 config mode 지원 (상호 배타, **상세**: `configs/README.md`):

```bash
# Mode 2: Modular (권장) — base + dataset + experiment merge
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e E0_1_facelift

# Mode 1: Legacy — 단일 standalone config
train_gslrm.py --config configs/mouse/D7_1_E2_gt_alpha.yaml

# Mode 3: Flexible — custom base + experiment
train_gslrm.py -b configs/base/gslrm_mouse.yaml -e E0_1_facelift

# CLI override (모든 mode에서 사용 가능)
train_gslrm.py -d M5t2 -e E0_1_facelift --set training.schedule.max_fwdbwd_passes 400
```

### Preprocessing
```bash
# Preset 기반 통합 전처리
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_2 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_2
```

**상세**: [[docs/experiments/COMMANDS]]

---

## 6. Key References (Backlinks)

### 데이터셋 & 전처리

| 문서 | 내용 |
|------|------|
| [[INDEX]] | 문서 허브 (★ 최우선) |
| [[PREPROCESSING_REGISTRY]] | 버전 이력, 프리셋 정의 |
| [[datasets/M5_SERIES_SPEC]] | M5 시리즈 상세 명세 |
| _(삭제됨)_ | PP/fx 이론 + 버그 분석 |

### 실험 & 결과

| 문서 | 내용 |
|------|------|
| [[experiments/EXPERIMENT_REGISTRY]] | 실험 설정, 우선순위 |
| [[experiments/comprehensive_analysis_report]] | Tier A/B/C 종합 분석 + View Ablation |
| [[experiments/COMMANDS]] | 명령어 SSOT |
| _(삭제됨)_ | Train/Val 성능 Gap 분석 |

---

## 7. Conventions

> **상세**: `configs/README.md` 참조

| 항목 | 규칙 |
|------|------|
| **데이터셋 명명** | M-Series: M1~M5. **M5t2** = 표준 split (80:10:10 temporal) |
| **실험 명명** | `E{Cat}_{Num}_{keywords}` |
| **Config** | 3 modes: Modular(`-d`+`-e`), Legacy(`--config`), Flexible(`-b`+`-e`) → `configs/README.md` |
| **문서 날짜** | 파일명 `YYMMDD`, 본문 `YYYY-MM-DD` |
| **Git** | Conventional commits, Co-Authored-By 포함 |

---

## 8. Anti-Patterns (절대 금지)

- ⛔ `killall python` 사용 금지
- ⛔ 실험 자동 실행 금지 (명령어만 제공)
- ⛔ GPU 0~3번 사용 금지
- ⛔ PP=256 강제 + Object-centered crop (MVG 부정합)
- ⛔ **E2E 추론 결과 검증 없이 보고 금지** (2026-03-03 교훈)
  - 필수: `grep 'Batch Path' LOG` → "Path 2b: MVDiffusion" 확인
  - 필수: `run_config.json`에서 `input_view_idx ≠ null` 확인
  - 필수: PSNR sanity check (E2E=7~9 dB, GS-LRM only=20~24 dB)
  - `--data_dir` 사용 시 `--input_view_idx 0` 반드시 지정해야 E2E

---

## 9. Known Issues

| 이슈 | 상태 | 참조 |
|------|------|------|
| D4 PP Bug (cx=cy=256 강제) | ✅ 해결됨 | _(삭제됨)_ |
| bf16 NaN (opacity_reg) | ✅ 해결됨 | `.float().clamp(1e-4)` |
| normalize_after_zoom PP | ✅ 해결됨 | [[PREPROCESSING_REGISTRY]] |
| Per-sample zoom_after_transform | ✅ 해결됨 | [[PREPROCESSING_REGISTRY]] |
| Val metrics WandB 미전송 (16GB+ GPU) | ✅ 해결됨 | `_offloaded_optim` 블록 외부로 이동 (260323) |
| summary.csv empty (path depth bug) | ✅ 해결됨 | nested dir 탐색 지원 (260323) |
| Train/Val 로그 구분 불가 | ✅ 해결됨 | `[Train]`/`[Val]` prefix 통일 (260323) |
| metrics.txt 1-UID only | ✅ 해결됨 | `_save_visualizations` 내부 → run() 루프로 분리 (260323) |
| outputs/ 구조 비일관 | ✅ 해결됨 | v2 구조 마이그레이션 완료 — `paths.py` SSOT + symlink backward compat (260323) |
| Deform V2 (param MSE) | ✅ **중단 (260331)** | param MSE ≠ rendering quality (3/3 audit). 237G cache 삭제. → V3 전환 |
| Deform V2 BG 낭비 | ✅ **V3에서 해결** | 97.5% BG Gaussians 학습 → FG-only cache (3MB/frame vs 85MB) |
| OMP_NUM_THREADS 미설정 | ✅ 해결됨 | dl_base.sh → .bashrc interactive guard 위 이동. OMP=1 (260331) |
| 5view ablation 수렴 | ✅ **중단 (260331)** | plateau 23.0-23.1 dB (step 12200). 결과 기록 완료 |
| DataLoader 과다 프로세스 | ✅ 완화됨 (rat_ft_v2) | base num_workers=8 → rat_ft_v2에서 4로 override, prefetch_factor 8→2. 다른 실험 config도 동일 적용 권장. cgroup v2 page cache 방지 (260331) |
| RAT2 v2 3D geometry 손상 | ✅ **v3에서 수정 (260402)** | `clip_xyz: true` + 비정규화 카메라 → Gaussian 위치 절삭. `mouse_dataset.py`에 `recenter_cameras` flag 추가. v2 체크포인트 삭제 (28.2GB). |
| videoio 극저 bitrate | ⚠️ 부분 해결 | `video_io.py` preset medium→slow 변경. videoio API에 crf 미지원. 근본 해결은 ffmpeg subprocess 대체 필요. |
| Novel view 512px 데이터셋 | 🔴 미완성 | 디렉토리 구조만 존재, 0 frames 생성됨. `outputs/datasets/novel_view/mouse_m5t2/` |
| Val PSNR clip 누락 | ✅ 해결됨 | `evaluation/metrics.py`에서 render [0,1] clip 누락 → PSNR 극저 보고 (21→1.2dB). `compute_per_view_metrics`에서 clip 추가 (260406) |

### Deformation V3 (진행 중, 260331~)

| 항목 | 상세 |
|------|------|
| **전략** | Rendering loss PRIMARY + FG-only + ARAP + L_zero (4× MoA Audit) |
| **FG Cache** | `cache_fg_gaussians.py` → FG-only, ~3MB/frame, float16 (V2: 85MB) |
| **학습** | `train_deform.py` (unified, config-driven) + `deform_v3.yaml` |
| **다음** | FG cache 완료 → smoke test → rendering loss 검증 |
| **이론** | Obsidian `theory/DEFORMATION_STRATEGY.md`, `BOTTOM_VIEW_ENHANCEMENT_STRATEGY.md` |
| **계획** | `docs/experiments/DEFORMATION_ROADMAP.md` §2 |

---



## 10. Baseline Comparison Protocol (FaceLift vs Pose-Splatter)

### Quick Reference

| Item | Detail |
|------|--------|
| **Comparison script** | `mouse_extensions/scripts/eval/compare_with_baseline.py` |
| **Config** | `mouse_extensions/scripts/eval/unified_eval_config.yaml` |
| **Design doc** | `docs/experiments/FL_vs_PS_comparison.md` |
| **PS metrics (joon)** | `output/facelift_compare_5cam/latest/paper_standard_evaluation.json` |
| **FL metrics (gpu03)** | `experiments/comparison/tier/*_fair.json` (fair eval) |

### Metric Protocol Warning

- **FaceLift**: White-BG composite + full-image (PSNR/SSIM inflated by background)
- **Pose-Splatter**: Masked foreground-only (lower absolute numbers)
- **Comparable**: L1 (masked), IoU (same formula)
- **Non-comparable without re-eval**: PSNR, SSIM

### Usage

```bash
# Load pre-computed metrics
python -m mouse_extensions.scripts.eval.compare_with_baseline \
    --facelift_metrics outputs/h5_e2e/cfgr_ckpt10000/metrics_v2.json \
    --baseline_metrics baselines/pose_splatter/paper_standard_evaluation.json \
    --output_dir experiments/comparison/FL_vs_PS/
```

---

## 11. Key Results Summary (260324 updated)

### View Ablation (Fair Eval, M5t2 test set 3240-3599, 360f × 5 views)

| Views | PSNR_gt | IoU | PSNR_int | 학습 상태 |
|:-----:|:-------:|:---:|:--------:|:---------:|
| 1 | 10.47 | 0.028 | 10.47 | ✅ |
| 2 | 15.95 | 0.858 | 17.91 | 🔄 P1 resume 중단 |
| 3 | 18.56 | 0.899 | 19.54 | ✅ |
| 4 | 20.66 | 0.926 | 21.29 | ✅ |
| 5 | — | — | — | ✅ **중단 (260331, plateau 23.0-23.1, step 12200)** |
| 6 | **23.84** | **0.954** | **24.02** | ✅ |

> 5view: val PSNR 23.0-23.1에서 plateau (step 11800-12200). Fair eval 미실행. 수렴 판정으로 중단.

### Best Checkpoints — 3-Best Rule (260326 Comprehensive Eval)

> **SSOT**: `docs/experiments/MASTER_RESULTS_TABLE.md` + `docs/experiments/ALPHA_LOSS_NOVEL_VIEW_ANALYSIS.md` §Comprehensive
> **모두 M5t2 split (80:10:10)으로 학습**. α=0.0 이름에 M5t2 미포함이나 동일 split 사용.

| 기준 | α | Checkpoint | PSNR_gt | IoU | 용도 |
|------|:-:|-----------|:-------:|:---:|------|
| **FG Quality** | 0.0 | `base_uniform_v2_6view_v2` | **20.12** | 0.886 | PSNR_gt 정량 비교 |
| **Trade-off (Default)** ⭐ | 0.3 | `M5t2_6view_alpha03_v3` | 20.01 | **0.913** | **공유/실용 default** |
| **Artifact** | 1.0 | `M5t2_6view_alpha10_v3` | 19.55 | **0.925** | 데모/시각화 |
| GS-LRM 4v | 0.0 | `base_uniform_v2_4view_v2` | — | — | View ablation |
| MVDiff | — | `mouse_M5t2/checkpoint-5000` | — | — | E2E |

> ⚠️ PSNR_gt (masked FG) ≠ 이전 "23.84" (다른 eval 프로토콜). 상세: `ALPHA_LOSS_NOVEL_VIEW_ANALYSIS.md` §Comprehensive Evaluation

### Phase 3 Conclusion

- All E2E strategies converge: PSNR_gt 7.90-8.44, IoU 0.47-0.53
- MVDiff = sole bottleneck (86% of quality loss, -15.64 dB from 6v GT)
- Training strategy optimization is saturated → architecture change needed
- GS-LRM 6v > PS by +7.13 dB (Tier A fair eval)

### Rat FT (RAT1→RAT2 v1→v2→v3)

| Version | Val PSNR | Data | Status | Notes |
|---------|:--------:|:----:|:------:|-------|
| RAT1 v1 | 17.49 | 1001 frames | ✅ Baseline | overfitting 17.5dB gap |
| RAT2 v2 (despilled) | 18.67 | 2967 frames | ❌ **INVALID** | 3D geometry 손상 (clip_xyz + 비정규화 카메라). 체크포인트 삭제됨 (260402) |
| RAT2 v3 (recentered) | — | 2967 frames | ❌ **FAILED** | centroid recenter → dist collapse (2.6→0.4). 체크포인트 삭제됨 |
| RAT2 v4b (convergence) | 0.90 | 2967 frames | ❌ **FAILED** | convergence-point recenter + clip_xyz=false. PSNR 0.9 → 학습 실패. 체크포인트 삭제됨 (260406) |
| **RAT2 v5 (fx-norm)** | TBD | 2968 frames | 🔄 **학습 중** | `--normalize_fx` 전처리 (fx=549). GPU5, step 0→15000 (260406~). Val PSNR clip 버그 수정 후 재시작 |

> **v3→v4b→v5 진화**: v3(centroid recenter→거리 붕괴), v4(clip_xyz), v4b(convergence point→PSNR 0.9), v5(근본원인=fx 미정규화. 전처리 단계에서 fx=549로 통일).
> **Checkpoint**: `RAT2_despilled_rat_ft_v5/` | Config: `rat_ft_v5.yaml` | Dataset: `RAT2_despilled_fxnorm.yaml`

### PS M5 Retraining (in progress)

- Coordinate system fixed (auto_orient space)
- Training loss: 1.20 → 0.58 (decreasing)
- Expected: fair Tier B comparison after completion

---

*Last Updated: 2026-03-23 | 상세 문서는 [[INDEX]] 참조*
