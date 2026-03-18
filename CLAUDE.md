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

**카메라/좌표계 작업 시 필수 검토:**

| 항목 | 체크 | 상세 |
|------|------|------|
| **Intrinsics** | fx, fy, cx, cy | 단위, 스케일 확인. _(삭제됨)_ |
| **PP (Principal Point)** | 256 vs 가변 | Center-aligned 권장 |
| **좌표계** | OpenGL vs OpenCV | Y-up vs Z-up 구분 |
| **정규화** | fx=549, trans=2.7 | Pretrained 호환 필수 |

**⚠️ 카메라 파라미터 변경 시**: 반드시 Ray Error 계산 및 검증

```python
# Ray Error 계산
θ_error = arctan(sqrt((Δcx/fx)² + (Δcy/fy)²))
```

### 2.4 Rotation Direction Convention

**Orbit vs Camera 좌표계 차이 (2026-02-09 수정):**

| 함수 | 좌표계 | CCW 의미 |
|------|--------|----------|
| get_turntable_cameras | cos->x, sin->y (from +X) | Standard math CCW |
| compute_camera_order | atan2(x,y) (from +Y) | 90 deg rotated |

**규칙**: rotation_direction=ccw 일 때:
- Camera order는 물리적 CCW (위에서 반시계)
- Orbit은 clockwise=True 전달 (math CW = physical CCW)

**Smooth Trajectory**: smooth_trajectory: true (base config) -> CubicSpline + RotationSpline, smoothstep easing. Linear SLERP fallback.

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

---

## 4. Project Structure

```
FaceLift/
├── mouse_extensions/       # ★ 모든 확장 구현은 여기에
│   ├── preprocessing/      # 전처리 (preset 기반 통합)
│   ├── model/             # loss, visualization
│   └── scripts/           # 유틸리티, 진단
├── configs/mouse/          # 실험 설정 (base + dataset + experiment)
├── gslrm/                  # GS-LRM 코어 (최소 수정)
└── docs/                   # 기술 문서
```

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
| [[datasets/M5_SERIES_SPEC]] | M3 시리즈 상세 명세 |
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
| **데이터셋 명명** | M-Series: M1, M2, M3_1, M3_2 |
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

## 11. Key Results Summary (260220)

### View Ablation (Fair Eval, test set, 360f × 5 views)

| Views | PSNR_gt | IoU | PSNR_int |
|:-----:|:-------:|:---:|:--------:|
| 1 | 10.47 | 0.028 | 10.47 |
| 2 | 15.95 | 0.858 | 17.91 |
| 3 | 18.56 | 0.899 | 19.54 |
| 4 | 20.66 | 0.926 | 21.29 |
| 5 | 22.16 | 0.942 | 22.56 |
| 6 | **23.84** | **0.954** | **24.02** |

### Best Checkpoints (M5t2)

| Component | Checkpoint | Key Metric |
|-----------|-----------|:----------:|
| GS-LRM 6v | `6view_v2/best_psnr.pt` | **PSNR=23.84** (test, fair eval) |
| GS-LRM 4v | `M5t2_E0_1_facelift/best_psnr.pt` | PSNR=20.66 (test, fair eval) |
| MVDiff | `mouse_M5t2/checkpoint-5000` | E2E PSNR_wh=21.29 |

### Phase 3 Conclusion

- All E2E strategies converge: PSNR_gt 7.90-8.44, IoU 0.47-0.53
- MVDiff = sole bottleneck (86% of quality loss, -15.64 dB from 6v GT)
- Training strategy optimization is saturated → architecture change needed
- GS-LRM 6v > PS by +7.13 dB (Tier A fair eval)

### PS M5 Retraining (in progress)

- Coordinate system fixed (auto_orient space)
- Training loss: 1.20 → 0.58 (decreasing)
- Expected: fair Tier B comparison after completion

---

*Last Updated: 2026-02-20 | 상세 문서는 [[INDEX]] 참조*
