# FaceLift Mouse Project

GS-LRM 기반 Multi-view Mouse 3D Reconstruction

---

## 1. Research Goal

**목표**: Template-free, monocular input → 3D reconstruction 모델을 **small non-rigid moving object**로 확장

- **첫 대상**: 6개 카메라 뷰 생쥐 데이터
- **후속 계획**: Pose-Splatter와 비교, behavior analysis downstream task용 feature로 활용

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
├── RESEARCH_HYPOTHESES.md  ← 가설 MoC
├── hypotheses/        # 연구 가설 문서 (H1~H8, HP)
├── experiments/       # 실험 운영 가이드, 명령어
├── datasets/          # 데이터셋 명세
├── guides/            # How-to 가이드
├── theory/            # 이론 문서
└── _archive/          # 통합/폐기된 문서
```

**문서 생성/수정 시 자동 업데이트:**

| 트리거 | 업데이트 대상 | 내용 |
|--------|-------------|------|
| 새 문서 생성 | `INDEX.md` | Backlink 추가 |
| 전처리 변경 | `PREPROCESSING_REGISTRY.md` | 버전 이력 |
| 실험 완료 | `experiments/EXPERIMENT_REGISTRY.md` | 결과 반영 |
| 이론/스펙 추가 | `theory/*.md` | 해당 문서 업데이트 |

**통합 규칙:**
- 동일 주제 reports 3개 이상 → `theory/`로 통합
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

**상세**: [[docs/_archive/MOUSE_QUICK_REFERENCE.md]]

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
| [[_archive/MOUSE_QUICK_REFERENCE]] | 명령어 참조 |
| _(삭제됨)_ | Train/Val 성능 Gap 분석 |

---

## 7. Conventions

> **상세**: [[_archive/MOUSE_QUICK_REFERENCE.md#Conventions]]

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

---

## 9. Known Issues

| 이슈 | 상태 | 참조 |
|------|------|------|
| D4 PP Bug (cx=cy=256 강제) | ✅ 해결됨 | _(삭제됨)_ |
| bf16 NaN (opacity_reg) | ✅ 해결됨 | `.float().clamp(1e-4)` |
| normalize_after_zoom PP | ✅ 해결됨 | [[PREPROCESSING_REGISTRY]] |
| Per-sample zoom_after_transform | ✅ 해결됨 | [[PREPROCESSING_REGISTRY]] |

---

*Last Updated: 2026-02-09 | 상세 문서는 [[INDEX]] 참조*
