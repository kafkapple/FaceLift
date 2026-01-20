# Code Location Registry

> FaceLift Mouse 프로젝트의 코드 위치 관리 문서
> `mouse_extensions/` 패키지 외부에 있는 스크립트들의 명시적 관리

## 코드 구조 개요

```
FaceLift/
├── mouse_extensions/          # ✅ 메인 패키지 (정리됨)
│   ├── preprocessing/         # 전처리 파이프라인
│   ├── model/                 # 모델 확장 (alpha_loss 등)
│   ├── scripts/               # 패키지 내부 스크립트
│   └── utils/                 # 유틸리티
│
├── scripts/                   # ⚠️ 외부 스크립트 (이 문서에서 관리)
│   ├── visualization/         # 시각화 도구
│   ├── *.py                   # 패치, 유틸리티
│   └── _archive/              # 아카이브
│
├── configs/mouse/             # 실험 설정 파일
├── gslrm/                     # 원본 FaceLift 코드 (수정됨)
└── docs/                      # 문서
```

---

## 외부 스크립트 레지스트리

### 1. 시각화 (`scripts/visualization/`)

| 파일 | 용도 | 생성일 | 의존성 |
|------|------|--------|--------|
| `visualize_gaussian_rerun.py` | Rerun.io 기반 3DGS 시각화 | 2026-01-19 | `rerun-sdk`, `plyfile` |
| `batch_visualize.py` | 배치 시각화 자동화 | 2026-01-19 | `visualize_gaussian_rerun.py` |

**사용 예시:**
```bash
# 단일 파일 시각화
python scripts/visualization/visualize_gaussian_rerun.py gaussians.ply --save

# 배치 시각화
python scripts/visualization/batch_visualize.py --exp-pattern "D7_t_*" --output-dir viz_output
```

### 2. 패치 스크립트 (`scripts/`)

| 파일 | 용도 | 대상 파일 | 생성일 |
|------|------|----------|--------|
| `apply_vis_mask_patch.py` | 시각화 마스크 일관성 패치 | `gslrm/model/gslrm.py` | 2026-01-19 |
| `add_mask_label_patch.py` | 시각화에 마스크 타입 라벨 추가 | `gslrm/model/gslrm.py` | 2026-01-19 |
| `patch_visualization_mask.py` | 위 두 패치 통합 | `gslrm/model/gslrm.py` | 2026-01-19 |

**패치 적용 상태:** ✅ gpu03 서버에 적용 완료 (2026-01-19)

### 3. 전처리/분할 (`scripts/`)

| 파일 | 용도 | 생성일 |
|------|------|--------|
| `generate_temporal_split.py` | Temporal train/val/test 분할 | - |
| `generate_mouse_prompt_embeds.py` | 프롬프트 임베딩 생성 | - |
| `visualize_alpha_masks.py` | Alpha 마스크 시각화 | - |

### 4. 아카이브 (`scripts/_archive/`)

사용되지 않는 레거시 스크립트들. 참조용으로만 보관.

---

## gslrm/ 수정 사항

### `gslrm/model/gslrm.py`

| 수정 내용 | 라인 | 설명 |
|----------|------|------|
| Alpha Loss 지원 | ~L1200 | `compute_alpha_loss()` 추가 |
| 시각화 마스크 일관성 | ~L2022 | mask_mode 기반 시각화 |
| 마스크 타입 라벨 | ~L2050 | PIL ImageDraw 라벨 추가 |

### `gslrm/data/mouse_dataset.py`

Mouse 데이터셋 로더. 원본 `dataset.py`와 분리됨.

---

## 설정 파일 (`configs/mouse/`)

### D7_t 실험 시리즈 (8개)

| 파일 | 실험 ID | 핵심 설정 |
|------|---------|----------|
| `D7_t_E1_1_paper_random.yaml` | E1.1 | 4v, random=true, no mask |
| `D7_t_E1_2_paper_fixed.yaml` | E1.2 | 4v, random=false, no mask |
| `D7_t_E2_1_rgb_mask.yaml` | E2.1 | 4v, rgb_pred mask |
| `D7_t_E2_2_gt_mask.yaml` | E2.2 | 4v, gt mask |
| `D7_t_E2_3_alpha_mask.yaml` | E2.3 | 4v, alpha mask |
| `D7_t_E3_2_5v_alpha.yaml` | E3.2 | 5v, alpha mask |
| `D7_t_E4_2_5v_alpha_loss.yaml` | E4.2 | 5v, alpha + alpha_loss |
| `D7_t_E5_1_5v_alpha_random.yaml` | E5.1 | 5v, alpha + random |

---

## 문서 (`docs/`)

| 파일 | 내용 |
|------|------|
| `00_Index.md` | 프로젝트 가이드 인덱스 |
| `Project_Structure.md` | 프로젝트 구조 설명 |
| `D7_t_experiment_matrix.md` | 실험 가설 및 비교군 |
| `Code_Location_Registry.md` | 이 문서 |
| `PREPROCESSING_REGISTRY.md` | 전처리 데이터셋 레지스트리 |

---

## 향후 정리 계획

### 단기
- [ ] `scripts/` 패치 스크립트들 → `mouse_extensions/patches/`로 이동 고려

### 중기
- [ ] 시각화 스크립트 패키지화 (`mouse_extensions/visualization/`)

---

*Created: 2026-01-19*
*Last Updated: 2026-01-19*
