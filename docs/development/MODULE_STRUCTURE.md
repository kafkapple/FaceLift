# FaceLift Module Structure

> mouse_extensions 모듈 구조 및 역할 정의
> Updated: 2026-01-27

---

## 디렉토리 구조

```
mouse_extensions/
├── model/                      # 모델 확장
│   ├── loss_extensions.py      # 확장 loss 함수 (perceptual, alpha 등)
│   ├── mask_losses.py          # 마스크 관련 loss
│   ├── gslrm_patches.py        # GSLRM 모델 패치
│   ├── gaussian_pruning.py     # Gaussian pruning 유틸
│   ├── alpha_renderer.py       # Alpha mask 렌더러
│   ├── visualization.py        # 내부 시각화 (IoU, 마스크)
│   └── visualization_extensions.py  # 학습 시각화 (에러 히트맵)
│
├── preprocessing/              # 전처리 모듈
│   ├── presets.py              # 프리셋 정의 (M1, M2, M3_*)
│   ├── preprocess.py           # 메인 전처리기
│   ├── camera_normalizer.py    # 카메라 정규화
│   ├── center_estimation.py    # 3D 중심 추정
│   ├── data_loader.py          # 데이터 로더
│   ├── split_manager.py        # Split 전략 관리 (CLI)
│   ├── split_verifier.py       # Split 검증 (CLI)
│   ├── create_temporal_split.py # Temporal split 생성 (CLI)
│   └── format_validator.py     # 포맷 검증 (CLI)
│
├── visualization/              # 외부 시각화 모듈
│   ├── turntable_config.py     # Turntable 설정 (카메라 순서)
│   ├── alpha_visualization.py  # Alpha 시각화
│   └── error_annotation.py     # 에러 어노테이션
│
├── validation/                 # 학습/추론 결과 검증
│   └── validator.py            # 결과 검증기
│
├── scripts/                    # 유틸리티 스크립트
│   ├── diagnostics/            # 진단 도구
│   └── report_generator/       # 리포트 생성
│
└── reports/                    # 리포트 템플릿
```

---

## 모듈별 역할

### model/ - 모델 확장

| 파일 | 역할 | 사용처 |
|------|------|--------|
| `loss_extensions.py` | Perceptual, SSIM, LPIPS, alpha loss | gslrm_patches.py |
| `mask_losses.py` | Masked L2, normalized masked loss | gslrm.py |
| `gslrm_patches.py` | GSLRM forward 패치 | train_gslrm.py |
| `gaussian_pruning.py` | Opacity 기반 pruning | (optional) |
| `alpha_renderer.py` | Alpha mask 렌더링 | alpha loss 계산 |
| `visualization.py` | IoU 계산, 마스크 비교 | 내부 유틸 |
| `visualization_extensions.py` | 에러 히트맵, 학습 시각화 | gslrm_patches.py |

### preprocessing/ - 전처리

| 파일 | 역할 | 실행 방식 |
|------|------|----------|
| `presets.py` | 프리셋 정의 (VERSION_HIERARCHY) | import |
| `preprocess.py` | 메인 전처리 실행기 | CLI |
| `camera_normalizer.py` | fx=549, PP=256 정규화 | import |
| `center_estimation.py` | 3D triangulation | import |
| `split_*.py` | Train/val split 관리 | CLI |
| `format_validator.py` | 데이터 포맷 검증 | CLI |

### visualization/ vs model/visualization

| 위치 | 용도 | 예시 |
|------|------|------|
| `visualization/` | **외부용** (validation, reporting) | turntable, alpha viz |
| `model/visualization*.py` | **내부용** (학습 루프) | 에러 히트맵, IoU |

### preprocessing/ vs validation/

| 위치 | 검증 대상 |
|------|-----------|
| `preprocessing/` | 전처리 데이터 품질 (포맷, split) |
| `validation/` | 학습/추론 결과 (PSNR, IoU) |

---

## 삭제된 폴더 (2026-01-27)

| 폴더 | 이유 |
|------|------|
| `patches/` | One-time 패치 스크립트 (더 이상 필요 없음) |
| `preprocessing/_archive/` | Legacy 전처리기 (D1, D2, D6) |
| `experiments/validation/_archive/` | 구 실험 결과 (2.6GB) |

---

## Import 관계

```
train_gslrm.py
    └── mouse_extensions.model.gslrm_patches
            ├── loss_extensions
            ├── mask_losses
            └── visualization_extensions

gslrm/model/gslrm.py
    └── mouse_extensions.model.mask_losses

mouse_extensions.model.__init__
    ├── gslrm_patches
    ├── gaussian_pruning
    └── alpha_renderer
```

---

*FaceLift Development Docs | 2026-01-27*
