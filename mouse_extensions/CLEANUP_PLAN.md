# mouse_extensions 정리 완료

> **목표**: 중복 제거, 모듈 통합, 미사용 스크립트 아카이브
> **완료일**: 2026-01-25

---

## 정리 결과

### Before → After

| 항목 | Before | After |
|------|--------|-------|
| preprocessing/ 활성 파일 | 13개 | **9개** |
| scripts/ 루트 파일 | 21개 | **2개** |
| 아카이브된 preprocessing | 12개 | **16개** |
| 아카이브된 scripts | 42개 | **55개** |

---

## 최종 구조

```
mouse_extensions/
├── data/                    # 데이터 로더 (1 파일)
├── model/                   # 모델 확장 (7 파일)
├── preprocessing/           # 전처리 (9 파일)
│   ├── preprocess.py        # 메인 진입점
│   ├── presets.py           # 프리셋 정의
│   ├── camera_normalizer.py # 카메라 정규화
│   ├── center_estimation.py # 중심점 추정
│   ├── data_loader.py       # 데이터 로딩
│   ├── split_manager.py     # split 관리
│   ├── split_verifier.py    # split 검증
│   └── format_validator.py  # 형식 검증
├── scripts/
│   ├── analysis/            # 분석 (4 파일)
│   ├── diagnostics/         # 진단 (4 파일)
│   ├── experiments/         # 실험 실행 (1 파일)
│   ├── inference/           # 추론 (8 파일)
│   ├── setup/               # 환경 설정 (4 파일)
│   ├── solutions/           # 솔루션 (3 파일)
│   ├── validation/          # 검증 (1 파일)
│   └── _archive/            # 아카이브 (55 파일)
├── utils/                   # 유틸리티 (4 파일)
├── reports/                 # 리포트 (2 파일)
└── tests/                   # 테스트
```

---

## 이동된 루트 스크립트

| 파일 | 새 위치 |
|------|---------|
| `run_experiments.sh` | `scripts/experiments/` |
| `setup_env.sh` | `scripts/setup/` |
| `setup_gpu03.sh` | `scripts/setup/` |

---

## 아카이브된 주요 파일

### preprocessing/_archive/
- `preprocess_fix.py` - preprocess.py에 통합됨
- `generate_split.py` - split_manager.py로 통합
- `create_split.py` - split_manager.py로 통합
- `split_dataset.py` - split_manager.py로 통합

### scripts/_archive/
- `dataset_comparison_v5.py` - 분석 완료
- `validate_center_research_v4.py` - 연구 완료
- `comprehensive_report_v7.py` - 버전 관리
- 기타 일회성 분석/디버깅 스크립트

---

## 핵심 모듈 (코어 코드에서 사용)

| 모듈 | 사용처 |
|------|--------|
| `data/` | `gslrm/data/mouse_dataset.py` |
| `model/` | `gslrm/model/gslrm.py` |
| `model/mask_losses.py` | `gslrm/model/gslrm.py` |
| `scripts/solutions/` | `mouse_dataset.py` (조건부) |
| `utils/` | `train_gslrm.py` |
| `scripts/test_evaluation_extension.py` | `train_gslrm.py` |

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-01-25 | 정리 완료: 아카이브 + 디렉토리 재구성 |

