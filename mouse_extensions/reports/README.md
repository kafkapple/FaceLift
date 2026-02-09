# FaceLift Preprocessing Report System

> **Version**: 1.0.0
> **Updated**: 2026-01-27

## Overview

모듈화된 전처리 레포트 생성 시스템입니다.

## Quick Start

```bash
# 통합 레포트 생성
cd /home/joon/dev/FaceLift
python -m mouse_extensions.reports.unified_report --output-dir ./reports/unified

# 특정 데이터셋만 분석
python -m mouse_extensions.reports.unified_report \
    --datasets D7.1,D8,M3_1,M3_2,M3_2b,M3_3,M4 \
    --output-dir ./reports/unified
```

## Module Structure

```
mouse_extensions/reports/
├── __init__.py
├── unified_report.py      # 메인 CLI
├── modules/
│   ├── __init__.py
│   ├── theory.py          # MVG 이론 + MathJax 수식
│   ├── camera_viz.py      # 카메라 3D/2D 시각화
│   ├── pp_analysis.py     # PP 분포 + Ray Error 분석
│   └── dataset_comparison.py  # M-Series 비교 + 가설
├── templates/
│   └── (HTML templates)
└── README.md
```

## Modules

### 1. TheoryModule
MVG (Multi-View Geometry) 이론 섹션 생성:
- 좌표계 변환 수식
- Ray direction 계산
- PP 에러 영향 분석
- zoom_center_mode 설명
- Affine vs Homography 비교

### 2. CameraVisualizationModule
카메라 배치 시각화:
- 3D view
- Top view (X-Z plane)
- Side view (Azimuth vs Elevation)
- 파라미터 테이블

### 3. PPAnalysisModule
Principal Point 분석:
- 데이터셋별 PP 분포
- Ray Error 계산
- 시각화 그래프

### 4. DatasetComparisonModule
데이터셋 비교:
- M1/M2/M3 시리즈 비교 테이블
- 가설 검증 섹션 (H1, H2)
- 권장 설정
- 전처리 명령어

## Output

생성되는 파일:
- `unified_report.html` - 통합 HTML 레포트
- `camera_top_view.png` - 카메라 Top View
- `camera_side_view.png` - 카메라 Side View
- `camera_3d_view.png` - 카메라 3D View
- `pp_distribution.png` - PP 분포 그래프
- `ray_error_comparison.png` - Ray Error 비교

## Programmatic Usage

```python
from mouse_extensions.reports import UnifiedReportGenerator

generator = UnifiedReportGenerator(
    output_dir='./my_reports',
    datasets=['M3_1', 'M3_2', 'M4']
)
report_path = generator.generate_report()
print(f"Report: {report_path}")
```

## Related Documents

- `docs/datasets/PREPROCESSING_REGISTRY.md` - 데이터셋/전처리 SSOT
- `docs/datasets/M5_SERIES_SPEC.md` - M5 시리즈 명세
- `docs/theory/METRICS_PROTOCOL.md` - 평가 메트릭 프로토콜

---

*FaceLift Mouse Extension | Report System v1.0*
