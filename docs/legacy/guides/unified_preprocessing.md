# Unified Preprocessing Guide

통합 전처리 모듈을 사용하여 markerless mouse 데이터를 GS-LRM 형식으로 변환합니다.

## Quick Start

```bash
cd /home/joon/dev/FaceLift

# D3 preset (권장) - Triangulation center + Correct PP
python -m mouse_extensions.preprocessing.unified_preprocessor \
    --preset D3 \
    --input_dir /home/joon/data/markerless_mouse_1_nerf \
    --output_dir /home/joon/data/preprocessed/FaceLift_mouse/D3

# Custom frame range
python -m mouse_extensions.preprocessing.unified_preprocessor \
    --preset D3 --frame_step 5 --end_frame 18000 \
    --input_dir /path/to/data --output_dir /path/to/output
```

## Presets

| Preset | Center Method | PP Method | 용도 |
|--------|---------------|-----------|------|
| **D3** | triangulation | correct | ✅ **권장** - 최고 정확도 |
| D2 | bbox | correct | 실험용 (bbox 중심) |
| D1 | pp_centered | force_256 | Legacy 호환 |
| v13 | per_view_2d | force_256 | ⛔ Legacy (PP 버그) |

## 왜 D3를 권장하는가?

### 1. Triangulation 기반 3D 중심
- 2D centroid는 뷰마다 다름 → cross-view inconsistency
- DLT triangulation으로 3D 중심 → 모든 뷰에 일관된 2D back-projection
- Reprojection error: 14.4px → 0px

### 2. Correct PP
- 이미지 shift 없이 실제 PP 값 기록
- 정보 손실 없음

## Python API

```python
from mouse_extensions.preprocessing import (
    UnifiedPreprocessor, PreprocessConfig
)

config = PreprocessConfig.from_preset("D3", input_dir, output_dir)
preprocessor = UnifiedPreprocessor(config)
preprocessor.run()
```

## Legacy 스크립트

`preprocessing/_archive/`로 이동됨:
- convert_unified.py, create_d1_pp_centered.py, create_d2_from_raw.py

---
*Last updated: 2026-01-17*
