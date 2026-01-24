# M3 통합 설계 근거

> Created: 2026-01-24

## 1. 기존 구현 분석

### 이미 구현된 기능

| 기능 | 파일 | 라인 | M3 활용 |
|------|------|------|---------|
| Up-alignment | preprocess.py | 60-166 | ✅ 재사용 |
| Adaptive zoom (bbox) | preprocess.py | 167-206 | ✅ 확장 |
| DLT Triangulation | center_estimation.py | 전체 | ✅ 재사용 |
| Camera normalization | camera_normalizer.py | 전체 | ✅ 재사용 |
| D10 paradigm | preprocess.py | 878-940 | ✅ 기반으로 확장 |

### 결론: 별도 스크립트(preprocess_m.py) 불필요

기존 `preprocess.py`를 확장하는 것이 효율적:
- 중복 코드 방지
- 기존 테스트/검증 활용
- 일관된 인터페이스 유지

## 2. 통합 설계

### 버전 매핑

| M-series | D-series 확장 | 설명 |
|----------|---------------|------|
| M1 | D7.1 (기존) | 기하학적 기준선 |
| M2 | D8 (기존) | Precision homography |
| M3 | **D10.3** (신규) | Object-centered zoom |

### M3 = D10.3 설계

D10 paradigm 기반으로 확장:

```yaml
# D10 (기존)
paradigm: up_aligned_zoom
adaptive_zoom: false  # 고정 zoom 또는 bbox 기반

# D10.3 (신규 = M3)
paradigm: up_aligned_zoom  # 기존 paradigm 재사용
adaptive_zoom: true
zoom_method: coverage_based  # NEW: coverage 기반
target_fg_coverage: 0.05     # NEW: 5% 목표
min_fg_coverage: 0.03        # NEW: 최소 3%
center_method: triangulation # 기존 center_estimation.py 활용
```

## 3. 필요한 변경사항

### A. presets.py 수정 (+30줄)

```python
"D10.3": {
    "paradigm": "up_aligned_zoom",
    "description": "M3 equivalent - Object-centered zoom for optimal performance",
    
    # Up-alignment (D10 기본)
    "up_alignment": True,
    "up_source": "camera_y_mean",  # fallback (vertical_lines 없을 때)
    
    # Adaptive zoom (확장)
    "adaptive_zoom": True,
    "zoom_method": "coverage_based",  # NEW
    "target_fg_coverage": 0.05,       # NEW
    "zoom_range": [1.0, 2.5],
    
    # Quality constraints (NEW)
    "min_fg_coverage": 0.03,
    "max_center_offset_px": 20,
    
    # Camera normalization (M1/D7.1 기반)
    "target_fx": 549.0,
    "normalize_distance": True,
    "target_distance": 2.7,
}
```

### B. preprocess.py 수정 (+50줄)

```python
# 1. compute_adaptive_zoom() 확장
def compute_adaptive_zoom_coverage(masks, target_coverage=0.05, zoom_range=(1.0, 2.5)):
    """Coverage 기반 adaptive zoom (기존 bbox 방식 대체)"""
    current_coverage = np.mean([m.sum() / m.size for m in masks])
    zoom = np.sqrt(target_coverage / max(current_coverage, 1e-6))
    return np.clip(zoom, *zoom_range)

# 2. coverage 필터 추가
def compute_fg_coverage_after_transform(mask, transform_matrix):
    """변환 후 foreground coverage 계산"""
    transformed = cv2.warpAffine(mask, transform_matrix, ...)
    return transformed.sum() / transformed.size

# 3. run() 메서드에 coverage 체크 추가
if config.min_fg_coverage > 0:
    coverage = compute_fg_coverage_after_transform(...)
    if coverage < config.min_fg_coverage:
        logger.warning(f"Frame {idx}: coverage {coverage:.3f} < {config.min_fg_coverage}")
```

### C. PreprocessConfig 수정 (+10줄)

```python
@dataclass
class PreprocessConfig:
    # 기존 필드...
    
    # M3/D10.3 신규 필드
    zoom_method: str = "bbox"  # "bbox" or "coverage_based"
    target_fg_coverage: float = 0.05
    min_fg_coverage: float = 0.0
    max_center_offset_px: float = float('inf')
```

## 4. 통합 이점

1. **코드 재사용**: DLT triangulation, camera normalization 등 기존 검증된 코드 활용
2. **일관성**: 동일한 CLI 인터페이스 (`--preset D10.3`)
3. **유지보수**: 단일 코드베이스
4. **테스트**: 기존 테스트 프레임워크 활용

## 5. 사용법

```bash
# M3 = D10.3 실행
python -m mouse_extensions.preprocessing.preprocess \
    --preset D10.3 \
    --input-dir /path/to/raw \
    --output-dir /path/to/D10.3

# 또는 alias
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3 \  # D10.3의 alias
    --input-dir /path/to/raw \
    --output-dir /path/to/M3
```

---

*통합 설계 완료: 2026-01-24*
