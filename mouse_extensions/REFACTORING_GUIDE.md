# Mouse Extensions 리팩토링 가이드

모듈화 리팩토링의 핵심 원칙과 방법을 정리합니다.

## 1. 리팩토링 원칙

### 1.1 단계적 접근
```
인라인 코드 → 독립 모듈 → 하위 모듈 분리 → 위치 이동
```

### 1.2 핵심 원칙
- **Fallback 유지**: 모듈 없어도 기존 코드 동작
- **역호환성**: 기존 import 경로도 동작
- **테스트 우선**: 매 단계마다 import 테스트

---

## 2. 리팩토링 단계별 방법

### Step 1: 인라인 코드를 모듈로 추출

**Before (인라인)**
```python
# gslrm.py
def _compute_all_losses(self, ...):
    use_pred_mask = self.config.training.losses.get("use_predicted_mask", False)
    if use_pred_mask:
        mask = (rendering - 1.0).abs().mean(dim=1) > threshold
    # ... 복잡한 로직
```

**After (모듈 추출)**
```python
# mouse_extensions/loss_extensions.py
def compute_mask_from_config(config, rendering, gt_mask, rendered_alpha):
    """Mask 계산 로직을 독립 함수로 분리"""
    use_pred_mask = config.training.losses.get("use_predicted_mask", False)
    if use_pred_mask:
        mask = (rendering - 1.0).abs().mean(dim=1) > threshold
    return mask, mask_type
```

### Step 2: Fallback 패턴으로 import

**핵심: try-except로 모듈 유무 처리**
```python
# gslrm.py
try:
    from mouse_extensions.model import compute_mask_from_config
    MOUSE_EXTENSIONS_AVAILABLE = True
except ImportError:
    MOUSE_EXTENSIONS_AVAILABLE = False

def _compute_all_losses(self, ...):
    if MOUSE_EXTENSIONS_AVAILABLE:
        mask, _ = compute_mask_from_config(self.config, rendering, ...)
    else:
        # 기존 인라인 로직 (fallback)
        use_pred_mask = self.config.training.losses.get(...)
```

### Step 3: 하위 모듈 분리

**구조 설계 원칙**
```
mouse_extensions/
├── __init__.py      # 통합 진입점 (re-export)
├── data/            # 데이터 관련
├── model/           # 모델 학습 관련
└── utils/           # 유틸리티
```

**__init__.py 패턴**
```python
# mouse_extensions/__init__.py (통합)
from .data import preprocess_cameras      # re-export
from .model import compute_mask_from_config
from .utils import get_experiment_info

# mouse_extensions/data/__init__.py (하위 모듈)
from .preprocessing import (
    normalize_camera_distance,
    preprocess_cameras,
)
```

### Step 4: import 경로 일괄 변경

**sed로 일괄 변경**
```bash
# 경로 변경: gslrm.mouse_extensions → mouse_extensions
sed -i 's|from gslrm.mouse_extensions|from mouse_extensions|g' file.py

# 하위 모듈 경로: mouse_extensions → mouse_extensions.data
sed -i 's|from mouse_extensions.preprocessing|from mouse_extensions.data|g' file.py
```

---

## 3. Import 패턴 비교

### 3.1 직접 import (권장)
```python
from mouse_extensions.data import preprocess_cameras
from mouse_extensions.model import compute_mask_from_config
```
- 장점: 명확한 출처, IDE 자동완성 지원
- 단점: 경로 길어짐

### 3.2 통합 import (역호환)
```python
from mouse_extensions import preprocess_cameras, compute_mask_from_config
```
- 장점: 짧은 경로, 기존 코드 호환
- 단점: 출처 불명확

---

## 4. 테스트 체크리스트

```bash
# 1. 모듈 import 테스트
python -c "from mouse_extensions import __version__; print(__version__)"

# 2. 하위 모듈 테스트
python -c "from mouse_extensions.data import preprocess_cameras"
python -c "from mouse_extensions.model import compute_mask_from_config"
python -c "from mouse_extensions.utils import get_experiment_info"

# 3. 의존 파일 테스트
python -c "from gslrm.data.mouse_dataset import MouseViewDataset"
python -c "from gslrm.model.gslrm import GSLRM"
```

---

## 5. 실수 방지 팁

### 5.1 순환 import 방지
```python
# Bad: 모듈 레벨에서 상호 import
from .module_a import func_a  # module_a가 이 모듈을 import하면 순환

# Good: 함수 내부에서 import
def my_func():
    from .module_a import func_a
```

### 5.2 __all__ 명시
```python
# 명시적 export 목록
__all__ = ["preprocess_cameras", "get_bg_color"]
```

### 5.3 버전 관리
```python
__version__ = "1.2.0"  # 구조 변경시 마이너 버전 업
```

---

## 6. 커밋 메시지 템플릿

```
Refactor: [모듈명] - [변경 내용]

STRUCTURE:
- 변경된 폴더 구조

CHANGES:
- 변경 파일 목록

IMPORT PATTERN:
- 새 import 방식

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

---

## 6. 원본 수정 파일 목록 (2026-01-29 기준)

| 파일 | 수정 정도 | 내용 |
|------|----------|------|
| `gslrm/model/gslrm.py` | Heavy | 6 mouse_extensions imports |
| `gslrm/model/gaussians_renderer.py` | Heavy | diff_gauss + 446줄 trajectory |
| `gslrm/data/mouse_dataset.py` | Shim | redirect only |
| `gslrm/data/dataset.py` | Minor | conditional import |
| `gslrm/model/transform_data.py` | Minor | conditional import |
| `gslrm/model/utils_metrics.py` | Minor | conditional import |
| `gslrm/model/utils_train.py` | Minor | conditional import |
| `train_gslrm.py` | Minor | 3 conditional imports |

### 커스텀 패키지
- `diff_gauss 1.0.10.0` — `diff_gaussian_rasterization` 대체 (alpha mask 지원)

### 향후 리팩토링 TODO
- [ ] `gaussians_renderer.py` trajectory 446줄 → `mouse_extensions/visualization/` 이동
- [ ] `gslrm.py` 직접 import → plugin/registry 패턴 전환
- [ ] `reports/` 내 `generate_comprehensive_d8_report.py` 정리 (D8 전용, 범용화 필요)
