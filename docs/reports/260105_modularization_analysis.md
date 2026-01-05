# 260105: 코드 모듈화 분석 및 제안

**날짜:** 2026-01-05
**주제:** gslrm 코드 구조 분석 및 모듈화 개선 제안
**상태:** 분석 완료

---

## 1. 현재 코드 구조

### 파일별 라인 수
| 파일 | 라인 수 | 역할 |
|------|---------|------|
| `gslrm/model/gslrm.py` | 2,035 | 메인 모델, Loss, 시각화 |
| `gslrm/model/gaussians_renderer.py` | 1,036 | Gaussian 렌더링 |
| `gslrm/data/mouse_dataset.py` | 771 | Mouse 데이터셋 |
| `gslrm/model/transform_data.py` | 418 | 데이터 변환 |
| `gslrm/model/utils_losses.py` | 379 | Loss 유틸리티 |
| `gslrm/data/dataset.py` | 307 | Random 데이터셋 |
| `gslrm/model/utils_transformer.py` | 302 | Transformer 유틸리티 |

### 클래스 구조 (gslrm.py)
```python
class Renderer(nn.Module)           # 64-148: 렌더링
class GaussiansUpsampler(nn.Module) # 150-235: 업샘플링
class LossComputer(nn.Module)       # 237-828: ⚠️ Loss + 시각화 혼합
class GSLRM(nn.Module)              # 830-2035: 메인 모델
```

---

## 2. 현재 모듈화 장점

### ✅ 잘 분리된 부분

1. **LossComputer 클래스**
   - Loss 계산 함수들이 `_compute_*_loss()` 패턴으로 일관됨
   - Mask 관련 로직 한 곳에 집중

2. **Dataset 분리**
   - `RandomViewDataset` (원본 FaceLift)
   - `MouseViewDataset` (Mouse 전용)
   - 설정으로 선택: `use_mouse_dataset: true/false`

3. **유틸리티 분리**
   - `utils_losses.py`: SSIM, Perceptual Loss
   - `utils_transformer.py`: Attention 관련
   - `utils_train.py`: 학습 유틸리티

---

## 3. 개선 제안

### 🔴 P1: LossComputer 분리 (권장)

**현재 문제**: LossComputer가 Loss 계산 + 시각화를 모두 담당 (592줄)

**제안**: 시각화 로직 분리
```python
# 새 파일: gslrm/model/utils_visualization.py
class VisualizationHelper:
    def create_error_heatmap(self, rendering, target, mask):
        """Blue→Red error heatmap 생성"""
        
    def create_comparison_grid(self, gt, rendered, mask, error_map):
        """5행 비교 그리드 생성"""
        
    def add_error_scale_annotation(self, visual, error_stats):
        """Error 범위 텍스트 추가"""
```

**장점**:
- LossComputer가 순수 loss 계산에 집중
- 시각화 코드 재사용 가능 (inference, debug 등)
- 테스트 용이

### 🟡 P2: Mask 유틸리티 분리

**현재**: mask 관련 함수가 여러 곳에 분산

**제안**:
```python
# 새 파일: gslrm/model/utils_mask.py
def compute_mask_iou(pred_mask, gt_mask):
    """Mask IoU 계산"""
    
def threshold_to_mask(image, threshold=0.1, bg_color=[1,1,1]):
    """이미지에서 mask 추출"""
    
def apply_mask_to_loss(loss_map, mask, reduction='mean'):
    """Mask 적용 loss 계산"""
```

### 🟢 P3: Config 검증 모듈

**현재**: config 검증이 산발적

**제안**:
```python
# gslrm/config/validator.py
class ConfigValidator:
    def validate_mouse_config(self, config):
        """Mouse 설정 필수 항목 확인"""
        required = ['use_mouse_dataset', 'masked_l2_loss']
        warnings = []
        if config.mouse.use_mouse_dataset and not config.training.losses.masked_l2_loss:
            warnings.append(Mouse dataset without masked_l2_loss may underperform)
        return warnings
```

---

## 4. 즉시 적용 가능한 개선

### 4.1 상수 분리
```python
# gslrm/model/constants.py
HEATMAP_COLORS = {
    'low': [0, 0, 1],    # Blue
    'mid': [0, 1, 0],    # Green  
    'high': [1, 0, 0],   # Red
}
DEFAULT_BG_COLOR = [1.0, 1.0, 1.0]  # White
MASK_THRESHOLD = 0.5
ERROR_NORMALIZATION = 0.3
```

### 4.2 타입 힌트 추가
```python
from typing import Dict, Tuple, Optional
import torch
from torch import Tensor

def _compute_l2_loss(
    self, 
    rendering: Tensor,      # [B*V, 3, H, W]
    target: Tensor,         # [B*V, 3, H, W]
    mask: Optional[Tensor]  # [B*V, 1, H, W] or None
) -> Tensor:
```

### 4.3 Docstring 표준화
```python
def _compute_mask_iou(self, rendering, target, gt_mask):
    """
    Compute IoU between GT mask and predicted mask.
    
    The predicted mask is computed by thresholding distance from background.
    
    Args:
        rendering: Rendered images [B*V, 3, H, W] in [0, 1]
        target: Target images [B*V, 3, H, W] in [0, 1]
        gt_mask: Ground truth mask [B*V, 1, H, W] or None
        
    Returns:
        IoU score as scalar tensor
        
    Example:
        >>> iou = self._compute_mask_iou(rendered, target, mask)
        >>> print(fMask IoU: {iou:.4f})
    """
```

---

## 5. 장기 로드맵

| 단계 | 작업 | 예상 효과 |
|------|------|----------|
| 1 | 시각화 모듈 분리 | 재사용성 ↑ |
| 2 | Mask 유틸리티 분리 | 테스트 용이 |
| 3 | Config 검증기 추가 | 설정 오류 감소 |
| 4 | 타입 힌트 + docstring | 유지보수성 ↑ |

---

## 6. 현재 상태 평가

### 점수 (10점 만점)
| 항목 | 점수 | 설명 |
|------|------|------|
| 클래스 분리 | 7/10 | 기본 분리 양호, 시각화 혼합 |
| 함수 명명 | 8/10 | `_compute_*_loss` 일관성 좋음 |
| 설정 분리 | 9/10 | YAML config 잘 활용 |
| 문서화 | 5/10 | Docstring 부족 |
| 타입 힌트 | 4/10 | 대부분 미적용 |

**총평**: 기능적으로 잘 작동하며, 향후 유지보수를 위해 시각화 분리와 문서화 개선 권장

---

*🤖 Generated with Claude Code*
