# 시각화 업데이트 구현 완료 보고서

> **작성일**: 2026-01-25
> **상태**: 부분 구현 완료 (추가 통합 필요)

---

## 1. 완료된 작업

### 1.1 문서/설정 업데이트 [완료]

| 파일 | 변경 내용 |
|------|-----------|
| configs/datasets/M3_1.yaml | 신규 생성 |
| configs/datasets/M3_2.yaml | 신규 생성 |
| docs/practical/QUICK_START.md | v7.0 - M3_1, M3_2 권장, M3_norm/persample 폐기 표기 |
| docs/practical/EXPERIMENT_REGISTRY.md | M3_1, M3_2 추가, 기존 M3 폐기 표기 |

### 1.2 Alpha Mask 시각화 모듈 [완료]

**파일**: `mouse_extensions/visualization/alpha_visualization.py`

**기능**:
- `visualize_alpha_comparison()`: GT mask vs rendered alpha 4-row 비교 이미지
  - Row 0: GT Mask (binary)
  - Row 1: Rendered Alpha (continuous)
  - Row 2: Alpha > 0.5 (binary)
  - Row 3: Difference (Green=TP, Red=FP, Blue=FN)
- `compute_alpha_metrics()`: IoU, Precision, Recall, F1 계산
- `should_visualize_alpha()`: config에서 alpha_loss_weight > 0 확인

### 1.3 Turntable 설정 모듈 [완료]

**파일**: `mouse_extensions/visualization/turntable_config.py`

**카메라 순서** (시계방향 물리 배치):
```
MOUSE_CAMERA_ORDER = [0, 4, 2, 1, 3, 5]
```

**기능**:
- `interpolate_camera_extrinsics()`: SLERP 회전 + Linear 이동 보간
- `create_turntable_trajectory()`: 전체 360도 궤적 생성
- `get_grid_row_labels()`: Grid 행 라벨 (Cam 0 -> 4 등)

**기본 설정**:
```python
DEFAULT_TURNTABLE_CONFIG = {
    "camera_order": [0, 4, 2, 1, 3, 5],
    "fps": 15,  # 기존 30의 절반 (2x 느림)
    "interpolation_steps": 6,
    "grid_rows": 6,
    "grid_cols": 6,
    "add_row_labels": True,
    "use_camera_interpolation": True,
}
```

---

## 2. 추가 통합 필요 (TODO)

### 2.1 train_gslrm.py 수정

```python
# 추가 필요한 import
from mouse_extensions.visualization import (
    visualize_alpha_comparison,
    should_visualize_alpha,
    MOUSE_CAMERA_ORDER,
)

# save_visuals_if_needed 또는 _log_visuals_to_wandb에 추가:
if should_visualize_alpha(self.config) and rendered_alpha is not None and mask is not None:
    alpha_vis = visualize_alpha_comparison(mask, rendered_alpha, num_views=v)
    Image.fromarray(alpha_vis).save(os.path.join(vis_dir, "alpha_comparison.jpg"))
    wandb_images["train/alpha_comparison"] = wandb.Image(alpha_vis)
```

### 2.2 gslrm/model/gslrm.py 수정

Turntable 렌더링 부분에서 `MOUSE_CAMERA_ORDER` 사용:

```python
from mouse_extensions.visualization import MOUSE_CAMERA_ORDER, create_turntable_trajectory

# render_turntable 호출 시 camera_order 전달
turntable_cfg["camera_order"] = MOUSE_CAMERA_ORDER
turntable_cfg["fps"] = 15  # 느리게
```

### 2.3 Val turntable 방식 Train에 추가

현재 Val의 기본 turntable.mp4 저장 방식을 Train에도 추가 필요.

---

## 3. 사용 방법

### 3.1 Alpha 시각화 테스트

```python
from mouse_extensions.visualization import visualize_alpha_comparison

# gt_mask: [B*V, 1, H, W], rendered_alpha: [B*V, 1, H, W]
vis_np = visualize_alpha_comparison(gt_mask, rendered_alpha, num_views=6)
Image.fromarray(vis_np).save("alpha_comparison.jpg")
```

### 3.2 Turntable 궤적 생성

```python
from mouse_extensions.visualization import create_turntable_trajectory

# camera_c2ws: [6, 4, 4] - 각 카메라의 c2w 행렬
trajectory = create_turntable_trajectory(camera_c2ws, interpolation_steps=6)
# trajectory: List[np.ndarray] - 총 36개 (6 cameras x 6 steps)
```

---

## 4. 파일 위치

```
mouse_extensions/
+-- visualization/
    +-- __init__.py
    +-- alpha_visualization.py  [NEW]
    +-- turntable_config.py     [NEW]
```

---

## 5. 다음 단계

1. [ ] train_gslrm.py에 alpha 시각화 통합
2. [ ] gslrm.py turntable에 camera order 적용
3. [ ] Val과 동일한 기본 turntable을 Train에 추가
4. [ ] Grid 레이아웃 row 기준으로 변경

---

*Visualization Update Report v1.0 | 2026-01-25*
