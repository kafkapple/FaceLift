# Rollback Documentation - 2026-01-21

## 개요

D7_1_E2_2_gt_mask 실험 재현 실패로 인한 코드 롤백 기록.

## 문제 상황

### 증상
- D7_1_E2_2_gt_mask 실험이 이전 성공 결과를 재현하지 못함
- WandB 기록상 2026-01-21 00:00:09에 성공적으로 실행된 실험과 현재 결과가 다름
- 시각화 결과에서 데이터가 다르게 보임

### 성공 당시 WandB Config (2026-01-21 00:00:09)
```yaml
mask_source: "gt"
use_masked_loss: true
total_views: 5
num_input_views: 4
dataset_path: /home/joon/data/preprocessed/FaceLift_mouse/D7_1/data_mouse_val.txt
```

### 현재 Config (문제 발생)
```yaml
mask_mode: gt  # 설정 키 이름 변경됨
masked_l2_loss: true
num_views: 6  # 5에서 6으로 변경됨
num_input_views: 4
```

## 발견된 차이점

### 1. 마스크 설정 체계 변경
| 이전 | 현재 |
|------|------|
| `mask_source: "gt"` | `mask_mode: "gt"` |
| `use_masked_loss: true` | `masked_l2_loss: true` |

### 2. 뷰 수 변경
- 성공 당시: `total_views: 5`
- 현재: `num_views: 6`

### 3. 코드 버그 (수정됨)
`gslrm.py`에서 `rendered_alpha`가 `_compute_all_losses()`에 전달되지 않는 버그 발견 및 수정.

```python
# 수정 전
losses = self._compute_all_losses(
    rendering_flat, target_flat, img_aligned_xyz, input, mask, b, v, h, w
)

# 수정 후
rendered_alpha_flat = None
if rendered_alpha is not None:
    rendered_alpha_flat = rendered_alpha.reshape(b * v, 1, h, w)

losses = self._compute_all_losses(
    rendering_flat, target_flat, img_aligned_xyz, input, mask, b, v, h, w,
    rendered_alpha=rendered_alpha_flat
)
```

## 롤백 정보

### 백업 브랜치
- **브랜치명**: `backup/wip-260121-before-rollback`
- **포함 내용**: 롤백 전 모든 uncommitted 변경사항 (281개 파일)
- **주요 변경**: gslrm.py 버그 수정, loss_extensions.py 디버그 로깅, 다수 config 파일

### 롤백 대상 커밋
- **커밋**: `84d2f4f` - "fix: use rendered_alpha as fallback mask for visualization"
- **이유**: 이 커밋 시점에서 성공적인 실험이 실행됨

## 복구 방법

백업된 변경사항을 다시 적용하려면:
```bash
cd /home/joon/dev/FaceLift
git checkout backup/wip-260121-before-rollback
# 또는 특정 파일만 복구
git checkout backup/wip-260121-before-rollback -- <file_path>
```

## 다음 단계

1. 84d2f4f 커밋으로 롤백
2. 성공 당시 config와 동일한 설정으로 실험 재시도
3. 결과 비교 후 문제 원인 확정

---

*Generated: 2026-01-21*
