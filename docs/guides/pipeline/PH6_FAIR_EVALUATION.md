# Phase 6: Fair Evaluation

> **Navigation**: [← Hub](../PIPELINE_DEEP_DIVE.md) | [Prev: PH5](PH5_E2E_INFERENCE.md) | [Next: PH7 →](PH7_3D_KEYPOINT.md)
>
> **핵심 파일**: `mouse_extensions/scripts/eval/fair_comparison.py`

---

> **Why**: FaceLift (feed-forward) vs PoseSplatter (per-scene optimization)은
> 근본적으로 다른 모델 유형. 공정한 비교를 위해 5가지 공정성 보장이 필요.

### 6.1 Fairness Guarantees

```
1. Test-only:    학습 데이터 제외 (M5t2 test: frame 3240-3599)
2. GT alpha mask: 양측 모두 GT RGBA alpha로 FG/BG 분리
3. Unified metric: 동일 함수로 PSNR/SSIM/L1/IoU 계산
4. FG-only:      배경 제외, 전경 픽셀만 평가
5. Coverage-aware: pred_fg ∩ gt_fg 영역과 gt_fg 전체 영역 별도 추적
```

### 6.2 Metric Computation

**File**: `mouse_extensions/scripts/eval/fair_comparison.py`

#### compute_all_metrics (L156-225) — 핵심 함수

```python
# Input: pred [H,W,3], gt [H,W,3], gt_mask [H,W], pred_mask [H,W] (opt.)
#
# Output metrics dict:
#   psnr_gt_masked     — GT FG 영역의 PSNR (GT mask 기준)
#   psnr_intersection  — pred_fg ∩ gt_fg 영역의 PSNR (순수 색상 정확도)
#   ssim_gt_masked     — GT mask bbox crop 기반 SSIM
#   l1_gt_masked       — FG 픽셀 L1: sum|diff| / (3 * sum(mask))
#   iou                — 실루엣 binary IoU (pred_fg vs gt_fg)
#   coverage           — pred가 gt_fg를 얼마나 커버하는지 (recall)
#   pred_precision     — pred_fg 중 gt_fg와 겹치는 비율 (precision)
#   color_bias_r/g/b   — intersection에서 pred-gt 평균 색 편차
```

#### evaluate_facelift (L288-467) — 평가 루프

```
1. Render path 자동 감지 (L331-337):
   E2E: cam_000/render_view_NN.png
   GS-LRM standalone: render_view_NN.png

2. GT 로드 (L360-365): M5 RGBA → alpha > 127 → binary mask

3. 해상도 불일치 처리 (L368-376): 중앙 crop

4. Per-frame metrics 수집 → Overall 통계 (mean, std, median)

5. Output JSON: model, evaluation, config, fairness, overall, per_view
```

### 6.3 Visualization Grid (L232-281)

```
┌──────────┬──────────┬──────────┐
│ GT       │ Render   │ GT Mask  │
│ (white)  │ (white)  │ (gray)   │
├──────────┼──────────┼──────────┤
│ Mask     │ Error    │ Raw      │
│ Compare  │ Map (5×) │ Render   │
│ G=inter  │          │          │
│ R=GT only│          │          │
│ B=pred   │          │          │
└──────────┴──────────┴──────────┘
Filename: {frame}_view{NN}_psnr{X.X}_int{X.X}_iou{X.XX}_cov{XX%}.png
```

### 6.4 Commands

```bash
# FL 평가
python mouse_extensions/scripts/eval/fair_comparison.py evaluate_fl \
    --render_dir outputs/phase3_e2e/<experiment>/samples \
    --gt_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --views 1 2 3 4 5 \
    --output outputs/phase3_e2e/<experiment>/fair_eval.json \
    --save_vis outputs/phase3_e2e/<experiment>/vis --vis_every 10

# FL vs PS 비교 리포트
python mouse_extensions/scripts/eval/fair_comparison.py compare \
    --facelift outputs/.../fair_eval.json \
    --baseline outputs/.../ps_eval.json \
    --output_dir outputs/.../comparison/
```

---

*← [PH5](PH5_E2E_INFERENCE.md) | [Hub](../PIPELINE_DEEP_DIVE.md) | [Next: PH7 →](PH7_3D_KEYPOINT.md)*
