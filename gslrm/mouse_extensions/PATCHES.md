# Minimal Patches for FaceLift

원본 코드에 적용할 최소한의 수정 사항입니다.

## 필수 설치

```bash
# diff_gauss (alpha 반환 지원 rasterizer)
pip install git+https://github.com/slothfulxtx/diff-gaussian-rasterization.git
```

---

## Patch 1: gaussians_renderer.py (Line 24-27)

### 변경 전
```python
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
```

### 변경 후
```python
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
```

---

## Patch 2: gaussians_renderer.py - render_opencv_cam (Line ~820)

### 변경 전
```python
result = rasterizer(
    means3D=means3D,
    means2D=means2D,
    shs=shs,
    colors_precomp=None,
    opacities=opacity,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=None,
)
if len(result) == 2:
    rendered_image, radii = result
else:
    rendered_image, radii = result[0], result[1]

return {
    "render": rendered_image,
    ...
}
```

### 변경 후
```python
color, depth, norm, alpha, radii, extra = rasterizer(
    means3D=means3D,
    means2D=means2D,
    shs=shs,
    colors_precomp=None,
    opacities=opacity,
    scales=scales,
    rotations=rotations,
    cov3Ds_precomp=None,
    norm3Ds_precomp=None,
    extra_attrs=None,
)

return {
    "render": color,
    "alpha": alpha,
    "depth": depth,
    "norm": norm,
    ...
}
```

---

## Patch 3: gaussians_renderer.py - DeferredGaussianRender.forward (Line ~888)

### 변경 전
```python
renders = []
for i in range(b):
    pc = gaussians_model.set_data(...)
    for j in range(v):
        renders.append(render_opencv_cam(...)[\"render\"])
renders = torch.stack(renders, dim=0)
renders = renders.reshape(b, v, 3, height, width)
renders = renders.requires_grad_()
return renders
```

### 변경 후
```python
renders = []
alphas = []
for i in range(b):
    pc = gaussians_model.set_data(...)
    for j in range(v):
        result = render_opencv_cam(...)
        renders.append(result["render"])
        alphas.append(result["alpha"])
renders = torch.stack(renders, dim=0).reshape(b, v, 3, height, width)
alphas = torch.stack(alphas, dim=0).reshape(b, v, 1, height, width)
renders = renders.requires_grad_()
alphas = alphas.requires_grad_()
return renders, alphas
```

---

## Patch 4: gaussians_renderer.py - DeferredGaussianRender.backward (Line ~913)

### 변경 전
```python
def backward(ctx, grad_output):
    ...
    render.backward(grad_output[i, j])
```

### 변경 후
```python
def backward(ctx, grad_renders, grad_alphas):
    ...
    if grad_renders is not None:
        render.backward(grad_renders[i, j], retain_graph=(grad_alphas is not None))
    if grad_alphas is not None:
        alpha.backward(grad_alphas[i, j])
```

---

## Patch 5: gslrm.py - _compute_all_losses (Line ~348)

### 추가 (기존 코드 상단에)
```python
def _compute_all_losses(self, rendering, target, img_aligned_xyz, input, mask, b, v, h, w, rendered_alpha=None):
    losses = {}

    # NEW: Mask type selection
    use_pred_mask = self.config.training.losses.get("use_predicted_mask", False)
    pred_mask_threshold = self.config.training.losses.get("pred_mask_threshold", 0.1)

    if use_pred_mask:
        bg_color = 1.0
        color_distance = (rendering - bg_color).abs().mean(dim=1, keepdim=True)
        mask = (color_distance > pred_mask_threshold).float()
    elif rendered_alpha is not None:
        use_rendered_alpha = self.config.training.losses.get("use_rendered_alpha_mask", False)
        if use_rendered_alpha:
            alpha_threshold = self.config.training.losses.get("alpha_mask_threshold", 0.5)
            mask = (rendered_alpha > alpha_threshold).float()

    # ... rest of loss computation
```

---

## Patch 6: train_gslrm.py - WandB logging (Line ~770)

### 추가
```python
# Ghost metrics logging
ghost_metrics = ["ghost_fg_coverage", "ghost_alpha_std", "ghost_opacity_std", "ghost_opacity_mean"]

for k, v in loss_name2value:
    # ... existing code ...
    elif k in ghost_metrics:
        log_dict["ghost/" + k.replace("ghost_", "")] = v
```

---

## Config 설정

```yaml
training:
  dataset:
    random_view_selection: true  # 랜덤 뷰 선택

  losses:
    masked_l2_loss: true
    use_predicted_mask: false
    use_rendered_alpha_mask: true
    alpha_mask_threshold: 0.5
    pred_mask_threshold: 0.1
```

---

## 변경 요약

| 파일 | 변경 수 | 핵심 내용 |
|------|---------|----------|
| gaussians_renderer.py | 4 | diff_gauss, alpha 반환, DeferredRender 수정 |
| gslrm.py | 2 | Mask 옵션, ghost metrics |
| train_gslrm.py | 1 | WandB ghost logging |

총 **~50줄** 수정으로 alpha rendering + mask options 구현
