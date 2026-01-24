# VS Code Debug Guide: Mask/Loss 시스템 디버깅

**Version**: 2.0
**Updated**: 2026-01-24
**Purpose**: 마스크/알파/Loss 시스템 디버깅을 위한 VS Code 사용법

---

## 목차

1. [환경 설정](#1-환경-설정)
2. [핵심 브레이크포인트](#2-핵심-브레이크포인트)
3. [디버깅 체크리스트](#3-디버깅-체크리스트)
4. [mask_mode 비교표](#4-mask_mode-비교표)
5. [유용한 디버그 코드](#5-유용한-디버그-코드)

---

## 1. 환경 설정

### 1.1 VS Code Remote SSH
```
1. Ctrl+Shift+P → "Remote-SSH: Connect to Host"
2. "gpu03" 선택
3. /home/joon/dev/FaceLift 폴더 열기
4. Ctrl+Shift+P → "Python: Select Interpreter"
   → /home/joon/anaconda3/envs/facelift/bin/python
```

### 1.2 launch.json 설정
```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug: D7_1 E2_gt_alpha",
            "type": "debugpy",
            "request": "launch",
            "program": "train_gslrm.py",
            "args": ["-d", "D7_1", "-e", "E2_gt_alpha"],
            "env": {"CUDA_VISIBLE_DEVICES": "0"},
            "console": "integratedTerminal",
            "justMyCode": false,
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Debug: Quick 5 Steps",
            "type": "debugpy",
            "request": "launch",
            "program": "train_gslrm.py",
            "args": ["-d", "D7_1", "-e", "E2_gt_alpha", "--max_steps", "5"],
            "env": {"CUDA_VISIBLE_DEVICES": "0"},
            "console": "integratedTerminal",
            "justMyCode": false,
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

---

## 2. 핵심 브레이크포인트

### 2.1 Loss 계산 흐름 (gslrm/model/gslrm.py)

| 라인 | 코드 | 목적 |
|------|------|------|
| **347** | `mask = None` | GT 마스크 초기화 |
| **349** | `target_flat, mask = target_flat.split([3, 1], dim=1)` | RGBA → RGB + Mask 분리 |
| **427-428** | `alpha_loss_weight = ...` | Alpha supervision loss 시작 |
| **436-437** | `bg_loss_weight = ...` | Background penalty 시작 |

### 2.2 Masked Loss 함수 (mouse_extensions/model/mask_losses.py)

| 라인 | 함수명 | 용도 |
|------|--------|------|
| **100** | `compute_masked_rgb_loss()` | Masked RGB loss 계산 |
| **158** | `compute_alpha_supervision_loss()` | Alpha supervision loss |
| **457, 479, 492** | (호출부) | mask_mode별 분기 |
| **503** | `alpha_loss = compute_alpha_supervision_loss(...)` | Alpha loss 호출 |

### 2.3 단축키 필수
| 키 | 동작 |
|----|------|
| **F5** | 디버깅 시작 |
| **F9** | 브레이크포인트 토글 |
| **F10** | Step Over |
| **F11** | Step Into |

---

## 3. 디버깅 체크리스트

### Step 1: 데이터 로딩 확인 (Line 349)

```python
# Debug Console에서 실행
print(f"target_flat shape: {target_flat.shape}")  # [B*V, 4, H, W]
print(f"mask shape: {mask.shape}")                 # [B*V, 1, H, W]
print(f"mask range: [{mask.min():.3f}, {mask.max():.3f}]")  # [0, 1]
print(f"FG ratio: {(mask > 0.5).float().mean():.4f}")       # ~0.02-0.05

# ★ D7_1 정상 값: FG ratio ~0.024 (2.4%)
```

### Step 2: Mask Mode 확인

```python
# mask_losses.py의 분기 확인
mask_mode = config.training.losses.mask_mode
print(f"mask_mode: {mask_mode}")  # "gt", "alpha", "composite", "none"

# gt: GT mask 사용 (권장)
# alpha: rendered alpha 사용 (확장 위험)
# composite: 배경 합성
# none: 마스크 미적용
```

### Step 3: Alpha/BG Loss 확인 (Line 427-440)

```python
# Alpha loss
print(f"alpha_loss_weight: {alpha_loss_weight}")  # 0.1 권장
print(f"alpha_loss: {losses[alpha_loss].item():.6f}")

# Background loss  
print(f"bg_loss_weight: {bg_loss_weight}")        # 0.0 or 0.3
print(f"bg_loss: {losses.get(bg_loss, 0)}")

# IoU 확인 (shape 수렴도)
iou = ((mask > 0.5) & (rendered_alpha_flat > 0.5)).sum() / \
      ((mask > 0.5) | (rendered_alpha_flat > 0.5)).sum()
print(f"Mask IoU: {iou:.4f}")  # 0.5+ 양호, <0.3 문제
```

### Step 4: 시각화 저장

```python
import torchvision
torchvision.utils.save_image(mask[0], /tmp/gt_mask.png)
torchvision.utils.save_image(rendered_alpha_flat[0], /tmp/rendered_alpha.png)
torchvision.utils.save_image(pred_flat[0], /tmp/pred_rgb.png)
torchvision.utils.save_image(target_flat[0,:3], /tmp/gt_rgb.png)
```

---

## 4. mask_mode 비교표

### 4.1 설정별 동작

| mask_mode | RGB Loss 영역 | 특징 | 권장도 |
|-----------|--------------|------|--------|
| **gt** | GT mask 영역만 | 안정적, 고정 | ★★★ |
| **alpha** | Rendered α > threshold | 동적, 확장 위험 | ★☆☆ |
| **composite** | 전체 (배경 합성) | 흰 배경에 적합 | ★★☆ |
| **none** | 전체 이미지 | 마스크 미사용 | ★★☆ |

### 4.2 D7_1 마스크 분석 결과

| 방법 | Threshold | IoU vs GT | 비고 |
|------|-----------|-----------|------|
| GT alpha | 0.1-0.9 | **1.0** | 완벽 (이진 마스크) |
| rgb_pred | 0.02-0.3 | 0.06-0.08 | ❌ 부적합 |

**결론**: D7_1의 알파 채널은 깨끗한 이진값(0/255). `mask_mode: gt` 권장.

### 4.3 실험 설정 비교

| 실험 | mask_mode | alpha_w | bg_w | 문헌 근거 |
|------|-----------|---------|------|-----------|
| **E2_gt_alpha** ★ | gt | 0.1 | 0.0 | LGM+PS |
| E5_composite_strong | composite | 0.3 | 0.0 | Splatfacto-W |
| E6_lgm_full | gt | 1.0 | 0.0 | LGM |
| E6_bg_penalty_strong | none | 0.1 | 1.0 | 2DGS |
| E6_combined | gt | 0.2 | 0.3 | 다중 문헌 |

---

## 5. 유용한 디버그 코드

### 5.1 텐서 정보 출력

```python
def debug_tensor(name, t):
    if t is None:
        print(f"{name}: None")
        return
    print(f"{name}: shape={t.shape}, dtype={t.dtype}, "
          f"range=[{t.min():.3f}, {t.max():.3f}], mean={t.mean():.3f}")

# 사용
debug_tensor("mask", mask)
debug_tensor("rendered_alpha", rendered_alpha_flat)
debug_tensor("pred", pred_flat)
```

### 5.2 Loss 분해 분석

```python
import torch.nn.functional as F

# Masked vs Unmasked 비교
unmasked = F.mse_loss(pred_flat, target_flat[:,:3])
masked = (F.mse_loss(pred_flat, target_flat[:,:3], reduction=none) * mask).sum() / mask.sum()
print(f"Unmasked L2: {unmasked:.6f}")
print(f"Masked L2: {masked:.6f}")

# FG/BG 분리
diff = (pred_flat - target_flat[:,:3]).abs()
fg_err = (diff * mask).sum() / mask.sum()
bg_err = (diff * (1 - mask)).sum() / (1 - mask).sum()
print(f"FG error: {fg_err:.6f}, BG error: {bg_err:.6f}")
```

### 5.3 시각화 그리드 저장

```python
import torchvision

def save_debug_grid(tensors, names, path=/tmp/debug_grid.png):
    """여러 텐서를 그리드로 저장"""
    grids = []
    for t, n in zip(tensors, names):
        if t.dim() == 4:
            t = t[0]
        if t.size(0) == 1:
            t = t.repeat(3, 1, 1)
        grids.append(t.cpu())
    grid = torchvision.utils.make_grid(grids, nrow=len(grids), padding=2)
    torchvision.utils.save_image(grid, path)
    print(f"Saved: {path}")

# 사용
save_debug_grid(
    [mask, rendered_alpha_flat, pred_flat, target_flat[:,:3]],
    [mask, alpha, pred, gt]
)
```

---

## 관련 문서

- [Mask_Literature_Review](../theory/mask/Mask_Literature_Review.md)
- [GHOSTING_DIAGNOSIS](../analysis/GHOSTING_DIAGNOSIS.md)
- [Quick Reference](../practical/MOUSE_QUICK_REFERENCE.md)

---

*FaceLift Mouse | VS Code Debug Guide v2.0 | 2026-01-24*
