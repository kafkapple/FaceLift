# VS Code Debug Guide: Mask System Debugging

**Version**: 1.0
**Created**: 2026-01-24
**Purpose**: 마스크/알파 시스템 디버깅을 위한 VS Code 사용법

---

## 1. VS Code Remote SSH 설정

### 1.1 원격 서버 연결
```
1. VS Code에서 Ctrl+Shift+P → "Remote-SSH: Connect to Host"
2. "gpu03" 선택 (또는 ssh gpu03 설정 추가)
3. /home/joon/dev/FaceLift 폴더 열기
```

### 1.2 Python 인터프리터 설정
```
Ctrl+Shift+P → "Python: Select Interpreter"
→ /home/joon/anaconda3/envs/facelift/bin/python 선택
```

---

## 2. launch.json 디버그 설정

### 2.1 설정 파일 위치
```
/home/joon/dev/FaceLift/.vscode/launch.json
```

### 2.2 마스크 디버깅용 설정

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug: Mask Training (D7_1 E1)",
            "type": "debugpy",
            "request": "launch",
            "program": "train_gslrm.py",
            "args": [
                "--config", "configs/mouse/mask_exp/D7_mask_E1_gt_alpha_sup.yaml"
            ],
            "env": {
                "CUDA_VISIBLE_DEVICES": "0"
            },
            "console": "integratedTerminal",
            "justMyCode": false,
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Debug: Quick Test (5 steps)",
            "type": "debugpy",
            "request": "launch",
            "program": "train_gslrm.py",
            "args": [
                "--config", "configs/mouse/mask_exp/D7_mask_E1_gt_alpha_sup.yaml",
                "--max_steps", "5"
            ],
            "env": {
                "CUDA_VISIBLE_DEVICES": "0"
            },
            "console": "integratedTerminal",
            "justMyCode": false,
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

---

## 3. 핵심 브레이크포인트 위치

### 3.1 마스크 관련 핵심 코드 위치

| 파일 | 라인 | 목적 |
|------|------|------|
| `gslrm/model/gslrm.py:347` | `mask = None` | GT 마스크 추출 시작 |
| `gslrm/model/gslrm.py:349` | `target_flat, mask = target_flat.split` | RGBA → RGB + Mask 분리 |
| `gslrm/model/gslrm.py:427-431` | `alpha_loss_weight` | Alpha supervision loss 계산 |
| `gslrm/model/gslrm.py:436-440` | `bg_loss_weight` | Background penalty 계산 |
| `mouse_extensions/model/mask_losses.py:124` | `compute_masked_rgb_loss` | Masked RGB loss 함수 |
| `mouse_extensions/model/mask_losses.py:168` | `compute_alpha_supervision_loss` | Alpha loss 함수 |

### 3.2 브레이크포인트 설정 방법
```
1. 해당 파일 열기
2. 라인 번호 왼쪽 클릭 → 빨간 점 생성
3. 또는: 해당 라인에서 F9 누르기
```

---

## 4. 디버깅 단축키 (★ 필수)

| 단축키 | 동작 | 설명 |
|--------|------|------|
| **F5** | 디버깅 시작 | launch.json 설정으로 실행 |
| **F9** | 브레이크포인트 토글 | 현재 라인에 BP 추가/제거 |
| **F10** | Step Over | 현재 라인 실행 후 다음으로 |
| **F11** | Step Into | 함수 내부로 들어감 |
| **Shift+F11** | Step Out | 현재 함수 빠져나옴 |
| **F6** | Pause | 실행 중 일시정지 |
| **Shift+F5** | 디버깅 중지 | 완전 종료 |
| **Ctrl+Shift+F5** | 재시작 | 처음부터 다시 |

---

## 5. 변수 검사 방법

### 5.1 Debug Console 사용 (★ 핵심)

브레이크포인트에서 정지 후:

```python
# 텐서 기본 정보
mask.shape            # 예: torch.Size([6, 1, 512, 512])
mask.dtype            # 예: torch.float32
mask.device           # 예: cuda:0

# 텐서 값 확인
mask.min(), mask.max()        # 값 범위
mask.mean()                   # 평균값
(mask > 0.5).sum()           # 양수 픽셀 수
(mask > 0.5).float().mean()  # 전경 비율

# 특정 픽셀 값
mask[0, 0, 256, 256]         # 중앙 픽셀

# 시각화 (파일 저장)
import torchvision
torchvision.utils.save_image(mask, '/tmp/debug_mask.png')
torchvision.utils.save_image(rendered_alpha, '/tmp/debug_alpha.png')
```

### 5.2 Watch 패널 사용

왼쪽 "WATCH" 패널에 추가:
```
mask.shape
mask.min().item()
mask.max().item()
rendered_alpha.shape if rendered_alpha is not None else 'None'
```

### 5.3 Variables 패널

- **Locals**: 현재 함수 내 지역 변수
- **Globals**: 전역 변수 (보통 불필요)
- 변수 옆 **>** 클릭하여 내부 구조 확인

---

## 6. 마스크 디버깅 체크리스트

### Step 1: 데이터 로딩 확인 (gslrm.py:349)

브레이크포인트 후 확인:
```python
# GT 마스크 확인
print(f"target_flat shape: {target_flat.shape}")  # [B*V, 4, H, W]
print(f"mask shape: {mask.shape}")                 # [B*V, 1, H, W]
print(f"mask range: [{mask.min():.3f}, {mask.max():.3f}]")
print(f"foreground ratio: {(mask > 0.5).float().mean():.4f}")  # 약 0.03-0.05

# 시각화
torchvision.utils.save_image(mask[0], '/tmp/gt_mask.png')
```

### Step 2: Alpha 렌더링 확인 (gslrm.py:354)

```python
# Rendered alpha 확인
print(f"rendered_alpha shape: {rendered_alpha_flat.shape}")
print(f"alpha range: [{rendered_alpha_flat.min():.3f}, {rendered_alpha_flat.max():.3f}]")

# GT vs Pred 비교
iou = ((mask > 0.5) & (rendered_alpha_flat > 0.5)).sum() / \
      ((mask > 0.5) | (rendered_alpha_flat > 0.5)).sum()
print(f"Mask IoU: {iou:.4f}")

# 시각화
torchvision.utils.save_image(rendered_alpha_flat[0], '/tmp/rendered_alpha.png')
```

### Step 3: Loss 계산 확인 (gslrm.py:427-440)

```python
# Alpha loss 확인
print(f"alpha_loss_weight: {alpha_loss_weight}")
print(f"alpha_loss: {losses['alpha_loss'].item():.6f}")

# Background loss 확인
print(f"bg_loss_weight: {bg_loss_weight}")
print(f"bg_loss: {losses['bg_loss'].item():.6f}")

# L2 loss 비교 (masked vs unmasked)
print(f"l2_loss: {losses['l2_loss'].item():.6f}")
```

### Step 4: 마스크 적용 전후 비교

```python
# mask_losses.py:124 근처에서
import torch.nn.functional as F

# Unmasked L2
unmasked_loss = F.mse_loss(pred, gt)
print(f"Unmasked L2: {unmasked_loss.item():.6f}")

# Masked L2
masked_loss = compute_masked_rgb_loss(pred, gt, mask, 'l2', True)
print(f"Masked L2 (normalized): {masked_loss.item():.6f}")

# 차이 분석
diff = (pred - gt).abs()
fg_diff = (diff * mask).sum() / mask.sum()
bg_diff = (diff * (1 - mask)).sum() / (1 - mask).sum()
print(f"FG error: {fg_diff.item():.6f}, BG error: {bg_diff.item():.6f}")
```

---

## 7. 일반적인 문제 및 디버그 방법

### 7.1 마스크가 None인 경우
```python
# 확인
if mask is None:
    print("WARNING: mask is None")
    print(f"target_flat channels: {target_flat.shape[1]}")  # 3이면 RGB only
```
**원인**: 데이터에 알파 채널 없음 → RGBA 데이터 사용 확인

### 7.2 마스크 값이 비정상적인 경우
```python
# 범위 확인
if mask.min() < 0 or mask.max() > 1:
    print(f"WARNING: mask out of [0,1]: [{mask.min()}, {mask.max()}]")
```
**해결**: 전처리에서 정규화 확인

### 7.3 Alpha와 GT Mask 불일치
```python
# IoU 확인
iou = compute_iou(rendered_alpha > 0.5, mask > 0.5)
if iou < 0.5:
    print(f"WARNING: Low IoU ({iou:.3f}) - shape not converged")
```
**해결**: alpha_loss_weight 증가 또는 학습 step 추가

---

## 8. 디버그 이미지 저장 위치

디버그 중 생성하는 이미지:
```
/tmp/debug_mask.png          # GT 마스크
/tmp/debug_alpha.png         # Rendered alpha
/tmp/debug_diff.png          # RGB 차이
/tmp/debug_rendering.png     # 렌더링 결과
```

```python
# 유틸리티 함수
def save_debug_tensor(tensor, name):
    import torchvision
    path = f'/tmp/debug_{name}.png'
    if tensor.dim() == 4:
        tensor = tensor[0]  # 첫 번째 배치
    if tensor.size(0) == 1:
        tensor = tensor.repeat(3, 1, 1)  # 그레이스케일 → RGB
    torchvision.utils.save_image(tensor, path)
    print(f"Saved: {path}")
```

---

## 9. 참고 자료

### 관련 파일
- `gslrm/model/gslrm.py`: 메인 모델 (loss 계산)
- `mouse_extensions/model/mask_losses.py`: 마스크 loss 함수들
- `mouse_extensions/model/loss_extensions.py`: 기존 loss 확장

### 관련 문서
- [MoC INDEX](../00_MoC_INDEX.md)
- [ALPHA_MASK_COMPLETE_GUIDE](../theory/mask/ALPHA_MASK_COMPLETE_GUIDE.md)
- [Mask_Literature_Review](../theory/mask/Mask_Literature_Review.md)

---

*FaceLift Mouse | VS Code Debug Guide v1.0 | 2026-01-24*
