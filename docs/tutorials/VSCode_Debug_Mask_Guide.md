# VSCode Remote Debug Guide: Mask & Loss Analysis

> **목적**: VSCode Remote SSH로 FaceLift 학습 코드를 디버깅하여 mask_mode별 loss 계산 흐름을 분석
> **난이도**: 중급 (Python 디버깅 경험 필요)
> **최종 업데이트**: 2026-01-25

---

## 목차
1. [환경 설정](#1-환경-설정)
2. [launch.json 설정 상세](#2-launchjson-설정-상세)
3. [디버그 실행 방법](#3-디버그-실행-방법)
4. [핵심 브레이크포인트](#4-핵심-브레이크포인트)
5. [디버깅 체크리스트](#5-디버깅-체크리스트)
6. [디버그 유틸리티](#6-디버그-유틸리티)

---

## 1. 환경 설정

### 1.1 VSCode Remote SSH 연결

```
1. VSCode 좌측 하단 녹색 버튼 클릭
2. "Connect to Host..." 선택
3. gpu03 선택 (또는 ssh gpu03)
4. /home/joon/dev/FaceLift 폴더 열기
```

### 1.2 Python Interpreter 선택

```
Ctrl+Shift+P → "Python: Select Interpreter"
→ /home/joon/anaconda3/envs/facelift/bin/python
```

### 1.3 필수 VSCode 확장

| 확장 | 용도 |
|------|------|
| Python (ms-python) | Python 디버깅 |
| Remote - SSH | 원격 서버 연결 |
| Pylance | 코드 분석 (선택) |

### 1.4 GPU 호환성 ⚠️

> **중요**: gpu03의 GPU 0-3 (Blackwell)은 PyTorch 미지원

| GPU | 아키텍처 | 호환성 |
|-----|----------|--------|
| 0-3 | Blackwell (12.0) | ❌ 미지원 |
| 4-7 | A6000 (8.6) | ✅ 사용 가능 |

**launch.json에서 반드시 `CUDA_VISIBLE_DEVICES: "4"` 이상 설정**

### 1.5 디버그 드롭다운이 안 보일 때

1. **워크스페이스 루트 문제**: `/home/joon/dev/FaceLift` 폴더를 직접 열어야 함
   - `File → Open Folder → /home/joon/dev/FaceLift` 선택
   - 상위 폴더를 열면 `.vscode/launch.json`이 인식되지 않음

2. **Python Debugger 확장 미설치**: 원격 서버에 확장 설치 필요
   - `Extensions (Ctrl+Shift+X) → "Python Debugger" → "Install in SSH: gpu03"`

---

## 2. launch.json 설정 상세

### 2.1 파일 위치
`.vscode/launch.json` (프로젝트 루트)

### 2.2 설정 구조 설명

```jsonc
{
    "version": "0.2.0",
    "configurations": [
        {
            // === 기본 정보 ===
            "name": "Debug: D7_1 + E1_2_alpha",  // 드롭다운에 표시될 이름
            "type": "debugpy",                       // Python 디버거 타입
            "request": "launch",                     // 새 프로세스 시작 (attach도 가능)
            
            // === 실행 대상 ===
            "program": "${workspaceFolder}/train_gslrm.py",  // 디버그할 Python 파일
            "console": "integratedTerminal",                  // 출력 터미널
            "cwd": "${workspaceFolder}",                      // 작업 디렉토리
            
            // === Python 환경 ===
            "python": "/home/joon/anaconda3/envs/facelift/bin/python",
            
            // === 핵심: Arguments ===
            "args": [
                "-d", "D7_1",           // 데이터셋 (Modular mode)
                "-e", "E1_2_alpha",  // 실험 설정
                "-s", "training.schedule.max_fwdbwd_passes", "10"  // Override: 10 step만
            ],
            
            // === 환경 변수 ===
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",  // GPU 선택
                "WANDB_MODE": "disabled"       // WandB 비활성화 (디버그 시)
            },
            
            // === 디버거 옵션 ===
            "justMyCode": false  // 라이브러리 코드도 디버깅 (중요!)
        }
    ]
}
```

### 2.3 Arguments 상세 설명

#### 필수 Arguments (택일)

| 방식 | Arguments | 설명 |
|------|-----------|------|
| **Modular (권장)** | `-d D7_1 -e E1_2_alpha` | 데이터셋 + 실험 조합 |
| Legacy | `--config configs/mouse/full_config.yaml` | 단일 YAML 파일 |

**왜 Modular 방식을 권장하는가?**
- base + dataset + experiment 3단계 자동 merge
- 실험 조합 변경이 쉬움 (args만 수정)
- 설정 누락 방지 (base에 기본값 포함)

#### Override Arguments (`-s`)

```bash
# 형식: -s KEY VALUE (공백으로 구분)
-s training.schedule.max_fwdbwd_passes 10   # 10 step만 실행
-s training.losses.mask_mode gt             # mask_mode 강제 변경
-s validation.val_every 5                   # 5 step마다 validation
```

**자주 사용하는 Override:**

| Key | 값 예시 | 용도 |
|-----|---------|------|
| `training.schedule.max_fwdbwd_passes` | 10 | Quick test (10 step) |
| `training.losses.mask_mode` | gt/alpha/none | mask 방식 변경 |
| `training.losses.alpha_loss_weight` | 0.1 | alpha loss 가중치 |
| `validation.val_every` | 5 | validation 주기 |
| `data.batch_size` | 1 | 메모리 절약 |

### 2.4 권장 launch.json 설정 (복사용)

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug: Quick Test (10 steps)",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/train_gslrm.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "python": "/home/joon/anaconda3/envs/facelift/bin/python",
            "args": [
                "-d", "D7_1",
                "-e", "E1_2_alpha",
                "-s", "training.schedule.max_fwdbwd_passes", "10"
            ],
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "WANDB_MODE": "disabled"
            },
            "justMyCode": false
        },
        {
            "name": "Debug: Mask Mode = GT",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/train_gslrm.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "python": "/home/joon/anaconda3/envs/facelift/bin/python",
            "args": [
                "-d", "D7_1",
                "-e", "E1_1_base",
                "-s", "training.schedule.max_fwdbwd_passes", "5"
            ],
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "WANDB_MODE": "disabled"
            },
            "justMyCode": false
        },
        {
            "name": "Debug: M3 Dataset",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/train_gslrm.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "python": "/home/joon/anaconda3/envs/facelift/bin/python",
            "args": [
                "-d", "D10_3",
                "-e", "E1_2_alpha",
                "-s", "training.schedule.max_fwdbwd_passes", "10"
            ],
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "WANDB_MODE": "disabled"
            },
            "justMyCode": false
        }
    ]
}
```

---

## 3. 디버그 실행 방법

### 3.1 실행 파일

**항상 `train_gslrm.py`에서 실행** (프로젝트 루트)

```
train_gslrm.py (Entry Point)
    │
    ├─ Config 로딩 (ModularConfigLoader)
    ├─ 데이터 로딩 (MouseDataset)
    ├─ 모델 초기화 (GSLRM)
    └─ Training Loop
        ├─ Forward pass
        ├─ Loss 계산 ← 핵심 디버깅 포인트
        └─ Backward pass
```

### 3.2 실행 단계

```
1. F5 또는 Run → Start Debugging
2. 좌측 상단 드롭다운에서 설정 선택
3. 브레이크포인트에서 멈춤
4. Debug Console에서 변수 검사
```

### 3.3 주요 단축키

| 단축키 | 기능 |
|--------|------|
| F5 | 디버깅 시작/계속 |
| F10 | Step Over (다음 줄) |
| F11 | Step Into (함수 내부로) |
| Shift+F11 | Step Out (함수 밖으로) |
| F9 | 브레이크포인트 토글 |
| Ctrl+Shift+Y | Debug Console 열기 |

---

## 4. 핵심 브레이크포인트

### 4.1 Loss 계산 흐름

```
gslrm.py: forward()
    │
    ├─ Line ~347: images, masks 로딩
    │   → images.shape, masks 값 확인
    │
    ├─ Line ~427: compute_loss() 호출
    │   → 여기서 Step Into (F11)
    │
    └─ loss_extensions.py: compute_loss()
        │
        ├─ Line ~89: mask_mode 분기
        │   → config의 mask_mode 값 확인
        │
        ├─ Line ~120: compute_mask_from_config()
        │   → 실제 마스크 생성 로직
        │
        └─ Line ~180: masked_l2_loss 계산
            → loss 값, mask 적용 영역 확인
```

### 4.2 권장 브레이크포인트 위치

| 파일 | 라인 | 변수 확인 |
|------|------|-----------|
| `gslrm.py` | ~347 | `images.shape`, `masks.shape` |
| `gslrm.py` | ~427 | `loss_dict` 반환값 |
| `loss_extensions.py` | ~89 | `mask_mode` 값 |
| `loss_extensions.py` | ~120 | `mask` tensor |
| `loss_extensions.py` | ~180 | `l2_loss`, `masked_pixels` |
| `mask_losses.py` | ~45 | `alpha_loss` 계산 |

### 4.3 Debug Console 명령어

```python
# 브레이크포인트에서 실행 가능
images.shape          # torch.Size([B, V, 3, H, W])
masks.mean()          # 마스크 평균 (0~1)
masks.sum() / masks.numel()  # 마스크 커버리지

# 시각화 저장
import torchvision
torchvision.utils.save_image(masks[0], '/tmp/mask_debug.png')
```

---

## 5. 디버깅 체크리스트

### 5.1 mask_mode별 예상 동작

| mask_mode | RGB Loss 영역 | Alpha Loss | 배경 처리 |
|-----------|---------------|------------|-----------|
| `none` | 전체 이미지 | 없음 | 함께 학습 |
| `gt` | GT mask 영역만 | 선택적 | 제외 |
| `alpha` | Rendered α 영역 | 자동 | ⚠️ 확장 위험 |
| `composite` | GT × α 혼합 | 있음 | 가중 제외 |

### 5.2 단계별 체크

```
□ Step 1: 데이터 로딩 확인
  - images.shape == [B, num_views, 3, 512, 512]
  - masks는 GT mask (0 또는 1)

□ Step 2: mask_mode 확인
  - config['training']['losses']['mask_mode'] 값

□ Step 3: 마스크 적용 확인
  - compute_mask_from_config() 반환값
  - mask.sum() / mask.numel() → 예상 커버리지 (0.05 정도)

□ Step 4: Loss 값 확인
  - l2_loss: 정상 범위 (0.01 ~ 0.1)
  - alpha_loss: 설정된 경우 확인
  - perceptual_loss: 전체 이미지 기준
```

### 5.3 문제 진단

| 증상 | 가능한 원인 | 확인 방법 |
|------|-------------|-----------|
| Loss NaN | bf16 + log 연산 | `opacity.dtype` 확인 |
| 마스크 전체 1 | mask_mode 미적용 | `mask.mean()` 확인 |
| 커버리지 증가 | alpha mask 악순환 | `fg_coverage` 로그 확인 |
| PSNR 낮음 | 데이터/설정 문제 | validation 이미지 확인 |

---

## 6. 디버그 유틸리티

### 6.1 텐서 디버그 함수

```python
def debug_tensor(name: str, tensor: torch.Tensor):
    """텐서 상태 출력"""
    print(f"[DEBUG] {name}:")
    print(f"  shape: {tensor.shape}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  range: [{tensor.min():.4f}, {tensor.max():.4f}]")
    print(f"  mean: {tensor.mean():.4f}, std: {tensor.std():.4f}")
    if tensor.isnan().any():
        print(f"  ⚠️ Contains NaN!")
    if tensor.isinf().any():
        print(f"  ⚠️ Contains Inf!")
```

### 6.2 마스크 시각화 저장

```python
def save_debug_masks(masks: torch.Tensor, rendered_alpha: torch.Tensor, 
                     step: int, output_dir: str = "/tmp/debug"):
    """GT mask와 rendered alpha 비교 저장"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    import torchvision
    # GT mask
    torchvision.utils.save_image(
        masks[0], f"{output_dir}/step{step:04d}_gt_mask.png"
    )
    # Rendered alpha
    torchvision.utils.save_image(
        rendered_alpha[0], f"{output_dir}/step{step:04d}_rendered_alpha.png"
    )
    # Difference
    diff = (masks[0] - rendered_alpha[0]).abs()
    torchvision.utils.save_image(
        diff, f"{output_dir}/step{step:04d}_diff.png"
    )
```

### 6.3 Loss 히스토리 추적

```python
class LossTracker:
    def __init__(self):
        self.history = {'l2': [], 'alpha': [], 'perceptual': []}
    
    def update(self, loss_dict):
        for key in self.history:
            if key in loss_dict:
                self.history[key].append(loss_dict[key].item())
    
    def plot(self, save_path="/tmp/loss_history.png"):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for key, values in self.history.items():
            if values:
                ax.plot(values, label=key)
        ax.legend()
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
```

---

## 관련 문서

| 문서 | 위치 | 내용 |
|------|------|------|
| EXPERIMENT_QUICK_REFERENCE | `docs/practical/` | 실험 ID 및 설정 |
| MOUSE_QUICK_REFERENCE | `docs/practical/` | 전체 워크플로우 |
| MASK_SYSTEM_GUIDE | `docs/practical/config/` | 마스크 설정 상세 |

---

*FaceLift Debug Guide v2.0 | 2026-01-24*

---

## 7. 환경변수 기반 자동 브레이크포인트 (NEW)

### 7.1 개요

VSCode 브레이크포인트는 워크스페이스/세션 변경 시 사라질 수 있습니다.
**해결책**: 코드에 조건부 `debug_break()` 삽입

### 7.2 설정된 브레이크포인트 위치

| 위치 | 파일 | 라인 | 환경변수 | 용도 |
|------|------|------|----------|------|
| BP1 | `loss_extensions.py` | 140 | `DEBUG_MASK=1` | mask_mode 분기 체크 |
| BP2 | `loss_extensions.py` | 154 | `DEBUG_MASK=1` | GT mask 적용 시점 |
| BP3 | `mask_losses.py` | 134 | `DEBUG_LOSS=1` | masked RGB loss 계산 |
| BP4 | `gslrm.py` | 382 | `DEBUG_LOSS=1` | _compute_all_losses 진입 |
| BP5 | `gslrm.py` | 386 | `DEBUG_LOSS=1` | mask 계산 직후 (inspect) |
| BP6 | `gslrm.py` | 1458 | `DEBUG_LOSS=1` | loss_calculator 호출 전 |

### 7.3 사용법

```bash
# 마스크 관련 디버깅
DEBUG_MASK=1 python train_gslrm.py -d D7_1 -e E1_2_gt_alpha

# Loss 관련 디버깅
DEBUG_LOSS=1 python train_gslrm.py -d D7_1 -e E1_2_gt_alpha

# 전체 디버깅
DEBUG_ALL=1 python train_gslrm.py -d D7_1 -e E1_2_gt_alpha
```

### 7.4 launch.json 설정

이미 구성된 디버그 프로필:

| 프로필 | 환경변수 | 용도 |
|--------|----------|------|
| `Debug: Quick Test` | - | 일반 테스트 |
| `Debug: Mask Mode` | `DEBUG_MASK=1` | 마스크 흐름 |
| `Debug: Loss` | `DEBUG_LOSS=1` | Loss 계산 |
| `Debug: All` | `DEBUG_ALL=1` | 전체 추적 |

  VSCode Debug 드롭다운에서 선택:                                                                                                                                                  
  Debug: Mask Mode (DEBUG_MASK=1)  → BP1, BP2에서 멈춤                                                                                                                             
  Debug: Loss (DEBUG_LOSS=1)       → BP3, BP4, BP5, BP6에서 멈춤                                                                                                                   
  Debug: All (DEBUG_ALL=1)         → 모든 BP에서 멈춤           

### 7.5 커스텀 브레이크포인트 추가

```python
from mouse_extensions.utils.debug_breakpoints import debug_break, debug_inspect

# 조건부 브레이크포인트
debug_break("mask")  # DEBUG_MASK=1일 때만 멈춤

# 텐서 검사 + 이미지 저장
debug_inspect("my_tensor", tensor, "/tmp/debug.png")
```

---

*Updated: 2026-01-25 - 환경변수 기반 자동 브레이크포인트 추가*

### 7.6 디버그 흐름도

```
train_gslrm.py
    │
    ▼
gslrm.py: forward()
    │
    ├─ BP6 (line 1458): loss_calculator 호출 전
    │
    ▼
gslrm.py: _compute_all_losses()
    │
    ├─ BP4 (line 382): 진입점
    │
    ▼
loss_extensions.py: compute_mask_from_config()
    │
    ├─ BP1 (line 140): mask_mode 체크
    ├─ BP2 (line 154): GT mask 적용
    │
    ▼
gslrm.py: mask 반환
    │
    ├─ BP5 (line 386): mask 텐서 검사 (debug_inspect)
    │
    ▼
mask_losses.py: compute_masked_rgb_loss()
    │
    └─ BP3 (line 134): Loss 계산
```

### 7.7 권장 디버깅 시나리오

| 시나리오 | 환경변수 | 확인할 BP |
|----------|----------|-----------|
| mask_mode가 적용되는지 확인 | `DEBUG_MASK=1` | BP1 |
| GT mask 값이 올바른지 확인 | `DEBUG_MASK=1` | BP2 |
| Loss 계산 전체 흐름 추적 | `DEBUG_LOSS=1` | BP4→BP5→BP3 |
| 모든 흐름 상세 추적 | `DEBUG_ALL=1` | 전체 |

### 7.8 Debug Console에서 유용한 명령어

```python
# BP5에서 mask 검사
mask.shape                    # [B*V, 1, H, W]
mask.mean()                   # 마스크 평균 (0~1)
(mask > 0.5).sum() / mask.numel()  # FG 커버리지

# 텐서 이미지로 저장
import torchvision
torchvision.utils.save_image(mask[0], "/tmp/mask_bp5.png")
torchvision.utils.save_image(rendering[0], "/tmp/render_bp5.png")

# Loss 값 확인 (BP3 이후)
losses["l2"].item()
losses["psnr"].item()
```

---

## 8. 최신 업데이트 (2026-01-25)

### 8.1 새로운 데이터셋

| Dataset | 특징 | 용도 |
|---------|------|------|
| D3_normalized | PP=256 고정, fx=549 | 기존 실험 |
| M3 | D10.3 preset, adaptive zoom | 신규 권장 |
| M3_1 | Center-aligned, PP=256 | M3 변형 |

### 8.2 새로운 실험 Config

| Config | 용도 | 특징 |
|--------|------|------|
| **E_debug** | 빠른 테스트 | 36 views, 18초 영상 |
| **E_vis_quality** | 고품질 영상 | 180 views, 36초 영상 |
| **E_overfit_6v** | Overfitting 테스트 | 6개 뷰 모두 input |

### 8.3 View Selection 이슈 (중요!)

**문제**: Validation에서 View 4, 5가 항상 novel view → PSNR 10-13

**원인**: `mouse_dataset.py:276-279`
```python
# Validation에서 고정 input [0,1,2,3]
if split != "train":
    input_indices = list(range(4))  # View 4,5 항상 novel
```

**디버깅 시 확인**:
```python
# BP에서 확인
print(f"split: {self.split}")
print(f"input_indices: {input_indices}")
print(f"target views: {[i for i in range(6) if i not in input_indices]}")
```

### 8.4 Per-View PSNR 디버깅

```python
# Validation 후 per-view metrics 확인
# experiments/validation/{dataset}_{exp}/iter_*/*/perview_metrics.txt

# 패턴 확인 스크립트
import glob
files = glob.glob("experiments/validation/*/iter_*/*/perview_metrics.txt")
for f in files[-1:]:
    with open(f) as fp:
        print(fp.read())
```

---

## 9. 관련 문서 (백링크)

### 9.1 분석 문서

| 문서 | 경로 | 내용 |
|------|------|------|
| **View Selection 분석** | `docs/analysis/VIEW_SELECTION_GHOSTING_ANALYSIS.md` | Ghosting 원인, 해결방안 |
| PP/MVG 분석 | `docs/analysis/PP_MVG_COMPREHENSIVE_ANALYSIS.md` | Principal Point 이론 |

### 9.2 실험 문서

| 문서 | 경로 | 내용 |
|------|------|------|
| Quick Reference | `docs/practical/MOUSE_QUICK_REFERENCE.md` | 전체 워크플로우 |
| Experiment Registry | `docs/practical/EXPERIMENT_REGISTRY.md` | 실험 ID 목록 |

### 9.3 전처리 문서

| 문서 | 경로 | 내용 |
|------|------|------|
| M3 Series Spec | `docs/datasets/M3_SERIES_SPEC.md` | M3 데이터셋 상세 |
| Preprocessing Registry | `docs/PREPROCESSING_REGISTRY.md` | 전처리 방식 목록 |

### 9.4 시각화 모듈

| 파일 | 경로 | 내용 |
|------|------|------|
| video_generator | `mouse_extensions/visualization/video_generator.py` | 통합 비디오 생성 |
| turntable_config | `mouse_extensions/visualization/turntable_config.py` | 카메라 순서 [1,3,5,0,4,2] |

---

## 10. 문서 업데이트 체크리스트

이 문서 수정 시 함께 확인할 파일:

```
□ docs/analysis/VIEW_SELECTION_GHOSTING_ANALYSIS.md
  - View selection 로직 변경 시
  
□ docs/practical/EXPERIMENT_REGISTRY.md
  - 새 실험 config 추가 시
  
□ mouse_extensions/visualization/turntable_config.py
  - 카메라 순서 변경 시
  
□ configs/experiments/E_*.yaml
  - 실험 설정 변경 시
```

---

*Updated: 2026-01-25 - 최신 데이터셋, View Selection 이슈, 백링크 추가*
