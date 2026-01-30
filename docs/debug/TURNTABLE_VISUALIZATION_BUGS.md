# Turntable Visualization Bugs Analysis

> **Date**: 2026-01-30
> **Status**: 분석 완료, 수정 대기
> **Related**: GS-LRM Training/Validation Turntable

---

## 1. 이슈 요약

| # | 이슈 | 심각도 | 파일 |
|---|------|--------|------|
| A | Train/Val 입력 스트립 불일치 | Medium | `gslrm.py` vs `validator.py` |
| B | 카메라 인덱스 ↔ 이미지 불일치 | **High** | `gaussians_renderer.py:1490` |
| C | Grid에 카메라 라벨 중복 | Low | `gslrm.py:1619` + overlay |

---

## 2. 이슈 A: Train/Val 입력 스트립 불일치

### 현상
- **Train**: 하단에 카메라 라벨 + In/Pred 구분 표시
- **Val**: 단순 입력 이미지만 표시 (라벨 없음)

### 원인 코드

**Train** (`gslrm/model/gslrm.py:1653-1665`):
```python
labeled_input = create_labeled_input_strip(
    target_data.image[batch_idx],  # All 6 views
    camera_order=camera_order,
    target_h=input_strip_h,
    target_w=turntable_resolution,
    border=2,
    input_indices=input_indices,
)
input_seq = np.tile(labeled_input[None], (video_frames.shape[0], 1, 1, 1))
combined_frames = np.concatenate((video_frames, input_seq), axis=1)
```

**Validation** (`mouse_extensions/validation/validator.py:404-414`):
```python
def _save_with_input(self, frames, input_np, render_res, fps, output_dir):
    """Save turntable with input overlay."""
    # 단순 리사이즈 + 패딩, 라벨 없음
    resized = cv2.resize(input_np, (render_res - border * 2, target_h - border * 2))
    bordered = np.pad(resized, ((border, border), ...), mode="constant")
    input_seq = np.tile(bordered[None], (frames.shape[0], 1, 1, 1))
    combined = np.concatenate((frames, input_seq), axis=1)
```

### 해결 방안

**Option 1**: Validation에서 `create_labeled_input_strip` 사용
```python
# validator.py 수정
from gslrm.model.gaussians_renderer import create_labeled_input_strip

def _save_with_input(self, frames, target_images, camera_order, ...):
    labeled_input = create_labeled_input_strip(
        target_images,
        camera_order=camera_order,
        target_h=input_strip_h,
        target_w=render_res,
    )
```

**Option 2**: 공통 함수를 `mouse_extensions/visualization/` 로 이동

---

## 3. 이슈 B: 카메라 인덱스 ↔ 이미지 불일치 (핵심 버그)

### 현상
- 영상 상단: "Cam 0 -> 4" 순서로 자연스럽게 회전
- 하단 이미지: 라벨은 "0, 4, 2, 1, 3, 5" 이지만 실제 이미지가 다름

### 원인 분석

**`create_labeled_input_strip` 함수** (`gaussians_renderer.py:1490-1565`):

```python
def create_labeled_input_strip(
    all_images: torch.Tensor,      # [V, C, H, W] - 로드 순서대로
    camera_order: list,            # [0, 4, 2, 1, 3, 5] - azimuth 기반 물리적 순서
    ...
):
    # 문제 코드:
    for cam_idx in camera_order:   # cam_idx = 0, 4, 2, 1, 3, 5
        if cam_idx < num_views:
            img = all_images[cam_idx, :3, ...]  # ❌ 텐서 인덱스로 사용
            ...
            label = f'{cam_idx}'  # 라벨은 cam_idx (물리적 카메라 번호)
```

**문제점**:
- `all_images` 텐서는 **로드 순서** `[0, 1, 2, 3, 4, 5]`로 저장됨
- `camera_order`는 **azimuth 기반** `[0, 4, 2, 1, 3, 5]`
- `all_images[4]`는 "물리적 Cam 4"가 아니라 "5번째로 로드된 이미지"

**예시**:
```
camera_order = [0, 4, 2, 1, 3, 5]

Position 0: cam_idx=0 → all_images[0] → Cam 0 이미지 ✅
Position 1: cam_idx=4 → all_images[4] → Cam 4 이미지 ✅ (우연히 일치)
Position 2: cam_idx=2 → all_images[2] → Cam 2 이미지 ✅ (우연히 일치)
...
```

실제로는 일치하는 경우가 많지만, **데이터셋 로딩 순서가 카메라 인덱스와 다르면 불일치 발생**.

### 검증 필요 사항

```python
# 데이터셋에서 이미지 로드 순서 확인
# mouse_dataset.py의 __getitem__ 반환값 순서 확인
```

### 해결 방안

**Option 1**: 텐서 인덱스와 카메라 인덱스 분리
```python
def create_labeled_input_strip(
    all_images: torch.Tensor,      # [V, C, H, W]
    camera_order: list,            # 표시 순서 (물리적)
    tensor_to_cam_map: dict = None # {tensor_idx: cam_idx}
):
    for i, cam_idx in enumerate(camera_order):
        tensor_idx = cam_idx if tensor_to_cam_map is None else tensor_to_cam_map.get(cam_idx, cam_idx)
        img = all_images[tensor_idx, :3, ...]
        label = f'{cam_idx}'  # 물리적 카메라 번호
```

**Option 2**: 이미지 표시 순서는 로드 순서대로, 라벨만 매핑
```python
for tensor_idx in range(num_views):
    cam_idx = camera_order[tensor_idx] if camera_order else tensor_idx
    img = all_images[tensor_idx, :3, ...]
    label = f'{cam_idx}'
```

---

## 4. 이슈 C: Grid에 카메라 라벨 중복

### 현상
- turntable grid 이미지(`turntable_*.jpg`)에서 카메라 정보가 두 줄로 중복

### 원인 코드

**1차 라벨**: `render_dataset_trajectory()` (`gaussians_renderer.py:1368-1380`)
```python
if show_overlay:
    if is_hold:
        text = f"Cam {current_cam} [HOLD]"
    else:
        text = f"Cam {from_cam} -> {to_cam}"
    frame = add_camera_overlay(frame, text)  # 각 프레임에 텍스트 추가
```

**2차 라벨**: `add_row_labels_to_grid()` (`gslrm.py:1619-1631`)
```python
if turntable_cfg.get("add_row_labels", True):
    turntable_grid = add_row_labels_to_grid(
        turntable_grid, camera_order, grid_rows, grid_cols, h_img
    )
```

### 해결 방안

**Option 1**: Config로 분리 제어
```yaml
visualization:
  turntable:
    show_frame_overlay: true    # 각 프레임 텍스트 (영상용)
    add_row_labels: false       # Grid 상단 라벨 (중복 방지)
```

**Option 2**: Grid 생성 시 overlay 없는 프레임 사용
```python
# render_dataset_trajectory 호출 시 show_overlay=False
frames, segments = render_dataset_trajectory(..., show_overlay=False)
# Grid에만 라벨 추가
turntable_grid = add_row_labels_to_grid(...)
```

---

## 5. 관련 파일 목록

| 파일 | 역할 | 주요 함수 |
|------|------|----------|
| `gslrm/model/gslrm.py` | Training 시각화 | `save_training_results()` |
| `gslrm/model/gaussians_renderer.py` | 렌더링 유틸리티 | `create_labeled_input_strip()`, `add_camera_overlay()`, `add_row_labels_to_grid()`, `render_dataset_trajectory()` |
| `mouse_extensions/validation/validator.py` | Validation 시각화 | `_create_turntable()`, `_save_with_input()` |
| `mouse_extensions/visualization/turntable_config.py` | 카메라 순서 설정 | `get_dynamic_camera_order()` |

---

## 6. 권장 수정 순서

1. **[High]** 이슈 B 해결: `create_labeled_input_strip` 텐서 인덱싱 수정
2. **[Medium]** 이슈 A 해결: Validation에서 동일 함수 사용
3. **[Low]** 이슈 C 해결: Config 옵션으로 중복 제어

---

## 7. 테스트 계획

```bash
# 수정 후 검증
cd /home/joon/dev/FaceLift

# 1. 단일 샘플 테스트
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.test_turntable_labels \
    --checkpoint checkpoints/gslrm/M5t_E1_2_alpha/best_psnr.pt \
    --sample_idx 0

# 2. 영상-이미지 일치 확인
# - turntable_with_input_*.mp4 영상의 Cam 0 위치와
# - 하단 "0 (In)" 라벨 이미지가 동일한지 육안 확인
```

---

*Debug Note | Created: 2026-01-30 | Author: Claude Code*

---

## Appendix: MVDiffusion Checkpoint 저장 로직

### 현재 상태

**Config** (`mouse_mvdiffusion_M5t.yaml:92-93`):
```yaml
checkpointing_steps: 500        # 500 step마다 저장
checkpoints_total_limit: 5      # 최근 5개만 유지
```

**저장 로직** (`train_diffusion.py:1085-1107`):
```python
if global_step % cfg.checkpointing_steps == 0:
    # 오래된 체크포인트 삭제 (total_limit 초과 시)
    if cfg.checkpoints_total_limit is not None:
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        if len(checkpoints) >= cfg.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - cfg.checkpoints_total_limit + 1
            for removing_checkpoint in removing_checkpoints:
                shutil.rmtree(removing_checkpoint)
    
    # 새 체크포인트 저장
    save_path = os.path.join(model_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
```

### 문제점

| 항목 | GS-LRM | MVDiffusion |
|------|--------|-------------|
| Best 저장 | ✅ `best_psnr.pt` | ❌ 없음 |
| 저장 기준 | Val PSNR 기반 | Step 기반 |
| 유지 개수 | `keep_last_n` + best | `total_limit`개 |

### 개선 제안

**Option 1**: Validation loss 기반 best 저장 추가
```python
# train_diffusion.py 수정
best_val_loss = float('inf')

def log_validation(...):
    val_loss = compute_validation_loss(...)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = os.path.join(model_dir, "best")
        accelerator.save_state(save_path)
        logger.info(f"New best checkpoint (val_loss={val_loss:.4f})")
    
    return val_loss
```

**Option 2**: FID/CLIP Score 기반 best 저장
```python
# 생성 품질 지표 기반 (더 적절할 수 있음)
best_fid = float('inf')

def evaluate_generation_quality(...):
    fid_score = compute_fid(generated_images, gt_images)
    if fid_score < best_fid:
        save_best_checkpoint()
```

---

*Appendix added: 2026-01-30*

---

## 수정 완료 (2026-01-30)

### Issue B: Camera Index Bug - ✅ FIXED

**수정 파일**: `gslrm/model/gaussians_renderer.py`

**변경 내용**:
1. `create_labeled_input_strip()` 함수에 `view_indices` 파라미터 추가
2. `view_indices`가 제공되면 camera ID → tensor index 매핑 생성
3. `cam_idx` 대신 `tensor_idx`로 텐서 인덱싱

```python
# Before (buggy)
img = all_images[cam_idx, :3, ...]  # cam_idx != tensor_idx

# After (fixed)  
tensor_idx = cam_to_tensor_idx.get(cam_idx)
img = all_images[tensor_idx, :3, ...]  # correct tensor index
```

**호출 수정** (`gslrm.py`):
```python
view_indices = target_data.index[batch_idx, :, 0].cpu().numpy().tolist()
labeled_input = create_labeled_input_strip(
    ...,
    view_indices=view_indices,  # NEW: pass tensor->camera mapping
)
```

### Issue C: Duplicate Labels - ✅ FIXED

**수정 파일**: `gslrm/model/gslrm.py`

**변경 내용**:
- `show_overlay` 하드코딩 제거
- Config에서 `show_frame_overlay` 옵션으로 제어 (기본값: `False`)

```python
show_frame_overlay = turntable_cfg.get("show_frame_overlay", False)
turntable_frames, segments = render_dataset_trajectory(
    ...,
    show_overlay=show_frame_overlay,  # configurable, default False
)
```

**Config 예시**:
```yaml
visualization:
  turntable:
    add_row_labels: true        # Row labels on grid (default)
    show_frame_overlay: false   # Frame-level "Cam X" overlay (default False)
```

### Issue A: Train/Val Mismatch - ✅ FIXED

**수정 파일**: `mouse_extensions/validation/validator.py`

**변경 내용**:
1. `create_labeled_input_strip` import 추가
2. `_save_with_input()` 함수 확장 - 새 파라미터 지원
3. Training과 동일한 labeled input strip 사용

**변경된 함수 시그니처**:
```python
def _save_with_input(self, frames, input_np, render_res, fps, output_dir, 
                    target_images=None, camera_order=None, 
                    view_indices=None, input_indices=None):
```

**호출 수정**:
```python
self._save_with_input(
    frames, input_np, render_res, fps, output_dir,
    target_images=target_data.image[batch_idx],
    camera_order=camera_order,
    view_indices=view_indices,
    input_indices=input_indices,
)
```

### 검증 방법

```bash
# 1. Syntax check
python3 -m py_compile gslrm/model/gaussians_renderer.py
python3 -m py_compile gslrm/model/gslrm.py
python3 -m py_compile mouse_extensions/validation/validator.py

# 2. Run training for a few steps
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 train_gslrm.py \
    -d M5 -e M5t_E1 --vis_every 50

# 3. Check outputs
# - turntable_with_input_*.mp4: Input strip labels should match video frames
# - No duplicate "Cam X" labels (unless show_frame_overlay: true)
# - Train and Val turntables should have same input strip format
```

---

*Updated: 2026-01-30*
