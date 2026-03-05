# Phase 4: Pose Conditioning (MVDiffusion)

> **Navigation**: [← Hub](../PIPELINE_DEEP_DIVE.md) | [Prev: PH3](PH3_MODEL_FORWARD_LOSS.md) | [Next: PH5 →](PH5_E2E_INFERENCE.md)
>
> **핵심 파일**: `mouse_extensions/model/pose_conditioning.py`, `mouse_extensions/model/pose_conditioning_integration.py`

---

> **Why**: MVDiffusion은 6개 뷰의 상대적 카메라 위치를 implicit하게 학습하지만,
> 명시적 카메라 포즈를 주입하면 multi-view consistency가 향상될 수 있다.
> 3가지 인코딩 방식(Spherical, Extrinsic, Plucker)을 구현하여 비교 실험.

### 4.1 Pose Encoding Architectures

**File**: `mouse_extensions/model/pose_conditioning.py`

```
┌─────────────────────────────────────────────────────────────────┐
│ 3 Encoder Architectures                                         │
├──────────────────┬──────────────────┬───────────────────────────┤
│ SphericalPose    │ ExtrinsicPose    │ PluckerRay                │
│ (L44-114)        │ (L117-175)       │ (L178-291)               │
├──────────────────┼──────────────────┼───────────────────────────┤
│ c2w → spherical  │ c2w → R,t direct │ c2w + K → pixel rays     │
│ (az, el, dist)   │ 6D rotation repr │ direction + moment        │
│ ↓                │ ↓                │ ↓                         │
│ Fourier encode   │ flatten + MLP    │ 2-layer Conv2d 1×1        │
│ (3×128=384D)     │ (9D or 12D)      │ (6 → 128 → 320)          │
│ ↓                │ ↓                │ ↓                         │
│ MLP → 1024D      │ MLP → 1024D      │ [B, 320, H, W] spatial   │
│                  │                  │                           │
│ ★ Global token   │ ★ Global token   │ ★ Spatial features        │
│ [B, N, 1024]     │ [B, N, 1024]     │ Per-pixel geometry        │
└──────────────────┴──────────────────┴───────────────────────────┘
```

#### PluckerRay 상세 (`pose_conditioning.py:218-291`)

```python
# compute_plucker_coordinates(c2w, intrinsics, height, width)
#
# 1. Pixel grid 생성 (L250-256)
#    meshgrid(0..H-1, 0..W-1) → [H, W, 2]
#
# 2. Camera space normalization (L264-266)
#    x = (u - cx) / fx,  y = (v - cy) / fy,  z = 1.0
#
# 3. World space ray direction (L273-277)
#    direction = einsum('bij,hwj->bhwi', R, ray_cam)  # [B, H, W, 3]
#    direction = normalize(direction)
#
# 4. Camera origin in world (L280)
#    origin = c2w[:, :3, 3]  # [B, 3]
#
# 5. Plucker moment (L283)
#    moment = cross(origin, direction)  # [B, H, W, 3]
#
# 6. Concat + rearrange (L286-289)
#    plucker = cat([direction, moment])  # [B, H, W, 6] → [B, 6, H, W]
```

### 4.2 Injection into MVDiffusion UNet

**File**: `mouse_extensions/model/pose_conditioning_integration.py`

#### PoseConditioningInjector (L113-542) — 핵심 클래스

```
┌─────────────────────────────────────────────────────────────────┐
│ Injection Modes (L114-136)                                      │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│ "concat"     │ "add"        │ "replace"    │ "spatial_token"    │
│ prompt seq에  │ 첫 token에   │ 마지막 token │ Plucker 공간 정보   │
│ extra token  │ 더하기       │ 교체         │ 64개 토큰 추가      │
│ 추가          │              │              │                    │
│ H3 실험       │ H6a 실험     │ 미사용       │ H7/H7v2 실험       │
├──────────────┴──────────────┴──────────────┴────────────────────┤
│                                                                 │
│ UNet Input:  encoder_hidden_states [B*N, seq_len, 1024]         │
│                              │                                  │
│       ┌──────────────────────┼────────────────────────┐         │
│       ▼                      ▼                        ▼         │
│   inject()              inject()               inject_spatial() │
│   (concat/replace)      (add mode)             (spatial_token)  │
│       │                      │                        │         │
│       ▼                      ▼                        ▼         │
│ [B*N, seq+1, 1024]    [B*N, seq, 1024]    [B*N, seq+64, 1024]  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Spatial Token 상세 (H7/H7v2, L264-304)

```python
# _compute_spatial_tokens() — Plucker rays → spatial token sequence
#
# Plucker spatial features: [N, 320, H, W]
#                              │
# AdaptiveAvgPool2d(S=8) → [N, 320, 8, 8]
#                              │
# flatten + transpose    → [N, 64, 320]      (S*S = 64 tokens)
#                              │
# plucker_spatial_linear  → [N, 64, 1024]     ★ Zero-init (ControlNet)
#
# Why zero-init (L204-205):
#   초기 학습 시 spatial tokens가 0을 출력 → 기존 모델 동작 보존
#   학습 진행에 따라 점진적으로 공간 정보 기여 → 안정적 fine-tuning
```

#### Trainable vs Frozen Mode (L212-217, L327-334)

```
trainable=True  + training=True   → gradient 흐름, cache 무효화
trainable=True  + training=False  → no_grad (검증/추론)
trainable=False + any             → 항상 no_grad (레거시 frozen)

★ pose_injector weights are saved with MVDiffusion checkpoint
  (Bug fix 2026-02-27: 이전에는 누락되어 추론 시 random weights 사용됨)
```

#### M5 Camera Utilities (L39-106)

```python
# load_m5_cameras(json_path) — M5 rig 정의 로드
# Output: {c2w: [6, 4, 4], w2c: [6, 4, 4], intrinsics: [6, 4], n_views: 6}

# get_rotated_cameras(c2w, ref_view_idx, n_views)
# Random reference view augmentation: 카메라 순서를 ref 기준으로 회전
# indices = [(ref + i) % n for i in range(n)]
```

### 4.3 Config & Commands

```yaml
# configs/experiments/pose_conditioning config:
pose_conditioning:
  method: "plucker"           # spherical / extrinsic / plucker
  injection_mode: "add"       # concat / add / replace_last / spatial_token
  embed_dim: 1024
  trainable: true
  # Plucker-specific:
  plucker_hidden_dim: 128
  plucker_spatial_dim: 320
  spatial_token_size: 8       # H7: 8×8 = 64 tokens
```

```bash
# MVDiffusion fine-tuning with pose conditioning
CUDA_VISIBLE_DEVICES=5 accelerate launch train_diffusion.py \
    --config configs/mvdiffusion/mouse_M5t2.yaml \
    --pose_config configs/experiments/H6a_v2_plucker_add.yaml
```

---

*← [PH3](PH3_MODEL_FORWARD_LOSS.md) | [Hub](../PIPELINE_DEEP_DIVE.md) | [Next: PH5 →](PH5_E2E_INFERENCE.md)*
