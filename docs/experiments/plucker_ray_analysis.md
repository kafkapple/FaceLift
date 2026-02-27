# Plucker Ray Conditioning: Theory, Implementation & Experiment Roadmap

> FaceLift MVDiffusion | Created: 2026-02-26 | Updated: 2026-02-27 (v1.3)

---

## 1. Why — Plucker Ray가 필요한 이유

### 1.1 Transfer Gap 문제

FaceLift의 핵심 병목은 **MVDiffusion → GS-LRM 전달 손실**이다:

| Stage | Metric | 값 |
|-------|--------|:--:|
| GS-LRM (GT input) | Val PSNR | 22.34 |
| MVDiffusion (val) | Val PSNR | ~26.5 |
| **E2E (Diff→GS-LRM)** | **PSNR_fg** | **8.85** (H3) / **9.04** (H4b best) |
| E2E (Diff→GS-LRM) | IoU | 0.57 (H3) / 0.577 (H4b best) |

Val PSNR 26.5의 diffusion이 E2E에서 8.85로 떨어지는 **~60% 전달 손실**의 주범은 **silhouette 불일치 (IoU 0.57)**이다. 즉, diffusion이 생성한 multi-view 이미지들의 기하학적 일관성이 부족하여 GS-LRM이 올바른 3D reconstruction을 하지 못한다.

### 1.2 Pose Conditioning의 역할

기본 MVDiffusion은 6개 뷰를 생성할 때 뷰 간의 **기하학적 관계**를 명시적으로 인코딩하지 않는다. 각 뷰의 공간적 위치는 학습 데이터에서 암묵적으로만 학습된다. Pose conditioning은 이 정보를 명시적으로 제공하여 **multi-view consistency**를 향상시키는 접근이다.

### 1.3 왜 Plucker Ray인가? (vs Extrinsic, Spherical)

| Method | 표현 | Granularity | E2E 기대 효과 |
|--------|------|:-----------:|:-------------:|
| **Extrinsic (9D)** | R(6D)+t(3D) | 뷰-레벨 (1 token/view) | IoU 0.57, PSNR_fg 8.85 (H3 실측) |
| **Spherical (3D)** | azimuth+elevation+distance | 뷰-레벨 (1 token/view) | 미실험 |
| **Plucker (6D×H×W)** | direction(3D)+moment(3D) per pixel | **픽셀-레벨** | **기대: IoU↑, PSNR_fg↑** |

**핵심 차이**: Extrinsic/Spherical은 "이 뷰의 카메라가 어디에 있는가"를 1개 벡터로 요약. Plucker는 "이 뷰의 모든 픽셀이 3D 공간에서 어떤 ray에 대응하는가"를 밀집하게 인코딩. **픽셀 수준 기하학 정보**를 제공하므로 diffusion이 각 픽셀의 3D 위치를 인식하고 multi-view 간 정합성을 더 잘 학습할 수 있다.

**문헌 근거**:
- **Zero123++ (2023)**: Plucker ray conditioning으로 multi-view 일관성 확보
- **SV3D (ECCV'24)**: Plucker coordinates로 카메라 궤적 conditioning, 시점 일관성 향상
- **SPAD (CVPR'24)**: Plucker projection jointly trained
- **CAT3D (NeurIPS'24 Oral)**: Raymap encoder E2E trained, 최고 성능

---

## 2. How — 구현 아키텍처

### 2.1 Plucker Coordinates 수학

3D 직선의 Plucker representation은 6D 벡터 `(d, m)`:

```
d = direction (normalized ray direction in world space)
m = moment = origin × direction (cross product)
```

각 픽셀 `(u, v)`에 대해:

```
1. Camera space ray:  d_cam = normalize([(u-cx)/fx, (v-cy)/fy, 1])
2. World space ray:   d_world = R @ d_cam  (R = camera rotation)
3. Origin:            o = camera position in world space
4. Moment:            m = o × d_world
5. Plucker:           p = [d_world, m] ∈ R^6
```

결과: `[B, 6, H, W]` 텐서 — 각 픽셀마다 6D Plucker coordinate.

### 2.2 코드 구조

```
mouse_extensions/model/
├── pose_conditioning.py              # 핵심 인코더 클래스들
│   ├── FourierEncoder                # NeRF-style positional encoding
│   ├── SphericalPoseEncoder          # Spherical (3D → embed_dim)
│   ├── ExtrinsicPoseEncoder          # Extrinsic (9D → embed_dim)
│   ├── PluckerRayEncoder             # Plucker (6D×H×W → spatial features)
│   │   ├── Conv2d(6, 128, k=1)       # 6D input → hidden
│   │   ├── SiLU()
│   │   └── Conv2d(128, 320, k=1)     # hidden → 320ch spatial output
│   └── CameraPoseConditioner         # Unified factory
│
└── pose_conditioning_integration.py  # UNet 통합 레이어
    ├── load_m5_cameras()             # M5 카메라 JSON → c2w, intrinsics
    ├── get_rotated_cameras()         # Random ref view 대응
    └── PoseConditioningInjector      # Non-invasive UNet injection
        ├── plucker_to_token          # Spatial → Token 변환
        │   ├── AdaptiveAvgPool2d(1)  # [N, 320, H, W] → [N, 320, 1, 1]
        │   ├── Flatten()             # [N, 320]
        │   └── Linear(320, 1024)     # [N, 1024] (= embed_dim)
        └── inject()                  # Prompt에 pose token 주입
```

### 2.3 데이터 흐름 (H6a_v2 기준)

```
M5 Camera JSON ──────────────────────────────────────┐
 (c2w: [6, 4, 4], intrinsics: [6, 4])               │
                                                      ▼
 ref_view_idx ──→ get_rotated_cameras() ──→ c2w_rotated [6, 4, 4]
                                                      │
                                                      ▼
                 compute_plucker_coordinates(c2w, intr, H=64, W=64)
                                                      │
                    ┌──── d_cam = normalize(pixel → camera ray) ──┐
                    │     d_world = R @ d_cam                     │
                    │     m = origin × d_world                    │
                    └─────────────────────────────────────────────┘
                                                      │
                                                      ▼
                              Plucker [6×6, 6, 64, 64]  (B*N, 6, H, W)
                                                      │
                                                      ▼
                              PluckerRayEncoder (Conv 6→128→320)
                                                      │
                                                      ▼
                              Spatial features [6×6, 320, 64, 64]
                                                      │
                                     plucker_to_token │
                                                      ▼
                              AdaptiveAvgPool2d(1) → [36, 320]
                              Linear(320, 1024)    → [36, 1024]
                                                      │
                              ┌────────────────────────┘
                              │  integration="add"
                              ▼
               prompt_embeddings[:, 0, :] += pose_token
                              │
                              ▼
                         UNet forward (cross-attention sees pose-aware prompt)
```

### 2.4 Trainable Mode (H6a_v2 핵심 변경)

H6a_v2의 가장 중요한 설계 결정은 **pose encoder를 UNet과 함께 학습**하는 것이다:

| 항목 | H6a (Frozen) | H6a_v2 (Trainable) |
|------|:---:|:---:|
| Encoder gradients | `torch.no_grad()` | 활성 |
| Embedding cache | 사용 | 사용 안 함 (매번 재계산) |
| Optimizer | UNet only | UNet + Pose encoder (별도 AdamW) |
| Param dtype | fp16 (weight_dtype) | **fp32** (GradScaler 호환) |
| 학습 가능 파라미터 | 0 | **~371K** |

```python
# train_diffusion.py에서의 구현
if trainable_pose:
    pose_injector = pose_injector.to(device)  # fp32 유지
    pose_optimizer = torch.optim.AdamW(
        pose_injector.parameters(),
        lr=5e-5, betas=(0.9, 0.999), weight_decay=0.01
    )
```

**문헌 합의**: MVDream, CAT3D, SPAD, SV3D 모두 pose projection layer를 jointly training. Random initialization의 frozen projection보다 학습된 representation이 항상 우수.

### 2.5 Integration Method: "add"

현재 `integration="add"` 방식 사용:

```python
# prompt_embeddings shape: [B*N, seq_len, 1024]
# pose_token shape:        [B*N, 1024]
prompt_embeddings[:, 0, :] += pose_token  # 첫 번째 토큰에 가산
```

**대안**: `"concat"` (seq_len+1로 확장), `"replace_last"` (마지막 토큰 대체). 현재 `"add"`가 H3에서 검증된 안정적 방법.

---

## 3. What — 현재 결과와 분석

### 3.1 Val PSNR 비교 (MVDiffusion)

| 실험 | Method | Best Val PSNR | Best Step | 상태 |
|------|--------|:----:|:---:|------|
| E2 baseline | No pose | 26.82 | 15K | 완료 |
| H3 | Extrinsic (9D, add) | 26.55 | 7K | 완료 |
| H4 Extended | No pose, resume H3→20K | **26.94** | 9.8K | 완료 |
| H5 cfg010 | No pose, CFG=0.10 | 26.58 | 9.8K | 완료 |
| H4b Extended | No pose, resume H3→10K | 26.12 | 4K | 학습중 |
| **H6a_v2** | **Plucker (trainable, add)** | **27.34** | **9K** | **완료** |
| **H7** | **Plucker (spatial_token, trainable)** | **25.57+** | **1K+** | **학습중** |

**H6a_v2 Plucker가 전 실험 최고 val PSNR 27.34 달성 (step 9000, cfg3.0).**

### 3.2 Val PSNR 수렴 곡선 (H6a_v2)

```
Step   Val PSNR
0001    6.58
0200   17.19    ← 매우 빠른 초기 수렴 (Plucker 기하학 정보 효과)
0400   23.00
0600   24.67
0800   24.22
1200   25.72
2800   26.58    ← 첫 peak
4000   26.65
5400   26.76
7000   27.02
9000   27.34    ← 최종 best (cfg3.0) ⭐
```

- **주목할 점**: Step 200에서 이미 17.19. H4b는 step 200에서 6.62. → Plucker의 풍부한 기하학 정보가 초기 학습을 극적으로 가속.
- Step 9000에서 27.34 달성으로 수렴 확인.

### 3.3 H3 (Extrinsic) E2E 결과 (참고 baseline)

| Metric | 값 |
|--------|:--:|
| PSNR_fg | 8.85 |
| IoU | 0.569 |
| PSNR_wh | 21.95 |
| LPIPS | 0.055 |

→ H6a_v2 E2E에서 **IoU 개선이 핵심 검증 포인트**. Val PSNR +0.47은 작지만 Plucker의 **픽셀-레벨 기하학 정보**가 silhouette 일관성을 높여 E2E IoU를 개선할 수 있는지가 진짜 질문.

---

## 4. Experiment Results Summary

### 4.1 H6a_v2 (Plucker + Add, Trainable) — Completed
- **Config**: method=plucker, integration=add, trainable=true
- **Best Val PSNR**: 27.34 @ step 9000 (cfg3.0)
- **Val PSNR progression (cfg3.0)**: step 200: 23.49, 400: 24.16, 600: 24.29, 800: 24.89, 1000: 25.06, → best 27.34 @ 9000
- **E2E Results (no pose injection)**: PSNR_fg=8.952, IoU=0.574
  - Comparison: H3 baseline PSNR_fg=8.854, IoU=0.569
  - Delta: +0.098 PSNR_fg, +0.005 IoU — marginal improvement
  - Transfer gap ~67% persists
- **E2E with pose injection (fresh encoder)**: PSNR_fg=8.94, IoU=0.573
  - Confirmed: fresh (untrained) encoder = no improvement (as expected)
  - NOTE: Trained pose encoder weights were NOT saved in checkpoints (bug fixed for H7+)

### 4.2 H7 (Plucker + Spatial Token, Trainable) — Training
- **Config**: method=plucker, integration=spatial_token, trainable=true, spatial_token_size=8
- **Architecture**: Plucker [320,64,64] → Pool(8x8) → 64 tokens → concat to prompt, PLUS global token add to prompt[0]
- **Zero-init**: plucker_spatial_linear initialized with zeros (ControlNet strategy)
- **Prompt sequence**: 77 → 141 tokens (77 CLIP + 64 spatial)
- **Val PSNR progression (cfg3.0)**:
  - Step 1: 6.63 (baseline before training)
  - Step 200: 21.63, 400: 24.93, 600: 25.51, 800: 25.15, 1000: 25.57
  - Step 2000: 24.89, 3000: 25.81, 3200: 25.77, 3400: 26.09, 3600: 25.45
- **vs H6a_v2 at same steps**: H7 leads by +0.26 to +1.22 dB from step 400 onwards
- **Status**: Training on GPU 7, step ~3600/10K (36%)
- **Inference**: Pose encoder weights now saved with checkpoints (bug fixed)

### 4.3 Early Comparison: H7 vs H6a_v2

| Step | H6a_v2 (cfg3.0) | H7 (cfg3.0) | Delta |
|------|-----------------|-------------|-------|
| 200 | 23.49 | 21.63 | -1.86 |
| 400 | 24.16 | 24.93 | +0.77 |
| 600 | 24.29 | 25.51 | +1.22 |
| 800 | 24.89 | 25.15 | +0.26 |
| 1000 | 25.06 | 25.57 | +0.51 |

Key observation: H7 starts slow (zero-init warm-up) but surpasses H6a_v2 from step 400.
The spatial token approach preserves spatial information from Plucker rays that was lost in H6a_v2's global average pooling.

### 4.4 H4b Extended (No Pose, Extended Training) — Completed
- **Config**: Same as H3 but extended to 10K steps with LR=1e-5
- **Best Val PSNR**: 26.24 @ step 4600 (cfg3.0), but checkpoint pruned (limit=3)
- **Available best**: checkpoint-8000, val PSNR 25.79 (cfg3.0)
- **E2E Results**: PSNR_fg=9.04, IoU=0.577
  - **Best E2E result so far** — better than H6a_v2 despite lower val PSNR

### 4.5 Comprehensive E2E Comparison

| Experiment | Val PSNR (best) | PSNR_fg | PSNR_wh | IoU | vs H3 |
|------------|:---------------:|:-------:|:-------:|:---:|:-----:|
| H3 (Extrinsic, baseline) | ~26 | 8.854 | — | 0.569 | — |
| **H4b** (Extended 10K) | 26.24 | **9.04** | 22.09 | **0.577** | **+0.19** |
| H6a_v2 (Plucker+Add) | **27.34** | 8.952 | 22.04 | 0.574 | +0.10 |
| H6a_v2 (w/ pose, fresh) | — | 8.94 | 22.04 | 0.573 | +0.09 |
| H7 (Spatial Token) | TBD | TBD | TBD | TBD | TBD |

**Key finding**: Val PSNR does NOT predict E2E performance. H4b has lower val but better E2E than H6a_v2.
Possible explanation: extended training improves consistency/stability more than pose conditioning.

### 4.6 Cross-Model Comparison (FL vs PS)

| Model | Type | PSNR_fg (ALL) | PSNR_fg (HO) | IoU (ALL) |
|-------|------|:-------------:|:-------------:|:---------:|
| FL H4b (best) | Feed-forward | 9.04 | — | 0.577 |
| PS 6v | Per-scene | **13.78** | 13.16 | **0.846** |
| PS 5v (holdout 4,5) | Per-scene | **13.92** | 13.20 | **0.849** |

FL-PS gap: ~4.9 dB PSNR_fg, ~0.27 IoU. Per-scene optimization still dominates.

---

## 5. Bug Fix: Pose Encoder Weight Persistence

**Problem**: train_diffusion.py did not save pose_injector.state_dict() at checkpoints.
- H6a_v2's trained pose encoder (371K params) was lost after training
- Inference pipeline had no pose injection support

**Fix (2026-02-27)**:
1. train_diffusion.py: Added torch.save(pose_injector.state_dict(), ...) at checkpoint save and final save
2. mvdiffusion_pipeline.py: Added pose_config, pose_weights_path params + injection in generate_views()
3. end_to_end.py: Pass-through for pose params
4. run_e2e_inference.py: --pose_config_yaml and --pose_weights CLI args

**Impact**: H7 onwards will have trained pose weights saved for proper inference evaluation.

---

## 6. Inference Pipeline Pose Injection

Added support for pose conditioning at inference time:
- `run_e2e_inference.py --pose_config_yaml <yaml> --pose_weights <path>`
- Pipeline creates PoseConditioningInjector from config, loads weights if available
- Injection happens before diffusion pipe() call in generate_views()
- Supports both "add" and "spatial_token" integration methods

---

## 7. 후속 실험 로드맵

### 7.1 즉시 실행 가능 (GPU 5, 6 idle)

#### EXP-A: H6a_v2 E2E Eval (최우선)
- **목적**: Plucker conditioning이 실제 Transfer gap을 줄이는지 검증
- **방법**: H6a_v2 best checkpoint → MVDiff inference → GS-LRM → fair eval
- **성공 기준**: IoU > 0.60 (H3: 0.569), PSNR_fg > 9.0 (H3: 8.85)
- **GPU**: 5 or 6 (학습 완료 후 즉시)

#### EXP-B: Plucker Spatial Injection (concat latent)
- **가설**: 현재 `add` integration은 spatial → global avg pool → token으로 공간 정보를 **압축**함. Plucker의 핵심 강점인 **픽셀-레벨 공간 정보를 보존**하면 더 효과적일 수 있다.
- **방법**: `PluckerRayEncoder` 출력 `[B*N, 320, 64, 64]`을 UNet input latent에 직접 concatenate (channel-wise)
- **구현 변경**:
  ```python
  # 현재: spatial → avg_pool → token → add to prompt
  # 제안: spatial → resize to latent size → concat to latent_model_input
  plucker_spatial = encoder(plucker)  # [BN, 320, 64, 64]
  # UNet input latent은 [BN, 8, 64, 64] (4ch noisy + 4ch cond)
  # → plucker_spatial를 Conv2d(320, 4, 1)로 압축 후 concat → [BN, 12, 64, 64]
  ```
- **리스크**: UNet conv_in 수정 필요 (zero-init 추가 channels). 아키텍처 변경.
- **GPU**: 5 or 6

#### EXP-C: Plucker + Higher Resolution (128×128)
- **가설**: 현재 `plucker_resolution=64`로 ray를 계산. 이는 latent space 크기와 동일하나, UNet 내부의 더 높은 해상도 feature에도 정보를 전달하면 세밀한 기하학 학습 가능.
- **방법**: `plucker_resolution: 128`, plucker_to_token은 동일 (avg pool)
- **리스크**: 메모리 증가 (64→128이면 4배), A6000 48GB 내 가능 여부 확인 필요
- **GPU**: 5 or 6

### 7.2 중기 실험 (학습 완료 후)

#### EXP-D: Plucker + Multi-Scale Injection
- **가설**: CAT3D의 raymap encoder 방식 — Plucker features를 UNet의 **각 해상도 단계**에 주입
- **방법**:
  - 64×64 spatial features → UNet down blocks의 각 해상도에 resize + conv 후 주입
  - Cross-attention이 아닌 **spatial addition**으로 direct injection
- **문헌**: CAT3D가 이 방식으로 SOTA 달성

#### EXP-E: Integration Method Ablation
- **현재**: `"add"` (첫 토큰에 가산)
- **비교**: `"concat"` (sequence 확장), `"replace_last"` (마지막 토큰 대체)
- **목적**: Token-level integration 최적 방법 탐색

#### EXP-F: Plucker + Extrinsic Hybrid
- **가설**: Plucker (픽셀-레벨)과 Extrinsic (뷰-레벨)을 동시 사용하면 상호보완
- **방법**: 두 encoder 출력을 concat하여 prompt에 주입
- **근거**: Multi-scale pose information이 더 풍부한 geometric signal

### 7.3 아키텍처 레벨 변경 (장기)

#### EXP-G: Plucker as UNet Input Conditioning (Zero123++ 방식)
- **핵심**: Prompt token이 아닌, **UNet input에 Plucker map을 직접 concat**
- **구현**:
  1. Plucker [6, H, W] → Conv encoder → [C, H, W]
  2. concat with noisy_latent + cond_latent → UNet forward
  3. UNet conv_in을 확장 (zero_init_conv_in 활용)
- **장점**: 공간 정보 완전 보존. Cross-attention bottleneck 회피.
- **주의**: UNet 아키텍처 수정 → pretrained checkpoint 호환성 문제

#### EXP-H: Plucker + Silhouette Loss
- **가설**: Val PSNR 향상과 E2E IoU 향상이 직접 연결되지 않을 수 있음. Silhouette 일관성을 **직접 loss로 추가**하면 Transfer gap 해소에 더 효과적.
- **방법**: Diffusion 생성 이미지에서 alpha 추출 → GT alpha와의 IoU/BCE loss 추가
- **리스크**: Diffusion training loop에 non-standard loss 추가 → 학습 불안정

---

## 8. 우선순위 매트릭스

| 순위 | 실험 | 근거 | 필요 리소스 | 기대 영향 |
|:----:|------|------|:-----------:|:---------:|
| **1** | **EXP-A: E2E Eval** | Plucker 효과 최종 검증 | GPU 1개, ~2h | ⭐⭐⭐ |
| **2** | EXP-B: Spatial Injection | 공간 정보 보존이 핵심 | GPU 1개, ~30h | ⭐⭐⭐ |
| **3** | EXP-E: Integration Ablation | 저비용 비교 | GPU 1개, ~30h | ⭐⭐ |
| **4** | EXP-G: UNet Input Concat | Zero123++ 검증된 방법 | GPU 1개 + 코드 수정 | ⭐⭐⭐ |
| **5** | EXP-D: Multi-Scale | CAT3D SOTA 방법 | 코드 복잡 | ⭐⭐ |
| **6** | EXP-C: Resolution 128 | 메모리 확인 필요 | GPU 1개, ~30h | ⭐ |
| **7** | EXP-F: Hybrid | 조합 효과 불확실 | GPU 1개, ~30h | ⭐ |
| **8** | EXP-H: Silhouette Loss | 직접적이나 불안정 리스크 | 연구 필요 | ⭐⭐ |

---

## 9. 핵심 인사이트 요약

1. **Plucker ray의 근본적 장점**: 뷰-레벨 토큰(1D) vs 픽셀-레벨 ray map(2D spatial). 기하학 정보 밀도가 근본적으로 다름.

2. **현재 구현의 한계**: 픽셀-레벨 spatial features를 **global average pooling으로 1개 토큰으로 압축**하여 사용 중. Plucker의 공간적 장점을 상당 부분 상실. 이것이 H3 대비 val PSNR +0.47에 그친 이유일 수 있음.

3. **Val PSNR ≠ E2E 성능**: H6a_v2 (val 27.34) > H4b (val 26.24)이지만 E2E에서는 H4b (9.04) > H6a_v2 (8.95). Extended training이 multi-view consistency에 더 효과적일 수 있음. 또는 H6a_v2의 pose encoder weights 미저장 때문일 수 있음.

4. **H7 spatial token 유망**: Step 400 이후 H6a_v2를 +0.26~+1.22 dB 상회. 공간 정보 보존 가설 지지. H7은 pose weights도 저장됨 → 진정한 E2E 테스트 가능.

5. **FL vs PS gap**: FL best 9.04 vs PS 5v 13.92 = -4.88 dB. Per-scene optimization 우위 지속. IoU 격차 (0.577 vs 0.849)가 핵심 — silhouette quality가 성능 결정.

6. **Pose encoder weight persistence**: 버그 발견 및 수정. H7부터 올바르게 저장됨.

---

## 10. Spatial Token 구현 (Phase 1 — 2026-02-26)

### 10.1 구현 완료

**문제**: H6a_v2의 `AdaptiveAvgPool2d(1)`이 `[320, 64, 64]` → `[320]`으로 4096개 공간 위치를 1개로 압축.

**해결**: `spatial_token` integration method 추가.

```
Plucker [B*N, 320, 64, 64]
  ├─ AvgPool(8×8) → [B*N, 320, 8, 8]
  ├─ flatten    → [B*N, 64, 320]
  ├─ Linear(320→1024, zero-init) → [B*N, 64, 1024]  ← 64 spatial tokens
  └─ concat to prompt_embeddings

  + global token (add) = 기존 H6a_v2와 동일

Result: prompt [B*N, 77+64, 1024] = [B*N, 141, 1024]
```

### 10.2 수정 파일

| 파일 | 변경 |
|------|------|
| `mouse_extensions/model/pose_conditioning_integration.py` | `spatial_token_size` param, `_compute_spatial_tokens()`, `inject_spatial()` 추가 |
| `train_diffusion.py` | `inject()` → `inject_spatial()` 자동 dispatch |
| `configs/mvdiffusion/mouse_mvdiffusion_M5t2_H7_spatial_token.yaml` | H7 config |

### 10.3 설계 결정

- **Zero-init linear**: ControlNet 전략. 학습 초기에 spatial tokens가 0 기여 → UNet이 기존 학습 안정성 유지하면서 점진적으로 spatial 정보 활용 학습
- **8×8 = 64 tokens**: seq_len 77→141 (1.83× 증가). Cross-attention 메모리 영향 제한적
- **Dual injection**: global add (큰 그림) + spatial concat (세부 기하학) 조합

### 10.4 H7 실험

| 항목 | 값 |
|------|-----|
| Config | `mouse_mvdiffusion_M5t2_H7_spatial_token.yaml` |
| GPU | 7 (A6000) |
| Base | E2 checkpoint-20000 |
| 학습 가능 파라미터 | 699,584 |
| Wandb | `mvdiff_M5t2_H7_spatial_token` |
| 상태 | 학습중 (260226 22:12~) |

### 10.5 다음 단계 (Phase 2)

방안 1 (UNet Input Concatenation) 조사 완료:

- SD2.1-UnCLIP UNet conv_in: `in_channels=9→320`
- Era3D: `in_channels=8` (noisy 4ch + cond 4ch), `zero_init_conv_in` 로 추가 채널 zero 초기화
- **동일 패턴으로 Plucker 채널 추가 가능**: Conv2d(320→8, 1×1)로 bottleneck → concat → conv_in(17→320)
- 기존 9ch weights 보존, 신규 8ch zero-init

---

*FaceLift Experiment Document v1.2 | Updated: 2026-02-27*
