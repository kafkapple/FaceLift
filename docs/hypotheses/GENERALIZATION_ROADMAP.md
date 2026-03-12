# Generalization Roadmap: Camera & Subject

> **목적**: 현재 파이프라인의 일반화 한계 분석 + 단계별 확장 로드맵
> **상태**: 📋 계획 | **Updated**: 2026-02-13
> **이론 배경**: → [MULTIVIEW_DIFFUSION_THEORY](../theory/MULTIVIEW_DIFFUSION_THEORY.md)
> ← [hypothesis_roadmap (SSOT)](../experiments/hypothesis_roadmap.md)

---

## 1. 현재 파이프라인 제약

### 1.1 입출력 흐름

```
현재:
  입력 이미지 (cam_0 고정)
      ↓
  [MV-Diffusion]  ← CLIP text("top-front view") 로만 뷰 방향 인식
      ↓               카메라 파라미터 입력 없음
  6뷰 출력 (M5 rig 고정: cam_000~005)
      ↓
  [GS-LRM]  ← c2w + intrinsics (M5 rig 파라미터) 필수 입력
      ↓
  3D Gaussian
```

### 1.2 제약 요약

| 제약 | 원인 | 영향 |
|------|------|------|
| **입력 뷰 고정** (cam_0) | `reference_view_idx: 0` | 다른 방향 입력 시 품질 저하 |
| **출력 뷰 고정** (M5 6방향) | 학습 데이터 = M5 rig | 다른 카메라 배치 출력 불가 |
| **피사체 고정** (1마리 생쥐) | 학습 데이터 = 1개체 3,600장 | 다른 생쥐/종 입력 시 품질 저하 |
| **카메라 정보 미사용** | CLIP text만으로 뷰 지정 | 연속적 카메라 각도 표현 불가 |
| **카메라 분포 편향** | M5 = 위에서 아래 (10°~31°) | 수평/하방 시점 학습 데이터 없음 |

### 1.3 M5 카메라 분포 시각화

```
Elevation (°)
35 ┤
30 ┤ ─── cam_5 (30.8°)
25 ┤ ─── cam_4 (26.5°)
20 ┤ ─── cam_1 (20.6°)
15 ┤ ─── cam_0 (14.9°) ← 현재 고정 입력
10 ┤ ─── cam_2 (11.3°), cam_3 (10.7°)
 5 ┤
 0 ┤ ─── 수평 (eye-level) ← 학습 데이터 없음
-5 ┤ ─── 아래에서 위로   ← 학습 데이터 없음
   └──────────────────────────
```

---

## 2. 시나리오별 가능 여부

### 2.1 현재 모델 (변경 없음)

| 입력 | 동작 여부 | 이유 |
|------|:-:|------|
| M5 rig cam_0 이미지 | **동작** | 학습 분포 내 |
| M5 rig 다른 뷰 (cam_3 등) | 품질 저하 | 모델이 cam_0만 봄 |
| 다른 각도 (정면) | 실패 | 학습 분포 밖 |
| 다른 생쥐 M5 rig | 품질 저하 | 1개체만 학습 |
| 다른 생쥐 + 다른 카메라 | 실패 | 모든 분포 밖 |

### 2.2 + Random Reference (P0)

| 입력 | 동작 여부 | 이유 |
|------|:-:|------|
| M5 rig 어느 뷰든 | **동작** | 6뷰 모두 입력으로 학습 |
| M5와 비슷한 각도 | 약간 저하 | 가까운 학습 분포 |
| 전혀 다른 각도 | 실패 | M5 6방향만 학습 |

### 2.3 + Pose Conditioning (P1)

| 입력 | 동작 여부 | 이유 |
|------|:-:|------|
| M5 rig 어느 뷰든 (카메라 정보 포함) | **동작** | 연속적 pose 표현 |
| M5 인근 각도 (카메라 정보 포함) | **합리적** | interpolation |
| 매우 다른 각도 (카메라 정보 포함) | 성능 저하 | extrapolation |
| 카메라 정보 없는 입력 | **불가** | pose conditioning = 카메라 필수 |

### 2.4 + 다종 데이터 재학습 (P2-P3)

| 입력 | 동작 여부 | 조건 |
|------|:-:|------|
| 다른 생쥐 + M5 rig | **동작** | 다종 생쥐 학습 데이터 |
| 다른 생쥐 + 다른 카메라 | **가능** | 다양한 rig 학습 데이터 + pose cond |

### 2.5 + MV-Adapter 교체 (P4)

| 입력 | 동작 여부 | 이유 |
|------|:-:|------|
| 임의 피사체 + 임의 카메라 | **동작** | 구조적으로 임의 viewpoint 지원 |

---

## 3. 단계별 로드맵

### P0: Random Reference View (즉시 실행 가능)

```yaml
# Config 변경 1줄 (코드 이미 존재)
reference_view_idx: "random"   # was: 0
```

| 항목 | 내용 |
|------|------|
| **해결** | M5 rig 내 다른 뷰 입력 지원 |
| **미해결** | M5 밖 카메라, 다른 피사체 |
| **GS-LRM 호환** | ✅ 기존 체크포인트 그대로 |
| **소요** | Config 1줄, 학습 ~12-24h |
| **Config** | `mouse_mvdiffusion_M5t2_randref_sparse.yaml` |

### P1: Spherical Pose Conditioning (1-2주)

```
입력 이미지 + 카메라 파라미터(θ, φ, d)
    ↓
[Pose Encoder]  ← Fourier + MLP → [B, N, 1024]
    ↓
[encoder_hidden_states에 concat]  ← UNet 수정 없음!
    ↓
[MV-Diffusion UNet]  ← 추가 토큰으로 자연스럽게 attend
    ↓
6뷰 출력 (M5 rig 고정)
```

| 항목 | 내용 |
|------|------|
| **해결** | 연속적 카메라 각도 표현, M5 인근 일반화 |
| **미해결** | 분포 밖 카메라 (큰 차이), 다른 피사체 |
| **GS-LRM 호환** | ✅ 출력 6뷰 위치 불변 |
| **소요** | ~100줄 코드, train_diffusion.py 수정 ~10줄 |
| **Config** | `mouse_mvdiffusion_M5t2_pose_spherical.yaml` |

> UNet 비침투적 주입 상세 → [§4. Pose Conditioning 주입 메커니즘](#4-pose-conditioning-주입-메커니즘)

### P2: 다종 생쥐 데이터 수집 + 재학습 (1개월+)

| 항목 | 내용 |
|------|------|
| **해결** | 피사체 일반화 (다른 생쥐) |
| **필요** | 추가 데이터 수집, 전처리, M5 rig 동일 |
| **GS-LRM** | 재학습 권장 (다른 체형/크기) |

### P3: 다양한 카메라 rig + 재학습 (1개월+)

| 항목 | 내용 |
|------|------|
| **해결** | 카메라 배치 일반화 |
| **필요** | pose conditioning (P1) 선행 필수 |
| **GS-LRM** | 재학습 필수 (다른 c2w) |

### P4: MV-Adapter 교체 (2-3개월)

| 항목 | 내용 |
|------|------|
| **해결** | 구조적으로 임의 viewpoint 지원 |
| **장점** | Plug-and-play, 127M만 학습, camera guider 내장 |
| **필요** | 코드 포팅, 학습 파이프라인 구축 |

---

## 4. Pose Conditioning 주입 메커니즘

### 4.1 핵심 원리: Cross-Attention Token Injection

현재 UNet의 cross-attention은 **CLIP text embedding**(prompt)을 key/value로 사용합니다:

```
원래 동작:
  Q = UNet hidden states [B*N, H*W, C]
  K = prompt_embedding   [B*N, seq_len, C]    ← "top-front view" 등
  V = prompt_embedding   [B*N, seq_len, C]

  Attention(Q, K, V) = softmax(QK^T / √d) V
```

Pose conditioning은 **추가 토큰 1개**를 prompt에 concat합니다:

```
수정 후:
  pose_embed = SphericalEncoder(azimuth, elevation, distance)  → [B*N, 1, C]
  K' = concat(prompt_embedding, pose_embed) → [B*N, seq_len+1, C]
  V' = concat(prompt_embedding, pose_embed) → [B*N, seq_len+1, C]

  Attention(Q, K', V') = softmax(QK'^T / √d) V'
```

### 4.2 왜 UNet 수정이 불필요한가

Cross-attention은 **key/value의 sequence 길이에 제약이 없습니다**:

```python
# Cross-attention의 수학적 정의
# Q: [B, S_q, d]   ← query (UNet features, 길이 S_q)
# K: [B, S_k, d]   ← key   (prompt, 길이 S_k ← 여기가 늘어남)
# V: [B, S_k, d]   ← value (prompt, 길이 S_k)
#
# QK^T: [B, S_q, S_k]  ← S_q와 S_k가 달라도 됨!
# Output: [B, S_q, d]   ← 항상 query 길이로 출력
```

따라서 `encoder_hidden_states`에 토큰을 추가해도 UNet 코드 변경 없이 동작합니다.

### 4.3 코드 수준 통합 (train_diffusion.py 수정 ~10줄)

```python
# === 추가할 코드 (model setup 후) ===
from mouse_extensions.model.pose_conditioning_integration import (
    create_pose_injector_from_config
)
pose_injector = create_pose_injector_from_config(vars(cfg))
if pose_injector is not None:
    pose_injector = pose_injector.to(accelerator.device)

# === training loop 내부 (line ~665) ===
# 기존:
#   model_output = models['unet'](noisy_latents, timesteps,
#       encoder_hidden_states=prompt_embeddings, ...)

# 수정:
if pose_injector is not None:
    prompt_embeddings = pose_injector.inject(
        prompt_embeddings,
        ref_view_idx=batch.get('ref_view_idx', 0),
        n_views=cfg.n_views,
    )
model_output = models['unet'](noisy_latents, timesteps,
    encoder_hidden_states=prompt_embeddings, ...)
```

### 4.4 시각적 요약

```
┌────────────────────────────────────────────────────────────┐
│                    UNet Cross-Attention                      │
│                                                              │
│  Q ← UNet hidden features                                   │
│                                                              │
│  K/V (before):  ["top-front", "view", ...]    seq_len=N     │
│                                                              │
│  K/V (after):   ["top-front", "view", ..., POSE]  seq_len=N+1│
│                          ↑                    ↑               │
│                     기존 prompt           추가 토큰            │
│                     (변경 없음)          (카메라 정보)          │
│                                                              │
│  Output shape:  동일 (Q 길이로 출력)                          │
│  UNet 코드:     수정 없음 ✅                                  │
└────────────────────────────────────────────────────────────┘
```

---

## 5. GS-LRM 호환성 분석

### 5.1 왜 GS-LRM 재학습이 불필요한가 (P0-P1)

```
                    MV-Diffusion               GS-LRM
                    ┌──────────┐              ┌──────────┐
입력: cam_X  ──►   │ 6뷰 생성  │ ──► 6뷰 ──►│ 3D 복원  │ ──► 3D Gaussian
+ pose embed       │          │    (M5 rig)  │          │
                   └──────────┘    ← 동일!   └──────────┘
                                                  ↑
                                        c2w + intrinsics
                                        (M5 rig, 변경 없음)
```

- Pose conditioning은 MV-Diffusion의 **입력 조건만 변경**
- **출력 6뷰의 카메라 위치는 항상 M5 rig** (학습 데이터 = M5 이미지)
- GS-LRM은 "6개 이미지 + M5 카메라 파라미터" 입력 → 동일 인터페이스
- ∴ **기존 GS-LRM 체크포인트 호환**

### 5.2 언제 GS-LRM 재학습이 필요한가

| 상황 | GS-LRM 재학습 |
|------|:-:|
| P0 Random ref (M5 rig 유지) | **불필요** |
| P1 Pose conditioning (M5 rig 유지) | **불필요** |
| P2 다종 생쥐 (M5 rig 유지) | **권장** (다른 체형) |
| P3 다른 카메라 rig | **필수** (다른 c2w) |
| P4 MV-Adapter (임의 viewpoint) | **필수** |

---

## 6. "다른 카메라 + 다른 생쥐" 시나리오 상세

### 질문: 카메라 파라미터 추정 → 입력하면 가능한가?

**이론적**: YES — pose conditioning의 목적이 바로 이것.

```
미래 파이프라인:
  새 이미지 + 추정된 카메라 파라미터
      ↓
  [Pose-Conditioned MV-Diffusion]
      ↓
  "M5 rig에서 본 것처럼" 6뷰 생성
      ↓
  [GS-LRM]  ← M5 rig 파라미터
      ↓
  3D Gaussian
```

**실제적 한계**:

1. **카메라 파라미터 추정 정확도**: 단일 이미지에서 절대 카메라 포즈 추정은 어려움
   - 해결: keypoint detection (DANNCE) → 포즈 추정, 또는 사용자가 대략적 방향 선택
2. **분포 밖 일반화**: M5 rig (top-angled)만 학습 → 정면/하방은 extrapolation
   - 해결: 다양한 카메라 rig 데이터로 재학습 (P3)
3. **피사체 일반화**: 1마리 생쥐만 학습 → 다른 개체는 품질 저하
   - 해결: 다종 생쥐 데이터 (P2)

### 결론: MV-Adapter가 필수인가?

**아닙니다.** 각 단계는 독립적으로 가치가 있습니다:

| 목표 | 최소 필요 단계 | MV-Adapter 필요? |
|------|:-:|:-:|
| M5 내 다른 뷰 입력 | **P0만** | ❌ |
| M5 인근 연속 각도 | P0 + P1 | ❌ |
| 다른 생쥐 (같은 rig) | P0 + P2 | ❌ |
| 임의 카메라 + 피사체 | P1 + P2 + P3 | ❌ (가능하지만 한계) |
| **최대 일반화** (임의 viewpoint, 고품질) | P4 | **✅ 권장** |

MV-Adapter는 **"최대 일반화"의 가장 깔끔한 해결책**이지만, P0-P3만으로도 상당한 수준의 일반화 달성 가능.

---

*Generalization Roadmap v1.0 | 2026-02-13*
