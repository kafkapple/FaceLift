# MVDiffusion Improvement Roadmap

> **Created**: 2026-02-19 | **Status**: Planning | **Version**: 1.0
> **Context**: Phase 3 결론 — 학습 전략 수준 최적화 한계 도달, 아키텍처 수준 변경 필요

---

## 1. Problem Statement

### 1.1 핵심 병목: MVDiffusion

Phase 3 실험 결과:
- GS-LRM GT 6v: PSNR_gt = **21.02** (backbone 상한)
- E2E (MVDiff → GS-LRM): PSNR_gt = **8.20** (실제 파이프라인)
- **-12.82 dB gap** = MVDiffusion이 유일한 병목

| 실험 | 전략 | E2E PSNR_gt | 결론 |
|------|------|:-----------:|------|
| Baseline | ckpt-5000, constant LR | 7.93 | under-trained |
| E1 | cosine LR, 11K | 7.90 | LR 무관 |
| E2 | resume + LR decay, 20K | 8.20 | 미미한 개선 |
| E3 | pose=extrinsic, add | 8.10 | **pose conditioning 무효** |
| E5 | alpha loss 0.3 (GS-LRM) | ~8.0 | Stage 2 개선 전이 0% |

**결론**: 모든 학습 전략이 E2E PSNR 7.9-8.2로 수렴. 아키텍처 수준 변경이 필요.

### 1.2 Tier A: GS-LRM vs PS 동일 입력 조건 비교

**가장 공정한 비교 = 동일 뷰 수에서 reconstruction backbone 성능 비교**

**기존 결과 (각자 카메라 공간에서 평가 — Tier A 참고용)**:

| 모델 | Data | Input Views | PSNR_gt | IoU | 비고 |
|------|------|:-----------:|:-------:|:---:|------|
| GS-LRM GT 6v | **M5** | 6 GT | 21.02 | 0.943 | ✅ |
| GS-LRM GT 4v | **M5** | 4 GT | 20.66 | 0.926 | ✅ |
| GS-LRM GT 1v | **M5** | 1 GT | 10.47 | 0.028 | ✅ |
| PS (fj5_ds2) | **fj5_ds2** | 5 GT+1 holdout | 16.71 | 0.827 | ⚠️ 다른 카메라 공간 |

> ⚠️ 위 PS 결과는 fj5_ds2 데이터 (HFOV≈35°, fx≈810)에서 측정.
> GS-LRM 결과는 M5 데이터 (HFOV=50°, fx=549)에서 측정.
> **카메라 공간이 달라 직접 수치 비교에 제약** → 이것이 Option A의 동기.

**Option A 비교 (동일 M5 데이터 — 진정한 Tier A)**:

| 모델 | Data | Input Views | PSNR_gt | IoU | 상태 |
|------|------|:-----------:|:-------:|:---:|:----:|
| GS-LRM GT 6v | M5 | 6 GT | 21.02 | 0.943 | ✅ 완료 |
| **GS-LRM GT 5v** | M5 | **5 GT (0-4)** | **?** | **?** | **⭐ 미실행 — PS와 동일 조건** |
| GS-LRM GT 4v | M5 | 4 GT | 20.66 | 0.926 | ✅ 완료 |
| **PS (M5)** | **M5** | **5 GT (0-4)** | **?** | **?** | **⭐ 학습중 (m5_baseline_gs)** |
| FL E2E | M5 | 1 이미지 | 8.20 | 0.521 | ✅ 완료 |

**핵심 누락**: GS-LRM 5v GT 결과가 아직 없음.
- Checkpoint 존재: `base_uniform_v2_5view_v2/best_psnr.pt`
- PS가 5개 뷰(0-4)로 reconstruction → GS-LRM도 동일 5개 뷰로 비교해야 **가장 강력한 증거**
- GS-LRM 4v(20.66) >> PS 5v(16.71)에서 이미 우위이지만, 5v vs 5v가 가장 설득력 있음
- **Action**: gpu03에서 GS-LRM 5v GT 추론 + fair 평가 실행 필요

**비교 구조 (Option A 완료 시)**:

```
추론(Inference):                          평가(Evaluation):
───────────────                          ──────────────
PS:         5 GT뷰(0-4) → render 6뷰    ┐
GS-LRM 5v: 5 GT뷰(0-4) → render 6뷰    ├→ M5 GT 6뷰 대비 (동일 메트릭)
GS-LRM 4v: 4 GT뷰     → render 6뷰     │  psnr_gt_masked, IoU, coverage
GS-LRM 6v: 6 GT뷰     → render 6뷰     │
FL E2E:    1 이미지→MVDiff→GS-LRM        ┘
```

### 1.3 비균등 카메라 배치 문제

M5 카메라 6대의 azimuth 각도:

```
View 0: 0°      ─┐
View 1: 22.5°    │ 22.5° gap
View 2: 36°      │ 13.5° gap  (매우 가까움)
View 3: 73°      │ 37° gap
View 4: 88°      │ 15° gap    (매우 가까움)
View 5: 151°     │ 63° gap
[gap]             │ 208.6° gap (카메라 없는 영역)
View 0: 360°   ─┘
```

MVDiffusion (Era3D 기반)의 가정:
- 6개 뷰가 **60° 균등 간격** (Objaverse pretrained)
- Row-wise Multi-head Attention (RMA)이 이 균등 관계를 구조적으로 인코딩
- View index embedding이 이산적 위치 의미를 학습

**비균등 카메라 → 208.6° gap 영역의 생성 품질 저하** (per-view 분석에서 확인)

### 1.4 E3 Pose Conditioning 실험 (Phase 3) — 중요 실험 기록

#### 실험 설정

| 항목 | 값 |
|------|-----|
| **실험 ID** | E3 |
| **전략** | pose=extrinsic, integration=add |
| **방법** | 카메라 외부 파라미터 (3×3 rotation + 3 translation)를 MVDiff conditioning에 additive signal로 추가 |
| **Base** | ckpt-5000 (baseline MVDiff checkpoint) |
| **학습** | 10K steps, GPU 6 |
| **기대** | 카메라 위치 정보 제공 → 비균등 배치 인식 → 생성 품질 개선 |

#### 결과

| Metric | Baseline (ckpt-5000) | E2 (resume 20K) | **E3 (pose 10K)** | 변화 |
|--------|:-------------------:|:---------------:|:-----------------:|:----:|
| PSNR_gt | 7.93 | 8.20 | **8.10** | noise margin |
| IoU | 0.474 | 0.521 | **0.523** | noise margin |
| PSNR_int | 15.63 | 15.63 | **15.88** | +0.25 (미미) |
| Coverage | 69.3% | 71.5% | **71.6%** | noise margin |

**결론**: E3 ≈ E2 ≈ Baseline. Pose conditioning **무효** (F4).

#### 실패 분석

**E3가 시도한 것** (얕은 수준):
```
기존 MVDiff:  view_embed = learned_embedding[view_idx]
E3 추가:      pose_signal = MLP(flatten(R, T))
              conditioning = view_embed + pose_signal  # additive
```

**E3가 바꾸지 못한 것** (구조적 가정):
```
Row-wise Multi-head Attention (RMA):
  - Row = view index (0-5), 고정 6행 구조
  - Attention weight가 "view_0↔view_1은 이렇게 관계" 를 학습
  - 이 학습된 관계는 Objaverse 균등 60° 간격 기반
  - Additive pose signal이 이 관계를 override하지 못함
```

**핵심 교훈**:
1. 단순 signal 추가 (additive/concatenation)로는 **attention 구조에 내재된 균등 간격 가정**을 극복 불가
2. **attention 메커니즘 자체의 수정** (pose-aware attention bias, epipolar attention 등)이 필요
3. 이것은 "pose 정보가 불필요하다"는 뜻이 아니라, **정보 전달 방식이 부적절**했다는 뜻
4. 향후 Camera Pose Conditioning (§2.1)은 attention score에 직접 개입하는 방식으로 설계해야 함

#### E3 vs 제안하는 Camera Pose Conditioning의 차이

| | E3 (시도됨, 실패) | Camera Pose Conditioning (제안) |
|--|:----:|:----:|
| **정보** | 카메라 extrinsic (R, T) | 동일 |
| **통합 방식** | Additive signal to conditioning | Attention bias / Relative pose encoding |
| **수정 범위** | Conditioning MLP만 | **Cross-view attention 구조** |
| **RMA 가정 유지** | 유지 (6행 고정) | **제거** (pose-dependent) |
| **Pretrained 호환** | 호환 (추가만) | 비호환 (구조 변경) |
| **기대 효과** | 무효 (실험 확인) | 높음 (이론적) |

---

## 2. Proposed Approaches

### 2.1 Camera Pose Conditioning (아키텍처 변경)

**전략**: 모델을 데이터에 맞춤 — MVDiff가 임의 카메라 배치에서 작동하도록 수정

#### 2.1.1 핵심 아이디어

```python
# 현재: 이산적 view index embedding (Era3D)
# RMA attention에서 row = view index (0-5)
view_embed = learned_embedding[view_idx]  # shape: [6, dim]
# → 6개 고정 위치만 지원, 균등 간격 가정

# 제안: 연속적 camera pose encoding
camera_R, camera_T = get_extrinsics(azimuth, elevation, distance)
pose_embed = camera_pose_encoder(camera_R, camera_T)  # 임의 각도 지원
# → 어떤 카메라 배치에서도 작동
```

#### 2.1.2 구체적 구현 방안

**A. Pose-Aware Cross-View Attention (최소 수정)**
```
현재 RMA:
  Q, K, V = linear(features_per_view)
  attention = softmax(Q @ K^T / sqrt(d))  # 뷰 간 관계는 학습된 weight에 내재

제안:
  relative_pose = compute_relative_pose(cam_i, cam_j)
  pose_bias = pose_mlp(relative_pose)  # [N_views, N_views, N_heads]
  attention = softmax(Q @ K^T / sqrt(d) + pose_bias)  # 명시적 기하학 반영
```

**B. Sinusoidal Camera Encoding (NeRF 스타일)**
```python
def camera_pose_encoder(R, T, L=10):
    """NeRF-style positional encoding for camera parameters."""
    # Flatten R (3x3) + T (3,) → 12-dim vector
    pose_flat = torch.cat([R.flatten(), T], dim=-1)  # [12]
    # Sinusoidal encoding
    freqs = 2**torch.arange(L) * pi
    encoded = torch.cat([torch.sin(freqs * p) for p in pose_flat] +
                        [torch.cos(freqs * p) for p in pose_flat])
    return mlp(encoded)  # → embedding dim
```

**C. 전체 아키텍처 교체 (Epipolar Attention)**
- RMA 대신 epipolar line 기반 attention 사용
- 뷰 간 기하학적 관계를 명시적으로 활용
- 가장 근본적이지만 가장 많은 코드 변경 필요

#### 2.1.3 장단점

| 장점 | 단점 |
|------|------|
| 임의 카메라 배치에서 작동 | 아키텍처 수정 필요 |
| 근본적 해결 | Objaverse pretrained weight 활용 어려움 |
| 새로운 데이터셋에 일반화 | 상당한 재학습 필요 |
| 논문 contribution 높음 | 구현 복잡도 높음 |

#### 2.1.4 관련 선행 연구

- **Zero123++**: relative camera transformation을 conditioning으로 사용
- **SV3D**: camera trajectory를 연속적으로 conditioning
- **MVDream**: multi-view aware diffusion with explicit camera control
- **Wonder3D**: cross-domain attention with camera-aware features

---

### 2.2 Virtual Camera Interpolation (데이터 변경)

**전략**: 데이터를 모델에 맞춤 — 균등 간격 가상 뷰 데이터로 MVDiff 재학습

#### 2.2.1 핵심 아이디어

```
Step 1: GS-LRM GT 6v → 3D Gaussian Splatting 재구성
  Input:  M5 비균등 6뷰 GT 이미지
  Output: 3D Gaussian 표현

Step 2: 균등 간격 가상 카메라에서 렌더링
  Original: [0°, 22.5°, 36°, 73°, 88°, 151°]  (비균등, 208.6° gap)
  Virtual:  [0°, 60°, 120°, 180°, 240°, 300°]  (균등, Era3D 가정)

Step 3: 균등 뷰 데이터로 MVDiffusion 재학습
  - 기존 Era3D 아키텍처 그대로 사용
  - Pretrained weight 활용 가능

Step 4: E2E 파이프라인 재구성
  - MVDiff: 1 이미지 → 6 균등 뷰 생성
  - GS-LRM: 6 균등 뷰 → 3D 재구성 (이것도 재학습 필요)
```

#### 2.2.2 구체적 구현 방안

**A. GS-LRM 기반 가상 뷰 생성**
```bash
# Step 1: GS-LRM으로 3D 재구성 (이미 있는 코드)
python run_e2e_inference.py \
    --gslrm_checkpoint best_psnr.pt \
    --split train+val \
    --output_dir outputs/virtual_views

# Step 2: 균등 카메라에서 렌더링 (신규 스크립트 필요)
python render_virtual_views.py \
    --gaussian_dir outputs/virtual_views \
    --azimuths 0 60 120 180 240 300 \
    --output_dir data/virtual_uniform_views

# Step 3: MVDiffusion 재학습
python train_mvdiffusion.py \
    --data_dir data/virtual_uniform_views \
    --pretrained_model_path checkpoint-5000
```

**B. 하이브리드 접근: 원본 + 가상 뷰 혼합 학습**
```
Training data = Original 6 views + Virtual 6 views (augmentation)
- 원본 비균등 뷰: 실제 이미지 품질 보존
- 가상 균등 뷰: 아키텍처 가정과 일치하는 데이터 추가
- Curriculum: 먼저 가상 뷰 → 점진적으로 원본 뷰 비율 증가
```

#### 2.2.3 장단점

| 장점 | 단점 |
|------|------|
| MVDiffusion 아키텍처 변경 불필요 | GS-LRM 렌더링 아티팩트 포함 |
| Pretrained weight 활용 가능 | 208.6° gap 영역 렌더링 품질 불확실 |
| 구현 상대적 용이 | GS-LRM도 균등 뷰용 재학습 필요 |
| 단기 실험 가능 | 닭과 달걀: 좋은 3D가 먼저 필요 |

#### 2.2.4 핵심 제약

**GS-LRM GT 6v의 208.6° gap 영역 렌더링 품질**이 관건:
- GS-LRM GT 6v 전체 PSNR = 21.02
- Per-view: 입력 뷰 근접 ~24 dB, 먼 뷰 ~16.6 dB
- 208.6° gap 중앙(~256°)의 가상 뷰는 **모든 입력 뷰에서 멀리** 있음
- 이 영역 렌더링 품질이 낮으면 MVDiff가 잘못된 데이터로 학습

---

## 3. Comparison Matrix

| 기준 | Pose Conditioning | Virtual Camera | E3 (시도됨) | 현재 (Phase 3) |
|------|:-----------------:|:--------------:|:----------:|:--------------:|
| 아키텍처 변경 | **대규모** | 없음 | 없음 | 없음 |
| Pretrained 활용 | 제한적 | **가능** | 가능 | 가능 |
| 구현 난이도 | 높 | **중** | 낮 | - |
| 기대 효과 | **높** | 중 | 없음 | 없음 |
| 일반화 (새 데이터) | **높** | 낮 | 없음 | 없음 |
| 논문 contribution | **높** | 중 | - | - |
| 소요 시간 | 수 주 | **수 일** | 완료 | 완료 |
| E3와의 차이 | attention 구조 수정 | 데이터 변환 | signal 추가만 | - |

---

## 4. Experimental Evidence

### 4.1 Phase 3 결과 (이미 수행)

| Finding | 의미 |
|---------|------|
| **F3**: E1-E5 모두 PSNR 7.9-8.2 | 학습 전략 포화 |
| **F4**: E3 pose ≈ baseline | 얕은 pose conditioning 무효 |
| **F6**: Shape > Color bottleneck | 실루엣 예측이 핵심 |
| **F2**: MVDiff = -12.82 dB | 유일한 병목 확정 |

### 4.2 Per-View Analysis (view proximity bias)

```
View    Azimuth    fg_PSNR    Gap from nearest input
─────   ────────   ────────   ──────────────────────
View 0  0° (ref)   9.39       0° (입력 자체)
View 1  22.5°      8.47       22.5° (View 0에서)
View 2  36°        8.01       13.5° (View 1에서)
View 3  73°        7.14       37° (View 2에서)
View 4  88°        7.58       15° (View 3에서)
View 5  151°       6.77       63° (View 4에서) ← 최저, 208.6° gap 시작
```

**시사점**: angular distance와 생성 품질이 강하게 상관. 208.6° gap 영역은 더 낮을 것으로 예상.

### 4.3 A2 Oracle MVDiff 실험

GS-LRM 4v model에서 MVDiff 생성 뷰를 GT로 교체:
- v4, v5 GT 교체 → 영향 0 (GS-LRM이 이 뷰 미사용)
- v3 GT 교체 → **-5.2 dB** 급락
- v2 GT 교체 → -5.4 dB
- v1 GT 교체 → -2.5 dB

**시사점**: GS-LRM이 실제 사용하는 뷰(v0-v3)의 품질이 결정적. 특히 v2, v3 (36°, 73°)의 MVDiff 생성 품질이 E2E 성능의 핵심.

---

## 5. Recommended Strategy

### 5.1 현재 (단기): Option A 완료

**목표**: Fair Tier B comparison 확보
- PS M5 학습 진행 중 (m5_baseline_gs, 50 epochs)
- 완료 시 fair_comparison.py로 동일 조건 평가
- 결과로 "GS-LRM >> PS" (Tier A) + "E2E vs PS" (Tier B, 입력 비대칭 주석 포함) 확정

### 5.2 중기: Virtual Camera Interpolation (실험)

**목표**: MVDiffusion 생성 품질 개선 가능성 탐색
1. GS-LRM GT 6v로 균등 뷰 렌더링
2. 균등 뷰 데이터로 MVDiffusion fine-tuning
3. E2E 재평가

**실현성**: 높 (기존 코드 활용, 아키텍처 변경 없음)
**기대**: MVDiff 생성 뷰 품질 개선 → E2E PSNR 향상 (정도 미지수)

### 5.3 장기: Camera Pose Conditioning (연구)

**목표**: 비균등 카메라에서의 근본적 해결
1. Pose-aware attention 구현
2. 대규모 재학습
3. 새로운 데이터셋 일반화 검증

**실현성**: 중 (상당한 개발 + 학습 비용)
**기대**: 비균등 카메라 문제의 근본적 해결

### 5.4 논문 전략

```
Contribution 1: GS-LRM >> PS (Tier A, +4.31 dB)
  → Feed-forward 3D reconstruction backbone의 우수성

Contribution 2: MVDiffusion = 유일한 병목 (-12.82 dB)
  → E1-E5 학습 전략 포화 실험으로 증명
  → E3 pose conditioning 실패 분석

Contribution 3: Camera mismatch 발견 + Option A 방법론
  → 크로스-모델 비교 시 카메라 공간 정렬 필요성

Future Work: Camera Pose Conditioning + Virtual Camera Interpolation
  → 비균등 카메라 일반화를 위한 아키텍처 방향 제시
```

---

## 6. File Locations

| 파일 | 서버 | 역할 |
|------|------|------|
| `260216_PHASE3_REPORT.md` | gpu03: docs/experiments/ | Phase 3 종합 보고서 |
| `FL_vs_PS_comparison.md` | gpu03: docs/experiments/ | Tier A/B/C + Option A (v9.1) |
| `FaceLift_training_optimal_settings.md` | gpu03: docs/experiments/ | 학습 최적 설정 |
| `MVDiff_improvement_roadmap.md` | gpu03: docs/experiments/ | 본 문서 (신규) |
| `m5_baseline_gs.json` | joon: configs/experiments/ | PS M5 학습 config |

---

## 7. Key Definitions

| 용어 | 의미 |
|------|------|
| **RMA** | Row-wise Multi-head Attention (Era3D) — 뷰 간 고정 위치 관계 가정 |
| **E3** | Phase 3 실험: extrinsic additive pose conditioning → 무효 |
| **Option A** | M5 데이터로 PS 재학습 → 동일 카메라 공간 Tier B 비교 |
| **208.6° gap** | M5 카메라 View 5 (151°) → View 0 (360°) 사이 카메라 없는 영역 |
| **Tier A** | GS-LRM GT vs PS — 동일 입력 조건 정량 비교 |
| **Tier B** | E2E vs PS — 다른 입력 조건, 실제 파이프라인 비교 |

---

*MVDiffusion Improvement Roadmap v1.0 | 2026-02-19*
