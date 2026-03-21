# Temporal Consistency Study

GS-LRM novel view temporal flickering 분석 및 smoothing 방법 비교.

---

## 1. Background

GS-LRM은 프레임별 독립적 Gaussian 세트를 생성하여, novel view 렌더링 시 temporal flickering (frame-to-frame jitter) 발생. 원 FaceLift 논문은 faces에 대해 8-layer MLP deformation network로 완화했으나, 우리 mouse 시나리오에서는:
- 큰 non-rigid motion (running, grooming, rearing)
- Per-Gaussian identity 미보장 (수/순서가 프레임마다 다름)
- Deformation checkpoint 존재하나 비교 렌더 미수행

## 2. Evaluation Standard

**SSOT**: [[TEMPORAL_EVAL_STANDARD]]

| 항목 | 값 |
|------|-----|
| **Sparse Frames** | 3240-3599 (10개, ~40간격, test split) |
| **Dense Frames** | 3300-3320 (21 연속, temporal 분석) |
| **Views** | bottom(-70°), top(+70°), front_low(-30°), side_low(-30°/90°) |
| **Experiments** | baseline_6v, α=0.3, α=0.5, α=1.0 |
| **Resolution** | 384×384 |

## 3. Phase 1 Results: 2D Post-Processing

### Methods Tested

| Method | Description | Parameters |
|--------|-------------|------------|
| **Original** | Raw GS-LRM output | — |
| **EMA** | Exponential Moving Average on rendered images | α ∈ {0.1, 0.3, 0.5} |
| **OptFlow** | Farneback optical flow warping + blending | α ∈ {0.3, 0.5} |

### tOF Results (mean across 4 views, lower = more stable)

| Method | Baseline | α=0.3 | α=0.5 | α=1.0 | Reduction |
|--------|:--------:|:-----:|:-----:|:-----:|:---------:|
| Original | 0.2052 | 0.1972 | 0.1956 | 0.1967 | — |
| **EMA α=0.1** | **0.0230** | **0.0251** | **0.0250** | **0.0250** | **~87%** |
| EMA α=0.3 | 0.0634 | 0.0700 | 0.0701 | 0.0702 | ~65% |
| EMA α=0.5 | 0.1336 | 0.1385 | 0.1387 | 0.1378 | ~30% |
| OptFlow α=0.3 | 0.1835 | 0.1802 | 0.1790 | 0.1789 | ~9% |
| OptFlow α=0.5 | 0.1927 | 0.1977 | 0.1962 | 0.1970 | ~0% |

### Key Findings

1. **EMA α=0.1이 압도적** — tOF 87% 감소. 단, motion blur 대가 있음
2. **OptFlow 비효과적** — Farneback flow warping이 flickering을 거의 줄이지 못함 (flow 추정 자체가 jitter에 영향받음)
3. **Alpha loss weight는 temporal stability에 영향 없음** — 4개 설정 모두 유사한 tOF (±3%)
4. **Bottom view가 가장 불안정** — tOF=0.31 (side_low=0.25, front_low=0.15, top=0.10)
   - Ventral surface: 다리, 배 등 dynamic 영역 노출 → 더 많은 flickering

### Per-View tOF (Original, Baseline)

| View | tOF | 특성 |
|------|:---:|------|
| bottom | 0.3143 | Ventral, 다리/배 노출 |
| side_low | 0.2540 | Profile, 꼬리/다리 |
| front_low | 0.1478 | 얼굴/머리 |
| top | 0.1049 | Dorsal, 상대적 안정 |

## 4. Phase 2: DeformationV2 (Planned)

### Module Status

| Component | Path | Status |
|-----------|------|--------|
| DeformationNetworkV2 | `mouse_extensions/model/deformation/deformation_network.py` | ✅ 구현 |
| TemporalDeformInference | `model/deformation/temporal_deform_inference.py` | ✅ 구현 |
| Checkpoint | `/node_data/joon/checkpoints/FaceLift/deformation/default/checkpoint_010000.pt` | ✅ 존재 |
| Gaussian NPZ cache | `outputs/datasets/temporal_eval/*/gaussians/` | ❌ 미생성 |

### Execution Plan

1. **Gaussian NPZ 추출**: `collect_dataset.py` 수정 → 프레임별 Gaussian params 저장
2. **NN Matching**: `cKDTree(xyz_t).query(xyz_t1)` → N 불일치 해결
3. **Deformation 적용**: `TemporalDeformInference.process_sequence()`, blend_alpha sweep
4. **재렌더링**: deformed Gaussians → novel view renders → tOF 비교

### FaceLift Appendix 3.5 — Deformation Module 분석

#### GS-LRM의 근본 구조

GS-LRM은 **per-pixel Gaussian** 생성 모델:
- 입력 이미지의 각 pixel 위치가 14-param Gaussian을 출력 (xyz, rotation, scale, opacity, SH)
- 프레임마다 **독립적**으로 Gaussian 세트 생성 → **Gaussian identity 없음**
- 수와 순서가 프레임마다 다름 → scene flow 계산의 근본적 어려움

#### 논문의 Canonical Deformation 접근

```
[Canonical Frame]
     ↓
  GS-LRM → Canonical Gaussians (고정 세트)
     ↓
[Target Frame t]
     ↓
  8-layer MLP(canonical_xyz, target_features) → Δxyz, Δopacity, Δscale
     ↓
  Deformed Gaussians = Canonical + Δ  (같은 Gaussian의 시간 변형 추적)
```

핵심: **canonical Gaussians를 기준으로 deformation을 예측**하므로 Gaussian identity가 유지됨.
- Loss: photometric (렌더 vs GT) + ARAP (local rigidity 보존)
- Faces에서 유효: 일관된 topology, 작은 motion → canonical frame 하나로 충분

#### Mouse에서의 한계 (3-model 심의 결과)

| 문제 | 설명 |
|------|------|
| **Fixed Topology** | Canonical frame에서 가려진 부위(배, 다리 안쪽)에 Gaussian이 없음 → 해당 부위 노출 시 표현 불가 |
| **Large Motion** | Running/grooming/rearing → canonical에서 너무 먼 deformation → MLP 용량 초과 |
| **Self-Occlusion** | 꼬리 감싸기, 다리 접기 → topology 변화, Gaussian birth/death 필요하나 canonical은 고정 |
| **Feed-Forward 제약** | Per-scene optimization 없이 feed-forward로 작동해야 → generalization 어려움 |

#### 대안 접근법 (문헌 조사)

| 방법 | 핵심 | Mouse 적합도 |
|------|------|:-----------:|
| **Multi-Canonical** | 5-10 keyframe의 Gaussian을 fuse → 더 넓은 coverage | ⭐⭐⭐ |
| **Sliding-Window Canonical** | 인접 프레임 canonical → large motion 대응 | ⭐⭐⭐ |
| **4DGS (per-scene optim)** | 전체 시퀀스 최적화로 canonical + deform 동시 학습 | ⭐⭐ (feed-forward 아님) |
| **SC-GS** | Sparse control points + implicit deformation | ⭐⭐ |
| **Recurrent Gaussian Gen** | 이전 프레임 Gaussian을 참조해 현 프레임 생성 | ⭐⭐⭐ (미래 방향) |

#### 최소 실험 (Minimum Viable Experiment)

1. **Neutral pose canonical 선택** — 서있는 자세, 최대 body part 노출 프레임
2. **기존 checkpoint로 deform 적용** (10K steps, blend_alpha=0.3-0.7)
3. **NN matching**으로 N 불일치 해결: `cKDTree(canonical_xyz).query(target_xyz, k=1)`
4. 렌더 비교: 원본 vs deformed (tOF + 육안)
5. **예상**: 작은 motion 구간에서 부분적 개선, 큰 motion에서 stretching artifact

### Feasibility Assessment (from /deliberate, updated 2026-03-21)

| Factor | Face (논문) | Mouse (우리) | 대응 |
|--------|:-----------:|:-----------:|------|
| Topology consistency | High | Low | Multi-canonical 필요 |
| Inter-frame motion | Small | Large | Sliding window 또는 recurrent |
| Self-occlusion | Rare | Frequent | Gaussian birth/death 필요 |
| Feed-forward 제약 | N/A (per-scene) | 필수 | Generalized canonical 학습 |
| Expected benefit | High | **Low-Medium** | 최소 실험으로 검증 |

**결론**: 단일 canonical 접근은 mouse에서 근본적 한계. Multi-canonical 또는 recurrent 접근이 필요하나, 현재 NeurIPS 일정상 **DiFix (artifact removal) 우선**, deformation은 future work로 분류.

## 5. Phase 3: Advanced Methods (Survey)

| Method | Type | Effort | Expected Effect | Status |
|--------|------|:------:|:---------------:|--------|
| 3D Scene Flow | 3D | Very High | High | 미구현, 차기 |
| Test-Time Optim | 3D | High | High | 미구현 |
| Neural Temporal Embed | 3D | Very High | Highest | 아키텍처 변경 필요 |
| NN-Match + Interp | 3D | Medium | Medium | Phase 2와 함께 |

### Optical Flow 한계 분석

Farneback optical flow가 효과 없었던 이유:
- **Flow 추정 자체가 jitter에 오염**: GS-LRM flickering → noisy flow → noisy warp
- **Occlusion/disocclusion**: mouse motion에서 빈번 → warp artifact
- **3D motion → 2D projection**: novel view에서의 parallax가 2D flow로 캡처 불가

**대안**: RAFT (learned) optical flow가 Farneback보다 robust할 수 있으나, 핵심 한계(3D→2D)는 동일.

## 6. Output Locations

| Type | Path |
|------|------|
| Unified renders | `outputs/datasets/temporal_eval/{experiment}/` |
| Temporal strips | `outputs/report/temporal_comparison/strips/` |
| Method grids | `outputs/report/temporal_comparison/grids/` |
| Metrics JSON | `outputs/report/temporal_comparison/metrics/temporal_comparison.json` |
| Comparison table | `outputs/report/temporal_comparison/metrics/comparison_table.md` |

## 7. Phase 1.5 Results: Window-Based Methods (2026-03-21)

EMA의 motion blur 문제를 해결하기 위해 window-based 방법 추가 테스트.

| Method | tOF↓ | 감소율 | Motion Blur | 비고 |
|--------|:----:|:------:|:-----------:|------|
| **Median w=5** | 0.139 | 56% | Minimal | 최고 균형 |
| SavGol w=5 | 0.221 | 30% | Very Low | Edge 보존 |
| EMA α=0.3 | 0.080 | 75% | Medium | |
| Bilateral w=5 σ=25 | 0.307 | 2% | None | ❌ 비효과적 |

### 육안 평가 결론

> **⚠️ 모든 2D temporal smoothing 방법이 원본 대비 시각적 품질 저하.**
> - Median/SavGol: flickering 감소하나 디테일 손실, "뭉개짐" 느낌
> - EMA: motion blur로 인한 잔상
> - Bilateral: 효과 미미
> - **결론: 2D post-processing으로는 temporal consistency 개선 불가. 원본 유지.**
> - 근본 해결은 3D 수준 (deformation, scene flow, 또는 아키텍처 변경) 필요.

**Status: 보류 (on hold)**. 2D temporal smoothing 탐색 종료. 원본 렌더 사용.

## 8. Conclusion & Next Steps

1. ~~EMA α=0.1은 간단하고 효과적인 baseline~~ → **육안 평가 결과 시각적 품질 저하, 원본 유지**
2. **Alpha loss는 temporal stability에 영향 없음** — alpha loss의 기여는 novel view artifact 감소에 국한
3. **3D-level smoothing (DeformV2, Scene Flow) 필요** — 2D 후처리의 한계 확인됨
4. **Bottom view가 핵심 challenge** — 가장 많은 flickering, 향후 개선 타겟
5. **Deformation module 재검토 필요** — canonical Gaussian 기반 접근의 mouse 적용 가능성 분석 필요

---

## Related

- ↑ [[INDEX]] (문서 허브)
- ↔ [[TEMPORAL_EVAL_STANDARD]] (평가 기준 SSOT)
- ↔ [[COMMANDS]] (실험 명령어)
- ↓ `mouse_extensions/evaluation/temporal_smoothing.py` (구현)
- ↓ `mouse_extensions/scripts/eval/temporal_comparison.py` (비교 파이프라인)

---

*Created: 2026-03-21 | Phase 1 완료, Phase 2-3 planned*
