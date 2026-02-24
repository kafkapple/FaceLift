# RMA & Camera Layout Analysis: M5 vs Original FaceLift

> **목적**: M5 비균일 카메라 배치가 RMA에 미치는 영향 분석 + Random Reference 실험 설정 가이드
> **상태**: 📋 분석 완료 | **Created**: 2026-02-13
> **관련**: → [GENERALIZATION_ROADMAP](./GENERALIZATION_ROADMAP.md) | [H5_MVDIFFUSION](./H5_MVDIFFUSION.md)

---

## 1. 카메라 배치 비교

### 1.1 Original FaceLift Cameras (Era3D RMA architecture)

`utils_folder/opencv_cameras.json` — 6개 canonical views:

| View | Elevation | Azimuth | Distance |
|:----:|:---------:|:-------:|:--------:|
| cam_000 | **0.00°** | -180.0° | 2.700 |
| cam_001 | **0.00°** | -135.0° | 2.700 |
| cam_002 | **0.00°** | -90.0° | 2.700 |
| cam_003 | **0.00°** | -45.0° | 2.700 |
| cam_004 | **0.00°** | 0.0° | 2.700 |
| cam_005 | **0.00°** | 90.0° | 2.700 |

**특성**:
- **Elevation: 모두 0.00°** (완벽한 equatorial plane)
- Azimuth gaps: 45°, 45°, 45°, 45°, 90°, 90° (비균일하지만 수평면)
- Distance: 모두 2.700 (동일)

### 1.2 M5 Mouse Cameras

`mouse_extensions/inference/cameras/m5_cameras.json` — M5 rig 6개 카메라:

| View | Prompt | Elevation | Azimuth | Distance |
|:----:|--------|:---------:|:-------:|:--------:|
| cam_000 | top-front | **-7.87°** | -145.1° | 2.799 |
| cam_001 | top-front-right | **+7.14°** | +32.3° | 2.739 |
| cam_002 | top-right | **-6.68°** | +96.7° | 2.706 |
| cam_003 | top-back | **-8.53°** | -23.6° | 2.593 |
| cam_004 | top-left | **+6.42°** | +158.2° | 2.757 |
| cam_005 | top-front-left | **+9.58°** | -79.2° | 2.607 |

**특성**:
- **Elevation: -8.53° ~ +9.58° (18.1° 편차)**
- Azimuth gaps (정렬 순): 55.6°~65.9° (비교적 균일, 평균 ~60°)
- Distance: 2.593~2.799 (약간 변동)

### 1.3 핵심 차이 시각화

```
Elevation 비교:
                 Original              M5 Mouse
  +10° ┤                          cam_5 (+9.6°)
   +7° ┤                          cam_1 (+7.1°), cam_4 (+6.4°)
    0° ┤  ═══ ALL 6 cameras ═══
   -7° ┤                          cam_0 (-7.9°), cam_2 (-6.7°)
   -9° ┤                          cam_3 (-8.5°)
        ──────────────────────────────────────────────────

두 그룹이 존재:
  HIGH group: cam_1(+7.1°), cam_4(+6.4°), cam_5(+9.6°) — 그룹 내 ≤3.2°
  LOW group:  cam_0(-7.9°), cam_2(-6.7°), cam_3(-8.5°) — 그룹 내 ≤1.8°
  그룹 간: 13~18° 차이
```

---

## 2. RMA (Row-wise Multi-view Attention) 실제 구현 분석

### 2.1 이론 vs 코드

**Era3D 논문의 이론**:
> Canonical camera (elevation 0°, equispaced azimuth) → epipolar line = 수평 행
> → Row-wise attention으로 효율적 multi-view 일관성 달성

**실제 FaceLift 코드** (`mvdiffusion/models/transformer_mv2d_image.py:814-834`):

```python
# XFormersMVAttnProcessor.__call__()
if multiview_attention:
    if not sparse_mv_attention:
        # FULL: 모든 뷰의 모든 token을 하나로 concat
        key = my_repeat(rearrange(key_raw, "(b t) d c -> b (t d) c", t=num_views), num_views)
        value = my_repeat(rearrange(value_raw, "(b t) d c -> b (t d) c", t=num_views), num_views)
    else:
        # SPARSE: slot_0 (reference)의 token + 자기 자신의 token
        key_front = my_repeat(rearrange(key_raw, "(b t) d c -> b t d c", t=num_views)[:, 0, :, :], num_views)
        value_front = my_repeat(rearrange(value_raw, "(b t) d c -> b t d c", t=num_views)[:, 0, :, :], num_views)
        key = torch.cat([key_front, key_raw], dim=1)
        value = torch.cat([value_front, value_raw], dim=1)
```

> **핵심 발견: "Row-wise"가 코드로 강제되지 않음!**
>
> 코드는 **dense attention** — 모든 spatial token이 다른 모든 token에 attend 가능.
> "Row" 대응관계는 canonical camera에서 자연스럽게 **학습되는 암묵적 패턴**이지,
> attention mask로 강제하는 것이 아님.

### 2.2 Sparse vs Full 비교

| | Full (`sparse=false`) | Sparse (`sparse=true`) |
|---|---|---|
| 각 뷰가 attend 하는 범위 | 6개 뷰 전체 (6d tokens) | 자신 + reference (2d tokens) |
| 뷰 간 관계 수 | 6×6 = 36 쌍 | 6×1 = 6 쌍 (star topology) |
| 메모리 | O(36d²) | O(12d²) = **3x 절약** |
| Baseline PSNR (H5) | 26.55 (CFG 3.0) | **27.30 (CFG 3.0)** |
| 뷰 간 간접 통신 | 직접 (모든 쌍) | reference 경유만 |

**Sparse가 더 좋은 이유**: Star topology가 작은 데이터셋(3,600장)에서 유익한 inductive bias로 작용.

### 2.3 Sparse에서 Reference 선택의 의미

```python
key_front = rearrange(key_raw, "(b t) d c -> b t d c", t=num_views)[:, 0, :, :]
#                                                                      ↑
#                                                              항상 slot 0
```

- **Slot 0** = 데이터 로더가 첫 번째로 넣은 뷰 = reference view
- Baseline (`ref=0` 고정): slot_0 = 항상 cam_000
- Random ref (`ref="random"`): slot_0 = 매번 다른 카메라

---

## 3. Epipolar Tilt 분석

### 3.1 Epipolar Line의 기울기

두 카메라의 elevation이 다르면, 한 이미지에서의 epipolar line이 수평이 아닌 **기울어진 직선**이 됨:

```
Elevation 차이 = 0° (Original):     Elevation 차이 = 18° (M5 worst case):
┌──────────────────┐                ┌──────────────────┐
│ ════════════════  │ ← 수평        │ ╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲  │ ← ~18° 기울기
│ ════════════════  │               │  ╲╲╲╲╲╲╲╲╲╲╲╲╲╲╲ │
│ ════════════════  │               │   ╲╲╲╲╲╲╲╲╲╲╲╲╲╲ │
└──────────────────┘                └──────────────────┘
   완벽한 행 대응                      행이 어긋남
```

### 3.2 M5 Pairwise Elevation 차이

| 카메라 쌍 | Elev 차이 | 상태 |
|-----------|:---------:|:----:|
| cam_0 ↔ cam_3 | 0.7° | ✅ OK |
| cam_1 ↔ cam_4 | 0.7° | ✅ OK |
| cam_0 ↔ cam_2 | 1.2° | ✅ OK |
| cam_2 ↔ cam_3 | 1.8° | ✅ OK |
| cam_1 ↔ cam_5 | 2.4° | ✅ OK |
| cam_4 ↔ cam_5 | 3.2° | ✅ OK |
| cam_2 ↔ cam_4 | 13.1° | ⚠️ WARN |
| cam_1 ↔ cam_2 | 13.8° | ⚠️ WARN |
| cam_0 ↔ cam_4 | 14.3° | ⚠️ WARN |
| cam_3 ↔ cam_4 | 14.9° | ⚠️ WARN |
| cam_0 ↔ cam_1 | 15.0° | ⚠️ HIGH |
| cam_1 ↔ cam_3 | 15.7° | ⚠️ HIGH |
| cam_2 ↔ cam_5 | 16.3° | ⚠️ HIGH |
| cam_0 ↔ cam_5 | 17.4° | ⚠️ HIGH |
| **cam_3 ↔ cam_5** | **18.1°** | **⚠️ MAX** |

**패턴**: 같은 그룹(HIGH-HIGH 또는 LOW-LOW) 내 ≤3.2°, 다른 그룹 간 13~18°.

### 3.3 Baseline이 잘 동작하는 이유

Baseline (`ref=0` 고정, `sparse=true`):
- 항상 cam_000(elev -7.9°)이 reference
- 각 타겟 뷰와의 관계가 **고정** — 모델이 특정 기하학 하나만 학습
- Dense attention이므로 기울어진 epipolar도 attend **가능**
- 결과: PSNR 27.30 달성

---

## 4. Random Reference의 영향

### 4.1 데이터 로더 동작 (`mouse_dataset.py:194-211`)

```python
if self.reference_view_idx == "random":
    current_ref_idx = random.choice([0, 1, 2, 3, 4, 5])
    rotated_target_indices = [(current_ref_idx + i) % 6 for i in range(6)]
    # 예: ref=3 → [3, 4, 5, 0, 1, 2]

    # Prompt embedding도 동일하게 rotate
    rotated_prompt_embedding = self.color_prompt_embedding[rotated_target_indices]
```

### 4.2 Random Ref가 각 reference에서 만드는 기하학

| Reference | Targets | Max Elev Diff | Mean Elev Diff |
|:---------:|---------|:-------------:|:--------------:|
| cam_000 (e=-7.9°) | 1,2,3,4,5 | 17.4° | 9.7° |
| cam_001 (e=+7.1°) | 2,3,4,5,0 | 15.7° | 9.5° |
| cam_002 (e=-6.7°) | 3,4,5,0,1 | 16.3° | 9.2° |
| cam_003 (e=-8.5°) | 4,5,0,1,2 | **18.1°** | **10.2°** |
| cam_004 (e=+6.4°) | 5,0,1,2,3 | 14.9° | 9.2° |
| cam_005 (e=+9.6°) | 0,1,2,3,4 | **18.1°** | **11.5°** |

- 매 iteration마다 다른 reference → **다른 epipolar geometry**
- Baseline은 항상 cam_000 기준의 고정된 기하학만 학습
- Random ref는 6가지 기하학 패턴을 모두 학습해야 함

### 4.3 Original Camera에서는 왜 무해한가

Original (elevation 모두 0°):
- 어떤 카메라가 reference든 epipolar = **항상 수평**
- Cyclic rotation = 순수 대칭 변환 → **기하학 불변**
- Random ref = 무해 (실제로 Era3D가 이 설정에서 학습)

M5 (elevation 변동):
- Reference마다 epipolar tilt 패턴이 달라짐
- 모델이 6가지 다른 기하학을 학습해야 → **더 어려운 과제**

### 4.4 Cyclic 실패와의 관계

`mvdiff_M5t2_cyclic` (PSNR 21.80, DELETED):
- `sparse_mv_attention: false` + `reference_view_idx: [0,1,2,3,4,5]`

| 요인 | 영향 |
|------|------|
| Full attention (sparse=false) | Baseline 대비 -0.75 dB (cfgr 실험에서 확인) |
| Random ref + M5 비균일 | 매번 다른 epipolar geometry |
| Full × Random = **36쌍 × 다양한 기하학** | 과부하 → **-5.5 dB** |

`randref_sparse` (제안):
- `sparse_mv_attention: true` + `reference_view_idx: "random"`

| 요인 | 영향 |
|------|------|
| Sparse attention (유지) | Baseline과 동일 구조 |
| Random ref + M5 비균일 | 매번 다른 기하학 |
| Sparse × Random = **6쌍 × 다양한 기하학** | Cyclic 대비 **6배 적은 부담** |

---

## 5. 실험 설정 권장사항

### 5.1 randref_sparse가 적합한 이유

1. **단일 변수 ablation**: ref만 변경 (0 → "random"), sparse 유지 → 변인 통제
2. **Cyclic 대비 안전**: sparse mode가 cross-view 관계 복잡도 6배 감소
3. **P1 pose conditioning의 전제 조건**: 임의 방향 입력을 위해 random ref 동작 검증 필수
4. **실패해도 가치**: "M5 비균일 카메라에서 view-invariant 학습 가능 여부" 답 제공

### 5.2 Config 확인 (이미 준비됨)

`configs/mvdiffusion/mouse_mvdiffusion_M5t2_randref_sparse.yaml`:
- `reference_view_idx: "random"` ← KEY CHANGE
- `sparse_mv_attention: true` ← SAME as baseline
- 나머지 모든 설정 baseline과 동일

### 5.3 모니터링 지표

| 지표 | 확인 사항 |
|------|-----------|
| **Overall PSNR** | Baseline 27.30 대비 하락폭 (1-2 dB 이내 = 성공) |
| **Per-view PSNR** | HIGH↔LOW 그룹 간 성능 차이 |
| **Loss curve** | 안정적 수렴 여부 (cyclic은 발산/진동했을 수 있음) |
| **Validation 영상** | 뷰 간 일관성 (특히 cam_3↔cam_5 쌍) |

### 5.4 실패 시 대안

1. **Restricted random**: 같은 elevation 그룹 내에서만 random ref
   - HIGH group: cam_1, cam_4, cam_5 (그룹 내 ≤3.2°)
   - LOW group: cam_0, cam_2, cam_3 (그룹 내 ≤1.8°)
2. **Gradual warm-up**: 처음 N step은 ref=0 → 이후 점진적 random 도입
3. **Pose conditioning 선행**: random ref 없이 바로 P1 spherical pose로 진행

### 5.5 MV-Adapter와의 비교

| | FaceLift RMA (현재) | MV-Adapter |
|---|---|---|
| Row attention | **암묵적** (dense attention, 학습으로 패턴 형성) | **명시적** (`rearrange(..., "b v h hh ww c -> (b hh) h (v ww) c")`) |
| Elevation 차이 허용 | 가능 (dense이므로) | 제한적 (row-only mask 강제) |
| Camera conditioning | 없음 (text prompt만) | Plücker ray (6ch) + T2IAdapter |
| Random ref 대응 | 어려움 (기하학 변경 시 재학습) | 용이 (camera guider가 기하학 인코딩) |

→ **장기적으로 MV-Adapter가 M5 비균일 카메라에 더 적합** (camera guider가 기하학을 명시적으로 인코딩)
→ **단기적으로 현재 RMA + randref_sparse는 합리적 실험** (dense attention이 elevation 차이를 보상 가능)

---

## 6. 결론

### M5 비균일 카메라의 영향 요약

1. **Elevation 18.1° 편차**: Original(0°) 대비 큰 차이 → epipolar 기울기 발생
2. **RMA 코드는 dense attention**: Row 강제가 아니므로 기울어진 epipolar도 처리 가능
3. **Baseline이 잘 작동**: 고정된 ref=0 기하학 하나만 학습 → 성공 (27.30)
4. **Random ref의 도전**: 6가지 다른 기하학 학습 필요 → 더 어려운 과제
5. **Sparse mode가 완충**: 관계 복잡도를 6배 줄여 random ref 부담 경감

### 권장 실행 순서

1. **randref_sparse 실행** (현재 config 그대로, 단일 변수 ablation)
2. per-view PSNR 모니터링 → HIGH↔LOW 그룹 간 성능 차이 확인
3. 결과에 따라:
   - 성공 (≤2 dB 하락) → P1 pose conditioning 진행
   - 실패 → restricted random 또는 pose conditioning 선행

---

*RMA & Camera Layout Analysis v1.0 | 2026-02-13*
