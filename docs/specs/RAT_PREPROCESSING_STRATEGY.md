# RAT Preprocessing Strategy: Zero-Pad Analysis

> **Navigation**: [← INDEX](../INDEX.md) | [PREPROCESSING_REGISTRY](../datasets/PREPROCESSING_REGISTRY.md) | [M5_SERIES_SPEC](../datasets/M5_SERIES_SPEC.md)
> **Purpose**: RAT2 GS-LRM 전처리 전략 분석 — `/deliberate --audit --loop 3` (수렴)
> **Created**: 2026-03-30
> **Audit**: 3-model × 3-loop (Claude Haiku, Gemini 2.5 Pro, o3-mini). Loop 3에서 수렴.

---

## 1. Problem Statement

RAT2 데이터를 GS-LRM fine-tuning 포맷으로 변환 시, foreground coverage와 fx 차이에 대한 우려.

| Dataset | FG Coverage | FG Pixels (per 512²) | fx | PP |
|---------|:-----------:|:--------------------:|:--:|:--:|
| **Mouse M5t** (pretrained) | **2.5%** | ~6,500 | 549 | (256, 256) |
| **RAT2** (zero-pad) | **1.5%** | ~3,900 | 605 | (~255, ~245) |

> ⚠️ 이전 문서에서 mouse coverage를 2.9%로 기재했으나, 10-sample 측정 결과 **2.0-2.8% (mean 2.5%)**로 교정됨.

---

## 2. Audit Evolution (Loop 1→2→3)

### Loop 1: 초기 권장 (3-model)
- "CoverageNormalizer(6%) 적용" 권장
- "1.5% vs 2.9%는 2:1 차이, Critical"

### Loop 2: 증거 기반 교정 (3-model 합의로 **Loop 1 철회**)
- CoverageNormalizer의 6% target은 **문서 오류** (실제 M5t = 2.5%)
- CoverageNormalizer는 **inference-only** 코드 (학습 전처리 아님)
- M5 preset은 **zoom 미사용** (M3/M4만 zoom 사용)
- Coverage gap은 1pp (1.5% vs 2.5%) — 이전 분석에서 과장

### Loop 3: 수렴 — fx 분석 + 최종 결론
- fx=605는 **정확한 물리적 값** (오류 아님)
- GS-LRM은 fx를 explicit input으로 받음 → 모델이 geometry를 "알고 있음"
- **현재 zero-pad 전처리가 Pareto-optimal**: fx↓ 시도 시 FG↓ (상충)
- Gemini: "CONVERGED — no new findings beyond Loop 2"

---

## 3. CoverageNormalizer 상세

### 3.1 위치 및 용도

```
mouse_extensions/inference/preprocessing/coverage_normalizer.py
```

**Inference-only** 모듈. 학습 전처리(`mouse_extensions/preprocessing/`)와 별도.

### 3.2 동작 원리

```python
class CoverageNormalizer:
    target_coverage = 0.06  # ⚠️ 문서상 "M5 training ~6%" — 실제 M5t=2.5%와 불일치

    def compute_scale(self, mask):
        current_coverage = (mask > 127).sum() / (H * W)
        scale = sqrt(target_coverage / current_coverage)  # Area scales with scale²
        return clip(scale, 0.3, 3.0)

    def normalize(self, image, mask, centroid):
        scale = self.compute_scale(mask)
        M = [[scale, 0, center - scale*cx],
             [0, scale, center - scale*cy]]  # Affine: scale + translate centroid→center
        return warpAffine(image, M, (512, 512))
```

### 3.3 사용 맥락

| 맥락 | 사용 여부 | 비고 |
|------|:---------:|------|
| **M5 학습 전처리** | ❌ | M5는 zoom 없음 (affine + batch norm만) |
| **M3/M4 학습 전처리** | ⚠️ 유사 | `compute_adaptive_zoom_coverage()` (별도 함수, target 5%) |
| **Inference** | ✅ | 테스트 이미지를 학습 분포에 맞추기 위해 사용 |
| **RAT2 변환** | ❌ | `sdannce_to_gslrm.py`는 별도 파이프라인 |

### 3.4 6% Target 오류

CoverageNormalizer 문서에 "M5 training coverage ~6%"로 기재되어 있으나:
- 실측 M5t coverage: **2.0-2.8% (mean 2.5%)**
- M3 zoom target: **5%**
- 6%의 출처 불명 — 아마 M3 zoom 결과를 M5로 오인했거나, 개별 프레임 최대값

> **TODO**: `coverage_normalizer.py` docstring 교정 (6% → 2.5% 또는 M3 기준 명시)

---

## 4. Three Strategies Compared

### 4.1 Zero-Pad (현재 RAT2)

```
1920×1200 → pad(360,360) → 1920×1920 → resize → 512×512
```

- **FG**: 1.5% | **fx**: 605 | **PP**: (~255, ~245)
- **장점**: PP 안정, 구현 단순, 프레임간 일관성
- **단점**: FG 작음, fx mismatch 10%

### 4.2 Object-Centered Crop

```
1920×1200 → crop(COM_2d, bbox+pad) → resize → 512×512
```

- **FG**: 제어 가능 | **fx**: 프레임별 변동 | **PP**: 보정 필수
- **장점**: FG 크기 제어
- **단점**: PP 보정 실수 위험 (D4 Bug 전례), 프레임별 fx 변동

### 4.3 Zero-Pad + Center Zoom (Hybrid)

```
1920×1200 → pad → 1920×1920 → zoom → crop center → 512×512
```

- **FG**: 제어 가능 | **fx**: zoom 비례 증가 (549 달성 불가!) | **PP**: 안정
- **⚠️ 핵심 문제**: fx를 549로 맞추려면 **더 넓게** 패딩 (2100²) → FG **1.0%로 감소**
- zoom IN은 fx를 **증가**시킴 (반대 방향)

### 4.4 Pareto Analysis

```
fx와 FG는 상충 관계 (rat 카메라 물리적 제약):

                fx
            700 ┤      ·(zoom in)
                │     ·
            605 ┤ ★ 현재 (Pareto front)
                │   ·
            549 ┤     · (pad 2100²: FG=1.0%)
                │
                └──┬──┬──┬──┬──→ FG%
                  1.0 1.5 2.0 2.5

★ = 현재 zero-pad (Pareto-optimal)
```

**fx를 549로 맞추면 FG가 1.0%로 감소** — 두 지표를 동시에 개선하는 전처리는 없음.

---

## 5. D4 PP Bug Clarification

**CLAUDE.md**: `⛔ PP=256 강제 + Object-centered crop (MVG 부정합)`

**정확한 범위**: crop 후 cx/cy를 **offset 무시하고** 256으로 하드코딩한 것이 문제.
- Object-centered crop 자체가 금지가 아님
- PP를 올바르게 보정하면 유효한 전략
- Center zoom은 이 anti-pattern에 **해당하지 않음** (PP 자동 중앙)

---

## 6. Mouse 전처리 비교 실험 현황

### 6.1 존재하는 비교 데이터

| 비교 | 데이터 | 정량 결과 |
|------|--------|:---------:|
| v1(crop) vs v2(pad) vs v3(lone) vs v4(masked) | smoke test 디렉토리 존재 | **❌ 없음** |
| D4 vs D6 vs D7 vs D8 | D7_1, D8 전처리 완료 (각 3238) | **❌ 학습 미실행** |
| M5 vs M5_4 vs M5_5 | Config 설계 완료 | **❌ ablation 미실행** |
| M5 vs M3 (zoom 유무) | 둘 다 구현 | **❌ 비교 없음** |

### 6.2 결론

**이 프로젝트에서 전처리 전략 간 정량 비교 실험은 단 한 건도 수행되지 않았음.**
- M5가 "권장"인 이유: 기하학적 추론 (D3n PSNR~3 실패 → PP centering 필수)
- zoom 유무 비교: 미수행
- "M5가 best" 결론에 실험적 근거 없음 (다만 기하학적 근거는 강함)

---

## 7. 최종 권장 (3-model × 3-loop 합의)

### 판정: 현재 zero-pad 유지, 실험적 검증 후 판단

**근거**:
1. **Coverage gap은 과장됨**: 1.5% vs 2.5% = 1pp 차이 (6%는 문서 오류)
2. **fx=605는 정당한 물리값**: GS-LRM이 explicit input으로 받음, 오류가 아님
3. **현재 설정이 Pareto-optimal**: fx↓와 FG↑는 상충, 동시 개선 불가
4. **실험적 근거 없이 전처리 변경은 리스크**: 어떤 변경도 unintended effect 가능
5. **Fine-tuning이 domain gap 흡수**: 10% fx 차이는 FT로 적응 가능 범위

### 실행 계획

```
Phase 1: 현재 zero-pad (fg=1.5%, fx=605)로 RAT2 FT 실행
         → val PSNR, IoU 측정
Phase 2: 결과가 RAT1(17.49dB)보다 유의미 개선 → 전처리 변경 불필요
         결과가 기대 이하 → 전처리 ablation 실시:
           A) 현재 (baseline)
           B) Center zoom (fx↑ but FG↑)
           C) Extra pad (fx=549 but FG↓)
```

### CoverageNormalizer Docstring 교정 (TODO)

```python
# 현재: "Adjusts scale to match M5 training coverage (~6%)"
# 교정: "Adjusts scale for inference normalization (target configurable, M5t actual ~2.5%)"
```

---

## Related Documents

- [PREPROCESSING_REGISTRY](../datasets/PREPROCESSING_REGISTRY.md) — 전처리 버전 이력
- [M5_SERIES_SPEC](../datasets/M5_SERIES_SPEC.md) — 카메라 정규화 ablation 설계
- Obsidian `docs/theory/COORDINATE_SYSTEMS.md` — 카메라 컨벤션 SSOT
- `mouse_extensions/inference/preprocessing/coverage_normalizer.py` — CoverageNormalizer (inference-only)

---

*2026-03-30 | RAT Preprocessing Strategy | `/deliberate --audit --loop 3` (converged)*
