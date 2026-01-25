# View Selection & Novel View Synthesis Analysis

> **Navigation**: [← Index](./00_INDEX.md) | [Experiments](./EXPERIMENT_NAMING.md)
> **SSOT**: Ghosting 현상 분석 및 View 선택 전략

---

## 1. 핵심 발견

### ⚠️ "Ghosting"은 실제 Ghosting이 아님

```
기존 해석: Multi-view reconstruction ghosting artifact
실제 원인: Novel view synthesis 한계
```

**증거**:
- View 0-3: PSNR 27-33 (높음)
- View 4-5: PSNR 10-13 (낮음)
- 차이 원인: View 4-5가 validation에서 항상 novel view

---

## 2. View 선택 메커니즘

### 코드 분석 (mouse_dataset.py:276-279)

```python
if random_view_selection and split == "train":
    input_indices = sorted(random.sample(all_indices, 4))
else:
    input_indices = list(range(4))  # [0, 1, 2, 3] 고정
```

### 동작 방식

| Split | View Selection | Input Views | Novel Views |
|-------|---------------|-------------|-------------|
| **Train** (random=True) | Random 4개 | 가변 | 가변 |
| **Train** (random=False) | Fixed | [0,1,2,3] | [4,5] |
| **Validation** | **Always Fixed** | [0,1,2,3] | [4,5] |

### 결론

```
Validation에서 View 4-5는 항상 novel view
→ 학습 데이터에서 본 적 없는 각도
→ 낮은 PSNR은 정상 (generalization 한계)
```

---

## 3. View별 PSNR 패턴

### 실험 결과

| View | 역할 (Val) | PSNR 범위 | 해석 |
|------|-----------|-----------|------|
| 0 | Input | 27-33 | Reconstruction |
| 1 | Input | 27-33 | Reconstruction |
| 2 | Input | 27-33 | Reconstruction |
| 3 | Input | 27-33 | Reconstruction |
| **4** | **Novel** | **10-13** | Synthesis |
| **5** | **Novel** | **10-13** | Synthesis |

### 시각화

```
View Quality Distribution (Validation)

PSNR
 35 ┤
 30 ┤  ████████████████
 25 ┤  ████████████████
 20 ┤  ████████████████
 15 ┤  ████████████████
 10 ┤  ████████████████  ████████
  5 ┤                    ████████
    └──────────────────────────────
       V0  V1  V2  V3  V4  V5
       ──────────────  ────────
        Input Views    Novel Views
```

---

## 4. H6 가설: View Quality

### 가설 정의

**H6**: Validation의 View 4-5 낮은 PSNR은 ghosting이 아닌 novel view synthesis 한계

### 검증 방법

1. **Train에서 random view selection 활성화**
2. **모든 뷰가 input으로 학습되도록 설정**
3. **Validation에서 View 4-5 PSNR 변화 관찰**

### 예상 결과

```
random_view_selection=True 시:
- View 4-5도 학습 중 input으로 사용됨
- Validation에서 View 4-5 PSNR 상승 예상
```

---

## 5. 실험 권장사항

### A. Random View Selection 활성화

```yaml
# configs/experiments/E1_2_random_baseline.yaml
training:
  random_view_selection: true
  num_input_views: 4
```

### B. View Count 증가 (E3 시리즈)

| 설정 | Input | Novel | 기대 효과 |
|------|-------|-------|-----------|
| 4v | 4 | 2 | 기본 |
| 5v | 5 | 1 | Novel 1개만 |
| **6v** | **6** | **0** | **Novel 없음** ⭐ |

### C. 6v 설정 (권장)

```yaml
# configs/experiments/E3_3_6v.yaml
training:
  num_input_views: 6
  # 모든 뷰가 input → reconstruction only
```

---

## 6. 진단 도구

### View별 PSNR 분석 스크립트

```bash
python mouse_extensions/scripts/diagnostics/analyze_per_view_psnr.py \
    --checkpoint outputs/M3_2_E2_1/best.pt \
    --dataset M3_2 \
    --output analysis/view_psnr_breakdown.json
```

### 출력 예시

```json
{
  "per_view_psnr": {
    "view_0": 31.2,
    "view_1": 29.8,
    "view_2": 30.5,
    "view_3": 28.9,
    "view_4": 11.3,
    "view_5": 12.1
  },
  "input_views_avg": 30.1,
  "novel_views_avg": 11.7,
  "gap": 18.4
}
```

---

## 7. 결론 및 권장사항

### 핵심 인사이트

1. **View 4-5 낮은 PSNR은 정상** (novel view synthesis 한계)
2. **실제 ghosting artifact 아님**
3. **Random view selection으로 개선 가능**

### 실험 우선순위

| Priority | 실험 | 목적 |
|----------|------|------|
| P0 | M3_2 + E1_2_alpha | 기본 검증 |
| P1 | M3_2 + E1_2_random | H6 검증 |
| P2 | M3_2 + E3_3_6v | Novel view 제거 |

### 보고 시 주의사항

```
❌ "View 4-5에서 ghosting 발생"
✅ "View 4-5는 novel view로 synthesis 품질 측정"
```

---

## 8. 관련 문서

- [[00_INDEX]] - Dataset Hub
- [[EXPERIMENT_NAMING]] - 실험 명명규칙
- [[HYPOTHESIS_VERIFICATION]] - 가설 검증 (H6 포함)
- [[presets/M3_2]] - 권장 데이터셋

---

*View Selection Analysis v1.0 | 2026-01-26*
