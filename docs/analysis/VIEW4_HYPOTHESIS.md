# View 4 문제 가설 및 검증 계획

**작성일**: 2026-01-25  
**상태**: 가설 수립 → 검증 대기

---

## 1. 관찰된 현상

### 1.1 Per-View PSNR 패턴 (D3_normalized, 6000 step)

| View | PSNR | 상대 성능 |
|------|------|-----------|
| 0 | 31.6 | 100% |
| 1 | 27.8 | 88% |
| 2 | 32.8 | 104% |
| 3 | 31.3 | 99% |
| **4** | **10.8** | **34%** |
| **5** | **12.8** | **40%** |

### 1.2 View 4 특이사항

| 지표 | View 4 | 다른 뷰 평균 | 차이 |
|------|--------|-------------|------|
| **Coverage** | 5.37% | 7.70% | **-30%** |
| **Brightness** | 27.9 | 44.0 | **-37%** |
| **Edge Sharpness** | 0.49 | 0.62 | -21% |

---

## 2. 가설

### 가설 H1: View 4는 데이터 품질 문제

**논리 구조**:
```
전제 1: View 4는 coverage가 가장 낮음 (5.37%)
전제 2: View 4는 밝기가 가장 어두움 (27.9)
전제 3: Validation에서 View 4는 항상 novel view
───────────────────────────────────────────
결론: View 4의 낮은 데이터 품질 + novel view 조합이 PSNR 급락 원인
```

**세부 가설**:

#### H1a: Coverage 부족
- 5.37%는 다른 뷰 대비 30% 낮음
- Foreground 신호 부족 → gradient 약함 → 학습 불충분
- **예측**: Coverage 높은 뷰로 교체 시 개선

#### H1b: 밝기 불균형
- 27.9는 다른 뷰 대비 37% 어두움
- 어두운 영역의 detail 손실 → 재구성 어려움
- **예측**: 밝기 정규화 시 개선

#### H1c: 카메라 위치 문제
- View 4: azimuth=-58.3°, elevation=31.7°
- Input [0-3]에서 가장 가까운 뷰와 14° 차이
- **예측**: View 4를 input에 포함 시 개선

### 가설 H2: View 4,5 동시 novel view 문제

**논리 구조**:
```
전제 1: Validation input = [0,1,2,3] 고정
전제 2: View 4,5가 항상 novel target
전제 3: View 4,5는 서로 다른 방향 (azimuth 차이 94.5°)
───────────────────────────────────────────
결론: 2개 novel view가 서로 다른 방향이라 interpolation 어려움
```

---

## 3. 검증 실험 계획

### 3.1 실험 매트릭스

| 실험 ID | 변경 사항 | 검증 가설 | 우선순위 |
|---------|-----------|-----------|----------|
| **EXP-A** | 6뷰 모두 input | H1c, H2 | **P0** |
| **EXP-B** | View 4 제외 (5뷰) | H1a, H1b | **P1** |
| **EXP-C** | Input [0,2,4,5] | H1c | P2 |
| **EXP-D** | Validation random | H2 | P2 |

### 3.2 실험 상세

#### EXP-A: 6뷰 모두 input (최우선)

**목적**: View 4,5가 input으로 학습되면 개선되는지 확인

**Config**: `E_overfit_6v.yaml`
```yaml
model:
  num_views: 6
  num_input_views: 6
```

**예상 결과**:
- 성공 (모든 뷰 PSNR 25+): H1c, H2 지지 → novel view 문제
- 실패 (View 4 여전히 낮음): H1a, H1b 가능성 → 데이터 문제

**실행**:
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D3_normalized -e E_overfit_6v
```

#### EXP-B: View 4 제외 (5뷰)

**목적**: View 4 데이터 자체가 문제인지 격리

**Config**: `E_exclude_view4.yaml` (신규 작성 필요)
```yaml
model:
  num_views: 5
  num_input_views: 4

training:
  dataset:
    exclude_views: [4]  # View 4 제외
```

**예상 결과**:
- View 5 개선: View 4가 학습 방해 요인
- View 5 동일: View 4는 무관, novel view 문제

#### EXP-C: 다른 Input 조합

**목적**: View 4가 input에 포함되면 개선되는지

**Config**: 코드 수정 필요
```python
# mouse_dataset.py 수정
input_indices = [0, 2, 4, 5]  # View 1, 3 → novel
```

**예상 결과**:
- View 4 개선, View 1,3 저하: novel view 위치 문제
- View 4 동일: 데이터 품질 문제

---

## 4. 판단 기준

### 4.1 성공 기준

| 지표 | 기준 | 의미 |
|------|------|------|
| All-view PSNR > 20 | 최소 기준 | 기본 학습 성공 |
| All-view PSNR > 25 | 목표 | 양호한 품질 |
| View 4 PSNR > 20 | 핵심 | View 4 문제 해결 |

### 4.2 결과 해석 플로우

```
EXP-A 실행
    │
    ├─ 모든 뷰 PSNR 25+ → ✅ Novel view 문제 확정
    │                      → H1c, H2 지지
    │                      → 해결: Val random 또는 더 많은 뷰
    │
    └─ View 4 여전히 낮음 → EXP-B 실행
                              │
                              ├─ View 5 개선 → View 4 데이터 문제
                              │               → H1a, H1b 지지
                              │               → 해결: View 4 제외 또는 보정
                              │
                              └─ View 5 동일 → 다른 원인 탐색 필요
```

---

## 5. 우선순위 실행 순서

| 순서 | 실험 | 소요 시간 | 필요 작업 |
|------|------|-----------|-----------|
| **1** | EXP-A (6뷰 overfit) | ~1시간 | Config 준비됨 |
| **2** | 결과 분석 | 10분 | Per-view PSNR 확인 |
| **3** | EXP-B (필요시) | ~1시간 | Config 작성 필요 |

---

## 6. 관련 문서

- [VIEW_SELECTION_GHOSTING_ANALYSIS.md](./VIEW_SELECTION_GHOSTING_ANALYSIS.md)
- [VSCode_Debug_Mask_Guide.md](../tutorials/VSCode_Debug_Mask_Guide.md)

---

*FaceLift View 4 Hypothesis | 2026-01-25*
