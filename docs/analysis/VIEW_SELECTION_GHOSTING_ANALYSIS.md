# View Selection & Ghosting 분석 보고서

**작성일**: 2026-01-25  
**실험**: D3_normalized_E0_1_facelift, E0_1_3_facelift_alpha_fixed

---

## 1. 문제 현상

특정 뷰에서 지속적인 품질 저하:
- 생쥐 2마리가 겹쳐 보임 (다른 꼬리 방향)
- 몸통 일부가 하얗게 잘림
- Train/Val PSNR이 좋음에도 특정 뷰에서 심각한 artifact

### Per-View PSNR 분석 (D3_normalized, 6000 step)

| View | PSNR | 상태 |
|------|------|------|
| 0 | 31.6 | ✅ 우수 |
| 1 | 27.8 | ✅ 양호 |
| 2 | 32.8 | ✅ 우수 |
| 3 | 31.3 | ✅ 우수 |
| **4** | **10.8** | 🔴 매우 나쁨 |
| **5** | **12.8** | 🔴 나쁨 |

---

## 2. 원인 분석

### 2.1 원본 FaceLift/GS-LRM 방식 (합성 인간 얼굴)

```
전체 뷰: 32개 (Objaverse 렌더링)
     ↓
샘플링: 8개 뷰 (random)
     ↓
분할: 4개 input + 4개 target
     ↓
target_has_input: true → target에 input 뷰 포함 가능
```

**핵심**: 
- 32개 중 8개 랜덤 샘플링 → 다양한 뷰 조합 학습
- `target_has_input: true` → 평가 시 본 적 있는 뷰 포함
- 결과: **Novel view 일반화 압력 완화**

### 2.2 현재 Mouse 데이터 방식 (6개 뷰 고정)

```
전체 뷰: 6개 (고정 카메라)
     ↓
Training: 4개 input (random from 6)
     ↓
Validation: [0,1,2,3] 고정 input, [4,5] 항상 novel target
```

**문제점**:
- 6개 뷰 밖에 없어 선택지 제한
- Validation 시 input이 [0,1,2,3]으로 고정
- **View 4, 5는 validation에서 항상 "한 번도 본 적 없는" novel view**

### 2.3 코드 확인

```python
# mouse_dataset.py:276-279
if random_view_selection and split == "train":
    input_indices = sorted(random.sample(all_indices, 4))  # 랜덤
else:
    input_indices = list(range(4))  # [0,1,2,3] 고정 ← 문제!
```

---

## 3. 결론

**"Ghosting"이 아니라 "Novel View Synthesis 한계"**

- View 0-3: Training/Validation 모두에서 input으로 사용 → 잘 학습됨
- View 4-5: Training에서 가끔 input, Validation에서 **항상 novel** → 일반화 실패

---

## 4. 해결 방안

### Option A: 6개 뷰 모두 학습 (Overfitting 허용)

```yaml
model:
  num_views: 6
  num_input_views: 6  # 모든 뷰를 input으로

training:
  dataset:
    random_view_selection: false  # 고정
    target_has_input: true        # target도 동일 뷰
```

**장점**: 모든 뷰에서 재구성 품질 확인 가능  
**단점**: Novel view 일반화 능력 평가 불가 (overfitting)

### Option B: 5개 input + 1개 target (타협안)

```yaml
model:
  num_views: 6
  num_input_views: 5

training:
  dataset:
    random_view_selection: true  # 매번 다른 1개가 novel
```

**장점**: 약간의 일반화 압력 유지  
**단점**: 여전히 1개 뷰는 novel

### Option C: Validation도 random input (권장)

```python
# mouse_dataset.py 수정
# Before:
if random_view_selection and split == "train":
    input_indices = sorted(random.sample(all_indices, 4))
else:
    input_indices = list(range(4))  # [0,1,2,3]

# After:
if random_view_selection:
    input_indices = sorted(random.sample(all_indices, num_input_views))
else:
    input_indices = list(range(num_input_views))
```

**장점**: Train/Val 일관성, 공정한 평가  
**단점**: Val 재현성 감소 (랜덤)

---

## 5. 즉시 실행 권장: Option A

**목표**: "제대로 학습되는지 먼저 확인" (Overfitting OK)

```yaml
# E_overfit_all_views.yaml
model:
  num_views: 6
  num_input_views: 6

training:
  dataset:
    random_view_selection: false
    target_has_input: true
```

**예상 결과**:
- 모든 6개 뷰에서 PSNR 25+ 달성
- Ghosting 현상 해소 (모든 뷰가 input이므로)
- 실제 일반화 능력은 별도 평가 필요

---

## 6. 향후 과제

1. **6개 뷰 학습 확인 후** → Option B/C로 일반화 능력 테스트
2. **추가 카메라 확보** 가능하면 → 더 많은 뷰로 실험
3. **Temporal consistency** → 동일 시점 프레임 정렬 확인

---

*FaceLift Mouse Extension Analysis | 2026-01-25*
