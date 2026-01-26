# FaceLift Mouse: 가설 및 실험 계획서

> **SSOT**: 전처리/학습 가설 검증을 위한 체계적 실험 계획
> **최종 업데이트**: 2026-01-25
> **관련 문서**: [[EXPERIMENT_REGISTRY]], [[PREPROCESSING_REGISTRY]], [[QUICK_START]]

---

## 1. 문제 정의

### 1.1 관찰된 현상

| 데이터셋 | Coverage | fx | Val PSNR | 비고 |
|----------|----------|-----|----------|------|
| D3_normalized | 84% | 549 | **27.1** | ⭐ SOTA |
| D7_1 | 50% | 549 | 20.9 | Baseline |
| M3 | 78% | **739** | 10.6 | fx 버그 |

**핵심 질문**: D3_normalized가 왜 좋은가? M3가 왜 나쁜가?

### 1.2 잠재적 원인 목록

| # | 잠재적 원인 | 설명 | 관련 가설 |
|---|-------------|------|-----------|
| 1 | **Coverage 부족** | FG 영역 작음, 배경(흰색) 과다 | H1 |
| 2 | **fx 미정규화** | Pretrained 분포(549)와 불일치 | H2 |
| 3 | **PP 기하학 오류** | cx,cy가 실제 위치와 불일치 | H3 |
| 4 | **Aspect Ratio** | fx/fy 비율 왜곡 | H4 |
| 5 | **크기 불일치** | 샘플별 객체 크기 차이 | H5 |
| 6 | **특정 뷰 품질** | 일부 뷰의 데이터 품질 문제 | H6 |
| 7 | **Mask/Loss 설계** | RGB mask가 학습을 방해 | H7 |
| 8 | **PP 고정 효과** | PP=256 고정이 유리 | H8 |

---

## 2. 가설 (Hypotheses)

### H1: Coverage가 PSNR을 결정한다 (P0)

**가설**: FG Coverage 증가 → Gradient 신호 증가 → PSNR 향상

| 근거 | Dataset | Coverage | PSNR |
|------|---------|----------|------|
| 관찰 | D3_normalized | 84% | 27.1 |
| 관찰 | D7_1 | 50% | 20.9 |

**검증**: M3_norm (78%) vs D7_1 (50%) 비교
**예측**: M3_norm PSNR >= 25

---

### H2: fx=549 정규화는 필수다 ✅ (검증됨)

**가설**: Pretrained 모델이 fx=549 분포를 기대, 다른 값은 학습 실패

| 설정 | fx | PSNR | 결과 |
|------|-----|------|------|
| M3 (버그) | 739 | 10.6 | 실패 |
| D7_1 | 549 | 20.9 | 정상 |

**결론**: **H2 지지됨** - fx 정규화 필수

---

### H3: PP 가변은 허용된다 (P0)

**가설**: GS-LRM이 fxfycxcy 텐서로 가변 PP 처리 가능

| 설정 | PP | 예상 |
|------|-----|------|
| M3_norm | 가변 (std~37) | 정상 학습 |
| D7_1 | 256 고정 | 정상 학습 (확인됨) |

**검증**: M3_norm 실험 결과로 확인
**예측**: M3_norm PSNR이 25+이면 H3 지지

---

### H4: Aspect Ratio 영향은 미미하다 (P3)

**가설**: fx/fy 비율이 1.0이 아니어도 큰 영향 없음

| 설정 | fx/fy | PSNR |
|------|-------|------|
| D3_normalized | 0.9937 | 27.1 |
| D7_1 | 1.0000 | 20.9 |

**우선순위**: P3 (H1, H2, H3 검증 후)

---

### H5: Per-sample Zoom이 일반화에 유리하다 (P1)

**가설**: 샘플별 적응적 zoom이 global zoom보다 나음

| 설정 | Zoom | Coverage |
|------|------|----------|
| M3_norm | Global 1.35x | 78% |
| M3_persample | Per-sample | 50% |

**⚠️ 정정**: M3_persample Coverage는 50% (80% 아님)

---

### H6: 특정 뷰 품질 문제 (P3)

**가설**: 일부 카메라 뷰의 캘리브레이션/품질 문제

**검증 방법**:
- View 갯수 변화 실험 (3v/4v/5v/6v)
- Fixed view 실험으로 특정 뷰 조합 테스트

---

### H7: mask_mode=none이 일반화에 유리 ✅ (검증됨)

**근거**:

| 설정 | mask_mode | Val PSNR | Gap |
|------|-----------|----------|-----|
| **E0_paper** | **none** | **20.9** | +3.7 |
| E1_gt | gt | 17.9 | +10.3 |
| E2_gt_alpha | gt + alpha | 17.0 | +7.8 |

**결론**: GT mask가 과적합을 유도 → 일반화 성능 저하

---

### H8: PP 고정(256)이 일반화에 유리할 수 있음 (P0)

**근거**:

| 설정 | PP | Val PSNR | Gap |
|------|-----|----------|-----|
| D3_normalized | 256 고정 | 27.1 | -3.0 |
| M3_norm | 가변 | TBD | TBD |
| M3_persample | 256 고정 | TBD | TBD |

**검증**: M3_norm vs M3_persample 비교로 확인

---

## 3. 실험 결과 (Results)

### 3.1 D7_1 실험 결과

| 실험 | 설정 | Val PSNR | Step | 상태 |
|------|------|----------|------|------|
| D7_1_E0_paper | Baseline | **20.93** | 4901 | 유효 |
| D7_1_E0_mouse | Mouse-adapted | 19.86 | 3001 | 유효 |
| D7_1_E1_gt | GT mask only | 17.89 | 4901 | 낮음 |
| D7_1_E2_gt_alpha | GT mask + alpha | 16.96 | 2401 | 낮음 |

### 3.2 기타 데이터셋

| 실험 | Val PSNR | Step | 비고 |
|------|----------|------|------|
| D3_normalized_E0_paper | **27.09** | 3001 | ⭐ SOTA |
| D8_E0_paper | 20.21 | 4301 | D7_1과 유사 |
| M3_E1_2_gt_alpha | 10.56 | 301 | fx 버그 확인 |

### 3.3 핵심 발견

1. **Coverage 효과**: D3 (84%) >> D7_1 (50%) → +6 PSNR
2. **fx 정규화 필수**: M3 (fx=739) → PSNR 10.6 (실패)
3. **Mask 효과**: E0 (20.9) > E1 (17.9) > E2 (17.0)

---

## 4. 실험 우선순위

### Phase 0 (P0): SOTA 재현

| 순위 | 데이터셋 | 실험 | 목적 | 상태 |
|------|----------|------|------|------|
| **1** | D3_normalized | E0_1_facelift | SOTA 확인 | ⏳ |
| **2** | D3_normalized | E0_1_1_facelift_alpha | SOTA + alpha | ⏳ |
| 3 | D3_normalized | E0_1_2_facelift_fixed | Fixed view 비교 | ⏳ |
| 4 | D3_normalized | E0_1_3_facelift_alpha_fixed | Alpha + fixed | ⏳ |

### Phase 1 (P1): 가설 검증

| 순위 | 데이터셋 | 실험 | 검증 가설 |
|------|----------|------|-----------|
| 5 | M3_norm | E0_1_facelift | H1, H3 |
| 6 | M3_persample | E0_1_facelift | H5, H8 |

---

## 5. 실험 매트릭스

### E0 계열 (FaceLift 기반)

```
              | E0_1     | E0_1_2_facelift_fixed | E0_1_1_facelift_alpha | E0_1_3_facelift_alpha_fixed |
              | (random) | (fixed)    | (random)   | (fixed)          |
--------------+----------+------------+------------+------------------+
D3_normalized | ⭐ P0    | P0         | P0         | P0               |
D7_1 (M1)     | 20.9     | TBD        | TBD        | TBD              |
M3_norm       | TBD P1   | TBD        | TBD        | TBD              |
M3_persample  | TBD P1   | TBD        | TBD        | TBD              |
```

### E1 계열 (GT Mask 기반) - 참고용

```
              | E1_2_alpha |
--------------+------------+
D7_1          | 17.0       |
M3_norm       | TBD        |
```

---

## 6. E0 실험 설정 비교 ⭐

### 6.1 FaceLift 논문 원본 vs 확장

| 설정 | E0_1_facelift | E0_1_2_facelift_fixed | E0_1_1_facelift_alpha | E0_1_3_facelift_alpha_fixed |
|------|---------------|------------|------------|------------------|
| **mask_mode** | none | none | none | none |
| **alpha_loss** | 0.0 | 0.0 | **0.1** | **0.1** |
| **random_view** | ✅ true | ❌ false | ✅ true | ❌ false |
| **출처** | FaceLift 논문 | +fixed | +alpha | +alpha+fixed |

### 6.2 E0 vs E1 비교

| 설정 | E0 계열 | E1 계열 |
|------|---------|---------|
| **mask_mode** | **none** | gt |
| **alpha_loss** | 0.0 or 0.1 | 0.1 |
| **normalize_by_mask** | false | true |
| **Val PSNR (D7_1)** | **20.9** | 17.0 |

**결론**: **E0 계열 (mask_mode=none) 권장**

### 6.3 Alpha Loss 출처

| 설정 | 출처 | 설명 |
|------|------|------|
| alpha_loss_weight: 0.1 | LGM | Shape convergence 가속화 |
| alpha_loss_type: mse | LGM | MSE for stable gradients |

**참고**: FaceLift 논문 원본에는 alpha loss 없음

---

## 7. 결론 및 다음 단계

### 확정된 결론

1. ✅ **H2 검증됨**: fx=549 정규화 필수
2. ✅ **H7 검증됨**: mask_mode=none이 GT mask보다 우수
3. **Coverage 중요**: D3 (84%) >> D7_1 (50%) → +6 PSNR

### 대기중인 검증

1. **H1**: M3_norm 결과 대기
2. **H3**: M3_norm PP 가변 영향 확인
3. **H8**: PP 고정 vs 가변 비교

### 권장 실행 명령어

```bash
# P0-1: D3_normalized SOTA 확인
CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D3_normalized -e E0_1_facelift \
    > logs/D3_E0_1_facelift.log 2>&1 &

# P0-2: D3 + Alpha
CUDA_VISIBLE_DEVICES=1 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D3_normalized -e E0_1_1_facelift_alpha \
    > logs/D3_E0_1_1_facelift_alpha.log 2>&1 &

# P0-3: D3 + Fixed (random vs fixed 비교)
CUDA_VISIBLE_DEVICES=2 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D3_normalized -e E0_1_2_facelift_fixed \
    > logs/D3_E0_1_2_facelift_fixed.log 2>&1 &

# P0-4: D3 + Alpha + Fixed
CUDA_VISIBLE_DEVICES=3 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D3_normalized -e E0_1_3_facelift_alpha_fixed \
    > logs/D3_E0_1_3_facelift_alpha_fixed.log 2>&1 &
```

---

## 8. 데이터셋 세부 비교

### 8.1 카메라 파라미터 비교

| 항목 | D3_normalized | D7_1 | M3 | M3_norm | M3_persample |
|------|---------------|------|-----|---------|--------------|
| **fx** | 549 | 549 | **739** ❌ | 549 | 549 |
| **fy** | 552.5 | 549 | 739.7 | 549 | 549 |
| **fx/fy** | **0.9937** | 1.0 | 1.0 | 1.0 | 1.0 |
| **cx** | 256 | 256 | 267 | 198 | 256 |
| **cy** | 256 | 256 | 198 | 147 | 256 |
| **cx std** | 0 | 0 | 49 | 37 | 0 |
| **Coverage** | **84%** | 50% | 78% | 78% | 50% |
| **Val PSNR** | **27.1** | 20.9 | 10.6 | TBD | TBD |

### 8.2 D3_normalized 성공 요인

1. **높은 Coverage (84%)**: 유효 gradient 극대화
2. **PP=256 고정**: Pretrained 모델 완벽 호환
3. **fx/fy=0.9937**: 원본 비율 유지

---

## 9. Train-Val Gap 분석

| 실험 | Train | Val | **Gap** | 해석 |
|------|-------|-----|---------|------|
| **D3_normalized_E0** | 24.07 | **27.09** | **-3.0** | ⭐ 우수 일반화 |
| D7_1_E0_paper | 24.58 | 20.93 | +3.7 | ○ 양호 |
| D7_1_E1_gt | 28.21 | 17.89 | +10.3 | ✗ 과적합 |
| M3_E1_2_gt_alpha | 23.95 | 10.56 | +13.4 | ✗ 실패 |

→ 상세: [[TRAIN_VAL_GAP_ANALYSIS]]

---

## 10. 관련 문서

- [[EXPERIMENT_REGISTRY]] - 실험 ID 및 설정 목록
- [[PREPROCESSING_REGISTRY]] - 전처리 방식 상세
- [[QUICK_START]] - 빠른 실행 명령어
- [[TRAIN_VAL_GAP_ANALYSIS]] - Train-Val Gap 상세 분석
- [[DATASET_PREPROCESSING_COMPARISON]] - 데이터셋 전처리 비교

---

*Created: 2026-01-25 | Updated: 2026-01-25 | FaceLift Mouse Project*
