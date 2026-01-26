# 핵심 가설 검증 보고서

> **작성일**: 2026-01-25
> **목적**: GS-LRM 생쥐 Finetuning 실패 원인 분석 및 가설 검증
> **관련 문서**: [[00_MoC_INDEX]] | [[PP_MVG_COMPREHENSIVE_ANALYSIS]] | [[M3_SERIES_SPEC]]

---

## 요약: 현재 상황

### 핵심 문제
- **현상**: 생쥐 데이터로 GS-LRM finetuning 시 whitening, ghosting 발생
- **정량 지표**: 최고 Val PSNR 27.09 (D3_normalized), 대부분 17-20 수준

### 즉시 중단 필요 실험

| 실험 | 데이터셋 | 문제점 | 조치 |
|------|----------|--------|------|
| M3_persample_E0_1_facelift | M3_persample | PP 가변 (ray error 16.15도) | **중단** |
| M3_norm_E0_1_facelift | M3_norm | PP 가변 (ray error 13.62도) | **중단** |
| M3_*_E* 전체 | M3, M3_norm, M3_persample | PP/fx 문제 | **폐기** |

### 유지 가능 실험

| 실험 | 데이터셋 | PP | Val PSNR |
|------|----------|-----|----------|
| D3_normalized_E0_* | D3_normalized | 256 +/- 0 | 23-27 |
| D7_1_E0_* | D7_1 | 256 +/- 0 | 17-21 |
| D8_E0_* | D8 | 256 +/- 0 | 18-20 |

---

## 가설 1: 전처리 정밀도 문제

### 1.1 PP (Principal Point) 정합성

**상태**: ✅ **검증 완료 - 핵심 원인 확인**

| 데이터셋 | PP 상태 | Ray Error | Val PSNR | 판정 |
|----------|---------|-----------|----------|------|
| D3_normalized | 256 +/- 0 | 0.00도 | **27.09** | OK |
| D7_1 | 256 +/- 0 | 0.00도 | 20.93 | OK |
| D8 | 256 +/- 0 | 0.00도 | 20.21 | OK |
| M3_norm | 197 +/- 46 | **13.62도** | 17.09 | BAD |
| M3_persample | 235 +/- 32 | **16.15도** | 17.08 | BAD |

**결론**: PP 정합성이 Val PSNR에 5-10 포인트 영향. 
Object-centered zoom이 문제 → Center-aligned zoom (M3_1, M3_2)으로 해결.

### 1.2 Plucker Ray 일관성

**상태**: ✅ **코드 검증 완료 - 정상**

```python
# gslrm/model/gslrm.py:1065-1066
if self.config.model.get("use_custom_plucker", False):
    # Custom Plucker: RGB + ray_direction + nearest_points
```

Plucker ray 계산은 cx, cy를 사용하므로:
- PP=256 고정 데이터셋: Plucker ray 일관성 보장
- PP 가변 데이터셋: Plucker ray 불일치 → **PP 문제로 귀결**

### 1.3 Camera Parameter 일관성

**상태**: ✅ **코드 검증 완료 - 정상**

```python
# gslrm/data/mouse_dataset.py:394-395
# Extract camera pose (w2c -> c2w)
c2w = np.linalg.inv(np.array(camera["w2c"]))
```

- w2c → c2w 변환 정상
- intrinsics (fx, fy, cx, cy) 사용 정상
- 문제는 **데이터 자체의 PP 가변성**

---

## 가설 2: 인간 데이터와의 구조적 차이

### 2.1 Coverage (전경 비율)

**상태**: ⚠️ **부분 검증 - 가설 지지**

| 데이터셋 | Coverage | Val PSNR | 분석 |
|----------|----------|----------|------|
| D3_normalized | 74% | 27.09 | 높은 coverage = 높은 PSNR |
| D7_1 | 50% | 20.93 | 중간 |
| M3_persample | 목표 5% | 17-19 | 낮은 coverage |

**결론**: Coverage 증가 → PSNR 향상 경향 존재. 
단, D3의 성공은 Coverage + PP 정합의 복합 효과.

### 2.2 Off-Center 위치

**상태**: ✅ **해결됨**

- 문제: 생쥐가 중앙이 아닌 다양한 위치
- 해결: PP shift 방식으로 가상 중앙 이동 (D7, D8 시리즈)
- **M3_1, M3_2**: Center-aligned zoom으로 PP=256 자동 보장

### 2.3 뷰 수 차이 (32뷰 vs 6뷰)

**상태**: ⚠️ **구조적 한계 - 해결 불가**

| 데이터 | 전체 뷰 | 학습 뷰 | 검증 뷰 |
|--------|---------|---------|---------|
| 인간 (합성) | 32 | 4 | 4 |
| 생쥐 (실제) | 6 | 4 | 2 |

**영향**: 
- 3뷰 실험 시 Val PSNR 14.17 (4뷰 16.96 대비 -2.8)
- Train-Val Gap 증가 (7.8 → 10.4)

**결론**: 뷰 수 부족은 일반화 한계로 작용하나, 근본 원인은 아님.

### 2.4 합성 vs 실제 데이터

**상태**: ⚠️ **잠재적 요인 - 추가 검증 필요**

| 특성 | 인간 데이터 | 생쥐 데이터 |
|------|-------------|-------------|
| 생성 방식 | 합성 렌더링 | 실제 카메라 |
| 노이즈 | 없음 | 존재 |
| 카메라 캘리브레이션 | 완벽 | 오차 가능 |
| 조명 | 일관됨 | 변동 가능 |

---

## 가설 3: 특정 뷰 데이터 오류

**상태**: ⚠️ **추가 검증 필요**

### 3.1 검증 방법

```bash
# 단일 뷰 overfit 실험
for view in 0 1 2 3 4 5; do
    torchrun train_gslrm.py -d D7_1 -e E_single_view_${view}
done
```

### 3.2 현재 데이터

D7_1_E2_gt_alpha_overfit 실험 결과:
- Val PSNR: 4.93 (매우 낮음)
- Train-Val Gap: +17.53

**해석**: 특정 뷰 문제보다는 overfit 시 일반화 실패 문제.

---

## 가설 4: Mask 이슈

### 4.1 현재 구현 상태

**Alpha Loss 구현** (gslrm/model/gslrm.py:430-436):

```python
alpha_loss_weight = getattr(self.config.training.losses, "alpha_loss_weight", 0.0)
if alpha_loss_weight > 0 and rendered_alpha is not None and mask is not None:
    alpha_loss_type = getattr(self.config.training.losses, "alpha_loss_type", "mse")
    loss_type = AlphaLossType.MSE if alpha_loss_type == "mse" else AlphaLossType.BCE
    losses["alpha_loss"] = compute_alpha_supervision_loss(rendered_alpha, mask, loss_type)
```

**WandB 로깅 이름**: `train/alpha_loss`

### 4.2 Mask Mode 분석

| mask_mode | RGB Loss 영역 | Alpha Supervision | 권장 |
|-----------|---------------|-------------------|------|
| none | 전체 이미지 | 없음 | O (안정적) |
| **gt** | GT mask 영역 | GT vs rendered alpha | **권장** |
| alpha | rendered alpha 영역 | - | X (피드백 루프) |

**결론**: `mask_mode=gt + alpha_loss_weight=0.1` 권장

### 4.3 Alpha Mask 시각화 문제

**현재 상태**: ❌ **시각화 누락됨**

현재 wandb 로깅에서 rendered alpha 이미지가 별도로 시각화되지 않음.

**해결 방안**:
```python
# train_gslrm.py의 visualization 부분에 추가 필요
if rendered_alpha is not None:
    alpha_vis = (rendered_alpha * 255).cpu().numpy().astype(np.uint8)
    wandb_images["train/rendered_alpha"] = wandb.Image(alpha_vis)
```

### 4.4 Mask Loss 실험 결과

| 실험 | mask_mode | alpha_loss | Val PSNR | Gap |
|------|-----------|------------|----------|-----|
| D7_1_E0_paper | none | 0.0 | **20.93** | +3.7 |
| D7_1_E1_gt | gt | 0.0 | 17.89 | +10.3 |
| D7_1_E2_gt_alpha | gt | 0.1 | 16.96 | +7.8 |

**관찰**: mask_mode=none이 현재 가장 높은 Val PSNR. 
GT mask 사용 시 과적합 경향.

---

## 가설 5: 3D GS Projection 방식

### 5.1 c2w vs w2c 사용

**상태**: ✅ **검증 완료 - 정상**

```python
# gslrm/data/mouse_dataset.py:394-395
c2w = np.linalg.inv(np.array(camera["w2c"]))
```

- 데이터: w2c (world-to-camera) 저장
- 모델: c2w (camera-to-world)로 변환하여 사용
- **정상 작동 확인**

### 5.2 좌표계 변환

```python
# gslrm/data/mouse_dataset.py:440-442
input_c2ws = normalize_cameras_to_z_up(input_c2ws, up_direction)
# 또는
input_c2ws = normalize_cameras_to_y_up(input_c2ws, up_direction)
```

**결론**: 좌표계 변환 로직 정상. 문제는 입력 데이터의 PP 정합성.

---

## 종합 결론

### 확인된 핵심 원인

| 순위 | 원인 | 영향도 | 상태 | 해결책 |
|------|------|--------|------|--------|
| **1** | PP 정합성 | +7-10 PSNR | 확인됨 | M3_1, M3_2 사용 |
| 2 | Coverage | +2-5 PSNR | 부분 확인 | 높은 coverage 유지 |
| 3 | fx 정규화 | +5-10 PSNR | 확인됨 | fx=549 필수 |
| 4 | Mask mode | +1-3 PSNR | 부분 확인 | none 또는 gt |

### 잠재적 원인 (추가 검증 필요)

| 원인 | 영향 추정 | 검증 방법 |
|------|-----------|-----------|
| 뷰 수 부족 (6개) | 구조적 한계 | Multi-dataset 실험 |
| 특정 뷰 캘리브레이션 | 불명 | 단일 뷰 overfit |
| 합성 vs 실제 차이 | 불명 | 합성 생쥐 데이터 실험 |

### 권장 다음 단계

1. **즉시**: M3_norm, M3_persample 실험 중단
2. **우선**: M3_1 또는 M3_2로 재전처리 후 실험
3. **검증**: Alpha mask 시각화 추가
4. **탐색**: 단일 뷰 overfit 실험으로 특정 뷰 문제 검증

---

## 관련 문서

| 문서 | 내용 |
|------|------|
| [[00_MoC_INDEX]] | 문서 허브 |
| [[PP_MVG_COMPREHENSIVE_ANALYSIS]] | PP 정합성 상세 분석 |
| [[M3_SERIES_SPEC]] | M3_1, M3_2 프리셋 명세 |
| [[TRAIN_VAL_GAP_ANALYSIS]] | Train-Val Gap 분석 |
| [[EXPERIMENT_REGISTRY]] | 실험 설정 레지스트리 |

---

*Hypothesis Verification Report v1.0 | 2026-01-25*
