# Masked SSIM 음수 Loss 분석 및 해결

- **날짜**: 2024-12-30
- **주제**: Masked SSIM Loss에서 발생하는 음수 값 문제
- **프로젝트**: Mouse-FaceLift GS-LRM Finetuning

---

## 1. 문제 현상

학습 중 SSIM loss가 음수로 출력되는 현상 발견:

```
step 35: ssim_loss: 0.042   # 정상 (SSIM ≈ 0.958)
step 36: ssim_loss: -0.133  # 비정상! (SSIM ≈ 1.133)
step 37: ssim_loss: 0.048   # 정상
```

**기대 동작**: SSIM ∈ [-1, 1], Loss = 1 - SSIM ∈ [0, 2]
**실제 동작**: SSIM > 1.0 발생 → Loss < 0

---

## 2. 배경: SSIM 공식

### 2.1 SSIM 정의

```
SSIM(x, y) = l(x,y) · c(x,y) · s(x,y)

l(x,y) = (2μₓμᵧ + C₁) / (μₓ² + μᵧ² + C₁)     # Luminance (밝기)
c(x,y) = (2σₓσᵧ + C₂) / (σₓ² + σᵧ² + C₂)     # Contrast (대비)
s(x,y) = (σₓᵧ + C₃) / (σₓσᵧ + C₃)             # Structure (구조)

C₁, C₂, C₃ = 안정성 상수 (0으로 나누기 방지)
```

### 2.2 SSIM Loss

```python
# gslrm/model/utils_losses.py
class SsimLoss(nn.Module):
    def forward(self, x, y):
        return 1.0 - self.ssim_module(x, y)
```

---

## 3. Masked SSIM 구현

### 3.1 마스킹 목적

마우스 이미지 특성:
- **배경**: ~95% (흰색)
- **전경 (마우스)**: ~5%

일반 SSIM은 배경 픽셀이 loss를 지배 → 전경에 집중하기 위해 masking 적용

### 3.2 구현 방식

```python
# gslrm/model/gslrm.py - _compute_ssim_loss()
mask_binary = (mask > 0.5).float()
neutral_value = 0.5  # 배경을 neutral gray로 설정

masked_rendering = rendering * mask_binary + neutral_value * (1 - mask_binary)
masked_target = target * mask_binary + neutral_value * (1 - mask_binary)

return self.ssim_loss_module(masked_rendering, masked_target)
```

### 3.3 결과 이미지 구조

```
┌─────────────────────────────────────┐
│  배경 (95%): 모두 0.5로 동일        │
│  ┌─────────┐                        │
│  │ 마우스  │ ← 전경 (5%): 실제 픽셀 │
│  │  (5%)   │                        │
│  └─────────┘                        │
└─────────────────────────────────────┘
```

---

## 4. 문제 원인 분석

### 4.1 상수 영역에서의 SSIM

배경이 모두 neutral_value = 0.5일 때:

```
μₓ = μᵧ = 0.5
σₓ² = σᵧ² = 0   (모든 픽셀 동일 → 분산 0)
σₓᵧ = 0

l(x,y) = (2 × 0.5 × 0.5 + C₁) / (0.5² + 0.5² + C₁)
       = (0.5 + C₁) / (0.5 + C₁) = 1.0 ✅

c(x,y) = (2 × 0 × 0 + C₂) / (0 + 0 + C₂)
       = C₂ / C₂ = 1.0 ✅

s(x,y) = (0 + C₃) / (0 × 0 + C₃)
       = C₃ / C₃ = 1.0 ✅

이론적 SSIM = 1.0 × 1.0 × 1.0 = 1.0
```

### 4.2 수치 불안정성 발생 메커니즘

```python
# pytorch-msssim 내부 계산 (간략화)
mu1 = F.conv2d(img1, window)  # 로컬 평균
mu1_sq = mu1 * mu1

sigma1_sq = F.conv2d(img1 * img1, window) - mu1_sq  # 로컬 분산

# 문제: 부동소수점 오차
# 상수 이미지(0.5)에서:
# mu1_sq = 0.25000001 (실제 계산값)
# conv(img1*img1) = 0.24999999 (convolution 결과)

sigma1_sq = 0.24999999 - 0.25000001 = -0.00000002  # 음수 분산!
```

**결과 체인:**
```
분산이 아주 작은 음수
    ↓
나눗셈/제곱근 연산에서 불안정
    ↓
최종 SSIM이 1.0을 약간 초과 (예: 1.02, 1.13)
    ↓
Loss = 1.0 - 1.13 = -0.13 (음수!)
```

### 4.3 발생 조건

다음 조건에서 더 자주 발생:
1. 마우스 영역이 매우 작은 샘플
2. 렌더링이 GT와 매우 유사한 경우
3. 배경 비율이 특히 높은 이미지

---

## 5. 해결책

### 5.1 적용된 해결책: SSIM 클램핑

```python
# gslrm/model/utils_losses.py - SsimLoss.forward()

# Before (문제)
return 1.0 - self.ssim_module(x, y)

# After (해결)
ssim_value = self.ssim_module(x, y)
ssim_value = torch.clamp(ssim_value, 0.0, 1.0)  # [0, 1] 범위로 제한
return 1.0 - ssim_value
```

### 5.2 대안적 해결책 비교

| 방법 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **Clamp (적용됨)** | SSIM을 [0,1]로 제한 | 간단, 안정적 | 극단값 정보 손실 |
| 배경 제외 SSIM | 전경 픽셀만으로 SSIM 계산 | 정확한 전경 평가 | 구현 복잡, 작은 영역 불안정 |
| 다른 neutral value | 0.5 대신 다른 값 사용 | - | 근본 해결 안됨 |
| MS-SSIM | 다중 스케일 SSIM | 더 robust | 메모리/계산량 증가 |
| L1/L2로 대체 | SSIM 대신 픽셀 loss | 안정적 | 구조 정보 손실 |

### 5.3 왜 Clamp가 적절한가

1. **이론적 근거**: 실제 SSIM은 [-1, 1] 범위. > 1.0은 수치 오류.
2. **정보 보존**: 대부분의 정상 케이스에는 영향 없음
3. **안정성**: 학습 불안정 (gradient explosion) 방지
4. **간단함**: 한 줄 수정으로 해결

---

## 6. 관련 문제: Scaler 상태 오류

### 6.1 동시 발생 오류

```
RuntimeError: unscale_() has already been called on this optimizer
since the last update().
```

### 6.2 원인

```python
# train_gslrm.py - optimizer_step()

# grad norm이 큰 경우 skip
if total_grad_norm > max_allowed_norm:
    skip_optimizer_step = True

# 문제: skip 시 scaler.update() 미호출
if not skip_optimizer_step:
    self.scaler.step(self.optimizer)
    self.scaler.update()  # ← skip되면 호출 안됨!
```

**다음 iteration에서:**
```python
self.scaler.unscale_(self.optimizer)  # 이미 unscale된 상태에서 재호출 → 오류
```

### 6.3 해결

```python
# scaler.update()는 항상 호출 (step skip 여부와 무관)
if not skip_optimizer_step:
    self.scaler.step(self.optimizer)
    self.param_update_step += 1

# Always update scaler
self.scaler.update()  # ← skip 시에도 호출하여 상태 리셋
```

---

## 7. 핵심 교훈

### 7.1 Masked Loss 사용 시 주의사항

1. **수치 안정성**: 상수 영역이 많으면 통계 기반 메트릭 불안정
2. **범위 확인**: Loss 값이 예상 범위 내인지 항상 검증
3. **클램핑 고려**: 이론적 범위를 벗어나는 경우 클램핑 적용

### 7.2 AMP (Mixed Precision) 사용 시 주의사항

1. **Scaler 상태 관리**: `update()`는 step skip 시에도 호출 필요
2. **Gradient 검사**: NaN/Inf gradient 처리 로직 필수
3. **문서 참조**: PyTorch AMP 공식 가이드라인 준수

### 7.3 디버깅 팁

```python
# Loss 모니터링 추가
if ssim_loss < 0:
    print(f"WARNING: Negative SSIM loss {ssim_loss}, raw SSIM: {ssim_value}")

# Gradient 상태 확인
if hasattr(self.scaler, '_scale'):
    print(f"Scaler state: scale={self.scaler._scale}, growth_tracker={self.scaler._growth_tracker}")
```

---

## 8. 변경된 파일

| 파일 | 변경 내용 |
|------|----------|
| `gslrm/model/utils_losses.py` | SSIM 클램핑 추가 (line 375-379) |
| `train_gslrm.py` | scaler.update() 항상 호출 (line 688-690) |

---

## 참고 자료

- [SSIM 원 논문](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf)
- [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html)
- [pytorch-msssim GitHub](https://github.com/VainF/pytorch-msssim)
