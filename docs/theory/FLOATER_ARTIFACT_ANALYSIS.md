> **Navigation**: [← MoC](../../00_MoC_INDEX.md) | [Theory](../) | [Architecture](./GS-LRM_ARCHITECTURE_GUIDE.md)

# Floater Artifact Analysis

> **Version**: 1.0
> **Date**: 2026-01-23
> "두 마리 쥐" 현상 (Ghosting/Floater) 원인 분석 및 해결책

---

## 1. 현상 설명

### 1.1 증상

| 증상 | 설명 |
|------|------|
| **Ghosting** | 같은 객체가 다른 깊이에 복제되어 보임 |
| **Floater** | 공중에 떠있는 불필요한 Gaussian 점들 |
| **Blur** | 경계 영역이 흐릿하게 렌더링됨 |

### 1.2 문헌 정의

> "Floaters arise from a coupling in the optimization process that leads to local minima... when both opacity (α) and color (c) parameters are optimized simultaneously, the color optimization often takes precedence, leading to the persistence of floaters."
> — [StableGS, arXiv:2503.18458](https://arxiv.org/html/2503.18458)

> "In sparse-view settings, the fundamental ill-posedness emerges from a lack of geometric coverage—many scene regions are unobserved, leading to ambiguities in color, depth, and structure assignments."
> — [Gaussian Splatting Survey](https://link.springer.com/article/10.1007/s10462-025-11171-4)

---

## 2. 원인 분석 (3가지)

### 2.1 Depth Ambiguity (깊이 모호성)

```
문제: Feed-forward 모델이 깊이를 잘못 추정
     → 같은 객체가 다른 거리에 복제됨

원인:
- Sparse view (6개 뷰)에서 깊이 정보 부족
- Multi-view consistency 부재
- 모호한 영역에서 임의 깊이 할당
```

**영향**: 가장 심각한 Ghosting 원인

### 2.2 Opacity-Color Coupling (투명도-색상 커플링)

```
문제: 최적화 시 color가 먼저 수렴
     → opacity는 local minima에 갇힘

메커니즘:
1. Color loss 빠르게 감소 (쉬운 최적화)
2. Opacity는 0 또는 1로 수렴 못함
3. 반투명 Gaussian이 여러 위치에 남음
```

**영향**: 흐릿한 경계, 반투명 artifact

### 2.3 Training-Turntable View Mismatch

```
현재 설정:
┌───────────┬───────────────┬────────────────┐
│   항목    │   Training    │   Turntable    │
├───────────┼───────────────┼────────────────┤
│ Elevation │ 10.7° ~ 30.8° │ 0° (hardcoded) │
├───────────┼───────────────┼────────────────┤
│ Radius    │ 데이터 의존   │ 2.7 (고정)     │
└───────────┴───────────────┴────────────────┘

문제: Turntable elevation(0°)이 학습 범위 밖
     → Novel view 외삽 시 artifact 심화
```

**영향**: Turntable 렌더링에서만 발생하는 artifact

---

## 3. 해결책

### 3.1 즉시 적용: Turntable Elevation 조정

**현재 코드** (`gaussians_renderer.py:978`):
```python
elevation=0,  # For MAX SNEAK ← 하드코딩됨
```

**수정안**:
```python
# 방법 1: 고정값 변경
elevation=20,  # 학습 범위 중간값 (10.7°~30.8°)

# 방법 2: Config에서 조절
visualization:
  turntable_elevation: 20
```

**난이도**: 하 | **효과**: 높음 (Turntable artifact 해결)

### 3.2 단기: Opacity Regularization

**목표**: Opacity를 0 또는 1에 가깝게 유도 (binary 강제)

```python
# Entropy-based regularization
opacity_reg = -torch.mean(
    opacity * torch.log(opacity + 1e-8) +
    (1 - opacity) * torch.log(1 - opacity + 1e-8)
)

# L1-sparse regularization (0에 가깝게)
opacity_reg = torch.mean(torch.abs(opacity))

# Binary regularization (0 또는 1에 가깝게)
opacity_reg = torch.mean(opacity * (1 - opacity))
```

**Config 설정**:
```yaml
training:
  losses:
    opacity_reg_weight: 0.01  # 0.01 ~ 0.05
    opacity_reg_type: entropy  # entropy | l1_sparse | binary
```

**난이도**: 하 | **효과**: 중 (반투명 artifact 감소)

### 3.3 중기: Depth Regularization

**목표**: Monocular depth prior로 깊이 일관성 유도

```python
# Pseudo GT depth from pretrained model (MiDaS, DPT)
pseudo_depth = depth_model(input_image)

# Depth loss
depth_loss = F.l1_loss(rendered_depth, pseudo_depth)
```

**참고 논문**:
- DNGaussian: Depth-regularized 3DGS
- DepthRegularizedGS

**난이도**: 중 (pretrained depth model 필요) | **효과**: 높음

### 3.4 고급: Dual Opacity 구조

> StableGS의 Dual Opacity GS 아키텍처

**개념**: Opacity를 두 단계로 분리
1. **Global opacity**: 전체 장면에서의 존재 여부
2. **Local opacity**: 특정 뷰에서의 가시성

**구현**: 모델 아키텍처 변경 필요

**난이도**: 상 | **효과**: 매우 높음

---

## 4. 권장 우선순위

| 순위 | 해결책 | 난이도 | 예상 효과 | 적용 시점 |
|------|--------|--------|----------|----------|
| **1** | Turntable elevation 20° | 하 | 높음 | 즉시 |
| **2** | Opacity regularization | 하 | 중 | 단기 |
| **3** | Depth regularization | 중 | 높음 | 중기 |
| **4** | Dual Opacity 구조 | 상 | 매우 높음 | 장기 |

---

## 5. 실험 검증 방법

### 5.1 Turntable Elevation 테스트

```bash
# 현재 (0°) vs 수정 (20°) 비교
python -m mouse_extensions.scripts.analysis.compare_turntable \
    --checkpoint ckpt.pt \
    --elevations 0 10 20 30
```

### 5.2 Opacity Distribution 분석

```python
# Opacity histogram 확인
import torch
import matplotlib.pyplot as plt

gaussians = model.get_gaussians()
opacities = gaussians['opacity'].detach().cpu()

plt.hist(opacities.numpy(), bins=50)
plt.xlabel('Opacity')
plt.title('Opacity Distribution')
# 이상적: 0과 1 근처에 집중 (bimodal)
# 문제: 중간값에 분포 (unimodal)
```

### 5.3 Depth Consistency Check

```python
# 여러 뷰에서 동일 3D 점의 depth 일관성 확인
for view in views:
    rendered_depth = render(gaussians, view)
    # depth 분산이 크면 inconsistency
```

---

## 6. 관련 문서

| 문서 | 연관성 |
|------|--------|
| [GS-LRM_ARCHITECTURE_GUIDE](./GS-LRM_ARCHITECTURE_GUIDE.md) | 픽셀당 Gaussian 설계의 한계 |
| [ALPHA_MASK_COMPLETE_GUIDE](../mask/ALPHA_MASK_COMPLETE_GUIDE.md) | Opacity regularization 설정 |
| [GS-LRM_Loss_Formula](../loss/GS-LRM_Loss_Formula.md) | Loss 가중치 조절 |
| [METRICS_GUIDE](../metrics/METRICS_GUIDE.md) | Floater 진단 지표 |

---

## 7. 참고 문헌

1. StableGS: Stabilizing 3D Gaussian Splatting, arXiv:2503.18458
2. Gaussian Splatting: A Comprehensive Survey, Springer 2025
3. DNGaussian: Depth-regularized Gaussian Splatting
4. GitHub Issue: [Floater Removal Discussion](https://github.com/graphdeco-inria/gaussian-splatting/issues/852)

---

*Updated: 2026-01-23*
