# 260203 Research Notes

> **Date**: 2026-02-03
> **Topics**: Deformation Network, MVDiffusion CFG Ablation

---

## 1. Deformation Network 분석

### 1.1 논문 분석 결과

FaceLift 논문 Appendix 3.5 "Applying FaceLift on Videos"의 Deformation Network:
- **목적**: 프레임별 독립 재구성의 temporal flickering 해결
- **구조**: 8-layer MLP
- **방식**: Autoregressive generation (이전 → 다음 프레임 예측)

### 1.2 현재 코드베이스 상태

| 항목 | 상태 | 비고 |
|------|------|------|
| Deformation Network | ❌ 미구현 | 논문의 확장 기능, 코드 미공개 |
| 8-layer MLP | ❌ 없음 | utils_transformer.py의 MLP는 Transformer용 |
| Autoregressive | ❌ 없음 | 현재 프레임별 독립 처리 |
| Canonical Gaussians | ❌ 없음 | 앵커 프레임 개념 없음 |

**결론**: Deformation Network는 구현되어 있지 않음 → mouse_extensions에 새로 구현 필요

### 1.3 Deformation Network 아키텍처

```
Input: Gaussian positions (x, y, z) from Gₜ [N_gaussians, 3]
       ↓
Architecture: 8-layer MLP
       ├─ Linear(3 → hidden_dim)
       ├─ ReLU × 6 layers
       └─ Linear(hidden → output_dim)
       ↓
Output: Deformation parameters
       ├─ Δx, Δy, Δz (position offset)
       ├─ Δα (opacity change)
       └─ Δs (scale change)
```

### 1.4 Autoregressive Pipeline

```
Step 1: 초기 Gaussian 생성 (FaceLift)
        F₀ → G₀, F₁ → G₁, ..., Fₜ → Gₜ

Step 2: Anchor Frame 선택
        G₀ = Canonical Gaussians (기준)

Step 3: Autoregressive Deformation
        G₀ + D₀(G₀) → G\'₁
        G\'ₜ₋₁ + Dₜ₋₁(G\'ₜ₋₁) → G\'ₜ

결과: 시간적으로 연속적인 {G\'₀, G\'₁, ..., G\'ₜ}
```

---

## 2. MVDiffusion CFG Dropout & Attention Ablation

### 2.1 문제 제기

M5t2_consistent 모델이 기존 M5t baseline보다 안 좋아 보임.
변경된 설정이 품질 저하 유발 가능성 조사.

### 2.2 변경 이력

| 버전 | 주요 변경 |
|------|-----------|
| M5t | Baseline (original prompt, 1to1 split) |
| M5t_mp | Mouse prompt 적용 |
| M5t2 | t2 split (80:10:10) |
| M5t2_consistent | CFG dropout 제거, Full attention |

### 2.3 가설

**H1: CFG Dropout 제거가 품질 저하의 원인**
| 설정 | M5t/M5t2 | M5t2_consistent |
|------|----------|-----------------|
| condition_drop_rate | 0.05 | 0.0 |

- drop_rate=0.05: 학습 중 5% 확률로 conditioning 드롭 → unconditional 학습
- drop_rate=0.0: unconditional 학습 없음 → guidance 효과 미미

**H2: Full Attention이 과적합 유발**
| 설정 | M5t/M5t2 | M5t2_consistent |
|------|----------|-----------------|
| sparse_mv_attention | true | false |

- Sparse: 인접 뷰 간만 attention (regularization 효과)
- Full: 모든 뷰 쌍 간 attention → 과적합 위험

### 2.4 실험 설계

**통제 변수**: GS-LRM 고정, 테스트 데이터 고정
**비교 조건**:
| 실험 | MVDiffusion | CFG | Sparse |
|------|-------------|-----|--------|
| Ctrl | M5t/ckpt-8000 | 0.05 | true |
| Exp-A | M5t2/ckpt-5000 | 0.05 | true |
| Exp-B | M5t2_consistent/ckpt-6000 | 0.0 | false |

---

## References

- FaceLift Paper Appendix 3.5
- Ho & Salimans (2022). Classifier-Free Diffusion Guidance
- Shi et al. (2023). MVDiffusion

---

*FaceLift Research Notes | 2026-02-03*
