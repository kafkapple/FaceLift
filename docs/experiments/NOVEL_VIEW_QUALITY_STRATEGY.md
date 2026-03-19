# Novel View Rendering Quality Improvement Strategy

> Multi-model deliberation (Claude + Gemini Pro + GPT-4o) 기반 종합 전략
> Created: 2026-03-19

---

## 1. Problem Statement

GS-LRM 6-view 모델의 **extrapolated novel view** (특히 bottom view)에서 **needle/pancake-like white artifact**가 발생.

**Root Cause**: Gaussian covariance의 effective rank가 1로 수렴 (한 eigenvalue만 지배적)
- Training view에서는 정상 렌더링
- Extrapolated view에서 white streak으로 보임
- 6v baseline: anisotropy ratio 71,833 (mean), 328,469 (p95)

**Impact**: MVDiff E2E pipeline에서 -15.64 dB 성능 하락의 주요 원인 중 하나

---

## 2. Strategy Ranking (3-Model Consensus)

### Tier 1: 즉시 실행 (1-2일, 높은 ROI)

| # | 전략 | 구현 상태 | 기대 효과 | 타겟 |
|:-:|------|:---------:|:---------:|------|
| 1 | **Effective Rank Regularization** | ✅ 구현 완료 | 높음 | Needle/pancake 직접 제거 |
| 2 | **Opacity Regularization** (entropy) | ✅ 구현 완료 | 중간 | Floater/semi-transparent 제거 |
| 3 | **Difix3D+ zero-shot** | ⬜ PoC 데이터 준비됨 | 불확실 | 2D 렌더링 후처리 |

### Tier 2: 단기 (3-7일)

| # | 전략 | 구현 상태 | 기대 효과 |
|:-:|------|:---------:|:---------:|
| 4 | **Difix3D+ fine-tuning** | ⬜ 빌더 스크립트 완성 | 높음 |
| 5 | **Depth Regularization** (Marigold/DPT) | ⬜ 구조만 구현 | 중간-높음 |
| 6 | **Gaussian Pruning** (anisotropy-based) | ⬜ placeholder | 중간 |

### Tier 3: 중장기 (2주+, NeurIPS 이후)

| # | 전략 | 기대 효과 | 비고 |
|:-:|------|:---------:|------|
| 7 | **2DGS/SurfSplat 전환** | 매우 높음 | 근본 해결, pretrained 재훈련 |
| 8 | **PM-Loss** (pointmap-based) | 높음 | Feed-forward 특화, 3DV 2026 |

### 비권장 (GS-LRM 비호환)

- Scaffold-GS, GOF, Mini-Splatting, ReconFusion, SDS

---

## 3. ERR Implementation Details

### Loss Function

```
erank = exp(H(p))
where p_i = softmax(2 * log_scale)_i  (numerically stable)
      H(p) = -Σ p_i * log(p_i + eps)
Loss = mean(target_rank - erank)  # target_rank = 3.0 (sphere)
```

### Config

```yaml
training:
  losses:
    effective_rank_weight: 0.01  # sweep: 0.001, 0.01, 0.1
    effective_rank_target: 3.0
```

### 파일

| 파일 | 내용 |
|------|------|
| `mouse_extensions/model/mask_losses.py` | `compute_effective_rank_loss()` |
| `gslrm/model/gslrm.py` | Training loop 연결 |
| `configs/experiments/6view_alpha05_err.yaml` | 실험 config |

---

## 4. Alpha Loss Details

### 공식

```
L_alpha = MSE(rendered_alpha, gt_mask)  # default, LGM 방식
```

4가지 loss type: MSE (기본), BCE, Dice, Focal

### 핵심 특성

- **Mask**: Soft (continuous [0,1]), GT RGBA alpha channel 그대로 사용
- **mask_mode 독립**: RGB loss의 mask_mode와 무관
- **Clamping**: rendered_alpha.clamp(1e-6, 1-1e-6)

---

## 5. Difix3D+ Status

### 이미 준비된 것

| 구성 요소 | 경로 | 상태 |
|-----------|------|:----:|
| Training Strategy | `docs/experiments/DIFIX_TRAINING_STRATEGY.md` | PLANNING |
| Dataset Builder | `mouse_extensions/scripts/eval/build_difix_dataset.py` | 완성 |
| PoC 이미지 | `outputs/poc_mesh_gs_pairs/` (3프레임) | 수집 완료 |
| 모델 가중치 | `nvidia/difix` (HuggingFace) | 공개 |

### 3종 데이터 페어

| Type | Input (Degraded) | Target (Clean) | 규모 |
|:----:|-------------------|-----------------|:----:|
| 1 | N-view GS-LRM @ GT camera | Original GT RGB | 108K |
| 2 | 6-view GS-LRM @ novel camera | MAMMAL mesh render | 14.4K |
| 3 | N-view GS-LRM @ GT camera | 6-view GS-LRM | 108K |

### Next Action

1. Zero-shot 테스트 (30분) → feasibility 확인
2. 실패 시 fine-tuning Stage 1 (Type 3, self-supervised, ~8시간)

---

## 6. 2DGS Transition Roadmap

### Phase 1: ERR로 Pancake 억제 (현재)

- Effective Rank Reg로 erank → 2+ 유도
- 3DGS 프레임워크 내에서 최대 품질 확보

### Phase 2: 2DGS 탐색 (NeurIPS 이후)

| 단계 | 작업 | 예상 기간 |
|------|------|:---------:|
| 2.1 | gsplat 설치 + `rasterization_2dgs` API 테스트 | 0.5일 |
| 2.2 | `to_gs()` scale 차원 변경 (3→2) | 0.5일 |
| 2.3 | Rasterizer 교체 (`diff_gauss` → `gsplat`) | 2-3일 |
| 2.4 | Depth distortion + Normal consistency loss | 1-2일 |
| 2.5 | 학습 + 비교 실험 | 2-3일 |

### Key References

- [2DGS (SIGGRAPH 2024)](https://arxiv.org/abs/2403.17888)
- [SurfSplat (2025)](https://arxiv.org/abs/2602.02000) — feedforward 2DGS
- [ERR (NeurIPS 2024)](https://arxiv.org/abs/2406.11672)
- [Difix3D+ (CVPR 2025 Oral)](https://arxiv.org/abs/2503.01774)

---

## 7. Deliberation Summary

### 합의 사항 (3개 모델)

1. ERR이 최우선 — 직접적으로 needle artifact 타겟
2. 2DGS는 NeurIPS submission 이후
3. Difix zero-shot 빠른 테스트 먼저
4. GS-LRM 개선에 집중 (MVDiff 교체보다)

### 미합의 (논쟁 사항)

| 쟁점 | Gemini | Claude | GPT |
|------|--------|--------|-----|
| GS-LRM vs MVDiff 집중 | 100% GS-LRM | GS-LRM 우선 | 병행 |
| Depth Prior 포함 여부 | 핵심 기여 | Tier 2 | 미언급 |
| Difix zero-shot 기대 | 실패 예상 | 열어둠 | 조건부 |

### 채택 전략

**Gemini 서사 구조 + Claude 실행 순서**:
- "Geometrically-Aware Priors for Feed-Forward 3DGS" 프레임
- ERR sweep → Opacity Reg → Difix PoC → Depth Prior → Full ablation

---

*Created: 2026-03-19 | Multi-model deliberation (Claude Opus + Gemini Pro + GPT-4o)*
