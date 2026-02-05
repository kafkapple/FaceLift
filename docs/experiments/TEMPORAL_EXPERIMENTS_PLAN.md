# Temporal Consistency Experiments Plan

**Last Updated**: 2026-02-04  
**Status**: Active

---

## Current Status

### Completed ✅
- [x] Deformation Network 학습 (10K steps)
- [x] Gaussian Cache 생성 (2880 frames)
- [x] Temporal inference pipeline
- [x] Evaluation metrics (jitter 92.7% 감소)
- [x] Comparison video generation

### Identified Issues ⚠️
- **Autoregressive Drift**: 현재 구현이 anchor frame에서만 시작, 누적 오차 발생
- **원본 정보 미사용**: Per-frame GS-LRM 출력을 활용하지 않음

---

## Experiment Priority

### P0: Immediate Fixes (1-2 days)

| Exp ID | Name | Goal | Status |
|--------|------|------|--------|
| **T-P0-1** | Per-frame baseline | 각 프레임별 원본 Gaussian 품질 확인 | 🔲 |
| **T-P0-2** | Label fix | 비디오 레이블 명확화 | ✅ |

### P1: Core Algorithm (1 week)

| Exp ID | Name | Goal | Status |
|--------|------|------|--------|
| **T-P1-1** | ARAP Loss | Local rigidity regularization 구현 | 🔲 |
| **T-P1-2** | Per-frame + Reg | 원본 유지 + temporal regularization | 🔲 |
| **T-P1-3** | Velocity smooth | 속도 연속성 loss 추가 | 🔲 |

### P2: Enhanced Methods (2 weeks)

| Exp ID | Name | Goal | Status |
|--------|------|------|--------|
| **T-P2-1** | Optical Flow | RAFT flow alignment loss | 🔲 |
| **T-P2-2** | Sliding Window | Window-based joint optimization | 🔲 |
| **T-P2-3** | SC-GS style | Sparse control points 도입 | 🔲 |

### P3: E2E Pipeline (ongoing)

| Exp ID | Name | Goal | Status |
|--------|------|------|--------|
| **T-P3-1** | MVDiffusion → GS-LRM → Temporal | 전체 파이프라인 통합 | 🔲 |
| **T-P3-2** | Real-time inference | 캐시 없이 실시간 추론 | 🔲 |

---

## Proposed Architecture

### Current (Problematic)


### Proposed (Per-frame + Regularization)


---

## Key Metrics

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Temporal Jitter | 0.0012 | < 0.001 | ARAP + velocity |
| PSNR (vs GT) | TBD | > 25 | Per-frame quality |
| Visual Drift | Severe | Minimal | Per-frame baseline |

---

## Implementation Plan

### Week 1: Core Fixes
1. **T-P0-1**: Per-frame baseline 품질 확인
2. **T-P1-1**: ARAP loss 구현 ()
3. **T-P1-2**: Per-frame + regularization fusion

### Week 2: Enhancement
4. **T-P1-3**: Velocity smoothness
5. **T-P2-1**: Optical flow integration (optional)

### Week 3: Integration
6. **T-P3-1**: E2E pipeline 통합
7. Ablation study & documentation

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
|  | **NEW** | ARAP, velocity, isometry losses |
|  | **NEW** | Per-frame + regularization |
|  | MODIFY | Add per-frame option |
|  | MODIFY | Support new pipeline |

---

## Related Documents

- [Research Report](./research/260204_4D_Gaussian_Temporal_Methods.md)
- [Deformation Network](../mouse_extensions/model/deformation/)
- [Evaluation Metrics](../mouse_extensions/evaluation/)

---

*Engram | Experiment Plan | 2026-02-04*
