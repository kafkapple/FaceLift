# 4D Gaussian Splatting: Temporal Consistency Methods

**Date**: 2026-02-04  
**Author**: Claude Agent  
**Status**: Research Complete

---

## 1. Executive Summary

4D Gaussian Splatting의 temporal consistency 확보 전략은 3가지로 분류됨:
1. **Canonical + Deformation**: 기준 공간에서 시간별 변형 학습
2. **Native 4D Primitives**: 시간을 4번째 차원으로 직접 모델링
3. **Per-frame + Regularization**: 프레임별 재구성 + temporal 제약

---

## 2. 현재 FaceLift 구현의 문제점

### 2.1 Autoregressive Drift (녹아내림 현상)

현재 구현 (temporal_pipeline.py):
- 이전 DEFORMED 결과를 사용하여 누적 drift 발생
- Frame N = Frame 0 + Sum(Delta_i) -> 누적 오차

### 2.2 원본 정보 손실

- 각 프레임의 GS-LRM 출력(원본)을 전혀 사용하지 않음
- Anchor frame만 실제 reconstruction

---

## 3. 주요 방법론 비교

| 방법 | 핵심 아이디어 | Temporal 처리 | 장점 | 단점 |
|------|--------------|---------------|------|------|
| D-3DGS | Canonical + Deform MLP | 시간 조건부 변형 | 구현 단순 | Canonical 품질 의존 |
| Dynamic 3DGS | Per-frame + Rigidity | Local rigid constraints | Dense tracking | 큰 변형 불가 |
| SC-GS | Sparse control points | ARAP loss | 최고 품질, 편집 가능 | 초기화 민감 |
| 4D-GS | HexPlane decomposition | 4D voxel encoding | Real-time | 긴 시퀀스 메모리 |
| Native 4D | 4D ellipsoids | 시공간 통합 | 유연성 | 계산 비용 |

---

## 4. 핵심 Regularization Losses

### 4.1 Local Rigidity Loss
L_rigidity = Sum_i Sum_j in N(i) ||v_i - R*v_j||^2
- 인접 Gaussian들의 motion이 locally rigid하도록 강제

### 4.2 ARAP (As-Rigid-As-Possible) Loss
L_ARAP = Sum_i Sum_j in N(i) ||p'_j - p'_i - R_hat_i(p_j - p_i)||^2
- SC-GS에서 사용, 생쥐의 관절 움직임에 적합

### 4.3 Velocity Smoothness
L_velocity = ||v_t - v_{t-1}||^2
- 속도 연속성 강제

### 4.4 Isometry (Distance Preservation)
L_isometry = ||d(p_i,p_j)_t - d(p_i,p_j)_0||^2
- 점 간 거리 보존

### 4.5 Optical Flow Alignment
L_flow = w * ||Delta_p_2D - flow||^2 (uncertainty-weighted)
- 2D motion과 3D motion 정렬

---

## 5. 권장 접근법

### 5.1 Per-frame + Temporal Regularization (권장)

Per-frame GS-LRM Output (완전한 reconstruction)
- Photometric Loss (L1 + SSIM) -> 현재 프레임 충실도
- Temporal Regularization -> 시간적 일관성
  - ARAP Loss (lambda=0.1)
  - Velocity Smoothness (lambda=0.01)
  - Flow Alignment (optional)

핵심: Deformation을 replacement가 아닌 regularization으로 사용

### 5.2 Two-Stage Fusion

1. Stage 1: Per-frame GS-LRM inference (독립)
2. Stage 2: Temporal refinement with ARAP + velocity loss

---

## 6. 구현 우선순위

| Priority | Module | Description | Effort |
|----------|--------|-------------|--------|
| P0 | temporal_regularization.py | ARAP + velocity loss 구현 | Medium |
| P1 | temporal_pipeline_v2.py | Per-frame + regularization fusion | High |
| P2 | optical_flow_loss.py | RAFT flow alignment | Medium |
| P3 | sliding_window_trainer.py | Window-based joint training | High |

---

## 7. References

### Core Papers
- Deformable 3D Gaussians (arXiv 2309.13101)
- Dynamic 3D Gaussians (arXiv 2308.09713)
- SC-GS (arXiv 2312.14937)
- 4D-GS (arXiv 2310.08528)

### Temporal Regularization
- MotionGS (NeurIPS 2024)
- MD-Splatting (arXiv 2312.00583)
- Dynamic Gaussian Marbles (arXiv 2406.18717)

---

Engram | Research Report | 2026-02-04
