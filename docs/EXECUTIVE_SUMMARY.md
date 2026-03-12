# BehaviorSplatter: Executive Summary (1-Page)

> 2026-03-03 | Single-View → 3D Reconstruction → Behavior Analysis for Freely Moving Animals

---

## Problem

기존 동물 3D 재구성 방법들은 (1) 다수의 카메라 뷰 필요 (PoseSplatter: 6뷰), (2) 장면마다 30분 최적화 (per-scene), (3) 골격 템플릿 필수 (DANNCE, MAMMAL). → **실시간 행동 분석 불가, 새로운 종 적용 어려움.**

## Solution: BehaviorSplatter

**단일 이미지 → 3D Gaussian 재구성 → 행동 클러스터링** (Feed-forward, Template-free, ~2초/프레임)

| Stage | Component | Input → Output |
|:-----:|-----------|----------------|
| 1 | MVDiffusion (SD2.1-UnCLIP + Plücker Spatial Token) | 1 image → 6 multi-view images |
| 2 | GS-LRM (ViT 24L transformer) | 6 views → 16,386 3D Gaussians |
| 3 | Visual Embedding (ResNet + SH + AdvPCA) | 3D Gaussians → 50D behavior embedding |

## Key Results

| Comparison | PSNR_gt | IoU | Speed | Input |
|-----------|:-------:|:---:|:-----:|:-----:|
| **GS-LRM 6v (GT input)** | **23.84** | **0.954** | 0.1s | 6 GT views |
| PoseSplatter (SOTA) | 13.78 | 0.846 | 30min | 6 views |
| **BehaviorSplatter E2E** | **9.04** | **0.577** | 2s | **1 view** |

## Core Findings

1. **GS-LRM >> PoseSplatter** (+10.06 dB): Reconstruction stage 자체는 SOTA 초과. GT 입력 시 per-scene optimization 불필요.
2. **MVDiffusion = 유일한 병목** (86%): E2E gap -14.8 dB 중 86%가 뷰 합성 품질 문제. 모든 학습 전략이 7.75-9.04 dB 범위 수렴.
3. **Plücker Spatial Token 유망**: Val PSNR +0.79 dB (27.34). Pixel-level 기하학 정보 보존이 핵심.
4. **Template-free 행동 분석**: 50D visual embedding, Silhouette score 0.596. 종간 일반화 가능.

## Trade-off: Feed-Forward vs Per-Scene

```
              Quality              Speed          Template    Species
PS:           13.78 dB             30 min/scene   Required    Limited
BS (E2E):      9.04 dB              2 sec/frame   None        Any
BS (GT):      23.84 dB ←CEILING→   0.1 sec        None        Any
```

**핵심**: MVDiffusion 품질만 개선되면, Feed-forward 접근이 Per-scene을 **품질+속도 모두** 초월.

## NeurIPS Readiness

| 항목 | 현재 | 필요 | Gap |
|------|:----:|:----:|:---:|
| MVDiff 품질 | 9.04 dB | >12 dB | High |
| 다종 평가 | Mouse only | +Rat, Fly | High |
| 행동 클러스터링 | Synthetic test | Real behavior GT | Medium |
| Temporal 일관성 | Per-frame | 4D smooth | Medium |
| 논문 완성도 | Draft v1 | Camera-ready | Medium |

## Next: Top 5 Priority Experiments

1. **Silhouette-focused loss** (H8) → IoU 0.577 → 0.7+ 목표
2. **Domain Adaptation** (DA1) → MVDiff 합성 뷰로 GS-LRM fine-tune → 분포 정합
3. **H7v2 Spatial Token 완성** → 진정한 Plücker E2E 평가
4. **Multi-species** → Rat7M 데이터로 일반화 증명
5. **Real behavior validation** → Ground-truth 행동 라벨로 embedding 검증

---

*BehaviorSplatter Executive Summary v1.0 | 2026-03-03*
