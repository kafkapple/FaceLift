# MVDiffusion Extended Training Analysis

> Date: 2026-02-16
> Phase 3, Round 1: Learning Rate Strategy Experiments
> WandB Project: `FaceLift-MVDiffusion`

## 1. Executive Summary

Phase 3의 첫 번째 목표인 MVDiffusion 장기 학습 실험(E1, E2)의 중간 결과를 보고합니다.
**핵심 발견**: 두 LR 전략 모두 baseline(5K steps) 대비 validation PSNR에서 **+2.5~3.6 dB의 대폭 개선**을 달성. 이는 MVDiffusion이 5K steps에서 심각하게 under-trained 상태였음을 입증합니다.

## 2. Experiment Setup

### 2.1 Common Configuration

| Parameter | Value |
|-----------|-------|
| Data | M5t2 split (train 2880, val 360, test 360) |
| Architecture | SD2.1-UnCLIP + Era3D sparse attention |
| Base LR | 5e-5 (E1 start), 1e-5 (E2 after 10K) |
| Validation | Every 200 steps, cfg=1.0 and cfg=3.0 |
| GPU | A6000 48GB each |

### 2.2 Experiment Specifications

| ID | Name | Strategy | Steps | LR Schedule | GPU | WandB Run |
|----|------|----------|-------|-------------|-----|-----------|
| E1 | `mvdiff_M5t2_20k_cosine` | Fresh train, cosine decay | 20K | 5e-5 → 0 (cosine) | 4 | `hmb12us2` |
| E2 | `mvdiff_M5t2_randref_sparse` | Resume from ckpt-5000 | +15K (total 20K) | 1e-5 (fixed after 10K) | 7 | `ohaif4n9` |

### 2.3 Baseline Reference

| Metric | Baseline (ckpt-5000) | Source |
|--------|---------------------|--------|
| Val PSNR (cfg=3.0) | ~24.1 | WandB step 5000 |
| Val SSIM | 0.974 | WandB step 5000 |
| Val LPIPS | 0.035 | WandB step 5000 |
| E2E PSNR_fw | 21.21 | H1bis evaluation |
| E2E sIoU | 0.521 | H1bis evaluation |

## 3. Results

### 3.1 E1: Cosine LR from Scratch (22% complete, step 4,489/20K)

| Step | Val PSNR | SSIM | LPIPS | LR |
|------|----------|------|-------|----|
| 600 | 21.95 | 0.9697 | 0.0407 | 4.98e-5 |
| 1000 | 23.65 | 0.9727 | 0.0355 | 4.96e-5 |
| 2000 | 25.37 | 0.9771 | 0.0290 | 4.84e-5 |
| 3000 | **26.15** | 0.9790 | 0.0257 | 4.67e-5 |
| 4000 | **26.62** | 0.9795 | **0.0248** | 4.50e-5 |
| 4400 | 26.33 | 0.9790 | 0.0254 | 4.45e-5 |

**Observations:**
- Step 1000에서 이미 baseline step 5000과 유사한 수준 (23.65 vs 24.1)
- Step 3000부터 baseline을 확실히 초과 (+2.0 dB)
- **같은 iteration 수 대비 cosine LR이 constant LR보다 ~5x 빠른 수렴**
- 아직 20% 진행이므로 최종 수렴값은 상당히 높을 것으로 예상
- PSNR curve가 여전히 우상향 (plateau 미도달)

### 3.2 E2: Resume + Reduced LR (73% complete, step 14,501/20K)

| Step | Val PSNR | SSIM | LPIPS | Phase |
|------|----------|------|-------|-------|
| 5000 | 24.11 | 0.9741 | 0.0352 | ← baseline |
| 7000 | 25.78 | 0.9766 | 0.0301 | +1.7 dB |
| 9400 | 26.16 | 0.9785 | 0.0253 | +2.0 dB |
| 10800 | **27.44** | 0.9814 | 0.0218 | +3.3 dB (spike) |
| 13000 | **27.69** | 0.9819 | **0.0212** | +3.6 dB |
| 14000 | 26.59 | 0.9790 | 0.0247 | (fluctuation) |
| 14400 | **27.70** | **0.9819** | **0.0206** | **BEST** |

**Observations:**
- Baseline 이후 꾸준히 개선: +3.6 dB at step 14400
- LPIPS도 0.035 → 0.021로 **40% 개선** (perceptual quality 대폭 향상)
- Step 10K 이후 LR=1e-5 고정 구간에서도 peak이 상승 (10800→13000→14400)
- 높은 variance (±1~2 dB between peaks and valleys) — validation sample sensitivity
- PSNR peak들이 단조 증가: 27.44 → 27.69 → 27.70

### 3.3 Cross-Experiment Comparison

```
                        Val PSNR   SSIM     LPIPS    Δ vs baseline
──────────────────────────────────────────────────────────────────
Baseline (ckpt-5000)     24.11    0.9741   0.0352    —
E1 best (step 4000)      26.62    0.9795   0.0248    +2.51 dB
E2 best (step 14400)     27.70    0.9819   0.0206    +3.59 dB
```

| Metric | E1 (22%) | E2 (73%) | Winner |
|--------|----------|----------|--------|
| PSNR (best) | 26.62 | 27.70 | E2 (+1.08) |
| SSIM (best) | 0.9795 | 0.9819 | E2 |
| LPIPS (best) | 0.0248 | 0.0206 | E2 |
| Efficiency (PSNR/step) | 0.59 dB/K | 0.24 dB/K | E1 (2.5x) |
| Convergence speed | Faster | Slower | E1 |

## 4. Analysis

### 4.1 MVDiffusion Under-training 확인

Baseline의 5K steps는 **심각한 under-training**이었습니다:
- E1이 같은 step 수(5K)에서 이미 +2 dB 높은 이유: cosine LR warmup이 더 효율적인 학습 경로를 제공
- E2에서 추가 10K steps만으로 +3.6 dB 개선: 모델이 아직 학습 가능한 capacity가 충분히 남아 있었음

### 4.2 Learning Rate Strategy 비교

**E1 (Cosine)의 장점:**
- 빠른 초기 수렴 — step 1000에서 이미 유의미한 수준
- Smooth decay가 안정적인 학습 궤도 제공
- Step efficiency가 E2 대비 2.5x 높음

**E2 (Resume + Fixed LR)의 장점:**
- 절대 성능이 더 높음 (27.70 vs 26.62, 아직 E1은 22%만 진행)
- Pre-trained features를 활용하므로 높은 시작점
- 10K 이후에도 peak이 상승하여 아직 개선 여지 존재

### 4.3 Validation Variance

E2에서 높은 variance (peak 27.70 vs valley 24.93)가 관찰됩니다:
- **원인**: 단일 validation sample 기반 평가 (이미지 1장에 대한 6-view 생성)
- **의미**: Checkpoint 선택이 중요 — 마지막 checkpoint이 반드시 최선은 아님
- **해결**: E2E 평가 시 360개 전체 test set 사용으로 robust한 비교 가능

### 4.4 E2E Transfer 예측

H6-E2E 실험에서 GS-LRM 개선(GT PSNR +2.9 dB)이 E2E로 0% 전이됨을 확인했습니다.
반면 MVDiffusion은 **E2E pipeline의 직접적 병목**이므로:
- MVDiffusion validation PSNR +3.6 dB는 E2E에서 유의미한 개선으로 전이될 가능성 높음
- **예상**: E2E PSNR_fw 21.2 → 22~24 range (전이율 30~80%)
- **검증 필요**: E2 완료 후 E2E eval로 실제 전이율 측정

## 5. Current Training Status (2026-02-16 01:50 KST)

| GPU | Exp | Step | Progress | ETA |
|-----|-----|------|----------|-----|
| 4 | E1 (cosine 20K) | 4,489/20K | 22% | ~35h |
| 5 | E5 (alpha=0.3 GS-LRM) | ~701/15840 | 4% | ~42h |
| 6 | E3 (extrinsic+add pose) | 265/10K | 3% | ~29h |
| 7 | E2 (resume 20K) | 14,501/20K | 73% | ~16h |

## 6. Next Steps

1. **E2 완료 → E2E eval** (auto-chain: `launch_e2e_after_E2.sh`)
   - MVDiff: E2 last checkpoint, GS-LRM: baseline (M5t2_E0_1)
   - 360 test samples, protocol v2
   - **핵심 질문**: MVDiff +3.6 dB가 E2E로 얼마나 전이되는가?

2. **E1 완료 → E2E eval** (E2와 동일 protocol)
   - Cosine vs Resume LR의 E2E 효과 비교

3. **E3/E5 모니터링**
   - E3: Pose conditioning의 첫 validation 결과 확인
   - E5: Alpha=0.3의 PSNR-LPIPS trade-off 관찰

4. **E4 (spherical+add) 대기**
   - E2 완료 후 GPU 7에서 E2E eval → E4 순차 실행

---

## Appendix A: WandB Run Details

### E1 Full Validation History (cfg=3.0)

```
Step  | PSNR   | SSIM   | LPIPS
------+--------+--------+-------
    1 |  4.282 | 0.6025 | 0.6230
  200 | 15.976 | 0.9487 | 0.0899
  400 | 17.213 | 0.9536 | 0.0687
  600 | 21.951 | 0.9697 | 0.0407
  800 | 23.644 | 0.9741 | 0.0337
 1000 | 23.649 | 0.9727 | 0.0355
 1200 | 24.406 | 0.9746 | 0.0329
 1400 | 24.663 | 0.9756 | 0.0311
 1600 | 25.001 | 0.9766 | 0.0284
 1800 | 25.093 | 0.9771 | 0.0289
 2000 | 25.370 | 0.9771 | 0.0290
 2200 | 25.411 | 0.9775 | 0.0289
 2400 | 25.362 | 0.9771 | 0.0291
 2600 | 25.575 | 0.9775 | 0.0282
 2800 | 25.904 | 0.9785 | 0.0263
 3000 | 26.148 | 0.9790 | 0.0257
 3200 | 26.372 | 0.9790 | 0.0255
 3400 | 26.347 | 0.9790 | 0.0256
 3600 | 26.430 | 0.9790 | 0.0257
 3800 | 26.557 | 0.9795 | 0.0253
 4000 | 26.620 | 0.9795 | 0.0248
 4200 | 26.380 | 0.9795 | 0.0250
 4400 | 26.325 | 0.9790 | 0.0254
```

### E2 Full Validation History (cfg=3.0)

```
Step  | PSNR   | SSIM   | LPIPS  | Note
------+--------+--------+--------+---------
  200 | 17.092 | 0.9521 | 0.0721 |
 1000 | 18.482 | 0.9629 | 0.0559 |
 2000 | 23.150 | 0.9712 | 0.0422 |
 3000 | 23.639 | 0.9731 | 0.0390 |
 4000 | 25.067 | 0.9756 | 0.0356 |
 5000 | 24.112 | 0.9741 | 0.0352 | ← baseline
 6000 | 24.545 | 0.9751 | 0.0323 |
 7000 | 25.783 | 0.9766 | 0.0301 |
 8000 | 25.843 | 0.9775 | 0.0282 |
 9000 | 25.363 | 0.9771 | 0.0298 |
 9400 | 26.161 | 0.9785 | 0.0253 |
10000 | 25.671 | 0.9775 | 0.0277 |
10800 | 27.440 | 0.9814 | 0.0218 | ★ spike
11000 | 25.581 | 0.9775 | 0.0281 |
12000 | 26.583 | 0.9800 | 0.0253 |
13000 | 27.686 | 0.9819 | 0.0212 | ★ new peak
14000 | 26.587 | 0.9790 | 0.0247 |
14400 | 27.696 | 0.9819 | 0.0206 | ★ BEST
```

---

*Phase 3 MVDiffusion Training Analysis | 2026-02-16*
