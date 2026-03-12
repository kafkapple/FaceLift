# FaceLift 연구 가설 대시보드 (ARCHIVED)

> **⚠️ ARCHIVED (2026-03-12)**: 이 문서는 `experiments/hypothesis_roadmap.md`로 통합되었습니다.
> **SSOT → [[experiments/hypothesis_roadmap]]**
>
> 아래 내용은 참고용으로 보존되지만, 최신 데이터는 hypothesis_roadmap.md를 참조하세요.
> 특히 H4 View Ablation 결과는 이 문서가 정확합니다 (6v=23.84 > 5v=22.16, monotonic).
>
> Original Last Updated: 2026-02-20 | Project: FaceLift

---

## Executive Summary

| 가설 | 핵심 질문 | 상태 | 결과 |
|------|----------|------|------|
| **[HP](#hp-preprocessing)** | 전처리 기하학 설정은? | 🔄 | 이론 ✅, 실험 ablation 대기 |
| [H0](#h0-poc) | PoC 가능한가? | ✅ | PSNR ~27 (train) |
| [H1](#h1-temporal-split) | Temporal split이 유효한가? | ✅ | Data leakage 방지 확인 |
| [H2](#h2-data-amount) | 학습 데이터 양이 중요한가? | ✅ | 다양성 > 반복 학습 |
| [H3](#h3-e2e-bottleneck) | E2E 파이프라인 병목은? | ✅ | MV-Diffusion (undertrained 시). H3-bis 완료 |
| [H3-bis](#h3-bis-same-test-set-revalidation) | 동일 테스트셋 재검증? | ✅ | MVDiff 항상 병목 (13-17 dB gap), 데이터↑ → gap↓ ~3 dB |
| [H4](#h4-view-ablation) | 최적 입력 뷰 수는? | ✅ | **6-view 최적 (단조 증가)**. R1+R2 완료 |
| [H5](#h5-mvdiffusion) | MV-Diffusion 개선 방법은? | ✅ | Phase 3 완료: 모든 전략 PSNR_gt 7.9-8.4 수렴, 아키텍처 한계 |
| [H6](#h6-alpha-mask) | Alpha mask가 효과적인가? | 🔄 | v3 실행중: LPIPS 3x 개선, PSNR ~1dB trade-off |
| [H7](#h7-ssim-weight) | SSIM weight 최적값은? | ⏳ | 대기 |
| [H8](#h8-reduced-view-generation) | 생성 뷰 감소로 품질↑? | ✅ | ❌ 뷰 감소 → novel PSNR -7.7dB, 개선 없음 |

> **PS M5 재학습**: 좌표계 수정(auto_orient space) 후 학습 진행중. 완료 시 fair Tier B 비교 가능.

---

## Experiment Status Dashboard (260220)

### GPU Allocation

| GPU | 실험 | 상태 |
|-----|------|------|
| 0-3 | **IDLE** | - |
| 4 | **Phase 3 E1** (20K cosine, 새 학습) | ✅ complete (7.90 dB) |
| 5 | H6 alpha05_v3 (GS-LRM) | 🔄 ~15K (Feb 13~) |
| 6 | H6 alpha10_v3 (GS-LRM) | 🔄 ~15K (Feb 13~) |
| 7 | **Phase 3 E2** (P0 resume, LR=1e-5) | ✅ complete (8.20 dB) |

### Recently Completed (260211-260220)

| 실험 | 결과 |
|------|------|
| **H5 P0 randref_sparse** (10K) | ✅ ckpt-10K, PSNR ~27 but oscillation |
| **H5 P1 pose_spherical** (10K) | ✅ 완료, 5K 후 plateau |
| **H5 E1 20K cosine** | ✅ 7.90 dB (PSNR_int=16.15, best color) |
| **H5 E2 resume 20K** | ✅ 8.20 dB (best overall PSNR_gt) |
| **H5 E3 pose 10K** | ✅ 8.10 dB (pose conditioning ineffective) |
| H8 3-view E2E | ✅ PSNR_wh=22.67, novel avg=16.26 (❌ 6-view보다 -7.7dB) |
| H8 결론 | ✅ 뷰 감소는 품질 개선 불가 (GS-LRM 뷰 부족이 지배적) |

> **Phase 3 COMPLETE**: 모든 전략(baseline/cfgr/E1/E2/E3) PSNR_gt 7.90-8.44 dB로 수렴. Training strategy로는 MVDiff 아키텍처 병목 돌파 불가.

### Phase 3 MVDiff Status (COMPLETE)

| # | Config | 핵심 변경 | LR | 상태 | GPU | PSNR_gt |
|:-:|--------|----------|:--:|:----:|:---:|:-------:|
| **E1** | M5t2_20k_cosine | cosine 5e-5→0, 20K 새 학습 | cosine | ✅ | 4 | 7.90 |
| **E2** | M5t2_randref_20k_resume | P0 resume ckpt-10K, LR=1e-5 | piecewise | ✅ | 7 | 8.20 |
| **E3** | M5t2_pose_extrinsic_add | extrinsic 6D + add | cosine | ✅ | 4 | 8.10 |
| **E4** | M5t2_pose_spherical_add | spherical + add (vs P1 concat) | cosine | ⏳ R1 후 | 7 | - |
| **E5** | 4view_alpha03_v3 | GS-LRM alpha=0.3 | - | ⏳ H6 완료 후 | 5/6 | - |

### Next Priority

| 순위 | 실험 | GPU | ETA |
|:----:|------|:---:|:---:|
| **1** | E5 (alpha=0.3) | 5/6 | H6 v3 완료 후 |
| **2** | E4 (spherical+add) | 7 | GPU 해제 후 |
| 3 | H7 SSIM weight ablation | TBD | Phase 3 분석 후 |

---

## 핵심 정량 결과

### HP 전처리 (카메라 정규화)

| 모드 | 설정 | 기하학 | 결과 |
|------|------|--------|------|
| **Batch Uniform** | `recenter_cameras=True` | ✅ 보존 | ⭐ **권장** |
| Per-view | `normalize_translation=True` | ⚠️ 왜곡 | parallax 오류 |
| None | 둘 다 False | ✅ 보존 | pretrained 불일치 |

### H3 진단 (MV-Diffusion 병목)

| Dataset | GS-LRM Test | E2E Test | Gap | 해석 |
|---------|-------------|----------|-----|------|
| M5t2 (2880) | 19.58 | 19.63 | **-0.05** | ✅ 충분한 학습 |
| M5t (1198) | 20.56 | 19.15 | **+1.41** | ⚠️ MVDiff 병목 |

### H3-bis 재검증 (동일 테스트셋, metrics_v2)

| Condition | PSNR_wh | PSNR_fg | SSIM | IoU | 해석 |
|-----------|---------|---------|------|-----|------|
| A: GS-LRM only | 35.01±3.67 | 20.92±3.54 | 0.9929 | 0.946 | Upper bound |
| B: E2E M5t2 ckpt-5000 | 21.21±2.65 | 7.91±2.63 | 0.9662 | 0.521 | 13.8 dB gap |
| C: E2E M5t ckpt-8000 | 18.28±2.50 | 4.66±2.82 | 0.9566 | 0.263 | 16.7 dB gap |

### H4 View Ablation v2 (Uniform, ✅ 최종 260210)

| Views | PSNR (dB) | Best Step | Δ vs 4v |
|:-----:|:---------:|:---------:|:-------:|
| **6** | **24.49** | 4,201 | +2.78 |
| 5 | 23.02 | 13,101 | +1.31 |
| **4** | **21.71** | 9,201 | — (논문 기본) |
| 3 | 20.01 | 10,701 | -1.70 |
| 2 | 17.75 | 11,801 | -3.96 |
| 1 | 11.08 | 2,401 | -10.63 |
| baseline | 15.99 | 0 | -5.72 |

> 이전 R1(비균일): 3view best → v2(uniform): **단조 증가**. 실험 조건 통일의 중요성.

### H4 Fair Eval Test Set 결과 (260220)

> Val PSNR (학습 중 검증)과 Fair Test PSNR_gt (테스트셋 공정 평가)의 비교:

| Views | Val PSNR | Fair Test PSNR_gt | Fair Test IoU |
|:-----:|:--------:|:-----------------:|:-------------:|
| 1 | 11.08 | 10.47 | 0.028 |
| 2 | 17.75 | 15.95 | 0.858 |
| 3 | 20.01 | 18.56 | 0.899 |
| 4 | 21.71 | 20.66 | 0.926 |
| 5 | 23.02 | 22.16 | 0.942 |
| 6 | 24.49 | 23.84 | 0.954 |

> 6v > 5v > 4v > 3v > 2v > 1v (monotonic). Val과 Fair Test 모두 단조 증가 패턴 확인.

### H5 cfgr 결과

| Config | PSNR_wh | vs Baseline | 판정 |
|--------|---------|-------------|------|
| baseline (sparse=true) | 21.29 | - | ⭐ |
| cfgr (sparse=false) | 20.81 | **-0.48 dB** | ❌ worse |

### H8 4-view E2E 결과

| 조건 | PSNR_wh | Gap from GS-LRM |
|------|---------|-----------------|
| GS-LRM only (A) | 35.01 | - |
| 4-view E2E (views 2,3,5) | 19.87 | -15.14 dB |

---

## HP: Preprocessing (기초 MVG 가설) 🔄

> **PoC 성립을 위한 필수 전처리 설정**
>
> → 상세: [datasets/PREPROCESSING_REGISTRY.md](./datasets/PREPROCESSING_REGISTRY.md)
> → 상세: [datasets/M5_SERIES_SPEC.md](./datasets/M5_SERIES_SPEC.md)
> → Ablation 계획: [experiments/hp_preprocessing_ablation.md](./experiments/hp_preprocessing_ablation.md)

### HP-1: PP Centering

> Principal Point를 256으로 고정해야 pretrained 모델과 호환되는가?

| 설정 | PP 값 | 결과 |
|------|-------|------|
| **shift_to_256** | 256 (고정) | ✅ Pretrained 호환 |
| original | 가변 | ⚠️ 불안정 |

**결론**: ✅ PP=256 필수 (pretrained cx=cy=256 기대)

---

### HP-2: Camera Translation Normalization

> 6개 카메라 거리를 어떻게 정규화해야 기하학이 보존되는가?

**핵심 문제**: GS-LRM pretrained는 `trans_norm≈2.7` 기대. 하지만 정규화 방식에 따라 **기하학 왜곡** 발생 가능.

| 모드 | 동작 | 기하학 | 권장 |
|------|------|--------|------|
| **Batch Uniform** | Centroid→Origin + 동일 scale | ✅ **보존** | ⭐ |
| Per-view | 각 카메라 개별 2.7로 | ❌ 왜곡 | ✖ |
| None | raw translation | ✅ 보존 | △ |

**Batch Uniform 동작**:
```
1. 6개 카메라 위치의 centroid 계산
2. Centroid → Origin 이동
3. 평균 거리 = 2.7 되도록 단일 scale 적용
   → 개별 거리는 가변 (예: 2.5, 2.8, 3.0...)
   → 거리 비율 보존 (parallax 정확)
```

**Per-view 문제**:
```
각 카메라를 개별적으로 2.7로 스케일
→ 모든 카메라가 정확히 2.7m
→ 거리 비율 깨짐 (parallax 오류)
→ Ghosting 악화
```

**결론**: ✅ **Batch Uniform (`recenter_cameras=True`)** 권장

---

### HP-3: Affine vs Homography Transform

> 기하학적 변환 방식이 품질에 영향을 주는가?

| Transform | Skew 보정 | 복잡도 | 권장 |
|-----------|----------|--------|------|
| Affine | ❌ 무시 (0.46% 차이) | 낮음 | ✅ 기준선 |
| Homography | ✅ 보정 | 중간 | ✅ 정밀 |
| Homography+Zoom | ✅ 보정 + coverage | 높음 | ✅ 최적 |

**결론**: M5 (Affine)로 기준선, 필요시 M5h (Homography)로 확장

---

### HP-4: Intrinsics Normalization

> fx=549로 정규화해야 pretrained 모델과 호환되는가?

| 원본 fx | 정규화 후 | 결과 |
|---------|----------|------|
| 844 (unnormalized) | - | ❌ PSNR ~3 (학습 실패) |
| 549 (normalized) | 549 | ✅ 정상 학습 |

**결론**: ✅ fx=549 정규화 필수

---

### HP 실험 매트릭스 (2x3 Factorial)

|  | **No Norm** | **Per-view** | **Batch Uniform** |
|--|-------------|--------------|-------------------|
| **Center (PP=256)** | M5_4 | M5_5 | **M5** ⭐ |
| **No Center (PP 가변)** | M0 | M0_n | - |

**권장 조합**: M5 (Center + Batch Uniform)

---

### HP 현황

| 상태 | 내용 |
|------|------|
| ✅ 완료 | 이론적 분석 및 문서화 |
| ✅ 완료 | M5 계열 프리셋 구현 |
| ✅ 완료 | 2x3 Ablation 설계 |
| ✅ 완료 | Config 준비: hp_M0.yaml, hp_M5_4.yaml, hp_M5_5.yaml |
| ✅ 완료 | Split 파일 생성: M0, M5_4, M5_5 |
| ⏳ 대기 | Empirical ablation 실험 (H4 uniform GPU 해제 후) |

### HP Ablation 실행 명령어 (H4 uniform 완료 후)

```bash
cd /home/joon/dev/FaceLift
mkdir -p logs

# M5 baseline은 기존 uniform_v2 4-view 결과 재사용

# GPU X: M0 (No Center + No Norm = raw baseline)
CUDA_VISIBLE_DEVICES=X python train_gslrm.py \
  -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/uniform/hp_M0.yaml \
  > logs/hp_M0.log 2>&1

# GPU X: M5_4 (Center + No Norm)
CUDA_VISIBLE_DEVICES=X python train_gslrm.py \
  -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/uniform/hp_M5_4.yaml \
  > logs/hp_M5_4.log 2>&1

# GPU X: M5_5 (Center + Per-view Norm)
CUDA_VISIBLE_DEVICES=X python train_gslrm.py \
  -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/uniform/hp_M5_5.yaml \
  > logs/hp_M5_5.log 2>&1
```

---

## H0: PoC ✅

> M5 데이터로 GS-LRM fine-tuning이 가능한가?

| 항목 | 값 |
|------|-----|
| 결과 | ✅ Train PSNR ~27 달성 |
| 이슈 | Random split → data leakage 우려 |
| 다음 | H1 (Temporal split) |

---

## H1: Temporal Split ✅

> Temporal sequence 기반 분할이 data leakage를 방지하는가?

| 항목 | 값 |
|------|-----|
| 결과 | ✅ PoseSplatter 비교 기준 확립 |
| 이슈 | Train 샘플 감소 (1198개) |
| 다음 | H2 (Data amount) |

---

## H2: Data Amount ✅

> Train 샘플 수 (80% vs 33%)가 품질에 영향을 주는가?

| 항목 | 값 |
|------|-----|
| 결과 | ✅ **다양성 > 반복 학습** (2880x6ep > 1198x20ep) |
| 결론 | **M5t2 (8:1:1) 패러다임 채택** |
| 다음 | H3 (E2E 분석) |

---

## H3: E2E Bottleneck ✅

> MV-Diffusion → GS-LRM 파이프라인에서 병목은 어디인가?

| 항목 | 값 |
|------|-----|
| 결과 | ✅ **MVDiffusion이 병목** (undertrained 시 +1.41 gap) |
| 결론 | M5t2 충분 학습 시 gap 거의 없음 |
| 다음 | H3-bis (재검증), H4 (View Ablation), H5 (MVDiff 개선) |

→ 상세 보고서: [outputs/reports/h1_diagnosis_report.md](../outputs/reports/h1_diagnosis_report.md)

---

## H3-bis: Same-Test-Set Revalidation ✅

> **H1-bis-v2**: 동일 M5t2 테스트셋에서 GS-LRM vs E2E 파이프라인 차이를 정밀 측정
>
> → 상세: [hypotheses/H1bis_v2_REVALIDATION.md](./hypotheses/H1bis_v2_REVALIDATION.md)

### 실험 설계

동일 테스트셋(M5t2, N=360)에서 3가지 조건 비교 (metrics_v2):

| Condition | 설명 | MV-Diffusion | GS-LRM |
|-----------|------|-------------|--------|
| A | GS-LRM only (GT input) | ✖ 미사용 | M5t2 ckpt |
| B | E2E (M5t2 MVDiff) | M5t2 ckpt-5000 | M5t2 ckpt |
| C | E2E (M5t MVDiff) | M5t ckpt-8000 | M5t2 ckpt |

### 정량 결과

| Condition | PSNR_wh | PSNR_fg | SSIM | IoU |
|-----------|---------|---------|------|-----|
| **A: GS-LRM only** | **35.01 ± 3.67** | **20.92 ± 3.54** | **0.9929** | **0.946** |
| B: E2E M5t2 | 21.21 ± 2.65 | 7.91 ± 2.63 | 0.9662 | 0.521 |
| C: E2E M5t | 18.28 ± 2.50 | 4.66 ± 2.82 | 0.9566 | 0.263 |

### 핵심 분석

| 비교 | Gap | 해석 |
|------|-----|------|
| A vs B | **-13.80 dB** | MV-Diffusion 통과 시 발생하는 본질적 품질 손실 |
| A vs C | **-16.73 dB** | 학습 데이터 부족 시 더 큰 손실 |
| B vs C | **+2.93 dB** | 데이터 양 효과 (M5t2 2880 vs M5t 1198) |

### 결론

1. **MVDiffusion은 항상 병목**: GT input(A) 대비 E2E(B)에서 13-17 dB 손실 발생
2. **데이터 양이 gap 감소**: M5t2(2880) vs M5t(1198) = +2.93 dB 개선
3. **IoU 차이 극심**: A=0.946 vs C=0.263 → MVDiff가 형태 보존에도 큰 영향
4. **PSNR_fg 특히 심각**: A=20.92 vs B=7.91 → foreground 영역에서 MVDiff 성능이 매우 낮음

### Output Paths

```
Condition A: outputs/h8_e2e/4view
Condition B: outputs/h1bis_v2/e2e_M5t2
Condition C: outputs/h1bis_v2/e2e_M5t
```

---

## H4: View Ablation ✅ 완료

> 최적의 입력 뷰 수는 몇 개인가?

| 항목 | 값 |
|------|-----|
| 가설 | ~~3-4 view가 최적~~ → **6-view 최적 (단조 증가)** |
| 상태 | ✅ **전체 완료** (7 experiments, 15840 steps each) |
| 결론 | **뷰 수 ↑ = PSNR ↑ (단조 증가), diminishing returns 없음** |
| baseline | zero-shot = 15.99 dB |

### Uniform v2 Training Status (✅ 전체 완료 260210)

| Views | Config | 상태 | Final Step | Best PSNR | Best Step |
|:-----:|--------|:----:|:----------:|:---------:|:---------:|
| baseline | baseline_v2 | ✅ | 0 | 15.99 | 0 |
| 1-view | 1view_v2 | ✅ | 15,840 | 11.08 | 2,401 |
| 2-view | 2view_v2 | ✅ | 15,840 | 17.75 | 11,801 |
| 3-view | 3view_v2 | ✅ | 15,840 | 20.01 | 10,701 |
| 4-view | 4view_v2 | ✅ | 15,840 | 21.71 | 9,201 |
| 5-view | 5view_v2 | ✅ | 15,840 | 23.02 | 13,101 |
| 6-view | 6view_v2 | ✅ | 15,840 | 24.49 | 4,201 |

### H4 Final Results (val PSNR, 전체 완료 260210)

| Views | PSNR (dB) | Best Step | Δ vs Baseline | 비고 |
|:-----:|:---------:|:---------:|:-------------:|------|
| **6** | **24.49** | 4,201 | +8.50 | ⭐ Best, 가장 빠른 수렴 |
| 5 | 23.02 | 13,101 | +7.03 | |
| 4 | 21.71 | 9,201 | +5.72 | 원 논문 기본 설정 |
| 3 | 20.01 | 10,701 | +4.02 | |
| 2 | 17.75 | 11,801 | +1.76 | |
| 1 | 11.08 | 2,401 | -4.91 | ❌ Fine-tuning 역효과 |
| baseline (0-shot) | 15.99 | 0 | — | Pretrained only |

### H4 Fair Eval Test Set 결과 (260220)

> Val PSNR (학습 중 검증)과 Fair Test (테스트셋 공정 평가) 비교. 6v > 5v (monotonic).

| Views | Val PSNR | Fair Test PSNR_gt | Fair Test IoU |
|:-----:|:--------:|:-----------------:|:-------------:|
| 1 | 11.08 | 10.47 | 0.028 |
| 2 | 17.75 | 15.95 | 0.858 |
| 3 | 20.01 | 18.56 | 0.899 |
| 4 | 21.71 | 20.66 | 0.926 |
| 5 | 23.02 | 22.16 | 0.942 |
| 6 | 24.49 | 23.84 | 0.954 |

→ **상세 + 명령어**: [hypotheses/H4_VIEW_ABLATION.md](./hypotheses/H4_VIEW_ABLATION.md)

---

## H5: MV-Diffusion ✅

> MV-Diffusion fine-tuning으로 E2E 품질을 개선할 수 있는가?

### H5 실험 매트릭스 (260220 업데이트)

**원칙**: Baseline (sparse=true, ref=0)에서 **단일 변수만** 변경

| Config | 변경 변수 | LR | steps | 상태 | 결과 |
|--------|----------|:--:|:-----:|:----:|:----:|
| **M5t2** (baseline) | - | piecewise 5e-5 | 10K | ✅ | PSNR_wh=21.29 |
| **M5t2_cfgr** | sparse→full | piecewise 5e-5 | 10K | ✅ | -0.48 dB ❌ |
| **P0: randref_sparse** | ref=random | piecewise 5e-5 | 10K | ✅ | ~27, oscillation |
| **P1: pose_spherical** | pose+concat | piecewise 5e-5 | 10K | ✅ | plateau @5K |
| **E1: 20k_cosine** | **cosine LR**, 20K | **cosine** | 20K | ✅ | **7.90 dB** (PSNR_int=16.15) |
| **E2: randref_20k_resume** | P0 resume, **LR=1e-5** | **piecewise** | 20K | ✅ | **8.20 dB** (best PSNR_gt) |
| **E3: pose_extrinsic_add** | extrinsic+**add** | **cosine** | 10K | ✅ | **8.10 dB** (pose ineffective) |
| **E4: pose_spherical_add** | spherical+**add** | **cosine** | 10K | ⏳ | |

### H5 Phase 1-2 결론

| 가설 | 실험 | 결과 |
|------|------|------|
| Full attention이 품질 향상? | cfgr vs baseline | ❌ -0.48 dB (worse) |
| Random ref가 다양성 증가? | P0 randref_sparse | ⚠️ PSNR ~27 도달, LR 문제로 oscillation |
| Pose conditioning이 E2E 개선? | P1 pose_spherical | ⚠️ 5K 후 plateau, concat 효과 미미 |
| **근본 원인** | P0+P1 공통 | `step_rules "1:100000,0.5"` = 10K 내 LR decay 없음 |

### H5 Phase 3 결론 (260220, COMPLETE)

Phase 2 실패 분석 → **LR 개선 + Pose integration 변형** 실험 완료:

| # | 실험 | PSNR_gt | PSNR_int | 핵심 결과 |
|:-:|------|:-------:|:--------:|----------|
| baseline | M5t2 (sparse, 10K) | 8.44 | - | 기준선 |
| cfgr | full attention | - | - | -0.48 dB ❌ |
| **E1** | 20K cosine (새 학습) | **7.90** | **16.15** | Best color (PSNR_int) |
| **E2** | P0 resume LR=1e-5 | **8.20** | - | **Best PSNR_gt** |
| **E3** | extrinsic 6D + add | **8.10** | - | Pose conditioning 무효 |

**Phase 3 핵심 결론**:

1. **수렴 포화**: 모든 전략(baseline/cfgr/E1/E2/E3)이 PSNR_gt 7.90-8.44 dB 범위로 수렴
2. **Training strategy 한계**: Cosine LR, resume, pose conditioning 모두 아키텍처 병목 돌파 불가
3. **Stage 2 transfer rate = 0%**: Alpha regularization 등 Stage 2 개선이 E2E에 전달되지 않음
4. **결론**: MVDiff 아키텍처 자체 변경 필요 (attention mechanism, backbone 등)

→ 상세: [hypotheses/H5_MVDIFFUSION.md](./hypotheses/H5_MVDIFFUSION.md)

---

## H6: Alpha Mask 🔄

> Rendered alpha mask supervision이 foreground 품질을 개선하는가?

### H6 v3 결과 (bugfix 후, 260213~)

> v2는 `alpha_loss=0` bug로 무효화. v3: `original_gt_mask` 보존.

| Config | alpha_w | Best PSNR | LPIPS ↓ | SSIM ↑ | Alpha IoU ↑ | 상태 |
|--------|:-------:|:---------:|:-------:|:------:|:-----------:|:----:|
| baseline (4view_v2) | 0.0 | **21.82** | 0.0429 | 0.9473 | N/A | ✅ |
| alpha05_v3 | 0.5 | 21.20 | 0.0204 | 0.9725 | 0.9451 | 🔄 GPU 5 |
| **alpha10_v3** | **1.0** | 20.84 | **0.0147** | **0.9742** | **0.9562** | 🔄 GPU 6 |
| **alpha03_v3 (E5)** | **0.3** | - | - | - | - | ⏳ 대기 |

**핵심 발견**: PSNR은 baseline이 ~1dB 높지만, **perceptual quality에서 alpha=1.0이 압도적**:
- LPIPS: 0.015 vs 0.043 (**2.9x** 개선)
- SSIM: 0.974 vs 0.947 (+0.027)
- Alpha IoU: 0.956 (우수한 shape accuracy)

**권장**: 논문 제출 시 **alpha=1.0 기본 설정** 채택 (multi-metric 관점 우수)

→ 상세: [hypotheses/H6_ALPHA_MASK.md](./hypotheses/H6_ALPHA_MASK.md)

---

## H7: SSIM Weight ⏳

> SSIM loss weight를 높이면 구조적 보존이 개선되는가?

| 항목 | 값 |
|------|-----|
| 가설 | SSIM weight↑ → 구조 선명도↑, 가능 PSNR 소폭↓ |
| 상태 | ⏳ H4 완료 후 진행 |
| 관찰 | train/ssim_loss 초기 감소 후 증가 (weight 0.1 너무 낮음) |
| 설정 | ssim_weight=0.3/0.5/1.0 (기본 0.1) |
| Configs | 4view_ssim03_v2, 4view_ssim05_v2, 4view_ssim10_v2 |
| 문헌 | 3DGS(0.2), Instant-3D(0.5), Splatter Image(~0.5) |

→ 상세: [hypotheses/H7_SSIM_WEIGHT.md](./hypotheses/H7_SSIM_WEIGHT.md)

---

## H8: Reduced View Generation ✅

> MV-Diffusion 생성 뷰 수를 6→3~4로 줄이면 per-view 품질이 개선되는가?

| 항목 | 값 |
|------|-----|
| 가설 | 적은 뷰 = Attention 집중 → per-view 품질↑, inconsistency↓ |
| 상태 | ✅ **완료 — 뷰 감소는 품질 개선 불가** |
| 근거 | Era3D(fewer tokens=better), InstantMesh(fewer=less inconsistency), LGM/GRM(4뷰 SOTA) |
| 전제 | ✅ H4에서 3view(19.92) vs 4view(21.50) = 1.58dB gap |
| 교차 | H4 (뷰 수) + H5 (MVDiff 품질) → H8 |

### H8 실험 진행 상황 (260209)

| 실험 | 설정 | 상태 | 결과 |
|------|------|------|------|
| 4-view E2E | camera_indices=[0,2,3,5], views=[2,3,5] | ✅ 완료 | PSNR_wh=**19.87** |
| 3-view MVDiff training | camera_indices=[0,2,4], ckpt-10000 | ✅ 완료 | |
| 3-view E2E | 3-view MVDiff → 3-view GS-LRM | ✅ 완료 | PSNR_wh=**22.67** (overall) |

### H8 E2E 분석 (260211)

| 실험 | Views Eval | Overall PSNR_wh | Novel Avg | Input PSNR |
|------|:----------:|:---------------:|:---------:|:----------:|
| 6-view E2E (H3-bis) | 6 | 21.21 | ~24.0 | ~39.5 |
| 4-view E2E | 4 | 19.87 | - | - |
| **3-view E2E** | **3** | **22.67** | **16.26** | **35.51** |

> ⚠️ **Overall 직접 비교 불가**: 3-view(input 1/3)와 6-view(input 1/6)는 input view 비율이 다름.
> Input view(35+ dB)가 overall을 끌어올리므로, view 수가 적을수록 overall이 과대평가됨.
>
> **의미 있는 비교**: Novel view avg 기준 (3-view: 16.26 dB)
> 이전 E1 6-view novel avg ~24.0 dB 대비 **-7.7 dB** → 3-view E2E novel 품질 크게 하락.
>
> **결론**: 뷰 수 감소(6→3)가 per-view 품질 개선으로 이어지지 않음.
> GS-LRM 입력 뷰 부족이 per-view 품질 저하를 압도함.

→ 상세: hypothesis_roadmap.md (H8) 에 통합
→ 종합 분석: hypothesis_roadmap.md (H8) 에 통합

---

## 관련 문서

### 전처리
| 문서 | 내용 |
|------|------|
| [PREPROCESSING_REGISTRY.md](./datasets/PREPROCESSING_REGISTRY.md) | 전처리 SSOT |
| [M5_SERIES_SPEC.md](./datasets/M5_SERIES_SPEC.md) | M5 계열 상세 |

### 실험
| 문서 | 내용 |
|------|------|
| [hp_preprocessing_ablation.md](./experiments/hp_preprocessing_ablation.md) | HP Ablation (진행 중) |
| [H1bis_v2_REVALIDATION.md](./hypotheses/H1bis_v2_REVALIDATION.md) | H3-bis 재검증 |
| [H4_VIEW_ABLATION.md](./hypotheses/H4_VIEW_ABLATION.md) | View Ablation + 명령어 |
| [H5_MVDIFFUSION.md](./hypotheses/H5_MVDIFFUSION.md) | MVDiffusion 실험 |
| [H6_ALPHA_MASK.md](./hypotheses/H6_ALPHA_MASK.md) | Alpha Mask |
| [H7_SSIM_WEIGHT.md](./hypotheses/H7_SSIM_WEIGHT.md) | SSIM Weight |
| [hypothesis_roadmap.md (H8)](./experiments/hypothesis_roadmap.md) | 뷰 수 감소 생성 |
| *(archived — key findings in hypothesis_roadmap.md H8)* | 문헌 기반 개선 서베이 |
| [COMMANDS.md](./experiments/COMMANDS.md) | 실험 명령어 SSOT |
| [EXPERIMENT_REGISTRY.md](./experiments/EXPERIMENT_REGISTRY.md) | 실험 레지스트리 |

### 보고서
| 문서 | 내용 |
|------|------|
| [h1_diagnosis_report.md](../outputs/reports/h1_diagnosis_report.md) | H3 병목 분석 |
| [view_ablation_report.md](../outputs/reports/view_ablation_report.md) | View Ablation |
| [comprehensive_analysis_report.md](../experiments/comparison/comprehensive_analysis_report.md) | Tier A/B/C + View Ablation 종합 분석 |

---

## 실험 로드맵 (260209 업데이트)

### Phase 1: View Ablation ✅ 완료

| Views | Val PSNR | Best Step | 상태 |
|:-----:|:--------:|:---------:|:----:|
| 6 | **24.49** | 4,201 | ✅ |
| 5 | 23.02 | 13,101 | ✅ |
| 4 | 21.71 | 9,201 | ✅ |
| 3 | 20.01 | 10,701 | ✅ |
| 2 | 17.75 | 11,801 | ✅ |
| 1 | 11.08 | 2,401 | ✅ |
| baseline | 15.99 | 0 | ✅ |

**최종 결론**: 뷰 수↑ = PSNR↑ (단조 증가, diminishing returns 없음). 6-view 최적.

### Phase 1.5: H8 MVDiff + E2E (Phase 1과 병렬)

| 순위 | 실험 | 상태 | 결과 |
|------|------|------|------|
| P1-1 | H8 4-view E2E | ✅ 완료 | PSNR_wh=19.87 |
| P1-2 | H5/H8 3-view MVDiff training | 🔄 step 7600/10000 | ~5h |
| P1-3 | H8 3-view E2E | ⏳ P1-2 완료 후 | |

### Phase 2: Loss/Mask Ablation (Phase 1 완료 후)

| 순위 | 가설 | 실험 | Config | 변경 변수 |
|------|------|------|--------|----------|
| P2-1 | H6 | alpha_weight=0.1 | 4view_alpha01_v2 | alpha_loss_weight |
| P2-2 | H6 | alpha_weight=0.5 | 4view_alpha05_v2 | alpha_loss_weight |
| P2-3 | H6 | GT mask + alpha | 4view_maskgt_v2 | mask_mode+alpha+masked_l2 |
| P2-4 | H7 | ssim_weight=0.3 | 4view_ssim03_v2 | ssim_loss_weight |
| P2-5 | H7 | ssim_weight=0.5 | 4view_ssim05_v2 | ssim_loss_weight |
| P2-6 | H7 | ssim_weight=1.0 | 4view_ssim10_v2 | ssim_loss_weight |

GPU 4개 x 2-3 실험 = 1~2 라운드

### Phase 3: MVDiffusion + GS-LRM Systematic Improvement ✅ COMPLETE

> P0/P1 결과 분석 → LR 개선 + Pose 변형 설계 → **모든 전략 수렴, 아키텍처 한계 확인**

| Round | 실험 | 핵심 변경 | 상태 | 결과 |
|:-----:|------|----------|:----:|:----:|
| **R1** | E1 (20K cosine) + E2 (P0 resume LR=1e-5) | LR scheduling | ✅ | 7.90 / 8.20 dB |
| **R1** | E3 (extrinsic+add) | Pose integration | ✅ | 8.10 dB |
| **R2** | E4 (spherical+add) | Pose integration | ⏳ | - |
| **R3** | E5 (alpha=0.3) | GS-LRM alpha 최적점 | ⏳ H6 v3 후 | - |

**Phase 3 결론**: 모든 전략(baseline/cfgr/E1/E2/E3)이 PSNR_gt 7.9-8.4 dB로 수렴. Training strategy로는 MVDiff 아키텍처 병목 돌파 불가. 아키텍처 변경 필요.

이전 실험 (완료):
| 순위 | 실험 | 상태 |
|------|------|:----:|
| P3-0 | cfgr 품질 평가 | ✅ (-0.48 dB, 기각) |
| P3-1 | P0 randref_sparse (10K) | ✅ oscillation → E2로 계속 |
| P3-2 | P1 pose_spherical (10K) | ✅ plateau → E3/E4로 개선 |
| P3-3 | H8 MVDiff 3-view | ✅ 뷰 감소 비효과적 |

### Phase 4: 전처리 Ablation (Phase 1 GPU 해제 후)

| 순위 | 가설 | 실험 | 상태 |
|------|------|------|------|
| P4-1 | HP | M5_4 (Center + No Norm) | ⏳ config 준비완료 |
| P4-2 | HP | M5_5 (Center + Per-view) | ⏳ config 준비완료 |
| P4-3 | HP | M0 (No Center + No Norm) | ⏳ config 준비완료 |

→ 상세: [experiments/hp_preprocessing_ablation.md](./experiments/hp_preprocessing_ablation.md)

### 핵심 의사결정 포인트

```
Phase 1 (학습중)
    │
    ├─ 3view ≈ 4view (차이 <2dB) → H8 (3-4뷰 MVDiff) 진행 ✅ 확인
    ├─ 4view >> 3view → H8은 4뷰로 한정
    └─ 5-6view 최적 → H8 보류, Phase 2 집중
    │
Phase 1.5 (병렬 진행중)
    │
    ├─ 4-view E2E: 19.87 (vs 6-view E2E: 21.21 = -1.34 dB)
    └─ 3-view E2E: pending
    │
H5 cfgr 분기 (완료)
    │
    └─ cfgr < baseline → Sparse attention 유지
       → randref_sparse, 20k_sparse 진행
    │
Phase 2 완료
    │
    ├─ Alpha/SSIM이 유의미 → 최적 조합으로 Phase 3
    └─ 차이 미미 → 기본 설정으로 Phase 3
```

### MVDiff Cyclic 결론 (260207)

| 항목 | 내용 |
|------|------|
| 상태 | checkpoint-9000 (10K 중) |
| 품질 | ❌ 뷰 간 일관성 부족 |
| 원인 | `sparse_mv_attention: false` + `reference_view_idx: all` 동시 변경 → 변수 혼재 |
| 결정 | **Cyclic 중단, Phase 1 (H4) 결과 우선 확인** |
| M5t2 baseline | ✅ E2E gap -0.05 → **충분히 학습됨** |

---

## 데이터셋 요약

| 이름 | Split | Train | 용도 |
|------|-------|-------|------|
| **M5t2** | 8:1:1 temporal | 2,880 | **기본** ⭐ |
| M5t | 1:1:1 temporal | 1,198 | PoseSplatter 비교 |
| M5 | - | - | 전처리 기준 |

---

*MoC v8.0 | FaceLift Research Dashboard | 260220*
