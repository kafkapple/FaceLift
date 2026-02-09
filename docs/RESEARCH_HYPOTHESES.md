# FaceLift 연구 가설 대시보드

> **Map of Content (MoC)** - 모든 가설과 실험의 메인 허브
>
> Last Updated: 2026-02-09 | Project: FaceLift

---

## Executive Summary

| 가설 | 핵심 질문 | 상태 | 결과 |
|------|----------|------|------|
| **[HP](#hp-preprocessing)** | 전처리 기하학 설정은? | 🔄 | 이론 ✅, 실험 ablation 대기 |
| [H0](#h0-poc) | PoC 가능한가? | ✅ | PSNR ~27 (train) |
| [H1](#h1-temporal-split) | Temporal split이 유효한가? | ✅ | Data leakage 방지 확인 |
| [H2](#h2-data-amount) | 학습 데이터 양이 중요한가? | ✅ | 다양성 > 반복 학습 |
| [H3](#h3-e2e-bottleneck) | E2E 파이프라인 병목은? | ✅ | MVDiffusion (undertrained 시). H3-bis 완료 |
| [H3-bis](#h3-bis-same-test-set-revalidation) | 동일 테스트셋 재검증? | ✅ | MVDiff 항상 병목 (13-17 dB gap), 데이터↑ → gap↓ ~3 dB |
| [H4](#h4-view-ablation) | 최적 입력 뷰 수는? | 🔄 | 1/2-view ✅, 3/4/5/6-view 학습중 |
| [H5](#h5-mvdiffusion) | MVDiffusion 개선 방법은? | 🔄 | cfgr < baseline (-0.48 dB), 3-view 학습중 |
| [H6](#h6-alpha-mask) | Alpha mask가 효과적인가? | ⏳ | 대기 |
| [H7](#h7-ssim-weight) | SSIM weight 최적값은? | ⏳ | 대기 |
| [H8](#h8-reduced-view-generation) | 생성 뷰 감소로 품질↑? | 🔄 | 4-view E2E PSNR_wh=19.87, 3-view MVDiff 학습중 |

---

## Experiment Status Dashboard (260209)

### GPU Allocation

| GPU | 실험 | 상태 | Step | ETA |
|-----|------|------|------|-----|
| 4 | H4 4-view uniform_v2 | 🔄 resumed (OOM) | 8000/15840 | ~28h |
| 5 | H4 3-view uniform_v2 | 🔄 running | 12750/15840 | ~7h |
| 6 | H4 5-view uniform_v2 | 🔄 running | 8100/15840 | ~42h |
| 7 | H4 6-view uniform_v2 | 🔄 running | 5800/15840 | ~32h |
| (MVDiff) | H8 3-view MVDiff training | 🔄 running | 7600/10000 | ~5h |

### Completed Today

| 실험 | 결과 |
|------|------|
| H4 baseline uniform_v2 | ✅ completed |
| H4 1-view uniform_v2 | ✅ completed (step 15800) |
| H4 2-view uniform_v2 | ✅ completed (step 15800) |
| H5 cfgr evaluation | ✅ PSNR_wh=20.81 (worse than baseline 21.29) |
| H8 4-view E2E | ✅ PSNR_wh=19.87 |
| H3-bis (H1-bis-v2) | ✅ 3-condition revalidation completed |

### Pending (GPU 확보 후)

| 실험 | 의존성 |
|------|--------|
| HP ablation (M0, M5_4, M5_5) | H4 uniform GPU 해제 후 |
| H5 randref_sparse | cfgr 결과 분석 완료 |
| H5 20k_sparse | cfgr 결과 분석 완료 |
| H8 3-view E2E | 3-view MVDiff 학습 완료 후 |

---

## 핵심 정량 결과

### HP 전처리 (카메라 정규화)

| 모드 | 설정 | 기하학 | 결과 |
|------|------|--------|------|
| **Batch Uniform** | `recenter_cameras=True` | ✅ 보존 | ⭐ **권장** |
| Per-view | `normalize_translation=True` | ⚠️ 왜곡 | parallax 오류 |
| None | 둘 다 False | ✅ 보존 | pretrained 불일치 |

### H3 진단 (MVDiffusion 병목)

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

### H4 View Ablation v2 (Uniform, 260209)

| Views | PSNR | Best Step | 비고 |
|-------|------|-----------|------|
| **6** | **23.46** | 601 | ⭐ Best |
| 5 | 22.63 | 2901 | |
| 4 | 21.50 | 3801 | |
| 3 | 19.92 | 3801 | |
| 2 | 17.70 | 8801 | |
| 1 | 11.08 | 2401 | < baseline |
| baseline (0-shot) | 15.99 | 0 | 4-view pretrained |

> ⚠️ 이전 R1 (non-uniform): 3view=21.12 최적 → v2 (uniform): **단조 증가**, 조건 통일의 중요성
>
> ⚠️ 위 결과는 val PSNR (학습 중). 최종 test evaluation은 학습 완료 후 별도 진행 예정

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
> → Ablation 계획: [hypotheses/HP_PREPROCESSING_ABLATION.md](./hypotheses/HP_PREPROCESSING_ABLATION.md)

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

> MVDiffusion → GS-LRM 파이프라인에서 병목은 어디인가?

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

| Condition | 설명 | MVDiffusion | GS-LRM |
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
| A vs B | **-13.80 dB** | MVDiffusion 통과 시 발생하는 본질적 품질 손실 |
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

## H4: View Ablation 🔄

> 최적의 입력 뷰 수는 몇 개인가?

| 항목 | 값 |
|------|-----|
| 가설 | ~~3-4 view가 최적~~ → **6-view 최적 (단조 증가)** (val 기준) |
| 상태 | 🔄 **Uniform v2 학습 진행중** (base_uniform_v2.yaml) |
| 결론 (잠정) | 뷰 수와 PSNR이 단조 증가. 이전 R1(비균일)과 반전 |
| baseline | ✅ zero-shot 4-view = PSNR 15.99 |

### Uniform v2 Training Status (260209)

| Views | Config | 상태 | Step | GPU | 비고 |
|-------|--------|------|------|-----|------|
| baseline | base_uniform_v2 | ✅ completed | - | - | zero-shot |
| 1-view | uniform_v2_1view | ✅ completed | 15800 | - | |
| 2-view | uniform_v2_2view | ✅ completed | 15800 | - | |
| 3-view | uniform_v2_3view | 🔄 running | 12750/15840 | 5 | ~7h remaining |
| 4-view | uniform_v2_4view | 🔄 resumed (OOM) | 8000/15840 | 4 | ~28h remaining |
| 5-view | uniform_v2_5view | 🔄 running | 8100/15840 | 6 | ~42h remaining |
| 6-view | uniform_v2_6view | 🔄 running | 5800/15840 | 7 | ~32h remaining |

> ⚠️ 아래 early results는 val PSNR (학습 중 모니터링). 최종 test evaluation은 모든 학습 완료 후 진행

### H4 Early Results (val PSNR, 학습중)

| Views | PSNR | Best Step | 비고 |
|-------|------|-----------|------|
| **6** | **23.46** | 601 | ⭐ Best (학습중) |
| 5 | 22.63 | 2901 | 학습중 |
| 4 | 21.50 | 3801 | 학습중 |
| 3 | 19.92 | 3801 | 학습중 |
| 2 | 17.70 | 8801 | ✅ |
| 1 | 11.08 | 2401 | ✅ |
| baseline (0-shot) | 15.99 | 0 | ✅ |

→ **상세 + 명령어**: [hypotheses/H4_VIEW_ABLATION.md](./hypotheses/H4_VIEW_ABLATION.md)

---

## H5: MVDiffusion 🔄

> MVDiffusion fine-tuning으로 E2E 품질을 개선할 수 있는가?

### H5 실험 매트릭스 (260209 업데이트)

**원칙**: Baseline (sparse=true, ref=0)에서 **단일 변수만** 변경

| Config | sparse_mv | ref_view | steps | 변경 변수 | 상태 | 결과 |
|--------|:---------:|:--------:|:-----:|----------|:----:|:----:|
| **M5t2** (baseline) | true | 0 | 10K | - | ✅ ckpt-5000 | PSNR_wh=21.29 |
| **M5t2_cfgr** | **false** | 0 | 10K | sparse attention | ✅ 완료 | PSNR_wh=20.81 (**-0.48 dB**) |
| **M5t2_3view** | true | 0 | 10K | camera_indices=[0,2,4] | 🔄 step 7600/10000 (~5h) | H8용 |
| **M5t2_randref_sparse** | true | **random** | 10K | ref augmentation | ⏳ 대기 | |
| **M5t2_20k_sparse** | true | 0 | **20K** | 학습 길이 | ⏳ 대기 | |

**폐기된 실험:**
| Config | 사유 |
|--------|------|
| ~~M5t2_cyclic~~ | ❌ 삭제 (2변수 동시 변경: sparse+ref, 27GB 해제) |
| M5t2_randref | ⚠️ 2변수 (sparse=false + ref=random) |
| M5t2_symmetric | ⚠️ 2변수 (sparse=false + ref=[0,3]) |
| M5t2_20k | ⚠️ 2변수 (sparse=false + 20K) |

### H5 cfgr 결과 분석 (260209)

| Config | sparse_mv | PSNR_wh | vs Baseline |
|--------|-----------|---------|-------------|
| **baseline** | true (sparse) | **21.29** | - |
| cfgr | false (full) | 20.81 | **-0.48 dB** |

**판정**: cfgr < baseline → **Sparse attention이 더 우수**

**의사결정 흐름 결과**:
```
cfgr 품질 확인 (완료)
    │
    └─ cfgr < baseline → ✅ Sparse attention이 핵심
       → randref_sparse + 20k_sparse 진행 (sparse 유지, 다른 변수 탐색)
       → Full attention (cfgr) 경로 폐기
```

### H5 가설별 검증 상태

| 가설 | 실험 | 상태 | 결과 |
|------|------|------|------|
| Full attention이 품질 향상? | cfgr vs baseline | ✅ 완료 | ❌ -0.48 dB (worse) |
| Random ref가 다양성 증가? | randref_sparse vs baseline | ⏳ 대기 | |
| 학습 부족이 원인? | 20k_sparse vs baseline | ⏳ 대기 | |

→ 상세: [hypotheses/H5_MVDIFFUSION.md](./hypotheses/H5_MVDIFFUSION.md)

---

## H6: Alpha Mask ⏳

> Rendered alpha mask supervision이 foreground 품질을 개선하는가?

| 항목 | 값 |
|------|-----|
| 가설 | Alpha supervision → shape 수렴 가속, boundary 개선 |
| 상태 | ⏳ H4 완료 후 진행 |
| 설정 | alpha_w=0.1/0.5, mask_mode=gt |
| Configs | 4view_alpha01_v2, 4view_alpha05_v2, 4view_maskgt_v2 |

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

## H8: Reduced View Generation 🔄

> MVDiffusion 생성 뷰 수를 6→3~4로 줄이면 per-view 품질이 개선되는가?

| 항목 | 값 |
|------|-----|
| 가설 | 적은 뷰 = Attention 집중 → per-view 품질↑, inconsistency↓ |
| 상태 | 🔄 **4-view E2E 완료, 3-view MVDiff 학습중** |
| 근거 | Era3D(fewer tokens=better), InstantMesh(fewer=less inconsistency), LGM/GRM(4뷰 SOTA) |
| 전제 | ✅ H4에서 3view(19.92) vs 4view(21.50) = 1.58dB gap |
| 교차 | H4 (뷰 수) + H5 (MVDiff 품질) → H8 |

### H8 실험 진행 상황 (260209)

| 실험 | 설정 | 상태 | 결과 |
|------|------|------|------|
| 4-view E2E | camera_indices=[0,2,3,5], views=[2,3,5] | ✅ 완료 | PSNR_wh=**19.87** |
| 3-view MVDiff training | camera_indices=[0,2,4] | 🔄 step 7600/10000 (~5h) | |
| 3-view E2E | 3-view MVDiff → GS-LRM | ⏳ MVDiff 완료 후 | |

### H8 4-view E2E 분석

| 비교 | PSNR_wh | Gap | 해석 |
|------|---------|-----|------|
| GS-LRM only (H3-bis Cond.A) | 35.01 | - | Upper bound (GT input) |
| 4-view E2E (P1) | 19.87 | **-15.14 dB** | MVDiff 통과 손실 |
| E2E M5t2 6-view (H3-bis Cond.B) | 21.21 | -13.80 dB | 6-view 비교 기준 |

> 4-view E2E(19.87) vs 6-view E2E(21.21) = -1.34 dB → 뷰 감소 시 GS-LRM 입력 부족이 주요 원인

→ 상세: [hypotheses/H8_REDUCED_VIEW_GENERATION.md](./hypotheses/H8_REDUCED_VIEW_GENERATION.md)

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
| [HP_PREPROCESSING_ABLATION.md](./hypotheses/HP_PREPROCESSING_ABLATION.md) | HP Ablation 계획 |
| [H1bis_v2_REVALIDATION.md](./hypotheses/H1bis_v2_REVALIDATION.md) | H3-bis 재검증 |
| [H4_VIEW_ABLATION.md](./hypotheses/H4_VIEW_ABLATION.md) | View Ablation + 명령어 |
| [H5_MVDIFFUSION.md](./hypotheses/H5_MVDIFFUSION.md) | MVDiffusion 실험 |
| [H6_ALPHA_MASK.md](./hypotheses/H6_ALPHA_MASK.md) | Alpha Mask |
| [H7_SSIM_WEIGHT.md](./hypotheses/H7_SSIM_WEIGHT.md) | SSIM Weight |
| [H8_REDUCED_VIEW_GENERATION.md](./hypotheses/H8_REDUCED_VIEW_GENERATION.md) | 뷰 수 감소 생성 |
| [COMMANDS.md](./experiments/COMMANDS.md) | 실험 명령어 SSOT |
| [EXPERIMENT_REGISTRY.md](./experiments/EXPERIMENT_REGISTRY.md) | 실험 레지스트리 |

### 보고서
| 문서 | 내용 |
|------|------|
| [h1_diagnosis_report.md](../outputs/reports/h1_diagnosis_report.md) | H3 병목 분석 |
| [view_ablation_report.md](../outputs/reports/view_ablation_report.md) | View Ablation |

---

## 실험 로드맵 (260209 업데이트)

### Phase 1: View Ablation 🔄 (uniform v2 학습중)

| Views | Val PSNR | 상태 |
|-------|----------|------|
| 6-view | 23.46 | 🔄 step 5800/15840 |
| 5-view | 22.63 | 🔄 step 8100/15840 |
| 4-view | 21.50 | 🔄 step 8000/15840 (OOM resume) |
| 3-view | 19.92 | 🔄 step 12750/15840 |
| 2-view | 17.70 | ✅ completed |
| 1-view | 11.08 | ✅ completed |
| baseline | 15.99 | ✅ completed |

**잠정 결론**: 뷰 수↑ = PSNR↑ (단조 증가). **H8 진행 결정**: gap < 2dB 기준 충족

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

### Phase 3: MVDiffusion 추가 실험 (Phase 1-2 분석 후)

| 순위 | 가설 | 실험 | 상태 | 의존성 |
|------|------|------|------|--------|
| P3-0 | H5 | **cfgr 품질 평가** | ✅ 완료 (-0.48 dB) | - |
| P3-1 | H5 | randref_sparse (random ref + sparse) | ⏳ | P3-0 완료 |
| P3-2 | H5 | 20k_sparse (longer training) | ⏳ | P3-0 완료 |
| P3-3 | H8 | MVDiff 3-view fine-tune | 🔄 학습중 | H4 결과 |

### Phase 4: 전처리 Ablation (Phase 1 GPU 해제 후)

| 순위 | 가설 | 실험 | 상태 |
|------|------|------|------|
| P4-1 | HP | M5_4 (Center + No Norm) | ⏳ config 준비완료 |
| P4-2 | HP | M5_5 (Center + Per-view) | ⏳ config 준비완료 |
| P4-3 | HP | M0 (No Center + No Norm) | ⏳ config 준비완료 |

→ 상세: [hypotheses/HP_PREPROCESSING_ABLATION.md](./hypotheses/HP_PREPROCESSING_ABLATION.md)

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

*MoC v5.0 | FaceLift Research Dashboard | 260209*
