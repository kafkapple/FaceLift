# FaceLift 연구 가설 대시보드

> **Map of Content (MoC)** - 모든 가설과 실험의 메인 허브
> 
> Last Updated: 2026-02-07 | Project: FaceLift

---

## 📊 Executive Summary

| 가설 | 핵심 질문 | 상태 | 결과 |
|------|----------|------|------|
| **[HP](#hp-preprocessing)** | 전처리 기하학 설정은? | ✅ | Batch Uniform + PP=256 |
| [H0](#h0-poc) | PoC 가능한가? | ✅ | PSNR ~27 (train) |
| [H1](#h1-temporal-split) | Temporal split이 유효한가? | ✅ | Data leakage 방지 확인 |
| [H2](#h2-data-amount) | 학습 데이터 양이 중요한가? | ✅ | 다양성 > 반복 학습 |
| [H3](#h3-e2e-bottleneck) | E2E 파이프라인 병목은? | ✅ | MVDiffusion (undertrained 시) |
| [H4](#h4-view-ablation) | 최적 입력 뷰 수는? | 🔄 | Uniform v2 실행 중 |
| [H5](#h5-mvdiffusion) | MVDiffusion 개선 방법은? | 🔄 | Cyclic Aug 진행중 |
| [H6](#h6-alpha-mask) | Alpha mask가 효과적인가? | ⏳ | 대기 |
| [H7](#h7-ssim-weight) | SSIM weight 최적값은? | ⏳ | 대기 |
| [H8](#h8-reduced-view-generation) | 생성 뷰 감소로 품질↑? | ⏳ | H4 의존 |

---

## 🔬 핵심 정량 결과

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

### H4 View Ablation (Inference)

| Views | PSNR | 비고 |
|-------|------|------|
| **3** | **21.12** | ⭐ Best |
| 2 | 20.20 | |
| 4-6 | 19.58 | Baseline |

---

## HP: Preprocessing (기초 MVG 가설) ✅

> **PoC 성립을 위한 필수 전처리 설정**
>
> → 상세: [datasets/PREPROCESSING_REGISTRY.md](./datasets/PREPROCESSING_REGISTRY.md)
> → 상세: [datasets/M5_SERIES_SPEC.md](./datasets/M5_SERIES_SPEC.md)

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

### HP 실험 매트릭스 (2×3 Factorial)

|  | **No Norm** | **Per-view** | **Batch Uniform** |
|--|-------------|--------------|-------------------|
| **Center (PP=256)** | M5_4 | M5_5 | **M5** ⭐ |
| **No Center (PP 가변)** | M0 | M0_n | - |

**권장 조합**: M5 (Center + Batch Uniform)

---

### HP 현황

| 상태 | 내용 |
|------|------|
| ✅ 완료 | 정규화 방식 분석 및 문서화 |
| ✅ 완료 | M5 계열 프리셋 구현 |
| ✅ 완료 | 2×3 Ablation 설계 |
| ⏳ 대기 | Ablation 실험 (H4 완료 후) |

### HP Ablation 실행 명령어 (H4 완료 후)

```bash
cd /home/joon/dev/FaceLift
mkdir -p logs

# GPU 4: M5 (baseline - Center + Batch Uniform)
export CUDA_VISIBLE_DEVICES=4 && \
nohup python train_gslrm.py -d M5t2 -e E0_1_facelift \
  > logs/hp_M5_baseline.log 2>&1 &

# GPU 5: M5_4 (Center + No Norm)
export CUDA_VISIBLE_DEVICES=5 && \
nohup python train_gslrm.py \
  -b configs/base/gslrm_mouse.yaml \
  -e configs/mouse/hp_ablation/M5_4.yaml \
  > logs/hp_M5_4.log 2>&1 &

# GPU 6: M5_5 (Center + Per-view Norm)
export CUDA_VISIBLE_DEVICES=6 && \
nohup python train_gslrm.py \
  -b configs/base/gslrm_mouse.yaml \
  -e configs/mouse/hp_ablation/M5_5.yaml \
  > logs/hp_M5_5.log 2>&1 &

# GPU 7: M0_n (No Center + Per-view Norm)
export CUDA_VISIBLE_DEVICES=7 && \
nohup python train_gslrm.py \
  -b configs/base/gslrm_mouse.yaml \
  -e configs/mouse/hp_ablation/M0_n.yaml \
  > logs/hp_M0_n.log 2>&1 &
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
| 결과 | ✅ **다양성 > 반복 학습** (2880×6ep > 1198×20ep) |
| 결론 | **M5t2 (8:1:1) 패러다임 채택** |
| 다음 | H3 (E2E 분석) |

---

## H3: E2E Bottleneck ✅

> MVDiffusion → GS-LRM 파이프라인에서 병목은 어디인가?

| 항목 | 값 |
|------|-----|
| 결과 | ✅ **MVDiffusion이 병목** (undertrained 시 +1.41 gap) |
| 결론 | M5t2 충분 학습 시 gap 거의 없음 |
| 다음 | H4 (View Ablation), H5 (MVDiff 개선) |

→ 상세 보고서: [outputs/reports/h1_diagnosis_report.md](../outputs/reports/h1_diagnosis_report.md)

---

## H4: View Ablation 🔄

> 최적의 입력 뷰 수는 몇 개인가?

| 항목 | 값 |
|------|-----|
| 가설 | 3-4 view가 최적 |
| 상태 | 🔄 **Round 1 실행 중** (1-5view) |
| baseline | ✅ true zero-shot 준비 완료 (validate_before_training + early_stop_after_epochs=0) |

→ **상세 + 명령어**: [experiments/H4_VIEW_ABLATION.md](./experiments/H4_VIEW_ABLATION.md)

---

## H5: MVDiffusion 🔄

> MVDiffusion fine-tuning으로 E2E 품질을 개선할 수 있는가?

### H5 실험 매트릭스 (260207 재설계)

**원칙**: Baseline (sparse=true, ref=0)에서 **단일 변수만** 변경

| Config | sparse_mv | ref_view | steps | 변경 변수 | 상태 |
|--------|:---------:|:--------:|:-----:|----------|:----:|
| **M5t2** (baseline) | true | 0 | 10K | - | ✅ ckpt-5000 |
| **M5t2_cfgr** | **false** | 0 | 10K | sparse attention | ✅ 학습 완료 |
| **M5t2_randref_sparse** | true | **random** | 10K | ref augmentation | 🆕 대기 |
| **M5t2_20k_sparse** | true | 0 | **20K** | 학습 길이 | 🆕 대기 |

**폐기된 실험:**
| Config | 사유 |
|--------|------|
| ~~M5t2_cyclic~~ | ❌ 삭제 (2변수 동시 변경: sparse+ref, 27GB 해제) |
| M5t2_randref | ⚠️ 2변수 (sparse=false + ref=random) |
| M5t2_symmetric | ⚠️ 2변수 (sparse=false + ref=[0,3]) |
| M5t2_20k | ⚠️ 2변수 (sparse=false + 20K) |

### H5 가설별 검증 계획

| 가설 | 실험 | 검증 방법 |
|------|------|----------|
| Full attention이 품질 향상? | cfgr vs baseline | PSNR, 뷰 일관성 비교 |
| Random ref가 다양성 증가? | randref_sparse vs baseline | E2E PSNR, 정성 평가 |
| 학습 부족이 원인? | 20k_sparse vs baseline | 수렴 곡선 비교 |

### H5 의사결정 흐름

```
cfgr 품질 확인 (이미 학습됨)
    │
    ├─ cfgr > baseline → Full attention 채택, 그 위에 randref/20k 테스트
    ├─ cfgr ≈ baseline → Sparse 유지, randref_sparse + 20k_sparse 진행
    └─ cfgr < baseline → Sparse attention이 핵심, 다른 개선 탐색
```

→ 상세: [experiments/H5_MVDIFFUSION.md](./experiments/H5_MVDIFFUSION.md)

---

## H6: Alpha Mask ⏳

> Rendered alpha mask supervision이 foreground 품질을 개선하는가?

| 항목 | 값 |
|------|-----|
| 가설 | Alpha supervision → shape 수렴 가속, boundary 개선 |
| 상태 | ⏳ H4 완료 후 진행 |
| 설정 | alpha_w=0.1/0.5, mask_mode=gt |
| Configs | 4view_alpha01_v2, 4view_alpha05_v2, 4view_maskgt_v2 |

→ 상세: [experiments/H6_ALPHA_MASK.md](./experiments/H6_ALPHA_MASK.md)

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

→ 상세: [experiments/H7_SSIM_WEIGHT.md](./experiments/H7_SSIM_WEIGHT.md)

---

## H8: Reduced View Generation ⏳

> MVDiffusion 생성 뷰 수를 6→3~4로 줄이면 per-view 품질이 개선되는가?

| 항목 | 값 |
|------|-----|
| 가설 | 적은 뷰 = Attention 집중 → per-view 품질↑, inconsistency↓ |
| 상태 | ⏳ **H4 결과 의존** (3view PSNR 확인 후 결정) |
| 근거 | Era3D(fewer tokens=better), InstantMesh(fewer=less inconsistency), LGM/GRM(4뷰 SOTA) |
| 전제 | H4에서 3view가 4view 대비 -2dB 이내 |
| 교차 | H4 (뷰 수) + H5 (MVDiff 품질) → H8 |

→ 상세: [experiments/H8_REDUCED_VIEW_GENERATION.md](./experiments/H8_REDUCED_VIEW_GENERATION.md)

---

## 📁 관련 문서

### 전처리
| 문서 | 내용 |
|------|------|
| [PREPROCESSING_REGISTRY.md](./datasets/PREPROCESSING_REGISTRY.md) | 전처리 SSOT |
| [M5_SERIES_SPEC.md](./datasets/M5_SERIES_SPEC.md) | M5 계열 상세 |

### 실험
| 문서 | 내용 |
|------|------|
| [H4_VIEW_ABLATION.md](./experiments/H4_VIEW_ABLATION.md) | View Ablation + 명령어 |
| [H5_MVDIFFUSION.md](./experiments/H5_MVDIFFUSION.md) | MVDiffusion 실험 |
| [H6_ALPHA_MASK.md](./experiments/H6_ALPHA_MASK.md) | Alpha Mask |
| [H7_SSIM_WEIGHT.md](./experiments/H7_SSIM_WEIGHT.md) | SSIM Weight |
| [H8_REDUCED_VIEW_GENERATION.md](./experiments/H8_REDUCED_VIEW_GENERATION.md) | 뷰 수 감소 생성 |
| [COMMANDS.md](./experiments/COMMANDS.md) | 실험 명령어 SSOT |
| [EXPERIMENT_REGISTRY.md](./experiments/EXPERIMENT_REGISTRY.md) | 실험 레지스트리 |

### 보고서
| 문서 | 내용 |
|------|------|
| [h1_diagnosis_report.md](../outputs/reports/h1_diagnosis_report.md) | H3 병목 분석 |
| [view_ablation_report.md](../outputs/reports/view_ablation_report.md) | View Ablation |

---

## 🚀 실험 로드맵 (260207 업데이트)

### Phase 1: View Ablation (현재 진행 중)

| GPU | 실험 | Steps | 진행률 | 예상 완료 |
|-----|------|-------|--------|----------|
| 5 | 4view_v2 | /15840 | 🔄 | |
| 6 | 1view_v2 + 2view_v2 | /15840 | 🔄 | |
| 7 | 3view_v2 + 5view_v2 | /15840 | 🔄 | |
| 4 | baseline_v2 (zero-shot) | 즉시 | ⏳ 대기 | |
| 4 | 6view_v2 | /15840 | ⏳ baseline 후 | |

**완료 시 결정**: 최적 뷰 수 확정 → Phase 2 실험 뷰 수 결정

### Phase 2: Loss/Mask Ablation (H4 R1 완료 후)

| 순위 | 가설 | 실험 | Config | 변경 변수 |
|------|------|------|--------|----------|
| P2-1 | H6 | alpha_weight=0.1 | 4view_alpha01_v2 | alpha_loss_weight |
| P2-2 | H6 | alpha_weight=0.5 | 4view_alpha05_v2 | alpha_loss_weight |
| P2-3 | H6 | GT mask + alpha | 4view_maskgt_v2 | mask_mode+alpha+masked_l2 |
| P2-4 | H7 | ssim_weight=0.3 | 4view_ssim03_v2 | ssim_loss_weight |
| P2-5 | H7 | ssim_weight=0.5 | 4view_ssim05_v2 | ssim_loss_weight |
| P2-6 | H7 | ssim_weight=1.0 | 4view_ssim10_v2 | ssim_loss_weight |

GPU 4개 × 2-3 실험 = 1~2 라운드

### Phase 3: MVDiffusion (Phase 1-2 분석 후)

| 순위 | 가설 | 실험 | 의존성 |
|------|------|------|--------|
| P3-0 | H5 | **cfgr 품질 평가** (이미 학습됨) | - |
| P3-1 | H5 | randref_sparse (random ref + sparse) | P3-0 결과 |
| P3-2 | H5 | 20k_sparse (longer training) | P3-0 결과 |
| P3-3 | H8 | MVDiff 4-view fine-tune | H4 결과 |

### Phase 4: 전처리 Ablation (선택)

| 순위 | 가설 | 실험 |
|------|------|------|
| P4 | HP | 2×3 Centering/Norm Ablation |

### 핵심 의사결정 포인트

```
Phase 1 완료
    │
    ├─ 3view ≈ 4view (차이 <2dB) → H8 (3-4뷰 MVDiff) 진행
    ├─ 4view >> 3view → H8은 4뷰로 한정
    └─ 5-6view 최적 → H8 보류, Phase 2 집중
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

*MoC v4.2 | FaceLift Research Dashboard | 260207*
