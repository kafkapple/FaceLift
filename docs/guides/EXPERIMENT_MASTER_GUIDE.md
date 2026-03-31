# FaceLift Experiment Master Guide

> **Purpose**: FaceLift mouse 3D reconstruction 프로젝트의 전체 실험 흐름을 초심자가 처음부터 이해하고 재현할 수 있도록 안내합니다.
>
> ← [[INDEX]] | **Version**: v1.0 | **Updated**: 2026-02-23

---

## 1. Project Overview

**FaceLift**는 단일 이미지에서 3D Gaussian Splatting을 생성하는 2-stage feed-forward 파이프라인입니다.

```
Single Image ──→ [Multi-view Diffusion] ──→ 6-view Images ──→ [GS-LRM] ──→ 3D Gaussians
                  (Stage 1-2)                                  (Stage 3)
                  SD2.1-UnCLIP + Era3D RMA                    Transformer
```

> **용어 주의**: 코드 폴더 `mvdiffusion/`과 클래스명 `MVDiffusion*`은 Era3D 코드베이스에서 상속된 명칭입니다.
> Tang et al.의 "MVDiffusion" 논문과는 **별개**입니다. 본 프로젝트의 Stage 1은 **SD2.1-UnCLIP + Era3D Row-wise Multi-view Attention (RMA)** 기반입니다.

- **Multi-view Diffusion (Stage 1)**: SD2.1-UnCLIP + Era3D RMA 기반, 1장 → 6장 multi-view 이미지 생성
- **GS-LRM**: Transformer 기반, multi-view 이미지 → 16,386개 3D Gaussian 예측
- **Mouse 도메인**: DANNCE/MAMMAL 데이터셋 (6-camera, 3600 frames, M5 전처리)에 적용

---

## 2. Experiment Roadmap

아래 순서대로 읽으면 전체 실험 흐름을 이해할 수 있습니다.

```
[CH1] 환경 & 데이터                    [CH2] 코드 구조
  │  ├─ gpu03 서버 설정                   │  ├─ Config 3-Layer 시스템
  │  ├─ M5 데이터셋 구조                  │  ├─ GS-LRM 학습 루프
  │  ├─ 전처리 파이프라인                  │  ├─ GSLRM Forward Pass (14단계)
  │  └─ MouseViewDataset 코드             │  └─ Loss 계산 & Validation
  │                                        │
  └──────────────┬─────────────────────────┘
                 │
           [CH3] 실험 & 결과
             │
             ├─ Phase 1: GS-LRM Baseline (E0_1)
             │   └─ PSNR 22.34 (4-view, M5t2)
             │
             ├─ Phase 2: GS-LRM Ablation
             │   ├─ H4: View Ablation (1v→6v) → 6v=23.84, 1v→2v 최대 gain
             │   ├─ H6: Alpha Mask → ❌ 기각 (monotonic decline)
             │   ├─ H7: SSIM Weight → ❌ 기각 (collapse at 0.5+)
             │   └─ HP: Preprocessing → M5_4=21.80 beats baseline!
             │
             ├─ Phase 3: Multi-view Diffusion Optimization
             │   ├─ H5: Random Ref + Sparse Attn + Resume 20K
             │   ├─ Pose Conditioning (spherical, extrinsic)
             │   └─ 전략 수렴: 7.9~8.4 dB ceiling (아키텍처 한계)
             │
             ├─ Phase 4: E2E Pipeline & Bottleneck Analysis
             │   ├─ MVDiff transfer rate ~14%
             │   ├─ H_T1: Distribution Mismatch 가설
             │   └─ DA1: Domain Adaptation (best 10.08 dB, +1.64)
             │
             └─ Phase 5: Fair Comparison (FL vs PS)
                 ├─ FL GS-LRM 6v = 23.84 >> PS M5 = 13.78
                 └─ 9-Experiment Matrix (Temporal + Spatial)
```

---

## 3. Quick Reference: All Experiments

| ID | 가설 | Config Family | Config | Val PSNR | PSNR_fg (fair) | 상태 |
|----|------|:------------:|--------|:--------:|:---------:|:----:|
| **E0_1** | FaceLift baseline | **Original** | `E0_1_facelift` | 22.34 | — | ✅ 완료 |
| **H4-1v** | 1-view sanity | Uniform v2 | `1view_v2` | 11.08 | 10.47 | ✅ |
| **H4-2v** | 2-view minimum | Uniform v2 | `2view_v2` | 17.75 | 15.95 | ✅ |
| **H4-3v** | 3-view | Uniform v2 | `3view_v2` | 20.01 | 18.56 | ✅ |
| **H4-4v** | 4-view (uniform) | Uniform v2 | `4view_v2` | 21.71 | 20.66 | ✅ |
| **H4-5v** | 5-view | Uniform v2 | `5view_v2` | 23.02 | 22.16 | ✅ |
| **H4-6v** | 6-view upper bound | Uniform v2 | `6view_v2` | 24.49 | 23.84 | ✅ |

> **⚠️ Config Note**: E0_1은 논문 원본 config (`E0_1_facelift.yaml`), H4 시리즈는 통제된 ablation용 `base_uniform_v2.yaml` 기반.
> E0_1(22.34) vs H4-4v(21.71)의 0.63 dB 차이는 config 차이에 기인하며 직접 비교 불가.
> **PSNR_fg (fair)** = fair eval 기준 foreground-masked PSNR (test set). 비교 시 이 값 사용 권장.
| **H6-α0.3** | Alpha mask 0.3 | `4view_alpha03_v3` | - | 21.34 | ❌ 기각 |
| **H6-α0.5** | Alpha mask 0.5 | `4view_alpha05_v3` | - | 21.20 | ❌ 기각 |
| **H6-α1.0** | Alpha mask 1.0 | `4view_alpha10_v3` | - | 20.84 | ❌ 기각 |
| **H7-s0.3** | SSIM weight 0.3 | `4view_ssim03_v2` | - | 21.10→19.69 | ❌ 기각 |
| **H7-s0.5** | SSIM weight 0.5 | `4view_ssim05_v2` | - | 21.30→10.17 | ❌ collapse |
| **H7-s1.0** | SSIM weight 1.0 | `4view_ssim10_v2` | - | 21.20→4.72 | ❌ collapse |
| **HP-M0** | Raw data | `hp_M0` | - | 발산 | ❌ |
| **HP-M5_4** | Center only | `hp_M5_4` | 7 | **21.80** | 🔄 학습중 |
| **HP-M5_5** | Center+norm | `hp_M5_5` | 4 | 21.58 | 🔄 학습중 |
| **H5-E2** | Best Stage 1 | `M5t2_randref_sparse` | - | val 27.70 | ✅ |
| **H3** | E2+Pose conditioning | `M5t2_H3_resume_pose` | 5 | — | 🔄 step 4200/10K |
| **DA1** | Domain adaptation | `domain_adapt_E2_v1` | 6 | 10.08 | 🔄 학습중 |
| **PS-6v** | Pose-Splatter 6v | PS config | - | 13.78 | ✅ |
| **PS-5v** | Pose-Splatter 5v | `m5_5view_holdout4` | joon | — | 🔄 epoch 35/50 |

---

## 4. Chapter Guide

| Chapter | 내용 | 대상 독자 | 링크 |
|---------|------|-----------|------|
| **CH1** | 환경 설정, 데이터 구조, 전처리 | 프로젝트 시작 | [[chapters/CH1_ENVIRONMENT_AND_DATA]] |
| **CH2** | Config 시스템, 학습 코드, 모델 구조 | 코드 이해 | [[chapters/CH2_GSLRM_CODE_FLOW]] |
| **CH3** | 전체 실험 흐름, 결과, 분석 | 실험 재현 | [[chapters/CH3_EXPERIMENTS_AND_RESULTS]] |

---

## 5. Key Paths (gpu03)

```
/home/joon/dev/FaceLift/                    # Project root
├── train_gslrm.py                          # GS-LRM training entry
├── train_diffusion.py                      # Multi-view diffusion training entry
├── configs/
│   ├── base/gslrm_mouse.yaml              # GS-LRM base config
│   ├── datasets/M5t2.yaml                  # M5t2 dataset config
│   ├── mouse/uniform/                      # Uniform experiment configs
│   │   ├── base_uniform_v2.yaml            # Uniform base (lr=1e-6, 15K steps)
│   │   ├── 4view_v2.yaml                   # H4 baseline
│   │   ├── domain_adapt_E2_v1.yaml         # DA1
│   │   └── hp_M5_4.yaml / hp_M5_5.yaml    # HP ablation
│   └── mvdiffusion/                        # Multi-view diffusion configs (Era3D 코드명)
│       ├── M5t2.yaml                       # MVDiff baseline
│       └── M5t2_H3_resume_pose.yaml        # H3
├── mouse_extensions/                       # Custom code (120+ files)
│   ├── data/mouse_dataset.py               # Dataset class
│   ├── model/                              # Loss, alpha, pose conditioning
│   ├── scripts/eval/fair_comparison.py     # Fair eval
│   └── scripts/domain_adapt/              # DA1 scripts
├── gslrm/model/gslrm.py                   # GSLRM model core
├── outputs/                                # Training outputs
├── experiments/validation/                  # Val metrics CSVs
└── docs/                                   # Documentation
```

```
/home/joon/data/preprocessed/FaceLift_mouse/
├── M5/                                     # Primary dataset (3616 frames)
│   ├── 000000/ ... 003599/                 # Frame directories
│   │   ├── opencv_cameras.json             # 6-view camera params
│   │   └── images/cam_000~005.png          # RGBA 512x512
├── M0/                                     # Raw (no centering, no norm)
├── M5_4/                                   # Centering only
├── M5_5/                                   # Centering + per-view norm
└── M5_mvdiff/                              # MVDiff-generated (DA1)
```

---

## 6. Document Cross-References

| 주제 | 상세 문서 |
|------|-----------|
| FL vs PS 비교 | [[fl_vs_ps_comparison]] |
| MVDiff 병목 분석 | [[mvdiffusion_bottleneck_analysis]] |
| Domain Adaptation | [[domain_adaptation_DA1]] |
| 평가 프로토콜 | [[evaluation_protocol_v1]] |
| 가설 로드맵 | [[hypothesis_roadmap]] |
| 전처리 레지스트리 | [[PREPROCESSING_REGISTRY]] |
| 명령어 모음 | [[COMMANDS]] |
| Config 시스템 | [[EXPERIMENT_CONFIG_GUIDE]] |
| 메트릭 이론 | [[theory/METRICS_PROTOCOL]] |

---

## 7. Conventions

| 항목 | 규칙 |
|------|------|
| **PSNR 표기** | `val/psnr` = validation PSNR (GT views), 소수점 2자리 |
| **Config 참조** | `configs/mouse/uniform/4view_v2.yaml` 형태 |
| **데이터셋 이름** | M5t2 = M5 데이터 + temporal 80:10:10 split |
| **Step 표기** | `fwdbwd_pass_step` (forward-backward pass 기준) |
| **GPU 범위** | gpu03: GPU 4-7 only (0-3은 Blackwell, PyTorch 미지원) |

---

*Experiment Master Guide v1.0 | 2026-02-23*
