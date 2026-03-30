# Evaluation Metrics Protocol for FaceLift

> How to compute PSNR/SSIM/LPIPS for fair comparison with PoseSplatter and 3D reconstruction literature.
>
> **Implementation**: `mouse_extensions/scripts/eval/compute_e2e_metrics.py` v2.0
> **See also**: [evaluation_protocol_v1](./evaluation_protocol_v1.md) — 평가 유형, Split 전략, View Ablation 가이드

---

## 1. Problem Statement

FaceLift E2E 평가에서 **메트릭 프로토콜 차이**로 동일 렌더링의 PSNR이 ~13dB 변동:

| 방식 | PSNR | 원인 |
|------|------|------|
| FG-only masked (v1) | 7.8 | alpha > 0.5 이진 마스크, FG 픽셀만 비교 |
| **White-BG full-image (v2)** | **21.3** | GT를 흰배경 합성 후 전체 이미지 비교 |

원인: 마우스가 512x512 프레임의 **2.3%**만 차지 → BG boost = +16.4 dB.

**해결**: 문헌 표준(White-BG Full-Image)을 기본으로 채택하고, 진단용 FG-only를 함께 보고.

---

## 2. PoseSplatter 비교 분석

### 2.1 실험 설정 비교

| 항목 | PoseSplatter | FaceLift |
|------|-------------|----------|
| **논문** | Goffinet et al. (2025), Duke | - |
| **Task** | Multi-view video → 3DGS | Single image → MVDiff → GS-LRM → 3DGS |
| **입력 (추론)** | 4-6 calibrated cameras (실제 뷰) | E2E: 1장 → MVDiff, GS-LRM standalone: 1-6 GT views |
| **대상** | 마우스, 핀치, 쥐 | 마우스 |
| **3D 표현** | 3D Gaussian Splatting | 3D Gaussian Splatting |
| **추론 속도** | ~30ms (feed-forward) | ~수초 (diffusion + reconstruction) |
| **해상도 (논문)** | 384x512 (Duke 4x ds) | 512x512 |
| **해상도 (M5 Fair)** | 512x512 (576→crop) | 512x512 |
| **Gaussian 수** | ~8.5K (mouse) | 1,572,866 raw → ~12K effective (opacity>0.1) |
| **뷰 생성** | 불필요 (실제 캡처) | MVDiffusion으로 6뷰 생성 |
| **카메라** | 사전 캘리브레이션 | 고정 가상 카메라 |
| **Supervision** | IoU + Masked L1 | L2 + SSIM + LPIPS |
| **FG fraction** | ~10-20% (추정) | **2.3%** (측정) |

### 2.2 메트릭 프로토콜 비교

| | PoseSplatter | FaceLift v2 (구현 완료) |
|---|---|---|
| **PSNR** | White-BG full-image | ✅ `psnr_full_white` |
| **SSIM** | White-BG full-image | ✅ `ssim_full_white` |
| **L1** | Masked: `∑\|x̂-x\| / (3∑m)` | ✅ `masked_l1` |
| **IoU** | Silhouette: rendered α vs GT mask | ✅ `silhouette_iou` (white-bg threshold) |
| **LPIPS** | 보고 안 함 | ✅ `lpips_full_white` |
| **FG-only PSNR** | 보고 안 함 | ✅ `psnr_fg_only` (진단용) |

### 2.3 정량 비교 (실측 데이터)

#### Table A: 동일 프로토콜 비교 (White-BG Full-Image)

| Method | Input | PSNR↑ | SSIM↑ | LPIPS↓ | L1↓ | IoU↑ | FG% |
|--------|-------|:-----:|:-----:|:------:|:---:|:----:|:---:|
| **PoseSplatter** 6-cam | 6 real views | **33.5** | **0.989** | - | **0.317** | **0.868** | ~15% |
| **PoseSplatter** 5-cam | 5 real views | 29.0 | 0.982 | - | 0.632 | 0.760 | ~15% |
| **PoseSplatter** 4-cam | 4 real views | 28.2 | 0.982 | - | 0.753 | 0.721 | ~15% |
| FaceLift H4 6-view GT→GS-LRM | 6 GT views | ~40* | - | - | - | - | 2.3% |
| FaceLift H4 4-view GT→GS-LRM | 4 GT views | ~38* | - | - | - | - | 2.3% |
| **FaceLift H5 E2E** baseline | 1 img → 6 gen | 21.3 | 0.967 | 0.059 | **0.297** | 0.518 | 2.3% |
| **FaceLift H5 E2E** cfgr | 1 img → 6 gen | 20.8 | 0.965 | 0.063 | 0.319 | 0.491 | 2.3% |

> *H4 PSNR은 FG-only (GS-LRM validation) + BG boost 16.4dB로 추정. 실측 필요.

#### Table B: FG-Normalized 비교 (BG 편향 제거)

PSNR에서 BG boost를 제거하여 **순수 FG 재구성 품질**만 비교:

| Method | PSNR_full | FG% | BG boost | **PSNR_fg** |
|--------|:---------:|:---:|:--------:|:-----------:|
| **PoseSplatter** 6-cam | 33.5 | ~15% | +8.2 | **~25.3** |
| **PoseSplatter** 4-cam | 28.2 | ~15% | +8.2 | **~20.0** |
| FaceLift H4 6-view (GT) | ~40 | 2.3% | +16.4 | **~24** |
| FaceLift H4 4-view (GT) | ~38 | 2.3% | +16.4 | **~22** |
| FaceLift H5 E2E baseline | 21.3 | 2.3% | +16.4 | **7.8** |

#### Table C: FG-Fraction-Independent 메트릭 비교

**Masked L1**과 **IoU**는 FG fraction에 무관 → 직접 비교 가능:

| Method | L1↓ | IoU↑ | 비고 |
|--------|:---:|:----:|------|
| **PoseSplatter** 6-cam | 0.317 | **0.868** | 6 실제 뷰 |
| **PoseSplatter** 5-cam | 0.632 | 0.760 | |
| **PoseSplatter** 4-cam | 0.753 | 0.721 | |
| **FaceLift H5 E2E** baseline | **0.297** | 0.518 | 1 이미지 → 6 생성 뷰 |
| **FaceLift H5 E2E** cfgr | 0.319 | 0.491 | |

**핵심 발견**:
- **L1**: FaceLift(0.297) < PoseSplatter 6-cam(0.317) → **FaceLift의 FG 색상 정확도가 더 높음**
- **IoU**: FaceLift(0.518) << PoseSplatter(0.868) → **FaceLift의 형상 재구성 품질이 크게 열등**

**해석**: FaceLift는 전경 픽셀의 **색상**은 잘 복원하지만 **형상(silhouette)**이 부정확.
이는 MV-Diffusion→GS-LRM 파이프라인에서 3D 기하학적 일관성이 부족하기 때문.

### 2.4 직접 비교의 한계

| 요소 | 영향 | 비고 |
|------|------|------|
| **FG fraction 차이** | PSNR +8dB 차이 | 2.3% vs ~15% → 직접 비교 불가 |
| **Task 난이도** | FaceLift가 훨씬 어려움 | single image vs 6 calibrated views |
| **데이터셋** | 다른 마우스, 다른 환경 | 동일 종이지만 다른 설정 |
| **해상도** | M5 Fair Eval에서 양쪽 512x512 통일 | PS 논문은 384x512 (Duke data) |
| **평가 프레임** | 다름 | PoseSplatter: 비디오 기반, FaceLift: 독립 프레임 |

---

## 3. 문헌 메트릭 프로토콜 서베이

### 3.1 Object-Centric 3D Reconstruction

| 논문 | 방식 | BG | Masking | 코드 확인 |
|------|------|----|---------|-----------|
| **LGM** (ECCV'24 Oral) | Full-image | White | 없음 | ✅ `image[:3]*mask+(1-mask)` |
| **GS-LRM** (ECCV'24) | Full-image | White | 추정 | ❌ 비공개 |
| **LRM** | Full-image | White | 없음 | ✅ "pure white background" |
| **Splatter Image** (CVPR'24) | Full-image | White/Black | 없음 | ✅ `eval.py` |
| **InstantMesh** | Full-image | White | 불명 | △ |
| **PoseSplatter** (2025) | Full-image | White | L1만 masked | ✅ 논문 명시 |
| **CO3D Challenge** | Masked | Black | Binary mask | ✅ README |

### 3.2 De Facto Standard

```
Object-centric 3D 분야:
  ┌─────────────────────────────────────────────────────┐
  │  GT: RGBA → alpha composite onto WHITE background   │
  │  Pred: Render with WHITE background                  │
  │  PSNR/SSIM: MSE over FULL image (incl. background) │
  │  L1: Often MASKED (foreground only)                  │
  │  IoU: Silhouette (rendered α vs GT mask)             │
  └─────────────────────────────────────────────────────┘
```

### 3.3 BG Boost 수학적 관계

```
PSNR_full_white ≈ PSNR_fg + 10·log₁₀(1/fg_fraction)
```

| FG fraction | BG boost | 예시 |
|:-----------:|:--------:|------|
| 50% | +3.0 dB | 큰 객체 |
| 25% | +6.0 dB | 중간 객체 |
| 10% | +10.0 dB | 작은 객체 |
| **2.3%** | **+16.4 dB** | **FaceLift 마우스** |

---

## 4. FaceLift 메트릭 프로토콜 (v2.0)

### 4.1 Primary Metrics (논문 보고용)

| 메트릭 | 코드 키 | 용도 |
|--------|---------|------|
| **PSNR** | `psnr_full_white` | 문헌 비교 (LGM, PoseSplatter 등) |
| **SSIM** | `ssim_full_white` | 구조적 유사성 |
| **LPIPS** | `lpips_full_white` | 지각적 품질 |
| **Masked L1** | `masked_l1` | PoseSplatter 직접 비교 |
| **IoU** | `silhouette_iou` | 형상 품질 |

### 4.2 Diagnostic Metrics (내부 분석용)

| 메트릭 | 코드 키 | 용도 |
|--------|---------|------|
| **PSNR_fg** | `psnr_fg_only` | 순수 FG 품질 (BG 편향 제거) |
| **PSNR_alpha** | `psnr_alpha_weighted` | GS-LRM training과 일관 |
| **FG fraction** | `fg_fraction` | BG boost 계산용 |

### 4.3 Usage

```bash
# Standard evaluation (all modes)
python -m mouse_extensions.scripts.eval.compute_e2e_metrics \
    --output_dir outputs/h5_e2e/baseline_ckpt5000 \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --skip_input_view 0

# Comparison (multiple experiments)
python -m mouse_extensions.scripts.eval.compute_e2e_metrics \
    --output_dir outputs/h5_e2e/cfgr_ckpt10000 outputs/h5_e2e/baseline_ckpt5000 \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --skip_input_view 0

# Fast (skip LPIPS)
python -m mouse_extensions.scripts.eval.compute_e2e_metrics \
    --output_dir outputs/h5_e2e/baseline_ckpt5000 \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --skip_input_view 0 --no_lpips
```

Output: `metrics_v2.json` per experiment directory.

---

## 5. 가설 검증 현황 및 실험 우선순위

### 5.1 검증 완료된 가설

| 가설 | 핵심 질문 | 결론 | 메트릭 |
|------|----------|------|--------|
| **HP** | 전처리 설정? | Batch Uniform + PP=256 필수 | - |
| **H0** | PoC 가능? | ✅ PSNR ~27 (train) | FG-only |
| **H1** | Temporal split 유효? | ✅ Data leakage 방지 확인 | - |
| **H2** | 데이터 양? | 다양성 > 반복 학습 | FG-only |
| **H3** | E2E 병목? | MV-Diffusion (undertrained 시) | FG-only |
| **H4** | 최적 뷰 수? | **6-view 최적** (단조 증가) | FG-only (GS-LRM val) |
| **H5** | MVDiff cfgr vs baseline? | **baseline이 소폭 우수** | full_white (v2) |

### 5.2 진행 중

| 가설 | 상태 | GPU | 예상 완료 |
|------|------|-----|-----------|
| **H8** 4-view E2E | P1 추론 실행 중 | 4 | ~1h |
| **H8** 3-view MVDiff | P0 학습 실행 중 | 6 | ~22h |
| **H4** 3view/5view/6view | 학습 계속 진행 (best_psnr 갱신 중) | 5, 7 | 수 시간 |

### 5.3 미검증 가설 및 실험 우선순위

| 순위 | 가설 | 실험 | 필요 GPU | 의존성 | 비고 |
|:----:|------|------|:--------:|--------|------|
| **P0** | H8 | 3-view MVDiff training | 6 (실행중) | - | |
| **P1** | H8 | 4-view E2E inference + metrics_v2 | 4 (실행중) | - | |
| **P2** | H8 | 4-view E2E metrics_v2 평가 | 4 | P1 완료 | 즉시 |
| **P3** | H8 | 3-view E2E inference + metrics_v2 | free | P0 완료 | ~22h 후 |
| **P4** | H7 | SSIM weight 0.3/0.5/1.0 (3 실험) | free x3 | H4 완료 | GPU 확보 시 |
| **P5** | H6 | Alpha supervision ablation | free | H4 완료 | |
| **P6** | - | H4 full_white 재평가 | free | - | PoseSplatter 비교용 |

### 5.4 추가 필요 실험 (PoseSplatter 비교 강화)

| 실험 | 목적 | 비고 |
|------|------|------|
| **H4 metrics_v2 재평가** | GS-LRM GT→render를 full_white로 재계산 | H4와 PoseSplatter PSNR 직접 비교 가능 |
| **FG fraction 분석** | PoseSplatter FG%를 논문 figure에서 추정 | BG boost 보정 정확도 개선 |
| **IoU 개선** | GS-LRM rendered alpha 출력 추가 | 현재 white-bg threshold 기반 → 부정확 |
| **4-view GS-LRM only** | H4 4-view 결과를 PoseSplatter 4-cam과 비교 | 동일 뷰 수 조건 비교 |
| **Crop-and-center 평가** | 마우스 중심 crop 후 재평가 | FG% 증가로 PSNR BG 편향 감소 |

---

## 6. References

- Goffinet et al. "Pose Splatter: A 3D Gaussian Splatting Model for Quantifying Animal Pose and Appearance" (2025). arXiv:2505.18342
- Tang et al. "LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation" (ECCV 2024 Oral). arXiv:2402.05054
- Zhang et al. "GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting" (ECCV 2024)
- Hong et al. "LRM: Large Reconstruction Model for Single Image to 3D" (2023). arXiv:2311.04400
- Szymanowicz et al. "Splatter Image: Ultra-Fast Single-View 3D Reconstruction" (CVPR 2024). arXiv:2312.13150
- CO3D Challenge. github.com/facebookresearch/co3d
- Kulhánek & Sattler. "NerfBaselines" (2024). arXiv:2406.17345

---

## 7. Comprehensive Metric Set (v3.0, 2026-03-26)

> **Code SSOT**: `mouse_extensions/scripts/eval/comprehensive_eval.py`
> **Design SSOT**: Obsidian `docs/analysis/PRUNING_ABLATION_DESIGN.md` §4

### 7.1 Reconstruction Quality (표준)

| Metric | Code Key | Definition | 용도 |
|--------|----------|------------|------|
| **PSNR_gt** | `psnr_gt` | Masked FG only: `MSE = Σ(pred*mask - gt*mask)² / (3·Σmask)` → `-10·log₁₀(MSE)` | **핵심**: BG boost 없는 순수 FG 품질 |
| **PSNR_int** | `psnr_int` | White-BG full image MSE → dB | 문헌 비교 (LGM, PS 등) |
| **SSIM** | `ssim` | Structural similarity on white-BG composite | 텍스처 품질 |
| **LPIPS** | `lpips` | Perceptual distance (AlexNet) on white-BG | 인간 시각 상관 |
| **IoU** | `iou` | Silhouette: `pred_mask ∩ gt_mask / union` (pred_mask = pixel mean < 0.95) | 형태 정확도 |

### 7.2 Artifact-Specific (신규 v3.0)

| Metric | Code Key | Definition | 용도 |
|--------|----------|------------|------|
| **Silhouette Precision** | `sil_precision` | `intersection / pred_fg_area` — 낮으면 배경에 floater 존재 | **Floater 검출** |
| **Silhouette Recall** | `sil_recall` | `intersection / gt_fg_area` — 낮으면 geometry 누락 | Missing geometry |
| **Per-camera PSNR_gt** | `per_cam_psnr_gt` | 각 카메라 (0-5) 별 개별 PSNR_gt | **Bottom view 분리** |
| **N_final** | `n_gaussians` | 필터 후 최종 Gaussian 수 | 효율성 |
| **Opacity mean/std** | `opacity_mean`, `opacity_std` | 분포 특성 | α supervision 효과 |
| **Anisotropy % flat** | `aniso_pct_flat` | ratio ≥ 30인 비율 (%) | 형태 분석 |
| **Floater fraction** | `floater_frac` | DBSCAN 이상치 비율 (main cluster 외) | 배경 노이즈 |

### 7.3 PSNR_gt vs PSNR_int 관계

```
PSNR_int ≈ PSNR_gt + 10·log₁₀(1/fg_fraction)
FG fraction ≈ 2.3% → BG boost ≈ +16.4 dB

예: PSNR_gt = 20.1 → PSNR_int ≈ 36.5 (이론값)
    실측: PSNR_int = 31.1 (white-BG 합성 방식 차이)
```

> ⚠️ 두 metric을 혼용하면 ~10-16 dB 차이 발생. 논문에서 반드시 어떤 기준인지 명시.

### 7.4 Pareto Frontier (Pruning 평가용)

```
Y축: PSNR_gt (dB) — 품질
X축: log(N_final) — 효율성
각 점: filter 조건 (E1~E5)
색상: α 값 (0.0 vs 0.3 vs 1.0)
```

"같은 PSNR_gt에서 N_final 최소화" = Pareto optimal.

### 7.5 PLY 저장 용량 추정

| Pruning 수준 | Gaussians/frame | PLY 크기/frame | 360 test frames | 3600 all frames |
|:------------:|:---------------:|:--------------:|:---------------:|:---------------:|
| Raw (unfiltered) | ~100K | ~24 MB | 8.6 GB | 86 GB |
| apply_all_filters | ~13K | ~3.1 MB | 1.1 GB | 11.2 GB |
| +Visibility+Orient | ~9K | ~2.1 MB | 0.76 GB | 7.6 GB |
| Aggressive | ~5K | ~1.2 MB | 0.43 GB | 4.3 GB |

> PLY 공유 시 filtered (~13K) 권장. Raw PLY는 on-demand 생성.

---

## Related

- ↑ [[../INDEX]] — 문서 허브
- ↔ Obsidian `analysis/PRUNING_ABLATION_DESIGN.md` — Pruning 실험 설계 근거
- ↔ [[../hypotheses/H8_opacity_anisotropy_analysis]] — Opacity/Anisotropy 분석
- ↔ [[../guides/ORIENTATION_FILTER_GUIDE]] — Orientation filter 상세
- ↔ [[../experiments/CHECKPOINT_INVENTORY_260326]] — 체크포인트 현황

---

*FaceLift Metrics Protocol v3.0 | 2026-03-26 | §7 Comprehensive + Artifact metrics 추가*
