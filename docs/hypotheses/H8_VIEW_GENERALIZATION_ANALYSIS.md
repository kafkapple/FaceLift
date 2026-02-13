# H8: View Reduction & Generalization — Experiments

> **목적**: 뷰 수 최소화 실험 결과 + 카메라/데이터 일반화 로드맵
> **상태**: 📊 분석 완료 | **Updated**: 2026-02-13
> **이론 배경**: → [MULTIVIEW_DIFFUSION_THEORY](../theory/MULTIVIEW_DIFFUSION_THEORY.md)
> ← [RESEARCH_HYPOTHESES](../RESEARCH_HYPOTHESES.md) | [H8_REDUCED_VIEW_GENERATION](./H8_REDUCED_VIEW_GENERATION.md)

---

## Executive Summary

### 핵심 결과
- **최소 허용 뷰 수 = 4 views** (GS-LRM PSNR 21.71, 6-view 대비 -11.3%)
- 3 views는 경계 (20.01, -18.2%), E2E novel 16.26 dB로 실용성 부족
- 뷰 감소 시 **이중 열화**: MV-Diffusion per-view 품질↓ + GS-LRM coverage 부족
- E2 (aug OFF) PSNR 26.82 ≈ cfgr (aug ON) 26.86 → **augmentation 영향 미미**

### 파이프라인 개요
```
1장 입력 → [MV-Diffusion (SD2.1-UnCLIP + Era3D RMA)] → 6뷰 → [GS-LRM (Plücker ray)] → 3D Gaussian
```
> 모델 계보/이론 상세 → [MULTIVIEW_DIFFUSION_THEORY §1](../theory/MULTIVIEW_DIFFUSION_THEORY.md#1-용어-정의-및-파이프라인-명칭)

### 일반화 로드맵
| 순위 | 작업 | 소요 | 효과 | 이론 참조 |
|:---:|------|:---:|------|-----------|
| **P0** | Random reference 학습 | 1-2일 | Top/Bottom 뷰 지원 | [§4.4](../theory/MULTIVIEW_DIFFUSION_THEORY.md#44-구현-우선순위-근거-포함) |
| P1 | 배경 제거 (SAM/rembg) | 1일 | 임의 배경 입력 | - |
| P2 | Spherical pose conditioning | 1-2주 | 임의 카메라 배치 | [§4.2A](../theory/MULTIVIEW_DIFFUSION_THEORY.md#42-pose-conditioning-3가지-방식-구현됨-미활성) |
| P3 | 4-view 최적화 | 1주 | 카메라 수 최소화 | - |
| P4 | 다종 데이터 + 재학습 | 1개월+ | Subject 일반화 | - |

### 결과 확인 위치
| 종류 | 경로 |
|------|------|
| GS-LRM turntable | `checkpoints/gslrm/base_uniform_v2_{N}view_v2/iter_*/turntable_*.mp4` |
| 3-view E2E | `outputs/h8_e2e/3view/{time_fixed,time_rotating,time_grid_6view}.mp4` |
| 4-view E2E | `outputs/h8_e2e/4view/{time_fixed,time_rotating}.mp4` |
| 6-view E2E | `outputs/h5_e2e/{baseline,cfgr}_*/time_*.mp4` |
| WandB | FaceLift-GS-LRM / FaceLift-MVDiffusion |

> ⚠️ `checkpoints/` = `/node_data/joon/checkpoints/FaceLift`에 대한 **symlink**

---

## 1. GS-LRM View Ablation (GT 입력, H4)

| Views | Val PSNR (dB) | Best Step | Δ from 6-view | 상태 |
|:-----:|:---:|:---:|:---:|:---:|
| **6** | **24.49** | 4,201 | baseline | ✅ |
| 5 | 23.02 | 13,101 | -1.47 | ✅ |
| 4 | 21.71 | 9,201 | -2.78 | ✅ |
| **3** | **20.01** | **10,701** | **-4.48** | ✅ |
| 2 | 17.75 | 11,801 | -6.74 | ✅ |
| 1 | 11.08 | 2,401 | -13.41 | ✅ |
| zero-shot | 15.99 | 0 | -8.50 | ✅ |

```
PSNR (dB)
25 ┤ ─────────── 6v: 24.49
23 ┤       ╱──── 5v: 23.02
21 ┤    ╱─────── 4v: 21.71
20 ┤ ╱────────── 3v: 20.01
18 ┤              2v: 17.75
16 ┤              baseline: 15.99
11 ┤              1v: 11.08
   ├──┬──┬──┬──┬──┬──┐
   0  1  2  3  4  5  6  views
```

| Transition | Δ PSNR | dB/view | 판정 |
|:----------:|:------:|:-------:|:----:|
| 6→5 | -1.47 | 1.47 | ✅ 완만 |
| 5→4 | -1.31 | 1.31 | ✅ 완만 |
| **4→3** | **-1.70** | **1.70** | ⚠️ 경계 |
| 3→2 | -2.26 | 2.26 | ❌ 급격 |
| 2→1 | -6.67 | 6.67 | ❌ 극심 |

---

## 2. MV-Diffusion 학습 현황

| Config | Views | Camera Indices | Checkpoint | 상태 |
|--------|:-----:|:--------------:|:----------:|:----:|
| M5t2 (baseline) | 6 | [0,1,2,3,4,5] | ckpt-5000 | ✅ |
| M5t2_cfgr | 6 | [0,1,2,3,4,5] | ckpt-10000 | ✅ |
| M5t2_noaug (E2) | 6 | [0,1,2,3,4,5] | ckpt-10000 | ✅ |
| M5t2_4view | 4 | [0,2,3,5] | ckpt-10000 | ✅ |
| M5t2_3view | 3 | [0,2,4] | ckpt-10000 | ✅ |

---

## 3. E2E 추론 결과 (MV-Diffusion → GS-LRM)

| Pipeline | Views | Overall PSNR_wh | Novel PSNR | SSIM | LPIPS |
|----------|:-----:|:---:|:---:|:---:|:---:|
| **6-view E2E** (H3-bis) | 6 | 21.21 | ~24.0* | - | - |
| 4-view E2E (H8) | 4 | 19.87 | 19.87** | 0.9619 | 0.0702 |
| 3-view E2E (H8) | 3 | 22.67*** | 16.26 | 0.9637 | 0.0808 |

> \* 6-view novel-only는 E1 quick eval 기준
> \*\* 4-view E2E는 novel view만 평가 (views 2,3,5)
> \*\*\* 3-view overall 높은 이유: input view(35.51 dB)가 1/3 비중 → 과대평가

### E2E 이중 열화 분석

| | GS-LRM only (GT) | E2E (MVDiff) | MVDiff 손실 |
|:---:|:---:|:---:|:---:|
| 6-view | 24.49 | 21.21 | -3.28 dB |
| 4-view | 21.71 | 19.87 | -1.84 dB |
| 3-view | 20.01 | ~16.26 (novel) | -3.75 dB |

### 3-view 결론

E2E novel 16.26 dB로 **실용성 부족**. 최소 4-view, 이상적으로 6-view 권장.
> H8 가설 **기각 경향**: 뷰 수 감소가 per-view 품질 개선으로 이어지지 않음.

---

## 4. Top/Bottom View 대응: 실험 계획

### 방법 A: Random Reference View 학습

```yaml
# config 1줄 변경 (코드 이미 존재: mouse_dataset.py:64-73)
reference_view_idx: "random"   # was: 0
```

| 장점 | 단점 |
|------|------|
| Config 1줄, 아키텍처 변경 없음 | M5 고정 6방향에 제한 |
| 기존 체크포인트에서 fine-tune | 임의 각도(35° 등) 불가 |
| 추론 시 카메라 정보 불필요 | 6배 augmentation → 수렴 느릴 수 있음 |

### 방법 B: Spherical Pose Conditioning

> 이론 상세 → [THEORY §4.2](../theory/MULTIVIEW_DIFFUSION_THEORY.md#42-pose-conditioning-3가지-방식-구현됨-미활성)

| 장점 | 단점 |
|------|------|
| 임의 카메라 포즈 지원 | ~100줄 코드 수정 |
| 연속적 포즈 보간 가능 | 학습/추론 시 c2w 필수 |
| 다른 데이터셋 전이 용이 | Pose encoder 학습 시간 |

### 비교 요약

| 기준 | Random Ref (A) | Pose Cond (B) |
|------|:--------------:|:-------------:|
| 구현 난이도 | ⭐ (1줄) | ⭐⭐⭐ (~100줄) |
| 일반화 수준 | M5 6방향 | 임의 카메라 |
| Checkpoint 호환 | 완전 | 부분 (MLP re-init) |
| **권장 순서** | **Phase 1** | **Phase 2** |

### Pose Conditioning 시작점

> 상세 근거 → [THEORY §4.4](../theory/MULTIVIEW_DIFFUSION_THEORY.md#44-구현-우선순위-근거-포함)

**Mouse M5t2 checkpoint에서 시작** (NOT Era3D).
- Mouse 도메인 이미 적응됨 → pose encoder만 re-init, 나머지 UNet freeze
- Era3D에서 시작하면 mouse 재학습 비용 발생

---

## 5. 배경 및 임의 입력 처리

### 배경 제거 (P1)
```bash
# SAM 체크포인트: checkpoints/sam/
rembg i input.jpg output.png
```
구현 난이도: ⭐ (1일)

### 임의 1장 입력
1. **촬영 가이드** 제공 ("cam_0 각도로 촬영") — 가장 간단
2. **Keypoint detection** (DANNCE) → 포즈 추정 → 뷰 정규화
3. **Random ref + pose cond** → 임의 뷰에서 직접 추론

---

## 6. Checkpoint 경로

```
/home/joon/dev/FaceLift/checkpoints
  → symlink → /node_data/joon/checkpoints/FaceLift

├── gslrm/
│   ├── base_uniform_v2_{1..6}view_v2/   ← H4 view ablation
│   ├── base_uniform_v2_4view_alpha*_v2/ ← H6 alpha mask
│   └── base_uniform_v2_paper_aligned_4view/
├── mvdiffusion/
│   ├── mouse_M5t2/                      ← baseline
│   ├── mouse_M5t2_noaug/               ← E2
│   ├── mouse_M5t2_cfgr/                ← CFG restored
│   ├── mouse_M5t2_{3,4}view/           ← H8 reduced view
│   └── pipeckpts/                       ← pretrained SD2.1-UnCLIP
└── sam/
```

> **⚠️ 전체 `checkpoints/` 디렉토리가 symlink** (하위 디렉토리가 아님).

---

## 7. 진행중 실험 (260213)

| GPU | 실험 | 상태 | 비고 |
|:---:|------|:----:|------|
| 4 | FREE | - | E2 완료 |
| 5 | H6 alpha_05 | 🔄 Running | GS-LRM |
| 6 | H6 alpha_01 | 🔄 Running | GS-LRM |
| 7 | paper_aligned_4view | 🔄 Running | GS-LRM |

---

*H8 View Generalization Experiments v3.0 | 260213*
