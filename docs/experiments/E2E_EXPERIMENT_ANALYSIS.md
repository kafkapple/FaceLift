# E2E Experiment Comprehensive Analysis

> Created: 2026-03-03 | Status: Active

## 1. Background

FaceLift E2E pipeline: 1-view input → MVDiffusion (6-view synthesis) → GS-LRM (3D reconstruction).
MVDiffusion이 병목으로, GT 6뷰 대비 E2E는 50-70% 성능 하락.

## 2. Pipeline Path 분류 (중요!)

`run_e2e_inference.py`는 입력에 따라 3가지 경로로 분기:

| Path | 조건 | 파이프라인 | 용도 |
|------|------|-----------|------|
| **Path 1** | `--data_dir` only (no `--input_view_idx`) | GT 6뷰 → GS-LRM | Upper bound 측정 |
| **Path 2a** | `--input_image` | 외부 이미지 → MVDiff → GS-LRM | 데모 |
| **Path 2b** | `--data_dir --input_view_idx 0` | GT 1뷰 → MVDiff → GS-LRM | **진짜 E2E** |

⚠️ `--input_view_idx 0`을 빠뜨리면 Path 1으로 분기 → E2E가 아닌 GS-LRM only 결과.

## 3. All Fair Eval Results (M5t2 Test Set, 360 frames × 5 eval views)

### GS-LRM Only (GT Input, Upper Bounds)

| 모델 | Input Views | PSNR_gt | IoU | 비고 |
|------|:---:|:---:|:---:|------|
| gslrm_1view | 1 | 10.47 | 0.028 | 단일 뷰, 심각한 정보 부족 |
| gslrm_2view | 2 | 15.95 | 0.858 | |
| gslrm_3view | 3 | 18.56 | 0.899 | |
| gslrm_4view | 4 | 20.66 | 0.926 | E0_1 checkpoint |
| gslrm_5view | 5 | 22.16 | 0.942 | |
| **gslrm_6view** | **6** | **23.84** | **0.954** | 6view_v2 checkpoint, 절대 상한 |

### 진짜 E2E (1뷰 → MVDiff → GS-LRM)

| 실험 | MVDiff Checkpoint | PSNR_gt | IoU | Pose | 비고 |
|------|------------------|:---:|:---:|------|------|
| baseline_360f | ckpt-5000 (sparse) | 7.93 | 0.474 | None | 초기 모델 |
| facelift_fair | ckpt-5000 (sparse) | 7.83 | 0.518 | None | fair_comparison.py |
| cfgr | ckpt-5000 (full attn) | 7.75 | 0.491 | None | CFG 복구 |
| e1_cosine_11k | E1 cosine 11K | 7.90 | 0.522 | None | Cosine LR |
| e1_cosine_20k | E1 cosine 20K | 7.90 | 0.528 | None | 20K 확장 |
| e2_resume_20k | E2 resume 20K | 8.20 | 0.521 | None | Resume + LR decay |
| e3_pose | E3 extrinsic | 8.10 | 0.523 | Extrinsic+Add | 초기 pose 실험 |
| p1_bl_6view_e2e | baseline | 8.11 | 0.501 | None | |
| p1_e1_6view_e2e | E1 | 8.04 | 0.511 | None | |
| **p1_6view_e2e** | **ckpt-5000** | **8.44** | **0.495** | **None** | **이전 E2E best** |
| H4b_step20000_e2e | H4b ckpt-20000 | ⏳ 실행 중 | ⏳ | Extrinsic+Add | 진짜 E2E (재실행) |

### 잘못 분류된 결과 (GS-LRM only를 E2E로 오인)

| 실험 | 실제 Path | PSNR_gt | IoU | 오인 원인 |
|------|----------|:---:|:---:|----------|
| H4b_extended_step20000_with_pose | **Path 1 (GS-LRM only)** | 21.02 | 0.943 | `--input_view_idx 0` 누락 |

## 4. 핵심 관찰

### 4.1 MVDiffusion = 여전히 병목

| 비교 | PSNR_gt | Gap |
|------|:---:|:---:|
| GS-LRM 6view (GT) | 23.84 | — |
| GS-LRM 4view (GT) | 20.66 | -3.2 dB |
| Best E2E (e2_resume) | 8.20 | **-15.6 dB** |
| Best E2E (p1_6view) | 8.44 | **-15.4 dB** |

E2E에서 GT 대비 **-15 dB 이상** 하락 → MVDiffusion이 사실상 무작위에 가까운 뷰 생성.

### 4.2 Pose Conditioning 효과 (E2E)

| 실험 | Pose | PSNR_gt | IoU | ΔIoU |
|------|------|:---:|:---:|:---:|
| e2_resume_20k | None | 8.20 | 0.521 | — |
| e3_pose | Extrinsic+Add | 8.10 | 0.523 | +0.002 |
| H4b_step20000_e2e | Extrinsic+Add (20K) | ⏳ | ⏳ | ⏳ |

E2E에서의 pose conditioning 효과는 아직 미미. H4b@20K 결과 대기 중.

### 4.3 모든 E2E 전략의 수렴

```
PSNR_gt 범위: 7.75 ~ 8.44 (전체 0.69 dB 스팬)
IoU 범위:     0.474 ~ 0.528 (전체 0.054 스팬)
```

Cosine LR, Resume, Pose, CFG 변경 등 모든 학습 전략이 좁은 범위에 수렴.
→ **MVDiffusion 아키텍처 자체의 한계**에 도달한 것으로 판단.

## 5. Verification Incident (2026-03-03)

### 사건 요약

H4b@20K 추론에서 `--input_view_idx 0` 누락 → GS-LRM only 결과(21.02 dB)를
E2E 결과로 잘못 보고. 기존 best(8.44)보다 +12.6 dB 향상이라는 비현실적 수치를
검증 없이 보고함.

### Root Cause

1. 명령어 구성 시 `--input_view_idx 0` 누락
2. 실행 로그의 "Batch Path 1: GS-LRM only" 확인 미수행
3. +12.6 dB 향상에 대한 sanity check 미수행
4. 검증 불충분 상태에서 결과 보고

### Prevention Protocol

E2E 결과 보고 전 **반드시** 확인:

| # | 항목 | 명령어 | 기대값 |
|---|------|--------|--------|
| 1 | Pipeline Path | `grep 'Batch Path' LOG` | "Path 2b: MVDiffusion" |
| 2 | MVDiff 로딩 | `grep 'Loading MVDiffusion' LOG` | 1+ 회 |
| 3 | Config 확인 | `jq .args.input_view_idx run_config.json` | `≠ null` |
| 4 | Sanity Check | 기존 E2E 범위와 비교 | 7~9 dB (±3 dB 이내) |
| 5 | 중간 산출물 | `ls samples/*/mvdiff_views/` | 디렉토리 존재 |

## 6. Current Status (2026-03-03)

| 실험 | GPU | PID | 상태 | 예상 완료 |
|------|:---:|:---:|------|----------|
| H4b@20K **진짜 E2E** | 5 | 2686164 | 실행 중 | ~8시간 (360f × MVDiff + GS-LRM) |
| H7v2 Spatial Token 재학습 | 6 | 2684682 | Step 5/10000 | ~37시간 |

### 다음 단계

1. H4b E2E 완료 → fair_comparison 평가 → 기존 best와 비교
2. H7v2 학습 완료 → E2E 추론 → 평가
3. 비교 표 업데이트

---

*FaceLift E2E Analysis | v1.0 | 2026-03-03*
