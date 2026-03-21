# Session Handoff — 260322

> 다음 세션 시작 시 이 문서를 첫 메시지로 전달하면 맥락이 이어집니다.

---

## 1. 세션 요약 (260321-260322, S14-S15)

### 완료된 작업

| # | 작업 | 핵심 결과 | 산출물 |
|:-:|------|-----------|--------|
| 1 | **Temporal smoothing 6방법 비교** | EMA/OptFlow/Median/Bilateral/SavGol 모두 **육안 품질 저하** → 원본 유지 | `outputs/report/temporal_comparison_v2/` |
| 2 | **N>=2 Covariance 재추출** | 3585f, 154d static + 198d temporal. HLAC probe CV=91.5% | `outputs/report/clustering/features/covariance_n2/` |
| 3 | **6v Alpha Loss 발견 + 평가** | 3variant 학습 완료 확인. α=0.3 best (PSNR 34.10, SSIM 0.9912) | `outputs/report/6v_alpha_comparison_512/` |
| 4 | **RENDER_RESOLUTION 384→512 수정** | 7개 파일. 이전 모든 렌더/DiFix 페어 재생성 필요 | 코드 수정 완료 |
| 5 | **메트릭 프로토콜 통일** | masked foreground = 표준 (3DGS/GS-LRM 논문 기준). Full-image = 참고만 | `TEMPORAL_EVAL_STANDARD.md` |
| 6 | **Deformation 모듈 분석** | canonical 접근 mouse에서 근본 한계. Future work로 분류 | `TEMPORAL_CONSISTENCY_STUDY.md` |

### 생성/수정된 파일

**신규 코드:**
- `mouse_extensions/evaluation/temporal_smoothing.py` — EMA, median, bilateral, savgol, optflow, temporal metrics
- `mouse_extensions/scripts/eval/temporal_comparison.py` — 통합 비교 파이프라인

**신규 문서:**
- `docs/experiments/TEMPORAL_EVAL_STANDARD.md` — 프레임/뷰/메트릭 기준 SSOT
- `docs/experiments/TEMPORAL_CONSISTENCY_STUDY.md` — temporal 연구 종합 (Phase 1-3)
- `docs/experiments/SESSION_HANDOFF_260322.md` — 이 문서

**수정된 코드 (resolution fix):**
- `mouse_extensions/scripts/novel_view/collect_dataset.py:80` — 384→512
- `gslrm/model/gaussians_renderer.py:358,1343` — 384→512
- `mouse_extensions/inference/modules/renderer.py:14` — 384→512
- `mouse_extensions/scripts/eval/gaussian_quality_metrics.py:270` — 384→512
- `configs/visualization/turntable_video.yaml`, `turntable_6x6.yaml` — 384→512

**수정된 문서:**
- `docs/experiments/ALPHA_LOSS_NOVEL_VIEW_ANALYSIS.md` — §10 (6v 결과 추가)
- `docs/INDEX.md` — temporal 문서 백링크 추가

---

## 2. 현재 진행 중 (GPU 백그라운드)

```bash
# 확인 명령어:
ssh gpu03 "grep DONE /tmp/full_bl.log /tmp/full_a03.log /tmp/full_a05.log"
```

| GPU | 작업 | 출력 | 예상 완료 |
|:---:|------|------|:---------:|
| 4 | baseline_6v 3600f 512 렌더 | `outputs/datasets/novel_view_512/baseline_6v/` | ~02:40 KST |
| 5 | 6v_alpha03 3600f 512 렌더 | `outputs/datasets/novel_view_512/6v_alpha03/` | ~02:40 KST |
| 6 | 6v_alpha05 3600f 512 렌더 | `outputs/datasets/novel_view_512/6v_alpha05/` | ~02:40 KST |

**렌더 스크립트**: `/tmp/render_full_3600.sh GPU EXP CKPT` (200f 배치, 18배치/모델)

---

## 3. 다음 세션 TODO (우선순위 순)

### 즉시 (P0 완료 후)

1. **P0 완료 확인** + 6v_alpha10 렌더 시작:
   ```bash
   nohup bash /tmp/render_full_3600.sh 4 6v_alpha10 \
     /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_6view_alpha10_v3/ckpt_0000000000015840.pt \
     > /tmp/full_a10.log 2>&1 &
   ```

2. **Fair comparison 512 재실행** — `psnr_gt_masked` 프로토콜로:
   ```bash
   python -m mouse_extensions.scripts.eval.unified_eval run \
     --render_dir outputs/datasets/novel_view_512/baseline_6v \
     --gt_dir /home/joon/data/preprocessed/FaceLift_mouse/M5t
   ```

3. **4모델 비교 그리드/영상 재생성 (512)** — `6v_alpha_comparison_512` 업데이트

### 순차 진행

4. **P1: MAMMAL Pseudo-GT 렌더** — `conda activate mammal_stable`, GPU 5, ~2h
5. **P2: OpenCV Artifact Cleanup** — CPU, tier0_raw → tier1_cleaned
6. **P4: DiFix 페어 빌드** — 512 데이터로 재생성, symlinks + manifest
7. **P5: DiFix Stage 1 학습** — GPU 4, ~8h

### 병렬 가능

- **Rat7M 데이터 준비 (P8)** — 독립 작업, 어떤 GPU든

---

## 4. 핵심 수치

### 6-View Alpha Loss (512, test set, full-image)

| Model | PSNR↑ | SSIM↑ |
|-------|:-----:|:-----:|
| Baseline (α=0) | 34.00 ± 4.63 | 0.9898 |
| **α=0.3** | **34.10 ± 4.56** | **0.9912** |
| α=0.5 | 34.07 ± 4.46 | 0.9912 |
| α=1.0 | 33.82 ± 4.30 | 0.9911 |

> ⚠️ 위는 full-image PSNR. Masked foreground (psnr_gt_masked) 재계산 필요.

### Gaussian Count (6v baseline, frame 3310)

| Filter | Count |
|--------|:-----:|
| Total | 1,048,578 |
| Opacity > 0.01 | 36,380 (3.5%) |
| N≥2 multiview | ~39K-73K |

### HLAC Probe (N>=2 covariance, 154d)

| Feature | Accuracy | CV |
|---------|:--------:|:--:|
| Covariance N>=2 | 89.6% | **91.5%** |
| KP centered (66d) | 50.0% | 52.3% |

---

## 5. 주의사항

- `outputs/difix_pairs/` 기존 384 페어 → **512 재생성 필수**
- `outputs/datasets/novel_view/` 기존 384 → `novel_view_512/`로 교체 예정
- **Full-image PSNR (34 dB) ≠ masked PSNR (예상 23-25 dB)** — 프로토콜 혼동 주의
- Temporal smoothing **보류** — 2D 후처리 전부 실패, 3D 접근 필요 (future work)
- Deformation module: canonical 접근 mouse 한계 → DiFix 우선
- PS Gaussian 수 미확인 — PS 체크포인트/로그에서 확인 필요

---

## 6. 결과물 위치 (gpu03)

| 결과 | 경로 |
|------|------|
| 6v Alpha 비교 (512) | `outputs/report/6v_alpha_comparison_512/{grids,videos,metrics}/` |
| Temporal 비교 | `outputs/report/temporal_comparison_v2/{videos,strips,metrics}/` |
| HLAC N>=2 | `outputs/m5t2_hlac_analysis_n2/` |
| 512 test set 렌더 | `outputs/datasets/temporal_eval_512/` |
| **3600f 전체 렌더 (진행 중)** | `outputs/datasets/novel_view_512/` |
| N>=2 covariance | `outputs/report/clustering/features/covariance_n2/` |

---

## 7. 체크포인트 경로

```
/node_data/joon/checkpoints/FaceLift/gslrm/
├── base_uniform_v2_6view_v2/best_psnr.pt          ← 6v baseline
├── M5t2_6view_alpha03_v3/ckpt_0000000000015840.pt  ← 6v α=0.3
├── M5t2_6view_alpha05_v3/ckpt_0000000000015840.pt  ← 6v α=0.5
├── M5t2_6view_alpha10_v3/ckpt_0000000000015840.pt  ← 6v α=1.0
├── base_uniform_v2_4view_alpha{03,05,10}_v3/best_psnr.pt  ← 4v variants
└── deformation/default/checkpoint_010000.pt         ← Deformation (미사용)
```

---

*Handoff by S14-15 | 2026-03-22 | → 다음 세션에서 이 파일을 전달하세요*
