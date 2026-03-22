# Session Handoff — 260322 S18

> 다음 세션 시작 시 이 문서를 첫 메시지로 전달하면 맥락이 이어집니다.

---

## 1. 세션 요약 (S18, 260322 03:00~08:30 KST)

### 완료된 작업

| # | 작업 | 핵심 결과 |
|:-:|------|-----------|
| 1 | **alpha10 6v render** (3600f, 512) | GPU 4, 완료 |
| 2 | **View ablation 5v/4v/3v/2v/1v** (3600f each, 512) | GPU 4,5,6 + chain script |
| 3 | **SCN2A_WK1 lone rat 다운로드** | Harvard Dataverse API, 1.2GB, 23 files |
| 4 | **sdannce_to_gslrm.py lone 지원** | keypoint/COM 경로 수정 (lone: `bsl0.5_FM/`) |
| 5 | **GS-LRM v3 zero-shot smoke test** | 단일 쥐 추론 성공 (마스크 없이, 케이지 포함 재구성) |
| 6 | **DANNCE 투영 수정** | MATLAB convention 문서화, 6cam 정확 |
| 7 | **6-view KP+skeleton 시각화** | 4프레임 × 6cam, DANNCE projection |
| 8 | **SAM2 lone 마스크** | `kp_sam2_lone.py` 작성, 20f × 6cam = 120 masks + 영상 |
| 9 | **s-DANNCE Dataset Guide v2.0** | 10섹션 SSOT, INDEX 백링크 |
| 10 | **10-condition comparison grid** | GT + 4 alpha + 5 ablation, 이미지 + 영상 |
| 11 | **DiFix 512 pairs** | 43.2K pairs (1v+2v → 6v, Type 3, symlink) |
| 12 | **Fair eval 512** | vs 6v baseline: **best alpha = α=0.3** (PSNR 24.85) |
| 13 | **문서 감사 + 수정** | INDEX footer, CLAUDE.md 오류, RAW_DATA 분리, 핸드오프 정리 |
| 14 | **7× multi-model deliberation** | 데이터 전략, DiFix, SAM2, 우선순위 등 |

### Fair Eval 512 결과 (핵심 수치)

**Alpha variants (vs 6v baseline)**:

| Model | PSNR | IoU |
|-------|:----:|:---:|
| **α=0.3** | **24.85** | **0.966** |
| α=0.5 | 24.24 | 0.964 |
| α=1.0 | 23.41 | 0.959 |

**View ablation — Inference-time (6v 모델에 N개 view 슬라이싱, vs 6v baseline)**:

| Views | PSNR | IoU | DiFix 필요도 |
|:-----:|:----:|:---:|:----------:|
| 5v, 4v | 50.00 | 1.000 | 불필요 (**pixel-identical**, MD5 동일) |
| 3v | 21.33 | 0.921 | 낮음 |
| 2v | 15.04 | 0.795 | 중간 |
| 1v | 7.05 | 0.402 | 높음 |

> **Note**: 5v/4v가 6v와 완전 동일한 이유: 6-view 학습 모델에서 5번째/6번째 view 정보가 redundant. 3v부터 critical threshold 이하.
> **H4 학습 시 ablation**(전용 N-view 모델, 384res)과 비교: 학습 시 1v=10.47 > 추론 시 1v=7.05 (+3.4dB). 학습 시 전용 모델이 저뷰에서 더 강건.

**Gaussian Artifact Metrics (3-frame sample)**:

| Metric | Baseline | α=0.3 | α=0.5 | α=1.0 |
|--------|:--------:|:-----:|:-----:|:-----:|
| Alpha Entropy (top) | 4.44 | **0.27** | 0.31 | 0.37 |
| Opacity Ambiguity % | 0.84 | **0.75** | 0.74 | 0.73 |
| Aniso Ratio (mean) | **70K** | 110K | 95K | 80K |

> Alpha loss → Alpha Entropy **16× 감소** (깔끔한 렌더링), Anisotropy 증가 (elongated Gaussians).
> **α=0.3이 entropy 최소 + PSNR 최고의 균형점**.

---

## 2. 생성/수정된 파일

**코드 (수정):**
- `mouse_extensions/scripts/sdannce_to_gslrm.py` — lone keypoint/COM 경로 지원

**코드 (신규, gpu03):**
- `sdannce-poc/segmentation/kp_sam2_lone.py` — SAM2 lone 마스크 생성

**문서 (수정):**
- `docs/INDEX.md` v16.1 — SDANNCE 백링크, footer 수정
- `docs/datasets/SDANNCE_VIDEO_AVAILABILITY.md` v2.0 — 10섹션 SSOT
- `docs/datasets/RAW_DATA.md` — Camera Config 분리
- `CLAUDE.md` — M5 레이블, 날짜 수정

**서버 산출물 (gpu03):**
- `outputs/datasets/novel_view_512/6v_alpha10/` — 3600f
- `outputs/datasets/novel_view_512/baseline_6v/mouse_m5t2/ablation_{1-5}view/` — 각 3600f
- `outputs/datasets/novel_view_512/comparison_grids/` — 비교 이미지 + 영상
- `outputs/datasets/novel_view_512/fair_eval_512_vs6v.json` — fair eval 결과
- `outputs/datasets/difix_pairs_512/` — 43.2K pairs (symlink)
- `outputs/sdannce_smoke_test/gslrm_v3_lone/` — zero-shot 렌더 + 진단
- `/home/joon/data/sdannce/rat/dataverse/SCN2A_WK1_2022_09_16_M1/` — WK1 데이터
- `/home/joon/data/sdannce/rat/dataverse/SCN2A_WK1_2022_09_16_M1/sam2_masks/` — 120 masks

---

## 3. 다음 세션 TODO (우선순위 순)

### P0: 즉시

1. **GS-LRM v4 masked smoke test** — SAM2 마스크 적용한 lone rat 추론
   - `sdannce_to_gslrm.py --sam2_ann_dir .../sam2_masks`
   - 동물만 깔끔하게 재구성되는지 확인

2. **Rat fine-tuning 시작** — α=0.3 모델 (best) 기반
   - 데이터 준비: `sdannce_to_gslrm.py` 로 1K~5K frames + SAM2 masks
   - LR: 1e-4 ~ 5e-5 (원래의 1/10)
   - 비교: zero-shot vs mouse-FT vs pretrained-FT

3. **DiFix 1v 학습 시작** — 43.2K pairs 준비 완료
   - 1v → 6v, Type 3 pair
   - 목표: 1v artifact 제거 → ~3v 수준

### P1: 이후

4. 전체 SAM2 마스크 (WK1 전체 90K frames × 6cam)
5. Rat fine-tuning 3-way 비교 평가 + 시각화
6. DiFix 2v 학습 + 평가

### P2: 장기

7. MAMMAL pseudo-GT render
8. BALB/c 저자 이메일 (klibaite@fas.harvard.edu)
9. NeurIPS paper drafting
10. 추가 WK1 세션 다운로드 (29세션 remaining)

---

## 4. 핵심 발견

### s-DANNCE 카메라 규약 (CRITICAL)

```
DANNCE (MATLAB): M = [R;t] @ K,  pts2d = [pts3d|1] @ M  (row vectors)
OpenCV:          pts2d = K @ [R|t] @ pts3d              (column vectors)

K_opencv = K_matlab.T (intrinsics 변환은 맞음)
BUT: cv2.projectPoints()는 DANNCE 캘리브에서 사용 금지!
→ sdannce-poc/src/sdannce_utils/projection.py 의 project_keypoints() 사용
```

### Best Alpha = α=0.3

| Metric | α=0.3 | baseline |
|--------|:-----:|:--------:|
| PSNR vs 6v | 24.85 | (ref) |
| IoU | 0.966 | (ref) |

### DiFix 전략

- **1v 우선** (PSNR 7.05 → target ~21+)
- Type 3 pairs (1v → 6v, self-supervised)
- 43.2K pairs ready

---

## 5. 체크포인트 경로

```
/node_data/joon/checkpoints/FaceLift/gslrm/
├── base_uniform_v2_6view_v2/best_psnr.pt          ← 6v baseline (best)
├── M5t2_6view_alpha03_v3/ckpt_0000000000015840.pt  ← 6v α=0.3 (BEST ALPHA)
├── M5t2_6view_alpha05_v3/ckpt_0000000000015840.pt  ← 6v α=0.5
├── M5t2_6view_alpha10_v3/ckpt_0000000000015840.pt  ← 6v α=1.0
```

---

## 6. 문서 감사 잔여 (P1)

| 이슈 | 상태 |
|------|:----:|
| INDEX footer 버전 | ✅ 수정 |
| RAW_DATA Camera Config 분리 | ✅ 수정 |
| CLAUDE.md M5 레이블 | ✅ 수정 |
| CLAUDE.md 날짜 | ✅ 수정 |
| SDANNCE 다운로드 상태 | ✅ 수정 |
| PREPROCESSING_REGISTRY 날짜 고착 | ⬜ P1 |
| KEYPOINT_3D_PIPELINE 중복 등재 | ⬜ P1 |
| 세션 핸드오프 파일 정리 | ✅ 3개 → _archive/ |

---

## 7. 시각화 산출물 경로 (gpu03)

| 산출물 | 경로 |
|--------|------|
| 10-condition 비교 그리드 | `outputs/datasets/novel_view_512/comparison_grids/*.png` |
| 비교 영상 | `outputs/datasets/novel_view_512/comparison_grids/alpha_ablation_comparison.mp4` |
| Fair eval JSON | `outputs/datasets/novel_view_512/fair_eval_512_vs6v.json` |
| Gaussian quality JSON | `outputs/reports/gaussian_quality_512/gaussian_quality_comparison.json` |
| Rat smoke test 렌더 | `outputs/sdannce_smoke_test/gslrm_v3_lone/render_cam0_f0.png` |
| Rat 6-view KP overlay | `outputs/sdannce_smoke_test/gslrm_v3_lone/diagnostics/6view_kp_correct_*.png` |
| SAM2 마스크 영상 | `SCN2A_WK1_.../sam2_masks/overlay_6cam_grid.mp4` |
| SAM2 마스크 그리드 | `gslrm_v3_lone/diagnostics/mask_overlay_*.png` |
| DiFix 512 pairs | `outputs/datasets/difix_pairs_512/` (43.2K, symlink) |
| DiFix 전략 문서 | `docs/experiments/DIFIX_TRAINING_STRATEGY.md` v1.0 |

---

*Handoff by S18 | 2026-03-22 15:30 KST (updated)*
