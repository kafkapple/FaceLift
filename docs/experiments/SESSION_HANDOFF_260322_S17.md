# Session Handoff — 260322 S17

> 다음 세션 시작 시 이 문서를 첫 메시지로 전달하면 맥락이 이어집니다.

---

## 1. 세션 요약 (S17, 260322 01:00~02:30 KST)

### 완료된 작업

| # | 작업 | 핵심 결과 | 산출물 |
|:-:|------|-----------|--------|
| 1 | **s-DANNCE BALB/c 비디오 가용성 조사** | Dataverse 17개 전수조사 → BALB/c 비디오 **공개 없음**. 저자 이메일이 유일 경로 | `docs/datasets/SDANNCE_VIDEO_AVAILABILITY.md` |
| 2 | **Rat SCN2A 세션 전처리 파이프라인 구축** | DANNCE→OpenCV 캘리브 변환, 6cam 프레임 추출, KP reprojection 검증 | `outputs/sdannce_smoke_test/` |
| 3 | **DANNCE→GS-LRM 변환 스크립트 (v2)** | Zero-pad 1920→1920² + SAM2 마스크 지원 + CLI | `mouse_extensions/scripts/sdannce_to_gslrm.py` |
| 4 | **GS-LRM smoke test v1 (crop)** | 파이프라인 작동 확인, 품질 불량 (fx 불일치, BG sub 마스크) | `outputs/sdannce_smoke_test/gslrm_output/` |
| 5 | **GS-LRM smoke test v2 (pad+SAM2)** | fx=605 (1.1× vs 549), 케이지+동물 재구성. v1 대비 개선 | `outputs/sdannce_smoke_test/gslrm_v2_output/` |
| 6 | **진단 시각화** | 6-view KP overlay (4 frames), pipeline comparison grid | `outputs/sdannce_smoke_test/diagnostics/` |
| 7 | **3-model deliberation 4회** | 데이터 전략, FOV 해결, 품질 원인, 패딩 이론 | 대화 내 |

### 생성/수정된 파일

**신규:**
- `mouse_extensions/scripts/sdannce_to_gslrm.py` — DANNCE→GS-LRM 통합 변환 (v2: padding+SAM2)
- `docs/datasets/SDANNCE_VIDEO_AVAILABILITY.md` — BALB/c 비디오 조사 보고서

**서버 산출물 (gpu03):**
- `outputs/sdannce_smoke_test/frames/` — 6cam × 10 sample frames
- `outputs/sdannce_smoke_test/calibration/cameras_opencv.npz` — 6cam OpenCV 캘리브
- `outputs/sdannce_smoke_test/validation/` — KP reprojection overlay, BG subtraction test
- `outputs/sdannce_smoke_test/gslrm_format/` — v1 GS-LRM 입력 (center crop)
- `outputs/sdannce_smoke_test/gslrm_v2_padded/` — v2 GS-LRM 입력 (zero-pad+SAM2)
- `outputs/sdannce_smoke_test/gslrm_output/` — v1 GS-LRM 추론 결과
- `outputs/sdannce_smoke_test/gslrm_v2_output/` — v2 GS-LRM 추론 결과
- `outputs/sdannce_smoke_test/diagnostics/` — 6-view grids, pipeline comparison

---

## 2. 현재 진행 중 (GPU 백그라운드)

```bash
# 확인 명령어:
ssh gpu03 "grep -c 'complete' /tmp/full_bl.log /tmp/full_a03.log /tmp/full_a05.log"
```

| GPU | 작업 | 진행률 | 예상 완료 |
|:---:|------|:------:|:---------:|
| 4 | baseline_6v 3600f 512 렌더 | ~56% (2010/3600) | ~03:30 KST |
| 5 | 6v_alpha03 3600f 512 렌더 | ~72% (2613/3600) | ~03:00 KST |
| 6 | 6v_alpha05 3600f 512 렌더 | ~67% (2412/3600) | ~03:15 KST |

---

## 3. 다음 세션 TODO (우선순위 순)

### P0: s-DANNCE → GS-LRM 품질 개선 (최우선)

1. **6cam 전체 SAM2 마스크 생성**
   - `kp_sam2_segment.py` 활용: keypoint→SAM2 point prompt→자동 마스크
   - 또는 6cam 각각 수동 annotation 몇 프레임 + SAM2 video propagation
   - 현재 Camera1만 SAM2 마스크 있음 → 나머지 5대 필요

2. **v3 smoke test 실행** (6cam SAM2 + padding)
   ```bash
   python -m mouse_extensions.scripts.sdannce_to_gslrm \
     --session_dir /home/joon/dev/sdannce-poc/data/2022_09_22_M3_M4 \
     --output_dir outputs/sdannce_smoke_test/gslrm_v3_full_mask \
     --sam2_ann_dir /path/to/6cam_masks \
     --merge_animals --frame_indices 0 500 1000 5000
   ```

3. **BALB/c 저자 이메일 발송** (병렬)
   - 수신: klibaite@fas.harvard.edu 또는 timothy.dunn@duke.edu
   - 내용: MOUSE cohort raw video + calibration 요청

### P1: 기존 FaceLift 렌더 완료 후

4. **P0 렌더 완료 확인 → 6v_alpha10 렌더 시작** (GPU 4)
5. **Fair comparison 512 재실행** (`psnr_gt_masked` 프로토콜)
6. **4모델 비교 그리드/영상 재생성 (512)**

### P2: 이후

7. P1: MAMMAL Pseudo-GT 렌더
8. P4: DiFix 페어 빌드 (512)

---

## 4. 핵심 수치 & 발견

### s-DANNCE Rat 카메라 특성

| Camera | fx | fy | cx | cy | dist(mm) | k1 |
|:------:|:--:|:--:|:--:|:--:|:--------:|:--:|
| 1 | 2268 | 2276 | 940 | 617 | 1115 | -0.090 |
| 2 | 2293 | 2293 | 959 | 574 | 1235 | -0.063 |
| 3 | 2230 | 2233 | 977 | 536 | 1229 | -0.085 |
| 4 | 2229 | 2229 | 956 | 541 | 1047 | -0.081 |
| 5 | 2293 | 2295 | 959 | 549 | 1179 | -0.063 |
| 6 | 2237 | 2241 | 965 | 516 | 1103 | -0.084 |

### GS-LRM FOV 호환성

| 방식 | fx (512용) | vs 학습(549) | 정보 손실 |
|------|:---------:|:-----------:|:---------:|
| Tight crop | 1700-2700 | **3-5×** ❌ | 높음 |
| Wide crop 1200² | 968 | **1.8×** ⚠️ | 좌우 잘림 |
| **Zero-pad 1920²** | **605** | **1.1×** ✅ | **없음** |

### 품질 원인 분석 (3-model 합의)

| 순위 | 원인 | 해결 방법 |
|:----:|------|----------|
| 1 | **마스크 품질** (BG sub 0-12%) | SAM2 6cam 마스크 |
| 2 | **Social pair** (2마리 동시) | Union mask (rat1∪rat2) |
| 3 | **FOV 불일치** | Zero-pad → fx=605 (해결) |
| 4 | **Species gap** (mouse→rat) | Fine-tuning (P2) |
| 5 | **Zero-shot** (학습 없음) | Fine-tuning (P2) |

### BALB/c 비디오 가용성

- Harvard Dataverse MOUSE cohort: **.mat only (128개, 1.9GB)**
- 비디오 포함 cohort: SCN2A_SOC1/SOC3/WK1 (rat only)
- 저자 이메일이 유일 경로

---

## 5. 주의사항

- `sdannce_to_gslrm.py`는 Camera1만 SAM2 마스크 적용 가능 (현재 한계)
- Camera2-6에 마스크 없으면 **full-frame foreground** → 케이지 전체 재구성 시도
- 기존 `sdannce-poc/kp_sam2_segment.py`로 6cam 자동 마스크 가능성 있음 (미검증)
- `merge_animals=True` (union) 기본값 — 두 마리 동시 재구성이 자연스러움
- P0 렌더 (GPU 4,5,6) 아직 진행 중 — 완료 후 alpha10 + fair eval 필요

---

## 6. 체크포인트 경로

```
/node_data/joon/checkpoints/FaceLift/gslrm/
├── base_uniform_v2_6view_v2/best_psnr.pt          ← 6v baseline (smoke test에 사용)
├── M5t2_6view_alpha03_v3/ckpt_0000000000015840.pt  ← 6v α=0.3
├── M5t2_6view_alpha05_v3/ckpt_0000000000015840.pt  ← 6v α=0.5
├── M5t2_6view_alpha10_v3/ckpt_0000000000015840.pt  ← 6v α=1.0
```

---

*Handoff by S17 | 2026-03-22 02:30 KST | → 다음 세션에서 이 파일을 전달하세요*
