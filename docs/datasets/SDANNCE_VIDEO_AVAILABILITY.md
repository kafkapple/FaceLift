# s-DANNCE Dataset Guide (SSOT)

> **Version**: v2.0 | **Updated**: 2026-03-22 | **Status**: 🔬 ACTIVE
> **Purpose**: BehaviorSplatter NeurIPS 2026 — multi-species 3D reconstruction data source
> **Source**: [Harvard Dataverse socialDANNCE_data](https://dataverse.harvard.edu/dataverse/socialDANNCE_data) | [GitHub tqxli/sdannce](https://github.com/tqxli/sdannce)

---

## 1. Executive Summary

s-DANNCE (social DANNCE)는 6-camera multi-view 환경에서 자유 행동하는 쥐의 3D pose를 추적하는 시스템. Harvard Dataverse에 **17개 데이터셋** (1,690+ 세션) 공개. BehaviorSplatter에서 **cross-species generalization** 실증에 핵심.

**현재 전략**: SCN2A_WK1 (lone rat) → GS-LRM v3 smoke test → multi-species 논문 서사

---

## 2. Harvard Dataverse 전수 조사 (17 Datasets)

### 2.1 비디오 포함 데이터셋 (Raw Movies + 3D Tracking)

| Dataset | DOI | Type | Animals | Files | Size | Species |
|---------|-----|:----:|:-------:|:-----:|-----:|:-------:|
| **SCN2A_WK1** ⭐ | `DVN/BHQBB7` | **Lone** | **1** | 878 | ~34.5 GB | Rat |
| SCN2A_SOC1 | `DVN/Q7CA6F` | Social | 2 (dyad) | 504 | ~25.5 GB | Rat |
| SCN2A_SOC2 | `DVN/C1VOJX` | Social | 2 (dyad) | — | — | Rat |
| SCN2A_SOC3 | `DVN/TPRHHX` | Social | 2 (dyad) | 540 | — | Rat |
| TRIADS | `DVN/CYDZ2F` | Social | **3** (triad) | 205 | — | Rat |
| BEDDING | `DVN/696AK6` | Social | 2 | 199 | — | Rat |

### 2.2 키포인트 + 행동 매핑만 (비디오 없음)

| Dataset | DOI | Files | Species | Notes |
|---------|-----|:-----:|:-------:|-------|
| MOUSE | `DVN/VKJHTD` | 128 | Mouse (BALB/c) | .mat only, 1.9 GB |
| CHD8 | `DVN/WWNZOX` | 360 | Rat | ASD model |
| SCN2A | `DVN/7RNDJY` | 120 | Rat | ASD model |
| CNTNAP2 | `DVN/EMI8Z3` | 146 | Rat | ASD model |
| NRXN1 | `DVN/1DOCOQ` | 376 | Rat | ASD model |
| GRIN2B | `DVN/XNSYTC` | 120 | Rat | ASD model |
| ARID1B | `DVN/0LZZMU` | 212 | Rat | ASD model |
| FMR1 | `DVN/DCII4E` | 276 | Rat | ASD model |
| LONG EVANS | `DVN/4X8GTB` | 210 | Rat | WT control |

### 2.3 기타

| Dataset | DOI | Files | Content |
|---------|-----|:-----:|---------|
| Documentation | `DVN/UKNUWN` | 2 | Data organization docs |
| Behavioral data sheets | `DVN/F2UMKM` | 8 | PNG behavior sheets per cohort |

**라이선스**: CC0 1.0 Public Domain (전체)

---

## 3. SCN2A_WK1 상세 (우리의 주요 데이터)

### 3.1 개요

| Item | Value |
|------|-------|
| **DOI** | [`10.7910/DVN/BHQBB7`](https://doi.org/10.7910/DVN/BHQBB7) |
| **Type** | **Lone (single animal)** — open field behavior |
| **Sessions** | 30 (M1~M6 × 5 days: 2022-09-15 ~ 2022-09-19) |
| **Duration** | 30 min per session @ 50 fps |
| **Cameras** | 6 synchronized views |
| **Keypoints** | 23 joints (DANNCE format) |

### 3.2 파일 구성 (세션당)

```
{SESSION}/
├── calibration/           # 6 camera intrinsics + extrinsics
│   ├── hires_cam1_params.mat
│   ├── hires_cam2_params.mat
│   └── ... (cam3~cam6)
├── videos/                # 6-camera raw video
│   ├── Camera1/
│   │   ├── 0.mp4          # ~88-185 MB per camera
│   │   ├── frametimes.npy  # frame timestamps
│   │   └── metadata.tab    # video metadata
│   └── ... (Camera2~Camera6)
├── COM/predict00/
│   └── com3d.mat          # Center-of-mass predictions
└── SDANNCE/bsl0.5_FM/
    ├── save_data_AVG.mat   # Final keypoint predictions (110 MB)
    ├── save_data_AVG0.mat  # Initial predictions
    ├── init_save_data_AVG.mat
    └── com3d_used.mat      # COM used for inference
```

### 3.3 Behavioral Label File (별도 Dataverse)

> ⚠️ 행동 레이블은 비디오 데이터셋(DVN/BHQBB7)이 아닌 **별도 데이터셋(DVN/7RNDJY)**에 포함.

**다운로드**:
```bash
# SCN2A_M1_20220916_0077_L.mat (25.7 MB, file ID: 10930077)
wget -O SCN2A_M1_20220916_0077_L.mat \
  'https://dataverse.harvard.edu/api/access/datafile/10930077'
```

**파일 구조**:
```python
sdannce.ratgroup    = 'SCN2A'
sdannce.ratid       = 'M1'
sdannce.ratdate     = '20220916'
sdannce.issoc       = 0          # lone session
sdannce.ratgen      = uint8      # genotype code
sdannce.m1          = (90000, 3, 23)   # 3D keypoints (50fps × 30min)
sdannce.hlac        = (90000, 1)       # 8 HLAC classes
sdannce.llac        = (90000, 1)       # 130+ LLAC fine-grained classes
sdannce.cz_action   = (90000, 2)       # t-SNE 2D behavioral embedding
sdannce.m2          = empty            # no partner (lone)
sdannce.part_hlac   = empty            # no partner labels
```

**HLAC 9-Class Definitions (Klibaite et al. 2025, Cell — Table S2)**:

| Group # | Name | Description | LLAC Count |
|:-------:|:-----|:-----------|:----------:|
| 1 | **idle** | skeleton completely still | 16 |
| 2 | **sniff/head** | small movements, head bob/sweeps, animal stationary | 28 |
| 3 | **groom** | regular movements of head/limbs | 6 |
| 4 | **scrunched** | body compressed, anterior body elevated | 16 |
| 5 | **active crouched** | front limbs off ground, high-velocity anterior movements | 15 |
| 6 | **reared** | anterior of body high off ground | 24 |
| 7 | **explore** | variety of movements, steps, sweeps, active sniffing | 35 |
| 8 | **locomotion** | stereotyped repeated limb movements, body translation | 18 |
| 9 | **fast/error** | very fast movement, often tracking error | 4 |

> Source: Supplementary Table S2 "High-level class desc." sheet.
> 162 LLACs (unsupervised watershed) → 9 HLACs (human-annotated).

**HLAC 8-Class 분포 (PoC session: 2022_09_16_M1, 90,000 frames)**:

| Class | Name | Frames | % | Kinematic Signature |
|:-----:|:-----|-------:|----:|:----|
| 1 | idle | 28,573 | 31.7% | Speed=0.9, completely still |
| 2 | sniff/head | 28,505 | 31.7% | Speed=4.2, head movement only |
| 3 | groom | 803 | 0.9% | Snout low (18mm), body short (165mm) |
| 4 | scrunched | 853 | 0.9% | Snout=163mm, anterior elevated |
| 5 | active crouched | 1,222 | 1.4% | Speed=114, high-velocity anterior |
| 6 | reared | 6,428 | 7.1% | Snout=229mm (highest), body extended (249mm) |
| 7 | explore | 15,938 | 17.7% | Snout=15mm (floor), active head (91mm/s) |
| 8 | locomotion | 7,678 | 8.5% | Speed=128, limb=135, body translation |

> Class 9 (fast/error) absent — lone rat, 30min session, no tracking errors.
> Verified 260324: kinematic signatures match Table S2 descriptions 8/8.

**Frame Index Matching (DANNCE ↔ SocialMapper)**:

| 항목 | DANNCE (DVN/BHQBB7) | SocialMapper (DVN/7RNDJY) |
|------|:-------------------:|:-------------------------:|
| 파일 | `save_data_AVG.mat` | `SCN2A_M1_20220916_0077_L.mat` |
| 프레임 수 | **89,000** | **90,000** |
| 인덱스 범위 | 0-88999 (sampleID) | 0-89999 |
| 차이 | 마지막 1,000 프레임 누락 | 전체 포함 |
| Keypoints | `pred (89000, 3, 23)` | `m1 (90000, 3, 23)` |
| KP 일치 여부 | **다른 모델 출력** (Frame 0 diff=20.2mm) | SocialMapper 자체 추정 |

**매핑 규칙**:
```python
# DANNCE frame index → HLAC label
# sampleID[i] = i (0-indexed, 연속)
# hlac[frame_idx] for frame_idx in range(89000) — 직접 인덱싱 가능
hlac_label = social['sdannce'][0,0]['hlac'].flatten()
for dannce_idx in range(89000):
    video_frame = int(dannce['sampleID'].flatten()[dannce_idx])
    label = hlac_label[video_frame]  # HLAC class for this frame
```

> ⚠️ 두 .mat의 keypoints는 다른 DANNCE 모델 output (Frame 0 diff=20.2mm).
> HLAC 매핑 시 **frame index 기준**, keypoint 값 비교 아님.
> 비디오 프레임 인덱스가 공통 키.

**검증 결과 (260323 S37 audit)**:
- sampleID는 0-88999 연속 → DANNCE 마지막 1K 프레임은 추론 미수행 구간 (비디오 끝부분)
- `hlac[0:89000]` 직접 인덱싱 안전 (intersection set)
- HLAC stratified sampling은 **class index 기반** — 행동 명칭 미확인이어도 sampling 자체는 유효
- 80/20 dual-pool (behavioral 80% + reconstruction 20%) 비율은 검증 필요 hyperparameter
- SocialMapper 재실행 불필요: .mat subject=M1, date=20220916 일치 확인
- Smoke test 8/8 통과 (import, session detection, HLAC load, KP load, frame match, CameraPool, social KP)

**LLAC 130+ Classes**: fine-grained behavioral syllables. HLAC는 LLAC를 계층적으로 그룹핑한 상위 분류.

**Stratified Sampling 적용 예시** (RAT2, 3K target):

| HLAC | 원본 % | 3K 비례 할당 | 최소 보장 (50) | 최종 할당 |
|:----:|------:|:-----------:|:------------:|:--------:|
| 1 | 31.7% | 951 | — | 951 |
| 2 | 31.7% | 951 | — | 951 |
| 7 | 17.7% | 531 | — | 531 |
| 8 | 8.5% | 255 | — | 255 |
| 6 | 7.1% | 213 | — | 213 |
| 5 | 1.4% | 42 | **50** | 50 |
| 4 | 0.9% | 27 | **50** | 50 |
| 3 | 0.9% | 27 | **50** | 50 |
| **합계** | | | | **~3,051** |

### 3.4 파일 타입 통계

| Type | Count | Size | Content |
|------|:-----:|-----:|---------|
| .mp4 | 180 | 24.3 GB | 6cam × 30 sessions |
| .mat | 332 | 9.9 GB | calibrations + keypoints + COM |
| .npy | 180 | 0.24 GB | frame timestamps |
| .tab | 181 | ~0 | video metadata |
| .hdf5 | 2 | 0.12 GB | — |
| .yaml | 1 | ~0 | — |
| **Total** | **878** | **~34.5 GB** | |

### 3.4 PoC 세션: `2022_09_16_M1`

| Item | Value |
|------|-------|
| **Files** | 29 (6 videos + 6 calibs + 6 frametimes + 5 SDANNCE + 1 COM + 5 metadata) |
| **Size** | 1.14 GB |
| **Download** | `wget -O file https://dataverse.harvard.edu/api/access/datafile/{FILE_ID}` |
| **Location (gpu03)** | `/home/joon/data/sdannce/rat/dataverse/SCN2A_WK1_2022_09_16_M1/` |

#### File IDs (API download reference)

```bash
# Calibration
10826356  calibration/hires_cam1_params.mat
10826339  calibration/hires_cam2_params.mat
10826621  calibration/hires_cam3_params.mat
10827163  calibration/hires_cam4_params.mat
10826682  calibration/hires_cam5_params.mat
10826863  calibration/hires_cam6_params.mat

# Videos
10826497  videos/Camera1/0.mp4
10827049  videos/Camera2/0.mp4
10827052  videos/Camera3/0.mp4
10826478  videos/Camera4/0.mp4
10826463  videos/Camera5/0.mp4
10826531  videos/Camera6/0.mp4

# SDANNCE keypoints
10827101  SDANNCE/bsl0.5_FM/save_data_AVG.mat
10826781  SDANNCE/bsl0.5_FM/save_data_AVG0.mat
10826831  SDANNCE/bsl0.5_FM/init_save_data_AVG.mat
10826786  SDANNCE/bsl0.5_FM/com3d_used.mat
10827013  COM/predict00/com3d.mat
```

---

## 4. 명명 규칙

### 세션 이름

```
{COHORT}_{TYPE}{ROUND}_{DATE}_{SUBJECT}
```

| Component | Values | Example |
|-----------|--------|---------|
| COHORT | SCN2A, CHD8, ... | SCN2A |
| TYPE | WK (lone), SOC (social) | WK1 |
| DATE | YYYY_MM_DD | 2022_09_16 |
| SUBJECT | M{N} (lone), M{N}_M{N} (social) | M1 |

### 동물 수 판별

| 패턴 | Type | 동물 수 |
|------|------|:-------:|
| `WK` + `M1` | Lone | **1** |
| `SOC` + `M1_M6` | Social (dyad) | **2** |
| TRIADS dataset | Social (triad) | **3** |

---

## 5. 카메라 특성 (SCN2A Rat)

### 기존 social 세션 (2022_09_22_M3_M4)에서 측정

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
| Tight crop | 1700-2700 | 3-5× ❌ | 높음 |
| Wide crop 1200² | 968 | 1.8× ⚠️ | 좌우 잘림 |
| **Zero-pad 1920²** | **605** | **1.1×** ✅ | **없음** |

---

## 6. BALB/c Mouse 데이터 현황

### Dataverse MOUSE cohort (DVN/VKJHTD)

- 128개 .mat 파일 (1.9 GB) — **비디오 없음**
- Keypoints (23j, 3D) + behavior labels + t-SNE embedding 포함
- BALB/c = W (White strain), C57BL/6 = B (Black strain)

> ⚠️ **Mouse `hlac` 필드는 14-class SocialMapper 체계** (Rat 9-class HLAC와 다름!)
> Source: `mouseEmbedding.m` — 1:idle, 2:slow, 3:head, 4:groom, 5:crouched,
> 6:active_crouch, 7:steps_crouched, 8:rear, 9:high_rear, 10:slow_explore,
> 11:explore, 12:step_explore, 13:locomotion_slow, 14:locomotion_fast.
> Lone mouse W4에서 11개 활성 (5,8,9 부재), Social B1에서 13개 활성 (1 부재).

### .mat 구조

```python
# MOUSE_W4_20240426_0028_L.mat (Lone BALB/c)
sdannce.group       = 'mouse'
sdannce.mousestrain = 'BALB/c'
sdannce.issoc       = 0 (lone)
sdannce.m1          = (30000, 3, 23)  # 10 min @ 50fps
sdannce.hlac        = (30000,)        # behavior labels (unsupervised cluster IDs)
sdannce.llac        = (30000,)        # fine-grained labels
sdannce.cz_action   = (30000, 2)      # t-SNE embedding
```

### 비디오 확보 경로

저자 이메일이 유일:

| Name | Email | Role |
|------|-------|------|
| Ugne Klibaite | klibaite@fas.harvard.edu | Lead contact |
| Bence P. Olveczky | olveczky@fas.harvard.edu | PI (Harvard) |
| Timothy W. Dunn | timothy.dunn@duke.edu | PI (Duke) |

---

## 7. 대안 데이터셋 비교

| Dataset | Species | Animals | Video | Calibration | Behavior Labels | Duration | Suitable? |
|---------|:-------:|:-------:|:-----:|:-----------:|:---------------:|----------|:---------:|
| **SCN2A_WK1** ⭐ | Rat | 1 | ✅ | ✅ | ✅ **HLAC 8cls** (DVN/7RNDJY) | 30min × 30 | **Best** |
| SCN2A_SOC1 | Rat | 2 | ✅ | ✅ | Via pipeline | 30min × ? | Social only |
| Rat 7M | Rat | 1 | ✅ | ✅ | ❌ (pose only) | 5s clips | Too short |
| PAIR-R24M | Rat | 2 | ✅ | ✅ | ✅ (11 types) | Long | Pairs only |
| DANNCE markerless_mouse | Mouse | 1 | ✅ | ✅ | ❌ | 3-60min | Alt. for mouse |
| MOUSE cohort | Mouse | 1 | ❌ | ❌ | ✅ (HLAC) | 10min | No video |

---

## 8. BehaviorSplatter 파이프라인 적합성

| Data Source | Video | Calib | Keypoints | Behavior | GS-LRM Status |
|-------------|:-----:|:-----:|:---------:|:--------:|:-------------:|
| **M5t2 Mouse** (ours) | ✅ | ✅ | ✅ (22j) | ❌ | ✅ Trained (best PSNR 23.84) |
| **SCN2A_WK1 Rat** ⭐ | ✅ | ✅ | ✅ (23j) | ✅ **HLAC 8cls** | ✅ v3 smoke test 성공 (zero-shot) |
| **SCN2A_SOC Rat** | ✅ | ✅ | ✅ (23j) | Via pipeline | 🔄 v2 tested (mask quality issue) |
| **BALB/c Mouse** | ❌ | ❌ | ✅ (23j) | ✅ (HLAC) | ❌ No video |

### Behavior Labels 참고

s-DANNCE/SocialMapper는 **비지도 방식** (Cell 2025, Klibaite et al.):
- 3D keypoints → PCA (15 PCs) → Morlet wavelet (25 freq, 0.5-20Hz) → t-SNE → watershed
- 클러스터 ID → 사후적으로 연구자가 명명 (grooming, rearing, locomotion 등)
- **Lone animal에도 적용됨** — Action embedding은 per-animal (Joint만 dyadic)

### ⭐ Behavioral Labels 소재 (2026-03-23 확인)

**핵심 발견**: 행동 레이블은 **별도 Dataverse 데이터셋**에 SocialMapper aggregate .mat로 제공됨.

| 데이터 유형 | Dataverse | DOI | 내용 |
|------------|-----------|-----|------|
| **비디오 + DANNCE keypoints** | SCN2A_WK1 | `DVN/BHQBB7` | 878 files, 34.5 GB, save_data_AVG.mat (HLAC 없음) |
| **행동 레이블 + keypoints** | SCN2A | `DVN/7RNDJY` | 120 files, SocialMapper aggregate (HLAC 포함!) |

**SCN2A_M1_20220916_0077_L.mat (DVN/7RNDJY)** 구조:
```python
sdannce.ratgroup  = 'SCN2A'
sdannce.ratid     = 'M1'
sdannce.ratdate   = '20220916'
sdannce.issoc     = 0 (lone)
sdannce.m1        = (90000, 3, 23)   # 3D keypoints (30min @ 50fps)
sdannce.hlac      = (90000, 1)       # 8 HLAC classes, 100% coverage
sdannce.llac      = (90000, 1)       # 130+ LLAC classes
sdannce.cz_action = (90000, 2)       # t-SNE 2D embedding
```

**HLAC 8-Class 분포 (90,000 frames)**:
| Class | Frames | % | 비고 |
|:-----:|-------:|----:|------|
| 1 | 28,573 | 31.7% | 주요 행동 A |
| 2 | 28,505 | 31.7% | 주요 행동 B |
| 7 | 15,938 | 17.7% | |
| 8 | 7,678 | 8.5% | |
| 6 | 6,428 | 7.1% | |
| 5 | 1,222 | 1.4% | 희소 |
| 4 | 853 | 0.9% | 희소 |
| 3 | 803 | 0.9% | 희소 |

### Frame Index Matching

| 항목 | DANNCE (DVN/BHQBB7) | SocialMapper (DVN/7RNDJY) |
|------|---------------------|---------------------------|
| **프레임 수** | 89,000 | 90,000 |
| **인덱스 범위** | 0-88999 (sampleID) | 0-89999 |
| **차이** | 마지막 1,000 프레임 누락 | 전체 |
| **매핑** | `hlac[frame_idx]` for frame_idx in 0-88999 | |

> ⚠️ DANNCE pred와 SocialMapper m1의 keypoints는 다른 모델 output (Frame 0 diff=20.2mm).
> HLAC 매핑 시 **frame index 기준** (sampleID), keypoint 값 비교 아님.

---

## 9. gpu03 데이터 현황

```
/home/joon/data/sdannce/
├── rat/
│   └── dataverse/
│       └── SCN2A_WK1_2022_09_16_M1/   ← ✅ Downloaded (1.2 GB, 23 files)
│           ├── calibration/             # 6 cam params
│           ├── videos/Camera{1-6}/      # 6 cam MP4
│           ├── COM/predict00/           # center-of-mass
│           └── SDANNCE/bsl0.5_FM/       # 3D keypoints
├── mouse/
│   └── dataverse/                       # BALB/c .mat only (128 files)
└── metadata/

/home/joon/dev/sdannce-poc/data/
└── 2022_09_22_M3_M4/                    # Social pair (existing, 1.4 GB)
    ├── calibration/
    ├── videos/Camera{1-6}/
    └── SDANNCE/bsl0.5_FM_{rat1,rat2}/
```

---

## 10. Download Commands

### 단일 세션 (PoC)

```bash
# SCN2A_WK1_2022_09_16_M1 (1.14 GB)
# Script: /tmp/download_wk1_wget.sh on gpu03
wget -O /path/to/file https://dataverse.harvard.edu/api/access/datafile/{FILE_ID}
```

### 전체 WK1 데이터셋 (34.5 GB)

```bash
# Harvard Dataverse API — all files for DVN/BHQBB7
# Requires: script to parse JSON file listing and download each by ID
curl -s "https://dataverse.harvard.edu/api/datasets/export?exporter=dataverse_json&persistentId=doi:10.7910/DVN/BHQBB7" \
  | python3 -c "import json,sys; [print(f['dataFile']['id'], f.get('directoryLabel','')+'/'+f['dataFile']['filename']) for f in json.load(sys.stdin)['datasetVersion']['files']]"
```

---

*BehaviorSplatter | s-DANNCE Dataset Guide SSOT | v2.0 | 2026-03-22*

Related:
- ↑ [FaceLift Documentation Hub](../INDEX.md)
- ↔ [Multi-Animal Preprocessing Spec](MULTI_ANIMAL_PREPROCESSING.md)
- ↔ [Raw Data Sources](RAW_DATA.md)
- ↔ [sdannce-poc dataset_catalog](~/dev/sdannce-poc/docs/data/dataset_catalog.md)
- ↔ [CLAUDE.md s-DANNCE data convention](../../CLAUDE.md#sdannce-data-convention)
