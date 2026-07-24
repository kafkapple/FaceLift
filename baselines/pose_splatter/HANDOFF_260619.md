# HANDOFF — pose-splatter 복구 및 재실험

> **작성**: 2026-06-19. 이전 세션에서 accidental deletion 발생 후 re-clone 상태.  
> **목적**: 새 세션에서 conda env 재구성 → 데이터 연결 → 훈련 재실행 → BehaviorSplatter 비교 평가 완료.

---

## 현재 상태

> ✅ **260723 gpu03 실측 재확인 — 260619 작성 시점에서 진척 0. Step 0부터 미착수.**

| 항목 | 260619 | **260723 실측** |
|---|---|---|
| 코드 (`src/`, `scripts/`, `configs/`) | ✅ git 복구 | 🔴 **거짓이었음** — `src/modules/data/` 가 **git 미커밋**이라 entry point import 실패. 260723 `36b7fc2` 에서 복원하여 해소 |
| conda env `splatter` | ❌ | 🔴 **환경명 오기.** 실제명 `pose-splatter`. 게다가 **작동 환경 이미 존재**: `/node_data/joon/conda_envs/posesplatter` (torch 2.7.0, gsplat 1.5.3) → **재생성 불필요** |
| `data/raw/markerless_mouse_1_nerf` | ❌ | ❌ **symlink 없음** |
| `data/preprocessed/FaceLift_mouse` | ❌ | ❌ **symlink 없음** |
| `experiments/` (체크포인트) | ❌ 손실 | ❌ `fair/` 뿐, 체크포인트 없음 |
| `output/` (렌더링) | ❌ 손실 | ❌ **비어 있음** |
| `output/facelift_compare_5cam/` | — | ❌ **없음** |

**GPU 가용 (260723 확인)**: 4·5·6·7 전부 유휴 (2 MiB / 97,887 MiB). 프로젝트 규칙상 **4~7번만 사용**.

**참조 수치 (재현 목표)**:
- `5cam_baseline_gs`: PSNR=15.86, IoU=0.79, SSIM=0.955
- `5cam_4dgs_soft_keypoint`: PSNR=15.08, IoU=0.85, SSIM=0.969
- `6cam_facelift_compare` (NVS): PSNR=24.68, SSIM=0.963
- BS 비교 config: `configs/comparison/m5_4view_holdout3.json`, `m5_5view_holdout4.json`

---

## 🔴 재현 대상 누락 (260723 `/fact --int` 추가)

**본 핸드오프 Step 3-B는 FaceLift 논문의 핵심 baseline 수치를 복구하지 못함.**

| 항목 | 상태 |
|---|---|
| FaceLift SSOT 인용값 | **PSNR_gt_masked = 13.78, IoU = 0.846** (`FACELIFT_SSOT.md` §2.1) |
| 산출 실험 | `m5_baseline_gs` **6-view**, v11.0(260223), M5 same-camera 재학습 |
| 본 핸드오프 Step 3-B | `m5_4view_holdout3` + `m5_5view_holdout4` 만 재생성 → **6-view 누락** |
| 잔존 아티팩트 | `paper_standard_evaluation.json`=24.68(full-image) · `posesplatter_fair.json`=16.80(구본, "different camera") — **13.78 없음** |

→ **Step 3-B에 6-view config 추가 필요.**
→ 재평가는 **fair 프로토콜(FG-masked, test range 3240-3600, frame_step 5)** 으로 수행해야 13.78과 비교 가능. `paper_standard`(full-image)는 24.68 계열이라 대조 불가.

### 🔴 선행 차단 — Step 1(데이터 연결)이 성립하지 않음 (260723 실측)

`m5_baseline_gs.json` 이 요구하는 `data/preprocessed/markerless_mouse_1_nerf/m5_for_ps_fj1` 이 **없음**.
`~/data/preprocessed/markerless_mouse_1_nerf/` 는 **빈 디렉토리** — 본 핸드오프 Step 1의 symlink 두 개로는 해결 안 됨.

더 심각한 것: 이 데이터를 만든 **`convert_m5_for_ps.py` 가 git에 커밋된 적이 없어 소실**.
잔존 `scripts/fix_m5_*.py` 는 변환 후 사후 패치용이라 원 변환을 대체 못 함.

→ **Step 0(conda) 착수 전에 결정 필요**: M5→PS 변환기를 재작성할 것인가?
→ 재작성 시 결과는 **13.78 재현이 아니라 신규 baseline**. 상세 = `FaceLift/docs/FACELIFT_SSOT.md` §5.1

### Step 3-B' — 6-view 학습 (위 차단 해소 후)

```bash
conda activate splatter
cd /home/joon/dev/pose-splatter

# config 확인 — 없으면 m5_5view_holdout4.json 복사 후 holdout=[5], 6 cameras 로 수정
ls configs/comparison/m5_baseline_gs.json configs/comparison/m5_6view_holdout5.json 2>/dev/null

# 학습 (GPU 4~7 중 빈 것 사용. 260723 기준 전부 유휴)
CUDA_VISIBLE_DEVICES=4 nohup python scripts/run_experiment.py \
    configs/comparison/m5_6view_holdout5.json \
    --stage train --epochs 50 \
    > /node_data/joon/logs_posesplatter/m5_6view.log 2>&1 &
```

**camera space (필수 일치)**: HFOV=50°, fx=549, cx=cy=256, 512×512 — 불일치 시 v10.0의 "different camera" 무효 사례 재발.

**성공 기준**: fair 평가에서 `psnr_gt_masked ≈ 13.78 ± 0.5`, `iou ≈ 0.846 ± 0.02`.
- 재현 성공 → SSOT D11 해소, 인용 동결 해제 가능
- 재현 실패 → **13.78 폐기하고 신규값을 정본화**. 파생 gap 3종(+7.13/+9.62/+10.06) 전부 재산출 (D14)

> ⚠️ 예상 소요 ~25h. 프로젝트 규칙 `FaceLift/CLAUDE.md §8 ⛔ 실험 자동 실행 금지` 에 따라 **사용자가 직접 실행**할 것. 에이전트는 명령어만 제공.

**중요도**: 이 수치가 없으면 논문의 헤드라인 주장 `+10.06 dB over SOTA` 를 뒷받침할 근거가 없음. NeurIPS D&B Track = 재현성 심사 핵심.
상세 = `FaceLift/docs/FACELIFT_SSOT.md` §2.1 **C5** · Drift Ledger **D11**.

> ⚠️ 별건: PS 방법론은 per-scene optimization이 **아니라** feed-forward(dataset-trained)임. Step 4 주석의 "PS=masked(FG-only)" 는 유효하나, 방법론 분류 서술은 정정 대상 (D10).

---

> ✅ **260723 해소됨 — Step 0~2 불필요.** repo 실행 가능 상태 회복.
> 사용할 환경: `conda run -p /node_data/joon/conda_envs/posesplatter ...`
> 수정 2건: `src/data_utils.py`(git 36b7fc2 복원) · `src/modules/embedding/spherical_harmonics.py`(scipy≥1.17 shim)
> 상세 = `FaceLift/outputs/reports/260723_ps_repo_unblock.md`

## ~~Step 0 — Conda env 재생성~~ (불필요, 위 참조)

```bash
cd /home/joon/dev/pose-splatter
conda env create -f environment.yml
conda activate splatter

# 검증
python -c "import torch; import gsplat; print(torch.__version__, gsplat.__version__)"
# 기대: torch 2.0.x, gsplat 1.5+

# numpy 충돌 발생 시
pip install "numpy<2.0" --force-reinstall

# torch-scatter 설치 실패 시
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**성공 기준**: `python scripts/run_experiment.py --help` 오류 없이 실행.

---

## Step 1 — 데이터 연결

```bash
mkdir -p /home/joon/dev/pose-splatter/data/raw
mkdir -p /home/joon/dev/pose-splatter/data/preprocessed

# markerless_mouse_1_nerf (기본 baseline 데이터)
ln -s ~/data/preprocessed/markerless_mouse_1_nerf \
      /home/joon/dev/pose-splatter/data/raw/markerless_mouse_1_nerf

# M5 데이터 (BS 비교용)
ln -s ~/data/preprocessed/FaceLift_mouse \
      /home/joon/dev/pose-splatter/data/preprocessed/FaceLift_mouse
```

**검증**:
```bash
ls /home/joon/dev/pose-splatter/data/raw/markerless_mouse_1_nerf/ | head -3
# 기대: 프레임 디렉토리들 (000000, 000001, ...)
```

---

## Step 2 — Smoke test (환경 검증)

훈련 전 5 epoch 빠른 검증:

```bash
conda activate splatter
cd /home/joon/dev/pose-splatter

python scripts/run_experiment.py \
    configs/systematic/B0_baseline_3dgs.json \
    --stage train --epochs 5
```

**성공 기준**: loss 감소 확인, 오류 없이 5 epoch 완료.

---

## Step 3 — 우선 실험 (BS 비교 핵심)

### 3-A. Baseline 재현 (참조 수치 검증)

```bash
# 5cam 3DGS baseline — PSNR 15.86 재현 목표
nohup python scripts/run_experiment.py \
    configs/experiments/5cam_baseline_gs.json \
    --stage train --epochs 50 \
    > logs/5cam_baseline_gs.log 2>&1 &

# 완료 후 평가
python scripts/mouse/analysis/quick_evaluate.py \
    output/experiments/5cam_baseline_gs/latest --step 50
```

**성공 기준**: PSNR ≈ 15.86 ± 0.5, IoU ≈ 0.79 ± 0.02.  
수치 차이 > 1 PSNR → 데이터/환경 문제 의심, 중단 후 진단.

### 3-B. BS 비교 실험 (핵심 목적)

```bash
# M5 4-view (FL GS-LRM 3v 비교용)
nohup python scripts/run_experiment.py \
    configs/comparison/m5_4view_holdout3.json \
    --stage train --epochs 50 \
    > logs/m5_4view.log 2>&1 &

# M5 5-view (FL GS-LRM 4v 비교용)
nohup python scripts/run_experiment.py \
    configs/comparison/m5_5view_holdout4.json \
    --stage train --epochs 50 \
    > logs/m5_5view.log 2>&1 &
```

**성공 기준**: `benchmark_results.json`의 `5cam_3dgs_baseline` NVS PSNR=24.54 ± 0.5 재현.

### 3-C. Best model 재현 (optional, BS 논문 필요 시)

```bash
# 4DGS + Soft Keypoint (best IoU=0.85)
nohup python scripts/run_experiment.py \
    configs/experiments/5cam_4dgs_soft_keypoint.json \
    --stage train --epochs 50 \
    > logs/5cam_4dgs_soft_kpt.log 2>&1 &
```

---

## Step 4 — FaceLift 비교 평가

PS 결과를 FaceLift 비교 프로토콜로 평가:

```bash
# PS 평가 결과 생성
python scripts/eval/run_evaluation.sh \
    output/experiments/m5_4view_holdout3/latest

# FL 비교 (FaceLift repo에서 실행)
cd /home/joon/dev/FaceLift
python mouse_extensions/scripts/eval/compare_with_baseline.py \
    --ps_results /home/joon/dev/pose-splatter/output/facelift_compare_5cam/latest/paper_standard_evaluation.json
```

**주의**: PS=masked(FG-only), FL=white-BG(full-image). 비교 가능 지표: L1(masked), IoU. PSNR/SSIM 직접 비교 불가.

---

## 실험 진행 순서 요약

```
Step 0: conda env 재생성          (~30분)
Step 1: 데이터 symlink             (~5분)
Step 2: Smoke test (5 epoch)       (~15분)
Step 3-A: 5cam_baseline 재현       (~2-3시간)  ← 수치 검증 게이트
Step 3-B: m5_4view + m5_5view      (~4-6시간, 병렬 가능)  ← BS 비교 핵심
Step 3-C: 4dgs_soft_kpt (optional) (~4시간)
Step 4: FL 비교 평가               (~30분)
```

Step 3-A 수치가 기준치에서 크게 벗어나면 Step 3-B 진행 전 원인 파악 필수.

---

## 참고 문서

- 실험 결과 기록: `docs/practical/EXPERIMENT_RESULTS.md`
- 평가 프로토콜: `docs/practical/evaluation.md`
- 환경 트러블슈팅: `environment.yml` 주석
- WandB 프로젝트: `kafkapple-joon-kaist/posesplatter`
