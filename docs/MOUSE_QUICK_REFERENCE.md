# Mouse Quick Reference

> 명령어 중심 빠른 참조. 상세 이론/설정은 → [MOUSE_REFERENCE_DETAILS.md](MOUSE_REFERENCE_DETAILS.md)
> Last updated: 2026-02-03

---

## 1. Data Preparation

### 1.1 Raw Data 정보

| 항목 | 값 |
|------|-----|
| 위치 | `/home/joon/data/raw/markerless_mouse_1_nerf` |
| 프레임 | 18,000 (6 카메라) |
| 해상도 | 1152×1024 |
| FPS | 100 |

### 1.2 전처리 명령어

```bash
cd /home/joon/dev/FaceLift

# M5 (권장 - Affine, 512x512)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M5 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M5
```

### 1.3 데이터셋 현황

| Preset | Split | Train/Val/Test | 용도 |
|--------|-------|---------------|------|
| **M5** | random 80:10:10 | 2880/360/360 | 일반 |
| **M5t** | temporal 1:1:1 | 1198/1198/1204 | Pose-Splatter 비교 |
| **M5t2** | temporal 80:10:10 | 2880/360/360 | ⭐ **권장** |

→ 상세: [MOUSE_REFERENCE_DETAILS.md#datasets](MOUSE_REFERENCE_DETAILS.md#datasets)

---

## 2. Training

### 2.1 GS-LRM

**M5t2 Resume (권장)**
```bash
cd /home/joon/dev/FaceLift

# Resume from last checkpoint (자동 감지)
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e E0_1_facelift \
    > logs/M5t2_E0_1.log 2>&1 &
```

**M5t (참고)**
```bash
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t -e E0_1_facelift \
    > logs/M5t_E0_1.log 2>&1 &
```

**처음부터 학습**: `rm -rf /node_data/joon/checkpoints/FaceLift/gslrm/{dataset}_{experiment}/`

### 2.2 MVDiffusion

**M5t2 (checkpoint-5000 이후 resume)**
```bash
cd /home/joon/dev/FaceLift

export CUDA_VISIBLE_DEVICES=7 && nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t2.yaml \
    --resume_from_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_cfgr/checkpoint-10000 \
    > logs/mvdiff_M5t2.log 2>&1 &
```

**M5t2_consistent (⚠️ 실험적 - CFG=0, FullAttn)**
```bash
# 주의: CFG dropout=0, sparse_mv_attention=false 설정
# 결과 검증 필요 - 3.7절 참조
export CUDA_VISIBLE_DEVICES=7 && nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t2_consistent.yaml \
    --resume_from_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_consistent/checkpoint-6000 \
    > logs/mvdiff_M5t2_consistent.log 2>&1 &
```

**M5t (완료됨 - checkpoint-8000)**
```bash
# 필요 시 추가 학습
export CUDA_VISIBLE_DEVICES=7 && nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t.yaml \
    --resume_from_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t/checkpoint-8000 \
    > logs/mvdiff_M5t.log 2>&1 &
```

⚠️ **GPU 선택**: `export CUDA_VISIBLE_DEVICES=N &&` 형식 필수 (env_vars.sh 덮어쓰기 방지)

### 2.2.1 우선순위 학습 실험: CFG 복원 (H1 검증 후)

**배경**: M5t2_consistent (CFG=0, FullAttn) 결과가 baseline보다 안 좋음.
**가설**: CFG dropout 제거가 주 원인 → 복원 후 재학습

**Exp-C: CFG 복원, Full Attention 유지**
```bash
cd /home/joon/dev/FaceLift

# 1. Config 생성 (M5t2_cfgr.yaml)
# condition_drop_rate: 0.05 (복원)
# sparse_mv_attention: false (full 유지)

# 2. 학습
export CUDA_VISIBLE_DEVICES=4 && nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t2_cfgr.yaml \
    > logs/mvdiff_M5t2_cfgr.log 2>&1 &
```

**상세**: `docs/research/260203_MVDiffusion_CFG_Ablation.md` (7절)

### 2.3 체크포인트 현황 요약

| Model | Dataset | Last Ckpt | 상태 | 비고 |
|-------|---------|-----------|------|------|
| GS-LRM | M5t | iter_12401 | ✅ 완료 | best_psnr.pt |
| GS-LRM | M5t2 | iter_901 | ⏳ Resume | - |
| MVDiff | M5t | ckpt-8000 | ✅ 완료 | **Baseline** |
| MVDiff | M5t2 | ckpt-5000 | ✅/Resume | CFG=0.05 |
| MVDiff | M5t2_consistent | ckpt-6000 | ⚠️ 검증필요 | CFG=0, FullAttn |

→ 상세: [MOUSE_REFERENCE_DETAILS.md#training](MOUSE_REFERENCE_DETAILS.md#training)


---


### 2.4 Deformation Network (Temporal Consistency)

**목적**: 프레임별 독립 재구성 → 시간적 연속성 확보 (temporal flickering 해결)

**현재 상태**: ✅ 구현 완료, 학습 진행 중

**아키텍처**:
- 8-layer MLP (397K params)
- Input: Gaussian xyz [N, 3] → Output: Δxyz, Δα, Δs [N, 5]
- Autoregressive: G_t → D(G_t) → G'_{t+1}

**학습 명령어**:
```bash
# Deformation 학습 (캐시 생성 + MLP 학습)
cd /home/joon/dev/FaceLift
CUDA_VISIBLE_DEVICES=6 nohup python -m mouse_extensions.scripts.train_deformation \
    --config configs/deformation/default.yaml > logs/deformation_train.log 2>&1 &

# 진행 확인
grep -oP '\d+/10000' logs/deformation_train.log | tail -1
tail -f logs/deformation_train.log
```

**학습 설정** (`configs/deformation/default.yaml`):
| 항목 | 값 |
|------|-----|
| GS-LRM | M5t2_E0_1_facelift/best_psnr.pt |
| 데이터 | M5 + t2 split (2880 frames) |
| Steps | 10000 |
| 캐시 | ~158GB (56MB × 2880 frames) |
| 속도 | ~4 it/s, ~40분 total |

**상세**: `docs/guides/DEFORMATION_INTEGRATION_GUIDE.md`


## 3. Inference

### ⚠️ 중요 변경사항 (2026-02-03)

| 변경 | 이전 | 이후 | 영향 |
|------|------|------|------|
| `--input_view_idx` 기본값 | 0 | **None** | 미지정 시 GS-LRM only |
| `--split` 옵션 | 표시만 | **실제 적용** | Split 파일에서 샘플 로드 |
| `--num_frames` 추가 | - | **신규** | 리스트 슬라이싱 (프레임 번호 대신 개수) |
| `--end_frame` + split | 무시됨 | **적용됨** | Split 모드에서도 제한 가능 |
| Batch 출력 구조 | 플랫 | `samples/` | 정리된 폴더 구조 |
| Temporal 비디오 | ✅ 통합됨 | ✅ | turntable_first, time_rotating, turntable_grid 자동 생성 |
| `--save_gaussian` | 기본 True | **기본 False** | 명시적 옵션 필요 |

### 3.0.1 추론 스크립트 개요

| 스크립트 | 용도 | 입력 | 출력 |
|----------|------|------|------|
| ~~~~ | ⚠️ DEPRECATED | - | → 로 이동됨 |
| `run_e2e_inference.py` | E2E 또는 GS-LRM only | 1-view 또는 6-view | ✅ 동일 (temporal 비디오 포함) |
| `run.py` (unified) | Config 기반 통합 | config YAML | config에 따름 |

**권장 스크립트**:
- **GS-LRM only (GT 상한선)**: `run_e2e_inference.py` (without --input_view_idx)
- **E2E (단일 뷰 → 3D)**: `run_e2e_inference.py --input_view_idx {0-5}`
- **E2E (설정 기반)**: `run.py --config configs/inference/e2e.yaml`

### Quick Cheatsheet

```bash
cd /home/joon/dev/FaceLift

# GS-LRM only (6-view → 3D)
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    > logs/gslrm.log 2>&1 &

# E2E (1-view → 3D) - 최소 필수 옵션
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    > logs/e2e.log 2>&1 &
```
**기본값**: `--model M5t`, `--split test`, `--end_frame 200`, `--prefer_ema`, `--skip_preprocess`
⚠️ **E2E 모드 필수**: `--input_view_idx {0-5}` 지정해야 E2E 동작 (미지정 시 GS-LRM only)

### 3.1 공통 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | M5t | **M5t** 또는 **M5t2** |
| `--num_frames` | None | 프레임 개수 (리스트 슬라이싱, start/end보다 우선) |
| `--end_frame` | 200 | 처리할 프레임 인덱스 상한 |
| `--split` | test | Split 파일 (자동 적용) |
| `--fps` | 10 | 비디오 FPS |
| `--rotation_speed` | 0.3 | 회전 속도 |
| `--input_view_idx` | **None** | E2E 입력 뷰 (0-5). ⚠️ **미지정 시 GS-LRM only** |
| `--turntable_views` | 60 | Turntable 뷰 수 (run_e2e) |
| `--rotation_speed` | 0.3 | 회전 속도 (0.3=느림, 1.0=보통) |
| ~~~~ | ⚠️ DEPRECATED | - | → 로 이동됨 |

**Split 명명 규칙**:
| 명칭 | 분할 | 샘플 수 | 용도 |
|------|------|---------|------|
| **1to1** | temporal 1:1:1 | 1198/1198/1204 | Pose-Splatter 비교 |
| **t2** | temporal 80:10:10 | 2880/360/360 | ⭐ 권장 (train 최대화) |

**Split 파일 경로 (M5 디렉토리 내)**:
- M5t: `data_mouse_1to1_{train,val,test}.txt`
- M5t2: `data_mouse_t2_{train,val,test}.txt`
⚡ **자동 기본값**: `--split` 미지정 시 `--model`에 따라 test split 자동 적용
  - M5t → `data_mouse_1to1_test.txt`
  - M5t2 → `data_mouse_t2_test.txt`

💡 **왜 모델이 split을 결정하나?**: 각 모델은 특정 split 전략으로 학습됨
  - M5t = temporal 1:1:1로 학습 → 1to1 split 사용
  - M5t2 = temporal 80:10:10으로 학습 → t2 split 사용
  - 데이터 디렉토리(M5)는 동일, split 파일만 다름
  - **잘못된 split 사용 시 data leakage 위험** (train 데이터로 평가)


### 3.2 체크포인트 현황

| Model | Dataset | Checkpoint | 상태 |
|-------|---------|------------|------|
| GS-LRM | M5t | `M5t_E0_1_facelift/best_psnr.pt` | ✅ |
| GS-LRM | M5t2 | `M5t2_E0_1_facelift/best_psnr.pt` | ✅ |
| MVDiffusion | M5t | `mouse_M5t/checkpoint-8000` | ✅ |
| MVDiffusion | M5t2 | `mouse_M5t2_cfgr/checkpoint-10000` | ✅ |

경로: `/node_data/joon/checkpoints/FaceLift/{gslrm|mvdiffusion}/`

### 3.3 GS-LRM Only (6-view → 3D)

**기본 실행**
```bash
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model {M5t|M5t2} \
    > logs/gslrm_{model}.log 2>&1 &
```

**Val/Test Split 전체**
```bash
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model {M5t|M5t2} \
    --split ~/data/preprocessed/FaceLift_mouse/M5/{split_file} \
    --end_frame -1 \
    --output_dir outputs/gslrm_{model}_{split} \
    > logs/gslrm_{model}_{split}.log 2>&1 &
```

**변수 조합**:
| model | split_file |
|-------|------------|
| M5t | `data_mouse_1to1_train.txt`, `data_mouse_1to1_val.txt`, `data_mouse_1to1_test.txt` |
| M5t2 | `data_mouse_t2_train.txt`, `data_mouse_t2_val.txt`, `data_mouse_t2_test.txt` |

### 3.4 E2E (1-view → MVDiffusion → GS-LRM → 3D)

**뷰 선택 (`--input_view_idx`)**

| 뷰 | 각도 | 설명 |
|----|------|------|
| **0** | Top-front | ⭐ 기본 (정보량 최대) |
| 1-2 | Top-left/right | 측면 |
| 3 | Top-back | 후면 |
| 4-5 | Side-left/right | 낮은 각도 |

**기본 실행**
```bash
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model {M5t|M5t2} \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --start_frame 0 --end_frame 200 \
    --input_view_idx {0-5} \
    \
    --prefer_ema --skip_preprocess \
    --turntable_views 60 \
    --output_dir outputs/e2e_{model}_view{idx} \
    > logs/e2e_{model}_view{idx}.log 2>&1 &
```

**Val/Test Split 전체**
```bash
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model {M5t|M5t2} \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split ~/data/preprocessed/FaceLift_mouse/M5/{split_file} \
    --input_view_idx 0 \
    \
    --prefer_ema --skip_preprocess \
    --turntable_views 60 \
    --output_dir outputs/e2e_{model}_{split} \
    > logs/e2e_{model}_{split}.log 2>&1 &
```

**예시 (M5t2 test, view 0)**:
```bash
export CUDA_VISIBLE_DEVICES=7 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model M5t2 \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_test.txt \
    --input_view_idx 0 \
    \
    --prefer_ema --skip_preprocess \
    --turntable_views 60 \
    --output_dir outputs/e2e_M5t2_test \
    > logs/e2e_M5t2_test.log 2>&1 &
```
또는
```
export CUDA_VISIBLE_DEVICES=7 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
--model M5t2 --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
> logs/e2e_M5t2_test.log 2>&1 &
```

### 3.5 Wild Image → 3D

```bash
export CUDA_VISIBLE_DEVICES=6 && python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --input_image /path/to/mouse.jpg \
    --sam_checkpoint checkpoints/sam/sam_vit_b.pth \
    --output_dir outputs/wild
```

→ 상세: [MOUSE_REFERENCE_DETAILS.md#inference](MOUSE_REFERENCE_DETAILS.md#inference)

### 3.5.5 출력 구조

**run_e2e_inference (GS-LRM temporal)**
```
outputs/gslrm_M5t_test/
├── turntable_first.mp4 # 첫 프레임 360°
├── time_fixed.mp4      # 고정 각도, 시간 변화
├── time_rotating.mp4   # 회전하며 시간 변화
├── grid_6view.mp4      # 입력 6뷰 그리드
├── grid_first.jpg      # 첫 프레임 turntable 그리드
└── gaussians/          # --save_gaussian 시에만
    └── frame_XXXXXX.ply
```

**run_e2e_inference (batch)**
```
outputs/e2e_M5t_test_view0_260203/   # ⭐ Auto naming: {mode}_{model}_{split}_view{idx}_{YYMMDD}
├── run_config.json     # ⭐ 실행 설정 자동 저장
├── turntable_first.mp4 # temporal 비디오는 루트에
├── time_fixed.mp4
├── time_rotating.mp4
├── grid_6view.mp4
└── samples/            # 프레임별 폴더는 samples/ 하위
    ├── 000000/
    │   └── cam_000/    # E2E 모드
    │       ├── turntable.mp4
    │       └── gaussians.ply
    └── ...
```

**Auto 출력 폴더 명명**:
- `--output_dir` 미지정 시 자동 생성
- 패턴: `{mode}_{model}_{split}_view{idx}_n{frames}_{YYMMDD}`
- 예: `e2e_M5t_test_view0_n10_260203`, `gslrm_M5t2_val_260203`

### 3.6 추천 시각화 실험

| 실험 | 목적 |
|------|------|
| **GS-LRM only** | MVDiffusion 없이 GT 6뷰 → 3D (E2E 상한선) |
| **Train split** | overfitting check (학습 데이터 재현 확인) |
| **다른 입력 뷰** | view 0 (top-front) vs view 5 등 비교 |
| **Val split** | 학습 중 본 데이터로 sanity check |
| **M5t2 모델** | 더 많은 train 데이터로 학습된 모델 |

```bash
cd /home/joon/dev/FaceLift

# GS-LRM only (GT 6뷰 → 3D, E2E 상한선) - Test split
export CUDA_VISIBLE_DEVICES=7 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model M5t --num_frames 200 \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_1to1_test.txt \
    --output_dir outputs/gslrm_M5t_test \
    > logs/gslrm_M5t_test.log 2>&1 &

# GS-LRM only - Train split (overfitting check)
export CUDA_VISIBLE_DEVICES=7 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model M5t \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_1to1_train.txt \
    --num_frames 200 \
    --output_dir outputs/gslrm_M5t_train \
    > logs/gslrm_M5t_train.log 2>&1 &

# E2E view 0 (Top-front, 정보량 최대)
export CUDA_VISIBLE_DEVICES=7 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model M5t --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --input_view_idx 0 --num_frames 20 \
    --split ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_1to1_train.txt \
    --output_dir outputs/e2e_M5t_train \
    > logs/e2e_M5t_train.log 2>&1 &

 
# Val split E2E (sanity check) - input_view_idx 필수!
export CUDA_VISIBLE_DEVICES=7 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model M5t --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_1to1_val.txt \
    --input_view_idx 0 --num_frames 10 \
    --output_dir outputs/e2e_M5t_val \
    > logs/e2e_M5t_val.log 2>&1 &

# test
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model M5t --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --input_view_idx 0 --num_frames 20 \
    --output_dir outputs/e2e_M5t_test \
    > logs/e2e_M5t_test.log 2>&1 &

# Train split E2E (overfitting check) - input_view_idx 필수!
export CUDA_VISIBLE_DEVICES=7 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model M5t --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_1to1_train.txt \
    --input_view_idx 0 --num_frames 10 \
    --output_dir outputs/e2e_M5t_train \
    > logs/e2e_M5t_train.log 2>&1 &



```

**⚠️ `--num_frames` vs `--end_frame` 사용 시점**:
| 옵션 | 동작 | 사용 시점 |
|------|------|-----------|
| `--num_frames N` | 리스트 처음 N개 | Split 파일 사용 시 (프레임 번호 무관) |
| `--end_frame N` | 인덱스 N 미만 | 연속 인덱스 범위 지정 시 |

예: Split 파일에 `002396, 002401, ...` 포함 → `--end_frame 20`은 **0개** (번호가 20 미만인 프레임 없음)
→ 대신 `--num_frames 20` 사용 (처음 20개 샘플)

### 3.7 MVDiffusion CFG Ablation (260204 Updated)

**배경**: M5t2_consistent 결과가 baseline보다 안 좋음 → CFG/Attention 영향 검증
**상세**: `docs/research/260203_MVDiffusion_CFG_Ablation.md`

#### 통제 변수 ⚠️

**GS-LRM 고정** (MVDiffusion 품질만 비교):
```
/node_data/joon/checkpoints/FaceLift/gslrm/M5t_E0_1_facelift/best_psnr.pt
```

#### 설정 비교

| 실험 | CFG | Sparse | ckpt | 역할 |
|------|-----|--------|------|------|
| **Ctrl** | 0.05 | true | M5t/8000 | Baseline |
| **Exp-A** | 0.05 | true | M5t2/5000 | Split+Prompt |
| **Exp-B** | 0.0 | false | M5t2_consistent/6000 | H1+H2 검증 |

#### 명령어

```bash
cd /home/joon/dev/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh && conda activate facelift

# === Ctrl: M5t baseline ===
export CUDA_VISIBLE_DEVICES=4 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model M5t \
    --mvdiffusion_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t/checkpoint-8000 \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_1to1_test.txt \
    --input_view_idx 0 --num_frames 200 \
    --output_dir outputs/compare_mvdiff/ctrl_M5t_8k \
    > logs/compare_ctrl_M5t.log 2>&1 &

# === Exp-A: M5t2 ===
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model M5t \
    --mvdiffusion_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_cfgr/checkpoint-10000 \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_1to1_test.txt \
    --input_view_idx 0 --num_frames 200 \
    --output_dir outputs/compare_mvdiff/exp_A_M5t2_5k \
    > logs/compare_expA_M5t2.log 2>&1 &

# === Exp-B: M5t2_consistent ===
export CUDA_VISIBLE_DEVICES=7 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model M5t \
    --mvdiffusion_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_consistent/checkpoint-6000 \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_1to1_test.txt \
    --input_view_idx 0 --num_frames 200 \
    --output_dir outputs/compare_mvdiff/exp_B_M5t2_consistent_6k \
    > logs/compare_expB_M5t2_consistent.log 2>&1 &
```

#### Split 옵션

| Split | 파일명 | 샘플 수 |
|-------|--------|---------|
| **1to1 test** (기본) | `data_mouse_1to1_test.txt` | 1204 |
| 1to1 train | `data_mouse_1to1_train.txt` | 1198 |
| t2 train | `data_mouse_t2_train.txt` | 2880 |

### 3.8 Temporal Inference (Deformation Network)

**목적**: 비디오 시퀀스의 시간적 일관성 확보

#### 3.8.1 현재 상태

| 구성요소 | 상태 | 비고 |
|----------|------|------|
| Deformation Network | ✅ 완료 | checkpoint_010000.pt |
| Gaussian Cache | ✅ 완료 | 2880 frames (158GB) |
| 추론 스크립트 | ✅ 완료 | run_temporal_inference.py |
| **평가 모듈** | ✅ **완료** | temporal_evaluator.py |
| **WandB 로깅** | ✅ **완료** | FaceLift-Mouse 프로젝트 |

#### 3.8.2 평가 결과 (2880 프레임)

| 메트릭 | Before | After | 변화 |
|--------|--------|-------|------|
| **Temporal Jitter** | 0.0167 | 0.0012 | **-92.7%** |
| Mean Displacement | - | 0.110 | - |

#### 3.8.3 명령어

**추론**: 
**평가**: 

**상세**: 


---
## 4. Utilities

### 4.1 프로세스 관리

```bash
# GPU 프로세스 확인
gj           # 간략
gj -f        # 전체 명령어

# 특정 실험 종료
pgrep -af "M5t2.*E0_1"           # 확인
pkill -f "M5t2.*E0_1_facelift"   # 종료
kill <PID>                       # PID로 종료
```

### 4.2 로그 확인

```bash
tail -f logs/M5t2_E0_1.log       # GS-LRM
tail -f logs/mvdiff_M5t2.log     # MVDiffusion
```

### 4.3 검증

```bash
# PP/MVG 일관성
python mouse_extensions/scripts/diagnostics/verify_pp_mvg_consistency.py --datasets M5 --verbose

# 클리핑 분석
python mouse_extensions/scripts/analysis/clipping_analyzer.py \
    /home/joon/data/preprocessed/FaceLift_mouse/M5 -r 50 -m 10
```

### 4.4 Rerun 시각화

```bash
# 로컬
rerun outputs/temporal_M5t_slow/rerun/sequence.rrd

# SSH 터널 (원격)
ssh -L 9090:localhost:9090 gpu03
rerun --web-viewer --port 9090 outputs/.../sequence.rrd
```

---

## 5. Quick Cheatsheet

### 변수 조합

| 변수 | 값 | 설명 |
|------|-----|------|
| `MODEL` | M5t, M5t2 | 데이터셋 |
| `SPLIT` | 1to1 (M5t), t2 (M5t2) | Split 파일 |
| `MVDIFF_CKPT` | checkpoint-8000 (M5t), checkpoint-5000 (M5t2) | MVDiff 체크포인트 |

### GPU 할당 (gpu03)

| GPU | 용도 |
|-----|------|
| 0-3 | ❌ Blackwell (미지원) |
| 4-7 | ✅ A6000 |

### Inference 파라미터

| 파라미터 | 기본 | 설명 |
|----------|------|------|
| --fps | 10 | 비디오 FPS |
| --turntable_views | 60 | Turntable 뷰 수 (run_e2e) |
| --rotation_speed | 0.3 | 회전 속도 (0.3=느림, 1.0=보통) |
| --grid_views | 36 | Grid 이미지 뷰 수 (6x6) |

---

*Quick Reference | 2026-02-03*
*상세: [MOUSE_REFERENCE_DETAILS.md](MOUSE_REFERENCE_DETAILS.md)*
