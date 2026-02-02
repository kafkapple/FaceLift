# Mouse Quick Reference

> 명령어 중심 빠른 참조. 상세 이론/설정은 → [MOUSE_REFERENCE_DETAILS.md](MOUSE_REFERENCE_DETAILS.md)
> Last updated: 2026-02-02

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
    --resume_from_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2/checkpoint-5000 \
    > logs/mvdiff_M5t2.log 2>&1 &
```

**M5t2_consistent (처음부터 - 체크포인트 없음)**
```bash
export CUDA_VISIBLE_DEVICES=7 && nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t2_consistent.yaml \
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

### 2.3 체크포인트 현황 요약

| Model | Dataset | Last Ckpt | 권장 |
|-------|---------|-----------|------|
| GS-LRM | M5t2 | iter_901 | ⏳ Resume |
| GS-LRM | M5t | iter_12401 | ✅ 완료 |
| MVDiff | M5t2 | ckpt-5000 | ✅/Resume |
| MVDiff | M5t2_consistent | (none) | 🆕 시작 |
| MVDiff | M5t | ckpt-8000 | ✅ 완료 |

→ 상세: [MOUSE_REFERENCE_DETAILS.md#training](MOUSE_REFERENCE_DETAILS.md#training)


---

## 3. Inference

### Quick Cheatsheet

```bash
cd /home/joon/dev/FaceLift

# GS-LRM only (가장 간단)
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.simple_temporal \
    > logs/gslrm.log 2>&1 &

# E2E (1-view → 3D)
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 --end_frame 200 \
    --input_view_idx 0 --prompt_embed_path mouse_prompt_embeds_6view_1024 \
    --prefer_ema --skip_preprocess \
    > logs/e2e.log 2>&1 &
```

### 3.1 공통 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | M5t | **M5t** 또는 **M5t2** |
| `--end_frame` | 200 | 처리할 프레임 수 (-1 = 전체) |
| `--split` | (없음) | Split 파일 경로 (start/end 대신 사용) |
| `--fps` | 10 | 비디오 FPS |
| `--rotation_speed` | 0.3 | 회전 속도 |
| `--input_view_idx` | 0 | E2E 입력 뷰 (0-5) |

**Split 파일 경로**:
- M5t: `data_mouse_1to1_{train,val,test}.txt`
- M5t2: `data_mouse_t2_{train,val,test}.txt`

### 3.2 체크포인트 현황

| Model | Dataset | Checkpoint | 상태 |
|-------|---------|------------|------|
| GS-LRM | M5t | `M5t_E0_1_facelift/best_psnr.pt` | ✅ |
| GS-LRM | M5t2 | `M5t2_E0_1_facelift/best_psnr.pt` | ⏳ |
| MVDiffusion | M5t | `mouse_M5t/checkpoint-8000` | ✅ |
| MVDiffusion | M5t2 | `mouse_M5t2/checkpoint-5000` | ✅ |

경로: `/node_data/joon/checkpoints/FaceLift/{gslrm|mvdiffusion}/`

### 3.3 GS-LRM Only (6-view → 3D)

**기본 실행**
```bash
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.simple_temporal \
    --model {M5t|M5t2} \
    > logs/gslrm_{model}.log 2>&1 &
```

**Val/Test Split 전체**
```bash
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.simple_temporal \
    --model {M5t|M5t2} \
    --split ~/data/preprocessed/FaceLift_mouse/M5/{split_file} \
    --end_frame -1 \
    --output_dir outputs/gslrm_{model}_{split} \
    > logs/gslrm_{model}_{split}.log 2>&1 &
```

**변수 조합**:
| model | split_file |
|-------|------------|
| M5t | `data_mouse_1to1_val.txt`, `data_mouse_1to1_test.txt` |
| M5t2 | `data_mouse_t2_val.txt`, `data_mouse_t2_test.txt` |

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
    --prompt_embed_path mouse_prompt_embeds_6view_1024 \
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
    --prompt_embed_path mouse_prompt_embeds_6view_1024 \
    --prefer_ema --skip_preprocess \
    --turntable_views 60 \
    --output_dir outputs/e2e_{model}_{split} \
    > logs/e2e_{model}_{split}.log 2>&1 &
```

**예시 (M5t2 test, view 0)**:
```bash
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model M5t2 \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_test.txt \
    --input_view_idx 0 \
    --prompt_embed_path mouse_prompt_embeds_6view_1024 \
    --prefer_ema --skip_preprocess \
    --turntable_views 60 \
    --output_dir outputs/e2e_M5t2_test \
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

| 파라미터 | 기본 | 느린재생 | 설명 |
|----------|------|---------|------|
| --fps | 24 | 10 | 비디오 FPS |
| --rotation_speed | 0.5 | 0.3 | 회전 속도 |
| --num_views | 36 | 60 | 360° 분할 |

---

*Quick Reference | 2026-02-02*
*상세: [MOUSE_REFERENCE_DETAILS.md](MOUSE_REFERENCE_DETAILS.md)*
