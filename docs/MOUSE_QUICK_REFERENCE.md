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

```bash
cd /home/joon/dev/FaceLift

# 기본 (M5t2 + E0_1_facelift)
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e E0_1_facelift \
    > logs/M5t2_E0_1.log 2>&1 &

# 다른 조합
CUDA_VISIBLE_DEVICES=5 torchrun ... train_gslrm.py -d {M5|M5t|M5t2} -e {E0_1_facelift|E1_2_alpha}
```

**Resume**: 동일 명령어 (자동 감지)
**처음부터**: `rm -rf /node_data/joon/checkpoints/FaceLift/gslrm/{dataset}_{experiment}/`

### 2.2 MVDiffusion

```bash
cd /home/joon/dev/FaceLift

# M5t2 (권장)
export CUDA_VISIBLE_DEVICES=7 && nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t2.yaml \
    > logs/mvdiff_M5t2.log 2>&1 &

# Resume (별도 config)
export CUDA_VISIBLE_DEVICES=7 && nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t2_resume.yaml \
    > logs/mvdiff_M5t2.log 2>&1 &
```

⚠️ **GPU 선택**: `export CUDA_VISIBLE_DEVICES=N &&` 형식 필수 (env_vars.sh 덮어쓰기 방지)

→ 상세: [MOUSE_REFERENCE_DETAILS.md#training](MOUSE_REFERENCE_DETAILS.md#training)

---

## 3. Inference

### Quick Cheatsheet

```bash
cd /home/joon/dev/FaceLift

# GS-LRM (가장 간단)
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.simple_temporal > logs/gslrm.log 2>&1 &

# E2E (1-view → 3D)
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 --end_frame 200 \
    --prompt_embed_path mouse_prompt_embeds_6view_1024 --prefer_ema --skip_preprocess \
    > logs/e2e.log 2>&1 &
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | M5t | M5t 또는 M5t2 |
| `--end_frame` | 200 | 처리할 프레임 수 |
| `--fps` | 10 | 비디오 FPS (slow) |
| `--rotation_speed` | 0.3 | 회전 속도 |

### 3.1 체크포인트 현황

| Model | Dataset | Checkpoint | 상태 |
|-------|---------|------------|------|
| GS-LRM | M5t | `M5t_E0_1_facelift/best_psnr.pt` | ✅ |
| GS-LRM | M5t2 | `M5t2_E0_1_facelift/best_psnr.pt` | ✅ |
| MVDiffusion | M5t | `mouse_M5t/checkpoint-8000` | ✅ |
| MVDiffusion | M5t2 | `mouse_M5t2/checkpoint-5000` | ✅ |

경로: `/node_data/joon/checkpoints/FaceLift/{gslrm|mvdiffusion}/`

### 3.2 GS-LRM Only (6-view → 3D)

**기본 (M5t, slow playback 200 frames)**
```bash
cd /home/joon/dev/FaceLift
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.simple_temporal \
    > logs/temporal_M5t.log 2>&1 &
```

**M5t2 모델**
```bash
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.simple_temporal \
    --model M5t2 > logs/temporal_M5t2.log 2>&1 &
```

**커스텀 설정**
```bash
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.simple_temporal \
    --model M5t2 --end_frame 500 --fps 24 --rotation_speed 0.5 \
    --output_dir outputs/custom > logs/custom.log 2>&1 &
```

기본값: fps=10, rotation_speed=0.3, num_views=60, end_frame=200

### 3.3 E2E (1-view → MVDiffusion → GS-LRM → 3D)

**기본 (M5t 모델)**
```bash
cd /home/joon/dev/FaceLift
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --start_frame 0 --end_frame 200 \
    --prompt_embed_path mouse_prompt_embeds_6view_1024 \
    --prefer_ema --skip_preprocess \
    --turntable_views 60 \
    > logs/e2e_M5t.log 2>&1 &
```

**M5t2 모델**
```bash
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model M5t2 \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --start_frame 0 --end_frame 200 \
    --prompt_embed_path mouse_prompt_embeds_6view_1024 \
    --prefer_ema --skip_preprocess \
    > logs/e2e_M5t2.log 2>&1 &
```

⚠️ E2E는 `--data_dir`과 프레임 범위는 명시 필요 (GS-LRM only와 다름)

### 3.4 Wild Image → 3D (SAM 전처리 포함)

```bash
export CUDA_VISIBLE_DEVICES=6 && python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --input_image /path/to/wild_mouse.jpg \
    --sam_checkpoint checkpoints/sam/sam_vit_b.pth \
    --mvdiffusion_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t/checkpoint-8000 \
    --gslrm_checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t_E0_1_facelift/best_psnr.pt \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/wild_inference
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
