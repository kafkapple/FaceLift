# Mouse Reference Details

> 상세 이론, 설정 근거, 트러블슈팅. 명령어는 → [MOUSE_QUICK_REFERENCE.md](MOUSE_QUICK_REFERENCE.md)
> Last updated: 2026-02-02

---

## Datasets

### Raw Data: markerless_mouse_1_nerf

| 항목 | 값 |
|------|-----|
| 총 프레임 | 18,000 (6개 카메라 동일) |
| FPS | 100 |
| 해상도 | 1152×1024 |
| frame_interval=5 | → 3,600 샘플 |

#### Frame Discontinuity (불연속 위치)
```
DISCONTINUITY_FRAMES = {5900, 11800, 17700}
```
- 해당 위치에서 원본 녹화의 프레임 불연속(갭) 발생
- 해당 프레임 자체는 **정상** (제외 불필요)
- temporal smoothness 가정하는 알고리즘 사용 시 주의

### 데이터셋 버전 체계

| Preset | PP | fx | Transform | 용도 |
|--------|-----|-----|-----------|------|
| D7.1 (M1) | 256 | 549 | Affine | 기준선 |
| D8 (M2) | 256 | 549 | Homography | 정밀 기준선 |
| **M5** | 256 | 549 | Affine (512×512) | ⭐ **현재 권장** |

### Split 전략

| Split | 데이터셋 | Train/Val/Test | 용도 |
|-------|---------|---------------|------|
| random 80:10:10 | M5 | 2880/360/360 | 일반 학습 |
| temporal 1:1:1 | M5t | 1198/1198/1204 | Pose-Splatter 비교 |
| **temporal 80:10:10** | **M5t2** | 2880/360/360 | ⭐ **권장** (data leakage 방지 + train 최대화) |

### 폐기 데이터셋

| ID | Issue | Ray Error |
|----|-------|-----------|
| ~~M3~~ | fx=739 unnormalized | 6.96 deg |
| ~~M3_norm~~ | Variable PP | 13.62 deg |
| ~~M3_persample~~ | Variable PP | 16.15 deg |

---

## Training

### GS-LRM 실험 설정

| 설정 | mask_mode | alpha_loss | 용도 |
|------|-----------|------------|------|
| E0_1_facelift | none | 0.0 | 마스크 없음 (안정적) |
| **E1_2_alpha** | gt | 0.1 | ⭐ **권장** |
| E1_3_lgm | gt | 1.0 | 강한 alpha |

### WandB Auto-Resume

**동작 방식**: checkpoint 디렉토리에 `wandb_run_id.txt` 자동 저장/로드

```
첫 학습 시:
  wandb.init() → 새 run 생성 → wandb_run_id.txt 저장

Resume 시:
  wandb_run_id.txt 확인 → run_id 로드 → 이전 run 이어서
```

**수동 run_id 지정** (이전 실험용):
```bash
# WandB UI에서 run_id 확인 후
echo "yz9fl25q" > /node_data/joon/checkpoints/.../wandb_run_id.txt
```

### MVDiffusion Config 체계

| Config | 용도 | 핵심 설정 |
|--------|------|-----------|
| mouse_mvdiffusion_M5t.yaml | **새로 시작** | resume: null |
| mouse_mvdiffusion_M5t_resume.yaml | **이어서 학습** | resume: latest, wandb_run_id |

**왜 Config 분리?**
- resume_from_checkpoint: latest 남아있으면 → 새 학습 시 의도치 않게 resume
- wandb_run_id 남아있으면 → 새 실험 로그가 이전 run에 오염

### MVDiffusion 학습 특성

**Early Stopping 불필요:**
- Diffusion 모델은 학습이 매우 안정적 (mode collapse 없음)
- Validation loss가 noisy해서 best 판단 어려움
- 고정 step + EMA + 마지막 checkpoint 사용이 일반적

### GPU 선택 주의사항

**문제**: `conda activate facelift` 시 `env_vars.sh`가 `CUDA_VISIBLE_DEVICES=4` 자동 설정

**해결**: `export`로 먼저 설정
```bash
# 올바른 방법
export CUDA_VISIBLE_DEVICES=6 && nohup accelerate launch ...

# 잘못된 방법 (GPU 4로 덮어씌워짐)
CUDA_VISIBLE_DEVICES=6 nohup accelerate launch ...
```

---

## Inference

### 비디오 출력 종류

| 파일 | 설명 | 축 |
|------|------|-----|
| `turntable.mp4` | 각 프레임의 360도 회전 연결 | 회전 (시간 고정) |
| `time_fixed_0.mp4` | 고정 각도(0도)에서 시간 변화 | 시간 (회전 고정) |
| `time_rotating.mp4` | **시간+회전 동시 변화** | 시간 ↔ 회전 |
| `full_all.mp4` | 모든 시간 × 모든 각도 | T×V 전체 |

#### time_rotating 계산 방식
```python
for t in range(T):           # 시간 프레임
    angle = (t * V // T) % V  # 시간에 비례해서 각도 증가
    frame = turntables[t][angle]
```

### Slow Motion 설정

| 용도 | fps | rotation_speed | num_views |
|------|-----|----------------|-----------|
| 빠른 미리보기 | 24 | 0.5 | 36 |
| 발표용 | 15 | 0.4 | 48 |
| **분석용** | **10** | **0.3** | **60** |
| 디버깅 | 5 | 0.2 | 72 |

#### Slow Motion 잔상 문제 (260202 수정)

**원인**: `interpolate_frames_for_speed()`가 프레임 간 블렌딩
```python
blended = (1-t) * frame[lower] + t * frame[upper]  # 잔상 발생\!
```

**수정**: 기본값을 nearest neighbor로 변경 (`interpolate=False`)
```python
nearest = int(np.round(idx))
new_frames.append(frames[nearest])  # 잔상 없음
```

### E2E 카메라 파라미터 주의

MVDiffusion output → GS-LRM input 시 **M5 카메라 파라미터 필수**.
`MVDiffusionInference.compute_cameras()` 사용. **절대 identity matrix 사용 금지**.

---

## MVDiffusion Prompts

### 생쥐 전용 프롬프트

기존 MVDiffusion은 수평 뷰 (front, back) 사용.
생쥐 데이터는 **위에서 비스듬히** 촬영 → 뷰 방향 불일치.

| 항목 | 값 |
|------|-----|
| 경로 | `mouse_prompt_embeds_6view_1024/clr_embeds.pt` |
| Shape | [6, 77, 1024] (SD 2.1 unclip) |
| 뷰 | top-front, top-front-right, top-right, top-back, top-left, top-front-left |

### CLIP Embedding Dimension

| SD 버전 | Text Encoder | Dim |
|---------|--------------|-----|
| SD 1.5 | CLIP ViT-L/14 | 768 |
| **SD 2.x** | **CLIP ViT-H/14** | **1024** |

MVDiffusion은 SD 2.1 unclip 기반 → **1024-dim 필수**

### Prompt Ablation 실험

| Style | 설명 | Config |
|-------|------|--------|
| **Rendering** | "3D model rendering" | mouse_mvdiffusion_M5t_mouse_prompt.yaml |
| Real | "laboratory mouse photographed" | mouse_mvdiffusion_M5t_real_prompt.yaml |
| Hybrid | "a mouse, multi-view capture" | mouse_mvdiffusion_M5t_hybrid_prompt.yaml |

**문헌 근거**: MVDiffusion은 text prompt를 view control에 직접 사용하지 않음.
View control은 Correspondence-Aware Attention (depth 기반).

---

## Manual Segmentation GUI

### 실행

```bash
# 외부 접속 (Public URL 생성)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.segment_mouse_web \
    --sam_checkpoint checkpoints/sam/sam_vit_b.pth \
    --input_dir /path/to/images \
    --output_dir /path/to/output \
    --port 7860 --share
```

### 접속 방법

| 방법 | URL | 조건 |
|------|-----|------|
| Gradio Share | `https://xxxxx.gradio.live` | `--share` 필요 |
| SSH 터널 | `localhost:7860` | `ssh -L 7860:localhost:7860 gpu03` |

### SAM Checkpoint

```bash
# SAM ViT-B (358MB, 권장)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
    -O checkpoints/sam/sam_vit_b.pth
```

---

## Pose-Splatter 비교

### 데이터셋 비교

| 항목 | FaceLift M5t2 | Pose-Splatter fj5_ds2 |
|------|--------------|----------------------|
| 해상도 | 512×512 | 512×576 |
| Split | temporal 80:10:10 | temporal 80:10:10 |
| Views | 6 (all) | 5 (holdout 1) |

### 실행

```bash
ssh joon
cd /home/joon/dev/pose-splatter
conda activate splatter
python mouse_extensions/scripts/run_comparison.py
```

---

## Troubleshooting

### GPU 4로 강제 지정되는 문제
- 원인: `facelift` conda env의 `env_vars.sh`
- 해결: `export CUDA_VISIBLE_DEVICES=N &&` 형식 사용

### Slow motion 잔상
- 원인: 프레임 블렌딩
- 해결: 260202 수정됨 (`interpolate=False`)

### Resume 시 새 WandB run 생성
- 원인: `wandb_run_id.txt` 없음
- 해결: 수동 생성 또는 `_resume.yaml` config 사용

### pkill 패턴 매칭 실패
- 원인: 인자가 분리됨 (`-d M5t2 -e E0_1`)
- 해결: 정규식 사용 `pkill -f "M5t2.*E0_1"`

---

*Reference Details | 2026-02-02*
*명령어: [MOUSE_QUICK_REFERENCE.md](MOUSE_QUICK_REFERENCE.md)*
