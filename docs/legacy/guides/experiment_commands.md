# Mouse-FaceLift 실험 명령어 가이드

## 체크포인트 구조

```bash
checkpoints/mvdiffusion/mouse/
├── original_facelift_embeds/    # 기존 학습 (FaceLift prompt_embeds)
│   ├── checkpoint-4000/unet
│   ├── checkpoint-5000/unet
│   └── checkpoint-6000/unet     ← 최신 (권장)
├── mouse_cam/                   # Option B: Mouse prompt_embeds 학습 (예정)
│   └── checkpoint-XXX/unet
└── facelift_cam/                # Option A: FaceLift 카메라 데이터 (예정)
    └── checkpoint-XXX/unet
```

**체크포인트 확인 명령**:
```bash
ls -la checkpoints/mvdiffusion/mouse/
ls -la checkpoints/mvdiffusion/mouse/original_facelift_embeds/
```

---

## 환경 설정

```bash
cd ~/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mouse_facelift
```

---

## Stage 1: MVDiffusion 테스트 (prompt_embeds 비교)

**목적**: 학습 없이 prompt_embeds 변경 효과 확인

```bash
# Test A: FaceLift prompt_embeds (수평 6방향)
python test_stage1_mvdiffusion.py \
    --input_image data_mouse/sample_000000/images/cam_000.png \
    --output_dir outputs/compare_embeds/facelift_embeds \
    --unet checkpoints/mvdiffusion/mouse/original_facelift_embeds/checkpoint-6000/unet \
    --prompt_embeds mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt

# Test B: Mouse prompt_embeds (경사 6방향)
python test_stage1_mvdiffusion.py \
    --input_image data_mouse/sample_000000/images/cam_000.png \
    --output_dir outputs/compare_embeds/mouse_embeds \
    --unet checkpoints/mvdiffusion/mouse/original_facelift_embeds/checkpoint-6000/unet \
    --prompt_embeds mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt
```

---

## Option B: MVDiffusion 재학습 (Mouse prompt_embeds)

```bash
cd ~/FaceLift

# 새 prompt_embeds로 처음부터 학습
python train_diffusion.py \
    --config configs/mouse_mvdiffusion_mouse_cam.yaml

# 또는 기존 체크포인트에서 이어서 학습 (transfer learning)
python train_diffusion.py \
    --config configs/mouse_mvdiffusion_mouse_cam.yaml \
    --resume_from_checkpoint checkpoints/mvdiffusion/mouse/original_facelift_embeds/checkpoint-6000
```

**체크포인트 저장 위치**: `checkpoints/mvdiffusion/mouse/mouse_cam/`

---

## Option A: MVDiffusion 학습 (FaceLift 카메라)

**선행 작업**: 데이터 전처리 필요 (아직 미구현)

```bash
# 데이터 전처리 (구현 필요)
python scripts/convert_mouse_to_facelift_camera.py \
    --input_dir data_mouse \
    --output_dir data_mouse/facelift_cam

# 학습
python train_diffusion.py \
    --config configs/mouse_mvdiffusion_facelift_cam.yaml
```

**체크포인트 저장 위치**: `checkpoints/mvdiffusion/mouse/facelift_cam/`

---

## 체크포인트 경로 가이드

### MVDiffusion UNet 경로
| 설정 | 경로 |
|------|------|
| FaceLift pretrained | (빈 문자열 → pretrained 사용) |
| 기존 학습 (FaceLift embeds) | `checkpoints/mvdiffusion/mouse/original_facelift_embeds/checkpoint-6000/unet` |
| Option B 학습 후 | `checkpoints/mvdiffusion/mouse/mouse_cam/checkpoint-XXX/unet` |
| Option A 학습 후 | `checkpoints/mvdiffusion/mouse/facelift_cam/checkpoint-XXX/unet` |

### prompt_embeds 경로
| 설정 | 경로 |
|------|------|
| FaceLift (수평 6방향) | `mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt` |
| Mouse (경사 6방향) | `mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt` |

---

## Stage 2: 전체 파이프라인 테스트 (MVDiff → GS-LRM)

```bash
# 조합 1: 기존 MVDiff + Mouse embeds + GS-LRM pretrained
python test_full_pipeline.py \
    --input_image data_mouse/sample_000000/images/cam_000.png \
    --mvdiff_unet checkpoints/mvdiffusion/mouse/original_facelift_embeds/checkpoint-6000/unet \
    --prompt_embeds mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt \
    --camera_json data_mouse/sample_000000/opencv_cameras.json \
    --gslrm_checkpoint checkpoints/gslrm/ckpt_0000000000021125.pt \
    --output_dir outputs/pipeline_mouse_embeds_pretrained

# 조합 2: 기존 MVDiff + Mouse embeds + GS-LRM mouse_finetune
python test_full_pipeline.py \
    --input_image data_mouse/sample_000000/images/cam_000.png \
    --mvdiff_unet checkpoints/mvdiffusion/mouse/original_facelift_embeds/checkpoint-6000/unet \
    --prompt_embeds mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt \
    --camera_json data_mouse/sample_000000/opencv_cameras.json \
    --gslrm_checkpoint checkpoints/gslrm/mouse_finetune \
    --output_dir outputs/pipeline_mouse_embeds_finetuned

# 조합 3: 기존 MVDiff + FaceLift embeds + GS-LRM pretrained (baseline)
python test_full_pipeline.py \
    --input_image data_mouse/sample_000000/images/cam_000.png \
    --mvdiff_unet checkpoints/mvdiffusion/mouse/original_facelift_embeds/checkpoint-6000/unet \
    --prompt_embeds mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt \
    --camera_json data_mouse/sample_000000/opencv_cameras.json \
    --gslrm_checkpoint checkpoints/gslrm/ckpt_0000000000021125.pt \
    --output_dir outputs/pipeline_facelift_embeds_pretrained
```

---

## 실험 비교 매트릭스

| 실험 | MVDiff UNet | prompt_embeds | GS-LRM | 예상 결과 |
|------|-------------|---------------|--------|----------|
| 조합 1 | original_facelift_embeds/ckpt-6000 | **Mouse** | pretrained | ⭐ 확인 필요 |
| 조합 2 | original_facelift_embeds/ckpt-6000 | **Mouse** | mouse_finetune | 확인 필요 |
| 조합 3 (baseline) | original_facelift_embeds/ckpt-6000 | FaceLift | pretrained | ✅ 작동 (인간 모델) |
| Option B 학습 후 | mouse_cam/ckpt-XXX | **Mouse** | mouse_finetune | 최적 예상 |
