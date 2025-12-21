---
date: 2025-12-13
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, mvdiffusion, gslrm, training-strategy, domain-alignment]
project: mouse-facelift
status: in-progress
generator: ai-assisted
generator_tool: claude-code
---

# Mouse-FaceLift 2단계 학습 전략 수립

> MVDiffusion → GS-LRM 도메인 정렬을 위한 순차적 학습 파이프라인

---

## 1. 문제 진단 요약

### 1.1 이전 실험 결과 분석

| 테스트 | GS-LRM 모델 | 카메라 | render_view 크기 | 결과 |
|--------|-------------|--------|------------------|------|
| test_gslrm_pretrained | pretrained (human) | FaceLift | 51-89 KB (균일) | ✅ **성공** |
| test_facelift_camera | mouse_finetune | FaceLift | 13-101 KB (불균일) | ❌ 실패 |
| test_real_mouse | mouse_finetune | mouse | 5-120 KB (불균일) | ❌ 실패 |

### 1.2 핵심 발견

```
┌─────────────────────────────────────────────────────────────────┐
│  문제의 핵심: 카메라 설정 불일치 체인                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MVDiffusion 학습:                                               │
│    └─ FaceLift prompt_embeds 사용 (수평 뷰 기준)                  │
│                                                                 │
│  MVDiffusion 출력:                                               │
│    └─ 수평 6뷰 생성 (elevation ≈ 0°)                              │
│                                                                 │
│  GS-LRM mouse_finetune:                                          │
│    └─ Mouse 카메라로 학습됨 (elevation ≈ 20°, 경사)                │
│                                                                 │
│  결과:                                                           │
│    └─ MVDiffusion 출력 ≠ GS-LRM 기대 입력 → 3D 복원 실패           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 중요 개념 정리

| 구성요소 | 역할 | prompt_embeds 사용 |
|----------|------|-------------------|
| **MVDiffusion** | 1뷰 → 6뷰 생성 | ✅ 출력 뷰 방향 결정 |
| **GS-LRM** | 6뷰 → 3D | ❌ 카메라 파라미터 직접 사용 |

---

## 2. 해결 전략: 2단계 순차 학습

### 2.1 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: MVDiffusion Fine-tuning                               │
│  ─────────────────────────────────────────────                  │
│  입력: 실제 마우스 2000 샘플 × 6뷰 = 12,000 (6배 증강)            │
│  설정:                                                          │
│    - reference_view_idx: "random" (0~5)                         │
│    - prompt_embeds: mouse_prompt_embeds (경사 뷰)                │
│  학습: 임의 뷰 → 전체 6뷰 생성                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: 합성 데이터 생성                                       │
│  ─────────────────────────────────────────────                  │
│  입력: 실제 마우스 1뷰씩 (2000 × 6 = 12,000)                     │
│  출력: MVDiffusion 생성 6뷰 (12,000 샘플)                        │
│  카메라: mouse_prompt_embeds 기준 (경사 뷰)                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: GS-LRM Fine-tuning                                    │
│  ─────────────────────────────────────────────                  │
│  데이터: Phase 2 합성 6뷰 12,000 샘플                            │
│  시작: human pretrained (도메인 일치)                            │
│  결과: MVDiffusion 출력에 최적화된 GS-LRM                        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 기존 학습 중단 이유

현재 진행 중인 `mouse_gslrm_v2_stable` 학습:
- **문제**: 실제 마우스 데이터로 학습 → MVDiffusion 출력과 도메인 불일치
- **결론**: 중단 권장, 새 전략으로 재학습

---

## 3. 구현 상세

### 3.1 Phase 1: MVDiffusion 학습 설정

**Config**: `configs/mouse_mvdiffusion_6x_aug.yaml`

```yaml
# 핵심 변경사항
reference_view_idx: "random"  # 6배 데이터 증강!
prompt_embed_path: "mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt"
output_dir: 'checkpoints/mvdiffusion/mouse/mouse_embeds_6x_aug'
max_train_steps: 20000
```

**Dataset 수정**: `mvdiffusion/data/mouse_dataset.py`

```python
# reference_view_idx 지원 확장
ref_view_config = config.get("reference_view_idx", 0)
if ref_view_config == "random":
    self.reference_view_idx = "random"
    self.reference_view_choices = list(range(self.n_views))
```

### 3.2 Phase 2: 합성 데이터 생성

**Script**: `scripts/generate_gslrm_training_data.py`

```bash
python scripts/generate_gslrm_training_data.py \
    --mvdiff_checkpoint checkpoints/mvdiffusion/mouse/mouse_embeds_6x_aug/checkpoint-XXXX \
    --input_data data_mouse/data_mouse_train.txt \
    --output_dir data_mouse_synthetic \
    --prompt_embeds mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt
```

**출력 구조**:
```
data_mouse_synthetic/
├── data_train.txt              # GS-LRM 학습용
├── data_val.txt                # GS-LRM 검증용
└── sample_XXXXXX/
    ├── opencv_cameras.json     # Mouse 카메라 파라미터
    ├── metadata.json           # 생성 정보
    └── images/
        ├── cam_000.png         # MVDiffusion 생성 이미지
        ├── cam_001.png
        └── ...
```

### 3.3 Phase 3: GS-LRM 학습 설정

**Config**: `configs/mouse_gslrm_synthetic.yaml`

```yaml
training:
  dataset:
    dataset_path: "data_mouse_synthetic/data_train.txt"

  checkpointing:
    checkpoint_dir: "checkpoints/gslrm/mouse_synthetic"
    # Human pretrained에서 시작 (mouse_finetune 아님!)
    resume_ckpt: "checkpoints/gslrm/ckpt_0000000000021125.pt"
```

---

## 4. 실행 명령어

### 4.1 현재 학습 중단

```bash
# 현재 GS-LRM 학습 프로세스 확인
ps aux | grep train_gslrm

# 프로세스 종료
kill <PID>
```

### 4.2 Phase 1: MVDiffusion 학습

```bash
# GPU 05 서버에서 실행
ssh gpu05
cd /home/joon/FaceLift

# conda 환경 활성화
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mouse_facelift

# 학습 시작
CUDA_VISIBLE_DEVICES=0 accelerate launch train_diffusion.py \
    --config configs/mouse_mvdiffusion_6x_aug.yaml

# 또는 백그라운드 실행
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 accelerate launch train_diffusion.py \
    --config configs/mouse_mvdiffusion_6x_aug.yaml' \
    > logs/train_mvdiff_6x.log 2>&1 &
```

**모니터링**:
- WandB: https://wandb.ai → project: `mouse_facelift` → group: `mvdiffusion`
- 로그: `tail -f logs/train_mvdiff_6x.log`

### 4.3 Phase 2: 합성 데이터 생성

```bash
# MVDiffusion 학습 완료 후 실행
python scripts/generate_gslrm_training_data.py \
    --mvdiff_checkpoint checkpoints/mvdiffusion/mouse/mouse_embeds_6x_aug/checkpoint-20000 \
    --input_data data_mouse/data_mouse_train.txt \
    --output_dir data_mouse_synthetic \
    --prompt_embeds mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt \
    --camera_json data_mouse/sample_000000/opencv_cameras.json \
    --augment_all_views
```

**예상 시간**: ~2-4시간 (12,000 샘플 생성)

### 4.4 Phase 3: GS-LRM 학습

```bash
# 합성 데이터 생성 완료 후 실행
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse_gslrm_synthetic.yaml

# 또는 백그라운드 실행
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse_gslrm_synthetic.yaml' \
    > logs/train_gslrm_synthetic.log 2>&1 &
```

---

## 5. 품질 평가 방법

### 5.1 Phase 1 평가 (MVDiffusion)

```python
# Validation metrics (자동 기록)
metrics = {
    "PSNR": ">25 목표",
    "SSIM": ">0.9 목표",
    "LPIPS": "<0.15 목표",
}
```

- WandB에서 `validation/psnr`, `validation/ssim` 확인
- Validation 이미지: `checkpoints/mvdiffusion/mouse/mouse_embeds_6x_aug/val/`

### 5.2 Phase 3 평가 (GS-LRM)

```python
# Validation metrics (자동 기록)
metrics = {
    "l2_loss": "reconstruction quality",
    "lpips_loss": "perceptual quality",
    "ssim_loss": "structural similarity",
}
```

- WandB에서 loss curves 확인
- 시각화: `experiments/validation/mouse_gslrm_synthetic/`

### 5.3 최종 파이프라인 테스트

```bash
# 학습 완료 후 full pipeline 테스트
python test_full_pipeline.py \
    --input_image data_mouse/sample_000000/images/cam_000.png \
    --mvdiff_unet checkpoints/mvdiffusion/mouse/mouse_embeds_6x_aug/checkpoint-20000/unet \
    --prompt_embeds mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt \
    --gslrm_checkpoint checkpoints/gslrm/mouse_synthetic \
    --output_dir outputs/pipeline_synthetic
```

---

## 6. 체크리스트

### Phase 1 시작 전
- [ ] 현재 GS-LRM 학습 중단
- [ ] `mouse_prompt_embeds_6view/clr_embeds.pt` 존재 확인
- [ ] GPU 메모리 확인 (24GB 권장)

### Phase 2 시작 전
- [ ] MVDiffusion 학습 완료 확인 (WandB)
- [ ] 체크포인트 존재 확인
- [ ] 디스크 공간 확인 (~50GB for 12k samples)

### Phase 3 시작 전
- [ ] 합성 데이터 생성 완료
- [ ] `data_mouse_synthetic/data_train.txt` 존재 확인
- [ ] 카메라 파라미터 일관성 확인

---

## 7. 예상 타임라인

| Phase | 작업 | 예상 시간 |
|-------|------|----------|
| 1 | MVDiffusion 학습 (20k steps) | ~12-24h |
| 2 | 합성 데이터 생성 (12k samples) | ~2-4h |
| 3 | GS-LRM 학습 (30k steps) | ~12-24h |
| **총** | | **~30-50h** |

---

## 8. 관련 파일

| 파일 | 용도 |
|------|------|
| `configs/mouse_mvdiffusion_6x_aug.yaml` | Phase 1 config |
| `scripts/generate_gslrm_training_data.py` | Phase 2 script |
| `configs/mouse_gslrm_synthetic.yaml` | Phase 3 config |
| `mvdiffusion/data/mouse_dataset.py` | 수정된 dataset (random ref) |
| `mvdiffusion/data/mouse_prompt_embeds_6view/` | Mouse prompt embeds |

---

## 9. 다음 단계

1. [ ] Phase 1 시작 (MVDiffusion 학습)
2. [ ] 학습 모니터링 (WandB)
3. [ ] Phase 2 실행 (합성 데이터 생성)
4. [ ] Phase 3 시작 (GS-LRM 학습)
5. [ ] 최종 파이프라인 테스트
