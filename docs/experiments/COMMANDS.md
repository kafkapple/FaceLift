# 실험 명령어 (SSOT)

> **최종 업데이트**: 260213
> **원칙**: 이 문서가 명령어의 단일 소스

---

## 1. 기본 패턴

### GS-LRM (Uniform v2)

**표준 명령어:**
```bash
cd /home/joon/dev/FaceLift
export CUDA_VISIBLE_DEVICES=N && nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/{EXPERIMENT}.yaml \
    > logs/uniform_{name}_v2.log 2>&1 &
```

> ⚠️ Dead Config 주의: `max_steps`와 `training.schedule.val_every`는 코드에서 무시됨.
> 실제 종료: `max_fwdbwd_passes` (+1 epoch 올림). 실제 validation: `validation.val_every`.

### MV-Diffusion

```bash
export CUDA_VISIBLE_DEVICES=N && PYTHONUNBUFFERED=1 nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py --config configs/mvdiffusion/{CONFIG}.yaml \
    > logs/{name}.log 2>&1 &
```

---

## 2. H4: View Ablation v2 (현재 실행 중)

> **Baseline**: `4view_v2` = E0_1_facelift 동일 조건 (4-view, random=true, no mask)
> **Base config**: `base_uniform_v2.yaml` (15K fwdbwd_passes → 15840 actual, seed=42, M5t2)
> **목표**: 동일 조건에서 input view 수에 따른 성능 비교

### Round 1 (실행 중, 260207~)

| GPU | Config | Views | 상태 |
|-----|--------|-------|------|
| 4 | baseline_v2 (zero-shot) | 6 | ⏳ 재실행 준비 |
| 5 | 4view_v2 **⭐ Baseline** | 4 | 🔄 |
| 6 | 1view_v2, 2view_v2 | 1, 2 | 🔄 (GPU 공유) |
| 7 | 3view_v2, 5view_v2 | 3, 5 | 🔄 (GPU 공유) |

> 📌 MV-Diffusion M5t2_cyclic은 H5 실험 (이 표와 별개, GPU 4에서 실행 중)

> ⚠️ **baseline_v2 수정 이력**: (1) `max_steps: 1` = dead config, (2) `max_fwdbwd_passes: 1` = +1 올림으로 1440 steps,
> (3) `training.schedule.val_every: 1` = dead config path.
> **최종 해결**: `early_stop_after_epochs: 1` + `validation.val_every: 1`

### Round 2 (대기)

| Config | Views | 가설 | 명령어 |
|--------|-------|------|--------|
| 6view_v2 | 6 | 최대 뷰 성능 | `-e 6view_v2` |
| 4view_fixed_v2 | 4 | Random vs Fixed view selection | `-e 4view_fixed_v2` |

---


## 2.5 H5: MV-Diffusion Optimization (다음 실행)

> **목표**: Sparse vs Full attention E2E 비교 → 단일 변수 ablation
> **문서**: [H5_MVDIFFUSION.md](../hypotheses/H5_MVDIFFUSION.md)

### Phase 1: E2E 비교 (학습 불필요, GPU 4)

| 순서 | 실험 | 예상 시간 |
|------|------|----------|
| 1 | cfgr (full attn) E2E | ~30분 |
| 2 | baseline (sparse) E2E | ~30분 |

```bash
cd /home/joon/dev/FaceLift

# 1) cfgr E2E (full attention, ckpt-10000)
export CUDA_VISIBLE_DEVICES=6 && python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --mvdiffusion_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_cfgr/checkpoint-10000 \
    --gslrm_checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt \
    --output_dir outputs/h5_e2e/cfgr_ckpt10000 \
    --input_view_idx 0 \
    --guidance_scale 3.0

# 2) baseline E2E (sparse attention, ckpt-5000)
export CUDA_VISIBLE_DEVICES=4 && python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --mvdiffusion_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2/checkpoint-5000 \
    --gslrm_checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt \
    --output_dir outputs/h5_e2e/baseline_ckpt5000 \
    --input_view_idx 0 \
    --guidance_scale 3.0
```

### Phase 1 평가

```bash
# cfgr vs baseline 비교 (추론 완료 후)
python -m mouse_extensions.scripts.eval.compute_e2e_metrics \
    --output_dir outputs/h5_e2e/cfgr_ckpt10000 outputs/h5_e2e/baseline_ckpt5000 \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --skip_input_view 0
```

### Phase 2B: Generalization 실험 (우선순위 순)

> **로드맵**: [GENERALIZATION_ROADMAP](../hypotheses/GENERALIZATION_ROADMAP.md)
> **카메라 분석**: [RMA_CAMERA_ANALYSIS](../hypotheses/RMA_CAMERA_ANALYSIS.md)

| 순위 | 실험 | Config | 변경점 | 상태 |
|:---:|------|--------|--------|:----:|
| **P0** | randref_sparse | `M5t2_randref_sparse` | ref=random (단일 변수) | ⏳ |
| **P1** | pose_spherical | `M5t2_pose_spherical` | spherical pose + random ref | ⏳ |
| - | 20k_sparse | `M5t2_20k_sparse` | 학습 시간 연장 (baseline) | ⏳ |

```bash
cd /home/joon/dev/FaceLift

# P0: randref_sparse — Random reference view (single-variable ablation)
# Change: reference_view_idx: 0 → "random", sparse_mv_attention: true (SAME)
# Analysis: docs/hypotheses/RMA_CAMERA_ANALYSIS.md
export CUDA_VISIBLE_DEVICES=4 && PYTHONUNBUFFERED=1 nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py --config configs/mvdiffusion/mouse_mvdiffusion_M5t2_randref_sparse.yaml \
    > logs/mvdiff_M5t2_randref_sparse.log 2>&1 &

# P1: pose_spherical — Spherical pose conditioning (non-invasive UNet injection)
# Change: pose_conditioning.enabled=true, method=spherical, random ref
# Resumes from M5t2 baseline checkpoint
# Prereq: mouse_extensions/model/pose_conditioning_integration.py
# Theory: docs/hypotheses/GENERALIZATION_ROADMAP.md §4
export CUDA_VISIBLE_DEVICES=7 && PYTHONUNBUFFERED=1 nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py --config configs/mvdiffusion/mouse_mvdiffusion_M5t2_pose_spherical.yaml \
    > logs/mvdiff_M5t2_pose_spherical.log 2>&1 &

# (Optional) 20k_sparse — Longer training baseline
export CUDA_VISIBLE_DEVICES=6 && PYTHONUNBUFFERED=1 nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py --config configs/mvdiffusion/mouse_mvdiffusion_M5t2_20k_sparse.yaml \
    > logs/mvdiff_M5t2_20k_sparse.log 2>&1 &
```

### Phase 2B 모니터링

| 실험 | WandB | 비교 기준 | 핵심 지표 |
|------|-------|-----------|-----------|
| randref_sparse | `mvdiff_M5t2_randref_sparse` | Baseline 27.30 | Overall PSNR, per-view PSNR |
| pose_spherical | `mvdiff_M5t2_pose_spherical` | Baseline 27.30 | Overall PSNR, loss 안정성 |
| 20k_sparse | `mvdiff_M5t2_20k_sparse` | Baseline @5K | 10K vs 20K 수렴 여부 |

## 3. H6: Alpha/Mask Hypothesis

> **기준**: 4view_v2 (Baseline, alpha_w=0, mask=none)
> **목표**: Alpha supervision의 shape quality (mask_iou) 개선 효과 측정
> **문서**: [H6_ALPHA_MASK.md](../hypotheses/H6_ALPHA_MASK.md)

### v2 → v3 Bugfix (260213)

> **Bug**: `mask_mode: none`이면 `compute_mask_from_config()`가 GT mask를 `None`으로 덮어씀
> → `alpha_loss`와 `mask_iou`가 항상 0.0 (alpha supervision 무효)
> **Fix**: `gslrm.py`에서 `original_gt_mask`를 보존, alpha/bg/mask_iou에 독립 사용
> **v2 실험 결과**: 무효 (삭제됨), v3로 재실험

| Config | alpha_w | mask_mode | 비고 | 상태 |
|--------|:-------:|:---------:|------|:----:|
| 4view_v2 (baseline) | 0.0 | none | 비교 기준 | **완료** |
| ~~4view_alpha01_v2~~ | ~~0.1~~ | ~~none~~ | ~~Bug: alpha_loss=0~~ | **삭제** |
| ~~4view_alpha05_v2~~ | ~~0.5~~ | ~~none~~ | ~~Bug: alpha_loss=0~~ | **삭제** |
| 4view_alpha05_v3 | **0.5** | none | v3 bugfix | **실행중** (GPU 5) |
| 4view_alpha10_v3 | **1.0** | none | v3 bugfix (최대) | **실행중** (GPU 6) |

**v3 명령어:**
```bash
cd /home/joon/dev/FaceLift

# Alpha 0.5 (v3, GPU 5)
export CUDA_VISIBLE_DEVICES=5 && PYTHONUNBUFFERED=1 nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/4view_alpha05_v3.yaml \
    > logs/h6_alpha05_v3.log 2>&1 &

# Alpha 1.0 (v3, GPU 6)
export CUDA_VISIBLE_DEVICES=6 && PYTHONUNBUFFERED=1 nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/4view_alpha10_v3.yaml \
    > logs/h6_alpha10_v3.log 2>&1 &
```

---

## 4. H7: SSIM Weight Hypothesis (대기)

> **기준**: 4view_v2 (Baseline, ssim_weight=0.1)
> **목표**: SSIM loss weight 증가 시 구조적 보존 개선 여부
> **문서**: [H7_SSIM_WEIGHT.md](../hypotheses/H7_SSIM_WEIGHT.md)
> **관찰**: train/ssim_loss가 초기 감소 후 증가 → SSIM이 다른 loss에 밀림

| Config | ssim_weight | L2:SSIM 비율 | 가설 |
|--------|-------------|-------------|------|
| 4view_v2 (baseline) | 0.1 | 10:1 | 기준 (GS-LRM 기본) |
| 4view_ssim03_v2 | **0.3** | 3.3:1 | 적당한 구조 강조 (3DGS 수준) |
| 4view_ssim05_v2 | **0.5** | 2:1 | 균형점 (Instant-3D 수준) |
| 4view_ssim10_v2 | **1.0** | 1:1 | 극단적 구조 강조 |

**명령어:**
```bash
# SSIM 0.3
export CUDA_VISIBLE_DEVICES=4 && PYTHONUNBUFFERED=1 nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/4view_ssim03_v2.yaml \
    > logs/uniform_4view_ssim03_v2.log 2>&1 &

# SSIM 0.5
export CUDA_VISIBLE_DEVICES=5 && PYTHONUNBUFFERED=1 nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/4view_ssim05_v2.yaml \
    > logs/uniform_4view_ssim05_v2.log 2>&1 &

# SSIM 1.0
export CUDA_VISIBLE_DEVICES=6 && PYTHONUNBUFFERED=1 nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/4view_ssim10_v2.yaml \
    > logs/uniform_4view_ssim10_v2.log 2>&1 &
```

---

## 5. H8: Reduced View Generation

> **전제**: ✅ H4 R1 완료 — 3view(19.92) vs 4view(21.50) = 1.58dB gap (<2dB 기준 충족)
> **목표**: MV-Diffusion 6뷰 → 3~4뷰 생성으로 per-view 품질 개선
> **문서**: [H8_REDUCED_VIEW_GENERATION.md](../hypotheses/H8_REDUCED_VIEW_GENERATION.md)

### S0: H4 결과 분석 ✅

| Views | GS-LRM PSNR (GT input) | Gap vs 6-view |
|-------|------------------------|---------------|
| 6-view | 23.46 | - |
| 5-view | 22.63 | -0.83 |
| 4-view | 21.50 | -1.96 |
| **3-view** | **19.92** | **-3.54** |
| 2-view | 17.70 | -5.76 |
| 1-view | 11.08 | -12.38 |

**결론**: GS-LRM에서 뷰 수↑ = PSNR↑ (단조 증가). 하지만 MV-Diffusion 생성 품질이
핵심 변수 — per-view 품질 개선이 coverage 손실을 상쇄할 수 있는지 E2E로 검증 필요.

### S2-S3: MV-Diffusion n_views=3/4 fine-tune

**코드 변경** (260209):
- `mvdiffusion/data/mouse_dataset.py`: `camera_indices` 매핑 추가
- `train_diffusion.py`: `TrainingConfig.camera_indices` 필드 추가
- 기존 6-view 학습과 backward compatible (camera_indices=None → [0..5])

**뷰 선택**:
- 4-view: `[0, 2, 3, 5]` — 90° spread (top-front, top-right, top-back, top-front-left)
- 3-view: `[0, 2, 4]` — 120° spread (top-front, top-right, top-left)

**명령어:**
```bash
cd /home/joon/dev/FaceLift

# H8: 4-view MV-Diffusion (GPU 4)
export CUDA_VISIBLE_DEVICES=4 && PYTHONUNBUFFERED=1 nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py --config configs/mvdiffusion/mouse_mvdiffusion_M5t2_4view.yaml \
    > logs/mvdiff_M5t2_4view.log 2>&1 &

# H8: 3-view MV-Diffusion (GPU 5)
export CUDA_VISIBLE_DEVICES=5 && PYTHONUNBUFFERED=1 nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py --config configs/mvdiffusion/mouse_mvdiffusion_M5t2_3view.yaml \
    > logs/mvdiff_M5t2_3view.log 2>&1 &
```

### S4: E2E 평가 (S2-S3 완료 후)

```bash
# 4-view E2E inference
export CUDA_VISIBLE_DEVICES=4 && python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse_M5t2_4view/checkpoint-BEST \
    --gslrm_checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt \
    --output_dir outputs/h8_e2e/4view \
    --num_input_views 4 \
    --guidance_scale 3.0

# 3-view E2E inference
export CUDA_VISIBLE_DEVICES=4 && python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse_M5t2_3view/checkpoint-BEST \
    --gslrm_checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt \
    --output_dir outputs/h8_e2e/3view \
    --num_input_views 3 \
    --guidance_scale 3.0
```

### Config 위치

```
configs/mvdiffusion/
├── mouse_mvdiffusion_M5t2.yaml          # 6-view baseline
├── mouse_mvdiffusion_M5t2_4view.yaml    # H8: 4-view [0,2,3,5]
└── mouse_mvdiffusion_M5t2_3view.yaml    # H8: 3-view [0,2,4]

mvdiffusion/data/
├── mouse_prompt_embeds_6view_1024/      # 6-view embeddings
├── mouse_prompt_embeds_4view_1024/      # H8: 4-view (sliced)
└── mouse_prompt_embeds_3view_1024/      # H8: 3-view (sliced)
```

---

## 6. 주요 비교 메트릭

| 메트릭 | H4 (View) | H6 (Alpha) | H7 (SSIM) |
|--------|-----------|------------|-----------|
| val/psnr | ⭐ Primary | Secondary | Secondary |
| val/ssim | Secondary | Secondary | ⭐ Primary |
| val/mask_iou | Secondary | ⭐ Primary | Secondary |
| train/ssim_loss | Monitor | Monitor | ⭐ Monitor |

---

## 7. 실험 우선순위

```
H4 Round 1 (현재 실행 중)
    ├── 완료 후 → H4 Round 2 (6view, fixed)
    └── GPU 확보 시 → H6 + H7 동시 진행 (독립적)
```

| Round | 실험 | GPU 필요 | 예상 시간 |
|-------|------|---------|----------|
| H4 R1 | baseline~5view | 4개 | ~9h |
| H4 R2 | 6view, fixed | 2개 | ~9h |
| H6 | alpha01, alpha05, maskgt | 3개 | ~9h |
| H7 | ssim03, ssim05, ssim10 | 3개 | ~9h |

---

## 8. 데이터셋 옵션 (-d)

| 옵션 | Split | 샘플 | 용도 |
|------|-------|------|------|
| **M5t2** | Temporal 8:1:1 | 2880 | ⭐ 권장 |
| M5t | Temporal 1:1:1 | 1198 | Pose-Splatter 비교 |
| M5 | Random | 2880 | Baseline |

---

## 9. Config 위치

```
configs/mouse/uniform/          # ⭐ 현재 실험 체계
├── base_uniform_v2.yaml        # 공통 base (15K fwdbwd_passes → 15840 actual, seed=42, M5t2)
│
├── baseline_v2.yaml            # H4: Zero-shot (max_fwdbwd_passes=1)
├── 1view_v2.yaml ~ 6view_v2.yaml   # H4: View ablation
├── 4view_fixed_v2.yaml         # H4: Fixed view selection
│
├── 4view_alpha01_v2.yaml       # H6: Alpha supervision 0.1
├── 4view_alpha05_v2.yaml       # H6: Alpha supervision 0.5
├── 4view_maskgt_v2.yaml        # H6: GT mask + alpha 0.1
│
├── 4view_ssim03_v2.yaml        # H7: SSIM weight 0.3
├── 4view_ssim05_v2.yaml        # H7: SSIM weight 0.5
└── 4view_ssim10_v2.yaml        # H7: SSIM weight 1.0
```

---

## 10. 모니터링

```bash
# 로그 실시간
tail -f logs/uniform_{name}_v2.log

# GPU
gpujobs    # GPU별 프로세스 확인
watch -n 5 nvidia-smi

# WandB: https://wandb.ai/joon/FaceLift-Mouse (Group: view_ablation_v2)
```

---

## 11. 환경변수

| 변수 | 용도 |
|------|------|
| `CUDA_VISIBLE_DEVICES=N` | GPU 지정 (4-7: A6000) |
| `PYTHONUNBUFFERED=1` | 로그 실시간 출력 |

---

---

## 12. Turntable Visualization

> *Source: TURNTABLE_VIS_GUIDE.md (merged 2026-02-11)*

### Quick Test (verify_turntable.sh)

```bash
cd /home/joon/dev/FaceLift
nohup bash scripts/verify_turntable.sh 6 > ./logs/verify_turntable.log 2>&1 &
tail -f ./logs/verify_turntable.log

# Custom checkpoint
bash scripts/verify_turntable.sh 6 /path/to/checkpoint.pt
```

| Step | Script | Verifies |
|------|--------|----------|
| 1/2 | render_from_checkpoint.py | Inference path (orbit only) |
| 2/2 | TurntableRenderer.render_all() | Train/Val path (orbit + view_traj + grid) |

### Standalone Inference

```bash
CUDA_VISIBLE_DEVICES=6 python mouse_extensions/scripts/inference/render_from_checkpoint.py \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/ckpt_0000000000009200.pt \
    --config configs/base/gslrm_mouse.yaml \
    --data_path /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_val.txt \
    --output_dir outputs/verify_turntable/inference \
    --mode turntable --num_samples 1
```

### Result Check

```bash
ls outputs/verify_turntable/inference/
ls outputs/verify_turntable/renderer/
scp -r gpu03:~/dev/FaceLift/outputs/verify_turntable/ .
```

> **See also**: [VISUALIZATION_SETTINGS.md](VISUALIZATION_SETTINGS.md) for turntable config, output file system, rotation direction.

---

*Commands v4.4 | 260213*
