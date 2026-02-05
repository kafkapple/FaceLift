# 실험 명령어 (SSOT)

> **최종 업데이트**: 260205
> **원칙**: 이 문서가 명령어의 단일 소스

---

## 1. 기본 패턴

### GS-LRM
```bash
cd /home/joon/dev/FaceLift
export CUDA_VISIBLE_DEVICES=N && nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d {DATASET} -e {EXPERIMENT} \
    > logs/{name}.log 2>&1 &
```

### MVDiffusion
```bash
export CUDA_VISIBLE_DEVICES=N && nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py --config configs/mvdiffusion/{CONFIG}.yaml \
    > logs/{name}.log 2>&1 &
```

### E2E Inference
```bash
python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --model {MODEL} --num_frames 50 --prefer_ema
```

### Test Evaluation
```bash
python -m mouse_extensions.scripts.evaluate_test \
    --checkpoint {CKPT_PATH} --config {CONFIG} --output_dir {OUTPUT}
```

---

## 2. 데이터셋 옵션 (-d)

| 옵션 | Split | 샘플 | 용도 |
|------|-------|------|------|
| **M5t2** | Temporal 8:1:1 | 2880 | ⭐ 권장 |
| M5t | Temporal 1:1:1 | 1198 | Pose-Splatter 비교 |
| M5 | Random | 2880 | Baseline |

---

## 3. 현재 실험 (260205)

### View Ablation (GS-LRM)

| Views | GPU | 상태 | 명령어 |
|-------|-----|------|--------|
| 2 | 6 | 🔄 | `-d M5t2 -e view_ablation/E0_1_2view` |
| 3 | 5 | 🔄 | `-d M5t2 -e view_ablation/E0_1_3view` |
| 6 | 7 | 🔄 | `-d M5t2 -e view_ablation/E0_1_6view` |

**Inference 결과** (260205):
| Views | PSNR |
|-------|------|
| **3** | **21.12** ⭐ |
| 2 | 20.20 |
| 4-6 | 19.58 |

### Cyclic MVDiffusion

| Config | GPU | 상태 |
|--------|-----|------|
| mouse_mvdiffusion_M5t2_cyclic | 4 | 🔄 10% |

---

## 4. 실험 카테고리 (-e)

| 카테고리 | mask_mode | 예시 |
|----------|-----------|------|
| **E0** | none | E0_1_facelift (Baseline) |
| **E1** | gt | E1_2_alpha (GT Mask) |
| **view_ablation/** | none | E0_1_Nview |

---

## 5. Config 위치

```
configs/
├── datasets/           # -d 옵션
│   └── M5t2.yaml
├── experiments/        # -e 옵션
│   ├── E0_1_facelift.yaml
│   └── view_ablation/
│       └── E0_1_3view.yaml
└── mvdiffusion/        # --config
    └── mouse_mvdiffusion_M5t2_cyclic.yaml
```

---

## 6. 모니터링

```bash
# 로그
tail -f logs/view_ablation_3view.log

# GPU
watch -n 5 nvidia-smi

# WandB
# https://wandb.ai/joon/FaceLift-Mouse
```

---

*Commands v1.0 | 260205*
