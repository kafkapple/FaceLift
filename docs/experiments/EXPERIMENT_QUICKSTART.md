# Experiment Quick Start

> **최종 업데이트**: 260205
> **상세 문서**: [EXPERIMENT_REGISTRY](./EXPERIMENT_REGISTRY.md)

---

## 현재 우선순위

### P0: 진행 중

| 실험 | GPU | 명령어 | 상태 |
|------|-----|--------|------|
| View Ablation 3v | 5 | `-d M5t2 -e view_ablation/E0_1_3view` | 🔄 |
| View Ablation 2v | 6 | `-d M5t2 -e view_ablation/E0_1_2view` | 🔄 |
| Cyclic MVDiffusion | 4 | `mouse_mvdiffusion_M5t2_cyclic.yaml` | 🔄 10% |

### P1: 대기

| 실험 | 명령어 | 목적 |
|------|--------|------|
| View Ablation 1v | `-e view_ablation/E0_1_1view` | Sanity check |
| View Ablation 5v | `-e view_ablation/E0_1_5view` | Near-full |

---

## Quick Commands

### GS-LRM 학습

```bash
cd /home/joon/dev/FaceLift

# View Ablation
export CUDA_VISIBLE_DEVICES=X && nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e view_ablation/E0_1_Nview \
    > logs/view_ablation_Nview.log 2>&1 &

# 기본 학습
export CUDA_VISIBLE_DEVICES=X && nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e E0_1_facelift \
    > logs/gslrm_M5t2.log 2>&1 &
```

### MVDiffusion 학습

```bash
export CUDA_VISIBLE_DEVICES=X && nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py --config configs/mvdiffusion/CONFIG.yaml \
    > logs/mvdiff.log 2>&1 &
```

### 모니터링

```bash
# 로그 확인
tail -f logs/view_ablation_3view.log

# GPU 사용량
watch -n 5 nvidia-smi

# WandB
# https://wandb.ai/joon/FaceLift-Mouse
```

---

## 데이터셋 선택

| 옵션 | 설명 |
|------|------|
| `-d M5t2` | ⭐ **권장** (Temporal 8:1:1) |
| `-d M5t` | 비교용 (Temporal 1:1:1) |
| `-d M5` | Baseline (Random split) |

---

*Quick Start v3.0 | 260205*
