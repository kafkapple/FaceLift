# View Ablation 실험 명령어

> **최종 업데이트**: 260205
> **Config 위치**: `configs/experiments/view_ablation/`

---

## Config 목록

| Config | 뷰 수 | 선택 방식 | 설명 |
|--------|-------|----------|------|
| `E0_1_1view` | 1 | random | Sanity check (불가능) |
| `E0_1_2view` | 2 | random | Minimal |
| `E0_1_3view` | 3 | random | Triangle |
| `E0_1_5view` | 5 | random | Near-full |
| `E0_1_6view` | 6 | all | Baseline |

**공통 설정**: mask_mode=none, alpha=0.0 (FaceLift 논문 기반)

---

## 학습 명령어

```bash
cd /home/joon/dev/FaceLift

# 1-view (Sanity Check) - 예상 PSNR 10-15
export CUDA_VISIBLE_DEVICES=7 && nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e view_ablation/E0_1_1view \
    > logs/view_ablation_1view_training.log 2>&1 &

# 2-view
export CUDA_VISIBLE_DEVICES=6 && nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e view_ablation/E0_1_2view \
    > logs/view_ablation_2view_training.log 2>&1 &

# 3-view
export CUDA_VISIBLE_DEVICES=5 && nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e view_ablation/E0_1_3view \
    > logs/view_ablation_3view_training.log 2>&1 &

# 5-view
export CUDA_VISIBLE_DEVICES=X && nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e view_ablation/E0_1_5view \
    > logs/view_ablation_5view_training.log 2>&1 &

# 6-view (Baseline)
export CUDA_VISIBLE_DEVICES=X && nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e view_ablation/E0_1_6view \
    > logs/view_ablation_6view_training.log 2>&1 &
```

---

## 현재 상태 (260205)

| Views | GPU | Config | 상태 |
|-------|-----|--------|------|
| 2 | 6 | `E0_1_2view` | 🔄 진행 |
| 3 | 5 | `E0_1_3view` | 🔄 진행 |

---

## Inference-time 결과 (Reference)

| Views | PSNR | 비고 |
|-------|------|------|
| **3** | **21.12** | ⭐ 최고 |
| 2 | 20.20 | |
| 4 | 19.58 | Baseline |

---

## 구 Config (Deprecated)

다음 파일들은 OLD 명명으로, 사용하지 않음:
- `M5t2_view_ablation_2v.yaml` → `view_ablation/E0_1_2view.yaml`
- `M5t2_view_ablation_3v.yaml` → `view_ablation/E0_1_3view.yaml`
- `M5t2_view_ablation_6v.yaml` → `view_ablation/E0_1_6view.yaml`

---

*View Ablation Commands v1.0 | 260205*
