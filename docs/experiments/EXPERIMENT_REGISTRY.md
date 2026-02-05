# Experiment Registry (실험 레지스트리)

> **SSOT**: 활성 실험 설정의 중앙 관리
> **최종 업데이트**: 260205

---

## 1. 명명 규칙

```
E{카테고리}_{번호}_{이름}
view_ablation/E0_1_{N}view

예시:
E0_1_facelift          - FaceLift 논문 기본
view_ablation/E0_1_3view - 3-view ablation
```

---

## 2. 카테고리

| 카테고리 | mask_mode | 설명 |
|----------|-----------|------|
| **E0** | none | FaceLift Baseline ⭐ |
| **E1** | gt | GT Mask 기반 |

---

## 3. 활성 데이터셋

| Alias | Split | Samples | 상태 |
|-------|-------|---------|------|
| **M5t2** | Temporal 8:1:1 | 2880 train | ⭐ **권장** |
| M5t | Temporal 1:1:1 | 1198 train | 비교용 |
| M5 | Random | 2880 train | Baseline |

---

## 4. View Ablation 실험

### Config 위치
`configs/experiments/view_ablation/`

### 설정

| Config | Views | 선택 | 설명 |
|--------|-------|------|------|
| `E0_1_1view` | 1 | random | Sanity check |
| `E0_1_2view` | 2 | random | Minimal |
| `E0_1_3view` | 3 | random | ⭐ Best (inference) |
| `E0_1_5view` | 5 | random | Near-full |
| `E0_1_6view` | 6 | all | Baseline |

### 명령어

```bash
cd /home/joon/dev/FaceLift

# N-view 학습
export CUDA_VISIBLE_DEVICES=X && nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e view_ablation/E0_1_Nview \
    > logs/view_ablation_Nview_training.log 2>&1 &
```

### 결과

**Inference-time** (260205):
| Views | PSNR | 비고 |
|-------|------|------|
| **3** | **21.12** | ⭐ Best |
| 2 | 20.20 | |
| 4-6 | 19.58 | Baseline |

**Training-time** (진행중):
| Views | GPU | 상태 |
|-------|-----|------|
| 2 | 6 | 🔄 |
| 3 | 5 | 🔄 |

---

## 5. 기본 실험

### E0 (FaceLift Baseline)

| ID | 데이터셋 | 설명 |
|----|----------|------|
| E0_1_facelift | M5t2 | 기본 설정 |

```bash
export CUDA_VISIBLE_DEVICES=X && nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e E0_1_facelift \
    > logs/gslrm_M5t2_E0_1.log 2>&1 &
```

### E1 (GT Mask)

| ID | 데이터셋 | alpha | 설명 |
|----|----------|-------|------|
| E1_2_alpha | M5t2 | 0.1 | Mask + Alpha |

```bash
export CUDA_VISIBLE_DEVICES=X && nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e E1_2_alpha \
    > logs/gslrm_M5t2_E1_2.log 2>&1 &
```

---

## 6. MVDiffusion 실험

### 현재 학습

| Config | 설명 | GPU | 상태 |
|--------|------|-----|------|
| mouse_mvdiffusion_M5t2_cyclic | 6x Cyclic Aug | 4 | 🔄 10% |

```bash
export CUDA_VISIBLE_DEVICES=4 && nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py --config configs/mvdiffusion/mouse_mvdiffusion_M5t2_cyclic.yaml \
    > logs/mvdiff_cyclic.log 2>&1 &
```

---

## 7. 체크포인트 위치

| 모델 | 경로 | 용도 |
|------|------|------|
| GS-LRM Pretrained | `checkpoints/gslrm/ckpt_*.pt` | Fine-tuning 시작점 |
| GS-LRM Best | `checkpoints/gslrm/{exp}/best_psnr.pt` | Inference |
| MVDiffusion Best | `checkpoints/mvdiffusion/mouse_M5t2_cfgr/` | E2E Pipeline |

---

*Experiment Registry v3.0 | 260205*
