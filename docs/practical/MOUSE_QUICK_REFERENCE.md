# FaceLift Mouse Quick Reference v3.2

> Last Updated: 2026-01-23

## Quick Start (Modular Mode)

```bash
cd /home/joon/dev/FaceLift

# 권장 실험 (GT + Alpha Supervision)
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E2_gt_alpha
```

---

## Modular Config 구조

```
configs/
├── base/gslrm_mouse.yaml      # 공통 설정 (runtime, model, pretrained)
├── datasets/                   # 데이터셋 설정
│   ├── D7_1.yaml
│   ├── D7_2.yaml
│   └── v13.yaml
└── experiments/                # 실험 설정
    ├── E0_none.yaml
    ├── E1_gt.yaml
    ├── E2_gt_alpha.yaml ⭐
    ├── E3_alpha.yaml
    └── E4_bg_penalty.yaml
```

---

## 실험 명령어

### 단일 실험

```bash
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E2_gt_alpha
```

### 병렬 실행 (nohup)

```bash
# E0: Baseline (GPU 4)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E0_none > logs/D7_1_E0_none.log 2>&1 &

# E2: GT + Alpha ⭐ (GPU 5)
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E2_gt_alpha > logs/D7_1_E2_gt_alpha.log 2>&1 &

# E1: GT Only (GPU 6)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E1_gt > logs/D7_1_E1_gt.log 2>&1 &

# E4: BG Penalty (GPU 7)
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E4_bg_penalty > logs/D7_1_E4_bg_penalty.log 2>&1 &
```

---

## 마스크 모드 설정

| Experiment | mask_mode | alpha_loss | bg_loss |
|------------|-----------|------------|---------|
| E0_none | none | 0.0 | 0.0 |
| E1_gt | gt | 0.0 | 0.0 |
| **E2_gt_alpha** ⭐ | **gt** | **0.1** | 0.0 |
| E3_alpha | none | 0.1 | 0.0 |
| E4_bg_penalty | none | 0.1 | 0.5 |

---

## 모니터링

```bash
tail -f logs/D7_1_E2_gt_alpha.log
nvidia-smi -l 5
ps aux | grep train_gslrm
```

---

## 핵심 파일 위치

| 항목 | 위치 |
|------|------|
| Base config | `configs/base/gslrm_mouse.yaml` |
| Pretrained | `checkpoints/gslrm/ckpt_0000000000021125.pt` |

---

*FaceLift Mouse v3.2 | Modular Mode | 2026-01-23*

---

## 실험 변형 (Variants)

| Experiment | 설명 | 사용 |
|------------|------|------|
| E2_gt_alpha_4v | 4-view 입력 | View ablation |
| E2_gt_alpha_fixed | Fixed view order | Random vs Fixed 비교 |
| E2_gt_alpha_overfit | 1-view overfit | 디버깅/sanity check |
| E6_rgb_pred | RGB mask prediction | Deprecated, 비교용 |

```bash
# 4v ablation
torchrun ... -d D7_1 -e E2_gt_alpha_4v

# Fixed view order
torchrun ... -d D7_1 -e E2_gt_alpha_fixed

# Overfit test
torchrun ... -d D7_1 -e E2_gt_alpha_overfit
```
