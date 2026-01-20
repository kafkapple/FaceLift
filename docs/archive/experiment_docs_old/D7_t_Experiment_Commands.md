# D7_t Experiment Commands

> 우선순위 순서로 정렬된 실험 실행 명령어
> GPU: gpu03 서버, CUDA_VISIBLE_DEVICES=4

## 빠른 참조 테이블

| Priority | ID | Views | Random | Mask | 가설 | 명령어 |
|----------|-----|-------|--------|------|------|--------|
| 1 | E1.1 | 4 | ✅ | none | H1 baseline | `d7t_e1_1_paper_random` |
| 1 | E1.2 | 4 | ❌ | none | H1 control | `d7t_e1_2_paper_fixed` |
| 2 | E2.3 | 4 | ❌ | alpha | H2/H3 base | `d7t_e2_3_alpha_mask` |
| 2 | E2.1 | 4 | ❌ | rgb_pred | H2 | `d7t_e2_1_rgb_mask` |
| 2 | E2.2 | 4 | ❌ | gt | H2 oracle | `d7t_e2_2_gt_mask` |
| 3 | E3.2 | 5 | ❌ | alpha | H3/H4 base | `d7t_e3_2_5v_alpha` |
| 4 | E4.2 | 5 | ❌ | alpha+loss | H4 | `d7t_e4_2_5v_alpha_loss` |
| 5 | E5.1 | 5 | ✅ | alpha | H5 best | `d7t_e5_1_5v_alpha_random` |

---

## 전체 명령어 (복사용)

### Priority 1: Baseline (E1.1 + E1.2)

```bash
# E1.1: Paper Random Baseline
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E1_1_paper_random.yaml > logs/d7t_e1_1_paper_random.log 2>&1 &

# E1.2: Fixed Control
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E1_2_paper_fixed.yaml > logs/d7t_e1_2_paper_fixed.log 2>&1 &
```

### Priority 2: Mask Methods (E2.1, E2.2, E2.3) - H2 검증

```bash
# E2.1: RGB Pred Mask
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E2_1_rgb_mask.yaml > logs/d7t_e2_1_rgb_mask.log 2>&1 &

# E2.2: GT Mask (Oracle)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E2_2_gt_mask.yaml > logs/d7t_e2_2_gt_mask.log 2>&1 &

# E2.3: Alpha Mask
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E2_3_alpha_mask.yaml > logs/d7t_e2_3_alpha_mask.log 2>&1 &
```

### Priority 3: 5-View Alpha (E3.2)

```bash
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E3_2_5v_alpha.yaml > logs/d7t_e3_2_5v_alpha.log 2>&1 &
```

### Priority 4: Alpha Loss (E4.2)

```bash
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E4_2_5v_alpha_loss.yaml > logs/d7t_e4_2_5v_alpha_loss.log 2>&1 &
```

### Priority 5: Combined Best (E5.1)

```bash
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E5_1_5v_alpha_random.yaml > logs/d7t_e5_1_5v_alpha_random.log 2>&1 &
```

---

## 병렬 실행 (다중 GPU)

```bash
# GPU 4: Priority 1-2
CUDA_VISIBLE_DEVICES=4 nohup torchrun ... D7_t_E1_1_paper_random.yaml > logs/d7t_e1_1_paper_random.log 2>&1 &
# → 완료 후
CUDA_VISIBLE_DEVICES=4 nohup torchrun ... D7_t_E2_3_alpha_mask.yaml > logs/d7t_e2_3_alpha_mask.log 2>&1 &

# GPU 5: Priority 1, 3
CUDA_VISIBLE_DEVICES=5 nohup torchrun ... D7_t_E1_2_paper_fixed.yaml > logs/d7t_e1_2_paper_fixed.log 2>&1 &
# → 완료 후
CUDA_VISIBLE_DEVICES=5 nohup torchrun ... D7_t_E3_2_5v_alpha.yaml > logs/d7t_e3_2_5v_alpha.log 2>&1 &

# GPU 6: Priority 4-6
CUDA_VISIBLE_DEVICES=6 nohup torchrun ... D7_t_E2_2_gt_mask.yaml > logs/d7t_e2_2_gt_mask.log 2>&1 &
# → 완료 후
CUDA_VISIBLE_DEVICES=6 nohup torchrun ... D7_t_E4_2_5v_alpha_loss.yaml > logs/d7t_e4_2_5v_alpha_loss.log 2>&1 &
# → 완료 후
CUDA_VISIBLE_DEVICES=6 nohup torchrun ... D7_t_E5_1_5v_alpha_random.yaml > logs/d7t_e5_1_5v_alpha_random.log 2>&1 &
```

---

## 모니터링

```bash
# 로그 확인
tail -f logs/d7t_*.log

# 프로세스 확인
ps aux | grep train_gslrm

# GPU 사용량
watch -n 5 nvidia-smi
```

---

*Created: 2026-01-19*
