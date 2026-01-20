# D7_t Experiment Overview

> D7_t 데이터셋 기반 체계적 실험 설계 종합 문서
> Created: 2026-01-19

---

## 1. 데이터셋 정보

### D7_t Dataset

| 항목 | 값 |
|------|-----|
| Split 방식 | Uniform Temporal 1:1:1 (PoseSplatter 방식) |
| Train | 1,221 samples |
| Val | 1,186 samples |
| Test | 1,187 samples |
| **Total** | **3,594 samples** |
| Source | D7 dataset with temporal split |
| Location | `/home/joon/data/preprocessed/FaceLift_mouse/D7_t/` |

### Split Method

```
Time ──────────────────────────────────────────────────────►
     |◄──── Train ────►|◄──── Val ────►|◄──── Test ────►|
           33.4%             33.0%            33.6%
        frame 0-6110     frame 6115-12050  frame 12055-17995
```

---

## 2. Loss 공식

### GS-LRM 논문 (Baseline)

$$
\mathcal{L}_{\text{paper}} = \mathcal{L}_{\text{MSE}} + 0.5 \cdot \mathcal{L}_{\text{perc}}
$$

### FaceLift Mouse Extension

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}^{M} + 0.5 \cdot \mathcal{L}_{\text{perc}}^{M} + \lambda_{\text{bg}} \cdot \mathcal{L}_{\text{bg}} + \lambda_{\alpha} \cdot \mathcal{L}_{\alpha}
$$

| Loss | 설명 | 논문 |
|------|------|------|
| $\mathcal{L}_{\text{MSE}}$ | Pixel-wise L2 loss | ✅ GS-LRM |
| $\mathcal{L}_{\text{perc}}$ | VGG perceptual loss ($\lambda=0.5$) | ✅ GS-LRM |
| $\mathcal{L}_{\text{bg}}$ | Background color loss | 🆕 Extension |
| $\mathcal{L}_{\alpha}$ | Alpha channel supervision (BCE) | 🆕 Extension |

---

## 3. 실험 매트릭스

### 전체 실험 비교표

| ID | Views | Random | Mask | BG Loss | α Loss | 검증 가설 |
|----|-------|--------|------|---------|--------|----------|
| **E1.1** | 4 | ✅ | none | ❌ | ❌ | H1 baseline |
| **E1.2** | 4 | ❌ | none | ❌ | ❌ | H1 control |
| **E2.1** | 4 | ❌ | rgb_pred | ✅ | ❌ | H2 |
| **E2.2** | 4 | ❌ | gt | ✅ | ❌ | H2 oracle |
| **E2.3** | 4 | ❌ | alpha | ✅ | ❌ | H2/H3 |
| **E3.2** | 5 | ❌ | alpha | ✅ | ❌ | H3/H4 |
| **E4.2** | 5 | ❌ | alpha | ✅ | ✅ 0.1 | H4 |
| **E5.1** | 5 | ✅ | alpha | ✅ | ❌ | H5 |

### 가설 요약

| 가설 | 질문 | 비교 |
|------|------|------|
| **H1** | Random view selection 효과? | E1.1 vs E1.2 |
| **H2** | 최적 mask 방법? | E1.2 < E2.1 < E2.3 ≈ E2.2 |
| **H3** | View 수 증가 효과? | E2.3 (4v) vs E3.2 (5v) |
| **H4** | Alpha loss 효과? | E3.2 vs E4.2 |
| **H5** | Combined best? | E5.1 vs {E3.2, E1.1} |

---

## 4. 실험 명령어

### 우선순위 순서

```bash
# P1: Baseline (E1.1, E1.2)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E1_1_paper_random.yaml > logs/d7t_e1_1_paper_random.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E1_2_paper_fixed.yaml > logs/d7t_e1_2_paper_fixed.log 2>&1 &

# P2: Mask Methods (E2.1, E2.2, E2.3)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E2_1_rgb_mask.yaml > logs/d7t_e2_1_rgb_mask.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E2_2_gt_mask.yaml > logs/d7t_e2_2_gt_mask.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E2_3_alpha_mask.yaml > logs/d7t_e2_3_alpha_mask.log 2>&1 &

# P3: 5-View (E3.2)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E3_2_5v_alpha.yaml > logs/d7t_e3_2_5v_alpha.log 2>&1 &

# P4: Alpha Loss (E4.2)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E4_2_5v_alpha_loss.yaml > logs/d7t_e4_2_5v_alpha_loss.log 2>&1 &

# P5: Combined Best (E5.1)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py --config configs/mouse/D7_t_E5_1_5v_alpha_random.yaml > logs/d7t_e5_1_5v_alpha_random.log 2>&1 &
```

---

## 5. 코드 위치

### gpu03 서버

```
/home/joon/dev/FaceLift/
├── configs/mouse/D7_t_*.yaml     # 8개 실험 설정
├── scripts/visualization/
│   ├── visualize_gaussian_rerun.py
│   └── batch_visualize.py
├── mouse_extensions/
│   └── utils/logging_utils.py    # split_method logging
└── docs/
    ├── D7_t_experiment_matrix.md
    ├── D7_t_Experiment_Commands.md
    ├── GS-LRM_Loss_Formula.md
    └── Code_Location_Registry.md
```

### 로컬 미러

```
/Users/joon/Documents/Obsidian/30_Projects/_CODES/code_Face_Lift/
├── docs/                         # 문서 (동일)
├── configs/mouse/                # 설정 (동일)
└── scripts/visualization/        # 시각화 (동일)
```

---

## 6. WandB 로깅

### Config 항목

```yaml
experiment:
  split_method: "temporal"       # 🆕 추가됨
  dataset_name: "D7_t"
  num_input_views: 4 or 5
  random_view_selection: true/false
  mask_source: "none/gt/alpha/rgb_pred"
```

### 주요 메트릭

| 메트릭 | 설명 |
|--------|------|
| `val/psnr` | Primary metric |
| `val/lpips` | Perceptual quality |
| `val/mask_iou` | Mask quality |
| `train_meta/step` | Current step |
| `train_meta/epoch` | Current epoch |

---

## 7. 시각화

### Rerun Gaussian Visualization

```bash
# 단일 파일
python scripts/visualization/visualize_gaussian_rerun.py gaussians.ply --save

# 배치 실행
python scripts/visualization/batch_visualize.py \
    --exp-pattern "D7_t_*" \
    --output-dir viz_output \
    --num-samples 5
```

---

## 8. 관련 문서

- [D7_t_experiment_matrix.md](./D7_t_experiment_matrix.md) - 상세 가설 및 비교군
- [D7_t_Experiment_Commands.md](./D7_t_Experiment_Commands.md) - 전체 명령어
- [GS-LRM_Loss_Formula.md](./GS-LRM_Loss_Formula.md) - Loss 공식 상세
- [Code_Location_Registry.md](./Code_Location_Registry.md) - 코드 위치

---

*Created: 2026-01-19*
