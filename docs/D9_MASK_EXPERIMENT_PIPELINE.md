# D9 Mask Mode Experiment Pipeline

> **목적**: D9 (원본 해상도 1152x1024) 데이터셋으로 mask_mode=gt 학습 후 alpha threshold 시각화
> **생성일**: 2026-01-22
> **예상 GPU**: A6000 (GPU 4-7), 48GB+ 권장

---

## 전체 파이프라인 요약

```
[1] D9 전처리 → [2] Split 생성 → [3] mask_mode=gt 학습 → [4] Alpha 시각화
```

---

## Step 1: D9 데이터셋 전처리

```bash
cd /home/joon/dev/FaceLift

# D9 전처리 (원본 해상도 1152x1024, ~48GB GPU 필요)
python -m mouse_extensions.preprocessing.preprocess \
    --preset D9 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D9
```

**예상 시간**: ~30분 (전체 프레임)
**출력 파일**:
- `D9/train/000000/`, `000001/`, ...
- `D9/data_mouse_all.txt` (전체 샘플 목록)

---

## Step 2: Train/Val Split 생성

```bash
cd /home/joon/dev/FaceLift

# 기본 split (val_ratio=0.1)
python -m mouse_extensions.preprocessing.generate_split \
    --data-dir /home/joon/data/preprocessed/FaceLift_mouse/D9 \
    --val-ratio 0.1

# 또는 다른 비율/시드
python -m mouse_extensions.preprocessing.generate_split \
    --data-dir /home/joon/data/preprocessed/FaceLift_mouse/D9 \
    --val-ratio 0.15 --seed 123
```

**출력 파일**:
- `D9/data_mouse_train.txt`
- `D9/data_mouse_val.txt`

---

## Step 3: mask_mode=gt 빠른 학습

```bash
cd /home/joon/dev/FaceLift

# D9 + E_quick_alpha (mask_mode=gt, 500 steps)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D9 -e E_quick_alpha \
    > logs/d9_quick_alpha.log 2>&1 &

# 학습 모니터링
tail -f logs/d9_quick_alpha.log

# 또는 D7_1로 먼저 테스트 (16GB GPU 가능)
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E_quick_alpha \
    > logs/d7_1_quick_alpha.log 2>&1 &
```

**학습 설정** (`configs/experiments/E_quick_alpha.yaml`):
- `mask_mode: gt` - GT mask 영역에서만 L2 loss 계산
- `alpha_loss_weight: 0.0` - alpha supervision 없음
- `max_fwdbwd_passes: 500` - 빠른 테스트

**체크포인트 위치**: `checkpoints/gslrm/D9_E_quick_alpha/ckpt_step_500.pt`

---

## Step 4: Alpha Threshold 시각화

### 4.1 WandB 스타일 시각화 (GT vs Rendered)

```bash
cd /home/joon/dev/FaceLift

# D9 학습 체크포인트 사용
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_rendered_alpha \
    --checkpoint checkpoints/gslrm/D9_E_quick_alpha/ckpt_step_500.pt \
    --config configs/mouse/D9_E_quick_alpha.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D9 \
    --output_dir alpha_analysis/D9_quick_alpha \
    --sample_idx 0

# D7_1 학습 체크포인트 사용
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_rendered_alpha \
    --checkpoint checkpoints/gslrm/D7_1_E_quick_alpha/ckpt_step_500.pt \
    --config configs/mouse/D7_1_E_quick_alpha.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --output_dir alpha_analysis/D7_1_quick_alpha \
    --sample_idx 0
```

**출력물**:
```
alpha_analysis/D9_quick_alpha/
├── figures/
│   └── gt_vs_pred_wandb.png   # 5 rows × 6 views
└── alpha_analysis_report.md
```

### 4.2 Alpha Threshold 비교 시각화

```bash
cd /home/joon/dev/FaceLift

# 다양한 threshold에서 foreground mask 비교
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_alpha_thresholds \
    --checkpoint checkpoints/gslrm/D9_E_quick_alpha/ckpt_step_500.pt \
    --config configs/mouse/D9_E_quick_alpha.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D9 \
    --output_dir alpha_threshold_comparison/D9 \
    --thresholds 0.3 0.5 0.7 0.9 0.99

# D7_1 버전
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_alpha_thresholds \
    --checkpoint checkpoints/gslrm/D7_1_E_quick_alpha/ckpt_step_500.pt \
    --config configs/mouse/D7_1_E_quick_alpha.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --output_dir alpha_threshold_comparison/D7_1 \
    --thresholds 0.3 0.5 0.7 0.9 0.99
```

**출력물**:
```
alpha_threshold_comparison/D9/
├── threshold_comparison.png   # 각 threshold별 mask overlay
└── report.md                  # Alpha 통계 + GT mask 비교
```

---

## Quick One-Liner (전체 파이프라인)

```bash
# === D7_1 빠른 테스트 (16GB GPU) ===
cd /home/joon/dev/FaceLift

# 1. 학습 (D7_1은 이미 전처리됨)
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E_quick_alpha

# 2. 시각화 (학습 완료 후)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_alpha_thresholds \
    --checkpoint checkpoints/gslrm/D7_1_E_quick_alpha/ckpt_step_500.pt \
    --config configs/mouse/D7_1_E_quick_alpha.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --output_dir alpha_threshold_comparison/D7_1 \
    --thresholds 0.3 0.5 0.7 0.9 0.99
```

---

## 체크포인트 경로 확인

학습 후 체크포인트 경로가 다를 수 있음. 실제 경로 확인:

```bash
# 최근 생성된 체크포인트 찾기
find checkpoints -name '*.pt' -mmin -60 | sort

# 또는 특정 실험
ls -la checkpoints/gslrm/D*_E_quick_alpha/
```

---

## 예상 결과

### mask_mode=gt 학습 후 예상 Alpha 분포

| Threshold | 예상 Coverage | 의미 |
|-----------|---------------|------|
| 0.3 | ~5-10% | 높은 확신 foreground |
| 0.5 | ~3-5% | GT mask와 유사 |
| 0.7 | ~2-4% | 엄격한 foreground |
| 0.9 | ~1-3% | 매우 엄격 |
| 0.99 | <1% | 거의 중심부만 |

**비교 대상**: GT mask coverage ~2.8%

### 성공 기준

- Alpha가 GT mask 영역에 집중됨 (이전: 전체 ~100%)
- Threshold 0.5에서 GT mask와 유사한 coverage
- Error heatmap이 foreground에 집중됨

---

## Troubleshooting

### Config 병합 오류
```
KeyError: 'D9' not found in datasets
```
→ `configs/datasets/D9.yaml` 존재 여부 확인

### GPU 메모리 부족
```
CUDA out of memory
```
→ D9 (1152x1024)는 48GB+ 필요. D7_1 (576x512)로 먼저 테스트

### 체크포인트 경로
```
FileNotFoundError: ckpt_step_500.pt
```
→ `find checkpoints -name '*.pt' -mmin -60`로 실제 경로 확인

---

*Created: 2026-01-22 | FaceLift Mouse Mask Experiment*
