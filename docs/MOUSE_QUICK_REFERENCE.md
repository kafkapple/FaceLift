# FaceLift Mouse Extension - Quick Reference

> **Last Updated**: 2026-01-22 22:30
> **Full Documentation**: Obsidian `30_Projects/_CODES/code_Face_Lift/docs/`

---

## 1. Preprocessing (통합 스크립트)

### 1.1 기본 명령어

```bash
cd /home/joon/dev/FaceLift

# 프리셋 목록 확인
python -m mouse_extensions.preprocessing.preprocess --list-presets

# D7.1 (권장) - 정규화 완료, 즉시 학습 가능
python -m mouse_extensions.preprocessing.preprocess \
    --preset D7.1 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1

# D9 (원본 해상도 1152x1024) - ⚠️ 정규화 미포함, 학습 불가
python -m mouse_extensions.preprocessing.preprocess \
    --preset D9 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D9
```

### 1.2 프리셋 목록

| Preset | 해상도 | fx | trans | 정규화 | 상태 |
|--------|--------|-----|-------|--------|------|
| **D7.1** | 512×512 | 549 | 2.7 | ✅ | **권장** |
| D8 | 512×512 | 549 | 2.7 | ✅ | 실험적 |
| D9 | 1152×1024 | 1632 | 246 | ❌ | ⚠️ 미정규화 |

### 1.3 D9 주의사항 ⚠️

**D9는 카메라 정규화 미적용 → Pretrained 모델과 불일치 → 학습 실패**

| 파라미터 | D7.1 (정규화) | D9 (원본) | Pretrained 기대 |
|----------|---------------|-----------|-----------------|
| fx | 549 | 1632 | ~549 |
| translation | 2.7 | 246 | ~2.7 |

**해결 방법**: D9 사용 시 translation 정규화 추가 필요 (246 → ~8.0)

### 1.4 전처리 완료 시 자동 생성

```bash
# 전처리 완료 시 자동 출력:
[Config Generated] /home/joon/dev/FaceLift/configs/datasets/D9.yaml

============================================================
Complete: /home/joon/data/preprocessed/FaceLift_mouse/D9
Total: 3597, Train: 3238, Val: 359
Split files: data_mouse_train.txt, data_mouse_val.txt, data_mouse_all.txt
============================================================
```

### 1.5 Split 재생성 (필요시)

```bash
python -m mouse_extensions.preprocessing.generate_split \
    --data-dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --val-ratio 0.1 \
    --update-config D7_1
```

---

## 2. Training (모듈화 Config)

### 2.1 기본 명령어 (권장)

```bash
cd /home/joon/dev/FaceLift

# 형식: train_gslrm.py -d {데이터셋} -e {실험}
# Config 자동 병합: base + datasets/{D}.yaml + experiments/{E}.yaml

# D7.1 + 기본 실험
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E1_1_paper_random

# D7.1 + mask_mode=gt 빠른 테스트 (500 steps)
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E_quick_alpha

# Background 실행
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E_quick_alpha \
    > logs/d7_1_quick_alpha.log 2>&1 &
```

### 2.2 사용 가능한 데이터셋 (-d)

```bash
ls configs/datasets/
# D7_1.yaml, D7_2.yaml, D8.yaml, D9.yaml, D3.yaml, ...
```

### 2.3 사용 가능한 실험 (-e)

```bash
ls configs/experiments/
# E1_1_paper_random.yaml  - 기본 학습 설정
# E_quick_alpha.yaml      - mask_mode=gt 빠른 테스트 (500 steps)
# ...
```

### 2.4 E_quick_alpha 설정

```yaml
# configs/experiments/E_quick_alpha.yaml
training:
  losses:
    mask_mode: gt              # GT mask 영역에서만 L2 loss
    alpha_loss_weight: 0.0     # alpha supervision 없음
  schedule:
    max_fwdbwd_passes: 500     # 빠른 테스트
```

---

## 3. Turntable Visualization

### 3.1 기본 설정

```yaml
visualization:
  turntable:
    num_views: 64              # 8×8 그리드 (기본)
    resolution: 384
    elevation: 20
    radius: 2.7
    trajectory_mode: "dataset_cameras"  # 기본값 ⭐
```

### 3.2 Trajectory 모드

| Mode | 설명 | 기본값 |
|------|------|--------|
| **`dataset_cameras`** | 6개 실제 카메라 간 보간 (1→3→5→0→4→2) | ⭐ 기본 |
| `turntable` | 생쥐 중심 360도 회전 | |
| `spiral` | Elevation 점진 변화 + 회전 | |

### 3.3 현재 생성 현황

| 항목 | Train | Validation |
|------|-------|------------|
| Turntable 이미지 | ✅ 생성 | ❌ 미생성 |
| WandB 로깅 | ✅ | ❌ |

### 3.4 Dataset Camera Trajectory 코드

```python
from gslrm.model.gaussians_renderer import render_dataset_trajectory

frames, segments = render_dataset_trajectory(
    gaussians,
    dataset_c2ws,        # [6, 4, 4]
    dataset_fxfycxcy,    # [6, 4]
    num_views=150,
    camera_order=[1, 3, 5, 0, 4, 2],  # 방문 순서
    loop=True
)
```

---

## 4. Mask Mode Analysis

### 4.1 mask_mode vs alpha_loss (독립적\!)

| 설정 | 역할 | 
|------|------|
| **mask_mode** | L2 loss 계산 영역 (none/gt/alpha) |
| **alpha_loss_weight** | rendered alpha → GT mask supervision |

```yaml
# 조합 예시
mask_mode: gt + alpha_loss: 0.0    # GT 영역만 L2, alpha 학습 안함
mask_mode: none + alpha_loss: 0.1  # 전체 L2 + alpha supervision
```

### 4.2 Alpha Threshold 분석

```bash
# 학습 완료 후 threshold별 mask 비교
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_alpha_thresholds \
    --checkpoint checkpoints/gslrm/D7_1_E_quick_alpha/ckpt_step_500.pt \
    --config configs/experiments/E_quick_alpha.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --output_dir alpha_threshold_comparison/D7_1 \
    --thresholds 0.3 0.5 0.7 0.9 0.99
```

### 4.3 WandB 스타일 시각화

```bash
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_rendered_alpha \
    --checkpoint checkpoints/gslrm/D7_1_E_quick_alpha/ckpt_step_500.pt \
    --config configs/experiments/E_quick_alpha.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --output_dir alpha_analysis/D7_1_quick_alpha \
    --sample_idx 0
```

**출력 레이아웃** (5 rows × 6 views):
- Row 1: GT RGB
- Row 2: Rendered RGB  
- Row 3: GT + Mask overlay (녹색)
- Row 4: Rendered + Alpha overlay (파란색)
- Row 5: Error heatmap

---

## 5. Inference

```bash
cd /home/joon/dev/FaceLift

# Pretrained 체크포인트
CUDA_VISIBLE_DEVICES=4 python inference_mouse.py \
    --sample_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1/train/000000 \
    --checkpoint checkpoints/gslrm/ckpt_0000000000021125.pt \
    --config configs/gslrm.yaml \
    --output_dir outputs/inference/sample_000000

# 학습된 체크포인트
CUDA_VISIBLE_DEVICES=4 python inference_mouse.py \
    --sample_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1/train/000000 \
    --checkpoint checkpoints/gslrm/D7_1_E_quick_alpha/ckpt_step_500.pt \
    --config configs/experiments/E_quick_alpha.yaml \
    --output_dir outputs/inference/D7_1_quick_alpha
```

---

## 6. GPU Reference

| GPU Index | Model | PyTorch 호환 |
|-----------|-------|--------------|
| 0-3 | RTX PRO 6000 Blackwell | ❌ |
| **4-7** | **RTX A6000** | **✅** |

**항상 `CUDA_VISIBLE_DEVICES=4` 이상 사용**

---

## 7. Troubleshooting

### D9 렌더링 실패 (기하학 곡선만 출력)
```
원인: 카메라 정규화 미적용 (translation 246 vs 기대값 2.7)
해결: D7.1 사용 또는 D9에 정규화 추가
```

### CUDA 호환성 오류
```
CUDA capability sm_120 is not compatible
```
→ GPU 4-7 (A6000) 사용: `CUDA_VISIBLE_DEVICES=4`

### 체크포인트 경로 찾기
```bash
find checkpoints -name "*.pt" -mmin -60 | sort
```

---

## 8. 파일 위치

### Config 구조
```
configs/
├── gslrm.yaml              # Base config
├── datasets/               # 데이터셋별 설정
│   ├── D7_1.yaml
│   ├── D9.yaml
│   └── ...
└── experiments/            # 실험별 설정
    ├── E1_1_paper_random.yaml
    └── E_quick_alpha.yaml
```

### 전처리 데이터
```
/home/joon/data/preprocessed/FaceLift_mouse/
├── D7_1/                   # 512×512, 정규화 ✅
│   ├── train/000000/
│   ├── val/000000/
│   └── data_mouse_train.txt
└── D9/                     # 1152×1024, 미정규화 ⚠️
    ├── 000000/
    └── data_mouse_train.txt
```

---

*Updated: 2026-01-22 22:30*
