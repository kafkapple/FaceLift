# FaceLift Mouse Extension - Quick Reference

> **Last Updated**: 2026-01-22 23:30
> **Full Documentation**: Obsidian `30_Projects/_CODES/code_Face_Lift/docs/`

---

## Quick Start: mask_mode=gt 실험

```bash
cd /home/joon/dev/FaceLift

# 1. 학습 (500 steps, ~10분)
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E_quick_alpha

# 2. 체크포인트 확인
find checkpoints -name '*.pt' -mmin -60 | sort

# 3. Alpha Threshold 비교 시각화
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_alpha_thresholds \
    --checkpoint checkpoints/gslrm/D7_1_E_quick_alpha/ckpt_step_500.pt \
    --config configs/mouse/D7_1_E_quick_alpha.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --output_dir alpha_threshold_comparison/D7_1 \
    --thresholds 0.3 0.5 0.7 0.9 0.99
```

**출력물**: `alpha_threshold_comparison/D7_1/threshold_comparison.png`

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
```

### 1.2 프리셋 목록

| Preset | 해상도 | fx | trans | 정규화 | 상태 |
|--------|--------|-----|-------|--------|------|
| **D7.1** | 512×512 | 549 | 2.7 | ✅ | **권장** |
| D9 | 1152×1024 | 1632 | 246 | ❌ | ⚠️ 미정규화 |

### 1.3 D9 주의사항 ⚠️

D9는 카메라 정규화 미적용 → Pretrained 모델과 불일치 → **학습 실패**

---

## 2. Training (모듈화 Config)

### 2.1 기본 명령어

```bash
# 형식: train_gslrm.py -d {데이터셋} -e {실험}

# D7.1 + mask_mode=gt 빠른 테스트 (500 steps)
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E_quick_alpha

# Background 실행
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E_quick_alpha \
    > logs/d7_1_quick_alpha.log 2>&1 &
```

### 2.2 E_quick_alpha 설정

```yaml
training:
  losses:
    mask_mode: gt              # GT mask 영역에서만 L2 loss
    alpha_loss_weight: 0.0     # alpha supervision 없음
  schedule:
    max_fwdbwd_passes: 500     # 빠른 테스트
```

---

## 3. Mask Analysis (학습 후)

### 3.1 Alpha Threshold 비교 ⭐

```bash
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_alpha_thresholds \
    --checkpoint {체크포인트_경로} \
    --config configs/mouse/D7_1_E_quick_alpha.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --output_dir alpha_threshold_comparison/D7_1 \
    --thresholds 0.3 0.5 0.7 0.9 0.99
```

**출력물**:
```
alpha_threshold_comparison/D7_1/
├── threshold_comparison.png   # 각 threshold별 mask overlay
└── report.md
```

### 3.2 WandB 스타일 GT vs Rendered

```bash
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_rendered_alpha \
    --checkpoint {체크포인트_경로} \
    --config configs/mouse/D7_1_E_quick_alpha.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --output_dir alpha_analysis/D7_1_quick_alpha \
    --sample_idx 0
```

### 3.3 mask_mode vs alpha_loss (독립적!)

| 설정 | 역할 | 
|------|------|
| **mask_mode** | L2 loss 계산 영역 (none/gt/alpha) |
| **alpha_loss_weight** | rendered alpha → GT mask supervision |

---

## 4. Turntable Visualization

### 4.1 기본 설정 (6×6 그리드)

```yaml
visualization:
  turntable:
    num_views: 36              # 6×6 그리드 (기본)
    grid_rows: 6
    grid_cols: 6
    camera_order: [1, 3, 5, 0, 4, 2]  # 360도 순회
    add_row_labels: true       # "Cam 1 -> 3" 레이블
```

### 4.2 출력물

| 단계 | 파일 | 설명 |
|------|------|------|
| Train | turntable_{uid}.jpg | 6×6 그리드 이미지 |
| Validation | turntable_grid.jpg | 6×6 그리드 이미지 |
| Validation | turntable.mp4 | 150프레임 비디오 |

---

## 5. GPU Reference

| GPU Index | Model | PyTorch 호환 |
|-----------|-------|--------------|
| 0-3 | RTX PRO 6000 Blackwell | ❌ |
| **4-7** | **RTX A6000** | **✅** |

**항상 `CUDA_VISIBLE_DEVICES=4` 이상 사용**

---

## 6. Troubleshooting

| 문제 | 원인 | 해결 |
|------|------|------|
| D9 렌더링 실패 | 카메라 미정규화 | D7.1 사용 |
| CUDA sm_120 오류 | Blackwell GPU | GPU 4-7 사용 |
| 체크포인트 못찾음 | 경로 다름 | `find checkpoints -name '*.pt' -mmin -60` |

---

## 7. 파일 위치

```
configs/
├── datasets/D7_1.yaml, D9.yaml    # -d 플래그
├── experiments/E_quick_alpha.yaml # -e 플래그
└── visualization/turntable_6x6.yaml

/home/joon/data/preprocessed/FaceLift_mouse/
├── D7_1/    # 512×512, 정규화 ✅
└── D9/      # 1152×1024, 미정규화 ⚠️
```

---

*Updated: 2026-01-22 23:30*

---

## 9. Experiment Recommendations

### 9.1 Dataset Selection Matrix

| 기준 | 권장 데이터셋 | 근거 |
|------|-------------|------|
| **기본/신규 실험** | D7.1 ★ | PP=256 + 기하학 정확 + 검증됨 |
| **최대 정밀도** | D8 | Skew 보정, fx=548.99 정확 |
| **확대 마우스** | D7_5 / D8.1 | 1.2-1.3× zoom |
| **원본 해상도** | D9 | 최대 디테일 (48GB+ GPU) |
| **이미지 무손실** | D6-2 | PP만 조정, 픽셀 미변경 |

### 9.2 Geometric Accuracy Comparison

| Dataset | PP | fx | Skew | Ray Error | FaceLift |
|---------|-----|-----|------|-----------|----------|
| D7.1 | 256 ✅ | 549.0 | ❌ (~0.9px) | 0° | ✅ |
| **D8** | 256 ✅ | 548.99 ✅ | ✅ (0px) | ≈0° | ✅ |
| D6-2 | ~208 ❌ | 549 | ❌ | 0° | ❌ PP불일치 |
| D9 | ~600 ❌ | ~1600 | ❌ | 0° | ❌ 적응필요 |

### 9.3 Recommended Experiment Priority

```
P1: D7.1 + E_quick_alpha (mask_mode=gt)     # 기본 검증
P2: D8 + E_quick_alpha                       # D7.1과 비교
P3: D7_5 + E_quick_alpha                     # 확대 마우스
P4: D9 + E_long_train                        # 원본 해상도 (48GB+)
```

### 9.4 D7 vs D6-2 Trade-off

| 항목 | D7.1 | D6-2 |
|------|------|------|
| 이미지 변형 | Shift (~50-90px 가장자리 손실) | 없음 (원본 유지) |
| PP 값 | 256 (고정) | ~208±65 (가변) |
| FaceLift 호환 | ✅ 완벽 | ❌ PP 불일치 |
| 이미지 품질 | ~30px 테두리 손실 | ✅ 완벽 |

---

*Updated: 2026-01-22*
