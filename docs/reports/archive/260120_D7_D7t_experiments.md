# D7 및 D7_t 실험 문서

> **버전**: v1.0 (2026-01-20)
> **상태**: 실험 진행 중

---

## 1. D7 데이터셋 개요

### 핵심 특징

| 항목 | 값 | 의미 |
|------|-----|------|
| **전처리 방법** | PP-centered shift | PP를 이미지 중앙으로 |
| **PP (cx, cy)** | 256, 256 | 정확히 달성 |
| **fx, fy** | 549.4, 549.4 | 정규화됨 |
| **기하학 정확성** | Ray error 0도 | 완벽 |

### D4 대비 개선

| 항목 | D4 | D7 |
|------|-----|-----|
| PP 값 | 256 (강제) | 256 (정확) |
| 실제 PP 오차 | 37px 평균 | 0px |
| Ray 오류 | 5-13도 | 0도 |
| Ghosting 예상 | Yes | No |

---

## 2. D7_t: Temporal Split

### 분할 정보

| Split | Frame Range | Count | Ratio |
|-------|-------------|-------|-------|
| Train | 0 - 6110 | 1222 | 34% |
| Val | 6115 - 12050 | 1187 | 33% |
| Test | 12055 - 17995 | 1188 | 33% |

### 목적

기존 Random split과 달리 **시간적으로 분리**하여:
1. 과거 프레임으로 학습
2. 미래 프레임으로 평가
3. 일반화 성능 측정

---

## 3. 실험 설정 매트릭스

### D7 기본 실험

| Config | Views | Random | Mask | 목적 |
|--------|-------|--------|------|------|
| D7_E1_baseline | 4 | Yes | None | 기본 baseline |
| D7_E2_gtmask | 4 | No | GT | GT 마스크 효과 |
| D7_E3_alpha_safe | 4 | No | Alpha (safe) | 안전한 alpha |
| **D7_E4_5v_alpha** | **5** | **No** | **Alpha** | **주력 실험** |
| D7_E6_5v_opacity | 5 | No | Alpha+Opacity | Opacity 정규화 |
| D7_E7_6v_alpha | 6 | No | Alpha | 최대 뷰 |

### D7_t 실험 (Temporal Split)

| Config | Views | Random | Mask | 가설 |
|--------|-------|--------|------|------|
| E1_1_paper_random | 4 | Yes | None | Paper 설정 |
| E1_2_paper_fixed | 4 | No | None | Random 효과 |
| E2_1_rgb_mask | 4 | No | RGB | RGB 마스크 |
| E2_2_gt_mask | 4 | No | GT | GT 마스크 |
| E2_3_alpha_mask | 4 | No | Alpha | Alpha 마스크 |
| E3_2_5v_alpha | 5 | No | Alpha | 5뷰 alpha |
| E4_2_alpha_loss | 5 | No | Alpha+Loss | Alpha loss |
| E5_1_combined | 5 | Yes | Alpha | 최종 조합 |

### 가설 검증 매트릭스

| 비교 | 가설 | 예상 결과 |
|------|------|-----------|
| E1.1 vs E1.2 | Random view 효과 | E1.1 > E1.2 |
| E2.3 vs E1.2 | Alpha mask 효과 | E2.3 > E1.2 |
| E3.2 vs E2.3 | 5뷰 효과 | E3.2 > E2.3 |
| E4.2 vs E3.2 | Alpha loss 효과 | E4.2 > E3.2 |

---

## 4. 실험 실행 명령어

### 기본 실행

```bash
cd /home/joon/dev/FaceLift
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_E4_5v_alpha.yaml
```

### D7_t 실험

```bash
# E1.1: Paper baseline (random)
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_t_E1_1_paper_random.yaml

# E3.2: 5v alpha (주력)
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_t_E3_2_5v_alpha.yaml
```

### 백그라운드 실행

```bash
CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_t_E1_1_paper_random.yaml \
    > logs/d7t_e1_1.log 2>&1 &
```

---

## 5. 성공 기준

### 정량적 목표

| 지표 | 목표 | D4 대비 |
|------|------|---------|
| Val PSNR | >= 28 | +0.13 |
| Val SSIM | >= 0.992 | 유지 |
| Val LPIPS | <= 0.012 | 유지 |

### 정성적 목표

| 항목 | 목표 | 검증 방법 |
|------|------|-----------|
| Ghosting | 없음 | Turntable 시각적 검토 |
| 기하학 | 정확 | Multi-view consistency |
| 일반화 | 우수 | D7_t test set |

---

## 6. 모니터링

### WandB

- Project: facelift-mouse
- Group: D7 또는 D7_t
- 주요 지표: val/psnr, val/lpips, val/mask_iou

### Turntable 검증

```bash
# 특정 step에서 turntable 확인
ls logs/[experiment]/vis/turntable_*.jpg
```

---

## 7. 관련 파일

### 설정 파일
```
/home/joon/dev/FaceLift/configs/mouse/
├── D7_E*.yaml       # D7 실험들
└── D7_t_E*.yaml     # D7_t 실험들
```

### 데이터셋
```
/home/joon/data/preprocessed/FaceLift_mouse/
├── D7/              # 현재 데이터셋
└── D7_t/            # Temporal split 데이터셋
```

---

*Generated: 2026-01-20*
