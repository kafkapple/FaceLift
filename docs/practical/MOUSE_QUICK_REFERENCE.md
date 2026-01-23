# FaceLift Mouse Quick Reference v3.1

> Last Updated: 2026-01-23

## Quick Start

```bash
cd /home/joon/dev/FaceLift

# 권장 실험 (GT + Alpha Supervision)
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E2_gt_alpha.yaml
```

---

## Config 파일 목록

### D7_1 데이터셋 (권장)

| Config | 키워드 | 설명 | Priority |
|--------|--------|------|----------|
| **D7_1_E0_baseline.yaml** | baseline | No mask, no alpha | P2 |
| **D7_1_E1_gt.yaml** | gt | GT mask only | P3 |
| **D7_1_E2_gt_alpha.yaml** ⭐ | gt_alpha | GT mask + alpha supervision | **P0** |
| **D7_1_E2_gt_alpha_4v.yaml** | gt_alpha_4v | 4-view ablation | P4 |
| **D7_1_E2_overfit_1v.yaml** | overfit_1v | 1-view overfit test | P5 |
| **D7_1_E3_alpha.yaml** | alpha | Alpha supervision only | P4 |
| **D7_1_E4_bg_penalty.yaml** | bg_penalty | Background penalty | P3 |

**Symlinks** (편의용):
- `D7_1_E0.yaml` → `D7_1_E0_baseline.yaml`
- `D7_1_E2.yaml` → `D7_1_E2_gt_alpha.yaml` ⭐

---

## 실험 명령어

### P0: 권장 실험 (GT + Alpha)

```bash
# 단일 실행
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E2_gt_alpha.yaml

# 백그라운드 실행
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E2_gt_alpha.yaml \
    > logs/D7_1_E2_gt_alpha.log 2>&1 &
```

### 전체 마스크 모드 비교 (병렬)

```bash
# E0: Baseline (GPU 4)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E0_baseline.yaml \
    > logs/D7_1_E0_baseline.log 2>&1 &

# E2: GT + Alpha ⭐ (GPU 5)
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E2_gt_alpha.yaml \
    > logs/D7_1_E2_gt_alpha.log 2>&1 &

# E1: GT Only (GPU 6)
CUDA_VISIBLE_DEVICES=6 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E1_gt.yaml \
    > logs/D7_1_E1_gt.log 2>&1 &

# E4: BG Penalty (GPU 7)
CUDA_VISIBLE_DEVICES=7 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E4_bg_penalty.yaml \
    > logs/D7_1_E4_bg_penalty.log 2>&1 &
```

### View Ablation

```bash
# 5v (기본)
CUDA_VISIBLE_DEVICES=4 ... --config configs/mouse/D7_1_E2_gt_alpha.yaml

# 4v
CUDA_VISIBLE_DEVICES=5 ... --config configs/mouse/D7_1_E2_gt_alpha_4v.yaml
```

---

## 마스크 모드 설정

| E# | mask_mode | normalize_by_mask | alpha_loss | bg_loss |
|----|-----------|-------------------|------------|---------|
| E0 | none | false | 0.0 | 0.0 |
| E1 | gt | true | 0.0 | 0.0 |
| **E2** ⭐ | **gt** | **true** | **0.1** | 0.0 |
| E3 | none | false | 0.1 | 0.0 |
| E4 | none | false | 0.1 | 0.5 |

---

## Config 검증

새 config 작성 후 반드시 검증:

```bash
python scripts/validate_config.py configs/mouse/my_config.yaml
python scripts/validate_config.py --all  # 전체 검증
```

---

## 모니터링

```bash
# 로그 확인
tail -f logs/D7_1_E2_gt_alpha.log

# GPU 사용량
nvidia-smi -l 5

# 프로세스 확인
ps aux | grep train_gslrm
```

---

## 핵심 파일 위치

| 항목 | 위치 |
|------|------|
| Config 스키마 | `docs/practical/config/CONFIG_SCHEMA.md` |
| 네이밍 규칙 | `docs/practical/experiments/EXPERIMENT_NAMING_CONVENTION.md` |
| 검증 스크립트 | `scripts/validate_config.py` |
| 체크포인트 | `checkpoints/gslrm/{experiment_name}/` |
| 로그 | `logs/{experiment_name}.log` |

---

## See Also

- [CONFIG_SCHEMA.md](config/CONFIG_SCHEMA.md) - Config 구조 상세
- [EXPERIMENT_NAMING_CONVENTION.md](experiments/EXPERIMENT_NAMING_CONVENTION.md) - 네이밍 규칙
- [configs/README.md](../../configs/README.md) - Config 시스템 개요

---

*FaceLift Mouse v3.1 | 2026-01-23*
