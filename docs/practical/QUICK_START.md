# FaceLift Mouse Quick Start

> **Version**: v6.0 | **Updated**: 2026-01-25
> **목적**: 실험 실행에 필요한 최소 명령어

---

## 1. 즉시 실행 명령어

### 권장 실험 (Production)

```bash
cd /home/joon/dev/FaceLift
conda activate facelift

# ⭐ M3 + GT mask + Alpha (권장)
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3 -e E1_2_alpha

# M3_norm (H1/H3 검증용)
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_norm -e E1_2_alpha

# M3_persample (H5 검증용)
CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_persample -e E1_2_alpha
```

### Background 실행

```bash
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_norm -e E1_2_alpha > logs/M3_norm_E1_2.log 2>&1 &
```

---

## 2. GPU 사용 현황

```bash
# GPU 상태 확인
nvidia-smi

# 실험 프로세스 확인
ps aux | grep train_gslrm
```

| GPU | 아키텍처 | 사용 가능 |
|-----|----------|-----------|
| 0-3 | Blackwell | ❌ PyTorch 미지원 |
| 4-7 | A6000 | ✅ 사용 |

---

## 3. 데이터셋 요약

| 데이터셋 | Coverage | fx | 용도 |
|----------|----------|-----|------|
| M1 (D7_1) | 50% | 549 | Baseline |
| M3 (D10_3) | 80% | 739 | H2 검증 |
| **M3_norm** | 80% | 549 | ⭐ 권장 |
| M3_persample | 80% | 549 | H5 검증 |

---

## 4. 실험 설정 요약

| 실험 | mask_mode | alpha_loss | 용도 |
|------|-----------|------------|------|
| E0_1_facelift | none | 0.0 | 원본 논문 |
| E1_1_gt | gt | 0.0 | GT mask만 |
| **E1_2_alpha** | gt | 0.1 | ⭐ 권장 |
| E1_3_lgm | gt | 1.0 | 강한 alpha |

---

## 5. 모니터링

```bash
# WandB 대시보드
# https://wandb.ai/{username}/facelift-mouse

# 로그 실시간 확인
tail -f logs/M3_norm_E1_2.log

# Validation 결과
ls experiments/validation/M3_norm_E1_2_alpha/
```

---

## 6. 관련 문서

| 문서 | 내용 |
|------|------|
| [EXPERIMENT_REGISTRY](./EXPERIMENT_REGISTRY.md) | 전체 실험/데이터셋 목록 |
| [Debug Guide](../tutorials/VSCode_Debug_Mask_Guide.md) | 디버깅 가이드 |

---

*FaceLift Mouse Quick Start v6.0 | 2026-01-25*
