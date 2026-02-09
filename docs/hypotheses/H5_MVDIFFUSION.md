# H5: MVDiffusion 학습 최적화

> **가설**: MVDiffusion을 mouse 데이터에 맞게 fine-tuning하면 E2E 품질이 개선될 것이다.
>
> ← [RESEARCH_HYPOTHESES.md](../RESEARCH_HYPOTHESES.md) | **상태**: 🔄 진행중 | **Updated**: 2026-02-09

---

## 1. 실험 설계 (v2.0 — 단일 변수 원칙)

### 1.1 Baseline

| 항목 | 값 |
|------|-----|
| Config | `mouse_mvdiffusion_M5t2.yaml` |
| sparse_mv_attention | **true** |
| reference_view_idx | **0** |
| max_train_steps | **10,000** |
| Checkpoint | `mouse_M5t2/checkpoint-5000` |
| Best PSNR (MVDiff) | 27.30 (CFG 3.0) |

### 1.2 실험 매트릭스 (단일 변수 변경만)

| Config | 변경 변수 | sparse_mv | ref_view | steps | 상태 |
|--------|----------|:---------:|:--------:|:-----:|:----:|
| **M5t2** (baseline) | - | true | 0 | 10K | ckpt-5000 ✅ |
| **M5t2_cfgr** | attention type | **false** | 0 | 10K | ckpt-10000 ✅ |
| **M5t2_randref_sparse** | reference augmentation | true | **random** | 10K | 대기 🆕 |
| **M5t2_20k_sparse** | training duration | true | 0 | **20K** | 대기 🆕 |

### 1.3 의사결정 흐름

```
Phase 1: cfgr vs baseline E2E 비교 (학습 완료, 평가만)
    │
    ├─ cfgr 승 (E2E gap < baseline) → sparse=false 채택
    │   └─ Phase 2A: randref (기존 2변수 config 활용 가능)
    │
    └─ baseline 승 (sparse=true 유지) → Phase 2B
        ├─ randref_sparse (reference augmentation 효과)
        └─ 20k_sparse (longer training 효과)
```

### 1.4 ⚠️ 2변수 Config (실행 비권장)

기존 config 중 baseline 대비 2개 이상 변수가 다른 것들. 경고 추가됨:

| Config | 변경 1 | 변경 2 | 비고 |
|--------|--------|--------|------|
| M5t2_randref | sparse=false | ref=random | → randref_sparse 사용 |
| M5t2_symmetric | sparse=false | ref=[0,3] | 보류 |
| M5t2_20k | sparse=false | 20K steps | → 20k_sparse 사용 |

### 1.5 삭제된 실험

| Config | 이유 | 일자 |
|--------|------|------|
| M5t2_cyclic | 2변수 변경 (sparse=false + ref=[0..5]), 성능 저조 (21.80 vs 27.30) | 260207 |
| M5t2_consistent | cfgr과 유사 (condition_drop 차이만), 중복 | 260207 |

---

## 2. 기존 결과 (Sparse vs Full Attention)

### 2.1 CFG 효과 비교

| 모델 | CFG 1.0 | CFG 3.0 | 변화 |
|------|---------|---------|------|
| M5t2 (sparse) | 26.60 | **27.30** | +0.70 ✅ |
| M5t2_cfgr (full) | **26.94** | 26.55 | -0.39 ❌ |

### 2.2 MVDiffusion 단독 결론

- **Full attention**: CFG 3.0에서 역효과 (-0.39)
- **Sparse attention + CFG 3.0**: 최고 성능 (27.30)
- **추가 학습 가치**: ❌ (cfgr은 step 8000에서 피크)

### 2.3 ⚠️ 미해결: E2E 비교 필요

MVDiffusion 단독 PSNR은 sparse > full이지만, **E2E 파이프라인에서의 차이는 미검증**.
Phase 1에서 E2E inference로 확인 필요.

---

## 3. 실행 명령어

### 3.1 Phase 1: E2E 비교 (학습 불필요)

```bash
cd /home/joon/dev/FaceLift

# cfgr E2E (full attention, ckpt-10000)
export CUDA_VISIBLE_DEVICES=4 && python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --mvdiff_ckpt /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_cfgr/checkpoint-10000 \
    --gslrm_ckpt /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt \
    --output_dir outputs/h5_e2e/cfgr_ckpt10000 \
    --guidance_scale 3.0

# baseline E2E (sparse attention, ckpt-5000)
export CUDA_VISIBLE_DEVICES=4 && python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --mvdiff_ckpt /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2/checkpoint-5000 \
    --gslrm_ckpt /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt \
    --output_dir outputs/h5_e2e/baseline_ckpt5000 \
    --guidance_scale 3.0
```

### 3.2 Phase 2B: 신규 학습 (baseline 승 시)

```bash
cd /home/joon/dev/FaceLift

# randref_sparse (reference augmentation)
export CUDA_VISIBLE_DEVICES=4 && PYTHONUNBUFFERED=1 nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py --config configs/mvdiffusion/mouse_mvdiffusion_M5t2_randref_sparse.yaml \
    > logs/mvdiff_M5t2_randref_sparse.log 2>&1 &

# 20k_sparse (longer training)
export CUDA_VISIBLE_DEVICES=4 && PYTHONUNBUFFERED=1 nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py --config configs/mvdiffusion/mouse_mvdiffusion_M5t2_20k_sparse.yaml \
    > logs/mvdiff_M5t2_20k_sparse.log 2>&1 &
```

---

## 4. 체크포인트 위치

| 실험 | 경로 | 용량 |
|------|------|------|
| M5t2 baseline | `mouse_M5t2/checkpoint-5000` | 13GB |
| M5t2_cfgr | `mouse_M5t2_cfgr/checkpoint-10000` | 13GB |
| pipeckpts (canonical) | `mvdiffusion/pipeckpts/` | 5.3GB (symlink 공유) |

Base: `/node_data/joon/checkpoints/FaceLift/mvdiffusion/`

---

## 5. 관련 문서

- [COMMANDS.md](../experiments/COMMANDS.md) — 명령어 SSOT
- [RESEARCH_HYPOTHESES.md](../RESEARCH_HYPOTHESES.md) — 전체 가설 맵
- [mvdiff_checkpoint_comparison.md](../../outputs/reports/mvdiff_checkpoint_comparison.md)

---

*H5 MVDiffusion | v2.0 | 2026-02-07*
