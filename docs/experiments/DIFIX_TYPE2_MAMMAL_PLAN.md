# DiFix Type 2: MAMMAL Mesh Bottom View Pair Generation Plan

Created: 2026-03-29 | MoA Audit: "Do NOT retry" 결론은 과잉 일반화 (3/3 합의)

---

## 1. Background

### 이전 실패 분석

| 항목 | 이전 PoC (실패) | 이번 계획 |
|------|----------------|----------|
| 데이터 | Type 3 only (zero gap) | **Type 2** (MAMMAL mesh pseudo-GT) |
| Loss | LPIPS + L2 + **Gram** (불안정) | LPIPS + L2 (Gram 제거) |
| Steps | 2000 의도 → 9500 bug | 2000 strict (early stop) |
| Target | GS-LRM 6v (same model) | **MAMMAL mesh render** (3D geometry) |
| 평가 | 2D image quality only | **3D-aware** (novel view re-render) |

### 왜 재시도하는가?

1. 이전 실패는 Type 3 (zero domain gap) → MLP가 identity function 학습
2. Gram loss 폭발 → loss 설계 문제 (근본 한계 아님)
3. Training bug (9500 steps) → 비제어 실험
4. **Type 2 (MAMMAL pseudo-GT) 미테스트** — 실제 3D geometry 감독 신호

## 2. Data Pair Generation

### Bottom View Camera

```python
# collect_dataset.py
NOVEL_VIEWS = {
    "bottom": {"elevation": -70.0, "azimuth": 0.0},
}
# radius=2.7, fx=fy=411.75 @ 512x512
```

### Step 1: GS-LRM Novel View Renders (Input)

```bash
ssh gpu03
cd /home/joon/dev/FaceLift
conda activate facelift

# Check if already exists
ls outputs/datasets/novel_view/mouse_m5t2/tier0_raw/bottom/ 2>/dev/null | wc -l

# If < 3600, generate:
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.novel_view.collect_dataset \
    --mode generate --phase gslrm \
    --frame_range 0 3600 \
    --views bottom
# 예상: ~30분 (6v checkpoint inference)
```

### Step 2: MAMMAL Mesh Renders (Target)

```bash
# Requires mammal_stable env for pyrender
conda activate mammal_stable
PYOPENGL_PLATFORM=egl python -m mouse_extensions.scripts.novel_view.collect_dataset \
    --mode generate --phase mammal \
    --frame_range 0 3600 \
    --views bottom
# 예상: ~18분
```

### Step 3: Build DiFix Pairs

```bash
conda activate facelift
python -m mouse_extensions.scripts.eval.build_difix_dataset \
    --types 2 \
    --views bottom \
    --use-symlinks
# Output: difix_pairs/type2_novel_view/{frame}_{view}/input.png + target.png
# 예상 pairs: 3,600 (bottom only)
```

## 3. Training Plan

### Stage 0: Zero-Shot Baseline

```bash
python -m mouse_extensions.scripts.eval.difix_zero_shot \
    --input_dir outputs/datasets/novel_view/mouse_m5t2/tier0_raw/bottom \
    --output_dir outputs/difix_zero_shot/type2_bottom \
    --n_samples 20
# 1-2시간, pretrained DiFix weights
```

### Stage 1: Type 2 Fine-Tuning

```bash
python -m mouse_extensions.scripts.eval.train_difix \
    --pairs_dir difix_pairs/type2_novel_view \
    --loss lpips+l2 \
    --max_steps 2000 \
    --eval_every 200 \
    --output_dir outputs/difix_training/type2_bottom_v1
# Gram loss 제거 (이전 폭발 원인)
```

### Stage 2: Mixed Training (선택)

Type 3 80% + Type 2 20% 혼합 (curriculum Stage 1.5):
```bash
python -m mouse_extensions.scripts.eval.train_difix \
    --pairs_dir difix_pairs/type2_novel_view:difix_pairs/type3_view_ablation \
    --mix_ratio 0.2:0.8 \
    --loss lpips+l2 \
    --max_steps 5000
```

## 4. Evaluation Protocol

### Style Transfer 위험 방지 (Gemini audit 경고)

Type 2는 GS-LRM look → MAMMAL look 스타일 전환만 학습할 위험.

**평가 방법**:
1. ❌ 2D image 직접 비교 (PSNR vs MAMMAL render) — 스타일 전환 보상
2. ✅ **3D-aware 평가**:
   - DiFix refined image → novel view consistency check
   - 여러 각도에서 DiFix 적용 후 multi-view consistency 측정
   - FG-PSNR (foreground masked, GT view) — 실제 GT 대비

### Metrics

| Metric | 설명 | Target |
|--------|------|--------|
| FG-PSNR (GT view) | Foreground PSNR vs real GT | > 20.01 dB (α=0.3 baseline) |
| LPIPS (GT view) | Perceptual quality | < baseline |
| Spike count | Needle artifact 정량화 | 감소 |
| Visual | Side-by-side comparison | 개선 확인 |

## 5. Resource Requirements

| 항목 | GPU | 시간 | 비고 |
|------|:---:|:----:|------|
| GS-LRM novel renders | 1 | ~30min | 1회성 |
| MAMMAL mesh renders | 0 (CPU/EGL) | ~18min | 1회성 |
| DiFix zero-shot | 1 | ~2h | 20 samples |
| DiFix Type 2 training | 1 | ~4-6h | 2000 steps |
| Evaluation | 1 | ~1h | FG-PSNR + visual |

**총 소요: ~1일 (GPU 1개)**

## 6. Risk Assessment

| Risk | Severity | Mitigation |
|------|:--------:|------------|
| Style transfer (MAMMAL look) | High | 3D-aware 평가, GT view 교차 검증 |
| Mode collapse (재발) | Medium | Gram loss 제거, 2000 step strict |
| Resolution mismatch (384 vs 512) | Low | 512 확인 후 생성 |
| MAMMAL fitting 품질 | Medium | Keyframe만 사용 (900 frames) |

---

> Related: [[DIFIX_TRAINING_STRATEGY]] | [[260329_GS-LRM_Architecture_and_Temporal_Analysis]]
