# H6: Alpha Mask Loss

> **가설**: Rendered alpha mask를 supervision에 사용하면 foreground 품질이 개선될 것이다.
>
> ← [[INDEX]] | [[hypothesis_roadmap]] | [[ALPHA_LOSS_NOVEL_VIEW_ANALYSIS]] | **상태**: 🔄 **재평가 중** | **Updated**: 2026-03-16

---

## 1. 배경

### 1.1 문헌 근거
| 논문 | 방식 | 효과 |
|------|------|------|
| LGM (ECCV 2024) | MSE alpha loss | "faster convergence of the shape" |
| GaussianObject (SIG Asia 2024) | BCE alpha loss | Object-level reconstruction |
| Pose Splatter (NeurIPS 2025) | normalized masked L1 | Mouse/rat 데이터 검증 |
| Object-Centric 2DGS (2025) | background penalty | foreground 집중 |
| Compact-3DGS (2024) | anisotropy regularization | Elongated Gaussian 억제 |

### 1.2 주의사항
- `mask_mode: alpha` → **피드백 루프 위험**
  - 부정확한 초기 alpha → 배경에 Gaussian → alpha 확장
  - fg_coverage 0.33 → 0.70 악화 사례
- `mask_mode: gt` → 안전 (GT mask 사용)

---

## 2. 실험 설계

### 2.1 설정 비교 (Uniform v2)

> **기준**: 4view_v2 (alpha_w=0, mask=none)

| Config | alpha_w | mask_mode | masked_l2 | 가설 |
|--------|---------|-----------|-----------|------|
| 4view_v2 (baseline) | 0.0 | none | false | 기준선 |
| 4view_alpha01_v2 | **0.1** | none | false | LGM 표준: shape 수렴 가속 |
| 4view_alpha05_v2 | **0.5** | none | false | 강한 alpha → boundary 선명화 |
| 4view_maskgt_v2 | 0.1 | **gt** | **true** | GT mask L2 + alpha |

### 2.2 평가 메트릭

| 메트릭 | 기대 변화 | 의미 |
|--------|----------|------|
| val/mask_iou | ↑ | 직접 최적화 대상 (alpha → shape) |
| val/psnr | ↔ or ↓ | alpha가 PSNR에 미치는 영향 |
| val/ssim | ↔ | 구조적 영향 측정 |

---

## 3. 실행 명령어 (H4 Round 1 완료 후)

```bash
cd /home/joon/dev/FaceLift

# Alpha 0.1 (LGM standard)
export CUDA_VISIBLE_DEVICES=5 && nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/4view_alpha01_v2.yaml \
    > logs/uniform_4view_alpha01_v2.log 2>&1 &

# Alpha 0.5 (strong)
export CUDA_VISIBLE_DEVICES=6 && nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/4view_alpha05_v2.yaml \
    > logs/uniform_4view_alpha05_v2.log 2>&1 &

# GT mask + alpha 0.1
export CUDA_VISIBLE_DEVICES=7 && nohup python \
    train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml \
    -e configs/mouse/uniform/4view_maskgt_v2.yaml \
    > logs/uniform_4view_maskgt_v2.log 2>&1 &
```

---

## 4. Config 파일 위치

```
configs/mouse/uniform/
├── 4view_alpha01_v2.yaml       # alpha_loss_weight=0.1
├── 4view_alpha05_v2.yaml       # alpha_loss_weight=0.5
└── 4view_maskgt_v2.yaml        # mask_mode=gt + alpha 0.1 + masked_l2
```

---

## 5. 실험 결과 (2026-02-21)

### 5.1 Results (v3 configs: alpha_w = 0.3, 0.5, 1.0)

| Alpha Weight | Best PSNR | vs Baseline | LPIPS ↓ | SSIM ↑ | Alpha IoU ↑ |
|:------------:|:---------:|:-----------:|:-------:|:------:|:-----------:|
| 0.0 (baseline) | **21.82** | — | 0.0429 | 0.9473 | N/A |
| 0.3 | 21.34 | -0.48 | — | — | — |
| 0.5 | 21.20 | -0.62 | 0.0204 | 0.9725 | 0.9451 |
| 1.0 | 20.84 | -0.98 | **0.0147** | **0.9742** | **0.9562** |

> **주목**: PSNR은 monotonic 하락이나, **LPIPS 3배 개선**, SSIM/IoU 대폭 상승.
> 이는 geometry 품질이 크게 개선되었다는 강한 신호.

### 5.2 Training-View 결론 (2026-02-21)

**Training view PSNR 관점**: ❌ — Alpha weight 증가 시 monotonic PSNR 하락.

**원인 분석**: Alpha loss가 mask boundary 근처에서 세밀한 텍스처(모피)를 약간 blurring → pixel-wise 오차(PSNR) 증가. 그러나 perceptual/structural metrics는 오히려 개선.

### 5.3 Novel View 재평가 (2026-03-16)

**새로운 평가 축**: Extrapolated novel view (bottom view -70°) artifact 감소.

Alpha loss → silhouette 외부 Gaussian opacity 억제 → 배경 "pancake" Gaussian 제거 → bottom view artifact 감소 기대.

**상태 변경**: ❌ 기각 → 🔄 **재평가 중** (novel view 관점)

상세 분석: [[ALPHA_LOSS_NOVEL_VIEW_ANALYSIS]]

### 5.4 현황

| 상태 | 내용 |
|------|------|
| ✅ 완료 | 문헌 조사 |
| ✅ 완료 | Config 생성 (v3: 3개) |
| ✅ 완료 | 실험 실행 (training view 분석) |
| 🔄 진행중 | **P0: Novel view artifact 평가** (bottom view 렌더링) |
| ⏳ 계획 | P2: Alpha + anisotropy reg 결합 |
| ⏳ 계획 | P5: 6-view + alpha loss 학습 |

### 5.5 Checkpoint 위치

| Config | Checkpoint (서버) |
|--------|------------------|
| baseline (4v) | `/node_data/joon/checkpoints/FaceLift/gslrm/base_uniform_v2_4view_v2/best_psnr.pt` |
| alpha 0.3 | `/node_data/joon/checkpoints/FaceLift/gslrm/base_uniform_v2_4view_alpha03_v3/best_psnr.pt` |
| alpha 0.5 | `/node_data/joon/checkpoints/FaceLift/gslrm/base_uniform_v2_4view_alpha05_v3/best_psnr.pt` |
| alpha 1.0 | `/node_data/joon/checkpoints/FaceLift/gslrm/base_uniform_v2_4view_alpha10_v3/best_psnr.pt` |

---

## 6. Related Documents

| 문서 | 관계 |
|------|------|
| ↑ [[../INDEX]] | MoC |
| ↔ [[../experiments/ALPHA_LOSS_NOVEL_VIEW_ANALYSIS]] | Novel view artifact 상세 분석 |
| ↔ [[../experiments/PHASE2_NOVEL_VIEW_ROADMAP]] | Phase 2 artifact removal |
| ↔ [[../experiments/EXPERIMENT_REGISTRY]] | 실험 결과 기록 |
| ↔ [[../../mouse_extensions/docs/MESH_GUIDED_REFINEMENT]] | Mesh-guided 전략 |
| ↓ `mouse_extensions/model/mask_losses.py` | Alpha loss 구현 (SSOT) |

---

*H6 Alpha Mask | v4.0 | 2026-03-16*
