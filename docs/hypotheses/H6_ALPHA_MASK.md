# H6: Alpha Mask Loss

> **가설**: Rendered alpha mask를 supervision에 사용하면 foreground 품질이 개선될 것이다.
>
> ← [[INDEX]] | [[hypothesis_roadmap]] | **상태**: ❌ **기각** | **Updated**: 2026-02-22

---

## 1. 배경

### 1.1 문헌 근거
| 논문 | 방식 | 효과 |
|------|------|------|
| LGM | MSE alpha loss | "faster convergence of the shape" |
| Pose Splatter | normalized masked L1 | Mouse/rat 데이터 검증 |
| Object-Centric 2DGS | background penalty | foreground 집중 |

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

| Alpha Weight | Best PSNR | vs Baseline (21.71) | Trend |
|:------------:|:---------:|:-------------------:|:-----:|
| 0.0 (baseline) | **21.71** | — | — |
| 0.3 | 21.34 | -0.37 | ↓ |
| 0.5 | 21.20 | -0.51 | ↓ |
| 1.0 | 20.84 | -0.87 | ↓ |

### 5.2 결론

**❌ 기각**: Alpha weight 증가에 따른 **monotonic PSNR 하락**. 모든 설정에서 baseline 이하.
Alpha mask supervision은 mouse GS-LRM 학습에 해로움.

**원인 추정**: Alpha loss가 L2/perceptual loss와 경쟁하며 색상 재구성 품질을 희생시킴.
GT alpha 대신 rendered alpha를 사용했으므로 피드백 루프 가능성도 있음.

### 5.3 현황

| 상태 | 내용 |
|------|------|
| ✅ 완료 | 문헌 조사 |
| ✅ 완료 | Config 생성 (v3: 3개) |
| ✅ 완료 | 실험 실행 + 결과 분석 |
| **❌ 기각** | Baseline 최적. 후속 실험 불필요. |

---

*H6 Alpha Mask | v3.0 | 2026-02-22*
