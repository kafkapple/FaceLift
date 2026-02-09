# H6: Alpha Mask Loss

> **가설**: Rendered alpha mask를 supervision에 사용하면 foreground 품질이 개선될 것이다.
>
> ← [RESEARCH_HYPOTHESES.md](../RESEARCH_HYPOTHESES.md) | **상태**: ⏳ 대기 | **Updated**: 2026-02-09

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

## 5. 현황

| 상태 | 내용 |
|------|------|
| ✅ 완료 | 문헌 조사 |
| ✅ 완료 | Config 생성 (3개, uniform_v2 체계) |
| ✅ 완료 | mask_iou 항상 계산 (코드 수정) |
| ⏳ 대기 | H4 Round 1 완료 후 진행 |

---

*H6 Alpha Mask | v2.0 | 2026-02-07*
