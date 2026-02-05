# MVDiffusion CFG Dropout & Attention Ablation

**Date**: 2026-02-03 | **Updated**: 2026-02-04
**Status**: In Progress
**Tags**: #mvdiffusion #ablation #cfg #attention

---

## 1. 문제 제기

M5t2_consistent 모델의 결과가 기존 M5t baseline보다 안 좋아 보임.
변경된 설정이 오히려 품질 저하를 유발했을 가능성 조사.

---

## 2. 변경 이력

| 버전 | 날짜 | 주요 변경 |
|------|------|-----------|
| M5t | 2026-01-28 | Baseline (original prompt, 1to1 split) |
| M5t_mp | 2026-02-01 | Mouse prompt 적용 |
| M5t2 | 2026-02-01 | t2 split (80:10:10) |
| M5t2_consistent | 2026-02-02 | CFG dropout 제거, Full attention |

---

## 3. 가설

### H1: CFG Dropout 제거가 품질 저하의 원인

| 설정 | M5t/M5t2 | M5t2_consistent |
|------|----------|-----------------|
| `condition_drop_rate` | **0.05** | **0.0** |

**메커니즘**:
- `condition_drop_rate=0.05`: 학습 중 5% 확률로 conditioning 드롭 → unconditional 학습
- Inference 시 `guidance_scale > 1.0` 적용하면 조건부/비조건부 차이로 guidance 수행
- `condition_drop_rate=0.0`: unconditional 학습 없음 → guidance 효과 미미

### H2: Full Attention이 과적합 유발

| 설정 | M5t/M5t2 | M5t2_consistent |
|------|----------|-----------------|
| `sparse_mv_attention` | **true** | **false** |

**메커니즘**:
- Sparse: 인접 뷰 간만 attention (regularization 효과)
- Full: 모든 뷰 쌍 간 attention → 과적합 위험 (t2 train: 2880 samples)

### H3: Prompt 임베딩 (가능성 낮음)
- M5t_mp가 M5t와 비슷하다면 prompt는 원인 아님

---

## 4. 실험 설계

### 4.1 통제 변수 ⚠️

**GS-LRM 고정** (MVDiffusion 품질만 비교):
```
/node_data/joon/checkpoints/FaceLift/gslrm/M5t_E0_1_facelift/best_psnr.pt
```

**테스트 데이터 고정**:
```
data_mouse_1to1_test.txt (1204 samples)
```

### 4.2 실험 조건

| 실험 | MVDiffusion | CFG | Sparse | 역할 |
|------|-------------|-----|--------|------|
| **Ctrl** | M5t/ckpt-8000 | 0.05 | true | Baseline |
| **Exp-A** | M5t2/ckpt-5000 | 0.05 | true | Split+Prompt 변경 |
| **Exp-B** | M5t2_consistent/ckpt-6000 | 0.0 | false | +CFG제거+FullAttn |

### 4.3 Split 옵션

| Split | Train | Val | Test | 비율 |
|-------|-------|-----|------|------|
| **1to1** | 1198 | 1198 | 1204 | 1:1:1 |
| **t2** | 2880 | 360 | 360 | 80:10:10 |

Train 데이터로 테스트 시: `--split .../data_mouse_1to1_train.txt`

---

## 5. 가설 검증 계획

### 5.1 검증 매트릭스

| 비교 | 결과 예상 | 검증되는 가설 |
|------|----------|---------------|
| Ctrl ≈ Exp-A | 비슷 | H1, H2 아님 → Split/Prompt 영향 없음 |
| Ctrl > Exp-A > Exp-B | 점진적 저하 | H1 + H2 복합 |
| Ctrl ≈ Exp-A > Exp-B | Exp-B만 저하 | H1 또는 H2 (추가 분리 필요) |

### 5.2 추가 실험 (결과에 따라)

**H1 확인 시** → Exp-C: CFG 복원
```yaml
condition_drop_rate: 0.05  # 원복
sparse_mv_attention: false  # full 유지
```

**H2 확인 시** → WandB train/val loss 곡선 분석

---

## 6. 평가 지표

| 지표 | 확인 방법 | 기대 |
|------|-----------|------|
| **Visual Quality** | 6-view 이미지 검토 | Ctrl > Exp-B? |
| **View Consistency** | 형태/색상 일관성 | Ctrl ≥ Exp-A > Exp-B? |
| **3D Quality** | Turntable 렌더링 | Ctrl > Exp-B? |
| **Artifact** | Ghosting, 불일치 | Exp-B에서 더 많을 것 |

---

## 7. 결과 (TBD)

### 7.1 실험 현황

| 실험 | 상태 | 출력 |
|------|------|------|
| Ctrl | ⏳ | `outputs/compare_mvdiff/ctrl_M5t_8k/` |
| Exp-A | ✅ | `outputs/compare_mvdiff/exp_A_M5t2_5k/` |
| Exp-B | ⏳ | `outputs/compare_mvdiff/exp_B_M5t2_consistent_6k/` |

### 7.2 정성 평가

*(결과 후 작성)*

### 7.3 결론

*(결과 후 작성)*

---

## 8. 명령어

→ **Quick Reference**: `docs/MOUSE_QUICK_REFERENCE.md` §3.8 참조

---

## 9. References

- Ho & Salimans (2022). Classifier-Free Diffusion Guidance. arXiv:2207.12598
- Shi et al. (2023). MVDiffusion. CVPR 2024.

---

*FaceLift Mouse Project | Research Notes*
